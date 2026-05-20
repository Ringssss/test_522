#!/usr/bin/env python3
"""
v0.1.15.8g — S_mask stability data collection (HetEval-32)

Runs C10-v6 (original K_A + K_B_v3) on full HetEval-32 (gen_length=256),
records every hot-path S_mask per layer per forward, then analyzes
inter-forward stability (Jaccard similarity).

Output: JSON with per-layer Jaccard curves + summary statistics.
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path
from collections import defaultdict

import torch
import triton
import triton.language as tl

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

from test_fused_eb_triton import (
    fused_routing, FusedEBController,
    _kernel_A, _kernel_A_cold, _kernel_B_v2,
    _kernel_C, _kernel_D_v2, _kernel_zero_init,
)
from baseline_optimizations import apply_all_optimizations

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
GEN_LENGTH = 256  # full HetEval-32

PROMPTS = [
    "Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h. The total distance between the two cities is 240 km. What is the average speed for the entire round trip?\n\nProblem 2: A rectangular garden has a perimeter of 56 meters.",
    "Write a detailed essay about the history of artificial intelligence, covering the Dartmouth conference of 1956, the AI winters, the rise of machine learning in the 1990s, and deep learning breakthroughs.",
    "You are a chemistry professor. Explain Le Chatelier's principle with examples and how it applies to industrial ammonia production via the Haber process.",
    "Design a complete REST API for an e-commerce platform with endpoints for user authentication, product management, shopping cart operations, and order processing.",
    "Analyze the global economic impact of climate change across agriculture, energy, real estate, and healthcare sectors with specific examples.",
    "Explain quantum computing to a classical CS background: qubits, superposition, entanglement, Shor's algorithm, and current hardware approaches.",
    "You are a systems architect. Design a distributed message queue with partition-based storage, consumer groups, replication, and exactly-once semantics.",
    "Write a comprehensive guide to training large language models covering data collection, tokenizer training, architecture decisions, and distributed training strategies.",
    "Solve the quadratic equation x^2 - 5x + 6 = 0 step by step. Show the factoring method, then verify both solutions by substituting them back into the original equation.",
    "Explain the mathematical foundations of neural networks: backpropagation, gradient descent, loss functions, and the universal approximation theorem.",
    "Design a microservices architecture for a ride-sharing application with real-time matching, pricing, routing, payments, and driver management.",
    "Write about the history of cryptography from Caesar ciphers through RSA, elliptic curve cryptography, and post-quantum cryptographic algorithms.",
    "Explain database indexing strategies: B-trees, hash indexes, bitmap indexes, and their trade-offs for OLTP vs OLAP workloads.",
    "Solve this logic puzzle step by step: If A is true, then B is true. If B is true, then C is true. A is true. What can we conclude about B and C? Then, if D is true only when both B and C are true, what can we conclude about D?",
    "Design a CI/CD pipeline for a large monorepo with microservices, including build caching, parallel testing, canary deployments, and rollback strategies.",
    "Explain the theory of relativity to a physics undergraduate, covering special relativity, time dilation, length contraction, and general relativity basics.",
    "Write a comprehensive comparison of Python, Rust, and Go for systems programming, covering memory safety, concurrency models, and ecosystem maturity.",
    "Design a real-time recommendation engine for a video streaming platform that handles cold start, user preferences, and content diversity.",
    "Explain the CAP theorem and its practical implications for distributed database design, with examples from Cassandra, MongoDB, and CockroachDB.",
    "Write a Python function to compute the nth Fibonacci number. Show the function, then compute fib(1) through fib(10) step by step and list all 10 values.",
    "Design a fraud detection system for a payment processing company using machine learning, rule engines, and real-time streaming analytics.",
    "Explain compiler optimization techniques including SSA form, loop unrolling, vectorization, and register allocation strategies.",
    "Write about the history and future of space exploration, from Apollo missions through SpaceX reusability to planned Mars colonization.",
    "Design an observability platform with distributed tracing, log aggregation, metrics collection, and intelligent alerting for microservices.",
    "Explain the mathematics behind public key cryptography, including modular arithmetic, Euler's theorem, and the RSA algorithm step by step.",
    "Write a guide to modern CSS layout techniques including Flexbox, Grid, Container Queries, and responsive design best practices.",
    "Design a multi-tenant SaaS platform architecture with data isolation, custom domains, billing integration, and horizontal scaling.",
    "Explain how garbage collectors work in JVM, Go, and Python, comparing mark-sweep, generational, and reference counting approaches.",
    "List all 8 planets in our solar system in order from closest to farthest from the Sun. For each planet, state whether it is a terrestrial or gas/ice giant planet, and give its approximate orbital period in Earth years.",
    "Design a real-time collaborative document editor like Google Docs with conflict resolution, offline support, and version history.",
    "Explain operating system memory management: virtual memory, page tables, TLB, demand paging, and memory-mapped files.",
    "Write a comprehensive guide to Kubernetes architecture including pods, services, ingress, operators, and cluster autoscaling.",
]


# ================================================================
# K_B_v3 (tl.sort)
# ================================================================
@triton.jit
def _kernel_B_v3(pop_ptr, s_mask_ptr, K_init, E: tl.constexpr):
    offs = tl.arange(0, E)
    pop = tl.load(pop_ptr + offs).to(tl.float32)
    pop_sorted = tl.sort(pop, descending=True)
    kth_idx = K_init - 1
    threshold = tl.max(tl.where(offs == kth_idx, pop_sorted, -float("inf")), axis=0)
    gt_mask = pop > threshold
    gt_count = tl.sum(gt_mask.to(tl.int32), axis=0)
    eq_mask = pop == threshold
    need_eq = K_init - gt_count
    eq_rank = tl.cumsum(eq_mask.to(tl.int32), axis=0)
    take_eq = eq_mask & (eq_rank <= need_eq)
    sel = gt_mask | take_eq
    tl.store(s_mask_ptr + offs, sel.to(tl.int32))
    tl.store(pop_ptr + offs, tl.zeros([E], dtype=tl.float32))


# ================================================================
# Recording Controller: captures every S_mask
# ================================================================
class RecordingEBController(FusedEBController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_init_history = []
        # Recording state
        self.records = []  # list of (layer_idx, is_cold, block_idx, fwd_in_block, s_mask_cpu)
        self._block_idx = {}    # layer_idx → current block index
        self._fwd_in_block = {} # layer_idx → forward count within current block

    def cold_path(self, layer_idx, logits, bias):
        N, E = logits.shape
        b = self._get_bufs(N, E, logits.device)

        _kernel_zero_init[(1,)](b['pop'], b['sat_flag'], b['sat_count'],
                                b['G'], b['H'], E=E)
        lf = logits.float()
        bf = bias.float()

        _kernel_A_cold[(N,)](
            lf, bf, b['pop'],
            b['topkm_idx'], b['topkm_w'], b['r'],
            N, self.rsf, self.quality_floor,
            lf.stride(0), lf.stride(1),
            b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
            E=E, KEXT=self.K_ext, KEXT_PAD=16, K=self.K)

        _kernel_B_v2[(1,)](b['pop'], b['s_mask'], self.K_target,
                           E=E, MAX_K=256)

        q_major_x1000 = int(self.q_major * 1000)
        for _ in range(self.MAX_ROUNDS):
            _kernel_C[(N,)](
                b['topkm_idx'], b['topkm_w'], b['r'],
                b['s_mask'], b['sat_flag'],
                b['sat_count'], b['G'], b['H'],
                N, b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                E=E, KEXT=self.K_ext, KEXT_PAD=16)
            _kernel_D_v2[(1,)](
                b['s_mask'], b['sat_flag'], b['sat_count'],
                b['G'], b['H'],
                N, q_major_x1000,
                E=E, CAP=self.cap)

        actual_s = int(b['s_mask'].sum().item())
        self.K_init[layer_idx] = actual_s
        self.k_init_history.append(actual_s)

        # Record
        bi = self._block_idx.get(layer_idx, -1) + 1
        self._block_idx[layer_idx] = bi
        self._fwd_in_block[layer_idx] = 0
        self.records.append((layer_idx, True, bi, 0, b['s_mask'].cpu().clone()))

        self.cold_count += 1
        return b['s_mask']

    def hot_path(self, layer_idx, logits, bias):
        N, E = logits.shape
        K_init = self.K_init.get(layer_idx, 103)
        b = self._get_bufs(N, E, logits.device)

        lf = logits.float()
        _kernel_A[(N,)](lf, bias.float(), b['pop'],
                        N, self.rsf, lf.stride(0), lf.stride(1),
                        E=E, KEXT=self.K_ext, KEXT_PAD=16)

        _kernel_B_v3[(1,)](b['pop'], b['s_mask'], K_init, E=E)

        # Record
        fi = self._fwd_in_block.get(layer_idx, 0) + 1
        self._fwd_in_block[layer_idx] = fi
        bi = self._block_idx.get(layer_idx, 0)
        self.records.append((layer_idx, False, bi, fi, b['s_mask'].cpu().clone()))

        self.hot_count += 1
        return b['s_mask']


# ================================================================
# Analysis
# ================================================================
def analyze_stability(records):
    """Compute Jaccard similarity curves from recorded S_masks."""
    # Organize by (layer_idx, block_idx) → list of (fwd_in_block, s_mask)
    by_layer_block = defaultdict(list)
    for layer_idx, is_cold, block_idx, fwd_in_block, s_mask in records:
        by_layer_block[(layer_idx, block_idx)].append((fwd_in_block, is_cold, s_mask))

    # Sort each list by fwd_in_block
    for key in by_layer_block:
        by_layer_block[key].sort(key=lambda x: x[0])

    # Compute Jaccard: adjacent (S_t vs S_{t-1}) and vs cold (S_t vs S_cold)
    per_layer = defaultdict(lambda: {
        'jaccard_adjacent': [],     # (fwd_in_block, jaccard)
        'jaccard_vs_cold': [],      # (fwd_in_block, jaccard)
        'n_changed_adjacent': [],   # (fwd_in_block, n_changed)
    })

    for (layer_idx, block_idx), seq in by_layer_block.items():
        if len(seq) < 2:
            continue
        cold_mask = seq[0][2].bool()  # first entry is cold start

        for i in range(1, len(seq)):
            fwd_i = seq[i][0]
            curr = seq[i][2].bool()
            prev = seq[i-1][2].bool()

            # Jaccard adjacent
            inter_adj = (curr & prev).sum().item()
            union_adj = (curr | prev).sum().item()
            jac_adj = inter_adj / union_adj if union_adj > 0 else 1.0

            # Jaccard vs cold
            inter_cold = (curr & cold_mask).sum().item()
            union_cold = (curr | cold_mask).sum().item()
            jac_cold = inter_cold / union_cold if union_cold > 0 else 1.0

            # n_changed
            n_changed = (curr != prev).sum().item()

            per_layer[layer_idx]['jaccard_adjacent'].append((fwd_i, jac_adj))
            per_layer[layer_idx]['jaccard_vs_cold'].append((fwd_i, jac_cold))
            per_layer[layer_idx]['n_changed_adjacent'].append((fwd_i, n_changed))

    return dict(per_layer)


def print_analysis(per_layer):
    """Print summary tables."""
    print(f"\n{'='*80}")
    print("S_MASK STABILITY ANALYSIS")
    print(f"{'='*80}")

    # Per-layer average Jaccard
    print(f"\n  {'Layer':<8s} {'Adj Jaccard':>12s} {'vs Cold':>12s} {'Adj Changed':>12s} {'Samples':>8s}")
    print(f"  {'-'*55}")

    all_adj = []
    all_cold = []
    all_changed = []

    for li in sorted(per_layer.keys()):
        data = per_layer[li]
        adj_vals = [v for _, v in data['jaccard_adjacent']]
        cold_vals = [v for _, v in data['jaccard_vs_cold']]
        changed_vals = [v for _, v in data['n_changed_adjacent']]

        avg_adj = sum(adj_vals) / len(adj_vals) if adj_vals else 0
        avg_cold = sum(cold_vals) / len(cold_vals) if cold_vals else 0
        avg_changed = sum(changed_vals) / len(changed_vals) if changed_vals else 0

        all_adj.extend(adj_vals)
        all_cold.extend(cold_vals)
        all_changed.extend(changed_vals)

        print(f"  L{li:<6d} {avg_adj:>12.4f} {avg_cold:>12.4f} {avg_changed:>12.1f} {len(adj_vals):>8d}")

    print(f"  {'GLOBAL':<8s} {sum(all_adj)/len(all_adj):>12.4f} "
          f"{sum(all_cold)/len(all_cold):>12.4f} "
          f"{sum(all_changed)/len(all_changed):>12.1f} {len(all_adj):>8d}")

    # Jaccard by fwd_in_block (averaged across layers and blocks)
    print(f"\n  Jaccard by fwd_in_block (adjacent):")
    by_fwd = defaultdict(list)
    by_fwd_cold = defaultdict(list)
    for li in per_layer:
        for fwd_i, jac in per_layer[li]['jaccard_adjacent']:
            by_fwd[fwd_i].append(jac)
        for fwd_i, jac in per_layer[li]['jaccard_vs_cold']:
            by_fwd_cold[fwd_i].append(jac)

    print(f"  {'fwd_in_blk':>12s} {'Adj Jaccard':>12s} {'vs Cold':>12s} {'Samples':>8s}")
    print(f"  {'-'*45}")
    for fi in sorted(by_fwd.keys())[:30]:  # first 30
        avg_a = sum(by_fwd[fi]) / len(by_fwd[fi])
        avg_c = sum(by_fwd_cold.get(fi, [0])) / len(by_fwd_cold.get(fi, [1]))
        print(f"  {fi:>12d} {avg_a:>12.4f} {avg_c:>12.4f} {len(by_fwd[fi]):>8d}")

    # Safety thresholds for M
    print(f"\n  Safe M estimation (adjacent Jaccard > 0.95):")
    for M in [2, 3, 5, 10]:
        # For M-skip: compare S_t with S_{t-M}
        safe_count = 0
        total_count = 0
        for li in per_layer:
            data = per_layer[li]
            adj_list = data['jaccard_adjacent']
            # Approximate M-skip Jaccard from adjacent data
            # Jaccard(t, t-M) ≈ product of adjacent Jaccards (rough lower bound)
            for i in range(M-1, len(adj_list)):
                # Compute cumulative Jaccard over M steps
                jacs = [adj_list[i-j][1] for j in range(M-1)]
                # Conservative estimate: min of adjacent Jaccards
                min_jac = min(jacs) if jacs else 1.0
                if min_jac > 0.95:
                    safe_count += 1
                total_count += 1
        pct = safe_count / total_count * 100 if total_count > 0 else 0
        print(f"    M={M}: {pct:.1f}% of windows have min adjacent Jaccard > 0.95")


# ================================================================
# Main
# ================================================================
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoTokenizer, AutoConfig

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print("S_mask Stability Data Collection — HetEval-32")
    print(f"  gen_length={GEN_LENGTH}, batch=32, block=32, threshold=0.90")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0),
                      use_cache=False)

        print("\nApplying baseline optimizations...")
        n_rms, n_fa = apply_all_optimizations(model)
        print(f"  Fused RMSNorm: {n_rms}, Flash-attn: {n_fa}")

        BATCH_SIZE = 32
        all_ids = []
        for i in range(BATCH_SIZE):
            text = PROMPTS[i]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=False)
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                  if ids.shape[0] < mx else ids for ids in all_ids]
        input_ids = torch.stack(padded, dim=0).to(device)
        print(f"  Input shape: {input_ids.shape}")

        decoder_t0 = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm(decoder):
            return BlockDiffusionLLM(
                model, decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True,
                inplace_cache_update=True)

        orig_routings = {}
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                orig_routings[name] = mod.routing

        def restore():
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate" and name in orig_routings:
                    mod.routing = orig_routings[name]

        # Patch with recording controller
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--q-major', type=float, default=0.95)
        args, _ = parser.parse_known_args()
        Q_MAJOR = args.q_major
        print(f"  q_major={Q_MAJOR}")

        ctrl = RecordingEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=Q_MAJOR, per_round_cap=8)

        idx = 0
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                     mod.top_k, mod.n_group, mod.topk_group)
                li = idx
                def mk(bb, rr, tt, nn, gg, layer_i, cc):
                    def fn(hs, go, topk, renorm):
                        sm = cc.get_s_mask(layer_i, go, bb)
                        w, i = fused_routing(go, bb, rr, s_mask=sm,
                                             K=tt, ng=nn, tkg=gg)
                        return w.to(go.dtype), i
                    return fn
                mod.routing = mk(b, r, tk, ng, tkg, li, ctrl)
                idx += 1
        print(f"  Patched {idx} gates with recording controller")

        # Warmup
        print("\nWarmup...")
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd, "
              f"cold={ctrl.cold_count}, hot={ctrl.hot_count}")

        # Reset for data collection run
        ctrl.records.clear()
        ctrl.prev_N.clear(); ctrl.K_init.clear()
        ctrl.cold_count = 0; ctrl.hot_count = 0
        ctrl._bufs.clear(); ctrl.k_init_history.clear()
        ctrl._block_idx.clear(); ctrl._fwd_in_block.clear()

        # Data collection run
        print("\nData collection run...")
        dllm = make_dllm(decoder_t0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        total_fwd = dllm.diff_iteration.num_forwards
        print(f"  Done: {total_fwd} fwd, {t1-t0:.3f}s")
        print(f"  Records: {len(ctrl.records)} S_masks captured")
        print(f"  Cold: {ctrl.cold_count}, Hot: {ctrl.hot_count}")

        # Analyze
        per_layer = analyze_stability(ctrl.records)
        print_analysis(per_layer)

        # Save summary (not raw S_masks — too large for JSON)
        summary = {}
        for li in sorted(per_layer.keys()):
            data = per_layer[li]
            adj_vals = [v for _, v in data['jaccard_adjacent']]
            cold_vals = [v for _, v in data['jaccard_vs_cold']]
            changed_vals = [v for _, v in data['n_changed_adjacent']]
            summary[f"L{li}"] = {
                'avg_jaccard_adjacent': sum(adj_vals) / len(adj_vals) if adj_vals else 0,
                'min_jaccard_adjacent': min(adj_vals) if adj_vals else 0,
                'avg_jaccard_vs_cold': sum(cold_vals) / len(cold_vals) if cold_vals else 0,
                'avg_n_changed': sum(changed_vals) / len(changed_vals) if changed_vals else 0,
                'n_samples': len(adj_vals),
            }

        # Jaccard by fwd_in_block
        by_fwd = defaultdict(list)
        for li in per_layer:
            for fwd_i, jac in per_layer[li]['jaccard_adjacent']:
                by_fwd[fwd_i].append(jac)
        fwd_curve = {str(fi): sum(by_fwd[fi])/len(by_fwd[fi])
                     for fi in sorted(by_fwd.keys())}

        save_data = {
            'q_major': Q_MAJOR,
            'total_fwd': total_fwd,
            'total_time_s': t1 - t0,
            'cold_count': ctrl.cold_count,
            'hot_count': ctrl.hot_count,
            'per_layer_summary': summary,
            'jaccard_by_fwd_in_block': fwd_curve,
            'k_init_history': ctrl.k_init_history,
        }

        qm_tag = str(Q_MAJOR).replace('.', '')
        out_path = REPO_ROOT / "codex_coding" / "results" / f"s_mask_stability_qm{qm_tag}.json"
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")

        restore()
        print("\nDone.")


if __name__ == "__main__":
    main()
