#!/usr/bin/env python3
"""
v0.1.15.8i — C10-overlap: Real CUDA stream overlap for EB hot path

K_A + K_B run on eb_stream, overlapped with fused_experts on main stream.
Hot path returns s_mask_old (from previous forward), computes s_mask_new async.
Cold path uses q_major=1.0 (full coverage) for best delay tolerance.

Full HetEval-32 (gen_length=256).
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path
from collections import OrderedDict

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
GEN_LENGTH = 256

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
VERIFIABLE = {
    0: "average speed = 480/7 ~ 68.57 km/h", 8: "x = 2 and x = 3",
    13: "B is true, C is true, D is true", 19: "fib(10) = 55",
    28: "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune",
}


# K_B_v3
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
# Overlap EB Controller
# ================================================================
class OverlapEBController(FusedEBController):
    """
    Hot path: return s_mask from previous forward (double buffer read slot),
    compute new s_mask on eb_stream (write slot), pointer swap (no copy).
    Cold path: sync, compute fresh, no delay.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_init_history = []
        self.eb_stream = None
        self.logits_event = None

        # Per-layer double buffer: [buf_a, buf_b], buf_idx points to current read slot
        self.s_mask_buf = {}   # layer_idx → [tensor, tensor]
        self.buf_idx = {}      # layer_idx → 0 or 1  (read slot)
        self.pop_buf = {}      # layer_idx → tensor[256]

    def _ensure_stream(self, device):
        if self.eb_stream is None:
            self.eb_stream = torch.cuda.Stream(device)
            self.logits_event = torch.cuda.Event()

    def _ensure_layer_bufs(self, layer_idx, E, device):
        if layer_idx not in self.s_mask_buf:
            self.s_mask_buf[layer_idx] = [
                torch.ones(E, device=device, dtype=torch.int32),
                torch.ones(E, device=device, dtype=torch.int32),
            ]
            self.buf_idx[layer_idx] = 0
            self.pop_buf[layer_idx] = torch.zeros(E, device=device, dtype=torch.float32)

    def cold_path(self, layer_idx, logits, bias):
        """Cold: sync eb_stream, compute fresh s_mask on main stream, no delay."""
        N, E = logits.shape
        device = logits.device
        self._ensure_stream(device)
        self._ensure_layer_bufs(layer_idx, E, device)
        b = self._get_bufs(N, E, device)

        self.eb_stream.synchronize()

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

        # Write cold result into BOTH double buffer slots
        self.s_mask_buf[layer_idx][0].copy_(b['s_mask'])
        self.s_mask_buf[layer_idx][1].copy_(b['s_mask'])
        self.buf_idx[layer_idx] = 0

        self.cold_count += 1
        return b['s_mask']

    def hot_path(self, layer_idx, logits, bias):
        """
        1. Return read slot (s_mask from previous forward)
        2. Launch K_A + K_B on eb_stream, K_B writes to write slot
        3. Swap pointer (Python int, no GPU copy)
        """
        N, E = logits.shape
        device = logits.device
        K_init = self.K_init.get(layer_idx, 103)
        self._ensure_stream(device)
        self._ensure_layer_bufs(layer_idx, E, device)

        read_idx = self.buf_idx[layer_idx]
        write_idx = 1 - read_idx
        s_mask_old = self.s_mask_buf[layer_idx][read_idx]
        s_mask_new = self.s_mask_buf[layer_idx][write_idx]

        # Record event: logits ready on main stream
        self.logits_event.record(torch.cuda.current_stream(device))

        # Launch EB on eb_stream
        pop = self.pop_buf[layer_idx]
        lf = logits.float()

        with torch.cuda.stream(self.eb_stream):
            self.eb_stream.wait_event(self.logits_event)
            _kernel_A[(N,)](lf, bias.float(), pop,
                            N, self.rsf, lf.stride(0), lf.stride(1),
                            E=E, KEXT=self.K_ext, KEXT_PAD=16)
            _kernel_B_v3[(1,)](pop, s_mask_new, K_init, E=E)
            # NO copy_! K_B writes directly to s_mask_new (write slot)

        # Swap pointer for next forward (Python only, zero GPU work)
        self.buf_idx[layer_idx] = write_idx

        self.hot_count += 1
        return s_mask_old


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
    print("v0.1.15.8i — C10-overlap: Real CUDA stream overlap")
    print(f"  gen_length={GEN_LENGTH}, q_major=1.0, HetEval-32")
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
        prompt_len = input_ids.shape[1]
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

        def patch_routing(eb_ctrl):
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
                    mod.routing = mk(b, r, tk, ng, tkg, li, eb_ctrl)
                    idx += 1
            return idx

        def patch_c5():
            restore()
            idx = 0
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate":
                    b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                         mod.top_k, mod.n_group, mod.topk_group)
                    def mk(bb, rr, tt, nn, gg):
                        def fn(hs, go, topk, renorm):
                            w, i = fused_routing(go, bb, rr, s_mask=None, K=tt, ng=nn, tkg=gg)
                            return w.to(go.dtype), i
                        return fn
                    mod.routing = mk(b, r, tk, ng, tkg)
                    idx += 1
            return idx

        def reset_ctrl(ctrl):
            ctrl.prev_N.clear(); ctrl.K_init.clear()
            ctrl.cold_count = 0; ctrl.hot_count = 0
            ctrl._bufs.clear(); ctrl.k_init_history.clear()
            ctrl.s_mask_buf.clear()
            ctrl.buf_idx.clear()
            ctrl.pop_buf.clear()

        results = OrderedDict()

        # ---- C5 ----
        print(f"\n{'='*60}")
        print("C5: Fused routing only")
        print(f"{'='*60}")
        patch_c5()
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        c5_times, c5_fwds = [], []
        for ri in range(2):
            dllm = make_dllm(decoder_t0); torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            c5_times.append(t1-t0); c5_fwds.append(dllm.diff_iteration.num_forwards)
            print(f"    Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd")
        results['C5'] = {'avg_time': sum(c5_times)/2, 'avg_fwd': sum(c5_fwds)/2,
                         'ms_per_fwd': sum(c5_times)/2 / (sum(c5_fwds)/2) * 1000}

        # ---- C10-overlap ----
        print(f"\n{'='*60}")
        print("C10-overlap: Real CUDA stream overlap (q_major=1.0)")
        print(f"{'='*60}")
        restore()
        ctrl_ov = OverlapEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8)
        patch_routing(ctrl_ov)

        # Warmup
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Warmup: {dllm.diff_iteration.num_forwards} fwd, "
              f"cold={ctrl_ov.cold_count}, hot={ctrl_ov.hot_count}")

        ov_times, ov_fwds = [], []
        for ri in range(2):
            reset_ctrl(ctrl_ov)
            dllm = make_dllm(decoder_t0); torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            ov_times.append(t1-t0); ov_fwds.append(dllm.diff_iteration.num_forwards)
            print(f"    Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd "
                  f"| cold={ctrl_ov.cold_count} hot={ctrl_ov.hot_count}")
        results['C10_overlap'] = {
            'avg_time': sum(ov_times)/2, 'avg_fwd': sum(ov_fwds)/2,
            'ms_per_fwd': sum(ov_times)/2 / (sum(ov_fwds)/2) * 1000,
            'cold': ctrl_ov.cold_count, 'hot': ctrl_ov.hot_count}

        # ---- Quality check ----
        print(f"\n{'='*60}")
        print("Quality check — C10-overlap (temp=0.7)")
        print(f"{'='*60}")
        restore()
        ctrl_q = OverlapEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8)
        patch_routing(ctrl_q)
        decoder_t7 = ThresholdParallelDecoder(
            temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm_q = make_dllm(decoder_t7)
        with torch.inference_mode():
            _ = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        reset_ctrl(ctrl_q)
        dllm_q = make_dllm(decoder_t7)
        with torch.inference_mode():
            out = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        gen_tokens = out[:, prompt_len:]
        for bi in sorted(VERIFIABLE.keys()):
            gt = gen_tokens[bi]
            valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
            text = tokenizer.decode(valid, skip_special_tokens=True)
            print(f"  #{bi}: {text[:150]}")

        # ---- Summary ----
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        c5_t = results['C5']['avg_time']
        c5_f = results['C5']['avg_fwd']
        print(f"  {'Config':<35s} {'Time(s)':>8s} {'Fwd':>5s} {'ms/fwd':>8s} {'vs C5':>8s} {'dFwd':>6s}")
        print(f"  {'-'*73}")
        for cn, r in results.items():
            d = (r['avg_time'] - c5_t) / c5_t * 100
            df = r['avg_fwd'] - c5_f
            print(f"  {cn:<35s} {r['avg_time']:>8.3f} {r['avg_fwd']:>5.0f} "
                  f"{r['ms_per_fwd']:>8.2f} {d:>+7.1f}% {df:>+5.0f}")

        # Reference: C10-delay (simulation) was 8.728s, 272 fwd
        print(f"\n  Reference: C10-delay qm=1.0 (simulation, no overlap) = 8.728s, 272 fwd")
        if 'C10_overlap' in results:
            ov_t = results['C10_overlap']['avg_time']
            hidden = 8.728 - ov_t
            print(f"  EB time hidden by overlap: {hidden:.3f}s ({hidden/8.728*100:.1f}%)")

        out_path = REPO_ROOT / "codex_coding" / "results" / "c10_overlap_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
