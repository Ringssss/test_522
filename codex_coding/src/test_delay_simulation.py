#!/usr/bin/env python3
"""
v0.1.15.8h — Step 1: Simulate 1-forward delay on S_mask

Each layer uses the PREVIOUS forward's s_mask (not current).
Cold start: compute fresh s_mask (no delay).
Purpose: measure fwd count impact of 1-forward delay (quality check).

Tests both q_major=0.95 and q_major=1.0.
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
# Delayed S_mask Controller: uses previous forward's s_mask
# ================================================================
class DelayedEBController(FusedEBController):
    """
    Hot path: return s_mask_old (from previous forward), then compute s_mask_new.
    Simulates 1-forward delay without actual stream overlap.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_init_history = []
        # Per-layer double buffer: s_mask_old (returned) and s_mask_new (computed)
        self.s_mask_cache = {}  # layer_idx → tensor[256] (previous forward's s_mask)

    def cold_path(self, layer_idx, logits, bias):
        """Cold path: compute fresh s_mask, store in cache. No delay."""
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

        # Store in cache for next forward to use as s_mask_old
        self.s_mask_cache[layer_idx] = b['s_mask'].clone()

        self.cold_count += 1
        return b['s_mask']  # cold path returns fresh (no delay)

    def hot_path(self, layer_idx, logits, bias):
        """
        1. Return s_mask_old (from cache) immediately
        2. Compute s_mask_new and store in cache for next forward
        """
        N, E = logits.shape
        K_init = self.K_init.get(layer_idx, 103)
        b = self._get_bufs(N, E, logits.device)

        # Get old s_mask to return
        s_mask_old = self.s_mask_cache.get(layer_idx, None)
        if s_mask_old is None:
            # Shouldn't happen (cold always runs first), but fallback
            s_mask_old = torch.ones(E, device=logits.device, dtype=torch.int32)

        # Compute new s_mask (same as normal hot path)
        lf = logits.float()
        _kernel_A[(N,)](lf, bias.float(), b['pop'],
                        N, self.rsf, lf.stride(0), lf.stride(1),
                        E=E, KEXT=self.K_ext, KEXT_PAD=16)
        _kernel_B_v3[(1,)](b['pop'], b['s_mask'], K_init, E=E)

        # Store new s_mask for next forward
        self.s_mask_cache[layer_idx] = b['s_mask'].clone()

        self.hot_count += 1
        return s_mask_old  # return OLD s_mask (1-forward delay)


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
    print("Step 1: 1-Forward Delay S_mask Simulation")
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

        def run_config(name, eb_ctrl):
            restore()
            patch_routing(eb_ctrl)

            # Warmup
            dllm = make_dllm(decoder_t0)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()

            eb_ctrl.prev_N.clear(); eb_ctrl.K_init.clear()
            eb_ctrl.cold_count = 0; eb_ctrl.hot_count = 0
            eb_ctrl._bufs.clear(); eb_ctrl.k_init_history.clear()
            if hasattr(eb_ctrl, 's_mask_cache'):
                eb_ctrl.s_mask_cache.clear()

            # Timing
            times, fwds = [], []
            for ri in range(2):
                eb_ctrl.prev_N.clear(); eb_ctrl.K_init.clear()
                eb_ctrl.cold_count = 0; eb_ctrl.hot_count = 0
                eb_ctrl._bufs.clear(); eb_ctrl.k_init_history.clear()
                if hasattr(eb_ctrl, 's_mask_cache'):
                    eb_ctrl.s_mask_cache.clear()

                dllm = make_dllm(decoder_t0)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                      block_length=BLOCK_LENGTH)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)
                fwds.append(dllm.diff_iteration.num_forwards)
                print(f"    Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd "
                      f"| cold={eb_ctrl.cold_count} hot={eb_ctrl.hot_count}")

            avg_t = sum(times) / len(times)
            avg_f = sum(fwds) / len(fwds)
            return {'avg_time': avg_t, 'avg_fwd': avg_f,
                    'ms_per_fwd': avg_t / avg_f * 1000,
                    'cold': eb_ctrl.cold_count, 'hot': eb_ctrl.hot_count}

        results = OrderedDict()

        # C5 reference (no EB)
        print(f"\n{'='*60}")
        print("C5: Fused routing only (reference)")
        print(f"{'='*60}")
        restore()
        idx = 0
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                     mod.top_k, mod.n_group, mod.topk_group)
                def mk_c5(bb, rr, tt, nn, gg):
                    def fn(hs, go, topk, renorm):
                        w, i = fused_routing(go, bb, rr, s_mask=None,
                                             K=tt, ng=nn, tkg=gg)
                        return w.to(go.dtype), i
                    return fn
                mod.routing = mk_c5(b, r, tk, ng, tkg)
                idx += 1

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
            c5_times.append(t1 - t0); c5_fwds.append(dllm.diff_iteration.num_forwards)
            print(f"    Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd")
        results['C5'] = {'avg_time': sum(c5_times)/2, 'avg_fwd': sum(c5_fwds)/2,
                         'ms_per_fwd': sum(c5_times)/2 / (sum(c5_fwds)/2) * 1000}

        # Test configurations
        for qm in [0.95, 1.0]:
            # No delay (normal C10-v6)
            label_no = f"C10-v6 qm={qm} (no delay)"
            print(f"\n{'='*60}")
            print(label_no)
            print(f"{'='*60}")
            from test_c10_v6 import FusedEBControllerV6
            ctrl_no = FusedEBControllerV6(
                num_layers=19, K=8, M=4, K_target=40,
                quality_floor=0.70, q_major=qm, per_round_cap=8)
            results[label_no] = run_config(label_no, ctrl_no)

            # With 1-forward delay
            label_d = f"C10-delay qm={qm} (1-fwd delay)"
            print(f"\n{'='*60}")
            print(label_d)
            print(f"{'='*60}")
            ctrl_d = DelayedEBController(
                num_layers=19, K=8, M=4, K_target=40,
                quality_floor=0.70, q_major=qm, per_round_cap=8)
            results[label_d] = run_config(label_d, ctrl_d)

        # Quality check: delay + q_major=1.0
        print(f"\n{'='*60}")
        print("Quality check — C10-delay qm=1.0 (temp=0.7)")
        print(f"{'='*60}")
        restore()
        ctrl_q = DelayedEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8)
        patch_routing(ctrl_q)
        decoder_t7 = ThresholdParallelDecoder(
            temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm_q = make_dllm(decoder_t7)
        with torch.inference_mode():
            _ = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        ctrl_q.prev_N.clear(); ctrl_q.K_init.clear()
        ctrl_q.cold_count = 0; ctrl_q.hot_count = 0
        ctrl_q._bufs.clear()
        if hasattr(ctrl_q, 's_mask_cache'):
            ctrl_q.s_mask_cache.clear()
        dllm_q = make_dllm(decoder_t7)
        with torch.inference_mode():
            out = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        gen_tokens = out[:, prompt_len:]
        for bi in sorted(VERIFIABLE.keys()):
            gt = gen_tokens[bi]
            valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
            text = tokenizer.decode(valid, skip_special_tokens=True)
            print(f"  #{bi}: {text[:150]}")

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        c5_t = results['C5']['avg_time']
        c5_f = results['C5']['avg_fwd']
        print(f"  {'Config':<40s} {'Time(s)':>8s} {'Fwd':>5s} {'ms/fwd':>8s} "
              f"{'vs C5':>8s} {'dFwd':>6s}")
        print(f"  {'-'*78}")
        for cn, r in results.items():
            d = (r['avg_time'] - c5_t) / c5_t * 100
            df = r['avg_fwd'] - c5_f
            print(f"  {cn:<40s} {r['avg_time']:>8.3f} {r['avg_fwd']:>5.0f} "
                  f"{r['ms_per_fwd']:>8.2f} {d:>+7.1f}% {df:>+5.0f}")

        out_path = REPO_ROOT / "codex_coding" / "results" / "delay_simulation_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
