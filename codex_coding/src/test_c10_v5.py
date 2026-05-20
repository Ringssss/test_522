#!/usr/bin/env python3
"""
v0.1.15.8e — New K_A_v2 (local_pop) + K_B_v3 (tl.sort)

Phase 1: Micro-bench (correctness + kernel speed)
Phase 2: E2E (C5 vs C10-v3 vs C10-v5)

Based on: docx/cites/warmstart_triton_final.md
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
GEN_LENGTH = 64

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


# ================================================================
# NEW K_A_v2: local_pop accumulation, reduced atomic contention
# ================================================================
@triton.jit
def _kernel_A_v2(
    logits_ptr, bias_ptr, pop_ptr,
    N, rsf,
    stride_n, stride_e,
    E: tl.constexpr,          # 256
    BLOCK_N: tl.constexpr,    # tokens per program
    KEXT: tl.constexpr,       # 12
    KEXT_PAD: tl.constexpr,   # 16 (power of 2)
):
    pid = tl.program_id(0)
    expert_offsets = tl.arange(0, E)
    bias = tl.load(bias_ptr + expert_offsets).to(tl.float32)
    local_pop = tl.zeros([E], dtype=tl.float32)

    for i in tl.static_range(BLOCK_N):
        tok = pid * BLOCK_N + i
        valid_tok = tok < N

        lg = tl.load(
            logits_ptr + tok * stride_n + expert_offsets * stride_e,
            mask=valid_tok, other=-float("inf"),
        ).to(tl.float32)

        scores = tl.sigmoid(lg) + bias

        # top-12 via iterative argmax (padded to 16)
        st = scores
        top_val = tl.zeros([KEXT_PAD], dtype=tl.float32)
        top_idx = tl.zeros([KEXT_PAD], dtype=tl.int32)
        for k in tl.static_range(KEXT):
            bx = tl.argmax(st, axis=0)
            bv = tl.max(st, axis=0)
            top_idx = tl.where(tl.arange(0, KEXT_PAD) == k, bx, top_idx)
            top_val = tl.where(tl.arange(0, KEXT_PAD) == k, bv, top_val)
            st = tl.where(expert_offsets == bx, -float("inf"), st)

        # normalize (only first KEXT valid)
        valid = tl.arange(0, KEXT_PAD) < KEXT
        top_val = tl.where(valid, top_val, tl.zeros([KEXT_PAD], dtype=tl.float32))
        s_sum = tl.sum(top_val, axis=0) + 1e-20
        top_w = top_val / s_sum * rsf

        # accumulate into local_pop (no atomics here)
        for k in tl.static_range(KEXT):
            idx = tl.sum(tl.where(tl.arange(0, KEXT_PAD) == k, top_idx,
                                  tl.zeros([KEXT_PAD], dtype=tl.int32)))
            w = tl.sum(tl.where(tl.arange(0, KEXT_PAD) == k, top_w,
                                tl.zeros([KEXT_PAD], dtype=tl.float32)))
            local_pop = tl.where(expert_offsets == idx, local_pop + w, local_pop)

    # one atomic_add per program (vector of 256)
    tl.atomic_add(pop_ptr + expert_offsets, local_pop)


# ================================================================
# NEW K_B_v3: tl.sort based, O(E log E) parallel
# ================================================================
@triton.jit
def _kernel_B_v3(
    pop_ptr,          # [256] float32
    s_mask_ptr,       # [256] int32
    K_init,           # runtime int
    E: tl.constexpr,  # 256
):
    offs = tl.arange(0, E)
    pop = tl.load(pop_ptr + offs).to(tl.float32)

    # parallel sort descending
    pop_sorted = tl.sort(pop, descending=True)

    # threshold = value at position K_init-1
    kth_idx = K_init - 1
    threshold = tl.max(
        tl.where(offs == kth_idx, pop_sorted, -float("inf")),
        axis=0,
    )

    # elements strictly greater than threshold -> always in
    gt_mask = pop > threshold
    gt_count = tl.sum(gt_mask.to(tl.int32), axis=0)

    # tie-break among equal values by expert index ascending
    eq_mask = pop == threshold
    need_eq = K_init - gt_count
    eq_rank = tl.cumsum(eq_mask.to(tl.int32), axis=0)
    take_eq = eq_mask & (eq_rank <= need_eq)

    sel = gt_mask | take_eq
    tl.store(s_mask_ptr + offs, sel.to(tl.int32))

    # clear popularity for next call
    tl.store(pop_ptr + offs, tl.zeros([E], dtype=tl.float32))


# ================================================================
# Python reference for correctness check
# ================================================================
def compute_s_mask_python_ref(logits, bias, k_init, rsf=2.5):
    scores_full = torch.sigmoid(logits.float()) + bias.float()
    topkm_score, topkm_idx = torch.topk(scores_full, k=12, dim=1)
    topkm_weight = topkm_score / (topkm_score.sum(dim=1, keepdim=True) + 1e-20) * rsf
    popularity = torch.zeros(256, device=logits.device, dtype=torch.float32)
    popularity.scatter_add_(0, topkm_idx.reshape(-1), topkm_weight.reshape(-1))
    _, pop_order = popularity.sort(descending=True)
    s_mask = torch.zeros(256, dtype=torch.int32, device=logits.device)
    s_mask[pop_order[:k_init]] = 1
    return s_mask


# ================================================================
# V5 Controller: new K_A_v2 + K_B_v3
# ================================================================
class FusedEBControllerV5(FusedEBController):
    def __init__(self, *args, block_n=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_n = block_n
        self.k_init_history = []

    def cold_path(self, layer_idx, logits, bias):
        """Original cold path + store actual |S|."""
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
        self.cold_count += 1
        return b['s_mask']

    def hot_path(self, layer_idx, logits, bias):
        """New hot path: K_A_v2 (local_pop) + K_B_v3 (tl.sort)."""
        N, E = logits.shape
        K_init = self.K_init.get(layer_idx, 103)
        b = self._get_bufs(N, E, logits.device)
        lf = logits.float()

        grid_a = (triton.cdiv(N, self.block_n),)
        _kernel_A_v2[grid_a](
            lf, bias.float(), b['pop'],
            N, self.rsf, lf.stride(0), lf.stride(1),
            E=E, BLOCK_N=self.block_n, KEXT=self.K_ext, KEXT_PAD=16)

        _kernel_B_v3[(1,)](
            b['pop'], b['s_mask'], K_init,
            E=E)

        self.hot_count += 1
        return b['s_mask']


# ================================================================
# Main
# ================================================================
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    print("=" * 80)
    print("v0.1.15.8e — New K_A_v2 (local_pop) + K_B_v3 (tl.sort)")
    print("=" * 80)

    # ============================================================
    # Phase 1: Micro-bench
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Micro-benchmark (correctness + kernel speed)")
    print("=" * 60)

    results = OrderedDict()

    for N in [1024, 3200]:
        K_init = 91
        print(f"\n  N={N}, K_init={K_init}")

        logits = torch.randn(N, 256, dtype=torch.float32, device=device)
        bias = torch.randn(256, dtype=torch.float32, device=device) * 0.01
        rsf = 2.5

        # --- Correctness: Python ref ---
        ref_mask = compute_s_mask_python_ref(logits, bias, K_init, rsf)

        # --- Correctness: Old K_A + K_B_v2 ---
        pop_old = torch.zeros(256, device=device, dtype=torch.float32)
        _kernel_A[(N,)](logits, bias, pop_old,
                        N, rsf, logits.stride(0), logits.stride(1),
                        E=256, KEXT=12, KEXT_PAD=16)
        mask_old = torch.zeros(256, device=device, dtype=torch.int32)
        _kernel_B_v2[(1,)](pop_old, mask_old, K_init, E=256, MAX_K=256)

        # --- Correctness: New K_A_v2 + K_B_v3 ---
        for BN in [8, 16, 32]:
            pop_new = torch.zeros(256, device=device, dtype=torch.float32)
            mask_new = torch.zeros(256, device=device, dtype=torch.int32)
            grid_a = (triton.cdiv(N, BN),)
            _kernel_A_v2[grid_a](logits, bias, pop_new,
                                 N, rsf, logits.stride(0), logits.stride(1),
                                 E=256, BLOCK_N=BN, KEXT=12, KEXT_PAD=16)
            _kernel_B_v3[(1,)](pop_new, mask_new, K_init, E=256)

            sum_new = int(mask_new.sum().item())
            overlap_ref = int((ref_mask & mask_new).sum().item())
            jaccard_ref = overlap_ref / int((ref_mask | mask_new).sum().item()) if int((ref_mask | mask_new).sum().item()) > 0 else 0
            overlap_old = int((mask_old & mask_new).sum().item())
            jaccard_old = overlap_old / int((mask_old | mask_new).sum().item()) if int((mask_old | mask_new).sum().item()) > 0 else 0

            print(f"    BN={BN}: sum={sum_new} (expect {K_init}), "
                  f"Jaccard vs ref={jaccard_ref:.4f}, vs old={jaccard_old:.4f}")

        # --- Speed: K_A old vs new ---
        print(f"\n    Kernel speed (100 runs each):")

        # Old K_A
        for _ in range(10):
            pop_t = torch.zeros(256, device=device, dtype=torch.float32)
            _kernel_A[(N,)](logits, bias, pop_t, N, rsf,
                            logits.stride(0), logits.stride(1),
                            E=256, KEXT=12, KEXT_PAD=16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            pop_t = torch.zeros(256, device=device, dtype=torch.float32)
            _kernel_A[(N,)](logits, bias, pop_t, N, rsf,
                            logits.stride(0), logits.stride(1),
                            E=256, KEXT=12, KEXT_PAD=16)
        torch.cuda.synchronize()
        t_ka_old = (time.perf_counter() - t0) / 100 * 1000

        # New K_A_v2 (sweep BLOCK_N)
        for BN in [8, 16, 32]:
            for _ in range(10):
                pop_t = torch.zeros(256, device=device, dtype=torch.float32)
                _kernel_A_v2[(triton.cdiv(N, BN),)](
                    logits, bias, pop_t, N, rsf,
                    logits.stride(0), logits.stride(1),
                    E=256, BLOCK_N=BN, KEXT=12, KEXT_PAD=16)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                pop_t = torch.zeros(256, device=device, dtype=torch.float32)
                _kernel_A_v2[(triton.cdiv(N, BN),)](
                    logits, bias, pop_t, N, rsf,
                    logits.stride(0), logits.stride(1),
                    E=256, BLOCK_N=BN, KEXT=12, KEXT_PAD=16)
            torch.cuda.synchronize()
            t_ka_new = (time.perf_counter() - t0) / 100 * 1000
            print(f"      K_A old: {t_ka_old:.3f}ms | K_A_v2 BN={BN}: {t_ka_new:.3f}ms | "
                  f"speedup: {t_ka_old/t_ka_new:.2f}x")

        # Old K_B_v2 vs New K_B_v3
        pop_for_kb = torch.zeros(256, device=device, dtype=torch.float32)
        _kernel_A[(N,)](logits, bias, pop_for_kb, N, rsf,
                        logits.stride(0), logits.stride(1),
                        E=256, KEXT=12, KEXT_PAD=16)
        pop_snapshot = pop_for_kb.clone()

        # Old K_B_v2
        for _ in range(10):
            pop_t = pop_snapshot.clone()
            mask_t = torch.zeros(256, device=device, dtype=torch.int32)
            _kernel_B_v2[(1,)](pop_t, mask_t, K_init, E=256, MAX_K=256)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            pop_t = pop_snapshot.clone()
            mask_t = torch.zeros(256, device=device, dtype=torch.int32)
            _kernel_B_v2[(1,)](pop_t, mask_t, K_init, E=256, MAX_K=256)
        torch.cuda.synchronize()
        t_kb_old = (time.perf_counter() - t0) / 100 * 1000

        # New K_B_v3
        for _ in range(10):
            pop_t = pop_snapshot.clone()
            mask_t = torch.zeros(256, device=device, dtype=torch.int32)
            _kernel_B_v3[(1,)](pop_t, mask_t, K_init, E=256)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            pop_t = pop_snapshot.clone()
            mask_t = torch.zeros(256, device=device, dtype=torch.int32)
            _kernel_B_v3[(1,)](pop_t, mask_t, K_init, E=256)
        torch.cuda.synchronize()
        t_kb_new = (time.perf_counter() - t0) / 100 * 1000

        print(f"      K_B_v2: {t_kb_old:.3f}ms | K_B_v3: {t_kb_new:.3f}ms | "
              f"speedup: {t_kb_old/t_kb_new:.2f}x")

        results[f"micro_N{N}"] = {
            "K_A_old_ms": t_ka_old, "K_B_v2_ms": t_kb_old,
            "K_B_v3_ms": t_kb_new, "K_B_speedup": t_kb_old / t_kb_new,
        }

    # ============================================================
    # Phase 2: E2E
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 2: E2E Integration")
    print("=" * 60)

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

        def patch_routing(eb_ctrl=None):
            idx = 0
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate":
                    b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                         mod.top_k, mod.n_group, mod.topk_group)
                    li = idx
                    def mk(bb, rr, tt, nn, gg, layer_i, cc):
                        def fn(hs, go, topk, renorm):
                            sm = cc.get_s_mask(layer_i, go, bb) if cc else None
                            w, i = fused_routing(go, bb, rr, s_mask=sm,
                                                 K=tt, ng=nn, tkg=gg)
                            return w.to(go.dtype), i
                        return fn
                    mod.routing = mk(b, r, tk, ng, tkg, li, eb_ctrl)
                    idx += 1
            return idx

        def run_config(name, eb_ctrl=None):
            restore()
            patch_routing(eb_ctrl)

            dllm = make_dllm(decoder_t0)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()

            if eb_ctrl:
                eb_ctrl.prev_N.clear(); eb_ctrl.K_init.clear()
                eb_ctrl.cold_count = 0; eb_ctrl.hot_count = 0
                eb_ctrl._bufs.clear()
                if hasattr(eb_ctrl, 'k_init_history'):
                    eb_ctrl.k_init_history.clear()

            times, fwds = [], []
            for ri in range(3):
                if eb_ctrl:
                    eb_ctrl.prev_N.clear(); eb_ctrl.K_init.clear()
                    eb_ctrl.cold_count = 0; eb_ctrl.hot_count = 0
                    eb_ctrl._bufs.clear()
                    if hasattr(eb_ctrl, 'k_init_history'):
                        eb_ctrl.k_init_history.clear()

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
                extra = ""
                if eb_ctrl:
                    extra = f" | cold={eb_ctrl.cold_count} hot={eb_ctrl.hot_count}"
                print(f"    Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd{extra}")

            avg_t = sum(times) / len(times)
            avg_f = sum(fwds) / len(fwds)
            return {'avg_time': avg_t, 'avg_fwd': avg_f,
                    'ms_per_fwd': avg_t / avg_f * 1000,
                    'cold': eb_ctrl.cold_count if eb_ctrl else 0,
                    'hot': eb_ctrl.hot_count if eb_ctrl else 0}

        # C5
        print(f"\n{'='*60}")
        print("C5: Fused routing only")
        print(f"{'='*60}")
        results['C5'] = run_config("C5")

        # C10-v3: old K_A + K_B_v2 + actual |S|
        print(f"\n{'='*60}")
        print("C10-v3: old K_A + K_B_v2, K_init=actual |S|")
        print(f"{'='*60}")
        from test_c10_v3 import FusedEBControllerV3
        ctrl_v3 = FusedEBControllerV3(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=0.95, per_round_cap=8)
        results['C10_v3'] = run_config("C10-v3", ctrl_v3)

        # C10-v5: new K_A_v2 + K_B_v3 + actual |S|
        print(f"\n{'='*60}")
        print("C10-v5: K_A_v2 (local_pop) + K_B_v3 (tl.sort), K_init=actual |S|")
        print(f"{'='*60}")
        ctrl_v5 = FusedEBControllerV5(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=0.95, per_round_cap=8, block_n=16)
        results['C10_v5'] = run_config("C10-v5", ctrl_v5)

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        c5_t = results['C5']['avg_time']
        print(f"  {'Config':<50s} {'Time(s)':>8s} {'Fwd':>5s} {'ms/fwd':>8s} {'vs C5':>8s}")
        print(f"  {'-'*82}")
        for cn, r in results.items():
            if cn.startswith("micro_"):
                continue
            d = (r['avg_time'] - c5_t) / c5_t * 100
            print(f"  {cn:<50s} {r['avg_time']:>8.3f} {r['avg_fwd']:>5.0f} "
                  f"{r['ms_per_fwd']:>8.2f} {d:>+7.1f}%")

        out_path = REPO_ROOT / "codex_coding" / "results" / "c10_v5_comparison.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
