#!/usr/bin/env python3
"""
v0.1.15.7 — Fused MoE Routing Triton Kernel

Fuses the 12 post-GEMM routing ops into a single Triton kernel:
  sigmoid → group_limited_topk → gather → normalize → scale

Step 1: Micro-benchmark (correctness + kernel speed)
Step 2: Integration monkey-patch into gate.routing
Step 3: E2E HetEval-32 timing

Model params: E=256, K=8, n_group=8, topk_group=4, group_size=32
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

# ================================================================
# Fused MoE Routing Triton Kernel
# ================================================================

@triton.jit
def _fused_moe_routing_kernel(
    # Pointers
    logits_ptr,         # [N, E] float32
    expert_bias_ptr,    # [E] float32
    topk_ids_ptr,       # [N, K] int32
    topk_weights_ptr,   # [N, K] float32
    # Scalars
    N,                  # number of tokens
    routed_scaling_factor,  # float
    stride_logits_n,    # logits stride for token dim
    stride_logits_e,    # logits stride for expert dim
    stride_ids_n,
    stride_ids_k,
    stride_w_n,
    stride_w_k,
    # Constants
    E: tl.constexpr,            # 256
    K: tl.constexpr,            # 8
    N_GROUP: tl.constexpr,      # 8
    TOPK_GROUP: tl.constexpr,   # 4
    GROUP_SIZE: tl.constexpr,   # 32 = E / N_GROUP
):
    """Fused routing kernel: one program per token."""
    pid = tl.program_id(0)
    if pid >= N:
        return

    # ---- Load logits[pid, :E] and expert_bias[:E] ----
    offs_e = tl.arange(0, E)
    logits = tl.load(logits_ptr + pid * stride_logits_n + offs_e * stride_logits_e)
    bias = tl.load(expert_bias_ptr + offs_e)

    # ---- Op 1-2: sigmoid + add bias ----
    scores = tl.sigmoid(logits)
    scores_biased = scores + bias

    # ---- Op 3-6: Group-limited topk (find top-4 groups out of 8) ----
    # For each group g in [0, N_GROUP), find top-2 experts, sum → group_score
    # Then select top-4 groups
    # group_scores: [N_GROUP]
    # We compute this by iterating over groups

    # Compute group scores: for each group, sum of top-2 values
    group_scores = tl.zeros([N_GROUP], dtype=tl.float32)
    for g in tl.static_range(N_GROUP):
        g_start = g * GROUP_SIZE
        g_offs = tl.arange(0, GROUP_SIZE)
        g_vals = tl.load(
            logits_ptr + pid * stride_logits_n + (g_start + g_offs) * stride_logits_e
        )
        g_scores = tl.sigmoid(g_vals) + tl.load(expert_bias_ptr + g_start + g_offs)

        # Top-1
        max1 = tl.max(g_scores, axis=0)
        # Top-2: mask out max1 position, find next max
        g_scores_m = tl.where(g_scores == max1, float('-inf'), g_scores)
        max2 = tl.max(g_scores_m, axis=0)
        # Handle case where max2 is -inf (all same values)
        max2 = tl.where(max2 == float('-inf'), 0.0, max2)

        group_scores = tl.where(
            tl.arange(0, N_GROUP) == g,
            max1 + max2,
            group_scores
        )

    # Find top-4 groups: iterative argmax on group_scores [N_GROUP=8]
    group_mask = tl.zeros([N_GROUP], dtype=tl.int32)
    gs_temp = group_scores
    for _k in tl.static_range(TOPK_GROUP):
        g_max_val = tl.max(gs_temp, axis=0)
        g_max_idx = tl.argmax(gs_temp, axis=0)
        group_mask = tl.where(
            tl.arange(0, N_GROUP) == g_max_idx,
            1,
            group_mask
        )
        gs_temp = tl.where(
            tl.arange(0, N_GROUP) == g_max_idx,
            float('-inf'),
            gs_temp
        )

    # ---- Op 7-9: Expand group_mask to [E] and apply ----
    # For expert e, its group = e // GROUP_SIZE
    expert_group = offs_e // GROUP_SIZE  # [E], values in [0, N_GROUP)
    # Gather group_mask for each expert
    expert_allowed = tl.zeros([E], dtype=tl.int32)
    for g in tl.static_range(N_GROUP):
        is_g = (expert_group == g)
        g_selected = tl.sum(tl.where(tl.arange(0, N_GROUP) == g, group_mask, tl.zeros([N_GROUP], dtype=tl.int32)))
        expert_allowed = tl.where(is_g, g_selected, expert_allowed)

    masked_scores = tl.where(expert_allowed == 1, scores_biased, float('-inf'))

    # ---- Op 10: Global topk(8) via iterative argmax ----
    topk_indices = tl.zeros([K], dtype=tl.int32)
    ms_temp = masked_scores
    for _k in tl.static_range(K):
        best_idx = tl.argmax(ms_temp, axis=0)
        topk_indices = tl.where(
            tl.arange(0, K) == _k,
            best_idx,
            topk_indices
        )
        ms_temp = tl.where(offs_e == best_idx, float('-inf'), ms_temp)

    # ---- Op 11-12: Gather original scores, normalize, scale ----
    # Gather scores at topk positions
    topk_scores = tl.zeros([K], dtype=tl.float32)
    for _k in tl.static_range(K):
        idx = tl.sum(tl.where(tl.arange(0, K) == _k, topk_indices, tl.zeros([K], dtype=tl.int32)))
        val = tl.sum(tl.where(offs_e == idx, scores, tl.zeros([E], dtype=tl.float32)))
        topk_scores = tl.where(tl.arange(0, K) == _k, val, topk_scores)

    # Normalize
    score_sum = tl.sum(topk_scores, axis=0) + 1e-20
    topk_weights = topk_scores / score_sum * routed_scaling_factor

    # ---- Store outputs ----
    offs_k = tl.arange(0, K)
    tl.store(topk_ids_ptr + pid * stride_ids_n + offs_k * stride_ids_k,
             topk_indices)
    tl.store(topk_weights_ptr + pid * stride_w_n + offs_k * stride_w_k,
             topk_weights)


def fused_moe_routing(
    logits: torch.Tensor,      # [N, E] float32
    expert_bias: torch.Tensor,  # [E] float32
    routed_scaling_factor: float,
    top_k: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Python wrapper for the fused routing kernel."""
    N, E = logits.shape
    assert E == 256, f"Expected E=256, got {E}"
    assert top_k == 8, f"Expected K=8, got {top_k}"
    assert n_group == 8, f"Expected n_group=8, got {n_group}"
    assert topk_group == 4, f"Expected topk_group=4, got {topk_group}"
    GROUP_SIZE = E // n_group  # 32

    topk_ids = torch.empty((N, top_k), dtype=torch.int32, device=logits.device)
    topk_weights = torch.empty((N, top_k), dtype=torch.float32, device=logits.device)

    # Ensure float32
    logits_f32 = logits.float() if logits.dtype != torch.float32 else logits
    bias_f32 = expert_bias.float() if expert_bias.dtype != torch.float32 else expert_bias

    grid = (N,)
    _fused_moe_routing_kernel[grid](
        logits_f32, bias_f32,
        topk_ids, topk_weights,
        N, routed_scaling_factor,
        logits_f32.stride(0), logits_f32.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        topk_weights.stride(0), topk_weights.stride(1),
        E=E, K=top_k, N_GROUP=n_group,
        TOPK_GROUP=topk_group, GROUP_SIZE=GROUP_SIZE,
    )

    return topk_weights, topk_ids


# ================================================================
# Reference (eager) implementation for correctness check
# ================================================================

def eager_moe_routing(logits, expert_bias, routed_scaling_factor,
                      top_k=8, n_group=8, topk_group=4):
    """Exact replica of gate.routing logic."""
    scores = torch.sigmoid(logits.float())
    scores_for_routing = scores + expert_bias
    N = scores.shape[0]
    E = scores.shape[1]
    group_size = E // n_group

    # group_limited_topk
    group_scores = scores_for_routing.view(N, n_group, group_size).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(N, n_group, group_size)
        .reshape(N, -1)
    )
    masked_scores = scores_for_routing.masked_fill(~score_mask.bool(), float('-inf'))
    _, topk_idx = torch.topk(masked_scores, k=top_k, dim=-1)

    # gather + normalize + scale
    gathered = torch.gather(scores, dim=1, index=topk_idx)
    topk_weight = gathered / (gathered.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * routed_scaling_factor

    return topk_weight, topk_idx


# ================================================================
# Main: Micro-bench + E2E
# ================================================================

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
    0: "average speed = 480/7 ~ 68.57 km/h",
    8: "x = 2 and x = 3",
    13: "B is true, C is true, D is true",
    19: "fib(10) = 55",
    28: "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune",
}


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    print("=" * 80)
    print("v0.1.15.7 — Fused MoE Routing Triton Kernel")
    print("=" * 80)

    # ============================================================
    # Step 1: Micro-benchmark
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 1: Micro-benchmark (correctness + speed)")
    print("=" * 70)

    results = OrderedDict()

    for N in [1024, 3200]:
        print(f"\n  N={N}")
        logits = torch.randn(N, 256, dtype=torch.float32, device=device)
        expert_bias = torch.randn(256, dtype=torch.float32, device=device) * 0.01
        rsf = 2.5

        # Correctness
        w_eager, idx_eager = eager_moe_routing(logits, expert_bias, rsf)
        w_fused, idx_fused = fused_moe_routing(logits, expert_bias, rsf)

        # Compare topk_ids (sorted per row for fair comparison, since topk order may differ)
        idx_eager_sorted = idx_eager.sort(dim=-1)[0]
        idx_fused_sorted = idx_fused.sort(dim=-1)[0]
        ids_match = (idx_eager_sorted == idx_fused_sorted).all().item()

        # Compare weights (after sorting by ids for alignment)
        w_eager_reorder = torch.gather(w_eager, 1, idx_eager.argsort(dim=-1))
        w_fused_reorder = torch.gather(w_fused, 1, idx_fused.argsort(dim=-1))
        # Re-sort both by sorted indices for alignment
        _, eager_order = idx_eager.sort(dim=-1)
        _, fused_order = idx_fused.sort(dim=-1)
        w_eager_aligned = torch.gather(w_eager, 1, eager_order)
        w_fused_aligned = torch.gather(w_fused, 1, fused_order)
        if ids_match:
            cos_sim = F.cosine_similarity(w_eager_aligned.flatten().unsqueeze(0),
                                           w_fused_aligned.flatten().unsqueeze(0)).item()
            max_diff = (w_eager_aligned - w_fused_aligned).abs().max().item()
        else:
            # If ids don't match perfectly, compute row-level match rate
            row_match = (idx_eager_sorted == idx_fused_sorted).all(dim=-1).float().mean().item()
            cos_sim = -1
            max_diff = -1

        print(f"    IDs exact match: {ids_match}")
        if ids_match:
            print(f"    Weight cosine sim: {cos_sim:.6f}")
            print(f"    Weight max diff: {max_diff:.8f}")
        else:
            print(f"    IDs row match rate: {row_match:.4f}")

        # Speed: eager
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            eager_moe_routing(logits, expert_bias, rsf)
        torch.cuda.synchronize()
        t_eager = (time.perf_counter() - t0) / 100 * 1000  # ms

        # Speed: fused
        # Warmup
        for _ in range(10):
            fused_moe_routing(logits, expert_bias, rsf)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            fused_moe_routing(logits, expert_bias, rsf)
        torch.cuda.synchronize()
        t_fused = (time.perf_counter() - t0) / 100 * 1000  # ms

        speedup = t_eager / t_fused if t_fused > 0 else 0
        print(f"    Eager: {t_eager:.3f} ms")
        print(f"    Fused: {t_fused:.3f} ms")
        print(f"    Speedup: {speedup:.2f}x")

        results[f"micro_N{N}"] = {
            "ids_match": ids_match,
            "cos_sim": cos_sim,
            "max_diff": max_diff,
            "eager_ms": t_eager,
            "fused_ms": t_fused,
            "speedup": speedup,
        }

    # Check if micro-bench passes
    any_fail = any(not r["ids_match"] for r in results.values() if "ids_match" in r)
    if any_fail:
        print("\n  ⚠ Correctness check FAILED. Skipping E2E test.")
        print("  Saving micro-bench results only.")
        out_path = REPO_ROOT / "codex_coding" / "results" / "fused_routing_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved to {out_path}")
        return

    # ============================================================
    # Step 2-3: E2E Integration + Timing
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 2-3: E2E Integration + HetEval-32 Timing")
    print("=" * 70)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoTokenizer, AutoConfig
    from baseline_optimizations import apply_all_optimizations

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

        # Build input
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
        padded = []
        for ids in all_ids:
            if ids.shape[0] < mx:
                ids = torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
            padded.append(ids)
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

        # ---- C0: Baseline ----
        print(f"\n{'='*60}")
        print("C0: Baseline (eager routing)")
        print(f"{'='*60}")

        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        c0_times, c0_fwds = [], []
        for run_i in range(2):
            dllm = make_dllm(decoder_t0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            c0_times.append(t1 - t0)
            c0_fwds.append(dllm.diff_iteration.num_forwards)
            print(f"  Run {run_i+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd, "
                  f"{dllm.diff_iteration.num_forwards/(t1-t0):.1f} fwd/s")

        c0_avg = sum(c0_times) / len(c0_times)
        c0_fwd = sum(c0_fwds) / len(c0_fwds)
        results["C0_baseline"] = {
            "avg_time": c0_avg, "avg_fwd": c0_fwd,
            "fwd_per_s": c0_fwd / c0_avg,
            "ms_per_fwd": c0_avg / c0_fwd * 1000,
        }

        # ---- C5: Fused routing kernel ----
        print(f"\n{'='*60}")
        print("C5: Fused Triton routing kernel (monkey-patch gate.routing)")
        print(f"{'='*60}")

        # Monkey-patch all gate.routing methods
        patched = 0
        for name, module in model.named_modules():
            if module.__class__.__name__ == "LLaDA2MoeGate":
                gate = module
                rsf = gate.routed_scaling_factor
                bias = gate.expert_bias
                top_k = gate.top_k
                ng = gate.n_group
                tkg = gate.topk_group

                def make_fused_routing(g_bias, g_rsf, g_topk, g_ng, g_tkg):
                    def fused_routing_fn(hidden_states, gating_output, topk, renormalize):
                        w, idx = fused_moe_routing(
                            gating_output, g_bias, g_rsf,
                            top_k=g_topk, n_group=g_ng, topk_group=g_tkg)
                        return w.to(gating_output.dtype), idx
                    return fused_routing_fn

                gate.routing = make_fused_routing(bias, rsf, top_k, ng, tkg)
                patched += 1

        print(f"  Patched {patched} gate.routing methods")

        # Warmup
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        # Timing
        c5_times, c5_fwds = [], []
        for run_i in range(2):
            dllm = make_dllm(decoder_t0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            c5_times.append(t1 - t0)
            c5_fwds.append(dllm.diff_iteration.num_forwards)
            print(f"  Run {run_i+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd, "
                  f"{dllm.diff_iteration.num_forwards/(t1-t0):.1f} fwd/s")

        c5_avg = sum(c5_times) / len(c5_times)
        c5_fwd = sum(c5_fwds) / len(c5_fwds)
        delta = (c5_avg - c0_avg) / c0_avg * 100
        fwd_match = c5_fwd == c0_fwd
        print(f"  C5 avg: {c5_avg:.3f}s, {c5_fwd:.0f} fwd, {c5_fwd/c5_avg:.1f} fwd/s")
        print(f"  vs C0: {delta:+.1f}%")
        print(f"  Fwd count match: {'YES' if fwd_match else 'NO'}")

        results["C5_fused_routing"] = {
            "avg_time": c5_avg, "avg_fwd": c5_fwd,
            "fwd_per_s": c5_fwd / c5_avg,
            "ms_per_fwd": c5_avg / c5_fwd * 1000,
            "delta_vs_c0_pct": delta,
            "fwd_count_match": fwd_match,
        }

        # Quality
        print("\n  Quality check (temp=0.7)...")
        decoder_t7 = ThresholdParallelDecoder(
            temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm_q = make_dllm(decoder_t7)
        with torch.inference_mode():
            dllm_q.diff_iteration.num_forwards = 0
            _ = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        with torch.inference_mode():
            dllm_q.diff_iteration.num_forwards = 0
            out = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        gen_tokens = out[:, prompt_len:]
        quality = {}
        for bi in sorted(VERIFIABLE.keys()):
            gt = gen_tokens[bi]
            valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
            text = tokenizer.decode(valid, skip_special_tokens=True)
            quality[bi] = text[:300]
            print(f"    #{bi}: {text[:150]}")
        results["C5_fused_routing"]["quality"] = quality

        # ---- Summary ----
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Config':<35s} {'Time(s)':>8s} {'Fwd':>5s} {'Fwd/s':>6s} {'ms/fwd':>7s} {'vs C0':>7s}")
        print(f"  {'-'*70}")
        for cname, r in results.items():
            if cname.startswith("micro_"):
                continue
            d = f"{r.get('delta_vs_c0_pct', 0):+.1f}%" if 'delta_vs_c0_pct' in r else "—"
            print(f"  {cname:<35s} {r['avg_time']:>8.3f} {r['avg_fwd']:>5.0f} "
                  f"{r['fwd_per_s']:>6.1f} {r['ms_per_fwd']:>7.2f} {d:>7s}")

        out_path = REPO_ROOT / "codex_coding" / "results" / "fused_routing_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
