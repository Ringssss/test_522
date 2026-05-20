#!/usr/bin/env python3
"""
v0.1.15.7b — Fused Routing + Expert Budgeting (Plan B & C)

Stacks Expert Budgeting on top of the fused routing kernel:
  C5:  Fused routing only (re-confirm ~7.7s)
  C6c: Plan C — per-block popularity top-K_target, reuse S across iterations
       K_target sweep: 120, 100, 80
  C6b: Plan B — per-block batch-add (first iter), reuse S across iterations

Key design:
  - S_mask computed once per block (~8 times total), reused across ~33 iterations
  - Fused routing kernel extended to accept S_mask (one extra load + AND)
  - Per-block refresh detected by model forward counter
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

# ================================================================
# Fused Routing Kernel WITH S_mask support
# ================================================================

@triton.jit
def _fused_moe_routing_kernel_v2(
    logits_ptr, expert_bias_ptr,
    s_mask_ptr,             # [E] int32, 1=allowed 0=blocked. NULL if no mask.
    topk_ids_ptr, topk_weights_ptr,
    N, routed_scaling_factor,
    stride_logits_n, stride_logits_e,
    stride_ids_n, stride_ids_k,
    stride_w_n, stride_w_k,
    HAS_S_MASK: tl.constexpr,
    E: tl.constexpr, K: tl.constexpr,
    N_GROUP: tl.constexpr, TOPK_GROUP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    offs_e = tl.arange(0, E)
    logits = tl.load(logits_ptr + pid * stride_logits_n + offs_e * stride_logits_e)
    bias = tl.load(expert_bias_ptr + offs_e)

    scores = tl.sigmoid(logits)
    scores_biased = scores + bias

    # Group-limited topk
    group_scores = tl.zeros([N_GROUP], dtype=tl.float32)
    for g in tl.static_range(N_GROUP):
        g_start = g * GROUP_SIZE
        g_offs = tl.arange(0, GROUP_SIZE)
        g_vals = tl.load(logits_ptr + pid * stride_logits_n + (g_start + g_offs) * stride_logits_e)
        g_scores = tl.sigmoid(g_vals) + tl.load(expert_bias_ptr + g_start + g_offs)
        max1 = tl.max(g_scores, axis=0)
        g_scores_m = tl.where(g_scores == max1, float('-inf'), g_scores)
        max2 = tl.max(g_scores_m, axis=0)
        max2 = tl.where(max2 == float('-inf'), 0.0, max2)
        group_scores = tl.where(tl.arange(0, N_GROUP) == g, max1 + max2, group_scores)

    group_mask = tl.zeros([N_GROUP], dtype=tl.int32)
    gs_temp = group_scores
    for _k in tl.static_range(TOPK_GROUP):
        g_max_idx = tl.argmax(gs_temp, axis=0)
        group_mask = tl.where(tl.arange(0, N_GROUP) == g_max_idx, 1, group_mask)
        gs_temp = tl.where(tl.arange(0, N_GROUP) == g_max_idx, float('-inf'), gs_temp)

    expert_group = offs_e // GROUP_SIZE
    expert_allowed = tl.zeros([E], dtype=tl.int32)
    for g in tl.static_range(N_GROUP):
        is_g = (expert_group == g)
        g_sel = tl.sum(tl.where(tl.arange(0, N_GROUP) == g, group_mask, tl.zeros([N_GROUP], dtype=tl.int32)))
        expert_allowed = tl.where(is_g, g_sel, expert_allowed)

    # Apply S_mask if provided (AND with group mask)
    if HAS_S_MASK:
        s_mask = tl.load(s_mask_ptr + offs_e)
        expert_allowed = expert_allowed & s_mask

    masked_scores = tl.where(expert_allowed == 1, scores_biased, float('-inf'))

    # Global topk(8) via iterative argmax
    topk_indices = tl.zeros([K], dtype=tl.int32)
    ms_temp = masked_scores
    for _k in tl.static_range(K):
        best_idx = tl.argmax(ms_temp, axis=0)
        topk_indices = tl.where(tl.arange(0, K) == _k, best_idx, topk_indices)
        ms_temp = tl.where(offs_e == best_idx, float('-inf'), ms_temp)

    topk_scores = tl.zeros([K], dtype=tl.float32)
    for _k in tl.static_range(K):
        idx = tl.sum(tl.where(tl.arange(0, K) == _k, topk_indices, tl.zeros([K], dtype=tl.int32)))
        val = tl.sum(tl.where(offs_e == idx, scores, tl.zeros([E], dtype=tl.float32)))
        topk_scores = tl.where(tl.arange(0, K) == _k, val, topk_scores)

    score_sum = tl.sum(topk_scores, axis=0) + 1e-20
    topk_weights = topk_scores / score_sum * routed_scaling_factor

    offs_k = tl.arange(0, K)
    tl.store(topk_ids_ptr + pid * stride_ids_n + offs_k * stride_ids_k, topk_indices)
    tl.store(topk_weights_ptr + pid * stride_w_n + offs_k * stride_w_k, topk_weights)


def fused_moe_routing_v2(logits, expert_bias, routed_scaling_factor,
                          s_mask=None, top_k=8, n_group=8, topk_group=4):
    N, E = logits.shape
    GROUP_SIZE = E // n_group
    topk_ids = torch.empty((N, top_k), dtype=torch.int32, device=logits.device)
    topk_weights = torch.empty((N, top_k), dtype=torch.float32, device=logits.device)
    logits_f32 = logits.float() if logits.dtype != torch.float32 else logits
    bias_f32 = expert_bias.float() if expert_bias.dtype != torch.float32 else expert_bias

    has_mask = s_mask is not None
    s_mask_ptr = s_mask if has_mask else torch.empty(0, dtype=torch.int32, device=logits.device)

    _fused_moe_routing_kernel_v2[(N,)](
        logits_f32, bias_f32, s_mask_ptr,
        topk_ids, topk_weights,
        N, routed_scaling_factor,
        logits_f32.stride(0), logits_f32.stride(1),
        topk_ids.stride(0), topk_ids.stride(1),
        topk_weights.stride(0), topk_weights.stride(1),
        HAS_S_MASK=has_mask,
        E=E, K=top_k, N_GROUP=n_group,
        TOPK_GROUP=topk_group, GROUP_SIZE=GROUP_SIZE,
    )
    return topk_weights, topk_ids


# ================================================================
# S_mask computation
# ================================================================

def compute_s_mask_popularity(logits, expert_bias, k_target):
    """Plan C: simple popularity top-K_target."""
    scores = torch.sigmoid(logits.float()) + expert_bias.float()
    popularity = scores.sum(dim=0)
    _, top_idx = torch.topk(popularity, k=k_target)
    s_mask = torch.zeros(logits.shape[1], dtype=torch.int32, device=logits.device)
    s_mask[top_idx] = 1
    return s_mask, int(k_target)


def compute_s_mask_batchadd(logits, expert_bias, routed_scaling_factor,
                             k_target=40, k_ext=12, quality_floor=0.70,
                             q_major=0.95, per_round_cap=8, max_rounds=999,
                             alpha=1.0, beta=0.5):
    """Plan B: expanded top-(K+M) with batch-add for quality guarantee."""
    N, E = logits.shape
    K = 8
    scores_full = torch.sigmoid(logits.float())
    rsf = routed_scaling_factor

    topkm_score, topkm_idx = torch.topk(scores_full, k=k_ext, dim=1)
    topkm_weight = topkm_score / (topkm_score.sum(dim=1, keepdim=True) + 1e-20) * rsf

    sorted_w, _ = topkm_weight.sort(dim=1, descending=True)
    r = quality_floor * sorted_w[:, :K].sum(dim=1)

    popularity = torch.zeros(E, device=logits.device, dtype=torch.float32)
    popularity.scatter_add_(0, topkm_idx.reshape(-1), topkm_weight.reshape(-1))
    _, pop_order = popularity.sort(descending=True)
    S_mask = torch.zeros(E, dtype=torch.bool, device=logits.device)
    S_mask[pop_order[:k_target]] = True

    token_ids = torch.arange(N, device=logits.device).unsqueeze(1).expand(N, k_ext)

    n_rounds = 0
    for _ in range(max_rounds):
        in_S = S_mask[topkm_idx]
        c = (topkm_weight * in_S.float()).sum(dim=1)
        d = (r - c).clamp_min(0.0)
        satisfied = d <= 0
        sat_ratio = satisfied.float().mean().item()
        if sat_ratio >= q_major:
            break

        unsat = ~satisfied
        edge_mask = unsat.unsqueeze(1).expand_as(topkm_idx)
        edge_expert = topkm_idx[edge_mask]
        edge_weight = topkm_weight[edge_mask]
        edge_token = token_ids[edge_mask]
        keep = ~S_mask[edge_expert]
        edge_expert, edge_weight, edge_token = edge_expert[keep], edge_weight[keep], edge_token[keep]
        if edge_expert.numel() == 0:
            break

        G = torch.zeros(E, device=logits.device, dtype=torch.float32)
        H = torch.zeros(E, device=logits.device, dtype=torch.float32)
        G.scatter_add_(0, edge_expert, torch.minimum(edge_weight, d[edge_token]))
        H.scatter_add_(0, edge_expert, (c[edge_token] + edge_weight >= r[edge_token]).float())
        Score = alpha * H + beta * G
        Score[S_mask] = -1e30
        _, add_order = Score.sort(descending=True)
        new_e = add_order[:per_round_cap]
        new_e = new_e[Score[new_e] > 0]
        if new_e.numel() == 0:
            break
        S_mask[new_e] = True
        n_rounds += 1

    active = S_mask.sum().item()
    s_mask_int = S_mask.int()
    return s_mask_int, active, n_rounds


# ================================================================
# Controller: manages per-layer S_mask with caching
# ================================================================

class EBController:
    def __init__(self, num_layers, mode='popularity', k_target=120,
                 refresh_interval=33, **eb_kwargs):
        self.num_layers = num_layers
        self.mode = mode
        self.k_target = k_target
        self.refresh_interval = refresh_interval
        self.eb_kwargs = eb_kwargs

        self.s_masks = {}         # layer_idx → s_mask tensor
        self.call_count = 0       # total routing calls
        self.fwd_count = 0        # model forward count
        self.stats = {'active': [], 'rounds': []}

    def new_model_forward(self):
        """Call at start of each model forward."""
        self.fwd_count += 1

    def should_refresh(self):
        return (self.fwd_count - 1) % self.refresh_interval == 0

    def get_s_mask(self, layer_idx, logits, expert_bias, rsf):
        if self.should_refresh() or layer_idx not in self.s_masks:
            if self.mode == 'popularity':
                s_mask, active = compute_s_mask_popularity(logits, expert_bias, self.k_target)
                self.s_masks[layer_idx] = s_mask
                self.stats['active'].append(active)
                self.stats['rounds'].append(0)
            elif self.mode == 'batchadd':
                s_mask, active, rounds = compute_s_mask_batchadd(
                    logits, expert_bias, rsf,
                    k_target=self.k_target, **self.eb_kwargs)
                self.s_masks[layer_idx] = s_mask
                self.stats['active'].append(active)
                self.stats['rounds'].append(rounds)
        return self.s_masks[layer_idx]


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
    from baseline_optimizations import apply_all_optimizations

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print("v0.1.15.7b — Fused Routing + Expert Budgeting (Plan B & C)")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)

        print("\nApplying baseline optimizations...")
        n_rms, n_fa = apply_all_optimizations(model)
        print(f"  Fused RMSNorm: {n_rms}, Flash-attn: {n_fa}")

        BATCH_SIZE = 32
        all_ids = []
        for i in range(BATCH_SIZE):
            text = PROMPTS[i]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                  if ids.shape[0] < mx else ids for ids in all_ids]
        input_ids = torch.stack(padded, dim=0).to(device)
        prompt_len = input_ids.shape[1]
        print(f"  Input shape: {input_ids.shape}")

        decoder_t0 = ThresholdParallelDecoder(temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm(decoder):
            return BlockDiffusionLLM(
                model, decoder, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Save original routing methods for restore
        orig_routings = {}
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                orig_routings[name] = mod.routing

        def install_fused_only():
            """C5: fused routing, no EB."""
            idx = 0
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate":
                    bias, rsf, tk, ng, tkg = mod.expert_bias, mod.routed_scaling_factor, mod.top_k, mod.n_group, mod.topk_group
                    def mk(b, r, t, n, g):
                        def fn(hs, go, topk, renorm):
                            w, i = fused_moe_routing_v2(go, b, r, s_mask=None, top_k=t, n_group=n, topk_group=g)
                            return w.to(go.dtype), i
                        return fn
                    mod.routing = mk(bias, rsf, tk, ng, tkg)
                    idx += 1

        def install_eb(ctrl):
            """C6: fused routing + EB with S_mask from controller."""
            idx = 0
            # Add pre-forward hook to model to track forward count
            def pre_hook(module, args):
                ctrl.new_model_forward()
            handle = model.register_forward_pre_hook(pre_hook)

            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate":
                    bias, rsf, tk, ng, tkg = mod.expert_bias, mod.routed_scaling_factor, mod.top_k, mod.n_group, mod.topk_group
                    li = idx
                    def mk(b, r, t, n, g, layer_i, c):
                        def fn(hs, go, topk, renorm):
                            sm = c.get_s_mask(layer_i, go, b, r)
                            w, i = fused_moe_routing_v2(go, b, r, s_mask=sm, top_k=t, n_group=n, topk_group=g)
                            return w.to(go.dtype), i
                        return fn
                    mod.routing = mk(bias, rsf, tk, ng, tkg, li, ctrl)
                    idx += 1
            return handle

        def restore_routing():
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate" and name in orig_routings:
                    mod.routing = orig_routings[name]

        def run_config(label, setup_fn, cleanup_fn=None):
            setup_fn()
            # Warmup
            dllm = make_dllm(decoder_t0)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            # Timing
            times, fwds = [], []
            for ri in range(2):
                dllm = make_dllm(decoder_t0)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)
                fwds.append(dllm.diff_iteration.num_forwards)
                print(f"  Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd, "
                      f"{dllm.diff_iteration.num_forwards/(t1-t0):.1f} fwd/s")
            if cleanup_fn:
                cleanup_fn()
            avg_t = sum(times) / len(times)
            avg_f = sum(fwds) / len(fwds)
            return {"avg_time": avg_t, "avg_fwd": avg_f, "fwd_per_s": avg_f/avg_t,
                    "ms_per_fwd": avg_t/avg_f*1000, "times": times, "fwds": fwds}

        results = OrderedDict()

        # ---- C5: Fused routing only ----
        print(f"\n{'='*60}")
        print("C5: Fused routing only (no EB)")
        print(f"{'='*60}")
        r = run_config("C5", install_fused_only, restore_routing)
        results["C5_fused_only"] = r
        c5_time = r["avg_time"]
        c5_fwd = r["avg_fwd"]
        print(f"  Avg: {r['avg_time']:.3f}s, {r['avg_fwd']:.0f} fwd, {r['fwd_per_s']:.1f} fwd/s")

        # ---- C6c: Plan C — popularity top-K, sweep K_target ----
        for kt in [120, 100, 80]:
            label = f"C6c_K{kt}"
            print(f"\n{'='*60}")
            print(f"{label}: Plan C — popularity top-{kt}, refresh ~per-block")
            print(f"{'='*60}")
            ctrl = EBController(19, mode='popularity', k_target=kt, refresh_interval=33)
            handle = None
            def setup():
                nonlocal handle
                handle = install_eb(ctrl)
            def cleanup():
                nonlocal handle
                if handle: handle.remove()
                restore_routing()
                ctrl.s_masks.clear()
                ctrl.fwd_count = 0
                ctrl.stats = {'active': [], 'rounds': []}
            r = run_config(label, setup, cleanup)
            delta = (r["avg_time"] - c5_time) / c5_time * 100
            r["delta_vs_c5_pct"] = delta
            r["k_target"] = kt
            r["avg_active"] = kt  # popularity always exactly K_target
            results[label] = r
            print(f"  Avg: {r['avg_time']:.3f}s, {r['avg_fwd']:.0f} fwd, {r['fwd_per_s']:.1f} fwd/s")
            print(f"  vs C5: {delta:+.1f}%")

        # ---- C6b: Plan B — batch-add ----
        print(f"\n{'='*60}")
        print("C6b: Plan B — batch-add (K_target=40, QF=0.70, q_major=0.95), refresh ~per-block")
        print(f"{'='*60}")
        ctrl_b = EBController(19, mode='batchadd', k_target=40, refresh_interval=33,
                              k_ext=12, quality_floor=0.70, q_major=0.95,
                              per_round_cap=8, max_rounds=999, alpha=1.0, beta=0.5)
        handle_b = None
        def setup_b():
            global handle_b
            handle_b = install_eb(ctrl_b)
        def cleanup_b():
            global handle_b
            if handle_b: handle_b.remove()
            restore_routing()
        r_b = run_config("C6b", setup_b, cleanup_b)
        delta_b = (r_b["avg_time"] - c5_time) / c5_time * 100
        avg_active_b = sum(ctrl_b.stats['active']) / max(len(ctrl_b.stats['active']), 1)
        avg_rounds_b = sum(ctrl_b.stats['rounds']) / max(len(ctrl_b.stats['rounds']), 1)
        r_b["delta_vs_c5_pct"] = delta_b
        r_b["avg_active"] = avg_active_b
        r_b["avg_rounds"] = avg_rounds_b
        results["C6b_batchadd"] = r_b
        print(f"  Avg: {r_b['avg_time']:.3f}s, {r_b['avg_fwd']:.0f} fwd, {r_b['fwd_per_s']:.1f} fwd/s")
        print(f"  vs C5: {delta_b:+.1f}%")
        print(f"  Avg |S|: {avg_active_b:.1f}, avg rounds: {avg_rounds_b:.1f}")

        # ---- Quality check on best EB config ----
        # Find best
        eb_configs = {k: v for k, v in results.items() if k.startswith("C6")}
        best_name = min(eb_configs, key=lambda k: eb_configs[k]["avg_time"])
        print(f"\n{'='*60}")
        print(f"Quality check on {best_name} (temp=0.7)")
        print(f"{'='*60}")
        # Re-install best config
        if "batchadd" in best_name:
            ctrl_q = EBController(19, mode='batchadd', k_target=40, refresh_interval=33,
                                  k_ext=12, quality_floor=0.70, q_major=0.95,
                                  per_round_cap=8, max_rounds=999, alpha=1.0, beta=0.5)
        else:
            kt_best = results[best_name].get("k_target", 120)
            ctrl_q = EBController(19, mode='popularity', k_target=kt_best, refresh_interval=33)
        h_q = install_eb(ctrl_q)

        decoder_t7 = ThresholdParallelDecoder(temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm_q = make_dllm(decoder_t7)
        with torch.inference_mode():
            _ = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        with torch.inference_mode():
            out = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        gen_tokens = out[:, prompt_len:]
        quality = {}
        for bi in sorted(VERIFIABLE.keys()):
            gt = gen_tokens[bi]
            valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
            text = tokenizer.decode(valid, skip_special_tokens=True)
            quality[bi] = text[:300]
            print(f"  #{bi}: {text[:150]}")
        results[best_name]["quality"] = quality
        h_q.remove()
        restore_routing()

        # ---- Summary ----
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        # Get C0 baseline from previous experiment
        c0_time = 8.792  # from fused_routing_results.json
        print(f"  (C0 baseline reference: {c0_time:.3f}s, 262 fwd, 29.8 fwd/s)")
        print(f"  {'Config':<30s} {'Time(s)':>8s} {'Fwd':>5s} {'Fwd/s':>6s} {'ms/fwd':>7s} {'vs C5':>7s} {'vs C0':>7s} {'|S|':>5s}")
        print(f"  {'-'*80}")
        for cname, r in results.items():
            d5 = f"{r.get('delta_vs_c5_pct', 0):+.1f}%" if 'delta_vs_c5_pct' in r else "—"
            d0 = f"{(r['avg_time'] - c0_time) / c0_time * 100:+.1f}%"
            act = f"{r.get('avg_active', 0):.0f}" if r.get('avg_active') else "—"
            print(f"  {cname:<30s} {r['avg_time']:>8.3f} {r['avg_fwd']:>5.0f} "
                  f"{r['fwd_per_s']:>6.1f} {r['ms_per_fwd']:>7.2f} {d5:>7s} {d0:>7s} {act:>5s}")

        out_path = REPO_ROOT / "codex_coding" / "results" / "fused_routing_eb_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
