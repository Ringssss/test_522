#!/usr/bin/env python3
"""
v0.1.15.5 — Expert Budgeting Hook Version: Profiling & Quality Verification

Profiles the PROVEN hook-based Expert Budgeting on the optimized baseline.

Four rounds:
  1. Baseline end-to-end timing (no hooks)
  2. Hook EB end-to-end timing (no per-step sync) — real overhead
  3. Hook EB per-step profiling (with sync, 5 forwards only) — bottleneck analysis
  4. Quality verification (temp=0.7, baseline vs hook EB)

Uses compute_active_set from expert_budgeting_e2e.py (proven algorithm).
Baseline optimizations from baseline_optimizations.py (shared module).
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"

# Constants from expert_budgeting_e2e.py
TOP_K_ORIG = 8
SHARED_RATE = 0.419
ROUTING_RATE = 0.581
NUM_EXPERTS = 256

# Expert Budgeting config (proven: K40 + QF0.85 + tp0.75)
K_TARGET = 40
QUALITY_FLOOR = 0.85
TOP_P = 0.75

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


# ==============================================================
# compute_active_set (from expert_budgeting_e2e.py, unchanged)
# ==============================================================
def compute_active_set(gate_logits, topk_idx, topk_w, K_target, quality_floor,
                       top_p=0.75):
    N = gate_logits.shape[0]
    device = gate_logits.device
    gate_w = torch.softmax(gate_logits.float(), dim=-1)
    popularity = gate_w.sum(dim=0)
    _, pop_order = popularity.sort(descending=True)
    S_mask = torch.zeros(NUM_EXPERTS, dtype=torch.bool, device=device)
    S_mask[pop_order[:K_target]] = True

    sorted_rw, sort_order = topk_w.sort(dim=1, descending=True)
    total_routing = topk_w.sum(dim=1, keepdim=True)
    needed_frac = (top_p - SHARED_RATE) / ROUTING_RATE
    threshold = needed_frac * total_routing
    cumsum = sorted_rw.cumsum(dim=1)
    enough = cumsum >= threshold
    enough[:, -1] = True
    k_budgets = (enough.float().argmax(dim=1) + 1).int()

    sorted_idx = topk_idx.gather(1, sort_order)
    original_gate_vals = gate_w.gather(1, sorted_idx)
    positions = torch.arange(TOP_K_ORIG, device=device).unsqueeze(0)
    topp_mask = positions < k_budgets.unsqueeze(1)
    original_weight = (original_gate_vals * topp_mask.float()).sum(dim=1)

    n_iters = 0
    for _ in range(30):
        s_indices = S_mask.nonzero(as_tuple=True)[0]
        s_gate_w = gate_w[:, s_indices]
        s_sorted_w, _ = s_gate_w.sort(dim=1, descending=True)
        s_positions = torch.arange(s_sorted_w.shape[1], device=device).unsqueeze(0)
        s_topp_mask = s_positions < k_budgets.unsqueeze(1)
        covered = (s_sorted_w * s_topp_mask.float()).sum(dim=1)
        safe = original_weight > 1e-8
        cov_ratio = torch.where(safe, covered / original_weight.clamp(min=1e-8),
                                torch.ones(N, device=device))
        violated = (cov_ratio < quality_floor) & safe
        if not violated.any():
            break
        violated_idx = violated.nonzero(as_tuple=True)[0]
        v_gate = gate_w[violated_idx]
        v_gate_masked = v_gate.clone()
        v_gate_masked[:, S_mask] = -1
        best_missing = v_gate_masked.argmax(dim=1)
        for e in best_missing.unique():
            S_mask[e.item()] = True
        n_iters += 1

    return S_mask, n_iters


# ==============================================================
# Hook-based Expert Budgeting (proven algorithm)
# ==============================================================
class ProfilingEBController:
    """Expert Budgeting hook with optional per-step timing."""

    def __init__(self, K_target=40, quality_floor=0.85, top_p=0.75,
                 do_profile=False):
        self.K_target = K_target
        self.quality_floor = quality_floor
        self.top_p = top_p
        self.do_profile = do_profile
        self.stats = {"active_experts": [], "safety_iters": []}
        self.timings = defaultdict(list)  # step_name -> list of durations (us)

    def hook_forward(self, moe_mod, layer_idx, hidden_states):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)

        def tick():
            if self.do_profile:
                torch.cuda.synchronize()
                return time.perf_counter()
            return 0

        t0 = tick()

        # Step A: shared_experts
        res = moe_mod.shared_experts(hidden_states)
        t1 = tick()

        # Step B: gate.get_logits (baseline also has this)
        gate_logits = moe_mod.gate.get_logits(hs_flat)
        t2 = tick()

        # Step C: gate() full call — extra, for topk_idx/topk_weight
        topk_idx, topk_weight, _ = moe_mod.gate(hs_flat)
        t3 = tick()

        # Step D: compute_active_set (core algorithm)
        S_mask, n_iters = compute_active_set(
            gate_logits, topk_idx, topk_weight,
            self.K_target, self.quality_floor, self.top_p)
        t4 = tick()

        self.stats["active_experts"].append(S_mask.sum().item())
        self.stats["safety_iters"].append(n_iters)

        # Step E: mask logits
        masked_logits = gate_logits.clone()
        masked_logits[:, ~S_mask] = float('-inf')
        t5 = tick()

        # Step F: gate.routing with masked logits
        topk_weight_new, topk_idx_new = moe_mod.gate.routing(
            hs_flat, masked_logits, moe_mod.gate.top_k, True)
        t6 = tick()

        # Step G: top-p pruning
        sorted_w, sort_order = topk_weight_new.sort(dim=1, descending=True)
        total_routing = topk_weight_new.sum(dim=1, keepdim=True)
        needed_frac = (self.top_p - SHARED_RATE) / ROUTING_RATE
        threshold = needed_frac * total_routing
        cumsum = sorted_w.cumsum(dim=1)
        enough = cumsum >= threshold
        enough[:, -1] = True
        cutoff = enough.float().argmax(dim=1) + 1
        rank_pos = torch.arange(TOP_K_ORIG, device=hs_flat.device).unsqueeze(0)
        keep_sorted = rank_pos < cutoff.unsqueeze(1)
        pruning_mask = torch.zeros_like(topk_weight_new, dtype=torch.bool)
        pruning_mask.scatter_(1, sort_order, keep_sorted)
        kept_sum = (topk_weight_new * pruning_mask.float()).sum(dim=1, keepdim=True)
        orig_sum = topk_weight_new.sum(dim=1, keepdim=True)
        scale = orig_sum / (kept_sum + 1e-8)
        new_weights = topk_weight_new * pruning_mask.float() * scale
        t7 = tick()

        # Step H: fused_experts
        routed_y = fused_experts(
            hidden_states=hs_flat,
            w1=moe_mod.experts.w13_weight,
            w2=moe_mod.experts.w2_weight,
            topk_weights=new_weights,
            topk_ids=topk_idx_new,
            inplace=False)
        t8 = tick()

        # Combine
        routed_y = routed_y.view(bsz, seq_len, h)
        out = routed_y + res if moe_mod.config.num_shared_experts is not None else routed_y

        if self.do_profile:
            self.timings["A_shared_experts"].append((t1 - t0) * 1e6)
            self.timings["B_get_logits"].append((t2 - t1) * 1e6)
            self.timings["C_gate_full"].append((t3 - t2) * 1e6)
            self.timings["D_compute_active_set"].append((t4 - t3) * 1e6)
            self.timings["E_mask_logits"].append((t5 - t4) * 1e6)
            self.timings["F_gate_routing"].append((t6 - t5) * 1e6)
            self.timings["G_topp_pruning"].append((t7 - t6) * 1e6)
            self.timings["H_fused_experts"].append((t8 - t7) * 1e6)
            self.timings["total"].append((t8 - t0) * 1e6)

        return out


def install_eb_hooks(model, ctrl):
    hooks = []
    mi = 0
    for layer in model.model.layers:
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue
        moe = layer.mlp
        orig = moe.forward
        idx = mi
        def make(m, i, c):
            def f(hs): return c.hook_forward(m, i, hs)
            return f
        moe.forward = make(moe, idx, ctrl)
        hooks.append((moe, orig))
        mi += 1
    return hooks


def remove_eb_hooks(hooks):
    for moe, orig in hooks:
        moe.forward = orig


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
    print("Expert Budgeting Hook Profiling on Optimized Baseline")
    print(f"  EB config: K={K_TARGET}, QF={QUALITY_FLOOR}, tp={TOP_P}")
    print("  Baseline: max_unroll=4 + Fused RMSNorm + Classic flash-attn 2.8.3")
    print("  HetEval-32: batch=32, gen=256, block=32, threshold=0.90")
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

        # Apply baseline optimizations
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
                ids = torch.cat([torch.full((mx - ids.shape[0],), pad_id,
                                            dtype=ids.dtype), ids])
            padded.append(ids)
        input_ids = torch.stack(padded, dim=0).to(device)
        prompt_len = input_ids.shape[1]
        print(f"  Input shape: {input_ids.shape}")

        GEN_LENGTH = 256
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

        # Warmup
        print("\nWarmup...")
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print("  Done.")

        # ==============================================================
        # Round 1: Baseline end-to-end (no hooks)
        # ==============================================================
        print("\n" + "=" * 60)
        print("Round 1: Baseline end-to-end (no hooks)")
        print("=" * 60)
        dllm = make_dllm(decoder_t0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd_bl = dllm.diff_iteration.num_forwards
        time_bl = t1 - t0
        print(f"  Fwd={fwd_bl}, time={time_bl:.3f}s, fwd/s={fwd_bl/time_bl:.1f}")

        # ==============================================================
        # Round 2: Hook EB end-to-end (no per-step sync)
        # ==============================================================
        print("\n" + "=" * 60)
        print("Round 2: Hook EB end-to-end (no per-step sync)")
        print("=" * 60)
        ctrl_r2 = ProfilingEBController(
            K_TARGET, QUALITY_FLOOR, TOP_P, do_profile=False)
        hooks = install_eb_hooks(model, ctrl_r2)

        # Warmup with hooks
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        # Timed run
        ctrl_r2.stats = {"active_experts": [], "safety_iters": []}
        dllm = make_dllm(decoder_t0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd_eb = dllm.diff_iteration.num_forwards
        time_eb = t1 - t0
        overhead_pct = (time_eb - time_bl) / time_bl * 100

        avg_active = (sum(ctrl_r2.stats["active_experts"]) /
                      max(len(ctrl_r2.stats["active_experts"]), 1))
        avg_iters = (sum(ctrl_r2.stats["safety_iters"]) /
                     max(len(ctrl_r2.stats["safety_iters"]), 1))

        print(f"  Fwd={fwd_eb}, time={time_eb:.3f}s, fwd/s={fwd_eb/time_eb:.1f}")
        print(f"  vs baseline: {overhead_pct:+.1f}%")
        print(f"  Avg active experts: {avg_active:.1f}")
        print(f"  Avg safety iterations: {avg_iters:.1f}")
        print(f"  Delta fwd: {fwd_eb - fwd_bl:+d}")

        remove_eb_hooks(hooks)

        # ==============================================================
        # Round 3: Per-step profiling (with sync, limited forwards)
        # ==============================================================
        print("\n" + "=" * 60)
        print("Round 3: Per-step profiling (sync, 5 forwards)")
        print("=" * 60)
        ctrl_r3 = ProfilingEBController(
            K_TARGET, QUALITY_FLOOR, TOP_P, do_profile=True)
        hooks = install_eb_hooks(model, ctrl_r3)

        # Run exactly 5 forwards by using small gen_length
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=32,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        n_profiled = dllm.diff_iteration.num_forwards

        print(f"  Profiled {n_profiled} forwards × 19 layers")
        print(f"  Total samples per step: {len(ctrl_r3.timings.get('total', []))}")
        print()
        print(f"  {'Step':<25s} {'Avg (us)':>10s} {'Med (us)':>10s} "
              f"{'Total (ms)':>10s} {'%':>6s}")
        print(f"  {'-'*65}")

        total_all = sum(ctrl_r3.timings.get("total", [0]))
        for step in ["A_shared_experts", "B_get_logits", "C_gate_full",
                      "D_compute_active_set", "E_mask_logits",
                      "F_gate_routing", "G_topp_pruning", "H_fused_experts"]:
            vals = ctrl_r3.timings.get(step, [])
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            med = sorted(vals)[len(vals) // 2]
            tot = sum(vals) / 1000  # us -> ms
            pct = sum(vals) / total_all * 100 if total_all > 0 else 0
            marker = " ★" if step.startswith(("C_", "D_", "F_", "G_")) else ""
            print(f"  {step:<25s} {avg:>10.0f} {med:>10.0f} "
                  f"{tot:>10.1f} {pct:>5.1f}%{marker}")

        print(f"  {'-'*65}")
        tot_vals = ctrl_r3.timings.get("total", [])
        if tot_vals:
            avg_t = sum(tot_vals) / len(tot_vals)
            print(f"  {'TOTAL per layer':<25s} {avg_t:>10.0f}")
            print(f"  {'TOTAL per forward (×19)':<25s} {avg_t * 19 / 1000:>10.1f} ms")

        print(f"\n  ★ = steps NOT in baseline (pure overhead)")

        # Baseline comparison: how long does a normal MoE forward take?
        print(f"\n  For reference:")
        print(f"    Baseline MoE per forward: ~{time_bl/fwd_bl*1000*0.60:.1f} ms "
              f"(60% of {time_bl/fwd_bl*1000:.1f} ms)")

        remove_eb_hooks(hooks)

        # ==============================================================
        # Round 4: Quality verification (temp=0.7)
        # ==============================================================
        print("\n" + "=" * 60)
        print("Round 4: Quality verification (temp=0.7)")
        print("=" * 60)

        decoder_t7 = ThresholdParallelDecoder(
            temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        for cname, use_eb in [("Baseline", False), ("EB_K40_QF85_tp75", True)]:
            if use_eb:
                ctrl_q = ProfilingEBController(
                    K_TARGET, QUALITY_FLOOR, TOP_P, do_profile=False)
                hooks = install_eb_hooks(model, ctrl_q)

            dllm = make_dllm(decoder_t7)
            # Warmup
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            # Generate
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                out = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                    block_length=BLOCK_LENGTH)
            gen_tokens = out[:, prompt_len:]
            fwd_q = dllm.diff_iteration.num_forwards

            print(f"\n  [{cname}] Fwd={fwd_q}")
            for bi in sorted(VERIFIABLE.keys()):
                gt = gen_tokens[bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                print(f"    #{bi}: {text[:200]}")

            if use_eb:
                remove_eb_hooks(hooks)

        # ==============================================================
        # Summary
        # ==============================================================
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"  Baseline:   Fwd={fwd_bl}, time={time_bl:.3f}s")
        print(f"  Hook EB:    Fwd={fwd_eb}, time={time_eb:.3f}s "
              f"({overhead_pct:+.1f}%), ΔFwd={fwd_eb-fwd_bl:+d}")
        print(f"  Avg active: {avg_active:.1f}, Avg safety iters: {avg_iters:.1f}")

        if tot_vals:
            avg_t = sum(tot_vals) / len(tot_vals)
            # Overhead breakdown
            overhead_per_fwd = time_eb / fwd_eb - time_bl / fwd_bl
            print(f"\n  Overhead per forward: {overhead_per_fwd*1000:.1f} ms")
            print(f"  (profiled total per forward: {avg_t*19/1000:.1f} ms with sync)")

        # Save
        save_data = {
            "baseline": {"fwd": fwd_bl, "time_s": time_bl},
            "hook_eb": {"fwd": fwd_eb, "time_s": time_eb,
                        "overhead_pct": overhead_pct,
                        "avg_active": avg_active,
                        "avg_safety_iters": avg_iters},
            "per_step_profiling": {k: {"avg_us": sum(v)/len(v),
                                       "total_ms": sum(v)/1000}
                                   for k, v in ctrl_r3.timings.items() if v},
            "config": {"K_target": K_TARGET, "quality_floor": QUALITY_FLOOR,
                       "top_p": TOP_P},
        }
        out_path = (REPO_ROOT / "codex_coding" / "results" /
                    "expert_budgeting_hook_profiling.json")
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
