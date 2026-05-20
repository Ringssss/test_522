#!/usr/bin/env python3
"""
v0.1.15.4 Phase 3 Step 1 — Expert Budgeting Kernel Profiling

Goal: Determine whether reducing unique active experts (via Expert Budgeting)
translates to fused_experts kernel wall-clock savings.

Key insight from theoretical analysis:
  - Per expert weight: w13=4MB + w2=2MB = 6MB (bf16)
  - H100 HBM bandwidth: ~3.35 TB/s
  - 82 experts (homogeneous batch): theoretical 0.154ms, measured 0.30ms → launch overhead
  - 221 experts (heterogeneous batch): theoretical 0.415ms → MAY exceed 0.30ms floor!
  - 141 experts (Expert Budgeting): theoretical 0.265ms

Previous benchmark only tested homogeneous batch (82 experts). This script tests
heterogeneous batch (~221 experts) to see if the kernel time increases, and whether
Expert Budgeting (141 experts) provides real savings.

Method:
  1. Load model, run heterogeneous 32-prompt batch → real routing data
  2. Construct routing data with varying unique expert counts
  3. Time fused_experts for each case with CUDA Events
  4. Also provide ncu-compatible markers for optional deep profiling
"""

from __future__ import annotations
import os, sys, socket, json
from pathlib import Path

import torch

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
RESULTS_DIR = REPO_ROOT / "codex_coding" / "results"
NUM_EXPERTS = 256
TOP_K = 8

# 32 heterogeneous prompts (same as batch32_boundary.py)
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
    "Solve this logic puzzle step by step: If A is true, then B is true. If B is true, then C is true. A is true. What can we conclude about B and C?",
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
    "List all 8 planets in our solar system in order from closest to farthest from the Sun.",
    "Design a real-time collaborative document editor like Google Docs with conflict resolution, offline support, and version history.",
    "Explain operating system memory management: virtual memory, page tables, TLB, demand paging, and memory-mapped files.",
    "Write a comprehensive guide to Kubernetes architecture including pods, services, ingress, operators, and cluster autoscaling.",
]


def time_fused_experts(hidden_states, w1, w2, topk_weights, topk_ids,
                       warmup=20, repeats=100, label=""):
    """Time fused_experts with CUDA Events."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    for _ in range(warmup):
        fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, inplace=False)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, inplace=False)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def restrict_to_expert_set(topk_ids, topk_weights, allowed_set, num_experts=256):
    """
    Restrict topk_ids to only experts in allowed_set.
    Experts not in allowed_set get weight=0 (keep original topk_ids shape).
    """
    allowed_mask = torch.zeros(num_experts, dtype=torch.bool, device=topk_ids.device)
    allowed_mask[allowed_set] = True
    keep = allowed_mask[topk_ids]  # [N, k] bool
    new_weights = topk_weights * keep.float()
    # Renormalize
    w_sum = new_weights.sum(dim=1, keepdim=True)
    w_sum = w_sum.clamp(min=1e-8)
    new_weights = new_weights / w_sum * topk_weights.sum(dim=1, keepdim=True)
    return topk_ids, new_weights


def physically_remove_experts(topk_ids, topk_weights, allowed_set, num_experts=256):
    """
    Physically remove non-allowed experts from topk_ids.
    Returns variable-length per token (packed into fixed shape with min_k).
    For simplicity, keep top-k format but replace non-allowed with the best allowed expert.
    """
    allowed_mask = torch.zeros(num_experts, dtype=torch.bool, device=topk_ids.device)
    allowed_mask[allowed_set] = True
    keep = allowed_mask[topk_ids]  # [N, k]

    # For tokens where some experts are removed, replace with best remaining
    N, k = topk_ids.shape
    new_ids = topk_ids.clone()
    new_weights = topk_weights.clone()

    for i in range(N):
        if not keep[i].all():
            kept_idx = topk_ids[i][keep[i]]
            kept_w = topk_weights[i][keep[i]]
            if len(kept_idx) == 0:
                continue
            # Fill removed slots with copies of best remaining (weight=0)
            removed = ~keep[i]
            n_removed = removed.sum().item()
            best = kept_idx[0]
            new_ids[i][removed] = best
            new_weights[i][removed] = 0.0

    # Renormalize
    w_sum = new_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    new_weights = new_weights / w_sum * topk_weights.sum(dim=1, keepdim=True)
    return new_ids.contiguous(), new_weights.contiguous()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoTokenizer, AutoConfig

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print("v0.1.15.4 Phase 3 Step 1 — Expert Budgeting Kernel Profiling")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                               local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                        local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        with torch.inference_mode():
            _ = model(torch.arange(64, dtype=torch.long, device=device).unsqueeze(0),
                      use_cache=False)

        # ---- Step 1: Get heterogeneous routing data ----
        print("\n[1] Collecting heterogeneous routing data (32 unique prompts)...")

        # Tokenize 32 prompts
        all_ids = []
        for text in PROMPTS:
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
        print(f"  Input shape: {list(input_ids.shape)}")

        # Hook MoE layer 10 to capture routing
        moe_layer = model.model.layers[10].mlp
        w1 = moe_layer.experts.w13_weight
        w2 = moe_layer.experts.w2_weight
        print(f"  Weight shapes: w1={list(w1.shape)}, w2={list(w2.shape)}, dtype={w1.dtype}")
        w1_per_expert_mb = w1[0].numel() * 2 / 1024 / 1024
        w2_per_expert_mb = w2[0].numel() * 2 / 1024 / 1024
        print(f"  Per expert: w1={w1_per_expert_mb:.2f}MB, w2={w2_per_expert_mb:.2f}MB, "
              f"total={w1_per_expert_mb + w2_per_expert_mb:.2f}MB")

        captured = {}
        orig_forward = moe_layer.forward

        def capture_hook(hidden_states):
            bsz, seq_len, h = hidden_states.shape
            hs_flat = hidden_states.view(-1, h)
            topk_idx, topk_weight, gate_logits = moe_layer.gate(hs_flat)
            captured["hidden_states"] = hs_flat.detach()
            captured["topk_idx"] = topk_idx.detach()
            captured["topk_weight"] = topk_weight.detach()
            captured["gate_logits"] = gate_logits.detach()
            # Return valid output
            shared_res = moe_layer.shared_experts(hidden_states)
            router_logits = moe_layer.gate.get_logits(hs_flat)
            routed_y = moe_layer.experts.forward_impl(
                hidden_states=hs_flat, router_logits=router_logits)
            routed_y = routed_y.view(bsz, seq_len, h)
            return routed_y + shared_res

        moe_layer.forward = capture_hook
        try:
            with torch.inference_mode():
                model(input_ids, use_cache=False)
        finally:
            moe_layer.forward = orig_forward

        hs = captured["hidden_states"]
        idx_full = captured["topk_idx"]
        w_full = captured["topk_weight"]
        gate_logits = captured["gate_logits"]
        N = hs.shape[0]
        print(f"  Captured: N={N} tokens, topk_ids={list(idx_full.shape)}")

        baseline_unique = idx_full.unique().numel()
        print(f"  Baseline unique active experts: {baseline_unique}/256")

        # ---- Step 2: Construct test cases ----
        print("\n[2] Constructing test cases...")

        # Use compute_active_set for realistic Expert Budgeting
        sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))
        from expert_budgeting_e2e import compute_active_set

        # Expert Budgeting: K=40, routing_p=0.6
        S_mask = compute_active_set(gate_logits, idx_full, w_full,
                                     K_target=40, quality_floor=0.85, top_p=0.6)
        budgeted_experts = S_mask.nonzero(as_tuple=True)[0]
        budgeted_count = budgeted_experts.numel()
        print(f"  Expert Budgeting (K=40, routing_p=0.6): {budgeted_count} active experts")

        # Build test cases: vary unique expert count
        gate_w = torch.softmax(gate_logits.float(), dim=-1)
        popularity = gate_w.sum(dim=0)
        _, pop_order = popularity.sort(descending=True)

        test_cases = []

        # Case 0: Baseline (full routing, ~221 unique experts)
        test_cases.append(("baseline", idx_full, w_full, baseline_unique))

        # Cases with synthetic expert restriction (top-N by popularity)
        for n_exp in [200, 180, 160, 141, 120, 100, 80, 60]:
            allowed = pop_order[:n_exp]
            new_ids, new_w = physically_remove_experts(idx_full, w_full, allowed)
            actual_unique = new_ids[new_w > 0].unique().numel()
            # Also count including zero-weight (since kernel still processes them)
            all_unique = new_ids.unique().numel()
            test_cases.append((f"restrict_{n_exp}", new_ids, new_w, all_unique))

        # Case: Expert Budgeting realistic (re-route within S)
        eb_ids, eb_w = physically_remove_experts(idx_full, w_full, budgeted_experts)
        eb_unique = eb_ids.unique().numel()
        test_cases.append((f"EB_K40_rp60", eb_ids, eb_w, eb_unique))

        # ---- Step 3: Kernel timing ----
        print("\n[3] Kernel timing (CUDA Events, 20 warmup + 100 repeats)...")
        print(f"  N={N} tokens, using layer 10 weights\n")

        bw_bytes_ms = 3.35e12 / 1000  # H100 HBM bandwidth in bytes/ms
        per_expert_bytes = (w1[0].numel() + w2[0].numel()) * 2  # bf16

        results = {}
        header = f"  {'Case':<20s} {'Unique':>6s} {'Time(ms)':>9s} {'Theor(ms)':>10s} {'Ratio':>6s} {'vs_BL':>7s}"
        print(header)
        print(f"  {'-' * 60}")

        baseline_time = None
        for name, ids, weights, unique_count in test_cases:
            with torch.inference_mode():
                t_ms = time_fused_experts(hs, w1, w2, weights, ids,
                                          warmup=20, repeats=100, label=name)
            theoretical_ms = unique_count * per_expert_bytes / bw_bytes_ms
            if baseline_time is None:
                baseline_time = t_ms
            ratio_vs_bl = t_ms / baseline_time

            print(f"  {name:<20s} {unique_count:>6d} {t_ms:>9.4f} {theoretical_ms:>10.4f} "
                  f"{t_ms/theoretical_ms:>6.2f}x {ratio_vs_bl:>6.3f}")

            results[name] = {
                "unique_experts": unique_count,
                "time_ms": t_ms,
                "theoretical_hbm_ms": theoretical_ms,
                "overhead_ratio": t_ms / theoretical_ms,
                "ratio_vs_baseline": ratio_vs_bl,
            }

        # ---- Step 4: Analysis ----
        print(f"\n{'='*80}")
        print("ANALYSIS")
        print(f"{'='*80}")

        bl = results["baseline"]
        eb = results.get("EB_K40_rp60", results.get("restrict_141"))

        print(f"\n  Baseline: {bl['unique_experts']} experts, {bl['time_ms']:.4f}ms")
        print(f"  Theoretical HBM time: {bl['theoretical_hbm_ms']:.4f}ms")
        print(f"  Overhead ratio (measured/theoretical): {bl['overhead_ratio']:.2f}x")

        if eb:
            print(f"\n  Expert Budgeting: {eb['unique_experts']} experts, {eb['time_ms']:.4f}ms")
            print(f"  Theoretical HBM time: {eb['theoretical_hbm_ms']:.4f}ms")
            saving_pct = (1 - eb['ratio_vs_baseline']) * 100
            print(f"  Speedup vs baseline: {saving_pct:.1f}%")

        # Determine if kernel time scales with expert count
        restrict_cases = [(name, r) for name, r in results.items()
                          if name.startswith("restrict_")]
        if restrict_cases:
            times = [(r["unique_experts"], r["time_ms"]) for _, r in restrict_cases]
            times.sort()
            min_exp, min_t = times[0]
            max_exp, max_t = times[-1]
            if max_t > min_t * 1.05:
                print(f"\n  FINDING: Kernel time DOES scale with expert count")
                print(f"    {min_exp} experts: {min_t:.4f}ms")
                print(f"    {max_exp} experts: {max_t:.4f}ms")
                print(f"    Scaling ratio: {max_t/min_t:.3f}x for {max_exp/min_exp:.1f}x experts")
                print(f"    → Expert Budgeting CAN provide kernel-level speedup")
            else:
                print(f"\n  FINDING: Kernel time is CONSTANT regardless of expert count")
                print(f"    Range: {min_t:.4f}ms — {max_t:.4f}ms (variation < 5%)")
                print(f"    → Expert Budgeting CANNOT provide kernel-level speedup")
                print(f"    → The {bl['time_ms']:.2f}ms floor is launch/scheduling overhead")

        # Save
        out_path = RESULTS_DIR / "expert_budgeting_kernel_profiling.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
