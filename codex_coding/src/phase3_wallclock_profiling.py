#!/usr/bin/env python3
"""
v0.1.15.4 Phase 3 Step 2a + 2a+ — End-to-End Wall-Clock + Full Profiling

Step 2a: Wall-clock timing of full generation
  - baseline vs K40+rp60+D1 (hook version)
  - Same 32 heterogeneous prompts, gen_length=256, block_length=32, threshold=0.90
  - CUDA Event timing for total generation

Step 2a+: torch.profiler component breakdown
  - Profile 1 full generation (baseline), get per-operator time
  - Identify MoE kernel share of total forward time

Config consistency:
  - Identical to expert_budgeting_batch32_boundary.py
  - routing_p=0.6 (equivalent top_p = 0.6 * ROUTING_RATE + SHARED_RATE = 0.768)
"""

from __future__ import annotations
import os, sys, socket, json, time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
TOP_K_ORIG = 8
SHARED_RATE = 0.419
ROUTING_RATE = 0.581
NUM_EXPERTS = 256

# routing_p=0.6 → equivalent top_p for existing code
ROUTING_P = 0.6
EQUIV_TOP_P = ROUTING_P * ROUTING_RATE + SHARED_RATE  # = 0.768

# Same 32 prompts as batch32_boundary.py
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

sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))
from expert_budgeting_e2e import compute_active_set
from expert_budgeting_boundary import FullStackController, install_hooks, remove_hooks, gen_with_ctrl


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
    print("Phase 3 Step 2a + 2a+")
    print(f"  routing_p={ROUTING_P}, equiv_top_p={EQUIV_TOP_P:.4f}")
    print(f"  K_target=40, QF=0.85, D1: margin=0.90, L4-14, drift=0.02")
    print(f"  batch=32, gen_length=256, block_length=32, threshold=0.90")
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

        # ---- Build input ----
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
        print(f"Input: {input_ids.shape}, prompt_len={prompt_len}")

        GEN_LENGTH = 256
        REUSE_LAYERS = set(range(4, 15))

        decoder = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm = BlockDiffusionLLM(
            model, decoder,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=1, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # ---- Warmup ----
        print("\nWarmup...")
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print("Warmup done.\n")

        # ============================================================
        # Step 2a: End-to-End Wall-Clock Timing
        # ============================================================
        print("=" * 80)
        print("STEP 2a: End-to-End Wall-Clock Timing")
        print("=" * 80)

        results = {}

        # --- Baseline ---
        print("\n  [baseline] Running...")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            out_bl = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        bl_fwd = dllm.diff_iteration.num_forwards
        bl_time = t1 - t0
        results["baseline"] = {"fwd": bl_fwd, "wall_clock_s": bl_time}
        print(f"  [baseline] Fwd={bl_fwd}, time={bl_time:.3f}s, fwd/s={bl_fwd/bl_time:.1f}")

        # --- K40 + routing_p=0.6 + D1 (hook version) ---
        print("\n  [K40_rp60_D1] Running...")
        ctrl = FullStackController(
            K_target=40, quality_floor=0.85, top_p=EQUIV_TOP_P,
            margin_threshold=0.90, reuse_layers=REUSE_LAYERS)
        hooks = install_hooks(model, ctrl)
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_opt = gen_with_ctrl(dllm, input_ids, ctrl, gl=GEN_LENGTH)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            opt_fwd = dllm.diff_iteration.num_forwards
            opt_time = t1 - t0

            avg_active = (sum(ctrl.total_active_experts) /
                          max(len(ctrl.total_active_experts), 1)) if ctrl.total_active_experts else 0
            avg_ept = (sum(ctrl.total_experts_per_token) /
                       max(len(ctrl.total_experts_per_token), 1)) if ctrl.total_experts_per_token else 8.0
            reuse_pct = ctrl.reused_tokens / max(ctrl.total_tokens, 1) * 100
        finally:
            remove_hooks(hooks)

        results["K40_rp60_D1"] = {
            "fwd": opt_fwd, "wall_clock_s": opt_time,
            "avg_active": avg_active, "avg_ept": avg_ept, "reuse_pct": reuse_pct}
        print(f"  [K40_rp60_D1] Fwd={opt_fwd}, time={opt_time:.3f}s, fwd/s={opt_fwd/opt_time:.1f}")
        print(f"    Active={avg_active:.1f}, E/tok={avg_ept:.1f}, Reuse={reuse_pct:.1f}%")

        # --- K40 + routing_p=0.6 only (no D1, hook version) ---
        print("\n  [K40_rp60] Running...")
        ctrl2 = FullStackController(
            K_target=40, quality_floor=0.85, top_p=EQUIV_TOP_P,
            margin_threshold=None, reuse_layers=set())
        hooks = install_hooks(model, ctrl2)
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_opt2 = gen_with_ctrl(dllm, input_ids, ctrl2, gl=GEN_LENGTH)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            opt2_fwd = dllm.diff_iteration.num_forwards
            opt2_time = t1 - t0

            avg_active2 = (sum(ctrl2.total_active_experts) /
                           max(len(ctrl2.total_active_experts), 1)) if ctrl2.total_active_experts else 0
        finally:
            remove_hooks(hooks)

        results["K40_rp60"] = {"fwd": opt2_fwd, "wall_clock_s": opt2_time, "avg_active": avg_active2}
        print(f"  [K40_rp60] Fwd={opt2_fwd}, time={opt2_time:.3f}s, fwd/s={opt2_fwd/opt2_time:.1f}")

        # --- Summary ---
        print(f"\n  {'Config':<20s} {'Fwd':>4s} {'Time(s)':>8s} {'Fwd/s':>6s} {'vs BL':>7s}")
        print(f"  {'-'*50}")
        for name, r in results.items():
            f = r["fwd"]
            t = r["wall_clock_s"]
            fs = f / t
            diff = (t / bl_time - 1) * 100
            tag = f"{diff:+.1f}%" if name != "baseline" else "—"
            print(f"  {name:<20s} {f:>4d} {t:>8.3f} {fs:>6.1f} {tag:>7s}")

        # ============================================================
        # Step 2a+: torch.profiler Component Breakdown
        # ============================================================
        print(f"\n{'='*80}")
        print("STEP 2a+: torch.profiler Component Breakdown (baseline)")
        print("=" * 80)

        print("\n  Profiling 1 full generation (baseline)...")
        with torch.inference_mode():
            # Warmup for profiler
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_flops=True,
            ) as prof:
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
                torch.cuda.synchronize()

        # Full CUDA kernel breakdown (top 30)
        print("\n  Top 30 CUDA kernels by total CUDA time:")
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=30))

        # Group by category
        print("\n  Aggregated by kernel pattern:")
        events = prof.key_averages()
        categories = {}
        for e in events:
            if e.device_time_total == 0:
                continue
            name = e.key
            cuda_us = e.device_time_total
            count = e.count
            if "fused_moe" in name.lower():
                cat = "fused_moe_kernel"
            elif "gemm" in name.lower() or "cutlass" in name.lower():
                cat = "GEMM (attention+other)"
            elif "flash" in name.lower() or "fmha" in name.lower():
                cat = "Flash Attention"
            elif "elementwise" in name.lower() or "pointwise" in name.lower():
                cat = "Elementwise"
            elif "layer_norm" in name.lower() or "layernorm" in name.lower() or "rms_norm" in name.lower():
                cat = "LayerNorm/RMSNorm"
            elif "copy" in name.lower() or "memcpy" in name.lower() or "memset" in name.lower():
                cat = "Memory ops"
            elif "reduce" in name.lower() or "scatter" in name.lower() or "gather" in name.lower():
                cat = "Reduce/Scatter/Gather"
            elif "silu" in name.lower() or "gelu" in name.lower() or "activation" in name.lower():
                cat = "Activations"
            elif "embedding" in name.lower():
                cat = "Embedding"
            else:
                cat = "Other"

            if cat not in categories:
                categories[cat] = {"cuda_us": 0, "count": 0}
            categories[cat]["cuda_us"] += cuda_us
            categories[cat]["count"] += count

        total_cuda = sum(v["cuda_us"] for v in categories.values())
        print(f"\n  {'Category':<30s} {'CUDA Time':>12s} {'%':>6s} {'Calls':>8s}")
        print(f"  {'-'*58}")
        for cat, v in sorted(categories.items(), key=lambda x: -x[1]["cuda_us"]):
            pct = v["cuda_us"] / total_cuda * 100 if total_cuda > 0 else 0
            print(f"  {cat:<30s} {v['cuda_us']/1000:>10.1f}ms {pct:>5.1f}% {v['count']:>8d}")
        print(f"  {'TOTAL':<30s} {total_cuda/1000:>10.1f}ms {'100%':>6s}")

        # Save results
        save_data = {
            "wall_clock": results,
            "profiler_categories": {k: {"cuda_ms": v["cuda_us"]/1000, "count": v["count"],
                                         "pct": v["cuda_us"]/total_cuda*100 if total_cuda > 0 else 0}
                                    for k, v in categories.items()},
            "config": {
                "routing_p": ROUTING_P, "equiv_top_p": EQUIV_TOP_P,
                "K_target": 40, "quality_floor": 0.85,
                "D1_margin": 0.90, "D1_layers": "L4-L14", "D1_drift": 0.02,
                "batch_size": BATCH_SIZE, "gen_length": GEN_LENGTH,
                "block_length": BLOCK_LENGTH, "threshold": 0.90,
            }
        }
        out_path = REPO_ROOT / "codex_coding" / "results" / "phase3_wallclock_profiling.json"
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n  Results saved to {out_path}")
        print("Done.")


if __name__ == "__main__":
    main()
