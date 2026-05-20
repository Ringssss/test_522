#!/usr/bin/env python3
"""
v0.1.15.6b — MoE-Only Compile Experiment

Key insight: MoE block's forward only takes hidden_states tensor,
does NOT touch KV cache. Compiling only MoE avoids DynamicCache recompile.

Configs:
  C0: Baseline (no compile) — re-confirm 8.75s
  C4: torch.compile on each MoE block's forward only (19 blocks)
  C4b: C4 with mode='reduce-overhead' (more aggressive optimization)

HetEval-32: batch=32, gen=256, block=32, threshold=0.90, max_unroll=4
Baseline optimizations: fused RMSNorm + classic flash-attn 2.8.3
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn.functional as F

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


def apply_moe_only_compile(model, mode='default'):
    """Compile only MoE block forwards. Does NOT touch attention/cache."""
    compiled_count = 0
    # Disable Inductor internal CUDA graphs to avoid conflicts
    try:
        import torch._inductor.config as _ind_cfg
        if hasattr(_ind_cfg, 'triton') and hasattr(_ind_cfg.triton, 'cudagraphs'):
            _ind_cfg.triton.cudagraphs = False
        if hasattr(_ind_cfg, 'cudagraph_trees'):
            _ind_cfg.cudagraph_trees = False
    except Exception:
        pass

    for name, module in model.named_modules():
        if module.__class__.__name__ == "LLaDA2MoeSparseMoeBlock":
            try:
                module.forward = torch.compile(
                    module.forward,
                    mode=mode,
                    fullgraph=False,
                    dynamic=True,
                )
                compiled_count += 1
            except Exception as e:
                print(f"  [WARN] Failed to compile {name}: {e}")
    return compiled_count


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
    print("v0.1.15.6b — MoE-Only Compile Experiment")
    print("  HetEval-32: batch=32, gen=256, block=32, threshold=0.90, max_unroll=4")
    print("  Baseline: fused RMSNorm + classic flash-attn 2.8.3")
    print("  Key: compile ONLY MoE blocks (19x), leave attention/cache in eager")
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

        results = OrderedDict()

        # ============================================================
        # C0: Baseline
        # ============================================================
        print(f"\n{'='*70}")
        print("C0: Baseline (no compile)")
        print(f"{'='*70}")

        # Warmup
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
        print(f"  C0 avg: {c0_avg:.3f}s, {c0_fwd:.0f} fwd, {c0_fwd/c0_avg:.1f} fwd/s")
        results["C0_baseline"] = {
            "times": c0_times, "fwds": c0_fwds,
            "avg_time": c0_avg, "avg_fwd": c0_fwd,
            "fwd_per_s": c0_fwd / c0_avg,
            "ms_per_fwd": c0_avg / c0_fwd * 1000,
        }

        # ============================================================
        # C4: MoE-only compile (mode=default)
        # ============================================================
        print(f"\n{'='*70}")
        print("C4: MoE-only compile (mode='default', dynamic=True)")
        print(f"{'='*70}")

        torch._dynamo.reset()
        n_compiled = apply_moe_only_compile(model, mode='default')
        print(f"  Compiled {n_compiled} MoE blocks")

        # Compile warmup
        print("  Compile warmup...")
        t_comp_start = time.perf_counter()
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t_comp_end = time.perf_counter()
        print(f"  Compile warmup done: {t_comp_end - t_comp_start:.1f}s")

        c4_times, c4_fwds = [], []
        for run_i in range(2):
            dllm = make_dllm(decoder_t0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            c4_times.append(t1 - t0)
            c4_fwds.append(dllm.diff_iteration.num_forwards)
            print(f"  Run {run_i+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd, "
                  f"{dllm.diff_iteration.num_forwards/(t1-t0):.1f} fwd/s")

        c4_avg = sum(c4_times) / len(c4_times)
        c4_fwd = sum(c4_fwds) / len(c4_fwds)
        delta = (c4_avg - c0_avg) / c0_avg * 100
        print(f"  C4 avg: {c4_avg:.3f}s, {c4_fwd:.0f} fwd, {c4_fwd/c4_avg:.1f} fwd/s")
        print(f"  vs C0: {delta:+.1f}%")
        print(f"  Fwd count match: {'YES' if c4_fwd == c0_fwd else 'NO (' + str(c4_fwd) + ' vs ' + str(c0_fwd) + ')'}")
        results["C4_moe_compile_default"] = {
            "times": c4_times, "fwds": c4_fwds,
            "avg_time": c4_avg, "avg_fwd": c4_fwd,
            "fwd_per_s": c4_fwd / c4_avg,
            "ms_per_fwd": c4_avg / c4_fwd * 1000,
            "delta_vs_c0_pct": delta,
            "compile_warmup_s": t_comp_end - t_comp_start,
            "fwd_count_match": c4_fwd == c0_fwd,
        }

        # Quality check
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
        results["C4_moe_compile_default"]["quality"] = quality

        # ============================================================
        # C4b: MoE-only compile (mode=reduce-overhead)
        # ============================================================
        print(f"\n{'='*70}")
        print("C4b: MoE-only compile (mode='reduce-overhead', dynamic=True)")
        print(f"{'='*70}")

        # Need to reset and reapply since we can't re-compile already compiled modules easily
        # Reload is expensive; instead, test if reduce-overhead works by resetting dynamo
        # and re-wrapping. Since the compiled forward is already set, we'll skip reload
        # and instead just note that mode='default' was tested above.
        # For a clean test, we re-apply compile with reduce-overhead mode.
        torch._dynamo.reset()

        # Re-apply baseline optimizations on the same model (idempotent)
        # The MoE forwards were replaced by compiled versions; we need fresh ones.
        # Simplest: reload model. But that's slow. Instead, we test reduce-overhead
        # only if default mode shows promise.

        # Check if C4 showed improvement
        if delta < -1.0:
            # C4 was faster, try reduce-overhead
            print("  C4 showed improvement, testing reduce-overhead...")
            print("  (Skipping reload - reduce-overhead test requires fresh model)")
            print("  → Recommend testing reduce-overhead in a separate run if C4 is promising")
            results["C4b_moe_compile_reduce_overhead"] = {
                "status": "SKIPPED",
                "reason": "requires model reload; C4 default mode tested"
            }
        else:
            print("  C4 did not show improvement, skipping reduce-overhead test")
            results["C4b_moe_compile_reduce_overhead"] = {
                "status": "SKIPPED",
                "reason": "C4 default mode not faster"
            }

        # ============================================================
        # Summary
        # ============================================================
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Config':<40s} {'Time(s)':>8s} {'Fwd':>5s} {'Fwd/s':>6s} {'ms/fwd':>7s} {'vs C0':>7s}")
        print(f"  {'-'*75}")
        for cname, r in results.items():
            if r.get("status") == "SKIPPED":
                print(f"  {cname:<40s} {'SKIPPED':>8s}")
                continue
            d = f"{r.get('delta_vs_c0_pct', 0):+.1f}%" if 'delta_vs_c0_pct' in r else "—"
            print(f"  {cname:<40s} {r['avg_time']:>8.3f} {r['avg_fwd']:>5.0f} "
                  f"{r['fwd_per_s']:>6.1f} {r['ms_per_fwd']:>7.2f} {d:>7s}")

        out_path = REPO_ROOT / "codex_coding" / "results" / "moe_compile_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
