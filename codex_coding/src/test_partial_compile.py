#!/usr/bin/env python3
"""
v0.1.15.6 — Partial Compile (Prism-style) Experiment

Configs:
  C0: Baseline (no compile) — re-confirm existing 8.8s
  C1: torch.compile(model) + torch._dynamo.disable on MoE submodules
  C2: C1 + torch.compile(gate.forward) selectively
  C3: Full compile (MoE not excluded) — upper-bound probe, may fail

All on HetEval-32: batch=32, gen=256, block=32, threshold=0.90, max_unroll=4
Baseline optimizations: fused RMSNorm + classic flash-attn 2.8.3
"""

from __future__ import annotations
import os, sys, time, socket, json, logging
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


# ================================================================
# Compile helpers
# ================================================================

def apply_partial_compile_c1(model):
    """C1: compile entire model, disable dynamo on MoE submodules."""
    import torch._dynamo as _dynamo

    # Disable dynamo on MoE submodules
    disabled_count = 0
    for name, sub in model.named_modules():
        cname = sub.__class__.__name__
        if cname in ("LLaDA2MoeSparseMoeBlock",):
            try:
                sub.forward = _dynamo.disable(sub.forward)
                disabled_count += 1
            except Exception as e:
                print(f"  [WARN] Failed to disable {name}: {e}")
    print(f"  Disabled dynamo on {disabled_count} MoE submodules")

    # Disable Inductor internal CUDA graphs (KV cache compat)
    try:
        import torch._inductor.config as _ind_cfg
        if hasattr(_ind_cfg, 'triton') and hasattr(_ind_cfg.triton, 'cudagraphs'):
            _ind_cfg.triton.cudagraphs = False
        if hasattr(_ind_cfg, 'cudagraph_trees'):
            _ind_cfg.cudagraph_trees = False
    except Exception:
        pass

    compiled = torch.compile(model, mode='default', fullgraph=False)
    return compiled


def apply_partial_compile_c2(model):
    """C2: like C1, but also compile gate.forward separately."""
    import torch._dynamo as _dynamo

    # First: disable dynamo on MoE block level (but NOT gate)
    disabled_count = 0
    gate_compiled = 0
    for name, sub in model.named_modules():
        cname = sub.__class__.__name__
        if cname == "LLaDA2MoeSparseMoeBlock":
            # Disable the block-level forward, but we'll compile gate separately
            try:
                sub.forward = _dynamo.disable(sub.forward)
                disabled_count += 1
            except Exception as e:
                print(f"  [WARN] Failed to disable {name}: {e}")

    # Compile gate.forward individually
    for name, sub in model.named_modules():
        cname = sub.__class__.__name__
        if cname == "LLaDA2MoeGate":
            try:
                sub.forward = torch.compile(sub.forward, mode='default', fullgraph=False, dynamic=True)
                gate_compiled += 1
            except Exception as e:
                print(f"  [WARN] Failed to compile gate {name}: {e}")

    print(f"  Disabled dynamo on {disabled_count} MoE blocks, compiled {gate_compiled} gates")

    # Disable Inductor internal CUDA graphs
    try:
        import torch._inductor.config as _ind_cfg
        if hasattr(_ind_cfg, 'triton') and hasattr(_ind_cfg.triton, 'cudagraphs'):
            _ind_cfg.triton.cudagraphs = False
        if hasattr(_ind_cfg, 'cudagraph_trees'):
            _ind_cfg.cudagraph_trees = False
    except Exception:
        pass

    compiled = torch.compile(model, mode='default', fullgraph=False)
    return compiled


def apply_full_compile_c3(model):
    """C3: compile everything (MoE included), fullgraph=False, dynamic=True."""
    # Disable Inductor internal CUDA graphs
    try:
        import torch._inductor.config as _ind_cfg
        if hasattr(_ind_cfg, 'triton') and hasattr(_ind_cfg.triton, 'cudagraphs'):
            _ind_cfg.triton.cudagraphs = False
        if hasattr(_ind_cfg, 'cudagraph_trees'):
            _ind_cfg.cudagraph_trees = False
    except Exception:
        pass

    compiled = torch.compile(model, mode='default', fullgraph=False, dynamic=True)
    return compiled


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
    print("v0.1.15.6 — Partial Compile (Prism-style) Experiment")
    print("  HetEval-32: batch=32, gen=256, block=32, threshold=0.90, max_unroll=4")
    print("  Baseline: fused RMSNorm + classic flash-attn 2.8.3")
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

        # Dummy forward to init lazy modules
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0),
                      use_cache=False)

        # Apply baseline optimizations (fused RMSNorm + flash-attn)
        print("\nApplying baseline optimizations...")
        n_rms, n_fa = apply_all_optimizations(model)
        print(f"  Fused RMSNorm: {n_rms}, Flash-attn: {n_fa}")

        # Build input (HetEval-32)
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

        def make_dllm(m, decoder):
            return BlockDiffusionLLM(
                m, decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True,
                inplace_cache_update=True)

        # ============================================================
        # C0: Baseline (no compile)
        # ============================================================
        print(f"\n{'='*70}")
        print("C0: Baseline (no compile)")
        print(f"{'='*70}")

        # Warmup
        dllm = make_dllm(model, decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        # Timing (2 runs)
        c0_times = []
        c0_fwds = []
        for run_i in range(2):
            dllm = make_dllm(model, decoder_t0)
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

        c0_avg_time = sum(c0_times) / len(c0_times)
        c0_avg_fwd = sum(c0_fwds) / len(c0_fwds)
        print(f"  C0 avg: {c0_avg_time:.3f}s, {c0_avg_fwd:.0f} fwd, {c0_avg_fwd/c0_avg_time:.1f} fwd/s")

        results = OrderedDict()
        results["C0_baseline"] = {
            "times": c0_times, "fwds": c0_fwds,
            "avg_time": c0_avg_time, "avg_fwd": c0_avg_fwd,
            "avg_fwd_per_s": c0_avg_fwd / c0_avg_time,
        }

        # ============================================================
        # C1: Partial compile (MoE excluded)
        # ============================================================
        print(f"\n{'='*70}")
        print("C1: Partial compile (MoE excluded via dynamo.disable)")
        print(f"{'='*70}")

        # Reset dynamo state
        torch._dynamo.reset()

        try:
            compiled_model_c1 = apply_partial_compile_c1(model)

            # Compile warmup (first generate triggers compilation)
            print("  Compile warmup (may take 30-120s)...")
            t_comp_start = time.perf_counter()
            dllm = make_dllm(compiled_model_c1, decoder_t0)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t_comp_end = time.perf_counter()
            print(f"  Compile warmup done: {t_comp_end - t_comp_start:.1f}s")

            # Timing (2 runs)
            c1_times = []
            c1_fwds = []
            for run_i in range(2):
                dllm = make_dllm(compiled_model_c1, decoder_t0)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                c1_times.append(t1 - t0)
                c1_fwds.append(dllm.diff_iteration.num_forwards)
                print(f"  Run {run_i+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd, "
                      f"{dllm.diff_iteration.num_forwards/(t1-t0):.1f} fwd/s")

            c1_avg_time = sum(c1_times) / len(c1_times)
            c1_avg_fwd = sum(c1_fwds) / len(c1_fwds)
            delta_pct = (c1_avg_time - c0_avg_time) / c0_avg_time * 100
            print(f"  C1 avg: {c1_avg_time:.3f}s, {c1_avg_fwd:.0f} fwd, {c1_avg_fwd/c1_avg_time:.1f} fwd/s")
            print(f"  vs C0: {delta_pct:+.1f}%")

            results["C1_partial_compile"] = {
                "times": c1_times, "fwds": c1_fwds,
                "avg_time": c1_avg_time, "avg_fwd": c1_avg_fwd,
                "avg_fwd_per_s": c1_avg_fwd / c1_avg_time,
                "delta_vs_c0_pct": delta_pct,
                "compile_warmup_s": t_comp_end - t_comp_start,
                "status": "OK",
            }

            # Quality check (temp=0.7)
            print("\n  Quality check (temp=0.7)...")
            decoder_t7 = ThresholdParallelDecoder(
                temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
            dllm_q = make_dllm(compiled_model_c1, decoder_t7)
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
            results["C1_partial_compile"]["quality"] = quality

        except Exception as e:
            print(f"  C1 FAILED: {e}")
            import traceback; traceback.print_exc()
            results["C1_partial_compile"] = {"status": "FAILED", "error": str(e)}

        # ============================================================
        # C2: C1 + compile gate.forward
        # ============================================================
        print(f"\n{'='*70}")
        print("C2: Partial compile + compile gate.forward")
        print(f"{'='*70}")

        # Need fresh model for C2 (dynamo state from C1 may interfere)
        torch._dynamo.reset()

        try:
            compiled_model_c2 = apply_partial_compile_c2(model)

            # Compile warmup
            print("  Compile warmup...")
            t_comp_start = time.perf_counter()
            dllm = make_dllm(compiled_model_c2, decoder_t0)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t_comp_end = time.perf_counter()
            print(f"  Compile warmup done: {t_comp_end - t_comp_start:.1f}s")

            # Timing (2 runs)
            c2_times = []
            c2_fwds = []
            for run_i in range(2):
                dllm = make_dllm(compiled_model_c2, decoder_t0)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                c2_times.append(t1 - t0)
                c2_fwds.append(dllm.diff_iteration.num_forwards)
                print(f"  Run {run_i+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd, "
                      f"{dllm.diff_iteration.num_forwards/(t1-t0):.1f} fwd/s")

            c2_avg_time = sum(c2_times) / len(c2_times)
            c2_avg_fwd = sum(c2_fwds) / len(c2_fwds)
            delta_pct = (c2_avg_time - c0_avg_time) / c0_avg_time * 100
            print(f"  C2 avg: {c2_avg_time:.3f}s, {c2_avg_fwd:.0f} fwd, {c2_avg_fwd/c2_avg_time:.1f} fwd/s")
            print(f"  vs C0: {delta_pct:+.1f}%")

            results["C2_compile_gate"] = {
                "times": c2_times, "fwds": c2_fwds,
                "avg_time": c2_avg_time, "avg_fwd": c2_avg_fwd,
                "avg_fwd_per_s": c2_avg_fwd / c2_avg_time,
                "delta_vs_c0_pct": delta_pct,
                "compile_warmup_s": t_comp_end - t_comp_start,
                "status": "OK",
            }
        except Exception as e:
            print(f"  C2 FAILED: {e}")
            import traceback; traceback.print_exc()
            results["C2_compile_gate"] = {"status": "FAILED", "error": str(e)}

        # ============================================================
        # C3: Full compile (MoE not excluded)
        # ============================================================
        print(f"\n{'='*70}")
        print("C3: Full compile (MoE included, dynamic=True)")
        print(f"{'='*70}")

        torch._dynamo.reset()

        try:
            compiled_model_c3 = apply_full_compile_c3(model)

            # Compile warmup
            print("  Compile warmup (may be slow or fail)...")
            t_comp_start = time.perf_counter()
            dllm = make_dllm(compiled_model_c3, decoder_t0)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t_comp_end = time.perf_counter()
            print(f"  Compile warmup done: {t_comp_end - t_comp_start:.1f}s")

            # Timing (2 runs)
            c3_times = []
            c3_fwds = []
            for run_i in range(2):
                dllm = make_dllm(compiled_model_c3, decoder_t0)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                c3_times.append(t1 - t0)
                c3_fwds.append(dllm.diff_iteration.num_forwards)
                print(f"  Run {run_i+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd, "
                      f"{dllm.diff_iteration.num_forwards/(t1-t0):.1f} fwd/s")

            c3_avg_time = sum(c3_times) / len(c3_times)
            c3_avg_fwd = sum(c3_fwds) / len(c3_fwds)
            delta_pct = (c3_avg_time - c0_avg_time) / c0_avg_time * 100
            print(f"  C3 avg: {c3_avg_time:.3f}s, {c3_avg_fwd:.0f} fwd, {c3_avg_fwd/c3_avg_time:.1f} fwd/s")
            print(f"  vs C0: {delta_pct:+.1f}%")

            results["C3_full_compile"] = {
                "times": c3_times, "fwds": c3_fwds,
                "avg_time": c3_avg_time, "avg_fwd": c3_avg_fwd,
                "avg_fwd_per_s": c3_avg_fwd / c3_avg_time,
                "delta_vs_c0_pct": delta_pct,
                "compile_warmup_s": t_comp_end - t_comp_start,
                "status": "OK",
            }
        except Exception as e:
            print(f"  C3 FAILED: {e}")
            import traceback; traceback.print_exc()
            results["C3_full_compile"] = {"status": "FAILED", "error": str(e)}

        # ============================================================
        # Summary
        # ============================================================
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Config':<35s} {'Status':>8s} {'Time(s)':>8s} {'Fwd':>5s} {'Fwd/s':>6s} {'vs C0':>7s}")
        print(f"  {'-'*75}")
        for cname, r in results.items():
            if r.get("status") == "FAILED":
                print(f"  {cname:<35s} {'FAILED':>8s}   —     —      —       —")
            else:
                delta_str = f"{r.get('delta_vs_c0_pct', 0):+.1f}%" if 'delta_vs_c0_pct' in r else "—"
                print(f"  {cname:<35s} {'OK':>8s} {r['avg_time']:>8.3f} {r['avg_fwd']:>5.0f} "
                      f"{r['avg_fwd_per_s']:>6.1f} {delta_str:>7s}")

        # Save results
        out_path = REPO_ROOT / "codex_coding" / "results" / "partial_compile_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
