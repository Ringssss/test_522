#!/usr/bin/env python3
"""
v0.1.15.4 Phase 3 — Optimized Baseline: max_unroll=4 + Fused RMSNorm

Two baseline optimizations applied simultaneously:
  1. maximum_unroll=4 (reduces Python loop overhead)
  2. Fused RMSNorm via vllm kernel (1 kernel vs 7 small kernels)

Runs:
  - Wall-clock timing: old baseline vs optimized baseline
  - Correctness check: compare forward counts
  - nsys-ready: NVTX markers for profiling (use --nsys flag)

Usage:
  # Wall-clock timing:
  python optimized_baseline.py

  # nsys profiling:
  nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx \
    --output=optimized_baseline \
    python optimized_baseline.py --nsys
"""

from __future__ import annotations
import os, sys, socket, json, time, argparse
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"

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


def apply_fused_rmsnorm(model):
    """Monkey-patch layer-level RMSNorm to use vllm's fused kernel.
    Skip attention-internal QK norms (query_layernorm, key_layernorm)
    because their inputs may not be contiguous after reshape/transpose."""
    from vllm.model_executor.layers.layernorm import rms_norm as vllm_rms_norm
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeRMSNorm

    # Only replace these specific norms (layer-level, always contiguous input)
    targets = set()
    for name, module in model.named_modules():
        if isinstance(module, LLaDA2MoeRMSNorm):
            # Skip attention-internal norms
            if "query_layernorm" in name or "key_layernorm" in name:
                continue
            targets.add(name)

    count = 0
    for name, module in model.named_modules():
        if name in targets:
            w = module.weight
            eps = module.variance_epsilon

            def make_fused_forward(weight, variance_epsilon):
                def fused_forward(hidden_states):
                    return vllm_rms_norm(hidden_states, weight, variance_epsilon)
                return fused_forward

            module.forward = make_fused_forward(w, eps)
            count += 1
    return count


def install_nvtx_hooks(model):
    """Install Level 2/3 NVTX markers (same as nsys_full_profiling.py)."""
    hooks = []
    nvtx = torch.cuda.nvtx

    # Embedding
    emb_mod = getattr(model.model, 'word_embeddings', getattr(model.model, 'embed_tokens', None))
    if emb_mod is not None:
        hooks.append(emb_mod.register_forward_pre_hook(lambda m, i: nvtx.range_push("Embedding")))
        hooks.append(emb_mod.register_forward_hook(lambda m, i, o: nvtx.range_pop()))

    for li, layer in enumerate(model.model.layers):
        is_moe = hasattr(layer.mlp, 'gate')
        tag = f"L{li}"

        # RMSNorm pre/post
        def mk_pre(t): return lambda m, i: nvtx.range_push(f"RMSNorm_pre_{t}")
        def mk_post(t): return lambda m, i, o: nvtx.range_pop()
        hooks.append(layer.input_layernorm.register_forward_pre_hook(mk_pre(tag)))
        hooks.append(layer.input_layernorm.register_forward_hook(mk_post(tag)))
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(mk_pre(f"post_{tag}")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(mk_post(f"post_{tag}")))

        # Attention (whole)
        attn = layer.attention
        def mk_a_pre(t): return lambda m, i: nvtx.range_push(f"Attention_{t}")
        def mk_a_post(t): return lambda m, i, o: nvtx.range_pop()
        hooks.append(attn.register_forward_pre_hook(mk_a_pre(tag)))
        hooks.append(attn.register_forward_hook(mk_a_post(tag)))

        # Attention sub: QKV, OProj
        hooks.append(attn.query_key_value.register_forward_pre_hook(
            (lambda t: lambda m, i: nvtx.range_push(f"Attn_QKV_{t}"))(tag)))
        hooks.append(attn.query_key_value.register_forward_hook(
            (lambda t: lambda m, i, o: nvtx.range_pop())(tag)))
        hooks.append(attn.dense.register_forward_pre_hook(
            (lambda t: lambda m, i: nvtx.range_push(f"Attn_OProj_{t}"))(tag)))
        hooks.append(attn.dense.register_forward_hook(
            (lambda t: lambda m, i, o: nvtx.range_pop())(tag)))

        # MoE / Dense MLP
        if is_moe:
            moe = layer.mlp
            idx_tag = f"L{li}"
            def make_moe_forward(moe_mod, lt):
                def patched_forward(hidden_states):
                    nvtx.range_push(f"MoE_{lt}")
                    nvtx.range_push(f"MoE_Shared_{lt}")
                    res = moe_mod.shared_experts(hidden_states)
                    nvtx.range_pop()
                    bsz, seq_len, h = hidden_states.shape
                    hs_flat = hidden_states.view(-1, h)
                    nvtx.range_push(f"MoE_Gate_{lt}")
                    router_logits = moe_mod.gate.get_logits(hs_flat)
                    nvtx.range_pop()
                    nvtx.range_push(f"MoE_Routed_{lt}")
                    y = moe_mod.experts.forward_impl(hidden_states=hs_flat, router_logits=router_logits)
                    nvtx.range_pop()
                    y = y.view(bsz, seq_len, h)
                    if moe_mod.config.num_shared_experts is not None:
                        y = y + res
                    nvtx.range_pop()
                    return y
                return patched_forward
            moe.forward = make_moe_forward(moe, idx_tag)
        else:
            hooks.append(layer.mlp.register_forward_pre_hook(
                (lambda t: lambda m, i: nvtx.range_push(f"DenseMLP_{t}"))(tag)))
            hooks.append(layer.mlp.register_forward_hook(
                (lambda t: lambda m, i, o: nvtx.range_pop())(tag)))

    # Final norm + LM head
    hooks.append(model.model.norm.register_forward_pre_hook(lambda m, i: nvtx.range_push("FinalRMSNorm")))
    hooks.append(model.model.norm.register_forward_hook(lambda m, i, o: nvtx.range_pop()))
    hooks.append(model.lm_head.register_forward_pre_hook(lambda m, i: nvtx.range_push("LMHead")))
    hooks.append(model.lm_head.register_forward_hook(lambda m, i, o: nvtx.range_pop()))

    return hooks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys", action="store_true", help="nsys mode: add NVTX + cudaProfilerApi")
    args = parser.parse_args()

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
    print("Optimized Baseline: max_unroll=4 + Fused RMSNorm")
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
        print(f"Input: {input_ids.shape}")

        GEN_LENGTH = 256
        decoder = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        # ============================================================
        # Run 1: Old baseline (max_unroll=1, plain RMSNorm)
        # ============================================================
        if not args.nsys:
            print("\n--- Old Baseline (max_unroll=1, plain RMSNorm) ---")
            dllm_old = BlockDiffusionLLM(
                model, decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=1, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)
            with torch.inference_mode():
                dllm_old.diff_iteration.num_forwards = 0
                _ = dllm_old.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm_old.diff_iteration.num_forwards = 0
                _ = dllm_old.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            old_fwd = dllm_old.diff_iteration.num_forwards
            old_time = t1 - t0
            print(f"  Fwd={old_fwd}, time={old_time:.3f}s, fwd/s={old_fwd/old_time:.1f}")

        # ============================================================
        # Apply optimizations
        # ============================================================
        print("\n--- Applying Fused RMSNorm ---")
        n_fused = apply_fused_rmsnorm(model)
        print(f"  Replaced {n_fused} RMSNorm modules with vllm fused kernel")

        # ============================================================
        # Run 2: Optimized baseline (max_unroll=4, fused RMSNorm)
        # ============================================================
        print("\n--- Optimized Baseline (max_unroll=4, fused RMSNorm) ---")
        dllm_new = BlockDiffusionLLM(
            model, decoder,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=4, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Warmup
        with torch.inference_mode():
            dllm_new.diff_iteration.num_forwards = 0
            _ = dllm_new.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        if args.nsys:
            # nsys mode: install NVTX + cudaProfilerApi
            print("\nInstalling NVTX hooks...")
            nvtx_hooks = install_nvtx_hooks(model)
            print(f"  {len(nvtx_hooks)} hooks installed.")

            print("\nStarting profiled generation...")
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            with torch.inference_mode():
                dllm_new.diff_iteration.num_forwards = 0
                _ = dllm_new.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            new_fwd = dllm_new.diff_iteration.num_forwards
            print(f"  Profiled: Fwd={new_fwd}")

            for h in nvtx_hooks:
                h.remove()
        else:
            # Timing mode
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm_new.diff_iteration.num_forwards = 0
                _ = dllm_new.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            new_fwd = dllm_new.diff_iteration.num_forwards
            new_time = t1 - t0
            print(f"  Fwd={new_fwd}, time={new_time:.3f}s, fwd/s={new_fwd/new_time:.1f}")

            # Summary
            print(f"\n{'='*80}")
            print(f"SUMMARY")
            print(f"{'='*80}")
            print(f"  {'Config':<40s} {'Fwd':>4s} {'Time(s)':>8s} {'Fwd/s':>6s} {'vs Old':>8s}")
            print(f"  {'-'*68}")
            print(f"  {'Old (unroll=1, plain RMSNorm)':<40s} {old_fwd:>4d} {old_time:>8.3f} {old_fwd/old_time:>6.1f} {'—':>8s}")
            speedup = (old_time - new_time) / old_time * 100
            print(f"  {'New (unroll=4, fused RMSNorm)':<40s} {new_fwd:>4d} {new_time:>8.3f} {new_fwd/new_time:>6.1f} {speedup:>+7.1f}%")

            # Save
            save_data = {
                "old_baseline": {"fwd": old_fwd, "time_s": old_time, "config": "unroll=1, plain RMSNorm"},
                "optimized_baseline": {"fwd": new_fwd, "time_s": new_time, "config": "unroll=4, fused RMSNorm"},
                "speedup_pct": speedup,
            }
            out_path = REPO_ROOT / "codex_coding" / "results" / "optimized_baseline_timing.json"
            with open(out_path, "w") as f:
                json.dump(save_data, f, indent=2, default=str)
            print(f"\n  Saved to {out_path}")

        print("\nDone.")


if __name__ == "__main__":
    main()
