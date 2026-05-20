#!/usr/bin/env python3
"""
v0.1.15.4 — nsys Profiling on Optimized Baseline (Lightweight Instrumentation)

Optimized baseline = max_unroll=4 + Fused RMSNorm + Classic flash-attn 2.8.3

Instrumentation: module-level NVTX hooks ONLY (no forward rewrites).
  Per-layer: RMSNorm_pre, Attention, RMSNorm_post, MoE/DenseMLP
  Global: Embedding, FinalRMSNorm, LMHead

Three runs:
  1. Wall-clock (no hooks) — confirm baseline matches ~8.789s
  2. Wall-clock (with hooks) — verify hook overhead is small
  3. cudaProfilerApi capture — nsys profiled generation

Usage:
  # Timing only:
  python nsys_optimized_baseline_profiling.py

  # nsys profiling:
  nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx \
    --output=codex_coding/results/optimized_baseline_level2 \
    python codex_coding/src/nsys_optimized_baseline_profiling.py

  # Analysis:
  nsys stats --report nvtx_sum codex_coding/results/optimized_baseline_level2.nsys-rep
"""

from __future__ import annotations
import os, sys, socket, json, time
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


# ==============================================================
# Optimization 1: Fused RMSNorm (from optimized_baseline_v3.py)
# ==============================================================
def apply_fused_rmsnorm(model):
    """Fuse layer-level RMSNorm (skip attention-internal QK norms)."""
    from vllm.model_executor.layers.layernorm import rms_norm as vllm_rms_norm
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeRMSNorm
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, LLaDA2MoeRMSNorm):
            if "query_layernorm" in name or "key_layernorm" in name:
                continue
            w, eps = module.weight, module.variance_epsilon
            def _mk(ww, ee):
                def _f(hs): return vllm_rms_norm(hs, ww, ee)
                return _f
            module.forward = _mk(w, eps)
            count += 1
    return count


# ==============================================================
# Optimization 2: Classic flash-attn 2.8 (from optimized_baseline_v3.py)
# ==============================================================
def apply_flash_attn_classic(model):
    """Replace SDPA with classic flash-attn 2.8 + QK norm before transpose."""
    from flash_attn import flash_attn_func
    from vllm.model_executor.layers.layernorm import rms_norm as vllm_rms_norm
    from dinfer.model.modeling_llada2_moe import (
        LLaDA2MoeSdpaAttention, apply_rotary_pos_emb,
    )

    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, LLaDA2MoeSdpaAttention):
            continue

        def make_fa_forward(attn_mod):
            q_norm_w = attn_mod.query_layernorm.weight
            q_norm_eps = attn_mod.query_layernorm.variance_epsilon
            k_norm_w = attn_mod.key_layernorm.weight
            k_norm_eps = attn_mod.key_layernorm.variance_epsilon

            def fa_forward(
                hidden_states: torch.Tensor,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions: bool = False,
                use_cache: bool = False,
                position_embeddings=None,
                cache_position=None,
                replace_position=None,
                **kwargs,
            ):
                bsz, q_len, _ = hidden_states.size()
                num_heads = attn_mod.num_heads // attn_mod.tp_size
                num_kv_heads = attn_mod.num_key_value_heads // attn_mod.tp_size
                head_dim = attn_mod.head_dim

                qkv = attn_mod.query_key_value(hidden_states)
                qkv = qkv.view(bsz, q_len, num_heads + 2 * num_kv_heads, head_dim)
                query_states, key_states, value_states = qkv.split(
                    [num_heads, num_kv_heads, num_kv_heads], dim=-2)

                # QK Norm BEFORE transpose (contiguous -> fused RMSNorm works)
                query_states = vllm_rms_norm(query_states, q_norm_w, q_norm_eps)
                key_states = vllm_rms_norm(key_states, k_norm_w, k_norm_eps)

                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin)

                if past_key_value is not None:
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, attn_mod.layer_idx, replace_position)
                if use_cache:
                    past_key_value = (key_states, value_states)

                q_fa = query_states.transpose(1, 2)
                k_fa = key_states.transpose(1, 2)
                v_fa = value_states.transpose(1, 2)

                if attention_mask is not None:
                    from dinfer.model.modeling_llada2_moe import repeat_kv
                    num_kv_groups = num_heads // num_kv_heads
                    k_exp = repeat_kv(key_states, num_kv_groups).contiguous()
                    v_exp = repeat_kv(value_states, num_kv_groups).contiguous()
                    q_c = query_states.contiguous()
                    am = attention_mask.bool()
                    if am.dim() == 3:
                        am = am.unsqueeze(1)
                    attn_output = F.scaled_dot_product_attention(
                        q_c, k_exp, v_exp, attn_mask=am, dropout_p=0.0, is_causal=False)
                    attn_output = attn_output.transpose(1, 2).contiguous()
                else:
                    attn_output = flash_attn_func(
                        q_fa.contiguous(), k_fa.contiguous(), v_fa.contiguous(),
                        causal=False)

                attn_output = attn_output.reshape(bsz, q_len, -1)
                attn_output = attn_mod.dense(attn_output)
                return attn_output, None, past_key_value

            return fa_forward

        module.forward = make_fa_forward(module)
        count += 1

    return count


# ==============================================================
# Lightweight NVTX hooks (module-level only, no forward rewrites)
# ==============================================================
def install_lightweight_nvtx_hooks(model):
    """Install NVTX markers at module level only. No forward rewrites."""
    hooks = []
    nvtx = torch.cuda.nvtx

    def make_pre(tag):
        def f(mod, inp): nvtx.range_push(tag)
        return f
    def make_post(tag):
        def f(mod, inp, out): nvtx.range_pop()
        return f

    # Embedding
    emb = getattr(model.model, 'word_embeddings',
                  getattr(model.model, 'embed_tokens', None))
    if emb is not None:
        hooks.append(emb.register_forward_pre_hook(make_pre("Embedding")))
        hooks.append(emb.register_forward_hook(make_post("Embedding")))

    # Per-layer
    for li, layer in enumerate(model.model.layers):
        tag = f"L{li}"
        is_moe = hasattr(layer.mlp, 'gate')

        # Input RMSNorm
        hooks.append(layer.input_layernorm.register_forward_pre_hook(
            make_pre(f"RMSNorm_pre_{tag}")))
        hooks.append(layer.input_layernorm.register_forward_hook(
            make_post(f"RMSNorm_pre_{tag}")))

        # Attention (whole module)
        attn = layer.attention if hasattr(layer, 'attention') else layer.self_attn
        hooks.append(attn.register_forward_pre_hook(
            make_pre(f"Attention_{tag}")))
        hooks.append(attn.register_forward_hook(
            make_post(f"Attention_{tag}")))

        # Post-attention RMSNorm
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
            make_pre(f"RMSNorm_post_{tag}")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            make_post(f"RMSNorm_post_{tag}")))

        # MLP/MoE (whole module, no internal decomposition)
        mlp_tag = f"MoE_{tag}" if is_moe else f"DenseMLP_{tag}"
        hooks.append(layer.mlp.register_forward_pre_hook(make_pre(mlp_tag)))
        hooks.append(layer.mlp.register_forward_hook(make_post(mlp_tag)))

    # Final RMSNorm
    hooks.append(model.model.norm.register_forward_pre_hook(
        make_pre("FinalRMSNorm")))
    hooks.append(model.model.norm.register_forward_hook(
        make_post("FinalRMSNorm")))

    # LM Head
    hooks.append(model.lm_head.register_forward_pre_hook(make_pre("LMHead")))
    hooks.append(model.lm_head.register_forward_hook(make_post("LMHead")))

    return hooks


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
    print("nsys Profiling — Optimized Baseline (Lightweight Instrumentation)")
    print("  Optimizations: max_unroll=4 + Fused RMSNorm + Classic flash-attn 2.8.3")
    print("  HetEval-32: batch=32, gen=256, block=32, threshold=0.90, temp=0")
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

        # ---- Apply optimizations ----
        print("\nApplying optimizations...")
        n_rms = apply_fused_rmsnorm(model)
        print(f"  Fused RMSNorm: {n_rms} modules")
        n_fa = apply_flash_attn_classic(model)
        print(f"  Flash-attn classic: {n_fa} attention modules")

        # ---- Build input (HetEval-32) ----
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
        print(f"  Input shape: {input_ids.shape}")

        GEN_LENGTH = 256
        decoder = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # ---- Warmup ----
        print("\nWarmup...")
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print("  Warmup done.")

        # ============================================================
        # Run 1: Wall-clock timing WITHOUT hooks
        # ============================================================
        print("\n--- Run 1: Wall-clock (no hooks) ---")
        dllm = make_dllm()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd_nohook = dllm.diff_iteration.num_forwards
        time_nohook = t1 - t0
        print(f"  Fwd={fwd_nohook}, time={time_nohook:.3f}s, "
              f"fwd/s={fwd_nohook/time_nohook:.1f}")

        # ============================================================
        # Install lightweight NVTX hooks
        # ============================================================
        print("\n--- Installing lightweight NVTX hooks ---")
        nvtx_hooks = install_lightweight_nvtx_hooks(model)
        print(f"  {len(nvtx_hooks)} hooks installed "
              f"(module-level only, no forward rewrites)")

        # ============================================================
        # Run 2: Wall-clock timing WITH hooks
        # ============================================================
        print("\n--- Run 2: Wall-clock (with hooks) ---")
        dllm = make_dllm()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd_hook = dllm.diff_iteration.num_forwards
        time_hook = t1 - t0
        hook_overhead = (time_hook - time_nohook) / time_nohook * 100
        print(f"  Fwd={fwd_hook}, time={time_hook:.3f}s, "
              f"fwd/s={fwd_hook/time_hook:.1f}")
        print(f"  Hook overhead: {hook_overhead:+.1f}%")

        # ============================================================
        # Run 3: nsys profiled generation (cudaProfilerApi)
        # ============================================================
        print("\n--- Run 3: nsys profiled generation ---")
        print("  (cudaProfilerStart -> full generation -> cudaProfilerStop)")
        dllm = make_dllm()

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()

        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

        fwd_prof = dllm.diff_iteration.num_forwards
        print(f"  Profiled generation: {fwd_prof} forwards")

        # ============================================================
        # Summary
        # ============================================================
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Run':<30s} {'Fwd':>4s} {'Time(s)':>8s} {'Fwd/s':>6s}")
        print(f"  {'-'*52}")
        print(f"  {'No hooks (true baseline)':<30s} {fwd_nohook:>4d} "
              f"{time_nohook:>8.3f} {fwd_nohook/time_nohook:>6.1f}")
        print(f"  {'With NVTX hooks':<30s} {fwd_hook:>4d} "
              f"{time_hook:>8.3f} {fwd_hook/time_hook:>6.1f}")
        print(f"  {'Profiled (for nsys)':<30s} {fwd_prof:>4d} "
              f"{'—':>8s} {'—':>6s}")
        print(f"\n  Hook overhead: {hook_overhead:+.1f}%")

        # Save timing
        save_data = {
            "no_hooks": {"fwd": fwd_nohook, "time_s": time_nohook},
            "with_hooks": {"fwd": fwd_hook, "time_s": time_hook,
                           "overhead_pct": hook_overhead},
            "profiled": {"fwd": fwd_prof},
            "optimizations": "max_unroll=4 + fused_rmsnorm + flash_attn_classic_2.8.3",
            "config": "HetEval-32: batch=32, gen=256, block=32, threshold=0.90",
        }
        out_path = (REPO_ROOT / "codex_coding" / "results" /
                    "optimized_baseline_profiling_timing.json")
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n  Timing saved to {out_path}")

        # Cleanup
        for h in nvtx_hooks:
            h.remove()

        print("\nDone. If run under nsys, analyze with:")
        print("  nsys stats --report nvtx_sum <output>.nsys-rep")


if __name__ == "__main__":
    main()
