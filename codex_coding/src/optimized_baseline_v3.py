#!/usr/bin/env python3
"""
HetEval-32: Optimized Baseline v3
  max_unroll=4 + Fused RMSNorm + Flashinfer Attention + QK Norm fuse

Optimizations:
  1. maximum_unroll=4 (reduce Python loop overhead)
  2. Fused RMSNorm via vllm kernel (layer-level norms)
  3. QK Norm moved before transpose (contiguous → can use fused RMSNorm)
  4. Flashinfer attention replacing SDPA (native GQA, native NHD, no repeat_kv,
     no contiguous copies, pre-compiled H100 kernel)

Wall-clock timing + HetEval-32 quality verification.
"""

from __future__ import annotations
import os, sys, socket, json, time
from pathlib import Path
from typing import Optional, Tuple

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

VERIFIABLE = {
    0: "average speed = 480/7 ≈ 68.57 km/h",
    8: "x = 2 and x = 3",
    13: "B is true, C is true, D is true",
    19: "fib(10) = 55",
    28: "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune",
}


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


def apply_flash_attn_classic(model):
    """Replace SDPA with classic flash-attn 2.8 + QK norm before transpose + no repeat_kv."""
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
            # Pre-extract QK norm params for fused RMSNorm
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

                # Step 1: QKV projection → (B, S, H_total, D) contiguous
                qkv = attn_mod.query_key_value(hidden_states)
                qkv = qkv.view(bsz, q_len, num_heads + 2 * num_kv_heads, head_dim)
                query_states, key_states, value_states = qkv.split(
                    [num_heads, num_kv_heads, num_kv_heads], dim=-2)
                # Now: Q(B,S,H,D), K(B,S,Hkv,D), V(B,S,Hkv,D) - all contiguous

                # Step 2: QK Norm BEFORE transpose (contiguous → fused RMSNorm)
                query_states = vllm_rms_norm(query_states, q_norm_w, q_norm_eps)
                key_states = vllm_rms_norm(key_states, k_norm_w, k_norm_eps)

                # Step 3: Transpose for RoPE + cache (need B,H,S,D)
                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)

                # Step 4: RoPE
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin)

                # Step 5: Cache update
                if past_key_value is not None:
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, attn_mod.layer_idx, replace_position)
                if use_cache:
                    past_key_value = (key_states, value_states)

                # Step 6: Flash-Attn attention
                # Transpose back to (B,S,H,D) for flash_attn_func
                q_fa = query_states.transpose(1, 2)
                k_fa = key_states.transpose(1, 2)
                v_fa = value_states.transpose(1, 2)

                if attention_mask is not None:
                    # Fall back to SDPA when custom mask is present
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
                    # Classic flash-attn: native GQA, (B,S,H,D), no repeat_kv
                    attn_output = flash_attn_func(
                        q_fa.contiguous(), k_fa.contiguous(), v_fa.contiguous(),
                        causal=False)

                # Step 7: reshape + OProj
                attn_output = attn_output.reshape(bsz, q_len, -1)
                attn_output = attn_mod.dense(attn_output)

                return attn_output, None, past_key_value

            return fa_forward

        module.forward = make_fa_forward(module)
        count += 1

    return count


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
    print("HetEval-32: Optimized Baseline v3")
    print("  max_unroll=4 + fused RMSNorm + classic flash-attn 2.8 + QK norm fuse")
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
        GEN_LENGTH = 256

        decoder_t0 = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        # ---- Config A: prev best (unroll=4, fused RMSNorm, SDPA) ----
        print("\n--- Config A: unroll=4 + fused RMSNorm (SDPA) ---")
        n_rms = apply_fused_rmsnorm(model)
        print(f"  Fused RMSNorm: {n_rms} modules")

        dllm_a = BlockDiffusionLLM(
            model, decoder_t0,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=4, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)
        with torch.inference_mode():
            dllm_a.diff_iteration.num_forwards = 0
            _ = dllm_a.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm_a.diff_iteration.num_forwards = 0
            _ = dllm_a.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd_a, time_a = dllm_a.diff_iteration.num_forwards, t1 - t0
        print(f"  Fwd={fwd_a}, time={time_a:.3f}s, fwd/s={fwd_a/time_a:.1f}")

        # ---- Config B: + flashinfer + QK norm fuse ----
        print("\n--- Config B: A + classic flash-attn 2.8 + QK norm fuse ---")
        n_fa = apply_flash_attn_classic(model)
        print(f"  Flash-attn classic: {n_fa} attention modules")

        dllm_b = BlockDiffusionLLM(
            model, decoder_t0,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=4, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)
        with torch.inference_mode():
            dllm_b.diff_iteration.num_forwards = 0
            _ = dllm_b.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm_b.diff_iteration.num_forwards = 0
            _ = dllm_b.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd_b, time_b = dllm_b.diff_iteration.num_forwards, t1 - t0
        print(f"  Fwd={fwd_b}, time={time_b:.3f}s, fwd/s={fwd_b/time_b:.1f}")

        # ---- Quality check with temp=0.7 ----
        print(f"\n--- Quality Check (temp=0.7, Config B) ---")
        decoder_t7 = ThresholdParallelDecoder(
            temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm_q = BlockDiffusionLLM(
            model, decoder_t7,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=4, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)
        with torch.inference_mode():
            dllm_q.diff_iteration.num_forwards = 0
            _ = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        with torch.inference_mode():
            dllm_q.diff_iteration.num_forwards = 0
            out = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        gen_tokens = out[:, prompt_len:]

        for bi in sorted(VERIFIABLE.keys()):
            gt = gen_tokens[bi]
            valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
            text = tokenizer.decode(valid, skip_special_tokens=True)
            print(f"\n  [V] Prompt #{bi}: {PROMPTS[bi][:60]}...")
            print(f"  Expected: {VERIFIABLE[bi]}")
            print(f"  Output: {text[:500]}")

        # ---- Summary ----
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Config':<50s} {'Fwd':>4s} {'Time(s)':>8s} {'Fwd/s':>6s}")
        print(f"  {'-'*70}")
        print(f"  {'A: unroll=4 + fused RMSNorm (SDPA)':<50s} {fwd_a:>4d} {time_a:>8.3f} {fwd_a/time_a:>6.1f}")
        print(f"  {'B: A + classic flash-attn + QK norm fuse':<50s} {fwd_b:>4d} {time_b:>8.3f} {fwd_b/time_b:>6.1f}")
        if time_b < time_a:
            pct = (time_a - time_b) / time_a * 100
            print(f"\n  Flash-attn classic + QK fuse speedup: {pct:.1f}%")
        else:
            pct = (time_b - time_a) / time_a * 100
            print(f"\n  Flash-attn classic + QK fuse overhead: +{pct:.1f}%")

        print("\nDone.")


if __name__ == "__main__":
    main()
