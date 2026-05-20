#!/usr/bin/env python3
"""
v0.1.15.5 — Expert Budgeting Inline (Simple Version)

Baseline optimizations: max_unroll=4 + Fused RMSNorm + Classic flash-attn 2.8.3
+ Expert Budgeting: popularity top-S, no safety check, zero Python loops.

Tests:
  1. Wall-clock: baseline (no EB) vs EB with S_size = 128, 112, 96, 80
  2. Quality: HetEval-32 temp=0.7, 5 verifiable prompts

The inline mechanism:
  gate.get_logits -> router_logits [N, 256]
  NEW: sigmoid -> sum(dim=0) -> topk(S_size) -> mask -> logits[:, ~S] = -inf
  experts.forward_impl(x, masked_logits)
  -> internal sigmoid(-inf)=0 -> those experts never selected
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

VERIFIABLE = {
    0: "average speed = 480/7 ≈ 68.57 km/h",
    8: "x = 2 and x = 3",
    13: "B is true, C is true, D is true",
    19: "fib(10) = 55",
    28: "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune",
}


# ==============================================================
# Optimization 1: Fused RMSNorm
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
# Optimization 2: Classic flash-attn 2.8
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
                hidden_states: torch.Tensor, attention_mask=None,
                position_ids=None, past_key_value=None,
                output_attentions: bool = False, use_cache: bool = False,
                position_embeddings=None, cache_position=None,
                replace_position=None, **kwargs,
            ):
                bsz, q_len, _ = hidden_states.size()
                num_heads = attn_mod.num_heads // attn_mod.tp_size
                num_kv_heads = attn_mod.num_key_value_heads // attn_mod.tp_size
                head_dim = attn_mod.head_dim

                qkv = attn_mod.query_key_value(hidden_states)
                qkv = qkv.view(bsz, q_len, num_heads + 2 * num_kv_heads, head_dim)
                query_states, key_states, value_states = qkv.split(
                    [num_heads, num_kv_heads, num_kv_heads], dim=-2)

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
                        key_states, value_states, attn_mod.layer_idx,
                        replace_position)
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
                        q_c, k_exp, v_exp, attn_mask=am, dropout_p=0.0,
                        is_causal=False)
                    attn_output = attn_output.transpose(1, 2).contiguous()
                else:
                    attn_output = flash_attn_func(
                        q_fa.contiguous(), k_fa.contiguous(),
                        v_fa.contiguous(), causal=False)

                attn_output = attn_output.reshape(bsz, q_len, -1)
                attn_output = attn_mod.dense(attn_output)
                return attn_output, None, past_key_value
            return fa_forward

        module.forward = make_fa_forward(module)
        count += 1
    return count


# ==============================================================
# Optimization 3: Expert Budgeting (Simple Version)
# ==============================================================
def apply_expert_budgeting(model, S_size=128):
    """Monkey-patch MoE layers: restrict active experts to top-S by popularity.

    Mechanism: between gate.get_logits and experts.forward_impl,
    compute popularity (sigmoid -> sum), take top-S, mask the rest to -inf.
    Zero Python loops, 6 tensor operations per layer.
    """
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeSparseMoeBlock
    count = 0
    originals = []

    for name, module in model.named_modules():
        if not isinstance(module, LLaDA2MoeSparseMoeBlock):
            continue

        orig_forward = module.forward

        def make_budgeted_forward(moe_mod, s_size):
            def budgeted_forward(hidden_states):
                # Shared experts (unchanged)
                res = moe_mod.shared_experts(hidden_states)
                bsz, seq_len, h = hidden_states.shape
                hidden_states_flat = hidden_states.view(-1, h)

                # Gate logits (unchanged)
                router_logits = moe_mod.gate.get_logits(hidden_states_flat)

                # === Expert Budgeting: 6 tensor ops, zero loops ===
                gate_scores = torch.sigmoid(router_logits.float())
                popularity = gate_scores.sum(dim=0)
                _, top_idx = popularity.topk(s_size)
                s_mask = torch.zeros(
                    router_logits.shape[1], dtype=torch.bool,
                    device=router_logits.device)
                s_mask[top_idx] = True
                router_logits = router_logits.masked_fill(~s_mask.unsqueeze(0),
                                                          float('-inf'))
                # === End Expert Budgeting ===

                y = moe_mod.experts.forward_impl(
                    hidden_states=hidden_states_flat,
                    router_logits=router_logits)

                y = y.view(bsz, seq_len, h)
                if moe_mod.config.num_shared_experts is not None:
                    y = y + res
                return y
            return budgeted_forward

        originals.append((module, orig_forward))
        module.forward = make_budgeted_forward(module, S_size)
        count += 1

    return count, originals


def remove_expert_budgeting(originals):
    """Restore original MoE forwards."""
    for module, orig in originals:
        module.forward = orig


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
    print("v0.1.15.5 — Expert Budgeting Inline (Simple Version)")
    print("  Baseline: max_unroll=4 + Fused RMSNorm + Classic flash-attn 2.8.3")
    print("  + Expert Budgeting: popularity top-S, zero Python loops")
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
        n_rms = apply_fused_rmsnorm(model)
        print(f"  Fused RMSNorm: {n_rms} modules")
        n_fa = apply_flash_attn_classic(model)
        print(f"  Flash-attn classic: {n_fa} attention modules")

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

        # ============================================================
        # Warmup (baseline, no EB)
        # ============================================================
        print("\nWarmup...")
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print("  Done.")

        # ============================================================
        # Run 0: Baseline (no EB) timing
        # ============================================================
        print("\n--- Baseline (no Expert Budgeting) ---")
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

        # ============================================================
        # S_size sweep: 128, 112, 96, 80
        # ============================================================
        S_SIZES = [128, 112, 96, 80]
        results = {"baseline": {"fwd": fwd_bl, "time_s": time_bl}}

        for s_size in S_SIZES:
            print(f"\n--- Expert Budgeting S={s_size} ---")
            n_eb, originals = apply_expert_budgeting(model, S_size=s_size)
            print(f"  Patched {n_eb} MoE layers")

            # Warmup with EB
            dllm = make_dllm(decoder_t0)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()

            # Timed run
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
            delta_pct = (time_bl - time_eb) / time_bl * 100
            print(f"  Fwd={fwd_eb}, time={time_eb:.3f}s, fwd/s={fwd_eb/time_eb:.1f}")
            print(f"  vs baseline: {delta_pct:+.1f}% ({'faster' if delta_pct > 0 else 'slower'})")

            results[f"S={s_size}"] = {
                "fwd": fwd_eb, "time_s": time_eb,
                "delta_pct": delta_pct, "s_size": s_size,
            }

            # Remove EB for next iteration
            remove_expert_budgeting(originals)

        # ============================================================
        # Quality check: best S_size config + baseline
        # ============================================================
        print(f"\n{'='*80}")
        print("QUALITY CHECK (temp=0.7)")
        print(f"{'='*80}")

        decoder_t7 = ThresholdParallelDecoder(
            temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        quality_configs = [("baseline", None)] + [(f"S={s}", s) for s in S_SIZES]

        for cname, s_size in quality_configs:
            if s_size is not None:
                _, originals = apply_expert_budgeting(model, S_size=s_size)

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

            print(f"\n  [{cname}]")
            for bi in sorted(VERIFIABLE.keys()):
                gt = gen_tokens[bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                print(f"    #{bi}: {text[:200]}")

            if s_size is not None:
                remove_expert_budgeting(originals)

        # ============================================================
        # Summary
        # ============================================================
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Config':<25s} {'Fwd':>4s} {'Time(s)':>8s} {'Fwd/s':>6s} {'vs BL':>8s}")
        print(f"  {'-'*55}")
        print(f"  {'Baseline (no EB)':<25s} {fwd_bl:>4d} {time_bl:>8.3f} "
              f"{fwd_bl/time_bl:>6.1f} {'—':>8s}")
        for s_size in S_SIZES:
            r = results[f"S={s_size}"]
            print(f"  {f'EB S={s_size}':<25s} {r['fwd']:>4d} {r['time_s']:>8.3f} "
                  f"{r['fwd']/r['time_s']:>6.1f} {r['delta_pct']:>+7.1f}%")

        # Save
        out_path = (REPO_ROOT / "codex_coding" / "results" /
                    "expert_budgeting_inline_simple.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
