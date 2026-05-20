#!/usr/bin/env python3
"""
v0.1.15.4 Phase 3 — nsys Full Inference Profiling with Level 3 NVTX Markers

Instruments the baseline generation pipeline with fine-grained NVTX markers:
  - Per-layer: RMSNorm_pre, Attention, RMSNorm_post, MLP/MoE
  - MoE Level 3: Gate, Shared, Routed (fused_experts)
  - Attention Level 3: QKV, QKNorm+RoPE+Attn, OProj
  - Plus: Embedding, FinalNorm, LMHead, Decoder

Uses cudaProfilerApi to capture only 5 forwards (after warmup).

Usage:
  # nsys profiling:
  nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx \
    --output=baseline_level3 \
    python nsys_full_profiling.py

  # Analysis:
  nsys stats --report nvtx_sum baseline_level3.nsys-rep
"""

from __future__ import annotations
import os, sys, socket, json
from pathlib import Path
from functools import partial

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"

# Same 32 prompts
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


def install_nvtx_hooks(model):
    """Install NVTX markers on all architectural components."""
    import math
    hooks = []
    nvtx = torch.cuda.nvtx

    # ---- Embedding (word_embeddings, not embed_tokens) ----
    def emb_pre(mod, inp):
        nvtx.range_push("Embedding")
    def emb_post(mod, inp, out):
        nvtx.range_pop()
    emb_mod = getattr(model.model, 'word_embeddings', getattr(model.model, 'embed_tokens', None))
    if emb_mod is not None:
        hooks.append(emb_mod.register_forward_pre_hook(emb_pre))
        hooks.append(emb_mod.register_forward_hook(emb_post))

    # ---- Per-layer hooks ----
    for li, layer in enumerate(model.model.layers):
        is_moe = hasattr(layer.mlp, 'gate')
        tag = f"L{li}"

        # Input RMSNorm
        def make_norm_pre(t):
            def f(mod, inp): nvtx.range_push(f"RMSNorm_pre_{t}")
            return f
        def make_norm_post(t):
            def f(mod, inp, out): nvtx.range_pop()
            return f
        hooks.append(layer.input_layernorm.register_forward_pre_hook(make_norm_pre(tag)))
        hooks.append(layer.input_layernorm.register_forward_hook(make_norm_post(tag)))

        # Post-attention RMSNorm
        hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(make_norm_pre(f"post_{tag}")))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(make_norm_post(f"post_{tag}")))

        # ---- Attention Level 3 ----
        attn = layer.attention if hasattr(layer, 'attention') else layer.self_attn

        # QKV projection
        def make_qkv_pre(t):
            def f(mod, inp): nvtx.range_push(f"Attn_QKV_{t}")
            return f
        def make_qkv_post(t):
            def f(mod, inp, out): nvtx.range_pop()
            return f
        hooks.append(attn.query_key_value.register_forward_pre_hook(make_qkv_pre(tag)))
        hooks.append(attn.query_key_value.register_forward_hook(make_qkv_post(tag)))

        # Output projection
        def make_oproj_pre(t):
            def f(mod, inp): nvtx.range_push(f"Attn_OProj_{t}")
            return f
        def make_oproj_post(t):
            def f(mod, inp, out): nvtx.range_pop()
            return f
        hooks.append(attn.dense.register_forward_pre_hook(make_oproj_pre(tag)))
        hooks.append(attn.dense.register_forward_hook(make_oproj_post(tag)))

        # Whole attention (wraps QKV + compute + OProj)
        def make_attn_pre(t):
            def f(mod, inp): nvtx.range_push(f"Attention_{t}")
            return f
        def make_attn_post(t):
            def f(mod, inp, out): nvtx.range_pop()
            return f
        hooks.append(attn.register_forward_pre_hook(make_attn_pre(tag)))
        hooks.append(attn.register_forward_hook(make_attn_post(tag)))

        # ---- MLP / MoE ----
        if is_moe:
            # Monkey-patch MoE forward for Level 3 breakdown
            moe = layer.mlp
            orig_forward = moe.forward.__func__ if hasattr(moe.forward, '__func__') else None
            idx = li

            def make_moe_forward(moe_mod, layer_idx):
                def patched_forward(hidden_states):
                    nvtx.range_push(f"MoE_{layer_idx}")

                    # Shared experts
                    nvtx.range_push(f"MoE_Shared_{layer_idx}")
                    res = moe_mod.shared_experts(hidden_states)
                    nvtx.range_pop()

                    bsz, seq_len, h = hidden_states.shape
                    hidden_states_flat = hidden_states.view(-1, h)

                    # Gate
                    nvtx.range_push(f"MoE_Gate_{layer_idx}")
                    router_logits = moe_mod.gate.get_logits(hidden_states_flat)
                    nvtx.range_pop()

                    # Routed experts (includes routing + fused_experts)
                    nvtx.range_push(f"MoE_Routed_{layer_idx}")
                    y = moe_mod.experts.forward_impl(
                        hidden_states=hidden_states_flat,
                        router_logits=router_logits)
                    nvtx.range_pop()

                    y = y.view(bsz, seq_len, h)
                    if moe_mod.config.num_shared_experts is not None:
                        y = y + res

                    nvtx.range_pop()  # MoE_{layer_idx}
                    return y
                return patched_forward

            moe.forward = make_moe_forward(moe, f"L{li}")
        else:
            # Dense MLP
            def make_mlp_pre(t):
                def f(mod, inp): nvtx.range_push(f"DenseMLP_{t}")
                return f
            def make_mlp_post(t):
                def f(mod, inp, out): nvtx.range_pop()
                return f
            hooks.append(layer.mlp.register_forward_pre_hook(make_mlp_pre(tag)))
            hooks.append(layer.mlp.register_forward_hook(make_mlp_post(tag)))

    # ---- Final RMSNorm ----
    def fn_pre(mod, inp): nvtx.range_push("FinalRMSNorm")
    def fn_post(mod, inp, out): nvtx.range_pop()
    hooks.append(model.model.norm.register_forward_pre_hook(fn_pre))
    hooks.append(model.model.norm.register_forward_hook(fn_post))

    # ---- LM Head ----
    def lm_pre(mod, inp): nvtx.range_push("LMHead")
    def lm_post(mod, inp, out): nvtx.range_pop()
    hooks.append(model.lm_head.register_forward_pre_hook(lm_pre))
    hooks.append(model.lm_head.register_forward_hook(lm_post))

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
    print("nsys Full Inference Profiling — Level 3 NVTX Markers")
    print("  batch=32, gen_length=256, block_length=32, threshold=0.90")
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
        dllm = BlockDiffusionLLM(
            model, decoder,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=4, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # ---- Apply Fused RMSNorm ----
        print("Applying Fused RMSNorm...")
        from vllm.model_executor.layers.layernorm import rms_norm as vllm_rms_norm
        from dinfer.model.modeling_llada2_moe import LLaDA2MoeRMSNorm
        fused_count = 0
        for name, module in model.named_modules():
            if isinstance(module, LLaDA2MoeRMSNorm):
                if "query_layernorm" in name or "key_layernorm" in name:
                    continue
                w, eps = module.weight, module.variance_epsilon
                def _mk(ww, ee):
                    def _f(hs): return vllm_rms_norm(hs, ww, ee)
                    return _f
                module.forward = _mk(w, eps)
                fused_count += 1
        print(f"  Replaced {fused_count} RMSNorm modules")

        # ---- Warmup ----
        print("Warmup...")
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print("Warmup done.")

        # ---- Install NVTX hooks ----
        print("Installing NVTX hooks...")
        nvtx_hooks = install_nvtx_hooks(model)
        print(f"  {len(nvtx_hooks)} Level 2/3 hooks installed.")

        # ---- Install Level 0 NVTX (iteration-level: model vs decoder vs cache) ----
        print("Installing Level 0 NVTX (iteration internals)...")
        from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner
        nvtx = torch.cuda.nvtx

        orig_iter_forward = BlockDiffusionIteration.forward
        def patched_iter_forward(self_iter, model_arg, decoder_arg, x, kv_cache,
                                  block, block_loc, block_id, pos_ids, attn_mask,
                                  past_key_values, replace_position, backend,
                                  is_cross_block=False, block_length=32):
            nvtx.range_push("Iter_CachePrep")
            # Cache prep: clone, inplace update flag
            if kv_cache is not None:
                if self_iter.inplace_cache_update and hasattr(past_key_values, 'use_inplace_update'):
                    past_key_values.use_inplace_update = True
            nvtx.range_pop()

            nvtx.range_push("Iter_ModelForward")
            if kv_cache is None:
                output = model_arg(x.data[:, :block_loc.end],
                    attention_mask=attn_mask[:,:block_loc.end,:block_loc.end],
                    position_ids=pos_ids[:, :block_loc.end])
            else:
                if is_cross_block:
                    output = model_arg(block.clone(memory_format=torch.contiguous_format),
                        position_ids=pos_ids[:,block_loc.start:block_loc.end].clone(memory_format=torch.contiguous_format),
                        use_cache=True,
                        past_key_values=past_key_values,
                        attention_mask=attn_mask,
                        replace_position=(0,0) if backend=='sglang' else replace_position)
                else:
                    output = model_arg(block.clone(memory_format=torch.contiguous_format),
                        position_ids=pos_ids[:,block_loc.start:block_loc.end].clone(memory_format=torch.contiguous_format),
                        use_cache=True,
                        past_key_values=past_key_values,
                        replace_position=(0,0) if backend=='sglang' else replace_position)
            nvtx.range_pop()

            nvtx.range_push("Iter_CacheUpdate")
            logits = output.logits if kv_cache is not None else output.logits[:, block_loc.start:block_loc.end]
            if kv_cache is not None and backend == 'vllm' and not self_iter.lazy_cache_update:
                kv_cache.update(output.past_key_values)
            nvtx.range_pop()

            nvtx.range_push("Iter_DecoderDecode")
            if is_cross_block:
                decoder_arg.decode(logits[:, block_length:] if kv_cache is not None else logits,
                                   block_loc.start+block_length if kv_cache is not None else block_loc.start,
                                   block_loc.end, x)
            else:
                decoder_arg.decode(logits, block_loc.start, block_loc.end, x)
            nvtx.range_pop()

            self_iter.num_forwards += 1
            self_iter.iter_no += 1
            return output

        BlockDiffusionIteration.forward = patched_iter_forward

        orig_runner_decode = BlockDiffusionRunner.decode
        def patched_runner_decode(self_runner, model_arg, decoder_arg, x, kv_cache,
                                   block, block_loc, block_id, pos_ids, attn_mask,
                                   block_length=32, cross_block_attn_mask=None):
            nvtx.range_push("Runner_Decode")

            nvtx.range_push("Runner_SelectUndecoded")
            from dinfer.decoding.generate_uniform import select_undecoded, BlockLoc
            orig_x = x
            seq_idx = torch.arange(x.batch_size, device=block.device)
            seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder_arg.mask_id, writeback=False)
            block = x[:, block_loc.start:block_loc.end]
            batch_size = x.batch_size
            nvtx.range_pop()

            nvtx.range_push("Runner_CacheExtend")
            if kv_cache is not None:
                kv_cache.extend_cache(block_loc.end)
                past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            else:
                past_key_values, replace_position = None, None
            nvtx.range_pop()

            input_block_mask_number = 0
            output = None
            while (block == decoder_arg.mask_id).sum() > 0:
                unroll_k = int(max(min((block == decoder_arg.mask_id).sum()//self_runner.expected_tpf, self_runner.maximum_unroll), 1))
                for unroll_i in range(unroll_k):
                    input_block_mask_number = (block == decoder_arg.mask_id).sum()
                    if self_runner.need_cross_block_update:
                        nvtx.range_push("Runner_CrossBlockUpdate")
                        cross_block_loc = BlockLoc(block_loc.start-block_length, block_loc.end)
                        cross_block_x = x[:, block_loc.start-block_length:block_loc.end]
                        cross_block_replace_positions = (block_loc.start-block_length, block_loc.end)
                        output = self_runner.diff_iteration.forward(model_arg, decoder_arg, x, kv_cache, cross_block_x, cross_block_loc,
                                    block_id, pos_ids, cross_block_attn_mask, past_key_values, cross_block_replace_positions,
                                    self_runner.backend, is_cross_block=True, block_length=block_length)
                        if self_runner.backend=='vllm':
                            if isinstance(output.past_key_values, list):
                                kv_cache.update([past_key_value[:, :, :block_loc.start] for past_key_value in output.past_key_values])
                            else:
                                output.past_key_values.consolidate()
                                kv_cache.update(output.past_key_values._data[:, :, :, :, :block_loc.start])
                            kv_cache.extend_cache(block_loc.end)
                        past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
                        self_runner.need_cross_block_update = False
                        nvtx.range_pop()
                    else:
                        output = self_runner.diff_iteration.forward(model_arg, decoder_arg, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, past_key_values, replace_position, self_runner.backend)
                if batch_size > 1:
                    nvtx.range_push("Runner_SelectUndecoded_Loop")
                    seq_idx, x = select_undecoded(seq_idx, orig_x, x, block, block_loc, decoder_arg.mask_id, writeback=True)
                    block = x[:, block_loc.start:block_loc.end]
                    nvtx.range_pop()
                    if len(seq_idx) == 0:
                        break

            if output is None:
                output = self_runner.diff_iteration.forward(model_arg, decoder_arg, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, past_key_values, replace_position, self_runner.backend)

            nvtx.range_push("Runner_CachePostUpdate")
            if kv_cache is not None:
                self_runner.cache_update_count += 1
                if input_block_mask_number > 0:
                    self_runner.need_cross_block_update = True
                else:
                    self_runner.need_cross_block_update = False
                    self_runner.hidden_cache_update_count += 1
                    if self_runner.backend=='vllm':
                        kv_cache.update(output.past_key_values)
            nvtx.range_pop()

            nvtx.range_push("Runner_EOS")
            eos_idx = torch.any(orig_x[:, block_loc.start:block_loc.end] == decoder_arg.eos_id, dim=1)
            if self_runner.early_stop:
                orig_x[eos_idx, block_loc.end:] = decoder_arg.eos_id
            nvtx.range_pop()

            nvtx.range_pop()  # Runner_Decode
            return eos_idx

        BlockDiffusionRunner.decode = patched_runner_decode
        print("  Level 0 NVTX installed.")

        # ---- Profiled generation ----
        print("\nStarting profiled generation...")
        print("  (cudaProfilerStart → full generation → cudaProfilerStop)")

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()

        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            out = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

        fwd_count = dllm.diff_iteration.num_forwards
        print(f"\n  Profiled generation complete: {fwd_count} forwards")

        # ---- Cleanup ----
        for h in nvtx_hooks:
            h.remove()
        BlockDiffusionIteration.forward = orig_iter_forward
        BlockDiffusionRunner.decode = orig_runner_decode

        print("\nDone. Run nsys stats to analyze:")
        print("  nsys stats --report nvtx_sum baseline_level3.nsys-rep")


if __name__ == "__main__":
    main()
