#!/usr/bin/env python3
"""
v0.1.14.7e — Pure Token-Only Reuse (compute-then-replace)

Simplest possible policy: only token_margin > threshold, ALL 19 layers reuse.
No layer restriction, no drift guard. Uses compute-then-replace (not true-skip).

Tests at batch=8 and batch=32, prints full output for quality check.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MASK_ID = 156895
EOS_ID = 156892
BLOCK_LENGTH = 32
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
DEVICE = "cuda:0"


class SimpleTokenController:
    """Simplest controller: token_margin only, all layers, compute-then-replace."""

    def __init__(self, policy, margin_threshold=0.99):
        self.policy = policy
        self.margin_threshold = margin_threshold
        self.enabled = (policy != "baseline")
        self.routed_cache = {}
        self.qualifying_mask = None
        self.total_positions = 0
        self.reused_positions = 0

    def reset_block(self):
        self.routed_cache.clear()
        self.qualifying_mask = None

    def update_after_forward(self, logits):
        if not self.enabled:
            return
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            top2 = probs.topk(2, dim=-1).values
            margin = top2[:, :, 0] - top2[:, :, 1]
            self.qualifying_mask = (margin > self.margin_threshold)

    def cache_routed(self, layer_idx, routed):
        if self.enabled:
            self.routed_cache[layer_idx] = routed.detach().clone()

    def get_reuse_mask_and_cache(self, layer_idx, bsz, seq_len):
        if not self.enabled or self.qualifying_mask is None:
            return None, None
        if layer_idx not in self.routed_cache:
            return None, None
        cached = self.routed_cache[layer_idx]
        if cached.shape[0] < bsz or cached.shape[1] != seq_len:
            return None, None
        return self.qualifying_mask[:bsz], cached[:bsz]


def install_hooks(model, ctrl):
    hooks = []
    layers = model.model.layers
    moe_idx = 0
    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue
        moe_block = layer.mlp
        orig_forward = moe_block.forward
        mi = moe_idx

        def make_hook(moe_mod, layer_i, c):
            def hooked_forward(hidden_states):
                bsz, seq_len, h = hidden_states.shape
                hs_flat = hidden_states.view(-1, h)
                shared_res = moe_mod.shared_experts(hidden_states)
                router_logits = moe_mod.gate.get_logits(hs_flat)
                routed_y = moe_mod.experts.forward_impl(
                    hidden_states=hs_flat, router_logits=router_logits)
                routed_y = routed_y.view(bsz, seq_len, h)

                c.total_positions += bsz * seq_len

                mask, cached = c.get_reuse_mask_and_cache(layer_i, bsz, seq_len)
                if mask is not None and cached is not None:
                    n = mask.sum().item()
                    if n > 0:
                        routed_y = routed_y.clone()
                        routed_y[mask] = cached[mask].to(routed_y.device)
                        c.reused_positions += n

                c.cache_routed(layer_i, routed_y)
                out = (routed_y + shared_res
                       if moe_mod.config.num_shared_experts is not None
                       else routed_y)
                return out
            return hooked_forward

        moe_block.forward = make_hook(moe_block, mi, ctrl)
        hooks.append(('moe', moe_block, orig_forward))
        moe_idx += 1
    return hooks


def remove_hooks(hooks):
    for _, obj, orig in hooks:
        obj.forward = orig


def generate_with_ctrl(dllm, input_ids, ctrl, gen_length=128):
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner
    orig_if = BlockDiffusionIteration.forward
    orig_rd = BlockDiffusionRunner.decode

    def pd(self_runner, model, decoder, x, kv_cache, block, block_loc,
           block_id, pos_ids, attn_mask, block_length=32,
           cross_block_attn_mask=None):
        ctrl.reset_block()
        return orig_rd(self_runner, model, decoder, x, kv_cache, block, block_loc,
                       block_id, pos_ids, attn_mask, block_length, cross_block_attn_mask)

    def pf(self_iter, model, decoder, x, kv_cache, block, block_loc,
           block_id, pos_ids, attn_mask, past_key_values,
           replace_position, backend, is_cross_block=False, block_length=32):
        out = orig_if(self_iter, model, decoder, x, kv_cache, block, block_loc,
                      block_id, pos_ids, attn_mask, past_key_values,
                      replace_position, backend, is_cross_block, block_length)
        if not is_cross_block:
            ctrl.update_after_forward(out.logits)
        return out

    BlockDiffusionIteration.forward = pf
    BlockDiffusionRunner.decode = pd
    try:
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            dllm.diff_iteration.iter_no = 0
            output = dllm.generate(input_ids.clone(), gen_length=gen_length, block_length=BLOCK_LENGTH)
    finally:
        BlockDiffusionIteration.forward = orig_if
        BlockDiffusionRunner.decode = orig_rd
    return output


def main():
    import socket
    from contextlib import closing
    from transformers import AutoTokenizer, AutoConfig

    PROMPTS = [
        "Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h. The total distance between the two cities is 240 km. What is the average speed for the entire round trip?\n\nProblem 2: A rectangular garden has a perimeter of 56 meters.",
        "Write a detailed essay about the history of artificial intelligence, covering the Dartmouth conference of 1956, the AI winters, the rise of machine learning in the 1990s, and deep learning breakthroughs.",
        "You are a chemistry professor. Explain Le Chatelier's principle with examples and how it applies to industrial ammonia production via the Haber process.",
        "Design a complete REST API for an e-commerce platform with endpoints for user authentication, product management, shopping cart operations, and order processing.",
        "Analyze the global economic impact of climate change across agriculture, energy, real estate, and healthcare sectors with specific examples.",
        "Explain quantum computing to a classical CS background: qubits, superposition, entanglement, Shor's algorithm, and current hardware approaches.",
        "You are a systems architect. Design a distributed message queue with partition-based storage, consumer groups, replication, and exactly-once semantics.",
        "Write a comprehensive guide to training large language models covering data collection, tokenizer training, architecture decisions, and distributed training strategies.",
        "Compare and contrast TCP and UDP protocols, including their use cases in modern distributed systems, gaming, streaming, and IoT applications.",
        "Explain the mathematical foundations of neural networks: backpropagation, gradient descent, loss functions, and the universal approximation theorem.",
        "Design a microservices architecture for a ride-sharing application with real-time matching, pricing, routing, payments, and driver management.",
        "Write about the history of cryptography from Caesar ciphers through RSA, elliptic curve cryptography, and post-quantum cryptographic algorithms.",
        "Explain database indexing strategies: B-trees, hash indexes, bitmap indexes, and their trade-offs for OLTP vs OLAP workloads.",
        "Discuss the economic and social implications of universal basic income with examples from pilot programs in Finland, Kenya, and the United States.",
        "Design a CI/CD pipeline for a large monorepo with microservices, including build caching, parallel testing, canary deployments, and rollback strategies.",
        "Explain the theory of relativity to a physics undergraduate, covering special relativity, time dilation, length contraction, and general relativity basics.",
    ]

    GEN_LENGTH = 128
    TEMPERATURE = 0.0
    THRESHOLD = 0.90
    N_RUNS = 3

    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    print("=" * 80)
    print("v0.1.14.7e — Pure Token-Only Reuse (compute-then-replace)")
    print("=" * 80)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    sys.path.insert(0, str(REPO_ROOT / "lib_cite" / "dInfer" / "python"))
    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (
        BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
        ThresholdParallelDecoder,
    )
    from dinfer.model import LLaDA2MoeModelLM

    port = find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg_m = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_m)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)

        decoder = ThresholdParallelDecoder(
            temperature=TEMPERATURE, threshold=THRESHOLD,
            mask_id=MASK_ID, eos_id=EOS_ID)
        dllm = BlockDiffusionLLM(
            model, decoder,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=1, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True,
        )

        for batch_size in [8, 32]:
            print(f"\n{'='*80}")
            print(f"BATCH SIZE = {batch_size}")
            print(f"{'='*80}")

            all_ids = []
            for i in range(batch_size):
                text = PROMPTS[i % len(PROMPTS)]
                if hasattr(tokenizer, "apply_chat_template"):
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        add_generation_prompt=True, tokenize=False)
                ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
                all_ids.append(ids)
            max_len = max(x.shape[0] for x in all_ids)
            pad_id = tokenizer.pad_token_id or 0
            padded = []
            for ids in all_ids:
                if ids.shape[0] < max_len:
                    ids = torch.cat([torch.full((max_len - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                padded.append(ids)
            input_ids = torch.stack(padded, dim=0).to(device)
            prompt_len = input_ids.shape[1]
            print(f"  Input: {input_ids.shape}")

            # Warmup
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                dllm.diff_iteration.iter_no = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()

            policies = [
                ("baseline", "baseline"),
                ("token_only_0.99", "token_only"),
                ("token_only_0.95", "token_only_095"),
            ]

            results = {}
            for pname, ptype in policies:
                margin = 0.95 if "095" in ptype else 0.99
                times, fwds = [], []
                gen_tokens_last = None
                skip_info = {}

                for run_idx in range(N_RUNS):
                    ctrl = SimpleTokenController(
                        policy="baseline" if ptype == "baseline" else "token_only",
                        margin_threshold=margin)
                    hooks = install_hooks(model, ctrl)
                    try:
                        torch.cuda.synchronize()
                        t0 = time.time()
                        output = generate_with_ctrl(dllm, input_ids, ctrl, gen_length=GEN_LENGTH)
                        torch.cuda.synchronize()
                        elapsed = time.time() - t0
                        times.append(elapsed)
                        fwds.append(dllm.diff_iteration.num_forwards)
                        if run_idx == 0:
                            gen_tokens_last = output[:, prompt_len:].cpu()
                            skip_info = {
                                "reused": ctrl.reused_positions,
                                "total": ctrl.total_positions,
                            }
                    finally:
                        remove_hooks(hooks)

                avg_t = sum(times) / len(times)
                min_t = min(times)
                avg_fwd = sum(fwds) / len(fwds)
                results[pname] = {
                    "avg_time": avg_t, "min_time": min_t,
                    "times": times, "avg_fwd": avg_fwd,
                    "gen_tokens": gen_tokens_last, **skip_info,
                }

            # Table
            bl = results["baseline"]
            print(f"\n  {'Policy':<22s} {'AvgTime':>7s} {'MinTime':>7s} "
                  f"{'Speedup':>7s} {'Reuse%':>7s} {'AvgFwd':>6s}")
            print(f"  {'-'*62}")
            for pname in ["baseline", "token_only_0.99", "token_only_0.95"]:
                r = results[pname]
                spd = bl["avg_time"] / max(r["avg_time"], 1e-6)
                rp = r.get("reused", 0) / max(r.get("total", 1), 1) * 100
                print(f"  {pname:<22s} {r['avg_time']:>6.3f}s {r['min_time']:>6.3f}s "
                      f"{spd:>6.2f}x {rp:>6.1f}% {r['avg_fwd']:>6.0f}")

            # Token match
            bl_tok = results["baseline"]["gen_tokens"]
            for pname in ["token_only_0.99", "token_only_0.95"]:
                gt = results[pname]["gen_tokens"]
                ml = min(bl_tok.shape[1], gt.shape[1])
                non_pad = (bl_tok[:, :ml] != 0) & (bl_tok[:, :ml] != EOS_ID)
                match = ((bl_tok[:, :ml] == gt[:, :ml]) & non_pad).sum().item() / max(non_pad.sum().item(), 1)
                print(f"  {pname} token match: {match*100:.1f}%")

            # Output for first 4 batches
            if batch_size == 8:
                print(f"\n  --- Output comparison (batch=8, first 4 sequences) ---")
                for bi in range(4):
                    print(f"\n  BATCH {bi}: {PROMPTS[bi][:60]}...")
                    for pname in ["baseline", "token_only_0.99"]:
                        gt = results[pname]["gen_tokens"][bi]
                        valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                        text = tokenizer.decode(valid, skip_special_tokens=True)

                        if pname != "baseline":
                            bl_t = bl_tok[bi]
                            ml = min(len(bl_t), len(gt))
                            m = (bl_t[:ml] == gt[:ml]).float().mean().item() * 100
                            label = f"{pname} ({m:.0f}%)"
                        else:
                            label = pname

                        print(f"    [{label}]:")
                        print(f"    {text[:300]}")
                        if len(text) > 300:
                            print(f"    ... ({len(text)} chars)")

        print(f"\nDone.")


if __name__ == "__main__":
    main()
