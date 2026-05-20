#!/usr/bin/env python3
"""
v0.1.14.7b — True-Skip Batch Scaling: batch=8 vs batch=32

Tests whether gather/scatter overhead is amortized at larger batch sizes.
Runs true-skip selective recompute at batch=8 and batch=32, comparing:
  - Actual wall-clock speedup
  - Skip ratio
  - Output quality
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")

MASK_ID = 156895
EOS_ID = 156892
BLOCK_LENGTH = 32
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
NUM_MOE_LAYERS = 19


# ============================================================
# Controller (same as before)
# ============================================================
class TrueSkipController:
    def __init__(self, policy, margin_threshold=0.99,
                 reuse_layers=None, drift_budget=None):
        self.policy = policy
        self.margin_threshold = margin_threshold
        self.reuse_layers = reuse_layers
        self.drift_budget = drift_budget
        self.enabled = (policy != "baseline")
        self.routed_cache = {}
        self.shared_cache = {}
        self.qualifying_mask = None
        self.total_tokens = 0
        self.skipped_tokens = 0
        self.drift_blocked = 0

    def reset_block(self):
        self.routed_cache.clear()
        self.shared_cache.clear()
        self.qualifying_mask = None

    def update_after_forward(self, logits):
        if not self.enabled:
            return
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            top2 = probs.topk(2, dim=-1).values
            margin = top2[:, :, 0] - top2[:, :, 1]
            self.qualifying_mask = (margin > self.margin_threshold)

    def cache_outputs(self, layer_idx, routed, shared):
        if self.enabled:
            self.routed_cache[layer_idx] = routed.detach().clone()
            self.shared_cache[layer_idx] = shared.detach().clone()

    def get_flat_reuse_mask(self, layer_idx, shared_now, bsz, seq_len):
        if not self.enabled or self.qualifying_mask is None:
            return None
        if layer_idx not in self.routed_cache:
            return None
        if self.reuse_layers is not None and layer_idx not in self.reuse_layers:
            return None
        cached = self.routed_cache[layer_idx]
        if cached.shape[0] < bsz or cached.shape[1] != seq_len:
            return None
        mask = self.qualifying_mask[:bsz].clone()
        if self.drift_budget is not None and layer_idx in self.shared_cache:
            sp = self.shared_cache[layer_idx][:bsz].to(shared_now.device).float()
            sn = shared_now[:bsz].float()
            cos_sim = F.cosine_similarity(sn, sp, dim=-1)
            drift_ok = ((1.0 - cos_sim) < self.drift_budget)
            before = mask.sum().item()
            mask = mask & drift_ok
            self.drift_blocked += int(before - mask.sum().item())
        return mask.view(-1)


# ============================================================
# True-skip hooks
# ============================================================
def install_true_skip_hooks(model, controller):
    hooks = []
    layers = model.model.layers
    moe_idx = 0
    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue
        moe_block = layer.mlp
        orig_forward = moe_block.forward
        current_moe_idx = moe_idx

        def make_hook(moe_mod, mi, ctrl):
            def hooked_forward(hidden_states):
                bsz, seq_len, h = hidden_states.shape
                n_total = bsz * seq_len
                hs_flat = hidden_states.view(-1, h)
                shared_res = moe_mod.shared_experts(hidden_states)
                ctrl.total_tokens += n_total
                reuse_mask = ctrl.get_flat_reuse_mask(mi, shared_res, bsz, seq_len)

                if reuse_mask is not None and reuse_mask.any():
                    n_reuse = reuse_mask.sum().item()
                    n_fresh = n_total - n_reuse
                    if n_fresh == 0:
                        routed_flat = ctrl.routed_cache[mi][:bsz].view(-1, h).clone()
                        ctrl.skipped_tokens += n_total
                    else:
                        fresh_indices = (~reuse_mask).nonzero(as_tuple=True)[0]
                        fresh_hs = hs_flat[fresh_indices]
                        fresh_router_logits = moe_mod.gate.get_logits(fresh_hs)
                        fresh_routed = moe_mod.experts.forward_impl(
                            hidden_states=fresh_hs, router_logits=fresh_router_logits)
                        cached_flat = ctrl.routed_cache[mi][:bsz].view(-1, h)
                        routed_flat = cached_flat.clone()
                        routed_flat[fresh_indices] = fresh_routed
                        ctrl.skipped_tokens += n_reuse
                else:
                    router_logits = moe_mod.gate.get_logits(hs_flat)
                    routed_flat = moe_mod.experts.forward_impl(
                        hidden_states=hs_flat, router_logits=router_logits)

                routed_y = routed_flat.view(bsz, seq_len, h)
                ctrl.cache_outputs(mi, routed_y, shared_res)
                out = (routed_y + shared_res
                       if moe_mod.config.num_shared_experts is not None
                       else routed_y)
                return out
            return hooked_forward

        moe_block.forward = make_hook(moe_block, current_moe_idx, controller)
        hooks.append(('moe_forward', moe_block, orig_forward))
        moe_idx += 1
    return hooks


def remove_hooks(hooks):
    for _, obj, orig in hooks:
        obj.forward = orig


def generate_with_controller(dllm, input_ids, controller, gen_length=128):
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner
    orig_iter_forward = BlockDiffusionIteration.forward
    orig_runner_decode = BlockDiffusionRunner.decode

    def patched_decode(self_runner, model, decoder, x, kv_cache, block, block_loc,
                       block_id, pos_ids, attn_mask, block_length=32,
                       cross_block_attn_mask=None):
        controller.reset_block()
        return orig_runner_decode(self_runner, model, decoder, x, kv_cache, block,
                                  block_loc, block_id, pos_ids, attn_mask,
                                  block_length, cross_block_attn_mask)

    def patched_forward(self_iter, model, decoder, x, kv_cache, block, block_loc,
                        block_id, pos_ids, attn_mask, past_key_values,
                        replace_position, backend, is_cross_block=False,
                        block_length=32):
        output = orig_iter_forward(
            self_iter, model, decoder, x, kv_cache, block, block_loc,
            block_id, pos_ids, attn_mask, past_key_values, replace_position,
            backend, is_cross_block, block_length)
        if not is_cross_block:
            controller.update_after_forward(output.logits)
        return output

    BlockDiffusionIteration.forward = patched_forward
    BlockDiffusionRunner.decode = patched_decode
    try:
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            dllm.diff_iteration.iter_no = 0
            output = dllm.generate(input_ids.clone(), gen_length=gen_length,
                                    block_length=BLOCK_LENGTH)
    finally:
        BlockDiffusionIteration.forward = orig_iter_forward
        BlockDiffusionRunner.decode = orig_runner_decode
    return output


# ============================================================
# Main
# ============================================================
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

    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    print("=" * 80)
    print("v0.1.14.7b — True-Skip Batch Scaling: batch=8 vs batch=32")
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

    pcfg_model = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_model)):
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
            backend='vllm',
            lazy_cache_update=True, inplace_cache_update=True,
        )

        POLICIES = [
            ("baseline", {"policy": "baseline", "reuse_layers": None, "drift_budget": None}),
            ("P2_true_skip", {"policy": "token_layer", "reuse_layers": set(range(4, 15)), "drift_budget": None}),
            ("P4_drift_skip", {"policy": "drift", "reuse_layers": set(range(4, 15)), "drift_budget": 0.015}),
        ]

        BATCH_SIZES = [8, 32]
        N_RUNS = 3

        for batch_size in BATCH_SIZES:
            print(f"\n{'='*80}")
            print(f"BATCH SIZE = {batch_size}  (tokens per forward = {batch_size * 32})")
            print(f"{'='*80}")

            # Tokenize
            all_ids = []
            for i in range(batch_size):
                text = PROMPTS[i % len(PROMPTS)]
                if hasattr(tokenizer, "apply_chat_template"):
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        add_generation_prompt=True, tokenize=False)
                ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
                all_ids.append(ids)
            max_len = max(ids.shape[0] for ids in all_ids)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            padded = []
            for ids in all_ids:
                if ids.shape[0] < max_len:
                    pad = torch.full((max_len - ids.shape[0],), pad_id, dtype=ids.dtype)
                    ids = torch.cat([pad, ids])
                padded.append(ids)
            input_ids = torch.stack(padded, dim=0).to(device)
            prompt_len = input_ids.shape[1]
            print(f"  Input: {input_ids.shape}")

            # Warmup
            print("  Warmup...", flush=True)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                dllm.diff_iteration.iter_no = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()

            # Run each policy
            batch_results = {}
            for pname, pcfg in POLICIES:
                times = []
                fwd_counts = []
                skip_pct_last = 0
                drift_blk_last = 0

                for run_idx in range(N_RUNS):
                    ctrl = TrueSkipController(
                        policy=pcfg["policy"], margin_threshold=0.99,
                        reuse_layers=pcfg["reuse_layers"],
                        drift_budget=pcfg["drift_budget"])
                    hooks = install_true_skip_hooks(model, ctrl)
                    try:
                        torch.cuda.synchronize()
                        t0 = time.time()
                        output = generate_with_controller(dllm, input_ids, ctrl, gen_length=GEN_LENGTH)
                        torch.cuda.synchronize()
                        elapsed = time.time() - t0
                        times.append(elapsed)
                        fwd_counts.append(dllm.diff_iteration.num_forwards)
                        if run_idx == 0:
                            skip_pct_last = ctrl.skipped_tokens / max(ctrl.total_tokens, 1) * 100
                            drift_blk_last = ctrl.drift_blocked
                            gen_tokens = output[:, prompt_len:].cpu()
                    finally:
                        remove_hooks(hooks)

                avg_t = sum(times) / len(times)
                min_t = min(times)
                avg_fwd = sum(fwd_counts) / len(fwd_counts)

                batch_results[pname] = {
                    "avg_time": avg_t, "min_time": min_t,
                    "times": times, "avg_fwd": avg_fwd,
                    "skip_pct": skip_pct_last,
                    "drift_blocked": drift_blk_last,
                    "gen_tokens": gen_tokens,
                }

            # Print comparison
            bl = batch_results["baseline"]
            print(f"\n  {'Policy':<18s} {'AvgTime':>7s} {'MinTime':>7s} "
                  f"{'Speedup(avg)':>12s} {'Speedup(min)':>12s} "
                  f"{'Skip%':>6s} {'AvgFwd':>6s}")
            print(f"  {'-'*75}")

            for pname in ["baseline", "P2_true_skip", "P4_drift_skip"]:
                r = batch_results[pname]
                spd_avg = bl["avg_time"] / max(r["avg_time"], 1e-6)
                spd_min = bl["min_time"] / max(r["min_time"], 1e-6)
                print(f"  {pname:<18s} {r['avg_time']:>6.3f}s {r['min_time']:>6.3f}s "
                      f"{spd_avg:>11.3f}x {spd_min:>11.3f}x "
                      f"{r['skip_pct']:>5.1f}% {r['avg_fwd']:>6.0f}")

            # Token match for reuse policies
            bl_tok = batch_results["baseline"]["gen_tokens"]
            for pname in ["P2_true_skip", "P4_drift_skip"]:
                gt = batch_results[pname]["gen_tokens"]
                ml = min(bl_tok.shape[1], gt.shape[1])
                non_pad = (bl_tok[:, :ml] != 0) & (bl_tok[:, :ml] != EOS_ID)
                match = ((bl_tok[:, :ml] == gt[:, :ml]) & non_pad).sum().item() / max(non_pad.sum().item(), 1)
                print(f"  {pname} token match vs baseline: {match*100:.1f}%")

        # ============================================================
        # CROSS-BATCH SUMMARY
        # ============================================================
        print(f"\n{'='*80}")
        print(f"SUMMARY: Does batch scaling help true-skip?")
        print(f"{'='*80}")
        print(f"  At batch=8:  256 tokens/forward → gather/scatter overhead dominates")
        print(f"  At batch=32: 1024 tokens/forward → overhead amortized?")
        print(f"\n  Check the speedup numbers above to confirm.")

        print(f"\nDone.")


if __name__ == "__main__":
    main()
