#!/usr/bin/env python3
"""
v0.1.14.7 — True-Skip Selective Recompute (No Kernel Change)

Instead of "compute all then replace", this version genuinely skips
routed expert computation for reuse tokens by:
  1. index_select only fresh tokens from the flat hidden_states
  2. Pass only fresh tokens to gate.get_logits + experts.forward_impl
  3. Scatter fresh results back, fill reuse positions from cache

This achieves real computational savings without modifying vllm's fused_moe kernel.

Policies tested:
  P0: Baseline (all fresh)
  P2: Token+Layer no guard (margin>0.99, L4-14)
  P4: Token+Layer+Drift (margin>0.99, L4-14, shared_cos drift<0.015)

Prints full output text for quality inspection.
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
# Controller (same as drift-guarded version)
# ============================================================
class TrueSkipController:
    def __init__(self, policy: str, margin_threshold: float = 0.99,
                 reuse_layers: Optional[Set[int]] = None,
                 drift_budget: Optional[float] = None):
        self.policy = policy
        self.margin_threshold = margin_threshold
        self.reuse_layers = reuse_layers
        self.drift_budget = drift_budget
        self.enabled = (policy != "baseline")

        self.routed_cache: Dict[int, torch.Tensor] = {}  # [bsz, seq_len, h]
        self.shared_cache: Dict[int, torch.Tensor] = {}
        self.qualifying_mask: Optional[torch.Tensor] = None

        # Stats
        self.total_tokens = 0
        self.skipped_tokens = 0  # tokens where routed computation was skipped
        self.drift_blocked = 0
        self.gate_skipped = 0  # gate computations saved

    def reset_block(self):
        self.routed_cache.clear()
        self.shared_cache.clear()
        self.qualifying_mask = None

    def update_after_forward(self, logits: torch.Tensor):
        if not self.enabled:
            return
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            top2 = probs.topk(2, dim=-1).values
            margin = top2[:, :, 0] - top2[:, :, 1]
            self.qualifying_mask = (margin > self.margin_threshold)

    def cache_outputs(self, layer_idx: int, routed: torch.Tensor, shared: torch.Tensor):
        if self.enabled:
            self.routed_cache[layer_idx] = routed.detach().clone()
            self.shared_cache[layer_idx] = shared.detach().clone()

    def get_flat_reuse_mask(self, layer_idx: int, shared_now: torch.Tensor,
                            bsz: int, seq_len: int):
        """Returns flat reuse mask [bsz*seq_len] bool — True = skip routed computation."""
        if not self.enabled or self.qualifying_mask is None:
            return None
        if layer_idx not in self.routed_cache:
            return None
        if self.reuse_layers is not None and layer_idx not in self.reuse_layers:
            return None

        cached_routed = self.routed_cache[layer_idx]
        if cached_routed.shape[0] < bsz or cached_routed.shape[1] != seq_len:
            return None

        # Start with token_margin mask
        mask = self.qualifying_mask[:bsz].clone()  # [bsz, seq_len]

        # Drift guard
        if self.drift_budget is not None and layer_idx in self.shared_cache:
            shared_prev = self.shared_cache[layer_idx][:bsz]
            sn = shared_now[:bsz].float()
            sp = shared_prev.to(sn.device).float()
            cos_sim = F.cosine_similarity(sn, sp, dim=-1)
            drift_ok = ((1.0 - cos_sim) < self.drift_budget)
            blocked = mask.sum().item()
            mask = mask & drift_ok
            self.drift_blocked += int(blocked - mask.sum().item())

        return mask.view(-1)  # [bsz*seq_len]


# ============================================================
# True-skip hook installer
# ============================================================
def install_true_skip_hooks(model, controller: TrueSkipController):
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
                hs_flat = hidden_states.view(-1, h)  # [N, h]

                # Shared expert: always fresh
                shared_res = moe_mod.shared_experts(hidden_states)  # [bsz, seq_len, h]

                ctrl.total_tokens += n_total

                # Get reuse mask
                reuse_mask = ctrl.get_flat_reuse_mask(mi, shared_res, bsz, seq_len)

                if reuse_mask is not None and reuse_mask.any():
                    n_reuse = reuse_mask.sum().item()
                    n_fresh = n_total - n_reuse
                    fresh_mask = ~reuse_mask

                    if n_fresh == 0:
                        # All tokens reuse — skip gate + experts entirely
                        routed_flat = ctrl.routed_cache[mi][:bsz].view(-1, h).clone()
                        ctrl.skipped_tokens += n_total
                        ctrl.gate_skipped += n_total
                    else:
                        # === TRUE SKIP: only compute fresh tokens ===
                        fresh_indices = fresh_mask.nonzero(as_tuple=True)[0]  # [n_fresh]
                        fresh_hs = hs_flat[fresh_indices]  # [n_fresh, h]

                        # Gate only on fresh tokens
                        fresh_router_logits = moe_mod.gate.get_logits(fresh_hs)  # [n_fresh, E]

                        # Routed experts only on fresh tokens
                        fresh_routed = moe_mod.experts.forward_impl(
                            hidden_states=fresh_hs,
                            router_logits=fresh_router_logits)  # [n_fresh, h]

                        # Assemble full result
                        cached_flat = ctrl.routed_cache[mi][:bsz].view(-1, h)
                        routed_flat = cached_flat.clone()
                        routed_flat[fresh_indices] = fresh_routed

                        ctrl.skipped_tokens += n_reuse
                        ctrl.gate_skipped += n_reuse
                else:
                    # No reuse — standard full path
                    router_logits = moe_mod.gate.get_logits(hs_flat)
                    routed_flat = moe_mod.experts.forward_impl(
                        hidden_states=hs_flat,
                        router_logits=router_logits)

                routed_y = routed_flat.view(bsz, seq_len, h)

                # Cache for next iteration
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
    for kind, obj, orig in hooks:
        obj.forward = orig


# ============================================================
# Patched generation
# ============================================================
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
    ]

    GEN_LENGTH = 128
    TEMPERATURE = 0.0
    THRESHOLD = 0.90

    POLICIES = [
        ("P0_baseline", {"policy": "baseline", "margin_threshold": 0.99,
                         "reuse_layers": None, "drift_budget": None,
                         "desc": "All fresh (baseline)"}),
        ("P2_no_guard", {"policy": "token_layer", "margin_threshold": 0.99,
                         "reuse_layers": set(range(4, 15)), "drift_budget": None,
                         "desc": "margin>0.99, L4-14, no guard (true skip)"}),
        ("P4_drift_mid", {"policy": "token_layer_drift", "margin_threshold": 0.99,
                          "reuse_layers": set(range(4, 15)), "drift_budget": 0.015,
                          "desc": "margin>0.99, L4-14, drift<0.015 (true skip)"}),
    ]

    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    print("=" * 80)
    print("v0.1.14.7 — True-Skip Selective Recompute")
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

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
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

        BATCH_SIZE = 8
        print(f"\nTokenizing {BATCH_SIZE} prompts...", flush=True)
        all_ids = []
        for i in range(BATCH_SIZE):
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
        print(f"  Input shape: {input_ids.shape}")

        # ============================================================
        # Warmup run (important for fair timing)
        # ============================================================
        print("\nWarmup run...", flush=True)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            dllm.diff_iteration.iter_no = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print("  Warmup done.")

        # ============================================================
        # Run each policy (multiple times for stable timing)
        # ============================================================
        N_RUNS = 3
        results = {}
        all_outputs = {}

        for policy_name, pcfg_dict in POLICIES:
            print(f"\n{'='*60}")
            print(f"{policy_name}: {pcfg_dict['desc']}")
            print(f"{'='*60}")

            times = []
            fwd_counts = []
            gen_tokens_last = None

            for run_idx in range(N_RUNS):
                ctrl = TrueSkipController(
                    policy=pcfg_dict["policy"],
                    margin_threshold=pcfg_dict["margin_threshold"],
                    reuse_layers=pcfg_dict["reuse_layers"],
                    drift_budget=pcfg_dict["drift_budget"],
                )
                hooks = install_true_skip_hooks(model, ctrl)
                try:
                    torch.cuda.synchronize()
                    t0 = time.time()
                    output = generate_with_controller(dllm, input_ids, ctrl, gen_length=GEN_LENGTH)
                    torch.cuda.synchronize()
                    elapsed = time.time() - t0

                    times.append(elapsed)
                    fwd_counts.append(dllm.diff_iteration.num_forwards)
                    gen_tokens_last = output[:, prompt_len:].cpu()

                    if run_idx == 0:
                        # Detailed stats from first run
                        total_tok = ctrl.total_tokens
                        skipped_tok = ctrl.skipped_tokens
                        drift_blk = ctrl.drift_blocked
                        gate_skip = ctrl.gate_skipped
                finally:
                    remove_hooks(hooks)

            avg_time = sum(times) / len(times)
            min_time = min(times)
            avg_fwd = sum(fwd_counts) / len(fwd_counts)

            print(f"  Times: {[f'{t:.3f}s' for t in times]}")
            print(f"  Avg: {avg_time:.3f}s  Min: {min_time:.3f}s  Fwd: {avg_fwd:.0f}")
            if ctrl.enabled:
                skip_pct = skipped_tok / max(total_tok, 1) * 100
                print(f"  Skipped: {skipped_tok}/{total_tok} tokens ({skip_pct:.1f}%)")
                print(f"  Gate skipped: {gate_skip} computations")
                if drift_blk > 0:
                    print(f"  Drift-blocked: {drift_blk}")

            results[policy_name] = {
                "desc": pcfg_dict["desc"],
                "avg_time": avg_time, "min_time": min_time,
                "times": times, "avg_fwd": avg_fwd,
                "skipped_tokens": skipped_tok if ctrl.enabled else 0,
                "total_tokens": total_tok if ctrl.enabled else 0,
                "gen_tokens": gen_tokens_last,
            }
            all_outputs[policy_name] = gen_tokens_last

        # ============================================================
        # COMPARISON
        # ============================================================
        baseline_tokens = all_outputs["P0_baseline"]
        bl_avg = results["P0_baseline"]["avg_time"]
        bl_min = results["P0_baseline"]["min_time"]

        print(f"\n{'='*80}")
        print(f"TRUE-SKIP PERFORMANCE COMPARISON")
        print(f"{'='*80}")
        print(f"  {'Policy':<20s} {'AvgTime':>7s} {'MinTime':>7s} {'Speedup':>7s} "
              f"{'Skip%':>6s} {'AvgFwd':>6s} {'TokMatch':>8s}")
        print(f"  {'-'*68}")

        for pn, r in results.items():
            gt = r["gen_tokens"]
            ml = min(baseline_tokens.shape[1], gt.shape[1])
            bt = baseline_tokens[:, :ml]
            gtt = gt[:, :ml]
            non_pad = (bt != 0) & (bt != EOS_ID)
            tok_match = ((bt == gtt) & non_pad).sum().item() / max(non_pad.sum().item(), 1)

            spd_avg = bl_avg / max(r["avg_time"], 1e-6)
            spd_min = bl_min / max(r["min_time"], 1e-6)
            skip_pct = r["skipped_tokens"] / max(r["total_tokens"], 1) * 100

            print(f"  {pn:<20s} {r['avg_time']:>6.3f}s {r['min_time']:>6.3f}s "
                  f"{spd_avg:>6.2f}x {skip_pct:>5.1f}% {r['avg_fwd']:>6.0f} "
                  f"{tok_match*100:>7.1f}%")

        # ============================================================
        # FULL OUTPUT COMPARISON
        # ============================================================
        print(f"\n{'='*80}")
        print(f"OUTPUT QUALITY COMPARISON")
        print(f"{'='*80}")

        for batch_idx in range(min(BATCH_SIZE, 4)):  # Show first 4 for brevity
            print(f"\n{'─'*80}")
            print(f"  BATCH {batch_idx}: {PROMPTS[batch_idx][:70]}...")
            print(f"{'─'*80}")

            for pn in ["P0_baseline", "P2_no_guard", "P4_drift_mid"]:
                gt = all_outputs[pn][batch_idx]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)

                bl = baseline_tokens[batch_idx]
                ml = min(len(bl), len(gt))
                match_rate = (bl[:ml] == gt[:ml]).float().mean().item() * 100

                print(f"\n  [{pn} ({match_rate:.0f}% match)]:")
                print(f"  {text[:400]}")
                if len(text) > 400:
                    print(f"  ... ({len(text)} chars)")

        print(f"\n{'='*80}")
        print(f"CONCLUSION")
        print(f"{'='*80}")

        p2 = results["P2_no_guard"]
        p4 = results["P4_drift_mid"]
        bl = results["P0_baseline"]
        print(f"  Baseline:     {bl['avg_time']:.3f}s avg ({bl['avg_fwd']:.0f} forwards)")
        print(f"  P2 true-skip: {p2['avg_time']:.3f}s avg → {bl['avg_time']/p2['avg_time']:.2f}x speedup")
        print(f"  P4 true-skip: {p4['avg_time']:.3f}s avg → {bl['avg_time']/p4['avg_time']:.2f}x speedup")
        print(f"  Token skip rate: P2={p2['skipped_tokens']/max(p2['total_tokens'],1)*100:.1f}%, "
              f"P4={p4['skipped_tokens']/max(p4['total_tokens'],1)*100:.1f}%")

        print(f"\nDone.")

        save_data = {}
        for pn, r in results.items():
            save_data[pn] = {k: v for k, v in r.items()
                             if k not in ("gen_tokens",)}
        save_path = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction" / "true_skip_results.json"
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
