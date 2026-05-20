#!/usr/bin/env python3
"""
v0.1.14.6 — End-to-End Selective Recompute Verification

Integrates MoE selective recompute into the real generation loop and measures
actual output quality impact. Patches MoE layer forwards to reuse cached
routed outputs for qualifying tokens.

Three policy levels tested progressively:
  Policy 0 (baseline):  All fresh, no reuse
  Policy 1 (token-only): token_margin > threshold → reuse ALL 19 layers
  Policy 2 (token+layer): token_margin > threshold → reuse only middle layers
                          (skip L0-L3 and L15-L18, reuse L4-L14)
  Policy 3 (token+layer tight): reuse only sweet-spot L7-L13

For each policy, runs full multi-block generation and compares:
  - Token-level exact match vs baseline
  - Per-block match rate
  - Total forward count
  - Reuse ratio (how many (layer,token) pairs were reused)
  - Wall-clock time

Runs the same prompts at temperature=0 for deterministic comparison.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")

MASK_ID = 156895
EOS_ID = 156892
NUM_EXPERTS = 256
TOP_K = 8
BLOCK_LENGTH = 32
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
NUM_MOE_LAYERS = 19


# ============================================================
# Selective Recompute Controller
# ============================================================
class SelectiveRecomputeController:
    """Controls per-(layer, token) reuse decisions during generation.

    Maintains a cache of routed outputs from the previous iteration and
    decides which tokens can reuse cached values based on policy config.
    """

    def __init__(self, policy: str, margin_threshold: float = 0.99,
                 reuse_layers: Optional[Set[int]] = None):
        """
        Args:
            policy: "baseline" | "token_only" | "token_layer" | "token_layer_tight"
            margin_threshold: token_margin threshold for Level-1 gate
            reuse_layers: set of MoE layer indices allowed to reuse (None = all)
        """
        self.policy = policy
        self.margin_threshold = margin_threshold
        self.reuse_layers = reuse_layers  # None means all layers
        self.enabled = (policy != "baseline")

        # Cache: routed_output[layer_idx] = tensor [batch, block_len, hidden]
        self.routed_cache: Dict[int, torch.Tensor] = {}
        # Previous step's logits for computing margin
        self.prev_logits: Optional[torch.Tensor] = None
        # Qualifying token mask: [batch, block_len] bool
        self.qualifying_mask: Optional[torch.Tensor] = None

        # Statistics
        self.total_positions = 0  # total (layer, token) positions processed
        self.reused_positions = 0  # positions where cache was used
        self.fresh_positions = 0

    def reset_block(self):
        """Reset state at the start of a new block."""
        self.routed_cache.clear()
        self.prev_logits = None
        self.qualifying_mask = None

    def update_after_forward(self, logits: torch.Tensor):
        """Update qualifying mask from decoder logits (called after each iteration).

        Args:
            logits: [batch, block_len, vocab] from model output
        """
        if not self.enabled:
            return

        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            top2 = probs.topk(2, dim=-1).values  # [batch, block_len, 2]
            margin = top2[:, :, 0] - top2[:, :, 1]  # [batch, block_len]
            self.qualifying_mask = (margin > self.margin_threshold)

        self.prev_logits = logits.detach()

    def cache_routed_output(self, layer_idx: int, routed: torch.Tensor):
        """Cache routed output for use in next iteration."""
        if self.enabled:
            self.routed_cache[layer_idx] = routed.detach().clone()

    def should_reuse(self, layer_idx: int) -> bool:
        """Check if this layer is eligible for reuse."""
        if not self.enabled:
            return False
        if self.qualifying_mask is None:
            return False  # First iteration, no previous data
        if layer_idx not in self.routed_cache:
            return False  # No cached value from previous step
        if self.reuse_layers is not None and layer_idx not in self.reuse_layers:
            return False  # Layer not in allowed set
        return True

    def get_reuse_mask_and_cache(self, layer_idx: int):
        """Get the qualifying mask and cached routed output for this layer.

        Returns:
            (mask [batch, block_len] bool, cached [batch, block_len, hidden])
            or (None, None) if reuse not applicable
        """
        if not self.should_reuse(layer_idx):
            return None, None
        return self.qualifying_mask, self.routed_cache[layer_idx]


# ============================================================
# Hook installer
# ============================================================
def install_selective_recompute_hooks(model, controller: SelectiveRecomputeController):
    """Install MoE hooks that implement selective recompute.

    For each MoE layer:
      1. Run gate + shared expert normally (always fresh)
      2. Run routed experts normally
      3. For qualifying tokens: replace routed output with cached value
      4. Cache current routed output for next iteration

    Returns list of (kind, obj, orig) for cleanup.
    """
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
                hs_flat = hidden_states.view(-1, h)

                # Always fresh: shared expert + gate + routed experts
                shared_res = moe_mod.shared_experts(hidden_states)
                router_logits = moe_mod.gate.get_logits(hs_flat)
                routed_y = moe_mod.experts.forward_impl(
                    hidden_states=hs_flat, router_logits=router_logits)
                routed_y = routed_y.view(bsz, seq_len, h)

                # Track total positions
                ctrl.total_positions += bsz * seq_len

                # Selective reuse: replace qualifying tokens' routed output with cache
                mask, cached = ctrl.get_reuse_mask_and_cache(mi)
                if mask is not None and cached is not None:
                    # Ensure shapes match (batch size might differ due to select_undecoded)
                    if cached.shape[0] >= bsz and cached.shape[1] == seq_len:
                        routed_y = routed_y.clone()
                        # mask shape: [orig_batch, block_len], need to slice to current batch
                        m = mask[:bsz]  # [bsz, seq_len]
                        c = cached[:bsz].to(routed_y.device)
                        n_reused = m.sum().item()
                        if n_reused > 0:
                            routed_y[m] = c[m]
                            ctrl.reused_positions += n_reused
                            ctrl.fresh_positions += bsz * seq_len - n_reused
                        else:
                            ctrl.fresh_positions += bsz * seq_len
                    else:
                        ctrl.fresh_positions += bsz * seq_len
                else:
                    ctrl.fresh_positions += bsz * seq_len

                # Cache current routed output for next iteration
                ctrl.cache_routed_output(mi, routed_y)

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
        if kind == 'moe_forward':
            obj.forward = orig


# ============================================================
# Patched generation with selective recompute
# ============================================================
def generate_with_selective_recompute(dllm, input_ids, controller, gen_length=128):
    """Run generation with selective recompute controller active.

    Patches BlockDiffusionIteration.forward to call controller.update_after_forward
    after each model forward, providing the logits needed for margin calculation.
    Also resets controller state at each new block.
    """
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner

    orig_iter_forward = BlockDiffusionIteration.forward
    orig_runner_decode = BlockDiffusionRunner.decode

    def patched_decode(self_runner, model, decoder, x, kv_cache, block, block_loc,
                       block_id, pos_ids, attn_mask, block_length=32,
                       cross_block_attn_mask=None):
        # Reset controller at start of each block
        controller.reset_block()
        return orig_runner_decode(self_runner, model, decoder, x, kv_cache, block,
                                  block_loc, block_id, pos_ids, attn_mask,
                                  block_length, cross_block_attn_mask)

    def patched_forward(self_iter, model, decoder, x, kv_cache, block, block_loc,
                        block_id, pos_ids, attn_mask, past_key_values,
                        replace_position, backend, is_cross_block=False,
                        block_length=32):
        # Run original forward (with hooked MoE layers)
        output = orig_iter_forward(
            self_iter, model, decoder, x, kv_cache, block, block_loc,
            block_id, pos_ids, attn_mask, past_key_values, replace_position,
            backend, is_cross_block, block_length)

        # Update controller with logits for next iteration's margin computation
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
    # Use temperature=0 for deterministic comparison
    TEMPERATURE = 0.0
    THRESHOLD = 0.90

    POLICIES = {
        "P0_baseline": {
            "policy": "baseline",
            "margin_threshold": 0.99,
            "reuse_layers": None,
            "desc": "All fresh (no reuse)",
        },
        "P1_token_only": {
            "policy": "token_only",
            "margin_threshold": 0.99,
            "reuse_layers": None,  # all 19 layers
            "desc": "token_margin>0.99, all layers",
        },
        "P1b_token_relaxed": {
            "policy": "token_only",
            "margin_threshold": 0.95,
            "reuse_layers": None,
            "desc": "token_margin>0.95, all layers",
        },
        "P2_token_layer_mid": {
            "policy": "token_layer",
            "margin_threshold": 0.99,
            "reuse_layers": set(range(4, 15)),  # L4-L14
            "desc": "token_margin>0.99, reuse L4-L14 only",
        },
        "P3_token_layer_sweet": {
            "policy": "token_layer_tight",
            "margin_threshold": 0.99,
            "reuse_layers": set(range(7, 14)),  # L7-L13 (sweet spot from window sweep)
            "desc": "token_margin>0.99, reuse L7-L13 only",
        },
    }

    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    print("=" * 80)
    print("v0.1.14.6 — End-to-End Selective Recompute Verification")
    print("=" * 80)

    # Init model
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

        # Tokenize
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
        # Run each policy
        # ============================================================
        results = {}

        for policy_name, pcfg_dict in POLICIES.items():
            print(f"\n{'='*60}")
            print(f"Policy: {policy_name} — {pcfg_dict['desc']}")
            print(f"{'='*60}")

            controller = SelectiveRecomputeController(
                policy=pcfg_dict["policy"],
                margin_threshold=pcfg_dict["margin_threshold"],
                reuse_layers=pcfg_dict["reuse_layers"],
            )

            # Install hooks
            hooks = install_selective_recompute_hooks(model, controller)

            try:
                # Warmup (to stabilize CUDA)
                if policy_name == "P0_baseline":
                    controller_warmup = SelectiveRecomputeController(policy="baseline")
                    hooks_warmup = install_selective_recompute_hooks(model, controller_warmup)
                    remove_hooks(hooks_warmup)

                torch.cuda.synchronize()
                t0 = time.time()

                output = generate_with_selective_recompute(
                    dllm, input_ids, controller, gen_length=GEN_LENGTH)

                torch.cuda.synchronize()
                elapsed = time.time() - t0

                num_forwards = dllm.diff_iteration.num_forwards

                # Extract generated tokens (excluding prompt)
                gen_tokens = output[:, prompt_len:].cpu()

                total_pos = controller.total_positions
                reused_pos = controller.reused_positions
                reuse_ratio = reused_pos / max(total_pos, 1)

                print(f"  Time: {elapsed:.3f}s")
                print(f"  Forwards: {num_forwards}")
                print(f"  Output shape: {output.shape}")
                print(f"  Reuse: {reused_pos}/{total_pos} positions "
                      f"({reuse_ratio*100:.1f}%)")

                results[policy_name] = {
                    "desc": pcfg_dict["desc"],
                    "gen_tokens": gen_tokens,
                    "elapsed": elapsed,
                    "num_forwards": num_forwards,
                    "total_positions": total_pos,
                    "reused_positions": reused_pos,
                    "reuse_ratio": reuse_ratio,
                }

            finally:
                remove_hooks(hooks)

        # ============================================================
        # Compare all policies against baseline
        # ============================================================
        print(f"\n{'='*80}")
        print(f"END-TO-END COMPARISON")
        print(f"{'='*80}")

        baseline_tokens = results["P0_baseline"]["gen_tokens"]
        baseline_fwd = results["P0_baseline"]["num_forwards"]
        baseline_time = results["P0_baseline"]["elapsed"]

        print(f"\n  Baseline: {baseline_fwd} forwards, {baseline_time:.3f}s")
        print(f"  Generated tokens shape: {baseline_tokens.shape}")

        # Decode a sample for visual inspection
        print(f"\n  --- Sample output (batch 0, first 100 tokens) ---")
        sample_ids = baseline_tokens[0][:100]
        sample_text = tokenizer.decode(sample_ids, skip_special_tokens=True)
        print(f"  {sample_text[:200]}...")

        print(f"\n{'Policy':<25s} {'Fwd':>5s} {'Time':>7s} {'Speedup':>7s} "
              f"{'Reuse%':>7s} {'ExactM':>7s} {'TokMatch':>8s} {'FwdSave':>7s}")
        print(f"  {'-'*78}")

        for policy_name, r in results.items():
            gen_tok = r["gen_tokens"]
            # Compute match metrics vs baseline
            min_len = min(baseline_tokens.shape[1], gen_tok.shape[1])
            bt = baseline_tokens[:, :min_len]
            gt = gen_tok[:, :min_len]

            # Per-sequence exact match
            seq_match = (bt == gt).all(dim=1)
            exact_match_rate = seq_match.float().mean().item()

            # Per-token match (excluding padding/EOS)
            non_pad = (bt != 0) & (bt != EOS_ID)
            if non_pad.sum() > 0:
                token_match_rate = ((bt == gt) & non_pad).sum().item() / non_pad.sum().item()
            else:
                token_match_rate = 1.0

            speedup = baseline_time / max(r["elapsed"], 1e-6)
            fwd_save = 1.0 - r["num_forwards"] / max(baseline_fwd, 1)

            print(f"  {policy_name:<23s} {r['num_forwards']:>5d} "
                  f"{r['elapsed']:>6.3f}s {speedup:>6.2f}x "
                  f"{r['reuse_ratio']*100:>6.1f}% "
                  f"{exact_match_rate*100:>6.1f}% "
                  f"{token_match_rate*100:>7.2f}% "
                  f"{fwd_save*100:>6.1f}%")

            r["exact_match_rate"] = exact_match_rate
            r["token_match_rate"] = token_match_rate
            r["speedup"] = speedup

        # ============================================================
        # Per-sequence detailed comparison
        # ============================================================
        print(f"\n{'='*80}")
        print(f"PER-SEQUENCE EXACT MATCH DETAIL")
        print(f"{'='*80}")

        for policy_name, r in results.items():
            if policy_name == "P0_baseline":
                continue
            gen_tok = r["gen_tokens"]
            min_len = min(baseline_tokens.shape[1], gen_tok.shape[1])
            bt = baseline_tokens[:, :min_len]
            gt = gen_tok[:, :min_len]

            print(f"\n  {policy_name}:")
            for b in range(bt.shape[0]):
                match = (bt[b] == gt[b]).all().item()
                if not match:
                    # Find first difference
                    diff_pos = (bt[b] != gt[b]).nonzero(as_tuple=True)[0]
                    n_diff = len(diff_pos)
                    first_diff = diff_pos[0].item() if n_diff > 0 else -1
                    tok_match = (bt[b] == gt[b]).sum().item() / min_len * 100
                    print(f"    batch {b}: DIFF — {n_diff}/{min_len} tokens differ "
                          f"(first at pos {first_diff}, {tok_match:.1f}% match)")
                else:
                    print(f"    batch {b}: EXACT MATCH")

        # ============================================================
        # Also run with temperature=0.7 to check stochastic case
        # ============================================================
        print(f"\n{'='*80}")
        print(f"STOCHASTIC COMPARISON (temperature=0.7)")
        print(f"  Running 3 times each to measure consistency")
        print(f"{'='*80}")

        decoder.temperature = 0.7

        stoch_policies = ["P0_baseline", "P1_token_only", "P2_token_layer_mid"]
        stoch_results = defaultdict(list)

        for run_idx in range(3):
            for policy_name in stoch_policies:
                pcfg_dict = POLICIES[policy_name]
                controller = SelectiveRecomputeController(
                    policy=pcfg_dict["policy"],
                    margin_threshold=pcfg_dict["margin_threshold"],
                    reuse_layers=pcfg_dict["reuse_layers"],
                )
                hooks = install_selective_recompute_hooks(model, controller)
                try:
                    torch.cuda.synchronize()
                    t0 = time.time()
                    output = generate_with_selective_recompute(
                        dllm, input_ids, controller, gen_length=GEN_LENGTH)
                    torch.cuda.synchronize()
                    elapsed = time.time() - t0

                    gen_tok = output[:, prompt_len:].cpu()
                    stoch_results[policy_name].append({
                        "gen_tokens": gen_tok,
                        "elapsed": elapsed,
                        "forwards": dllm.diff_iteration.num_forwards,
                        "reuse_ratio": controller.reused_positions / max(controller.total_positions, 1),
                    })
                finally:
                    remove_hooks(hooks)

            print(f"  Run {run_idx+1}/3 complete", flush=True)

        # Compare stochastic runs: how much does reuse change output vs natural randomness?
        print(f"\n  --- Stochastic variability analysis ---")
        print(f"  {'Policy':<25s} {'AvgTime':>7s} {'AvgFwd':>6s} {'Reuse%':>7s} "
              f"{'Self-consistency':>16s}")

        for policy_name in stoch_policies:
            runs = stoch_results[policy_name]
            avg_time = sum(r["elapsed"] for r in runs) / len(runs)
            avg_fwd = sum(r["forwards"] for r in runs) / len(runs)
            avg_reuse = sum(r["reuse_ratio"] for r in runs) / len(runs)

            # Self-consistency: how similar are the 3 runs to each other?
            # Compare run 0 vs run 1, run 0 vs run 2, run 1 vs run 2
            consistencies = []
            for i in range(len(runs)):
                for j in range(i+1, len(runs)):
                    ti = runs[i]["gen_tokens"]
                    tj = runs[j]["gen_tokens"]
                    min_l = min(ti.shape[1], tj.shape[1])
                    match = (ti[:, :min_l] == tj[:, :min_l]).float().mean().item()
                    consistencies.append(match)
            avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 1.0

            print(f"  {policy_name:<25s} {avg_time:>6.3f}s {avg_fwd:>6.0f} "
                  f"{avg_reuse*100:>6.1f}% "
                  f"{avg_consistency*100:>15.1f}%")

        # Cross-policy comparison at t=0.7: how different are reuse outputs from baseline?
        print(f"\n  --- Cross-policy token match (vs baseline runs) ---")
        bl_runs = stoch_results["P0_baseline"]
        for policy_name in stoch_policies:
            if policy_name == "P0_baseline":
                continue
            pol_runs = stoch_results[policy_name]
            cross_matches = []
            for pr in pol_runs:
                for br in bl_runs:
                    min_l = min(pr["gen_tokens"].shape[1], br["gen_tokens"].shape[1])
                    match = (pr["gen_tokens"][:, :min_l] == br["gen_tokens"][:, :min_l]).float().mean().item()
                    cross_matches.append(match)
            avg_cross = sum(cross_matches) / len(cross_matches)
            print(f"  {policy_name} vs baseline: {avg_cross*100:.1f}% token match")

        # Also show baseline self-consistency for reference
        bl_self = []
        for i in range(len(bl_runs)):
            for j in range(i+1, len(bl_runs)):
                min_l = min(bl_runs[i]["gen_tokens"].shape[1], bl_runs[j]["gen_tokens"].shape[1])
                match = (bl_runs[i]["gen_tokens"][:, :min_l] == bl_runs[j]["gen_tokens"][:, :min_l]).float().mean().item()
                bl_self.append(match)
        if bl_self:
            print(f"  Baseline self-consistency: {sum(bl_self)/len(bl_self)*100:.1f}% "
                  f"(natural randomness at t=0.7)")

        print(f"\nDone.")

        # Save summary
        save_data = {}
        for pn, r in results.items():
            save_data[pn] = {k: v for k, v in r.items() if k != "gen_tokens"}
        results_dir = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction"
        save_path = results_dir / "e2e_selective_recompute_results.json"
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
