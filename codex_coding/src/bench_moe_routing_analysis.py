#!/usr/bin/env python3
"""
MoE routing & output profiling for dLLM Insight validation.

Collects per-position × per-iteration × per-layer data for one generation block:
  - Expert routing decisions (topk_idx from gate)
  - MoE block input/output hidden states
  - MASK vs decoded position tracking

Then performs offline analysis:
  A1: MASK vs decoded routing entropy (per layer)
  A2: Expert load balance (per layer, per iteration)
  B1: Cross-iteration routing stability (decoded positions)
  B2: Cross-iteration MoE output cosine similarity
  C1: Redundancy quantification
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
RESULTS_DIR = REPO_ROOT / "codex_coding" / "results"

MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
GEN_LENGTH = 128
BLOCK_LENGTH = 32
THRESHOLD = 0.90
MASK_ID = 156895
EOS_ID = 156892
DEVICE = "cuda:0"
NUM_EXPERTS = 256
TOP_K = 8
NUM_MOE_LAYERS = 19  # layers 1-19

LONG_PROMPT = """Please solve the following problems step by step.

Problem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h. The total distance between the two cities is 240 km. What is the average speed for the entire round trip?

Problem 2: A rectangular garden has a perimeter of 56 meters. If the length is 4 meters more than twice the width, find the dimensions of the garden.

Problem 3: In a class of 40 students, 25 study Mathematics, 20 study Physics, and 10 study both subjects. How many students study neither Mathematics nor Physics?

Problem 4: A cone has a radius of 7 cm and a slant height of 25 cm. Calculate the total surface area and the volume of the cone.

Problem 5: A bank offers compound interest at 8% per annum, compounded quarterly. If you deposit $5000, how much will you have after 3 years?

Problem 6: Two pipes can fill a tank. Pipe A fills the tank in 12 hours and Pipe B fills it in 18 hours. If both pipes are opened together, but Pipe B is closed after 4 hours, how long will it take Pipe A alone to fill the remaining tank?

Problem 7: A sequence is defined as follows: a(1) = 2, a(2) = 5, and for n >= 3, a(n) = 2*a(n-1) - a(n-2) + 3. Find the first 8 terms.

Problem 8: A factory produces widgets on two assembly lines. Line A produces 300 widgets per hour with a defect rate of 2%. Line B produces 200 widgets per hour with a defect rate of 1.5%. If the factory runs both lines for 8 hours, what is the overall defect rate?

Problem 9: A cylindrical water tank with radius 3 meters and height 10 meters is being filled at a rate of 2 cubic meters per minute while being drained at 0.5 cubic meters per minute. How long will it take to fill completely?"""

# Target block index to profile (0-indexed among generation blocks, skip block 0)
TARGET_BLOCK_IDX = 1
BATCH_SIZES = [1, 8, 32]


# ============================================================
# Data collector
# ============================================================
class MoEDataCollector:
    """Collects routing and hidden state data during MoE forwards."""

    def __init__(self, num_moe_layers, block_length, num_experts, top_k):
        self.num_moe_layers = num_moe_layers
        self.block_length = block_length
        self.num_experts = num_experts
        self.top_k = top_k

        # State tracking
        self.active = False           # only collect when True
        self.recording = False        # True only during a regular (non-cross-block) forward
        self.current_iter = -1        # current diffusion iteration within the block
        self.layer_counter = 0        # which MoE layer within current forward pass
        self.block_start = 0          # start position of the block in the full sequence
        self.block_end = 0            # end position of the block in the full sequence
        self.batch_size = 1

        # Collected data (per iteration)
        # routing_data[iter][layer] = topk_idx tensor [batch, block_len, top_k]
        self.routing_data = defaultdict(dict)
        # topk_weight_data[iter][layer] = topk_weight tensor [batch, block_len, top_k]
        self.weight_data = defaultdict(dict)
        # moe_input[iter][layer] = hidden_states [batch, block_len, hidden]
        self.moe_input = defaultdict(dict)
        # moe_output[iter][layer] = output [batch, block_len, hidden]
        self.moe_output = defaultdict(dict)
        # mask_state[iter] = bool tensor [batch, block_len] (True = MASK position)
        self.mask_state = {}

    def start_iteration(self, iter_idx, mask_positions, block_start, block_end, batch_size):
        """Called at the start of each diffusion iteration."""
        self.current_iter = iter_idx
        self.layer_counter = 0
        self.block_start = block_start
        self.block_end = block_end
        self.batch_size = batch_size
        self.recording = True
        self.mask_state[iter_idx] = mask_positions.clone().cpu()

    def end_iteration(self):
        """Called at the end of each diffusion iteration."""
        self.layer_counter = 0
        self.recording = False

    def record_routing(self, topk_idx, topk_weight):
        """Called by gate hook: record routing decision for block portion only.
        topk_idx/topk_weight arrive as [bsz*seq, top_k] from gate.
        In cache path, seq=block_length (32). In no-cache path, seq=full_seq."""
        if not self.active or not self.recording:
            return
        layer = self.layer_counter
        full_seq = topk_idx.shape[0] // self.batch_size
        topk_idx_3d = topk_idx.view(self.batch_size, full_seq, self.top_k)
        topk_weight_3d = topk_weight.view(self.batch_size, full_seq, self.top_k)
        block_len = self.block_end - self.block_start
        if full_seq <= block_len:
            # Cache path: MoE input IS the block (or smaller), keep as-is
            self.routing_data[self.current_iter][layer] = topk_idx_3d[:, -block_len:, :].detach().cpu()
            self.weight_data[self.current_iter][layer] = topk_weight_3d[:, -block_len:, :].detach().cpu()
        else:
            # No-cache path: MoE processes full sequence, slice to block portion
            self.routing_data[self.current_iter][layer] = topk_idx_3d[:, self.block_start:self.block_end, :].detach().cpu()
            self.weight_data[self.current_iter][layer] = topk_weight_3d[:, self.block_start:self.block_end, :].detach().cpu()

    def record_moe_io(self, input_hs, output_hs):
        """Called by MoE block hook: record output for block portion only.
        Only stores batch[0] for output similarity to save memory at large batch."""
        if not self.active or not self.recording:
            return
        layer = self.layer_counter
        full_seq = input_hs.shape[1]
        block_len = self.block_end - self.block_start
        if full_seq <= block_len:
            self.moe_output[self.current_iter][layer] = output_hs[0:1, -block_len:, :].detach().cpu()
        else:
            self.moe_output[self.current_iter][layer] = output_hs[0:1, self.block_start:self.block_end, :].detach().cpu()
        self.layer_counter += 1

    def get_num_iterations(self):
        return len(self.mask_state)


# ============================================================
# Hook installation
# ============================================================
def install_moe_hooks(model, collector):
    """Install forward hooks on gate.routing and MoE block."""
    hooks = []

    # Find all MoE blocks (layer 1-19)
    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue  # skip dense layer (layer 0)

        gate = layer.mlp.gate
        moe_block = layer.mlp

        # Hook on gate.routing to capture topk_idx
        orig_routing = gate.routing

        def make_routing_hook(orig_fn):
            def hooked_routing(self_gate, hidden_states, gating_output, topk, renormalize):
                topk_weight, topk_idx = orig_fn(hidden_states, gating_output, topk, renormalize)
                collector.record_routing(topk_idx, topk_weight)
                return topk_weight, topk_idx
            return hooked_routing

        gate.routing = make_routing_hook(orig_routing).__get__(gate)
        hooks.append(('routing', gate, orig_routing))

        # Hook on MoE block forward to capture input/output
        def make_moe_hook():
            def hook_fn(module, input_tuple, output):
                inp = input_tuple[0]  # hidden_states
                out = output if isinstance(output, torch.Tensor) else output[0]
                collector.record_moe_io(inp, out)
            return hook_fn

        h = moe_block.register_forward_hook(make_moe_hook())
        hooks.append(('moe_hook', h, None))

    return hooks


def remove_hooks(hooks):
    """Remove installed hooks."""
    for kind, obj, orig in hooks:
        if kind == 'routing':
            # restore original routing method
            obj.routing = orig
        elif kind == 'moe_hook':
            obj.remove()


# ============================================================
# Instrumented generate: monkey-patch to track iterations
# ============================================================
def run_instrumented_generate(dllm, input_ids, collector, target_block_idx):
    """Run generate with iteration-level tracking on the target block."""
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration

    orig_forward = BlockDiffusionIteration.forward
    current_block_idx = [0]  # mutable counter for block transitions
    iteration_in_block = [0]

    # We also need to track block transitions.
    # The generate loop calls block_runner.decode() per block.
    # Inside decode, it calls diff_iteration.forward() per iteration.
    # We track block index by counting how many blocks have started.

    from dinfer.decoding.generate_uniform import BlockDiffusionRunner
    orig_decode = BlockDiffusionRunner.decode

    def patched_decode(self_runner, model, decoder, x, kv_cache, block, block_loc,
                       block_id, pos_ids, attn_mask, block_length=32, cross_block_attn_mask=None):
        gen_block_idx = current_block_idx[0]
        is_target = (gen_block_idx == target_block_idx)

        if is_target:
            collector.active = True
            iteration_in_block[0] = 0
            print(f"  [Collector] Activated for gen block {gen_block_idx} "
                  f"(block_loc={block_loc.start}:{block_loc.end})", flush=True)

        result = orig_decode(self_runner, model, decoder, x, kv_cache, block,
                             block_loc, block_id, pos_ids, attn_mask, block_length,
                             cross_block_attn_mask)

        if is_target:
            collector.active = False
            print(f"  [Collector] Deactivated. Collected {collector.get_num_iterations()} iterations.",
                  flush=True)

        current_block_idx[0] += 1
        return result

    def patched_forward(self_iter, model, decoder, x, kv_cache, block, block_loc,
                        block_id, pos_ids, attn_mask, past_key_values, replace_position,
                        backend, is_cross_block=False, block_length=32):
        gen_block_idx = current_block_idx[0]
        is_target = (gen_block_idx == target_block_idx)

        if is_target and collector.active and not is_cross_block:
            # Record mask state for this iteration (block portion only)
            block_tokens = x.data[:, block_loc.start:block_loc.end]
            mask_positions = (block_tokens == decoder.mask_id)
            collector.start_iteration(iteration_in_block[0], mask_positions,
                                      block_loc.start, block_loc.end, x.data.shape[0])

        result = orig_forward(self_iter, model, decoder, x, kv_cache, block,
                              block_loc, block_id, pos_ids, attn_mask,
                              past_key_values, replace_position, backend,
                              is_cross_block, block_length)

        if is_target and collector.active and not is_cross_block:
            collector.end_iteration()
            iteration_in_block[0] += 1

        return result

    # Apply patches
    BlockDiffusionIteration.forward = patched_forward
    BlockDiffusionRunner.decode = patched_decode

    try:
        with torch.inference_mode():
            out = dllm.generate(input_ids, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
    finally:
        # Restore
        BlockDiffusionIteration.forward = orig_forward
        BlockDiffusionRunner.decode = orig_decode

    return out


# ============================================================
# Offline Analysis
# ============================================================
def compute_entropy(counts, base=2):
    """Compute entropy of a distribution given counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts.float() / total
    probs = probs[probs > 0]
    return -(probs * probs.log2()).sum().item()


def analyze_routing(collector, batch_size=1):
    """Perform full offline analysis of collected data."""
    num_iters = collector.get_num_iterations()
    if num_iters == 0:
        print("  No data collected!")
        return {}

    results = {}

    print(f"\n{'='*100}")
    print(f"  MoE ROUTING ANALYSIS — {num_iters} iterations collected")
    print(f"{'='*100}")

    # ============================
    # A1: MASK vs Decoded routing entropy (per layer)
    # ============================
    print(f"\n--- A1: MASK vs Decoded Routing Entropy (per layer, averaged over iterations) ---")
    print(f"  {'Layer':<8s} {'MASK_ent':>10s} {'Dec_ent':>10s} {'Ratio':>8s} "
          f"{'MASK_active':>14s} {'Dec_active':>14s}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*14} {'-'*14}")

    a1_data = {}
    for layer in sorted(collector.routing_data[0].keys()):
        mask_entropies = []
        dec_entropies = []
        mask_active_counts = []
        dec_active_counts = []

        for it in range(num_iters):
            if layer not in collector.routing_data[it]:
                continue
            topk_idx = collector.routing_data[it][layer]  # [batch, seq, top_k]
            mask = collector.mask_state[it]               # [batch, seq]

            # Flatten batch dim for simplicity (batch=1)
            topk_flat = topk_idx[0]   # [seq, top_k]
            mask_flat = mask[0]       # [seq]

            # MASK positions
            mask_pos = topk_flat[mask_flat]  # [n_mask, top_k]
            if mask_pos.numel() > 0:
                expert_counts = torch.zeros(NUM_EXPERTS)
                for e in mask_pos.flatten():
                    expert_counts[e.item()] += 1
                mask_entropies.append(compute_entropy(expert_counts))
                mask_active_counts.append((expert_counts > 0).sum().item())

            # Decoded positions
            dec_pos = topk_flat[~mask_flat]  # [n_dec, top_k]
            if dec_pos.numel() > 0:
                expert_counts = torch.zeros(NUM_EXPERTS)
                for e in dec_pos.flatten():
                    expert_counts[e.item()] += 1
                dec_entropies.append(compute_entropy(expert_counts))
                dec_active_counts.append((expert_counts > 0).sum().item())

        avg_mask_ent = sum(mask_entropies) / len(mask_entropies) if mask_entropies else 0
        avg_dec_ent = sum(dec_entropies) / len(dec_entropies) if dec_entropies else 0
        ratio = avg_mask_ent / avg_dec_ent if avg_dec_ent > 0 else 0
        avg_mask_active = sum(mask_active_counts) / len(mask_active_counts) if mask_active_counts else 0
        avg_dec_active = sum(dec_active_counts) / len(dec_active_counts) if dec_active_counts else 0

        print(f"  {layer:<8d} {avg_mask_ent:>10.3f} {avg_dec_ent:>10.3f} {ratio:>8.3f} "
              f"{avg_mask_active:>10.1f}/256 {avg_dec_active:>10.1f}/256")
        a1_data[layer] = {
            "mask_entropy": avg_mask_ent, "dec_entropy": avg_dec_ent,
            "ratio": ratio, "mask_active": avg_mask_active, "dec_active": avg_dec_active,
        }
    results["A1_entropy"] = a1_data

    # ============================
    # A2: Expert load balance (per layer, first vs last iteration)
    # ============================
    print(f"\n--- A2: Expert Load Balance (first iter vs last iter, all batch elements) ---")
    print(f"  {'Layer':<8s} {'First_max/avg':>14s} {'First_gini':>12s} "
          f"{'Last_max/avg':>14s} {'Last_gini':>12s} {'Tokens/fwd':>11s}")
    print(f"  {'-'*8} {'-'*14} {'-'*12} {'-'*14} {'-'*12} {'-'*11}")

    a2_data = {}
    for layer in sorted(collector.routing_data[0].keys()):
        layer_data = {}
        for label, it in [("first", 0), ("last", num_iters - 1)]:
            if layer not in collector.routing_data[it]:
                continue
            topk_idx = collector.routing_data[it][layer]  # [batch, seq, top_k]
            # Use ALL batch elements for load balance analysis
            expert_counts = torch.zeros(NUM_EXPERTS)
            for e in topk_idx.flatten():
                expert_counts[e.item()] += 1
            total_tokens = topk_idx.shape[0] * topk_idx.shape[1]  # batch * seq
            avg_load = expert_counts.mean().item()
            max_load = expert_counts.max().item()
            ratio = max_load / avg_load if avg_load > 0 else 0
            # Gini coefficient
            sorted_counts = expert_counts.sort()[0]
            n = len(sorted_counts)
            cumsum = sorted_counts.cumsum(0)
            gini = (2.0 * (torch.arange(1, n+1).float() * sorted_counts).sum() / (n * sorted_counts.sum()) - (n+1)/n).item() if sorted_counts.sum() > 0 else 0
            layer_data[label] = {"max_avg_ratio": ratio, "gini": gini, "total_tokens": total_tokens}

        if "first" in layer_data and "last" in layer_data:
            tok_per_fwd = layer_data["first"].get("total_tokens", "?")
            print(f"  {layer:<8d} {layer_data['first']['max_avg_ratio']:>14.2f} "
                  f"{layer_data['first']['gini']:>12.4f} "
                  f"{layer_data['last']['max_avg_ratio']:>14.2f} "
                  f"{layer_data['last']['gini']:>12.4f} {tok_per_fwd:>11}")
        a2_data[layer] = layer_data
    results["A2_load_balance"] = a2_data

    # ============================
    # B1: Cross-iteration routing stability (decoded positions)
    # ============================
    print(f"\n--- B1: Cross-Iteration Routing Stability ---")
    print(f"  {'Layer':<8s} {'Dec_change%':>12s} {'Mask_change%':>13s} "
          f"{'Dec_pairs':>10s} {'Mask_pairs':>11s}")
    print(f"  {'-'*8} {'-'*12} {'-'*13} {'-'*10} {'-'*11}")

    b1_data = {}
    for layer in sorted(collector.routing_data[0].keys()):
        dec_changes = 0
        dec_total = 0
        mask_changes = 0
        mask_total = 0

        for it in range(1, num_iters):
            if layer not in collector.routing_data[it] or layer not in collector.routing_data[it-1]:
                continue
            curr_routing = collector.routing_data[it][layer][0]    # [seq, top_k]
            prev_routing = collector.routing_data[it-1][layer][0]  # [seq, top_k]
            curr_mask = collector.mask_state[it][0]                # [seq]
            prev_mask = collector.mask_state[it-1][0]              # [seq]

            # Decoded positions: positions that were decoded in BOTH iterations
            decoded_both = ~curr_mask & ~prev_mask
            if decoded_both.any():
                curr_set = curr_routing[decoded_both]  # [n, top_k]
                prev_set = prev_routing[decoded_both]  # [n, top_k]
                # Compare sorted expert sets (order may differ)
                curr_sorted = curr_set.sort(dim=-1)[0]
                prev_sorted = prev_set.sort(dim=-1)[0]
                changed = (curr_sorted != prev_sorted).any(dim=-1)
                dec_changes += changed.sum().item()
                dec_total += decoded_both.sum().item()

            # MASK positions: positions that were MASK in BOTH iterations
            mask_both = curr_mask & prev_mask
            if mask_both.any():
                curr_set = curr_routing[mask_both]
                prev_set = prev_routing[mask_both]
                curr_sorted = curr_set.sort(dim=-1)[0]
                prev_sorted = prev_set.sort(dim=-1)[0]
                changed = (curr_sorted != prev_sorted).any(dim=-1)
                mask_changes += changed.sum().item()
                mask_total += mask_both.sum().item()

        dec_pct = dec_changes / dec_total * 100 if dec_total > 0 else 0
        mask_pct = mask_changes / mask_total * 100 if mask_total > 0 else 0
        print(f"  {layer:<8d} {dec_pct:>11.2f}% {mask_pct:>12.2f}% "
              f"{dec_total:>10d} {mask_total:>11d}")
        b1_data[layer] = {
            "dec_change_pct": dec_pct, "mask_change_pct": mask_pct,
            "dec_pairs": dec_total, "mask_pairs": mask_total,
        }
    results["B1_routing_stability"] = b1_data

    # ============================
    # B2: Cross-iteration MoE output cosine similarity
    # ============================
    print(f"\n--- B2: Cross-Iteration MoE Output Cosine Similarity ---")
    print(f"  {'Layer':<8s} {'Dec_cos_sim':>12s} {'Mask_cos_sim':>13s} "
          f"{'Dec_pairs':>10s} {'Mask_pairs':>11s}")
    print(f"  {'-'*8} {'-'*12} {'-'*13} {'-'*10} {'-'*11}")

    b2_data = {}
    for layer in sorted(collector.moe_output.get(0, {}).keys()):
        dec_sims = []
        mask_sims = []

        for it in range(1, num_iters):
            if layer not in collector.moe_output.get(it, {}) or layer not in collector.moe_output.get(it-1, {}):
                continue
            curr_out = collector.moe_output[it][layer][0].float()    # [seq, hidden]
            prev_out = collector.moe_output[it-1][layer][0].float()  # [seq, hidden]
            curr_mask = collector.mask_state[it][0]
            prev_mask = collector.mask_state[it-1][0]

            # Decoded in both
            decoded_both = ~curr_mask & ~prev_mask
            if decoded_both.any():
                c = curr_out[decoded_both]
                p = prev_out[decoded_both]
                cos = F.cosine_similarity(c, p, dim=-1)
                dec_sims.extend(cos.tolist())

            # MASK in both
            mask_both = curr_mask & prev_mask
            if mask_both.any():
                c = curr_out[mask_both]
                p = prev_out[mask_both]
                cos = F.cosine_similarity(c, p, dim=-1)
                mask_sims.extend(cos.tolist())

        avg_dec = sum(dec_sims) / len(dec_sims) if dec_sims else 0
        avg_mask = sum(mask_sims) / len(mask_sims) if mask_sims else 0
        print(f"  {layer:<8d} {avg_dec:>12.6f} {avg_mask:>13.6f} "
              f"{len(dec_sims):>10d} {len(mask_sims):>11d}")
        b2_data[layer] = {
            "dec_cos_sim": avg_dec, "mask_cos_sim": avg_mask,
            "dec_pairs": len(dec_sims), "mask_pairs": len(mask_sims),
        }
    results["B2_output_similarity"] = b2_data

    # ============================
    # C1: Redundancy quantification
    # ============================
    print(f"\n--- C1: Redundancy Quantification ---")
    total_computations = 0
    redundant_routing = 0
    for it in range(1, num_iters):
        for layer in collector.routing_data[it]:
            if layer not in collector.routing_data[it-1]:
                continue
            curr = collector.routing_data[it][layer][0]     # [seq, top_k]
            prev = collector.routing_data[it-1][layer][0]
            curr_sorted = curr.sort(dim=-1)[0]
            prev_sorted = prev.sort(dim=-1)[0]
            same = (curr_sorted == prev_sorted).all(dim=-1)  # [seq]
            total_computations += curr.shape[0]
            redundant_routing += same.sum().item()

    # First iteration has no "previous" → all are unique
    for layer in collector.routing_data.get(0, {}):
        total_computations += collector.routing_data[0][layer].shape[1]

    pct = redundant_routing / total_computations * 100 if total_computations > 0 else 0
    print(f"  Total token-expert-layer computations: {total_computations}")
    print(f"  Redundant (same routing as prev iter): {redundant_routing} ({pct:.1f}%)")
    print(f"  Unique computations:                   {total_computations - redundant_routing}")

    # Per-iteration breakdown
    print(f"\n  Per-iteration breakdown:")
    print(f"  {'Iter':<6s} {'#MASK':>6s} {'#Dec':>6s} {'Redundant%':>11s}")
    print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*11}")
    c1_per_iter = []
    for it in range(num_iters):
        n_mask = collector.mask_state[it][0].sum().item()
        n_dec = BLOCK_LENGTH - n_mask
        if it == 0:
            print(f"  {it:<6d} {int(n_mask):>6d} {int(n_dec):>6d} {'(baseline)':>11s}")
            c1_per_iter.append({"iter": it, "n_mask": int(n_mask), "n_dec": int(n_dec), "redundant_pct": 0})
        else:
            iter_total = 0
            iter_redundant = 0
            for layer in collector.routing_data[it]:
                if layer not in collector.routing_data[it-1]:
                    continue
                curr = collector.routing_data[it][layer][0].sort(dim=-1)[0]
                prev = collector.routing_data[it-1][layer][0].sort(dim=-1)[0]
                same = (curr == prev).all(dim=-1)
                iter_total += curr.shape[0]
                iter_redundant += same.sum().item()
            ipct = iter_redundant / iter_total * 100 if iter_total > 0 else 0
            print(f"  {it:<6d} {int(n_mask):>6d} {int(n_dec):>6d} {ipct:>10.1f}%")
            c1_per_iter.append({"iter": it, "n_mask": int(n_mask), "n_dec": int(n_dec), "redundant_pct": ipct})

    results["C1_redundancy"] = {
        "total_computations": total_computations,
        "redundant_routing": redundant_routing,
        "redundant_pct": pct,
        "per_iteration": c1_per_iter,
    }

    return results


# ============================================================
# Main
# ============================================================
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (
        BlockDiffusionLLM,
        BlockDiffusionLLMAttnmask,
        BlockIteratorFactory,
        KVCacheFactory,
        ThresholdParallelDecoder,
    )
    from dinfer.model import LLaDA2MoeModelLM

    port = find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("Loading model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    cfg = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=cfg).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        # Warmup
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)

        prompt_text = LONG_PROMPT
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True, tokenize=False,
            )
        long_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
        print(f"Prompt: {long_ids.shape[1]} tokens")
        print(f"Gen: {GEN_LENGTH}, Block: {BLOCK_LENGTH}, Target block idx: {TARGET_BLOCK_IDX}")

        # Install hooks once (they stay on the model across batch sizes)
        print("Installing MoE hooks ...", flush=True)
        collector = MoEDataCollector(NUM_MOE_LAYERS, BLOCK_LENGTH, NUM_EXPERTS, TOP_K)
        hooks = install_moe_hooks(model, collector)
        print(f"  Installed {len(hooks)} hooks")

        # Warmup with batch=1
        print("Warmup run (batch=1) ...", flush=True)
        dllm = BlockDiffusionLLM(
            model,
            ThresholdParallelDecoder(temperature=0.0, threshold=THRESHOLD, mask_id=MASK_ID, eos_id=EOS_ID),
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, lazy_cache_update=True, inplace_cache_update=True,
        )
        with torch.inference_mode():
            dllm.generate(long_ids, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)

        all_results = {}

        for bs in BATCH_SIZES:
            print(f"\n{'#'*100}")
            print(f"  BATCH SIZE = {bs}")
            print(f"{'#'*100}")

            # Reset collector for this batch size
            collector.routing_data = defaultdict(dict)
            collector.weight_data = defaultdict(dict)
            collector.moe_output = defaultdict(dict)
            collector.mask_state = {}
            collector.active = False
            collector.recording = False
            collector.current_iter = -1
            collector.layer_counter = 0

            batched_ids = long_ids.repeat(bs, 1)

            dllm = BlockDiffusionLLM(
                model,
                ThresholdParallelDecoder(temperature=0.0, threshold=THRESHOLD, mask_id=MASK_ID, eos_id=EOS_ID),
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, lazy_cache_update=True, inplace_cache_update=True,
            )

            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            try:
                out = run_instrumented_generate(dllm, batched_ids, collector, TARGET_BLOCK_IDX)
                torch.cuda.synchronize(device)
                wall = time.perf_counter() - t0
                print(f"  Generate: {wall:.3f}s, {dllm.num_forwards} forwards")

                results = analyze_routing(collector, batch_size=bs)
                results["batch_size"] = bs
                results["wall_s"] = wall
                results["num_forwards"] = dllm.num_forwards
                all_results[f"batch_{bs}"] = results
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at batch={bs}! Skipping.", flush=True)
                torch.cuda.empty_cache()
                continue

        # Remove hooks
        remove_hooks(hooks)

        # Save all results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "moe_routing_analysis_multibatch_results.json"

        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {str(k): clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj]
            elif isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return str(obj)
                return obj
            return obj

        out_path.write_text(json.dumps(clean_for_json(all_results), ensure_ascii=False, indent=2) + "\n")
        print(f"\nSaved: {out_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
