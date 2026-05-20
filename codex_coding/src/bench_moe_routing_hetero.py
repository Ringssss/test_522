#!/usr/bin/env python3
"""
MoE routing & output profiling for dLLM — heterogeneous batch + stochastic decoding.

Key changes from the homogeneous version:
  - Multiple DIFFERENT prompts to form non-homogeneous batches
  - temperature=0.7 (Gumbel noise) so each batch element decodes differently
  - Runs batch=8 and batch=32
  - Analysis aggregates across all batch elements

Collects per-position × per-iteration × per-layer data for one generation block:
  - Expert routing decisions (topk_idx from gate)
  - MoE block output hidden states (subset of batch for memory)
  - MASK vs decoded position tracking per batch element
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
TEMPERATURE = 0.7  # stochastic decoding
MASK_ID = 156895
EOS_ID = 156892
DEVICE = "cuda:0"
NUM_EXPERTS = 256
TOP_K = 8
NUM_MOE_LAYERS = 19

TARGET_BLOCK_IDX = 1
BATCH_SIZES = [8, 32]
MAX_OUTPUT_BATCH = 4  # only store MoE output for first N batch elements (memory)

# 8 diverse prompts of similar length for non-homogeneous batching
PROMPTS = [
    "Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h. The total distance between the two cities is 240 km. What is the average speed for the entire round trip?\n\nProblem 2: A rectangular garden has a perimeter of 56 meters. If the length is 4 meters more than twice the width, find the dimensions of the garden.\n\nProblem 3: In a class of 40 students, 25 study Mathematics, 20 study Physics, and 10 study both subjects. How many students study neither Mathematics nor Physics?\n\nProblem 4: A cone has a radius of 7 cm and a slant height of 25 cm. Calculate the total surface area and the volume of the cone.\n\nProblem 5: A bank offers compound interest at 8% per annum, compounded quarterly. If you deposit $5000, how much will you have after 3 years?",

    "Write a detailed essay about the history of artificial intelligence, covering the following topics: the Dartmouth conference of 1956, the AI winters of the 1970s and 1980s, the rise of machine learning in the 1990s, deep learning breakthroughs in the 2010s, and the current state of large language models. For each era, discuss the key researchers, major achievements, and limitations that defined that period. Also discuss the societal implications.",

    "You are a chemistry professor. Explain the following concepts in detail with examples: (1) Le Chatelier's principle and how it applies to industrial ammonia production, (2) the difference between SN1 and SN2 reaction mechanisms, (3) how buffer solutions work and calculate the pH of a buffer made from 0.1M acetic acid and 0.1M sodium acetate, (4) the principles behind electrochemistry and how a galvanic cell works, and (5) molecular orbital theory for diatomic molecules.",

    "Design a complete REST API for an e-commerce platform. The API should include endpoints for user authentication (register, login, logout, refresh token), product management (CRUD operations, search, filtering by category and price range), shopping cart operations (add item, remove item, update quantity, get cart), order processing (place order, cancel order, track order status, order history), and payment integration. For each endpoint, specify the HTTP method, URL path, request body, response format, and error codes.",

    "Analyze the global economic impact of climate change across different sectors. Cover agriculture (crop yields, water scarcity, food prices), energy (transition costs, infrastructure damage, renewable adoption rates), real estate (coastal flooding, insurance markets, migration patterns), healthcare (heat-related illness, vector-borne diseases, mental health), and finance (stranded assets, green bonds, carbon markets). Provide specific examples from at least three different countries.",

    "Explain quantum computing to someone with a background in classical computer science. Cover the following: (1) What are qubits and how they differ from classical bits, (2) superposition and entanglement with concrete examples, (3) quantum gates and how they compare to classical logic gates, (4) Shor's algorithm and its implications for cryptography, (5) Grover's search algorithm and its speedup, (6) current hardware approaches (superconducting, trapped ion, photonic), and (7) error correction challenges.",

    "You are a systems architect. Design a distributed message queue system similar to Apache Kafka. Describe the architecture including: (1) broker design with partition-based storage, (2) producer protocol with batching and compression, (3) consumer groups with offset management, (4) replication strategy for fault tolerance with leader election, (5) exactly-once delivery semantics, (6) data retention and compaction policies, (7) monitoring and alerting metrics, and (8) capacity planning guidelines.",

    "Write a comprehensive guide to training large language models. Cover the following stages: (1) data collection and cleaning including deduplication and quality filtering, (2) tokenizer training with BPE and SentencePiece, (3) model architecture decisions including attention mechanisms and positional encoding, (4) distributed training strategies including data parallelism, tensor parallelism, and pipeline parallelism, (5) learning rate scheduling and optimization tricks, (6) evaluation benchmarks and metrics, and (7) alignment techniques including RLHF and DPO.",
]


# ============================================================
# Data collector
# ============================================================
class MoEDataCollector:
    """Collects routing and hidden state data during MoE forwards."""

    def __init__(self, block_length, num_experts, top_k):
        self.block_length = block_length
        self.num_experts = num_experts
        self.top_k = top_k

        self.active = False
        self.recording = False
        self.current_iter = -1
        self.layer_counter = 0
        self.block_start = 0
        self.block_end = 0
        self.batch_size = 1

        # routing_data[iter][layer] = [batch, block_len, top_k]
        self.routing_data = defaultdict(dict)
        # moe_output[iter][layer] = [min(batch, MAX_OUTPUT_BATCH), block_len, hidden]
        self.moe_output = defaultdict(dict)
        # mask_state[iter] = [batch, block_len] bool
        self.mask_state = {}

    def reset(self):
        self.active = False
        self.recording = False
        self.current_iter = -1
        self.layer_counter = 0
        self.routing_data = defaultdict(dict)
        self.moe_output = defaultdict(dict)
        self.mask_state = {}

    def start_iteration(self, iter_idx, mask_positions, block_start, block_end, batch_size):
        self.current_iter = iter_idx
        self.layer_counter = 0
        self.block_start = block_start
        self.block_end = block_end
        self.batch_size = batch_size
        self.recording = True
        self.mask_state[iter_idx] = mask_positions.clone().cpu()

    def end_iteration(self):
        self.layer_counter = 0
        self.recording = False

    def record_routing(self, topk_idx, topk_weight):
        if not self.active or not self.recording:
            return
        layer = self.layer_counter
        bsz = self.batch_size
        full_tokens = topk_idx.shape[0]
        seq_per_batch = full_tokens // bsz
        block_len = self.block_end - self.block_start

        topk_3d = topk_idx.view(bsz, seq_per_batch, self.top_k)
        if seq_per_batch <= block_len:
            self.routing_data[self.current_iter][layer] = topk_3d[:, -block_len:, :].detach().cpu()
        else:
            self.routing_data[self.current_iter][layer] = topk_3d[:, self.block_start:self.block_end, :].detach().cpu()

    def record_moe_io(self, input_hs, output_hs):
        if not self.active or not self.recording:
            return
        layer = self.layer_counter
        full_seq = output_hs.shape[1]
        block_len = self.block_end - self.block_start
        store_bsz = min(output_hs.shape[0], MAX_OUTPUT_BATCH)
        if full_seq <= block_len:
            self.moe_output[self.current_iter][layer] = output_hs[:store_bsz, -block_len:, :].detach().cpu()
        else:
            self.moe_output[self.current_iter][layer] = output_hs[:store_bsz, self.block_start:self.block_end, :].detach().cpu()
        self.layer_counter += 1

    def get_num_iterations(self):
        return len(self.mask_state)


# ============================================================
# Hook installation
# ============================================================
def install_moe_hooks(model, collector):
    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue
        gate = layer.mlp.gate
        moe_block = layer.mlp

        orig_routing = gate.routing
        def make_routing_hook(orig_fn):
            def hooked_routing(self_gate, hidden_states, gating_output, topk, renormalize):
                topk_weight, topk_idx = orig_fn(hidden_states, gating_output, topk, renormalize)
                collector.record_routing(topk_idx, topk_weight)
                return topk_weight, topk_idx
            return hooked_routing
        gate.routing = make_routing_hook(orig_routing).__get__(gate)
        hooks.append(('routing', gate, orig_routing))

        def make_moe_hook():
            def hook_fn(module, input_tuple, output):
                inp = input_tuple[0]
                out = output if isinstance(output, torch.Tensor) else output[0]
                collector.record_moe_io(inp, out)
            return hook_fn
        h = moe_block.register_forward_hook(make_moe_hook())
        hooks.append(('moe_hook', h, None))
    return hooks


def remove_hooks(hooks):
    for kind, obj, orig in hooks:
        if kind == 'routing':
            obj.routing = orig
        elif kind == 'moe_hook':
            obj.remove()


# ============================================================
# Instrumented generate
# ============================================================
def run_instrumented_generate(dllm, input_ids, collector, target_block_idx):
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner

    orig_forward = BlockDiffusionIteration.forward
    orig_decode = BlockDiffusionRunner.decode
    current_block_idx = [0]
    iteration_in_block = [0]

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

    BlockDiffusionIteration.forward = patched_forward
    BlockDiffusionRunner.decode = patched_decode
    try:
        with torch.inference_mode():
            out = dllm.generate(input_ids, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
    finally:
        BlockDiffusionIteration.forward = orig_forward
        BlockDiffusionRunner.decode = orig_decode
    return out


# ============================================================
# Analysis
# ============================================================
def compute_entropy(counts, base=2):
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts.float() / total
    probs = probs[probs > 0]
    return -(probs * probs.log2()).sum().item()


def analyze_routing(collector, batch_size):
    num_iters = collector.get_num_iterations()
    if num_iters == 0:
        print("  No data collected!")
        return {}

    results = {}
    print(f"\n{'='*100}")
    print(f"  MoE ROUTING ANALYSIS — batch={batch_size}, {num_iters} iters, temperature={TEMPERATURE}")
    print(f"{'='*100}")

    # --- A1: MASK vs Decoded routing entropy (per layer, aggregated over all batch elements) ---
    print(f"\n--- A1: MASK vs Decoded Routing Entropy (per layer, all batch elements) ---")
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
            actual_bsz = min(topk_idx.shape[0], mask.shape[0])

            # Aggregate all batch elements
            all_mask_experts = []
            all_dec_experts = []
            for b in range(actual_bsz):
                mask_b = mask[b]        # [seq]
                topk_b = topk_idx[b]    # [seq, top_k]
                mask_pos = topk_b[mask_b]
                dec_pos = topk_b[~mask_b]
                if mask_pos.numel() > 0:
                    all_mask_experts.append(mask_pos)
                if dec_pos.numel() > 0:
                    all_dec_experts.append(dec_pos)

            if all_mask_experts:
                combined = torch.cat(all_mask_experts, dim=0).flatten()
                expert_counts = torch.zeros(NUM_EXPERTS)
                for e in combined:
                    expert_counts[e.item()] += 1
                mask_entropies.append(compute_entropy(expert_counts))
                mask_active_counts.append((expert_counts > 0).sum().item())

            if all_dec_experts:
                combined = torch.cat(all_dec_experts, dim=0).flatten()
                expert_counts = torch.zeros(NUM_EXPERTS)
                for e in combined:
                    expert_counts[e.item()] += 1
                dec_entropies.append(compute_entropy(expert_counts))
                dec_active_counts.append((expert_counts > 0).sum().item())

        avg_me = sum(mask_entropies) / len(mask_entropies) if mask_entropies else 0
        avg_de = sum(dec_entropies) / len(dec_entropies) if dec_entropies else 0
        ratio = avg_me / avg_de if avg_de > 0 else 0
        avg_ma = sum(mask_active_counts) / len(mask_active_counts) if mask_active_counts else 0
        avg_da = sum(dec_active_counts) / len(dec_active_counts) if dec_active_counts else 0

        print(f"  {layer:<8d} {avg_me:>10.3f} {avg_de:>10.3f} {ratio:>8.3f} "
              f"{avg_ma:>10.1f}/256 {avg_da:>10.1f}/256")
        a1_data[layer] = {"mask_entropy": avg_me, "dec_entropy": avg_de,
                          "ratio": ratio, "mask_active": avg_ma, "dec_active": avg_da}
    results["A1_entropy"] = a1_data

    # --- A2: Expert load balance (first vs last iteration, all batch elements) ---
    print(f"\n--- A2: Expert Load Balance (first vs last iter, all batch elements) ---")
    print(f"  {'Layer':<8s} {'First_max/avg':>14s} {'First_gini':>12s} "
          f"{'Last_max/avg':>14s} {'Last_gini':>12s}")
    print(f"  {'-'*8} {'-'*14} {'-'*12} {'-'*14} {'-'*12}")

    a2_data = {}
    for layer in sorted(collector.routing_data[0].keys()):
        layer_data = {}
        for label, it in [("first", 0), ("last", num_iters - 1)]:
            if layer not in collector.routing_data[it]:
                continue
            topk_idx = collector.routing_data[it][layer]  # [batch, seq, top_k]
            expert_counts = torch.zeros(NUM_EXPERTS)
            for e in topk_idx.flatten():
                expert_counts[e.item()] += 1
            avg_load = expert_counts.mean().item()
            max_load = expert_counts.max().item()
            ratio = max_load / avg_load if avg_load > 0 else 0
            sorted_counts = expert_counts.sort()[0]
            n = len(sorted_counts)
            gini = (2.0 * (torch.arange(1, n+1).float() * sorted_counts).sum() / (n * sorted_counts.sum()) - (n+1)/n).item() if sorted_counts.sum() > 0 else 0
            layer_data[label] = {"max_avg_ratio": ratio, "gini": gini}
        if "first" in layer_data and "last" in layer_data:
            print(f"  {layer:<8d} {layer_data['first']['max_avg_ratio']:>14.2f} "
                  f"{layer_data['first']['gini']:>12.4f} "
                  f"{layer_data['last']['max_avg_ratio']:>14.2f} "
                  f"{layer_data['last']['gini']:>12.4f}")
        a2_data[layer] = layer_data
    results["A2_load_balance"] = a2_data

    # --- B1: Cross-iteration routing stability (per batch element, then avg) ---
    print(f"\n--- B1: Cross-Iteration Routing Stability (averaged over batch elements) ---")
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
            curr_routing = collector.routing_data[it][layer]    # [batch, seq, top_k]
            prev_routing = collector.routing_data[it-1][layer]
            curr_mask = collector.mask_state[it]                # [batch, seq]
            prev_mask = collector.mask_state[it-1]
            actual_bsz = min(curr_routing.shape[0], prev_routing.shape[0],
                             curr_mask.shape[0], prev_mask.shape[0])

            for b in range(actual_bsz):
                decoded_both = ~curr_mask[b] & ~prev_mask[b]
                if decoded_both.any():
                    curr_sorted = curr_routing[b][decoded_both].sort(dim=-1)[0]
                    prev_sorted = prev_routing[b][decoded_both].sort(dim=-1)[0]
                    changed = (curr_sorted != prev_sorted).any(dim=-1)
                    dec_changes += changed.sum().item()
                    dec_total += decoded_both.sum().item()

                mask_both = curr_mask[b] & prev_mask[b]
                if mask_both.any():
                    curr_sorted = curr_routing[b][mask_both].sort(dim=-1)[0]
                    prev_sorted = prev_routing[b][mask_both].sort(dim=-1)[0]
                    changed = (curr_sorted != prev_sorted).any(dim=-1)
                    mask_changes += changed.sum().item()
                    mask_total += mask_both.sum().item()

        dec_pct = dec_changes / dec_total * 100 if dec_total > 0 else 0
        mask_pct = mask_changes / mask_total * 100 if mask_total > 0 else 0
        print(f"  {layer:<8d} {dec_pct:>11.2f}% {mask_pct:>12.2f}% "
              f"{dec_total:>10d} {mask_total:>11d}")
        b1_data[layer] = {"dec_change_pct": dec_pct, "mask_change_pct": mask_pct,
                          "dec_pairs": dec_total, "mask_pairs": mask_total}
    results["B1_routing_stability"] = b1_data

    # --- B2: Cross-iteration MoE output cosine similarity (first MAX_OUTPUT_BATCH elements) ---
    print(f"\n--- B2: Cross-Iteration MoE Output Cosine Similarity (first {MAX_OUTPUT_BATCH} batch elements) ---")
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
            curr_out = collector.moe_output[it][layer].float()
            prev_out = collector.moe_output[it-1][layer].float()
            curr_mask = collector.mask_state[it]
            prev_mask = collector.mask_state[it-1]
            store_bsz = min(curr_out.shape[0], prev_out.shape[0],
                            curr_mask.shape[0], prev_mask.shape[0])

            for b in range(store_bsz):
                decoded_both = ~curr_mask[b] & ~prev_mask[b]
                if decoded_both.any():
                    cos = F.cosine_similarity(curr_out[b][decoded_both], prev_out[b][decoded_both], dim=-1)
                    dec_sims.extend(cos.tolist())
                mask_both = curr_mask[b] & prev_mask[b]
                if mask_both.any():
                    cos = F.cosine_similarity(curr_out[b][mask_both], prev_out[b][mask_both], dim=-1)
                    mask_sims.extend(cos.tolist())

        avg_dec = sum(dec_sims) / len(dec_sims) if dec_sims else 0
        avg_mask = sum(mask_sims) / len(mask_sims) if mask_sims else 0
        print(f"  {layer:<8d} {avg_dec:>12.6f} {avg_mask:>13.6f} "
              f"{len(dec_sims):>10d} {len(mask_sims):>11d}")
        b2_data[layer] = {"dec_cos_sim": avg_dec, "mask_cos_sim": avg_mask,
                          "dec_pairs": len(dec_sims), "mask_pairs": len(mask_sims)}
    results["B2_output_similarity"] = b2_data

    # --- C1: Redundancy quantification ---
    print(f"\n--- C1: Redundancy Quantification (all batch elements) ---")
    total_computations = 0
    redundant_routing = 0
    for it in range(1, num_iters):
        for layer in collector.routing_data[it]:
            if layer not in collector.routing_data[it-1]:
                continue
            curr = collector.routing_data[it][layer]      # [batch, seq, top_k]
            prev = collector.routing_data[it-1][layer]
            actual_bsz = min(curr.shape[0], prev.shape[0])
            for b in range(actual_bsz):
                curr_sorted = curr[b].sort(dim=-1)[0]
                prev_sorted = prev[b].sort(dim=-1)[0]
                same = (curr_sorted == prev_sorted).all(dim=-1)
                total_computations += curr[b].shape[0]
                redundant_routing += same.sum().item()
    for layer in collector.routing_data.get(0, {}):
        r = collector.routing_data[0][layer]
        total_computations += r.shape[0] * r.shape[1]

    pct = redundant_routing / total_computations * 100 if total_computations > 0 else 0
    print(f"  Total token-expert-layer computations: {total_computations}")
    print(f"  Redundant (same routing as prev iter): {redundant_routing} ({pct:.1f}%)")

    # Per-iteration breakdown with mask diversity across batch
    print(f"\n  Per-iteration breakdown:")
    print(f"  {'Iter':<6s} {'Avg#MASK':>9s} {'Min#MASK':>9s} {'Max#MASK':>9s} {'Redundant%':>11s}")
    print(f"  {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*11}")
    c1_per_iter = []
    for it in range(num_iters):
        ms = collector.mask_state[it]  # [batch, seq]
        mask_counts = ms.sum(dim=1).float()
        avg_mask = mask_counts.mean().item()
        min_mask = mask_counts.min().item()
        max_mask = mask_counts.max().item()
        if it == 0:
            print(f"  {it:<6d} {avg_mask:>9.1f} {int(min_mask):>9d} {int(max_mask):>9d} {'(baseline)':>11s}")
            c1_per_iter.append({"iter": it, "avg_mask": avg_mask, "min_mask": int(min_mask),
                                "max_mask": int(max_mask), "redundant_pct": 0})
        else:
            iter_total = 0
            iter_redundant = 0
            for layer in collector.routing_data[it]:
                if layer not in collector.routing_data[it-1]:
                    continue
                curr = collector.routing_data[it][layer]
                prev = collector.routing_data[it-1][layer]
                abz = min(curr.shape[0], prev.shape[0])
                for b in range(abz):
                    curr_s = curr[b].sort(dim=-1)[0]
                    prev_s = prev[b].sort(dim=-1)[0]
                    same = (curr_s == prev_s).all(dim=-1)
                    iter_total += curr[b].shape[0]
                    iter_redundant += same.sum().item()
            ipct = iter_redundant / iter_total * 100 if iter_total > 0 else 0
            print(f"  {it:<6d} {avg_mask:>9.1f} {int(min_mask):>9d} {int(max_mask):>9d} {ipct:>10.1f}%")
            c1_per_iter.append({"iter": it, "avg_mask": avg_mask, "min_mask": int(min_mask),
                                "max_mask": int(max_mask), "redundant_pct": ipct})
    results["C1_redundancy"] = {
        "total_computations": total_computations, "redundant_routing": redundant_routing,
        "redundant_pct": pct, "per_iteration": c1_per_iter}

    return results


# ============================================================
# Main
# ============================================================
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_batch(prompts, tokenizer, batch_size, device):
    """Build a heterogeneous batch from diverse prompts, padded to same length."""
    all_ids = []
    for i in range(batch_size):
        text = prompts[i % len(prompts)]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=False)
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        all_ids.append(ids)

    # Pad to max length with pad_token_id (left padding)
    max_len = max(ids.shape[0] for ids in all_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    padded = []
    for ids in all_ids:
        if ids.shape[0] < max_len:
            pad = torch.full((max_len - ids.shape[0],), pad_id, dtype=ids.dtype)
            ids = torch.cat([pad, ids])  # left pad
        padded.append(ids)
    batch = torch.stack(padded, dim=0).to(device)
    lengths = [ids.shape[0] for ids in all_ids]
    print(f"  Batch: {batch_size} prompts, padded to {max_len} tokens "
          f"(original lengths: {min(lengths)}-{max(lengths)})")
    return batch


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (
        BlockDiffusionLLMAttnmask,
        BlockIteratorFactory,
        ThresholdParallelDecoder,
    )
    from dinfer.model import LLaDA2MoeModelLM

    port = find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("Loading model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=cfg).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)

        collector = MoEDataCollector(BLOCK_LENGTH, NUM_EXPERTS, TOP_K)
        hooks = install_moe_hooks(model, collector)
        print(f"Installed {len(hooks)} hooks")

        all_results = {}
        for batch_size in BATCH_SIZES:
            print(f"\n{'#'*100}")
            print(f"  BATCH SIZE = {batch_size}, temperature = {TEMPERATURE}")
            print(f"{'#'*100}")

            # Build heterogeneous batch
            batch_ids = build_batch(PROMPTS, tokenizer, batch_size, device)

            # Warmup (no collection)
            print("  Warmup ...", flush=True)
            collector.reset()
            dllm = BlockDiffusionLLMAttnmask(
                model,
                ThresholdParallelDecoder(temperature=TEMPERATURE, threshold=THRESHOLD,
                                         mask_id=MASK_ID, eos_id=EOS_ID),
                BlockIteratorFactory(use_block_diffusion=True),
                early_stop=True,
            )
            try:
                with torch.inference_mode():
                    dllm.generate(batch_ids, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            except Exception as e:
                print(f"  Warmup failed: {e}", flush=True)

            # Profiled run
            print(f"  Profiled run ...", flush=True)
            collector.reset()
            dllm = BlockDiffusionLLMAttnmask(
                model,
                ThresholdParallelDecoder(temperature=TEMPERATURE, threshold=THRESHOLD,
                                         mask_id=MASK_ID, eos_id=EOS_ID),
                BlockIteratorFactory(use_block_diffusion=True),
                early_stop=True,
            )
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            try:
                out = run_instrumented_generate(dllm, batch_ids, collector, TARGET_BLOCK_IDX)
                torch.cuda.synchronize(device)
                wall = time.perf_counter() - t0
                print(f"  Generate: {wall:.3f}s, {dllm.num_forwards} forwards")
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at batch={batch_size}!", flush=True)
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"  Failed: {e}", flush=True)
                import traceback; traceback.print_exc()
                continue

            results = analyze_routing(collector, batch_size)
            all_results[f"batch_{batch_size}"] = results
            torch.cuda.empty_cache()

        remove_hooks(hooks)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "moe_routing_analysis_hetero_batch.json"

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
