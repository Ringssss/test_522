#!/usr/bin/env python3
"""
v0.1.14.5 — MoE Combination Transfer Pre-check

Tests whether single-point safe proxy rules remain valid when multiple
(layer, token) pairs are reused simultaneously. This is the critical
go/no-go checkpoint before deploying selective recompute policies.

Three test scenarios:
  Test A: Single layer, ALL qualifying tokens reused simultaneously
  Test B: Multiple layers, single token reused across ALL qualifying layers
  Test C: Full deployment — ALL qualifying (layer, token) pairs reused simultaneously

Four proxy rules (conservative → aggressive):
  R1: highly_stable + topk_overlap >= 1.0     (31% reuse, 6.8% FNR)
  R2: highly_stable + topk_overlap >= 0.875   (34% reuse, 7.3% FNR)
  R3: stable + gate_cos >= 0.999              (56% reuse, 11.7% FNR)
  R4: token_margin > 0.99                     (dominant single predictor)

Metrics:
  - Per-token final_logits_kl (compared to clean baseline)
  - Per-token top1_changed rate
  - Block-level exact_match rate
  - Degradation amplification vs single-point expectations
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")

# Constants
MASK_ID = 156895
EOS_ID = 156892
NUM_EXPERTS = 256
TOP_K = 8
BLOCK_LENGTH = 32
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
NUM_MOE_LAYERS = 19


# ============================================================
# Proxy rules
# ============================================================
def classify_token_state(fresh_data, step, token_idx, batch_idx):
    """Classify token state at (step, batch_idx, token_idx)."""
    mask_state = fresh_data["mask_state"]
    if mask_state[step][batch_idx, token_idx].item():
        return "mask", 0
    if step > 0 and mask_state[step - 1][batch_idx, token_idx].item():
        return "newly_decoded", 1
    top1_preds = fresh_data["logits_info"]
    stable_len = 1
    curr_pred = top1_preds[step]["top1_pred"][batch_idx, token_idx].item()
    for s in range(step - 1, -1, -1):
        if mask_state[s][batch_idx, token_idx].item():
            break
        if top1_preds[s]["top1_pred"][batch_idx, token_idx].item() == curr_pred:
            stable_len += 1
        else:
            break
    return ("highly_stable" if stable_len >= 4 else "stable_decoded"), stable_len


def compute_proxy_for_token(fresh_data, step, layer, token_idx, batch_idx):
    """Compute proxy signals for one (step, layer, token) triple."""
    if step == 0:
        return {"gate_topk_overlap_prev": 0.0, "gate_cos_prev": 0.0,
                "token_confidence": 0.0, "token_margin": 0.0}
    signals = {}
    # Top-k overlap
    tk_curr = set(fresh_data["topk_indices"][step][layer][batch_idx, token_idx].tolist())
    tk_prev = set(fresh_data["topk_indices"][step - 1][layer][batch_idx, token_idx].tolist())
    signals["gate_topk_overlap_prev"] = len(tk_curr & tk_prev) / TOP_K
    # Gate cosine
    g_curr = fresh_data["gate_logits_data"][step][layer][batch_idx, token_idx].float()
    g_prev = fresh_data["gate_logits_data"][step - 1][layer][batch_idx, token_idx].float()
    signals["gate_cos_prev"] = F.cosine_similarity(
        g_curr.unsqueeze(0), g_prev.unsqueeze(0)).item()
    # Token-level signals
    li = fresh_data["logits_info"][step]
    signals["token_confidence"] = li["confidence"][batch_idx, token_idx].item()
    signals["token_margin"] = li["margin"][batch_idx, token_idx].item()
    return signals


RULES = {
    "R1_conservative": {
        "desc": "highly_stable + topk>=1.0",
        "check": lambda state, stable_len, proxy: (
            state == "highly_stable" and
            proxy["gate_topk_overlap_prev"] >= 1.0
        ),
    },
    "R2_moderate": {
        "desc": "highly_stable + topk>=0.875",
        "check": lambda state, stable_len, proxy: (
            state == "highly_stable" and
            proxy["gate_topk_overlap_prev"] >= 0.875
        ),
    },
    "R3_aggressive": {
        "desc": "stable+ + gate_cos>=0.999",
        "check": lambda state, stable_len, proxy: (
            state in ("stable_decoded", "highly_stable") and
            proxy["gate_cos_prev"] >= 0.999
        ),
    },
    "R4_margin": {
        "desc": "token_margin > 0.99",
        "check": lambda state, stable_len, proxy: (
            proxy["token_margin"] > 0.99
        ),
    },
}


# ============================================================
# Identify qualifying (layer, token) pairs per rule
# ============================================================
def find_qualifying_pairs(fresh_data, step, batch_idx, rule_name):
    """Find all (layer, token_idx) pairs that qualify under a given rule.

    Returns: list of (layer, token_idx) tuples
    """
    rule_fn = RULES[rule_name]["check"]
    pairs = []
    block_len = fresh_data["mask_state"][step].shape[1]

    for layer in range(NUM_MOE_LAYERS):
        for tok_idx in range(block_len):
            state, stable_len = classify_token_state(fresh_data, step, tok_idx, batch_idx)
            proxy = compute_proxy_for_token(fresh_data, step, layer, tok_idx, batch_idx)
            if rule_fn(state, stable_len, proxy):
                pairs.append((layer, tok_idx))

    return pairs


# ============================================================
# Multi-site intervention runner
# ============================================================
class CombinationRunner:
    """Runs multi-site counterfactual interventions with KV cache.

    Unlike CounterfactualRunner (single-site), this replaces routed output
    at MULTIPLE (layer, token) positions simultaneously in one forward pass.
    """

    def __init__(self, model, fresh_data, kv_info, device="cuda:0"):
        self.model = model
        self.fresh_data = fresh_data
        self.device = device

        # KV cache info
        self.kv_data = kv_info["kv_data"]
        self.block_start = kv_info["block_start"]
        self.block_end = kv_info["block_end"]
        self.replace_position = kv_info["replace_position"]
        self.block_pos_ids = kv_info["pos_ids"]

        # Identify MoE layers
        layers = model.model.layers
        self.moe_layers = [(idx, l) for idx, l in enumerate(layers)
                           if hasattr(l, 'mlp') and hasattr(l.mlp, 'gate')]

    def _make_kv_cache(self):
        from dinfer.decoding.utils import KVCache
        kv = KVCache(self.kv_data.clone(), backend='vllm')
        return kv

    def _run_forward(self, block_tokens, reuse_map=None):
        """Run one model forward with optional multi-site intervention.

        Args:
            block_tokens: [batch, block_len] token IDs
            reuse_map: None for baseline, or dict mapping
                moe_layer_idx -> list of (batch_idx, token_idx, cached_routed_tensor)

        Returns:
            logits [batch, block_len, vocab]
        """
        kv_cache = self._make_kv_cache()
        orig_forwards = {}

        if reuse_map is not None:
            for moe_idx, (abs_idx, decoder_layer) in enumerate(self.moe_layers):
                moe_block = decoder_layer.mlp
                orig_forwards[moe_idx] = moe_block.forward

                def make_hook(moe_mod, mi, rmap):
                    def hooked_forward(hidden_states):
                        bsz, seq_len, h = hidden_states.shape
                        hs_flat = hidden_states.view(-1, h)
                        shared_res = moe_mod.shared_experts(hidden_states)
                        router_logits = moe_mod.gate.get_logits(hs_flat)
                        routed_y = moe_mod.experts.forward_impl(
                            hidden_states=hs_flat, router_logits=router_logits)
                        routed_y = routed_y.view(bsz, seq_len, h)

                        # Apply cached routed outputs for qualifying tokens
                        if mi in rmap:
                            routed_y = routed_y.clone()
                            for b_idx, t_idx, cached_val in rmap[mi]:
                                routed_y[b_idx, t_idx] = cached_val

                        out = (routed_y + shared_res
                               if moe_mod.config.num_shared_experts is not None
                               else routed_y)
                        return out
                    return hooked_forward

                moe_block.forward = make_hook(moe_block, moe_idx, reuse_map)

        try:
            with torch.inference_mode():
                output = self.model(
                    block_tokens.to(self.device),
                    position_ids=self.block_pos_ids.to(self.device),
                    use_cache=True,
                    past_key_values=kv_cache,
                    replace_position=self.replace_position,
                )
            logits = output.logits
        finally:
            if reuse_map is not None:
                for moe_idx, (abs_idx, decoder_layer) in enumerate(self.moe_layers):
                    decoder_layer.mlp.forward = orig_forwards[moe_idx]

        return logits

    def build_reuse_map(self, step, pairs, batch_idx):
        """Build reuse_map from list of (layer, token_idx) pairs.

        Args:
            step: current denoising step
            pairs: list of (layer, token_idx)
            batch_idx: which batch element

        Returns:
            dict mapping moe_layer_idx -> [(batch_idx, token_idx, cached_tensor)]
        """
        rmap = defaultdict(list)
        for layer, tok_idx in pairs:
            cached_routed = self.fresh_data["routed_output"][step - 1][layer][
                batch_idx, tok_idx].to(self.device)
            rmap[layer].append((batch_idx, tok_idx, cached_routed))
        return dict(rmap)

    def run_combination_test(self, step, pairs, batch_idx, baseline_logits):
        """Run combination intervention and compute metrics.

        Args:
            step: iteration index
            pairs: list of (layer, token_idx) to reuse
            batch_idx: which batch element
            baseline_logits: [batch, block_len, vocab] clean baseline

        Returns:
            dict with per-token and aggregate metrics
        """
        if not pairs:
            return {"n_reused": 0, "skipped": True}

        block_tokens = self.fresh_data["token_ids"][step]
        reuse_map = self.build_reuse_map(step, pairs, batch_idx)

        # Run intervention forward
        interv_logits = self._run_forward(block_tokens, reuse_map=reuse_map)

        # Compute per-token metrics for the intervened batch element
        bl = baseline_logits[batch_idx].float().cpu()  # [block_len, vocab]
        il = interv_logits[batch_idx].float().cpu()

        bl_probs = F.softmax(bl, dim=-1)
        il_probs = F.softmax(il, dim=-1)

        block_len = bl.shape[0]
        per_token_kl = []
        per_token_top1_changed = []

        for t in range(block_len):
            kl = F.kl_div(il_probs[t].log().unsqueeze(0),
                          bl_probs[t].unsqueeze(0),
                          reduction='batchmean', log_target=False).item()
            top1_ch = (bl_probs[t].argmax().item() != il_probs[t].argmax().item())
            per_token_kl.append(kl)
            per_token_top1_changed.append(top1_ch)

        # Identify which tokens were actually reused
        reused_tokens = set()
        for layer, tok_idx in pairs:
            reused_tokens.add(tok_idx)

        # Aggregate
        reused_kls = [per_token_kl[t] for t in reused_tokens]
        non_reused_kls = [per_token_kl[t] for t in range(block_len) if t not in reused_tokens]

        n_top1_changed = sum(per_token_top1_changed)
        n_reused_top1_changed = sum(per_token_top1_changed[t] for t in reused_tokens)

        # Risk classification per token (same thresholds as single-point)
        n_unsafe = sum(1 for kl in per_token_kl if kl >= 1e-2)
        n_borderline = sum(1 for kl in per_token_kl if 1e-3 <= kl < 1e-2)
        n_safe = sum(1 for kl in per_token_kl if kl < 1e-3)

        # Reused-token risk
        reused_unsafe = sum(1 for t in reused_tokens
                           if per_token_kl[t] >= 1e-2 or per_token_top1_changed[t])
        reused_borderline = sum(1 for t in reused_tokens
                               if 1e-3 <= per_token_kl[t] < 1e-2
                               and not per_token_top1_changed[t])

        return {
            "n_reused": len(pairs),
            "n_unique_tokens_reused": len(reused_tokens),
            "n_layers_involved": len(set(l for l, _ in pairs)),
            "skipped": False,
            # Per-token KL stats
            "mean_kl_all": sum(per_token_kl) / block_len,
            "max_kl_all": max(per_token_kl),
            "mean_kl_reused": (sum(reused_kls) / len(reused_kls)) if reused_kls else 0,
            "max_kl_reused": max(reused_kls) if reused_kls else 0,
            "mean_kl_non_reused": (sum(non_reused_kls) / len(non_reused_kls)) if non_reused_kls else 0,
            "max_kl_non_reused": max(non_reused_kls) if non_reused_kls else 0,
            # Top-1 changes
            "n_top1_changed": n_top1_changed,
            "n_reused_top1_changed": n_reused_top1_changed,
            "top1_changed_rate": n_top1_changed / block_len,
            # Risk distribution (block-wide)
            "n_safe": n_safe,
            "n_borderline": n_borderline,
            "n_unsafe": n_unsafe,
            # Reused-token risk
            "reused_unsafe": reused_unsafe,
            "reused_borderline": reused_borderline,
            "per_token_kl": per_token_kl,
            "per_token_top1_changed": per_token_top1_changed,
        }


# ============================================================
# KV cache capture (reused from moe_counterfactual.py)
# ============================================================
def capture_kv_cache(dllm, input_ids, target_block_idx=1):
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner
    orig_decode = BlockDiffusionRunner.decode
    captured = {}

    def capturing_decode(self_runner, model, decoder, x, kv_cache, block, block_loc,
                         block_id, pos_ids, attn_mask, block_length=32,
                         cross_block_attn_mask=None):
        gen_block_idx = captured.get("_block_counter", 0)
        if gen_block_idx == target_block_idx:
            kv_cache.extend_cache(block_loc.end)
            past_kv, replace_pos = kv_cache.get_key_values(block_loc.start, block_loc.end)
            captured["kv_data"] = past_kv._data.clone()
            captured["block_start"] = block_loc.start
            captured["block_end"] = block_loc.end
            captured["replace_position"] = replace_pos
            captured["pos_ids"] = pos_ids[:, block_loc.start:block_loc.end].clone()
            print(f"  [KV Capture] block_loc={block_loc.start}:{block_loc.end}, "
                  f"KV shape={past_kv._data.shape}, replace_pos={replace_pos}", flush=True)
        result = orig_decode(self_runner, model, decoder, x, kv_cache, block,
                             block_loc, block_id, pos_ids, attn_mask,
                             block_length, cross_block_attn_mask)
        captured["_block_counter"] = gen_block_idx + 1
        return result

    BlockDiffusionRunner.decode = capturing_decode
    try:
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            dllm.diff_iteration.iter_no = 0
            _ = dllm.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
    finally:
        BlockDiffusionRunner.decode = orig_decode

    assert "kv_data" in captured, "Failed to capture KV cache"
    return captured


# ============================================================
# Main experiment
# ============================================================
def main():
    import socket
    from contextlib import closing
    from transformers import AutoConfig, AutoTokenizer

    TEMPERATURE = 0.7
    THRESHOLD = 0.90

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

    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    # --- Load fresh run data ---
    print("=" * 80)
    print("v0.1.14.5 — MoE Combination Transfer Pre-check")
    print("=" * 80)
    print("\nLoading fresh run data...", flush=True)
    data_path = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction" / "full_fresh_run_data.pt"
    fresh_data = torch.load(data_path, map_location="cpu")
    n_iters = len(fresh_data["mask_state"])
    n_layers = len(fresh_data["routed_output"][0])
    print(f"  {n_iters} iterations, {n_layers} MoE layers")

    # --- Load single-point results for comparison ---
    print("Loading single-point counterfactual results...", flush=True)
    sp_path = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction" / "counterfactual_kvcache_full.json"
    with open(sp_path) as f:
        single_point_results = json.load(f)
    print(f"  {len(single_point_results)} single-point samples loaded")

    # Compute single-point baseline stats per rule
    print("\n--- Single-point baseline (for comparison) ---")
    for rule_name, rule_info in RULES.items():
        # Count how many single-point samples would qualify and their risk
        qualifying = []
        for sp in single_point_results:
            state = sp["token_state"]
            stable_len = sp["stable_len"]
            proxy = {
                "gate_topk_overlap_prev": sp["gate_topk_overlap_prev"],
                "gate_cos_prev": sp["gate_cos_prev"],
                "token_confidence": sp["token_confidence"],
                "token_margin": sp["token_margin"],
            }
            if rule_info["check"](state, stable_len, proxy):
                qualifying.append(sp)
        if qualifying:
            n_q = len(qualifying)
            n_safe = sum(1 for s in qualifying if s["risk_label"] == "safe")
            n_unsafe = sum(1 for s in qualifying if s["risk_label"] == "unsafe")
            avg_kl = sum(s["final_logits_kl"] for s in qualifying) / n_q
            max_kl = max(s["final_logits_kl"] for s in qualifying)
            print(f"  {rule_name}: {n_q} qualifying ({n_q/len(single_point_results)*100:.1f}%), "
                  f"safe={n_safe/n_q*100:.1f}%, unsafe={n_unsafe/n_q*100:.1f}%, "
                  f"avg_kl={avg_kl:.6f}, max_kl={max_kl:.6f}")

    # --- Init model ---
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

    print("\nLoading model...", flush=True)
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

        # --- Tokenize ---
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
        print(f"  Input shape: {input_ids.shape}")

        # --- Capture KV cache ---
        print("\nCapturing KV cache...", flush=True)
        kv_info = capture_kv_cache(dllm, input_ids, target_block_idx=1)
        print(f"  Block: {kv_info['block_start']}:{kv_info['block_end']}")

        # --- Create runner ---
        runner = CombinationRunner(model, fresh_data, kv_info, device=DEVICE)

        # --- Experiment parameters ---
        # Sample steps: early, mid, late phases
        ALL_STEPS = list(range(1, n_iters))
        # Use 8 representative steps to keep runtime manageable
        # (each combination forward is expensive)
        SAMPLE_STEPS = [2, 5, 8, 11, 15, 19, 23, 27]
        SAMPLE_STEPS = [s for s in SAMPLE_STEPS if s < n_iters]
        BATCH_INDICES = [0, 1]  # Two batch elements for statistical diversity

        print(f"\n{'='*80}")
        print(f"COMBINATION TRANSFER EXPERIMENTS")
        print(f"  Steps: {SAMPLE_STEPS}")
        print(f"  Batch indices: {BATCH_INDICES}")
        print(f"  Rules: {list(RULES.keys())}")
        print(f"{'='*80}")

        all_experiment_results = {}
        t0_all = time.time()

        for rule_name in RULES:
            print(f"\n{'='*60}")
            print(f"Rule: {rule_name} — {RULES[rule_name]['desc']}")
            print(f"{'='*60}")

            rule_results = {
                "test_A": [],  # single-layer multi-token
                "test_B": [],  # multi-layer single-token
                "test_C": [],  # full deployment
            }

            for step in SAMPLE_STEPS:
                for batch_idx in BATCH_INDICES:
                    # Find all qualifying pairs for this step/batch/rule
                    all_pairs = find_qualifying_pairs(
                        fresh_data, step, batch_idx, rule_name)

                    if not all_pairs:
                        continue

                    # Organize by layer and by token
                    by_layer = defaultdict(list)
                    by_token = defaultdict(list)
                    for layer, tok_idx in all_pairs:
                        by_layer[layer].append((layer, tok_idx))
                        by_token[tok_idx].append((layer, tok_idx))

                    # Get baseline logits (clean, no intervention)
                    block_tokens = fresh_data["token_ids"][step]
                    baseline_logits = runner._run_forward(block_tokens)
                    baseline_logits_cpu = baseline_logits.float().cpu()

                    # ---- Test A: Single layer, all qualifying tokens ----
                    # Pick layers with most qualifying tokens
                    top_layers = sorted(by_layer.keys(),
                                       key=lambda l: len(by_layer[l]), reverse=True)
                    for layer in top_layers[:3]:  # top-3 layers
                        pairs_A = by_layer[layer]
                        if len(pairs_A) < 2:
                            continue  # Need at least 2 tokens for meaningful test
                        result_A = runner.run_combination_test(
                            step, pairs_A, batch_idx, baseline_logits_cpu)
                        result_A["step"] = step
                        result_A["batch_idx"] = batch_idx
                        result_A["test_layer"] = layer
                        result_A["test_type"] = "A_single_layer_multi_token"
                        rule_results["test_A"].append(result_A)

                    # ---- Test B: All qualifying layers, single token ----
                    # Pick tokens with most qualifying layers
                    top_tokens = sorted(by_token.keys(),
                                       key=lambda t: len(by_token[t]), reverse=True)
                    for tok_idx in top_tokens[:3]:  # top-3 tokens
                        pairs_B = by_token[tok_idx]
                        if len(pairs_B) < 2:
                            continue
                        result_B = runner.run_combination_test(
                            step, pairs_B, batch_idx, baseline_logits_cpu)
                        result_B["step"] = step
                        result_B["batch_idx"] = batch_idx
                        result_B["test_token"] = tok_idx
                        result_B["test_type"] = "B_multi_layer_single_token"
                        rule_results["test_B"].append(result_B)

                    # ---- Test C: Full deployment ----
                    result_C = runner.run_combination_test(
                        step, all_pairs, batch_idx, baseline_logits_cpu)
                    result_C["step"] = step
                    result_C["batch_idx"] = batch_idx
                    result_C["test_type"] = "C_full_deployment"
                    rule_results["test_C"].append(result_C)

                    print(f"  step={step} batch={batch_idx}: "
                          f"{len(all_pairs)} pairs, "
                          f"A={len(rule_results['test_A'])} "
                          f"B={len(rule_results['test_B'])} "
                          f"C={len(rule_results['test_C'])}", flush=True)

            # ---- Print per-rule summary ----
            print(f"\n  --- {rule_name} Summary ---")
            for test_name in ["test_A", "test_B", "test_C"]:
                results = [r for r in rule_results[test_name] if not r.get("skipped")]
                if not results:
                    print(f"  {test_name}: no valid results")
                    continue

                avg_kl = sum(r["mean_kl_all"] for r in results) / len(results)
                max_kl = max(r["max_kl_all"] for r in results)
                avg_reused_kl = sum(r["mean_kl_reused"] for r in results) / len(results)
                max_reused_kl = max(r["max_kl_reused"] for r in results)
                avg_top1_rate = sum(r["top1_changed_rate"] for r in results) / len(results)
                avg_n_reused = sum(r["n_reused"] for r in results) / len(results)
                avg_n_tokens = sum(r["n_unique_tokens_reused"] for r in results) / len(results)
                avg_n_layers = sum(r["n_layers_involved"] for r in results) / len(results)

                # Count how many tests had ANY unsafe token
                n_with_unsafe = sum(1 for r in results if r["n_unsafe"] > 0)
                # Count how many tests had unsafe on reused tokens
                n_reused_unsafe = sum(1 for r in results if r["reused_unsafe"] > 0)

                label = {
                    "test_A": "Single-layer multi-token",
                    "test_B": "Multi-layer single-token",
                    "test_C": "Full deployment",
                }[test_name]

                print(f"\n  {label} ({len(results)} runs):")
                print(f"    Avg reuse: {avg_n_reused:.1f} pairs "
                      f"({avg_n_tokens:.1f} tokens × {avg_n_layers:.1f} layers)")
                print(f"    KL (all tokens):   avg={avg_kl:.6f}  max={max_kl:.6f}")
                print(f"    KL (reused only):  avg={avg_reused_kl:.6f}  max={max_reused_kl:.6f}")
                print(f"    Top-1 changed rate: {avg_top1_rate:.4f}")
                print(f"    Runs with ANY unsafe token: {n_with_unsafe}/{len(results)} "
                      f"({n_with_unsafe/len(results)*100:.1f}%)")
                print(f"    Runs with reused-token unsafe: {n_reused_unsafe}/{len(results)} "
                      f"({n_reused_unsafe/len(results)*100:.1f}%)")

            # Store for later serialization (remove per_token arrays for JSON)
            for test_name in ["test_A", "test_B", "test_C"]:
                for r in rule_results[test_name]:
                    r.pop("per_token_kl", None)
                    r.pop("per_token_top1_changed", None)

            all_experiment_results[rule_name] = rule_results

        elapsed_all = time.time() - t0_all

        # ============================================================
        # FINAL CROSS-RULE COMPARISON
        # ============================================================
        print(f"\n{'='*80}")
        print(f"FINAL CROSS-RULE COMPARISON (Test C — Full Deployment)")
        print(f"{'='*80}")
        print(f"{'Rule':<20s} {'Runs':>5s} {'AvgReuse':>8s} "
              f"{'AvgKL':>10s} {'MaxKL':>10s} "
              f"{'Top1%':>7s} {'UnsafeRuns%':>11s} {'ReuseUnsafe%':>12s}")
        print("-" * 95)

        for rule_name in RULES:
            results = [r for r in all_experiment_results[rule_name]["test_C"]
                      if not r.get("skipped")]
            if not results:
                print(f"{rule_name:<20s} {'N/A':>5s}")
                continue
            n = len(results)
            avg_reuse = sum(r["n_reused"] for r in results) / n
            avg_kl = sum(r["mean_kl_all"] for r in results) / n
            max_kl = max(r["max_kl_all"] for r in results)
            avg_top1 = sum(r["top1_changed_rate"] for r in results) / n
            n_unsafe_runs = sum(1 for r in results if r["n_unsafe"] > 0)
            n_reuse_unsafe = sum(1 for r in results if r["reused_unsafe"] > 0)

            print(f"{rule_name:<20s} {n:>5d} {avg_reuse:>8.1f} "
                  f"{avg_kl:>10.6f} {max_kl:>10.6f} "
                  f"{avg_top1*100:>6.2f}% "
                  f"{n_unsafe_runs/n*100:>10.1f}% "
                  f"{n_reuse_unsafe/n*100:>11.1f}%")

        # ============================================================
        # DEGRADATION AMPLIFICATION ANALYSIS
        # ============================================================
        print(f"\n{'='*80}")
        print(f"DEGRADATION AMPLIFICATION: Combination vs Single-point")
        print(f"{'='*80}")
        print(f"  If combination KL ≈ single-point KL → rules transfer safely")
        print(f"  If combination KL >> single-point KL → error accumulates, rules don't transfer")
        print()

        # Compute single-point avg KL for qualifying samples per rule
        for rule_name, rule_info in RULES.items():
            sp_qualifying = []
            for sp in single_point_results:
                state = sp["token_state"]
                stable_len = sp["stable_len"]
                proxy = {
                    "gate_topk_overlap_prev": sp["gate_topk_overlap_prev"],
                    "gate_cos_prev": sp["gate_cos_prev"],
                    "token_confidence": sp["token_confidence"],
                    "token_margin": sp["token_margin"],
                }
                if rule_info["check"](state, stable_len, proxy):
                    sp_qualifying.append(sp)

            if not sp_qualifying:
                continue

            sp_avg_kl = sum(s["final_logits_kl"] for s in sp_qualifying) / len(sp_qualifying)
            sp_max_kl = max(s["final_logits_kl"] for s in sp_qualifying)
            sp_unsafe_rate = sum(1 for s in sp_qualifying
                               if s["risk_label"] == "unsafe") / len(sp_qualifying)

            combo_results = [r for r in all_experiment_results[rule_name]["test_C"]
                           if not r.get("skipped")]
            if not combo_results:
                continue

            combo_avg_kl = sum(r["mean_kl_all"] for r in combo_results) / len(combo_results)
            combo_max_kl = max(r["max_kl_all"] for r in combo_results)
            combo_unsafe_rate = sum(1 for r in combo_results
                                   if r["n_unsafe"] > 0) / len(combo_results)

            amplification = combo_avg_kl / (sp_avg_kl + 1e-10)

            print(f"  {rule_name}:")
            print(f"    Single-point: avg_kl={sp_avg_kl:.6f}  max_kl={sp_max_kl:.6f}  "
                  f"unsafe_rate={sp_unsafe_rate*100:.1f}%")
            print(f"    Combination:  avg_kl={combo_avg_kl:.6f}  max_kl={combo_max_kl:.6f}  "
                  f"unsafe_runs={combo_unsafe_rate*100:.1f}%")
            print(f"    Amplification ratio (combo/single avg_kl): {amplification:.2f}x")
            print()

        # ============================================================
        # GO / NO-GO DECISION
        # ============================================================
        print(f"{'='*80}")
        print(f"GO / NO-GO ASSESSMENT")
        print(f"{'='*80}")

        for rule_name in RULES:
            combo_C = [r for r in all_experiment_results[rule_name]["test_C"]
                      if not r.get("skipped")]
            if not combo_C:
                print(f"  {rule_name}: SKIP (no data)")
                continue

            avg_kl = sum(r["mean_kl_all"] for r in combo_C) / len(combo_C)
            max_kl = max(r["max_kl_all"] for r in combo_C)
            unsafe_run_rate = sum(1 for r in combo_C if r["n_unsafe"] > 0) / len(combo_C)
            avg_top1_rate = sum(r["top1_changed_rate"] for r in combo_C) / len(combo_C)

            # Decision criteria:
            #   GO:   avg_kl < 1e-3 AND max_kl < 1e-1 AND top1_rate < 5%
            #   CONDITIONAL: avg_kl < 1e-2 AND top1_rate < 10%
            #   NO-GO: otherwise
            if avg_kl < 1e-3 and max_kl < 1e-1 and avg_top1_rate < 0.05:
                verdict = "GO"
            elif avg_kl < 1e-2 and avg_top1_rate < 0.10:
                verdict = "CONDITIONAL GO"
            else:
                verdict = "NO-GO"

            print(f"  {rule_name}: {verdict}")
            print(f"    avg_kl={avg_kl:.6f} max_kl={max_kl:.6f} "
                  f"top1_rate={avg_top1_rate*100:.2f}% unsafe_runs={unsafe_run_rate*100:.1f}%")

        print(f"\nTotal experiment time: {elapsed_all:.1f}s")

        # --- Save results ---
        results_dir = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction"
        save_path = results_dir / "combination_transfer_results.json"
        with open(save_path, "w") as f:
            json.dump(all_experiment_results, f, indent=2, default=str)
        print(f"\nSaved to {save_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
