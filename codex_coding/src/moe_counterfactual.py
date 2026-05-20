#!/usr/bin/env python3
"""
MoE Single-Site Counterfactual Intervention Framework (v0.1.14.2 — KV cache version)

For each intervention sample (step t, layer l, token i):
  1. Re-run step t's forward WITH KV cache (matching real cache-path inference)
  2. At layer l's MoE, replace token i's routed output with cached value from step t-1
  3. Let the modified hidden state propagate through remaining layers
  4. Record: local MoE error, final hidden drift, final logits KL, top-1 change

Key design changes from v1:
  - Uses KV cache prefix captured from a real generation run
  - Baseline and intervention forwards both use the same KV cache
  - Covers ALL 19 MoE layers × ALL valid steps (full coverage)
  - Baseline logits cached per step (2 forwards → 1+1 per sample)
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ============================================================
# Data structures
# ============================================================
@dataclass
class InterventionSample:
    """One counterfactual intervention result."""
    step: int
    layer: int
    token_idx: int
    batch_idx: int

    token_state: str
    token_id: int
    token_confidence: float
    token_margin: float
    stable_len: int
    mask_ratio: float

    # Proxy signals
    hidden_cos_prev: float = 0.0
    hidden_rel_l2_prev: float = 0.0
    gate_cos_prev: float = 0.0
    gate_rel_l2_prev: float = 0.0
    gate_topk_overlap_prev: float = 0.0
    routing_weight_cos_prev: float = 0.0
    routing_weight_l1_prev: float = 0.0

    # Intervention results
    local_moe_cos: float = 0.0
    local_moe_rel_l2: float = 0.0
    final_logits_kl: float = 0.0
    final_top1_changed: bool = False
    confidence_shift: float = 0.0

    risk_label: str = "unknown"


# ============================================================
# Token classification & proxy computation (same as before)
# ============================================================
def classify_token_state(fresh_data, step, token_idx, batch_idx):
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


def compute_proxy_signals(fresh_data, step, layer, token_idx, batch_idx):
    if step == 0:
        return {k: 0.0 for k in [
            "hidden_cos_prev", "hidden_rel_l2_prev",
            "gate_cos_prev", "gate_rel_l2_prev",
            "gate_topk_overlap_prev",
            "routing_weight_cos_prev", "routing_weight_l1_prev"]}
    signals = {}
    # Hidden drift
    h_curr = fresh_data["pre_moe_hidden"][step][layer][batch_idx, token_idx].float()
    h_prev = fresh_data["pre_moe_hidden"][step - 1][layer][batch_idx, token_idx].float()
    signals["hidden_cos_prev"] = F.cosine_similarity(h_curr.unsqueeze(0), h_prev.unsqueeze(0)).item()
    signals["hidden_rel_l2_prev"] = (h_curr - h_prev).norm().item() / (h_curr.norm().item() + 1e-8)
    # Gate logits drift
    g_curr = fresh_data["gate_logits_data"][step][layer][batch_idx, token_idx].float()
    g_prev = fresh_data["gate_logits_data"][step - 1][layer][batch_idx, token_idx].float()
    signals["gate_cos_prev"] = F.cosine_similarity(g_curr.unsqueeze(0), g_prev.unsqueeze(0)).item()
    signals["gate_rel_l2_prev"] = (g_curr - g_prev).norm().item() / (g_curr.norm().item() + 1e-8)
    # Top-k overlap
    tk_curr = set(fresh_data["topk_indices"][step][layer][batch_idx, token_idx].tolist())
    tk_prev = set(fresh_data["topk_indices"][step - 1][layer][batch_idx, token_idx].tolist())
    signals["gate_topk_overlap_prev"] = len(tk_curr & tk_prev) / TOP_K
    # Routing weight drift
    w_curr = fresh_data["topk_weights"][step][layer][batch_idx, token_idx].float()
    w_prev = fresh_data["topk_weights"][step - 1][layer][batch_idx, token_idx].float()
    signals["routing_weight_cos_prev"] = F.cosine_similarity(w_curr.unsqueeze(0), w_prev.unsqueeze(0)).item()
    signals["routing_weight_l1_prev"] = (w_curr - w_prev).abs().sum().item()
    return signals


# ============================================================
# KV cache capture via generation hook
# ============================================================
def capture_kv_cache(dllm, input_ids, target_block_idx=1):
    """Run a real generation and capture KV cache + block info at target block start.

    Returns dict with: kv_data, block_start, block_end, pos_ids, replace_position
    """
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner

    orig_decode = BlockDiffusionRunner.decode
    captured = {}

    def capturing_decode(self_runner, model, decoder, x, kv_cache, block, block_loc,
                         block_id, pos_ids, attn_mask, block_length=32,
                         cross_block_attn_mask=None):
        gen_block_idx = captured.get("_block_counter", 0)
        if gen_block_idx == target_block_idx:
            # Capture KV cache state at start of this block
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
# Intervention engine (with KV cache)
# ============================================================
class CounterfactualRunner:
    """Runs single-site counterfactual interventions with KV cache."""

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
        """Create a fresh KVCache object from captured data."""
        from dinfer.decoding.utils import KVCache
        kv = KVCache(self.kv_data.clone(), backend='vllm')
        return kv

    def _run_forward(self, block_tokens, intervention=None):
        """Run one model forward with KV cache.

        Args:
            block_tokens: [batch, block_len] token IDs
            intervention: None for baseline, or dict with
                {layer, token_idx, batch_idx, cached_routed}
        Returns:
            logits [batch, block_len, vocab], intervention_results dict
        """
        kv_cache = self._make_kv_cache()
        past_kv, replace_pos = kv_cache, self.replace_position

        intervention_results = {}
        orig_forwards = {}

        if intervention is not None:
            target_layer = intervention["layer"]
            target_tok = intervention["token_idx"]
            target_batch = intervention["batch_idx"]
            cached_routed = intervention["cached_routed"]

            for moe_idx, (abs_idx, decoder_layer) in enumerate(self.moe_layers):
                moe_block = decoder_layer.mlp
                orig_forwards[moe_idx] = moe_block.forward

                def make_hook(moe_mod, tgt_layer, tgt_tok, tgt_batch, cached_val,
                              results, mi, total):
                    def hooked_forward(hidden_states):
                        bsz, seq_len, h = hidden_states.shape
                        hs_flat = hidden_states.view(-1, h)
                        shared_res = moe_mod.shared_experts(hidden_states)
                        router_logits = moe_mod.gate.get_logits(hs_flat)
                        routed_y = moe_mod.experts.forward_impl(
                            hidden_states=hs_flat, router_logits=router_logits)
                        routed_y = routed_y.view(bsz, seq_len, h)

                        if mi == tgt_layer:
                            fresh_r = routed_y[tgt_batch, tgt_tok].detach().float()
                            shared_t = shared_res[tgt_batch, tgt_tok].detach().float()
                            fresh_moe = fresh_r + shared_t
                            interv_moe = cached_val.float() + shared_t
                            results["local_moe_cos"] = F.cosine_similarity(
                                fresh_moe.unsqueeze(0), interv_moe.unsqueeze(0)).item()
                            results["local_moe_rel_l2"] = (
                                (fresh_moe - interv_moe).norm().item() /
                                (fresh_moe.norm().item() + 1e-8))
                            routed_y = routed_y.clone()
                            routed_y[tgt_batch, tgt_tok] = cached_val

                        out = (routed_y + shared_res
                               if moe_mod.config.num_shared_experts is not None
                               else routed_y)
                        return out
                    return hooked_forward

                moe_block.forward = make_hook(
                    moe_block, target_layer, target_tok, target_batch,
                    cached_routed, intervention_results, moe_idx, len(self.moe_layers))

        try:
            with torch.inference_mode():
                output = self.model(
                    block_tokens.to(self.device),
                    position_ids=self.block_pos_ids.to(self.device),
                    use_cache=True,
                    past_key_values=past_kv,
                    replace_position=self.replace_position,
                )
            logits = output.logits
        finally:
            if intervention is not None:
                for moe_idx, (abs_idx, decoder_layer) in enumerate(self.moe_layers):
                    decoder_layer.mlp.forward = orig_forwards[moe_idx]

        return logits, intervention_results

    def run_step_interventions(self, step, interventions_for_step):
        """Run baseline + all interventions for one step.

        Args:
            step: iteration index
            interventions_for_step: list of (layer, token_idx, batch_idx) tuples

        Returns:
            list of result dicts
        """
        block_tokens = self.fresh_data["token_ids"][step]

        # Baseline forward (shared across all interventions at this step)
        baseline_logits, _ = self._run_forward(block_tokens)
        baseline_logits_cpu = baseline_logits.float().cpu()

        results = []
        for layer, token_idx, batch_idx in interventions_for_step:
            cached_routed = self.fresh_data["routed_output"][step - 1][layer][
                batch_idx, token_idx].to(self.device)

            interv_logits, interv_results = self._run_forward(
                block_tokens,
                intervention={
                    "layer": layer,
                    "token_idx": token_idx,
                    "batch_idx": batch_idx,
                    "cached_routed": cached_routed,
                })

            # Compare
            bl = baseline_logits_cpu[batch_idx, token_idx]
            il = interv_logits[batch_idx, token_idx].float().cpu()

            bl_probs = F.softmax(bl, dim=-1)
            il_probs = F.softmax(il, dim=-1)

            kl = F.kl_div(il_probs.log().unsqueeze(0), bl_probs.unsqueeze(0),
                          reduction='batchmean', log_target=False).item()
            top1_changed = (bl_probs.argmax().item() != il_probs.argmax().item())
            conf_shift = il_probs.max().item() - bl_probs.max().item()

            results.append({
                "local_moe_cos": interv_results.get("local_moe_cos", 1.0),
                "local_moe_rel_l2": interv_results.get("local_moe_rel_l2", 0.0),
                "final_logits_kl": kl,
                "final_top1_changed": top1_changed,
                "confidence_shift": conf_shift,
            })
        return results


# ============================================================
# Sampling
# ============================================================
def generate_full_coverage_samples(fresh_data, all_layers, all_steps,
                                    tokens_per_group=2, batch_indices=None):
    """Generate stratified samples covering ALL layers × ALL steps."""
    if batch_indices is None:
        batch_indices = [0, 1]
    mask_state = fresh_data["mask_state"]
    block_len = mask_state[0].shape[1]

    samples_by_step = defaultdict(list)
    for step in all_steps:
        for layer in all_layers:
            for batch_idx in batch_indices:
                groups = {"mask": [], "newly_decoded": [],
                          "stable_decoded": [], "highly_stable": []}
                for tok_idx in range(block_len):
                    state, _ = classify_token_state(fresh_data, step, tok_idx, batch_idx)
                    groups[state].append(tok_idx)
                for state, tok_list in groups.items():
                    if not tok_list:
                        continue
                    n = min(tokens_per_group, len(tok_list))
                    stride = max(1, len(tok_list) // n)
                    selected = tok_list[::stride][:n]
                    for tok_idx in selected:
                        samples_by_step[step].append((layer, tok_idx, batch_idx))

    return samples_by_step


# ============================================================
# Main
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
    print("Loading fresh run data...", flush=True)
    data_path = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction" / "full_fresh_run_data.pt"
    fresh_data = torch.load(data_path, map_location="cpu")
    n_iters = len(fresh_data["mask_state"])
    n_layers = len(fresh_data["routed_output"][0])
    print(f"  {n_iters} iterations, {n_layers} MoE layers")

    # Full coverage: all layers, all valid steps
    ALL_LAYERS = list(range(n_layers))  # 0..18
    ALL_STEPS = list(range(1, n_iters))  # 1..29 (skip step 0)
    BATCH_INDICES = [0, 1]
    TOKENS_PER_GROUP = 2

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

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        # Warmup
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
        print("\nCapturing KV cache from real generation...", flush=True)
        kv_info = capture_kv_cache(dllm, input_ids, target_block_idx=1)
        print(f"  Block: {kv_info['block_start']}:{kv_info['block_end']}")
        print(f"  KV shape: {kv_info['kv_data'].shape}")
        print(f"  Replace position: {kv_info['replace_position']}")

        # --- Generate samples ---
        samples_by_step = generate_full_coverage_samples(
            fresh_data, ALL_LAYERS, ALL_STEPS,
            tokens_per_group=TOKENS_PER_GROUP, batch_indices=BATCH_INDICES)
        total_samples = sum(len(v) for v in samples_by_step.values())
        print(f"\nGenerated {total_samples} intervention samples across {len(samples_by_step)} steps")

        # Count by token state
        state_counts = defaultdict(int)
        for step, interventions in samples_by_step.items():
            for layer, tok_idx, batch_idx in interventions:
                state, _ = classify_token_state(fresh_data, step, tok_idx, batch_idx)
                state_counts[state] += 1
        print(f"  By state: {dict(state_counts)}")

        # --- Run interventions ---
        print(f"\n{'='*80}")
        print(f"Running counterfactual interventions (with KV cache)")
        print(f"  {total_samples} samples, {len(ALL_LAYERS)} layers × {len(ALL_STEPS)} steps")
        print(f"{'='*80}")

        runner = CounterfactualRunner(model, fresh_data, kv_info, device=DEVICE)

        all_results = []
        done = 0
        t0 = time.time()

        for step in sorted(samples_by_step.keys()):
            interventions = samples_by_step[step]
            step_results = runner.run_step_interventions(step, interventions)

            for (layer, tok_idx, batch_idx), interv in zip(interventions, step_results):
                state, stable_len = classify_token_state(fresh_data, step, tok_idx, batch_idx)
                proxy = compute_proxy_signals(fresh_data, step, layer, tok_idx, batch_idx)
                li = fresh_data["logits_info"][step]
                confidence = li["confidence"][batch_idx, tok_idx].item()
                margin = li["margin"][batch_idx, tok_idx].item()
                token_id = fresh_data["token_ids"][step][batch_idx, tok_idx].item()
                mask_ratio = fresh_data["mask_state"][step][batch_idx].float().mean().item()

                # Risk label
                if interv["final_top1_changed"] or interv["final_logits_kl"] >= 1e-2:
                    risk = "unsafe"
                elif interv["final_logits_kl"] >= 1e-3:
                    risk = "borderline"
                else:
                    risk = "safe"

                sample = InterventionSample(
                    step=step, layer=layer, token_idx=tok_idx, batch_idx=batch_idx,
                    token_state=state, token_id=token_id,
                    token_confidence=confidence, token_margin=margin,
                    stable_len=stable_len, mask_ratio=mask_ratio,
                    **proxy, **interv, risk_label=risk,
                )
                all_results.append(sample)

            done += len(interventions)
            elapsed = time.time() - t0
            if done % 200 == 0 or done == total_samples:
                safe_n = sum(1 for r in all_results if r.risk_label == "safe")
                unsafe_n = sum(1 for r in all_results if r.risk_label == "unsafe")
                border_n = sum(1 for r in all_results if r.risk_label == "borderline")
                print(f"  [{done}/{total_samples}] step={step} {elapsed:.1f}s — "
                      f"safe={safe_n} border={border_n} unsafe={unsafe_n}")

        elapsed_total = time.time() - t0
        print(f"\nTotal: {len(all_results)} samples in {elapsed_total:.1f}s "
              f"({elapsed_total/max(len(all_results),1)*1000:.1f}ms/sample)")

        # --- Summary ---
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}")
        total = len(all_results)

        # By risk label
        label_counts = defaultdict(int)
        for r in all_results:
            label_counts[r.risk_label] += 1
        for label in ["safe", "borderline", "unsafe"]:
            n = label_counts[label]
            pct = n / total * 100 if total > 0 else 0
            print(f"  {label:>12s}: {n:>5d} ({pct:.1f}%)")

        # By token state × risk
        print(f"\n  Token state × Risk:")
        state_risk = defaultdict(lambda: defaultdict(int))
        for r in all_results:
            state_risk[r.token_state][r.risk_label] += 1
        print(f"  {'State':<18s} {'safe':>6s} {'border':>8s} {'unsafe':>8s} {'total':>6s}")
        for state in ["mask", "newly_decoded", "stable_decoded", "highly_stable"]:
            s = state_risk[state]
            t = s["safe"] + s["borderline"] + s["unsafe"]
            if t > 0:
                print(f"  {state:<18s} {s['safe']:>6d} {s['borderline']:>8d} {s['unsafe']:>8d} {t:>6d}")

        # By layer × risk
        print(f"\n  Layer × Risk:")
        layer_risk = defaultdict(lambda: defaultdict(int))
        for r in all_results:
            layer_risk[r.layer][r.risk_label] += 1
        print(f"  {'Layer':<8s} {'safe':>6s} {'border':>8s} {'unsafe':>8s} {'safe%':>7s}")
        for layer in sorted(layer_risk.keys()):
            s = layer_risk[layer]
            t = s["safe"] + s["borderline"] + s["unsafe"]
            safe_pct = s["safe"] / t * 100 if t > 0 else 0
            print(f"  {layer:<8d} {s['safe']:>6d} {s['borderline']:>8d} {s['unsafe']:>8d} {safe_pct:>6.1f}%")

        # By step group × risk
        print(f"\n  Step group × Risk:")
        step_risk = {"early(1-5)": defaultdict(int), "mid(6-10)": defaultdict(int),
                     "late(11-20)": defaultdict(int), "final(21+)": defaultdict(int)}
        for r in all_results:
            if r.step <= 5: g = "early(1-5)"
            elif r.step <= 10: g = "mid(6-10)"
            elif r.step <= 20: g = "late(11-20)"
            else: g = "final(21+)"
            step_risk[g][r.risk_label] += 1
        print(f"  {'Group':<14s} {'safe':>6s} {'border':>8s} {'unsafe':>8s} {'safe%':>7s}")
        for g in ["early(1-5)", "mid(6-10)", "late(11-20)", "final(21+)"]:
            s = step_risk[g]
            t = s["safe"] + s["borderline"] + s["unsafe"]
            safe_pct = s["safe"] / t * 100 if t > 0 else 0
            print(f"  {g:<14s} {s['safe']:>6d} {s['borderline']:>8d} {s['unsafe']:>8d} {safe_pct:>6.1f}%")

        # --- Save ---
        results_dir = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction"
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / "counterfactual_kvcache_full.json"
        with open(save_path, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\n  Saved {len(all_results)} samples to {save_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
