#!/usr/bin/env python3
"""
v0.1.14.5.1 — Contamination Propagation Detection

Instruments the combination-reuse forward to capture per-layer contamination
signals. The goal: determine whether layer-level signals (gate_cos, topk_overlap,
etc.) can detect error accumulation during multi-site reuse, enabling adaptive
"stop reusing from this layer onward" decisions.

Two-level decision architecture:
  Level 1 (Token gate):  token_confidence / token_margin → admit or reject
  Level 2 (Layer gate):  gate_cos / topk_overlap / weight_cos / hidden_cos /
                          shared_cos → per-layer reuse-or-fresh

Experiment design:
  For each (step, batch, rule), run a full-deployment combination forward,
  but instrument every MoE layer's hook to record:
    1. Live gate logits    (from corrupted hidden state)
    2. Live topk indices   (from corrupted hidden state)
    3. Live topk weights   (from corrupted hidden state)
    4. Live pre-MoE hidden (the corrupted input)
    5. Live shared output  (fresh-computed on corrupted input)

  Then compare each against the clean fresh-run values to produce:
    - gate_cos_vs_fresh[layer]
    - topk_overlap_vs_fresh[layer]
    - weight_cos_vs_fresh[layer]
    - hidden_cos_vs_fresh[layer]
    - shared_cos_vs_fresh[layer]

  Cross-reference with final logits KL to see which signals predict
  the contamination blowup point.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

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
# Token classification & proxy (reused)
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


def compute_proxy_for_token(fresh_data, step, layer, token_idx, batch_idx):
    if step == 0:
        return {"gate_topk_overlap_prev": 0.0, "gate_cos_prev": 0.0,
                "token_confidence": 0.0, "token_margin": 0.0}
    signals = {}
    tk_curr = set(fresh_data["topk_indices"][step][layer][batch_idx, token_idx].tolist())
    tk_prev = set(fresh_data["topk_indices"][step - 1][layer][batch_idx, token_idx].tolist())
    signals["gate_topk_overlap_prev"] = len(tk_curr & tk_prev) / TOP_K
    g_curr = fresh_data["gate_logits_data"][step][layer][batch_idx, token_idx].float()
    g_prev = fresh_data["gate_logits_data"][step - 1][layer][batch_idx, token_idx].float()
    signals["gate_cos_prev"] = F.cosine_similarity(
        g_curr.unsqueeze(0), g_prev.unsqueeze(0)).item()
    li = fresh_data["logits_info"][step]
    signals["token_confidence"] = li["confidence"][batch_idx, token_idx].item()
    signals["token_margin"] = li["margin"][batch_idx, token_idx].item()
    return signals


# Proxy rules (same as v0.1.14.5)
RULES = {
    "R1_conservative": {
        "desc": "highly_stable + topk>=1.0",
        "check": lambda state, stable_len, proxy: (
            state == "highly_stable" and
            proxy["gate_topk_overlap_prev"] >= 1.0),
    },
    "R2_moderate": {
        "desc": "highly_stable + topk>=0.875",
        "check": lambda state, stable_len, proxy: (
            state == "highly_stable" and
            proxy["gate_topk_overlap_prev"] >= 0.875),
    },
    "R3_aggressive": {
        "desc": "stable+ + gate_cos>=0.999",
        "check": lambda state, stable_len, proxy: (
            state in ("stable_decoded", "highly_stable") and
            proxy["gate_cos_prev"] >= 0.999),
    },
    "R4_margin": {
        "desc": "token_margin > 0.99",
        "check": lambda state, stable_len, proxy: (
            proxy["token_margin"] > 0.99),
    },
}


def find_qualifying_pairs(fresh_data, step, batch_idx, rule_name):
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
# Instrumented Combination Runner — captures per-layer signals
# ============================================================
class InstrumentedCombinationRunner:
    """Runs combination reuse AND records per-layer contamination signals."""

    def __init__(self, model, fresh_data, kv_info, device="cuda:0"):
        self.model = model
        self.fresh_data = fresh_data
        self.device = device
        self.kv_data = kv_info["kv_data"]
        self.block_start = kv_info["block_start"]
        self.block_end = kv_info["block_end"]
        self.replace_position = kv_info["replace_position"]
        self.block_pos_ids = kv_info["pos_ids"]

        layers = model.model.layers
        self.moe_layers = [(idx, l) for idx, l in enumerate(layers)
                           if hasattr(l, 'mlp') and hasattr(l.mlp, 'gate')]

    def _make_kv_cache(self):
        from dinfer.decoding.utils import KVCache
        return KVCache(self.kv_data.clone(), backend='vllm')

    def run_instrumented(self, step, pairs, batch_idx):
        """Run combination forward with full per-layer signal capture.

        Args:
            step: denoising iteration
            pairs: list of (layer, token_idx) to reuse
            batch_idx: which batch element

        Returns:
            dict with per-layer signals and final metrics
        """
        block_tokens = self.fresh_data["token_ids"][step]

        # Build reuse map
        reuse_set = defaultdict(set)  # layer -> set of token_idx
        for layer, tok_idx in pairs:
            reuse_set[layer].add(tok_idx)

        # Per-layer signal storage
        layer_signals = {}  # layer_idx -> dict of signals

        # ---- Baseline forward (clean, no intervention) ----
        kv_cache_bl = self._make_kv_cache()
        with torch.inference_mode():
            bl_output = self.model(
                block_tokens.to(self.device),
                position_ids=self.block_pos_ids.to(self.device),
                use_cache=True,
                past_key_values=kv_cache_bl,
                replace_position=self.replace_position,
            )
        baseline_logits = bl_output.logits.float().cpu()

        # ---- Instrumented combination forward ----
        kv_cache = self._make_kv_cache()
        orig_forwards = {}

        for moe_idx, (abs_idx, decoder_layer) in enumerate(self.moe_layers):
            moe_block = decoder_layer.mlp
            orig_forwards[moe_idx] = moe_block.forward

            def make_hook(moe_mod, mi, rset, fd, st, bi, lsigs):
                def hooked_forward(hidden_states):
                    bsz, seq_len, h = hidden_states.shape
                    hs_flat = hidden_states.view(-1, h)

                    # --- Record live pre-MoE hidden ---
                    live_hidden = hidden_states[bi].detach().float().cpu()  # [block_len, h]

                    # --- Shared expert (always fresh) ---
                    shared_res = moe_mod.shared_experts(hidden_states)
                    live_shared = shared_res[bi].detach().float().cpu()  # [block_len, h]

                    # --- Gate ---
                    router_logits = moe_mod.gate.get_logits(hs_flat)
                    topk_idx, topk_weight, full_gate_logits = moe_mod.gate(hs_flat)

                    # Reshape gate outputs to [batch, seq, ...]
                    live_gate = full_gate_logits.view(bsz, seq_len, -1)[bi].detach().float().cpu()
                    live_topk_idx = topk_idx.view(bsz, seq_len, -1)[bi].detach().cpu()
                    live_topk_w = topk_weight.view(bsz, seq_len, -1)[bi].detach().float().cpu()

                    # --- Routed experts ---
                    routed_y = moe_mod.experts.forward_impl(
                        hidden_states=hs_flat, router_logits=router_logits)
                    routed_y = routed_y.view(bsz, seq_len, h)

                    # --- Apply cached routed outputs ---
                    tokens_in_layer = rset.get(mi, set())
                    if tokens_in_layer:
                        routed_y = routed_y.clone()
                        for t_idx in tokens_in_layer:
                            cached_val = fd["routed_output"][st - 1][mi][bi, t_idx].to(
                                hidden_states.device)
                            routed_y[bi, t_idx] = cached_val

                    out = (routed_y + shared_res
                           if moe_mod.config.num_shared_experts is not None
                           else routed_y)

                    # --- Compute contamination signals vs fresh run ---
                    fresh_hidden = fd["pre_moe_hidden"][st][mi][bi].float()   # [block_len, h]
                    fresh_gate = fd["gate_logits_data"][st][mi][bi].float()   # [block_len, E]
                    fresh_topk_idx = fd["topk_indices"][st][mi][bi]           # [block_len, k]
                    fresh_topk_w = fd["topk_weights"][st][mi][bi].float()     # [block_len, k]
                    fresh_shared = fd["shared_output"][st][mi][bi].float()    # [block_len, h]

                    block_len_actual = live_hidden.shape[0]
                    sig = {
                        "n_reused_tokens": len(tokens_in_layer),
                        # Per-token signals (averaged over block)
                        "per_token": {},
                    }

                    # Compute per-token signals
                    gate_cos_list = []
                    topk_ol_list = []
                    weight_cos_list = []
                    hidden_cos_list = []
                    shared_cos_list = []

                    # Also track separately for reused vs non-reused tokens
                    reused_gate_cos = []
                    reused_hidden_cos = []
                    non_reused_gate_cos = []
                    non_reused_hidden_cos = []

                    for t in range(block_len_actual):
                        # gate_cos
                        gc = F.cosine_similarity(
                            live_gate[t].unsqueeze(0),
                            fresh_gate[t].unsqueeze(0)).item()
                        gate_cos_list.append(gc)

                        # topk_overlap
                        live_set = set(live_topk_idx[t].tolist())
                        fresh_set = set(fresh_topk_idx[t].tolist())
                        ol = len(live_set & fresh_set) / TOP_K
                        topk_ol_list.append(ol)

                        # routing_weight_cos
                        wc = F.cosine_similarity(
                            live_topk_w[t].unsqueeze(0),
                            fresh_topk_w[t].unsqueeze(0)).item()
                        weight_cos_list.append(wc)

                        # hidden_cos
                        hc = F.cosine_similarity(
                            live_hidden[t].unsqueeze(0),
                            fresh_hidden[t].unsqueeze(0)).item()
                        hidden_cos_list.append(hc)

                        # shared_output_cos
                        sc = F.cosine_similarity(
                            live_shared[t].unsqueeze(0),
                            fresh_shared[t].unsqueeze(0)).item()
                        shared_cos_list.append(sc)

                        # Track by reuse status
                        if t in tokens_in_layer:
                            reused_gate_cos.append(gc)
                            reused_hidden_cos.append(hc)
                        else:
                            non_reused_gate_cos.append(gc)
                            non_reused_hidden_cos.append(hc)

                    sig["gate_cos_vs_fresh"] = {
                        "mean": sum(gate_cos_list) / len(gate_cos_list),
                        "min": min(gate_cos_list),
                    }
                    sig["topk_overlap_vs_fresh"] = {
                        "mean": sum(topk_ol_list) / len(topk_ol_list),
                        "min": min(topk_ol_list),
                    }
                    sig["weight_cos_vs_fresh"] = {
                        "mean": sum(weight_cos_list) / len(weight_cos_list),
                        "min": min(weight_cos_list),
                    }
                    sig["hidden_cos_vs_fresh"] = {
                        "mean": sum(hidden_cos_list) / len(hidden_cos_list),
                        "min": min(hidden_cos_list),
                    }
                    sig["shared_cos_vs_fresh"] = {
                        "mean": sum(shared_cos_list) / len(shared_cos_list),
                        "min": min(shared_cos_list),
                    }

                    # Reused vs non-reused breakdown
                    if reused_gate_cos:
                        sig["reused_gate_cos_mean"] = sum(reused_gate_cos) / len(reused_gate_cos)
                        sig["reused_hidden_cos_mean"] = sum(reused_hidden_cos) / len(reused_hidden_cos)
                    if non_reused_gate_cos:
                        sig["non_reused_gate_cos_mean"] = sum(non_reused_gate_cos) / len(non_reused_gate_cos)
                        sig["non_reused_hidden_cos_mean"] = sum(non_reused_hidden_cos) / len(non_reused_hidden_cos)

                    lsigs[mi] = sig
                    return out
                return hooked_forward

            moe_block.forward = make_hook(
                moe_block, moe_idx, dict(reuse_set), self.fresh_data,
                step, batch_idx, layer_signals)

        try:
            with torch.inference_mode():
                output = self.model(
                    block_tokens.to(self.device),
                    position_ids=self.block_pos_ids.to(self.device),
                    use_cache=True,
                    past_key_values=kv_cache,
                    replace_position=self.replace_position,
                )
            interv_logits = output.logits.float().cpu()
        finally:
            for moe_idx, (abs_idx, decoder_layer) in enumerate(self.moe_layers):
                decoder_layer.mlp.forward = orig_forwards[moe_idx]

        # ---- Final logits comparison ----
        bl = baseline_logits[batch_idx]  # [block_len, vocab]
        il = interv_logits[batch_idx]

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

        mean_kl = sum(per_token_kl) / block_len
        max_kl = max(per_token_kl)
        n_top1_changed = sum(per_token_top1_changed)
        n_unsafe = sum(1 for kl in per_token_kl if kl >= 1e-2)

        return {
            "step": step,
            "batch_idx": batch_idx,
            "n_total_pairs": len(pairs),
            "n_unique_tokens": len(set(t for _, t in pairs)),
            "mean_kl": mean_kl,
            "max_kl": max_kl,
            "n_top1_changed": n_top1_changed,
            "top1_changed_rate": n_top1_changed / block_len,
            "n_unsafe": n_unsafe,
            "layer_signals": layer_signals,
        }


# ============================================================
# KV cache capture (reused)
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
                  f"KV shape={past_kv._data.shape}", flush=True)
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
    assert "kv_data" in captured
    return captured


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

    print("=" * 80)
    print("v0.1.14.5.1 — Contamination Propagation Detection")
    print("=" * 80)

    # --- Load data ---
    print("\nLoading fresh run data...", flush=True)
    data_path = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction" / "full_fresh_run_data.pt"
    fresh_data = torch.load(data_path, map_location="cpu")
    n_iters = len(fresh_data["mask_state"])
    n_layers = len(fresh_data["routed_output"][0])
    print(f"  {n_iters} iterations, {n_layers} MoE layers")

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
        print(f"  Input shape: {input_ids.shape}")

        # KV cache
        print("\nCapturing KV cache...", flush=True)
        kv_info = capture_kv_cache(dllm, input_ids, target_block_idx=1)
        print(f"  Block: {kv_info['block_start']}:{kv_info['block_end']}")

        runner = InstrumentedCombinationRunner(model, fresh_data, kv_info, device=DEVICE)

        # ---- Experiment ----
        SAMPLE_STEPS = [2, 5, 8, 11, 15, 19, 23, 27]
        SAMPLE_STEPS = [s for s in SAMPLE_STEPS if s < n_iters]
        BATCH_INDICES = [0, 1]

        print(f"\n{'='*80}")
        print(f"CONTAMINATION PROPAGATION EXPERIMENT")
        print(f"  Steps: {SAMPLE_STEPS}, Batch: {BATCH_INDICES}")
        print(f"{'='*80}")

        all_results = {}
        t0 = time.time()

        for rule_name in RULES:
            print(f"\n--- {rule_name}: {RULES[rule_name]['desc']} ---")
            rule_runs = []

            for step in SAMPLE_STEPS:
                for batch_idx in BATCH_INDICES:
                    pairs = find_qualifying_pairs(fresh_data, step, batch_idx, rule_name)
                    if len(pairs) < 2:
                        continue

                    result = runner.run_instrumented(step, pairs, batch_idx)

                    # Print per-layer signal curve
                    print(f"\n  step={step} batch={batch_idx}: "
                          f"{result['n_total_pairs']} pairs, "
                          f"mean_kl={result['mean_kl']:.6f}, "
                          f"top1_changed={result['n_top1_changed']}")

                    print(f"  {'Layer':>5s} {'#Reuse':>6s} "
                          f"{'HidCos':>7s} {'GateCos':>7s} {'TopkOl':>7s} "
                          f"{'WgtCos':>7s} {'ShrCos':>7s}")

                    for li in range(NUM_MOE_LAYERS):
                        if li not in result["layer_signals"]:
                            continue
                        ls = result["layer_signals"][li]
                        print(f"  L{li:>3d}  {ls['n_reused_tokens']:>6d} "
                              f"{ls['hidden_cos_vs_fresh']['mean']:>7.4f} "
                              f"{ls['gate_cos_vs_fresh']['mean']:>7.4f} "
                              f"{ls['topk_overlap_vs_fresh']['mean']:>7.4f} "
                              f"{ls['weight_cos_vs_fresh']['mean']:>7.4f} "
                              f"{ls['shared_cos_vs_fresh']['mean']:>7.4f}")

                    rule_runs.append(result)

            all_results[rule_name] = rule_runs

        elapsed = time.time() - t0

        # ============================================================
        # CROSS-RULE LAYER SIGNAL AGGREGATION
        # ============================================================
        print(f"\n{'='*80}")
        print(f"AGGREGATED LAYER-WISE CONTAMINATION CURVES")
        print(f"{'='*80}")

        for rule_name in RULES:
            runs = all_results[rule_name]
            if not runs:
                continue

            print(f"\n  {rule_name} ({len(runs)} runs):")
            print(f"  {'Layer':>5s} {'HidCos':>8s} {'GateCos':>8s} {'TopkOl':>8s} "
                  f"{'WgtCos':>8s} {'ShrCos':>8s}  "
                  f"{'Reused_GC':>9s} {'NonReu_GC':>9s}")

            for li in range(NUM_MOE_LAYERS):
                hc_vals, gc_vals, ol_vals, wc_vals, sc_vals = [], [], [], [], []
                r_gc_vals, nr_gc_vals = [], []
                r_hc_vals, nr_hc_vals = [], []

                for run in runs:
                    if li not in run["layer_signals"]:
                        continue
                    ls = run["layer_signals"][li]
                    hc_vals.append(ls["hidden_cos_vs_fresh"]["mean"])
                    gc_vals.append(ls["gate_cos_vs_fresh"]["mean"])
                    ol_vals.append(ls["topk_overlap_vs_fresh"]["mean"])
                    wc_vals.append(ls["weight_cos_vs_fresh"]["mean"])
                    sc_vals.append(ls["shared_cos_vs_fresh"]["mean"])
                    if "reused_gate_cos_mean" in ls:
                        r_gc_vals.append(ls["reused_gate_cos_mean"])
                    if "non_reused_gate_cos_mean" in ls:
                        nr_gc_vals.append(ls["non_reused_gate_cos_mean"])

                if not hc_vals:
                    continue

                avg = lambda v: sum(v)/len(v) if v else 0
                print(f"  L{li:>3d}  {avg(hc_vals):>8.5f} {avg(gc_vals):>8.5f} "
                      f"{avg(ol_vals):>8.5f} {avg(wc_vals):>8.5f} {avg(sc_vals):>8.5f}  "
                      f"{avg(r_gc_vals):>9.5f} {avg(nr_gc_vals):>9.5f}")

        # ============================================================
        # SIGNAL CORRELATION WITH FINAL KL
        # ============================================================
        print(f"\n{'='*80}")
        print(f"SIGNAL vs FINAL KL CORRELATION (per-run)")
        print(f"{'='*80}")

        for rule_name in RULES:
            runs = all_results[rule_name]
            if not runs:
                continue

            print(f"\n  {rule_name}:")
            # For each run, compute min signal across layers and correlate with mean_kl
            signal_names = ["hidden_cos_vs_fresh", "gate_cos_vs_fresh",
                           "topk_overlap_vs_fresh", "weight_cos_vs_fresh",
                           "shared_cos_vs_fresh"]

            for sig_name in signal_names:
                pairs_data = []  # (min_signal, mean_kl)
                for run in runs:
                    min_sig = 1.0
                    for li in range(NUM_MOE_LAYERS):
                        if li in run["layer_signals"]:
                            val = run["layer_signals"][li][sig_name]["mean"]
                            min_sig = min(min_sig, val)
                    pairs_data.append((min_sig, run["mean_kl"]))

                # Spearman-like: just show if lower signal → higher KL
                pairs_data.sort(key=lambda x: x[0])
                n = len(pairs_data)
                q1 = pairs_data[:n//3]
                q3 = pairs_data[2*n//3:]
                avg_kl_low_sig = sum(p[1] for p in q1) / len(q1) if q1 else 0
                avg_kl_high_sig = sum(p[1] for p in q3) / len(q3) if q3 else 0
                avg_sig_low = sum(p[0] for p in q1) / len(q1) if q1 else 0
                avg_sig_high = sum(p[0] for p in q3) / len(q3) if q3 else 0

                short_name = sig_name.replace("_vs_fresh", "").replace("_cos", "Cos")
                print(f"    {short_name:<16s}: "
                      f"low_sig({avg_sig_low:.4f})→kl={avg_kl_low_sig:.6f}  "
                      f"high_sig({avg_sig_high:.4f})→kl={avg_kl_high_sig:.6f}  "
                      f"ratio={avg_kl_low_sig/(avg_kl_high_sig+1e-10):.1f}x")

        # ============================================================
        # LAYER TIPPING POINT ANALYSIS
        # ============================================================
        print(f"\n{'='*80}")
        print(f"LAYER TIPPING POINT: Where does contamination become critical?")
        print(f"{'='*80}")
        print(f"  (Layer where gate_cos first drops below threshold)")

        for rule_name in RULES:
            runs = all_results[rule_name]
            if not runs:
                continue

            thresholds = [0.999, 0.995, 0.99, 0.98, 0.95]
            print(f"\n  {rule_name}:")
            for thresh in thresholds:
                first_breach_layers = []
                for run in runs:
                    breached = False
                    for li in range(NUM_MOE_LAYERS):
                        if li in run["layer_signals"]:
                            gc = run["layer_signals"][li]["gate_cos_vs_fresh"]["mean"]
                            if gc < thresh:
                                first_breach_layers.append(li)
                                breached = True
                                break
                    if not breached:
                        first_breach_layers.append(NUM_MOE_LAYERS)  # never breached

                avg_layer = sum(first_breach_layers) / len(first_breach_layers)
                n_breached = sum(1 for l in first_breach_layers if l < NUM_MOE_LAYERS)
                print(f"    gate_cos < {thresh:.3f}: "
                      f"avg first breach at L{avg_layer:.1f}, "
                      f"{n_breached}/{len(runs)} runs breached")

        print(f"\nTotal time: {elapsed:.1f}s")

        # --- Save ---
        # Convert layer_signals keys to str for JSON
        save_results = {}
        for rule_name, runs in all_results.items():
            save_runs = []
            for run in runs:
                run_copy = {k: v for k, v in run.items() if k != "layer_signals"}
                run_copy["layer_signals"] = {
                    str(k): v for k, v in run["layer_signals"].items()
                }
                save_runs.append(run_copy)
            save_results[rule_name] = save_runs

        results_dir = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction"
        save_path = results_dir / "contamination_propagation_results.json"
        with open(save_path, "w") as f:
            json.dump(save_results, f, indent=2, default=str)
        print(f"Saved to {save_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
