#!/usr/bin/env python3
"""
v0.1.14.5.2 — Layer-Range Sweep: Which layers can safely reuse?

Systematically tests different "fresh range / reuse range" splits to find
the relationship between which layers are allowed to reuse and final quality.

Sweep designs:
  1. Forward sweep: fresh L0..K, reuse L(K+1)..18   (K = 0,1,...,18)
     → Answers: "how many shallow layers must be fresh?"

  2. Reverse sweep: reuse L0..K, fresh L(K+1)..18   (K = 0,1,...,18)
     → Answers: "what if we only reuse shallow layers?"

  3. Window sweep: reuse only a 5-layer window, fresh elsewhere
     → Answers: "which specific layers are safest to reuse?"

For each configuration, also records per-layer contamination signals
(shared_cos, hidden_cos) to see if the signals naturally correlate with
which splits work best — feeding into dynamic policy design.

Uses R4_margin (token_margin > 0.99) as the token-level gate since it has
the strongest single-point safety (0.8% FNR) and produced the clearest
contamination patterns in v0.1.14.5.1.
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
# Token classification (reused)
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


def get_qualifying_tokens(fresh_data, step, batch_idx):
    """Get tokens qualifying under R4_margin (token_margin > 0.99)."""
    li = fresh_data["logits_info"][step]
    block_len = fresh_data["mask_state"][step].shape[1]
    qualifying = []
    for tok_idx in range(block_len):
        margin = li["margin"][batch_idx, tok_idx].item()
        if margin > 0.99:
            qualifying.append(tok_idx)
    return qualifying


# ============================================================
# Layer-Range Runner with contamination signal capture
# ============================================================
class LayerRangeRunner:
    """Runs combination reuse with configurable per-layer reuse/fresh decisions."""

    def __init__(self, model, fresh_data, kv_info, device="cuda:0"):
        self.model = model
        self.fresh_data = fresh_data
        self.device = device
        self.kv_data = kv_info["kv_data"]
        self.replace_position = kv_info["replace_position"]
        self.block_pos_ids = kv_info["pos_ids"]

        layers = model.model.layers
        self.moe_layers = [(idx, l) for idx, l in enumerate(layers)
                           if hasattr(l, 'mlp') and hasattr(l.mlp, 'gate')]

    def _make_kv_cache(self):
        from dinfer.decoding.utils import KVCache
        return KVCache(self.kv_data.clone(), backend='vllm')

    def run_with_layer_mask(self, step, qualifying_tokens, batch_idx,
                            reuse_layers: Set[int]):
        """Run forward where only specified layers reuse cached routed output.

        Args:
            step: denoising iteration
            qualifying_tokens: list of token indices that pass Level-1
            batch_idx: which batch element
            reuse_layers: set of MoE layer indices (0..18) where reuse is allowed

        Returns:
            dict with metrics and per-layer signals
        """
        block_tokens = self.fresh_data["token_ids"][step]
        qt_set = set(qualifying_tokens)
        layer_signals = {}

        # Baseline forward
        kv_bl = self._make_kv_cache()
        with torch.inference_mode():
            bl_out = self.model(
                block_tokens.to(self.device),
                position_ids=self.block_pos_ids.to(self.device),
                use_cache=True, past_key_values=kv_bl,
                replace_position=self.replace_position,
            )
        baseline_logits = bl_out.logits.float().cpu()

        # Intervention forward with layer mask
        kv_int = self._make_kv_cache()
        orig_forwards = {}

        for moe_idx, (abs_idx, decoder_layer) in enumerate(self.moe_layers):
            moe_block = decoder_layer.mlp
            orig_forwards[moe_idx] = moe_block.forward

            def make_hook(moe_mod, mi, rlayers, qts, fd, st, bi, lsigs):
                def hooked_forward(hidden_states):
                    bsz, seq_len, h = hidden_states.shape
                    hs_flat = hidden_states.view(-1, h)

                    # Always compute shared (needed regardless)
                    shared_res = moe_mod.shared_experts(hidden_states)

                    # Always compute gate + routed (to compare)
                    router_logits = moe_mod.gate.get_logits(hs_flat)
                    routed_y = moe_mod.experts.forward_impl(
                        hidden_states=hs_flat, router_logits=router_logits)
                    routed_y = routed_y.view(bsz, seq_len, h)

                    # Record contamination signals
                    live_hidden = hidden_states[bi].detach().float().cpu()
                    live_shared = shared_res[bi].detach().float().cpu()
                    fresh_hidden = fd["pre_moe_hidden"][st][mi][bi].float()
                    fresh_shared = fd["shared_output"][st][mi][bi].float()

                    block_len = live_hidden.shape[0]
                    hc_list, sc_list = [], []
                    for t in range(block_len):
                        hc_list.append(F.cosine_similarity(
                            live_hidden[t].unsqueeze(0),
                            fresh_hidden[t].unsqueeze(0)).item())
                        sc_list.append(F.cosine_similarity(
                            live_shared[t].unsqueeze(0),
                            fresh_shared[t].unsqueeze(0)).item())

                    is_reuse_layer = mi in rlayers
                    n_reused = 0

                    # Apply reuse only if this layer is in the reuse set
                    if is_reuse_layer and qts:
                        routed_y = routed_y.clone()
                        for t_idx in qts:
                            cached_val = fd["routed_output"][st - 1][mi][bi, t_idx].to(
                                hidden_states.device)
                            routed_y[bi, t_idx] = cached_val
                            n_reused += 1

                    out = (routed_y + shared_res
                           if moe_mod.config.num_shared_experts is not None
                           else routed_y)

                    lsigs[mi] = {
                        "is_reuse_layer": is_reuse_layer,
                        "n_reused": n_reused,
                        "hidden_cos_mean": sum(hc_list) / len(hc_list),
                        "hidden_cos_min": min(hc_list),
                        "shared_cos_mean": sum(sc_list) / len(sc_list),
                        "shared_cos_min": min(sc_list),
                    }
                    return out
                return hooked_forward

            moe_block.forward = make_hook(
                moe_block, moe_idx, reuse_layers, qt_set, self.fresh_data,
                step, batch_idx, layer_signals)

        try:
            with torch.inference_mode():
                output = self.model(
                    block_tokens.to(self.device),
                    position_ids=self.block_pos_ids.to(self.device),
                    use_cache=True, past_key_values=kv_int,
                    replace_position=self.replace_position,
                )
            interv_logits = output.logits.float().cpu()
        finally:
            for moe_idx, (abs_idx, decoder_layer) in enumerate(self.moe_layers):
                decoder_layer.mlp.forward = orig_forwards[moe_idx]

        # Compare logits
        bl = baseline_logits[batch_idx]
        il = interv_logits[batch_idx]
        bl_probs = F.softmax(bl, dim=-1)
        il_probs = F.softmax(il, dim=-1)

        block_len = bl.shape[0]
        kl_list, top1_ch_list = [], []
        for t in range(block_len):
            kl = F.kl_div(il_probs[t].log().unsqueeze(0),
                          bl_probs[t].unsqueeze(0),
                          reduction='batchmean', log_target=False).item()
            top1_ch = bl_probs[t].argmax().item() != il_probs[t].argmax().item()
            kl_list.append(kl)
            top1_ch_list.append(top1_ch)

        return {
            "mean_kl": sum(kl_list) / block_len,
            "max_kl": max(kl_list),
            "n_top1_changed": sum(top1_ch_list),
            "top1_rate": sum(top1_ch_list) / block_len,
            "n_unsafe": sum(1 for kl in kl_list if kl >= 1e-2),
            "n_reuse_layers": len(reuse_layers),
            "n_qualifying_tokens": len(qualifying_tokens),
            "total_reuse_pairs": len(reuse_layers) * len(qualifying_tokens),
            "layer_signals": layer_signals,
        }


# ============================================================
# KV cache capture
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
    print("v0.1.14.5.2 — Layer-Range Sweep")
    print("=" * 80)

    # Load data
    print("\nLoading fresh run data...", flush=True)
    data_path = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction" / "full_fresh_run_data.pt"
    fresh_data = torch.load(data_path, map_location="cpu")
    n_iters = len(fresh_data["mask_state"])
    print(f"  {n_iters} iterations, {len(fresh_data['routed_output'][0])} MoE layers")

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
        print(f"  Input shape: {input_ids.shape}")

        # KV cache
        print("\nCapturing KV cache...", flush=True)
        kv_info = capture_kv_cache(dllm, input_ids, target_block_idx=1)

        runner = LayerRangeRunner(model, fresh_data, kv_info, device=DEVICE)

        # Sample steps and batches
        SAMPLE_STEPS = [5, 11, 19, 27]
        SAMPLE_STEPS = [s for s in SAMPLE_STEPS if s < n_iters]
        BATCH_INDICES = [0, 1]

        t0 = time.time()

        # ============================================================
        # Sweep 1: Forward — fresh L0..K, reuse L(K+1)..18
        # ============================================================
        print(f"\n{'='*80}")
        print(f"SWEEP 1: Fresh L0..K, Reuse L(K+1)..18")
        print(f"  'How many shallow layers must be fresh?'")
        print(f"{'='*80}")

        forward_results = {}  # K -> list of run results

        for K in range(-1, NUM_MOE_LAYERS):
            # K=-1: all reuse (no fresh), K=0: fresh L0 only, ...
            # K=18: all fresh (no reuse)
            reuse_layers = set(range(K + 1, NUM_MOE_LAYERS))
            fresh_layers = set(range(0, K + 1))
            label = f"fresh_0..{K}" if K >= 0 else "all_reuse"
            if K == NUM_MOE_LAYERS - 1:
                label = "all_fresh"

            runs = []
            for step in SAMPLE_STEPS:
                for batch_idx in BATCH_INDICES:
                    qt = get_qualifying_tokens(fresh_data, step, batch_idx)
                    if len(qt) < 2:
                        continue
                    result = runner.run_with_layer_mask(
                        step, qt, batch_idx, reuse_layers)
                    result["step"] = step
                    result["batch_idx"] = batch_idx
                    runs.append(result)

            if not runs:
                continue

            avg_kl = sum(r["mean_kl"] for r in runs) / len(runs)
            max_kl = max(r["max_kl"] for r in runs)
            avg_top1 = sum(r["top1_rate"] for r in runs) / len(runs)
            avg_reuse_pairs = sum(r["total_reuse_pairs"] for r in runs) / len(runs)
            n_unsafe_runs = sum(1 for r in runs if r["n_unsafe"] > 0)

            print(f"  K={K:>2d} ({label:>16s}): "
                  f"reuse_layers={len(reuse_layers):>2d}  "
                  f"avg_pairs={avg_reuse_pairs:>5.0f}  "
                  f"avg_kl={avg_kl:>10.6f}  max_kl={max_kl:>10.6f}  "
                  f"top1%={avg_top1*100:>5.2f}  "
                  f"unsafe_runs={n_unsafe_runs}/{len(runs)}")

            forward_results[K] = {
                "reuse_layer_count": len(reuse_layers),
                "avg_kl": avg_kl,
                "max_kl": max_kl,
                "avg_top1_rate": avg_top1,
                "unsafe_run_rate": n_unsafe_runs / len(runs),
                "avg_reuse_pairs": avg_reuse_pairs,
                "n_runs": len(runs),
                "runs": runs,
            }

        # ============================================================
        # Sweep 2: Reverse — reuse L0..K, fresh L(K+1)..18
        # ============================================================
        print(f"\n{'='*80}")
        print(f"SWEEP 2: Reuse L0..K, Fresh L(K+1)..18")
        print(f"  'What if we only reuse shallow layers?'")
        print(f"{'='*80}")

        reverse_results = {}

        for K in range(0, NUM_MOE_LAYERS):
            reuse_layers = set(range(0, K + 1))
            label = f"reuse_0..{K}"

            runs = []
            for step in SAMPLE_STEPS:
                for batch_idx in BATCH_INDICES:
                    qt = get_qualifying_tokens(fresh_data, step, batch_idx)
                    if len(qt) < 2:
                        continue
                    result = runner.run_with_layer_mask(
                        step, qt, batch_idx, reuse_layers)
                    result["step"] = step
                    result["batch_idx"] = batch_idx
                    runs.append(result)

            if not runs:
                continue

            avg_kl = sum(r["mean_kl"] for r in runs) / len(runs)
            max_kl = max(r["max_kl"] for r in runs)
            avg_top1 = sum(r["top1_rate"] for r in runs) / len(runs)
            avg_reuse_pairs = sum(r["total_reuse_pairs"] for r in runs) / len(runs)
            n_unsafe_runs = sum(1 for r in runs if r["n_unsafe"] > 0)

            print(f"  K={K:>2d} ({label:>16s}): "
                  f"reuse_layers={len(reuse_layers):>2d}  "
                  f"avg_pairs={avg_reuse_pairs:>5.0f}  "
                  f"avg_kl={avg_kl:>10.6f}  max_kl={max_kl:>10.6f}  "
                  f"top1%={avg_top1*100:>5.2f}  "
                  f"unsafe_runs={n_unsafe_runs}/{len(runs)}")

            reverse_results[K] = {
                "reuse_layer_count": len(reuse_layers),
                "avg_kl": avg_kl,
                "max_kl": max_kl,
                "avg_top1_rate": avg_top1,
                "unsafe_run_rate": n_unsafe_runs / len(runs),
                "avg_reuse_pairs": avg_reuse_pairs,
                "n_runs": len(runs),
                "runs": runs,
            }

        # ============================================================
        # Sweep 3: Window — reuse only 3-layer window
        # ============================================================
        print(f"\n{'='*80}")
        print(f"SWEEP 3: Reuse only 3-layer window, fresh elsewhere")
        print(f"  'Which specific layers are safest to reuse?'")
        print(f"{'='*80}")

        window_results = {}
        WINDOW_SIZE = 3

        for start in range(0, NUM_MOE_LAYERS - WINDOW_SIZE + 1):
            reuse_layers = set(range(start, start + WINDOW_SIZE))
            label = f"L{start}..{start+WINDOW_SIZE-1}"

            runs = []
            for step in SAMPLE_STEPS:
                for batch_idx in BATCH_INDICES:
                    qt = get_qualifying_tokens(fresh_data, step, batch_idx)
                    if len(qt) < 2:
                        continue
                    result = runner.run_with_layer_mask(
                        step, qt, batch_idx, reuse_layers)
                    result["step"] = step
                    result["batch_idx"] = batch_idx
                    runs.append(result)

            if not runs:
                continue

            avg_kl = sum(r["mean_kl"] for r in runs) / len(runs)
            max_kl = max(r["max_kl"] for r in runs)
            avg_top1 = sum(r["top1_rate"] for r in runs) / len(runs)
            n_unsafe_runs = sum(1 for r in runs if r["n_unsafe"] > 0)

            print(f"  {label:>10s}: "
                  f"avg_kl={avg_kl:>10.6f}  max_kl={max_kl:>10.6f}  "
                  f"top1%={avg_top1*100:>5.2f}  "
                  f"unsafe_runs={n_unsafe_runs}/{len(runs)}")

            window_results[start] = {
                "layers": list(reuse_layers),
                "avg_kl": avg_kl,
                "max_kl": max_kl,
                "avg_top1_rate": avg_top1,
                "unsafe_run_rate": n_unsafe_runs / len(runs),
                "n_runs": len(runs),
                "runs": runs,
            }

        # ============================================================
        # Sweep 4: Contamination signal vs fresh-layer-count
        # For each forward-sweep config, show per-layer hidden_cos/shared_cos
        # ============================================================
        print(f"\n{'='*80}")
        print(f"CONTAMINATION SIGNALS BY FRESH-LAYER-COUNT (Forward Sweep)")
        print(f"  Per-layer hidden_cos at first reuse layer and last reuse layer")
        print(f"{'='*80}")

        print(f"  {'K':>3s} {'FreshLayers':>11s} {'1stReuse':>8s} "
              f"{'HC@1stR':>8s} {'SC@1stR':>8s} "
              f"{'HC@L18':>8s} {'SC@L18':>8s} {'avg_kl':>10s}")

        for K in sorted(forward_results.keys()):
            if K >= NUM_MOE_LAYERS - 1:
                continue
            fr = forward_results[K]
            first_reuse = K + 1
            if first_reuse >= NUM_MOE_LAYERS:
                continue

            # Aggregate layer signals across runs
            hc_first, sc_first, hc_last, sc_last = [], [], [], []
            for run in fr["runs"]:
                ls = run.get("layer_signals", {})
                if first_reuse in ls:
                    hc_first.append(ls[first_reuse]["hidden_cos_mean"])
                    sc_first.append(ls[first_reuse]["shared_cos_mean"])
                if (NUM_MOE_LAYERS - 1) in ls:
                    hc_last.append(ls[NUM_MOE_LAYERS - 1]["hidden_cos_mean"])
                    sc_last.append(ls[NUM_MOE_LAYERS - 1]["shared_cos_mean"])

            avg = lambda v: sum(v)/len(v) if v else 0
            print(f"  {K:>3d} {'L0..'+str(K):>11s} L{first_reuse:<6d} "
                  f"{avg(hc_first):>8.5f} {avg(sc_first):>8.5f} "
                  f"{avg(hc_last):>8.5f} {avg(sc_last):>8.5f} "
                  f"{fr['avg_kl']:>10.6f}")

        elapsed = time.time() - t0

        # ============================================================
        # SUMMARY TABLE
        # ============================================================
        print(f"\n{'='*80}")
        print(f"SUMMARY: Forward Sweep Pareto (fresh shallow, reuse deep)")
        print(f"{'='*80}")
        print(f"  {'FreshL':>6s} {'ReuseL':>6s} {'Pairs':>6s} "
              f"{'AvgKL':>10s} {'MaxKL':>10s} {'Top1%':>6s} {'UnsafeRun%':>10s} {'Verdict':>12s}")
        print(f"  {'-'*70}")

        for K in sorted(forward_results.keys()):
            fr = forward_results[K]
            n_fresh = K + 1
            n_reuse = NUM_MOE_LAYERS - n_fresh
            verdict = "?"
            if fr["avg_kl"] < 1e-4 and fr["avg_top1_rate"] < 0.01:
                verdict = "SAFE"
            elif fr["avg_kl"] < 1e-3 and fr["avg_top1_rate"] < 0.03:
                verdict = "ACCEPTABLE"
            elif fr["avg_kl"] < 1e-2 and fr["avg_top1_rate"] < 0.05:
                verdict = "MARGINAL"
            else:
                verdict = "UNSAFE"

            print(f"  {n_fresh:>6d} {n_reuse:>6d} {fr['avg_reuse_pairs']:>6.0f} "
                  f"{fr['avg_kl']:>10.6f} {fr['max_kl']:>10.6f} "
                  f"{fr['avg_top1_rate']*100:>5.2f}% "
                  f"{fr['unsafe_run_rate']*100:>9.1f}% "
                  f"{verdict:>12s}")

        print(f"\n{'='*80}")
        print(f"SUMMARY: Reverse Sweep Pareto (reuse shallow, fresh deep)")
        print(f"{'='*80}")
        print(f"  {'ReuseL':>6s} {'FreshL':>6s} {'Pairs':>6s} "
              f"{'AvgKL':>10s} {'MaxKL':>10s} {'Top1%':>6s} {'UnsafeRun%':>10s} {'Verdict':>12s}")
        print(f"  {'-'*70}")

        for K in sorted(reverse_results.keys()):
            rr = reverse_results[K]
            n_reuse = K + 1
            n_fresh = NUM_MOE_LAYERS - n_reuse
            verdict = "?"
            if rr["avg_kl"] < 1e-4 and rr["avg_top1_rate"] < 0.01:
                verdict = "SAFE"
            elif rr["avg_kl"] < 1e-3 and rr["avg_top1_rate"] < 0.03:
                verdict = "ACCEPTABLE"
            elif rr["avg_kl"] < 1e-2 and rr["avg_top1_rate"] < 0.05:
                verdict = "MARGINAL"
            else:
                verdict = "UNSAFE"

            print(f"  {n_reuse:>6d} {n_fresh:>6d} {rr['avg_reuse_pairs']:>6.0f} "
                  f"{rr['avg_kl']:>10.6f} {rr['max_kl']:>10.6f} "
                  f"{rr['avg_top1_rate']*100:>5.2f}% "
                  f"{rr['unsafe_run_rate']*100:>9.1f}% "
                  f"{verdict:>12s}")

        print(f"\nTotal time: {elapsed:.1f}s")

        # Save
        save_data = {
            "forward_sweep": {str(k): {kk: vv for kk, vv in v.items() if kk != "runs"}
                              for k, v in forward_results.items()},
            "reverse_sweep": {str(k): {kk: vv for kk, vv in v.items() if kk != "runs"}
                              for k, v in reverse_results.items()},
            "window_sweep": {str(k): {kk: vv for kk, vv in v.items() if kk != "runs"}
                             for k, v in window_results.items()},
        }
        results_dir = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction"
        save_path = results_dir / "layer_range_sweep_results.json"
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Saved to {save_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
