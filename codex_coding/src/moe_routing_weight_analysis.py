#!/usr/bin/env python3
"""
v0.1.14.9a — Routing Weight Distribution Analysis

Analyze top-8 expert routing weights from fresh run data to answer:
  Q1: What does the weight distribution look like?
      - Per-rank weight (1st expert, 2nd, ..., 8th)
      - By token type (MASK, newly_decoded, stable, highly_stable)
      - By layer (shallow, middle, deep)
      - By step (early, mid, late)

  Q2: How many experts are actually needed under top-p?
      - For top-p = [0.80, 0.85, 0.90, 0.95], how many experts suffice?
      - What fraction of computation could be saved?
      - Broken down by token type

All analysis from existing full_fresh_run_data.pt — no model run needed.
"""

import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MASK_ID = 156895
TOP_K = 8
NUM_MOE_LAYERS = 19


def classify_token(fresh_data, step, token_idx, batch_idx):
    mask_state = fresh_data["mask_state"]
    if mask_state[step][batch_idx, token_idx].item():
        return "mask"
    if step > 0 and mask_state[step - 1][batch_idx, token_idx].item():
        return "newly_decoded"
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
    return "highly_stable" if stable_len >= 4 else "stable_decoded"


def main():
    print("=" * 80)
    print("v0.1.14.9a — Routing Weight Distribution Analysis")
    print("=" * 80)

    data_path = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction" / "full_fresh_run_data.pt"
    fresh_data = torch.load(data_path, map_location="cpu")
    n_iters = len(fresh_data["mask_state"])
    n_layers = len(fresh_data["topk_weights"][0])
    batch_size = fresh_data["topk_weights"][0][0].shape[0]
    block_len = fresh_data["topk_weights"][0][0].shape[1]
    print(f"  {n_iters} steps, {n_layers} layers, batch={batch_size}, block={block_len}")

    # ============================================================
    # Collect all weights with metadata
    # ============================================================
    print("\nCollecting routing weights...", flush=True)

    # Storage: sorted weights per sample
    all_sorted_weights = []  # list of [8] tensors, sorted descending
    by_token_type = defaultdict(list)
    by_layer_group = defaultdict(list)
    by_step_group = defaultdict(list)
    by_token_layer = defaultdict(list)  # (token_type, layer_group)

    for step in range(n_iters):
        if step <= 3:
            sg = "early(0-3)"
        elif step <= 8:
            sg = "mid(4-8)"
        elif step <= 15:
            sg = "late(9-15)"
        else:
            sg = "final(16+)"

        for layer in range(n_layers):
            if layer <= 3:
                lg = "shallow(0-3)"
            elif layer <= 13:
                lg = "middle(4-13)"
            else:
                lg = "deep(14-18)"

            weights = fresh_data["topk_weights"][step][layer].float()  # [batch, block, 8]

            for bi in range(batch_size):
                for ti in range(block_len):
                    w = weights[bi, ti]  # [8]
                    w_sorted, _ = w.sort(descending=True)

                    # Only classify token type for a subset to save time
                    # (classify is expensive due to history scan)
                    if step % 3 == 0 and layer % 4 == 0:
                        ttype = classify_token(fresh_data, step, ti, bi)
                        by_token_type[ttype].append(w_sorted)
                        by_token_layer[(ttype, lg)].append(w_sorted)

                    all_sorted_weights.append(w_sorted)
                    by_layer_group[lg].append(w_sorted)
                    by_step_group[sg].append(w_sorted)

    print(f"  Total samples: {len(all_sorted_weights)}")
    print(f"  Token-type samples: {sum(len(v) for v in by_token_type.values())}")

    # ============================================================
    # Q1: Weight distribution by rank
    # ============================================================
    def analyze_weights(weights_list, label):
        if not weights_list:
            return {}
        W = torch.stack(weights_list)  # [N, 8]
        N = W.shape[0]

        print(f"\n  {label} (N={N}):")
        print(f"    {'Rank':>6s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'P25':>8s} "
              f"{'Median':>8s} {'P75':>8s} {'Max':>8s} {'CumMean':>8s}")
        print(f"    {'-'*76}")

        cum = 0
        rank_stats = {}
        for r in range(8):
            col = W[:, r]
            cum += col.mean().item()
            stats = {
                "mean": col.mean().item(),
                "std": col.std().item(),
                "min": col.min().item(),
                "p25": col.quantile(0.25).item(),
                "median": col.median().item(),
                "p75": col.quantile(0.75).item(),
                "max": col.max().item(),
                "cum_mean": cum,
            }
            rank_stats[r] = stats
            print(f"    #{r+1:>4d}  {stats['mean']:>8.4f} {stats['std']:>8.4f} "
                  f"{stats['min']:>8.4f} {stats['p25']:>8.4f} "
                  f"{stats['median']:>8.4f} {stats['p75']:>8.4f} "
                  f"{stats['max']:>8.4f} {cum:>8.4f}")

        return rank_stats

    print(f"\n{'='*80}")
    print(f"Q1: WEIGHT DISTRIBUTION BY RANK")
    print(f"{'='*80}")

    analyze_weights(all_sorted_weights, "ALL TOKENS")

    print(f"\n  --- By token type ---")
    for ttype in ["mask", "newly_decoded", "stable_decoded", "highly_stable"]:
        if ttype in by_token_type:
            analyze_weights(by_token_type[ttype], f"TOKEN={ttype}")

    print(f"\n  --- By layer group ---")
    for lg in ["shallow(0-3)", "middle(4-13)", "deep(14-18)"]:
        if lg in by_layer_group:
            analyze_weights(by_layer_group[lg], f"LAYER={lg}")

    print(f"\n  --- By step group ---")
    for sg in ["early(0-3)", "mid(4-8)", "late(9-15)", "final(16+)"]:
        if sg in by_step_group:
            analyze_weights(by_step_group[sg], f"STEP={sg}")

    # ============================================================
    # Q2: Top-p analysis — how many experts needed?
    # ============================================================
    print(f"\n{'='*80}")
    print(f"Q2: TOP-P ANALYSIS — How many experts are needed?")
    print(f"{'='*80}")

    TOP_P_VALUES = [0.80, 0.85, 0.90, 0.95]

    def top_p_analysis(weights_list, label):
        if not weights_list:
            return
        W = torch.stack(weights_list)  # [N, 8]
        N = W.shape[0]
        cumsum = W.cumsum(dim=1)  # [N, 8]

        print(f"\n  {label} (N={N}):")
        print(f"    {'top-p':>6s} {'AvgExpert':>10s} {'Med':>5s} "
              f"{'1exp%':>6s} {'2exp%':>6s} {'3exp%':>6s} {'4exp%':>6s} "
              f"{'5exp%':>6s} {'6exp%':>6s} {'7exp%':>6s} {'8exp%':>6s} "
              f"{'SavedExp':>9s} {'Save%':>6s}")
        print(f"    {'-'*105}")

        for p in TOP_P_VALUES:
            # For each sample, find the first index where cumsum >= p
            enough = (cumsum >= p)  # [N, 8] bool
            # Set all-False rows (if any) to True at index 7
            enough[:, 7] = True
            first_enough = enough.float().argmax(dim=1) + 1  # [N], 1-indexed

            avg_k = first_enough.float().mean().item()
            med_k = first_enough.float().median().item()
            dist = [(first_enough == k).sum().item() / N * 100 for k in range(1, 9)]
            saved = 8 - avg_k
            save_pct = saved / 8 * 100

            print(f"    {p:>6.2f} {avg_k:>10.2f} {med_k:>5.0f} "
                  + " ".join(f"{d:>5.1f}%" for d in dist)
                  + f" {saved:>9.2f} {save_pct:>5.1f}%")

    top_p_analysis(all_sorted_weights, "ALL TOKENS")

    for ttype in ["mask", "newly_decoded", "stable_decoded", "highly_stable"]:
        if ttype in by_token_type:
            top_p_analysis(by_token_type[ttype], f"TOKEN={ttype}")

    for lg in ["shallow(0-3)", "middle(4-13)", "deep(14-18)"]:
        if lg in by_layer_group:
            top_p_analysis(by_layer_group[lg], f"LAYER={lg}")

    # ============================================================
    # Q2b: Bottom-K weight analysis
    # ============================================================
    print(f"\n{'='*80}")
    print(f"Q2b: BOTTOM-K EXPERT WEIGHT ANALYSIS")
    print(f"{'='*80}")

    def bottom_k_analysis(weights_list, label):
        if not weights_list:
            return
        W = torch.stack(weights_list)  # [N, 8] sorted descending
        N = W.shape[0]

        print(f"\n  {label} (N={N}):")
        print(f"    {'Bottom-K':>9s} {'MeanWgt':>8s} {'<1%':>6s} {'<2%':>6s} "
              f"{'<5%':>6s} {'<10%':>6s} {'<15%':>6s}")
        print(f"    {'-'*50}")

        for k in [1, 2, 3, 4]:
            bottom_sum = W[:, 8-k:].sum(dim=1)  # sum of bottom-k weights
            mean_wgt = bottom_sum.mean().item()
            lt1 = (bottom_sum < 0.01).sum().item() / N * 100
            lt2 = (bottom_sum < 0.02).sum().item() / N * 100
            lt5 = (bottom_sum < 0.05).sum().item() / N * 100
            lt10 = (bottom_sum < 0.10).sum().item() / N * 100
            lt15 = (bottom_sum < 0.15).sum().item() / N * 100

            print(f"    bottom-{k:>1d}   {mean_wgt:>8.4f} {lt1:>5.1f}% {lt2:>5.1f}% "
                  f"{lt5:>5.1f}% {lt10:>5.1f}% {lt15:>5.1f}%")

    bottom_k_analysis(all_sorted_weights, "ALL TOKENS")

    for ttype in ["mask", "newly_decoded", "stable_decoded", "highly_stable"]:
        if ttype in by_token_type:
            bottom_k_analysis(by_token_type[ttype], f"TOKEN={ttype}")

    # ============================================================
    # Special: MASK vs Stable weight concentration comparison
    # ============================================================
    print(f"\n{'='*80}")
    print(f"MASK vs STABLE: Weight Concentration Comparison")
    print(f"{'='*80}")

    for ttype in ["mask", "highly_stable"]:
        if ttype not in by_token_type:
            continue
        W = torch.stack(by_token_type[ttype])
        top1_frac = W[:, 0].mean().item()
        top2_frac = W[:, :2].sum(dim=1).mean().item()
        top4_frac = W[:, :4].sum(dim=1).mean().item()
        gini = 1 - (W ** 2).sum(dim=1).mean().item() * 8  # normalized
        entropy = -(W * (W + 1e-8).log()).sum(dim=1).mean().item()
        max_entropy = torch.tensor(8.0).log().item()

        print(f"\n  {ttype}:")
        print(f"    Top-1 fraction:  {top1_frac:.4f}")
        print(f"    Top-2 fraction:  {top2_frac:.4f}")
        print(f"    Top-4 fraction:  {top4_frac:.4f}")
        print(f"    Entropy:         {entropy:.4f} / {max_entropy:.4f} = {entropy/max_entropy:.4f}")

    # ============================================================
    # Cross: token_type × layer_group
    # ============================================================
    print(f"\n{'='*80}")
    print(f"TOKEN TYPE × LAYER GROUP: avg top-p=0.90 expert count")
    print(f"{'='*80}")
    print(f"  {'':>18s} {'shallow(0-3)':>13s} {'middle(4-13)':>13s} {'deep(14-18)':>13s}")
    print(f"  {'-'*60}")

    for ttype in ["mask", "newly_decoded", "stable_decoded", "highly_stable"]:
        vals = []
        for lg in ["shallow(0-3)", "middle(4-13)", "deep(14-18)"]:
            key = (ttype, lg)
            if key in by_token_layer and by_token_layer[key]:
                W = torch.stack(by_token_layer[key])
                cumsum = W.cumsum(dim=1)
                enough = (cumsum >= 0.90)
                enough[:, 7] = True
                avg_k = (enough.float().argmax(dim=1) + 1).float().mean().item()
                vals.append(f"{avg_k:.2f}")
            else:
                vals.append("—")
        print(f"  {ttype:>18s} {vals[0]:>13s} {vals[1]:>13s} {vals[2]:>13s}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
