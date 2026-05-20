#!/usr/bin/env python3
"""
v0.1.14.9d — Corrected Q2: Global Top-p with Shared Rate

Correct framework:
  global_info = shared_rate + routing_rate × (cumsum_topk / total_routing_weight)

  where shared_rate = 0.419, routing_rate = 0.581 (from Q3 measurement)

For each top-p threshold, compute how many experts each token needs.
Break down by token type, layer group, step group.
"""

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MASK_ID = 156895
TOP_K = 8
SHARED_RATE = 0.419
ROUTING_RATE = 0.581


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
    print("v0.1.14.9d — Corrected Q2: Global Top-p Analysis")
    print(f"  shared_rate={SHARED_RATE:.3f}, routing_rate={ROUTING_RATE:.3f}")
    print("=" * 80)

    data_path = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction" / "full_fresh_run_data.pt"
    fresh_data = torch.load(data_path, map_location="cpu")
    n_iters = len(fresh_data["mask_state"])
    n_layers = len(fresh_data["topk_weights"][0])
    batch_size = fresh_data["topk_weights"][0][0].shape[0]
    block_len = fresh_data["topk_weights"][0][0].shape[1]
    print(f"  {n_iters} steps, {n_layers} layers, batch={batch_size}, block={block_len}")

    # Collect sorted weights with metadata
    print("\nCollecting routing weights...", flush=True)

    all_weights = []
    by_token_type = defaultdict(list)
    by_layer_group = defaultdict(list)
    by_step_group = defaultdict(list)

    for step in range(n_iters):
        sg = "early(0-3)" if step <= 3 else "mid(4-8)" if step <= 8 else "late(9-15)" if step <= 15 else "final(16+)"
        for layer in range(n_layers):
            lg = "shallow(0-3)" if layer <= 3 else "middle(4-13)" if layer <= 13 else "deep(14-18)"
            weights = fresh_data["topk_weights"][step][layer].float()  # [batch, block, 8]

            for bi in range(batch_size):
                for ti in range(block_len):
                    w = weights[bi, ti]
                    w_sorted, _ = w.sort(descending=True)
                    all_weights.append(w_sorted)
                    by_layer_group[lg].append(w_sorted)
                    by_step_group[sg].append(w_sorted)

                    # Token type classification (subsample for speed)
                    if step % 3 == 0 and layer % 4 == 0:
                        ttype = classify_token(fresh_data, step, ti, bi)
                        by_token_type[ttype].append(w_sorted)

    print(f"  Total: {len(all_weights)}, Token-typed: {sum(len(v) for v in by_token_type.values())}")

    # ============================================================
    # Global top-p analysis
    # ============================================================
    TOP_P_VALUES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    def global_top_p_analysis(weights_list, label):
        if not weights_list:
            return
        W = torch.stack(weights_list)  # [N, 8] sorted descending
        N = W.shape[0]
        total_routing = W.sum(dim=1, keepdim=True)  # [N, 1]
        cumsum = W.cumsum(dim=1)  # [N, 8]

        # Global info retention for each top-k
        # global_info[k] = shared_rate + routing_rate × (cumsum[:, k] / total_routing)
        # We want: for each token, find min k where global_info >= top_p

        print(f"\n  {label} (N={N}):")

        # First show: what global info does each top-k give?
        print(f"    {'top-k':>6s} {'RoutingKept%':>13s} {'GlobalInfo%':>12s}")
        print(f"    {'-'*35}")
        for k in range(1, TOP_K + 1):
            routing_frac = (cumsum[:, k - 1] / total_routing.squeeze()).mean().item()
            global_info = SHARED_RATE + ROUTING_RATE * routing_frac
            print(f"    top-{k:>1d}   {routing_frac*100:>11.1f}%  {global_info*100:>11.1f}%")

        # Then: for each global top-p, how many experts needed?
        print(f"\n    {'GlobalTopP':>10s} {'AvgExpert':>10s} {'Med':>5s} "
              f"{'1exp':>5s} {'2exp':>5s} {'3exp':>5s} {'4exp':>5s} "
              f"{'5exp':>5s} {'6exp':>5s} {'7exp':>5s} {'8exp':>5s} "
              f"{'SavedExp':>9s}")
        print(f"    {'-'*95}")

        for p in TOP_P_VALUES:
            # For each token: find min k such that
            # shared_rate + routing_rate * (cumsum[:, k-1] / total_routing) >= p
            # → cumsum[:, k-1] / total_routing >= (p - shared_rate) / routing_rate
            needed_routing_frac = (p - SHARED_RATE) / ROUTING_RATE
            if needed_routing_frac <= 0:
                # shared alone covers it
                print(f"    {p:>10.2f} {'shared only':>10s}")
                continue

            threshold = needed_routing_frac * total_routing  # [N, 1]
            enough = (cumsum >= threshold)
            enough[:, -1] = True
            first_enough = enough.float().argmax(dim=1) + 1  # [N], 1-indexed

            avg_k = first_enough.float().mean().item()
            med_k = first_enough.float().median().item()
            dist = [(first_enough == k).sum().item() / N * 100 for k in range(1, 9)]
            saved = TOP_K - avg_k

            print(f"    {p:>10.2f} {avg_k:>10.2f} {med_k:>5.0f} "
                  + " ".join(f"{d:>4.1f}%" for d in dist)
                  + f" {saved:>9.2f}")

    print(f"\n{'='*80}")
    print(f"CORRECTED Q2: Global Top-p (shared_rate={SHARED_RATE:.1%})")
    print(f"{'='*80}")

    global_top_p_analysis(all_weights, "ALL TOKENS")

    print(f"\n  --- By token type ---")
    for ttype in ["mask", "newly_decoded", "stable_decoded", "highly_stable"]:
        if ttype in by_token_type:
            global_top_p_analysis(by_token_type[ttype], f"TOKEN={ttype}")

    print(f"\n  --- By layer group ---")
    for lg in ["shallow(0-3)", "middle(4-13)", "deep(14-18)"]:
        if lg in by_layer_group:
            global_top_p_analysis(by_layer_group[lg], f"LAYER={lg}")

    print(f"\n  --- By step group ---")
    for sg in ["early(0-3)", "mid(4-8)", "late(9-15)", "final(16+)"]:
        if sg in by_step_group:
            global_top_p_analysis(by_step_group[sg], f"STEP={sg}")

    # ============================================================
    # Cross: token_type × layer_group at global top-p=0.80
    # ============================================================
    print(f"\n{'='*80}")
    print(f"TOKEN TYPE × LAYER: avg expert count at global top-p=0.80")
    print(f"{'='*80}")

    # Need per (token_type, layer_group) data
    by_tl = defaultdict(list)
    for step in range(0, n_iters, 3):
        for layer in range(0, n_layers, 4):
            lg = "shallow(0-3)" if layer <= 3 else "middle(4-13)" if layer <= 13 else "deep(14-18)"
            weights = fresh_data["topk_weights"][step][layer].float()
            for bi in range(batch_size):
                for ti in range(block_len):
                    ttype = classify_token(fresh_data, step, ti, bi)
                    w = weights[bi, ti]
                    w_sorted, _ = w.sort(descending=True)
                    by_tl[(ttype, lg)].append(w_sorted)

    print(f"  {'':>18s} {'shallow(0-3)':>13s} {'middle(4-13)':>13s} {'deep(14-18)':>13s}")
    print(f"  {'-'*60}")

    for ttype in ["mask", "newly_decoded", "stable_decoded", "highly_stable"]:
        vals = []
        for lg in ["shallow(0-3)", "middle(4-13)", "deep(14-18)"]:
            wl = by_tl.get((ttype, lg), [])
            if wl:
                W = torch.stack(wl)
                total_r = W.sum(dim=1, keepdim=True)
                cumsum = W.cumsum(dim=1)
                needed = (0.80 - SHARED_RATE) / ROUTING_RATE
                threshold = needed * total_r
                enough = (cumsum >= threshold)
                enough[:, -1] = True
                avg_k = (enough.float().argmax(dim=1) + 1).float().mean().item()
                vals.append(f"{avg_k:.2f}")
            else:
                vals.append("—")
        print(f"  {ttype:>18s} {vals[0]:>13s} {vals[1]:>13s} {vals[2]:>13s}")

    # Same for top-p=0.85
    print(f"\n  Same at global top-p=0.85:")
    print(f"  {'':>18s} {'shallow(0-3)':>13s} {'middle(4-13)':>13s} {'deep(14-18)':>13s}")
    print(f"  {'-'*60}")
    for ttype in ["mask", "newly_decoded", "stable_decoded", "highly_stable"]:
        vals = []
        for lg in ["shallow(0-3)", "middle(4-13)", "deep(14-18)"]:
            wl = by_tl.get((ttype, lg), [])
            if wl:
                W = torch.stack(wl)
                total_r = W.sum(dim=1, keepdim=True)
                cumsum = W.cumsum(dim=1)
                needed = (0.85 - SHARED_RATE) / ROUTING_RATE
                threshold = needed * total_r
                enough = (cumsum >= threshold)
                enough[:, -1] = True
                avg_k = (enough.float().argmax(dim=1) + 1).float().mean().item()
                vals.append(f"{avg_k:.2f}")
            else:
                vals.append("—")
        print(f"  {ttype:>18s} {vals[0]:>13s} {vals[1]:>13s} {vals[2]:>13s}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
