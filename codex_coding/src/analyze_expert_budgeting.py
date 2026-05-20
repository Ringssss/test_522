#!/usr/bin/env python3
"""
v0.1.15.3 Phase 1b — Expert Budgeting Feasibility Analysis (Vectorized)

Fully vectorized with torch ops — no Python for-loops over tokens.
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
NUM_EXPERTS = 256
TOP_K_ORIG = 8
SHARED_RATE = 0.419
ROUTING_RATE = 0.581
EXPERT_TOTAL_MB = 6.0


def compute_topp_kbudget(topk_w, top_p=0.75):
    """Returns per-token k_budget [N] (int, 1-8)."""
    sorted_w, _ = topk_w.sort(dim=1, descending=True)
    total = topk_w.sum(dim=1, keepdim=True)
    needed = (top_p - SHARED_RATE) / ROUTING_RATE * total
    cumsum = sorted_w.cumsum(dim=1)
    enough = cumsum >= needed
    enough[:, -1] = True
    return (enough.float().argmax(dim=1) + 1).int()  # [N]


def expert_budgeting_vectorized(gate_w, topk_idx, topk_w, K_target,
                                 quality_floor, dominant_floor, top_p=0.75):
    """
    Vectorized expert budgeting.
    gate_w: [N, 256], topk_idx: [N, 8], topk_w: [N, 8]
    """
    N = gate_w.shape[0]
    device = gate_w.device

    # Per-token k_budget from top-p
    k_budgets = compute_topp_kbudget(topk_w, top_p)  # [N], values 1-8

    # Original top-p expert set — use gate_w (consistent scale) for comparison
    # Sort topk experts by their routing weight to identify the top-p set
    sorted_rw, sort_order = topk_w.sort(dim=1, descending=True)
    sorted_idx = topk_idx.gather(1, sort_order)  # expert ids in routing-weight order

    # Get gate_w values for these experts (consistent scale)
    original_gate_vals = gate_w.gather(1, sorted_idx)  # [N, 8]

    positions = torch.arange(TOP_K_ORIG, device=device).unsqueeze(0)  # [1, 8]
    topp_mask = positions < k_budgets.unsqueeze(1)  # [N, 8]
    original_weight = (original_gate_vals * topp_mask.float()).sum(dim=1)  # [N]
    original_top1 = original_gate_vals[:, 0]  # [N] — gate_w of highest-routing expert

    # Step 1: Initial S by popularity
    popularity = gate_w.sum(dim=0)  # [256]
    _, pop_order = popularity.sort(descending=True)
    S_mask = torch.zeros(NUM_EXPERTS, dtype=torch.bool, device=device)
    S_mask[pop_order[:K_target]] = True

    # Step 2: Iterative safety check (vectorized per iteration)
    max_rounds = 30
    n_exceptions = 0

    for round_idx in range(max_rounds):
        # For each token, get weights of experts in S: gate_w[:, S_mask]
        # Then find top-k_budget among them
        s_indices = S_mask.nonzero(as_tuple=True)[0]  # [|S|]
        s_gate_w = gate_w[:, s_indices]  # [N, |S|]

        # For each token, sort S-experts by weight, take top-k_budget
        s_sorted_w, s_sort_order = s_gate_w.sort(dim=1, descending=True)

        # Covered weight = sum of top-k_budget from S
        # k_budgets vary per token, so we need per-token masking
        max_k = k_budgets.max().item()
        s_positions = torch.arange(s_sorted_w.shape[1], device=device).unsqueeze(0)
        s_topp_mask = s_positions < k_budgets.unsqueeze(1)  # [N, |S|]
        covered = (s_sorted_w * s_topp_mask.float()).sum(dim=1)  # [N]

        # Top-1 in S
        top1_in_s = s_sorted_w[:, 0]  # [N]

        # Check constraints
        safe_orig = original_weight > 1e-8
        cov_ratio = torch.where(safe_orig, covered / original_weight.clamp(min=1e-8),
                                torch.ones(N, device=device))
        t1_ratio = torch.where(original_top1 > 1e-8,
                               top1_in_s / original_top1.clamp(min=1e-8),
                               torch.ones(N, device=device))

        violated = (cov_ratio < quality_floor) | (t1_ratio < dominant_floor)
        violated = violated & safe_orig  # only check tokens with non-trivial routing

        if not violated.any():
            break

        # For violated tokens, find their most-needed expert NOT in S
        # gate_w[violated, :] but mask out experts already in S
        violated_idx = violated.nonzero(as_tuple=True)[0]
        v_gate = gate_w[violated_idx]  # [n_violated, 256]
        v_gate_masked = v_gate.clone()
        v_gate_masked[:, S_mask] = -1  # mask out experts in S
        # For each violated token, best missing expert
        best_missing = v_gate_masked.argmax(dim=1)  # [n_violated]
        # Add all unique missing experts to S
        unique_missing = best_missing.unique()
        for e in unique_missing:
            S_mask[e.item()] = True
            n_exceptions += 1

    # Final statistics
    final_S_size = S_mask.sum().item()

    # Recompute final covered ratios for reporting
    s_indices = S_mask.nonzero(as_tuple=True)[0]
    s_gate_w = gate_w[:, s_indices]
    s_sorted_w, _ = s_gate_w.sort(dim=1, descending=True)
    s_positions = torch.arange(s_sorted_w.shape[1], device=device).unsqueeze(0)
    s_topp_mask = s_positions < k_budgets.unsqueeze(1)
    covered = (s_sorted_w * s_topp_mask.float()).sum(dim=1)
    top1_in_s = s_sorted_w[:, 0]

    safe = original_weight > 1e-8
    cov_ratio = torch.where(safe, covered / original_weight.clamp(min=1e-8),
                            torch.ones(N, device=device))
    t1_ratio = torch.where(original_top1 > 1e-8,
                           top1_in_s / original_top1.clamp(min=1e-8),
                           torch.ones(N, device=device))

    return {
        "K_target": K_target,
        "final_S": final_S_size,
        "n_exceptions": n_exceptions,
        "avg_covered_ratio": cov_ratio.mean().item(),
        "min_covered_ratio": cov_ratio.min().item(),
        "p10_covered_ratio": cov_ratio.quantile(0.1).item(),
        "avg_top1_ratio": t1_ratio.mean().item(),
        "min_top1_ratio": t1_ratio.min().item(),
        "n_violated_final": int((cov_ratio < quality_floor).sum().item()),
    }


def main():
    print("=" * 80)
    print("v0.1.15.3 Phase 1b — Expert Budgeting Feasibility (Vectorized)")
    print("=" * 80)

    data_path = REPO_ROOT / "codex_coding" / "results" / "expert_budgeting_routing_data.pt"
    raw = torch.load(data_path, map_location="cpu")
    data = raw["data"]
    print(f"Loaded: {raw['sampled_steps']} steps, {raw['n_layers']} layers")

    baseline_active = 220.8

    # ============================================================
    # SWEEP 1: K_target (QF=0.85, DF=0.5)
    # ============================================================
    print(f"\n{'='*80}")
    print(f"SWEEP 1: K_target (quality_floor=0.85, dominant_floor=0.5)")
    print(f"{'='*80}")

    K_TARGETS = [220, 200, 180, 150, 120, 100, 80]
    sample_layers = [0, 4, 9, 14, 18]

    sweep1 = defaultdict(lambda: defaultdict(list))

    for si, step_data in enumerate(data):
        for li in sample_layers:
            if li not in step_data:
                continue
            ld = step_data[li]
            for K in K_TARGETS:
                r = expert_budgeting_vectorized(
                    ld["gate_w"], ld["topk_idx"], ld["topk_w"],
                    K_target=K, quality_floor=0.85, dominant_floor=0.5)
                sweep1[K][li].append(r)
        if (si + 1) % 10 == 0:
            print(f"  Step {si+1}/{len(data)} done", flush=True)

    # Print per-layer results
    print(f"\n  {'K':>4s}  {'Layer':>5s}  {'|S|':>6s}  {'Exc':>5s}  "
          f"{'AvgCov':>7s}  {'MinCov':>7s}  {'AvgT1':>7s}")
    print(f"  {'-'*52}")

    for K in K_TARGETS:
        for li in sample_layers:
            entries = sweep1[K][li]
            if not entries:
                continue
            avg_S = sum(e["final_S"] for e in entries) / len(entries)
            avg_exc = sum(e["n_exceptions"] for e in entries) / len(entries)
            avg_cov = sum(e["avg_covered_ratio"] for e in entries) / len(entries)
            min_cov = min(e["min_covered_ratio"] for e in entries)
            avg_t1 = sum(e["avg_top1_ratio"] for e in entries) / len(entries)
            print(f"  {K:>4d}  {li:>5d}  {avg_S:>6.1f}  {avg_exc:>5.1f}  "
                  f"{avg_cov:>7.4f}  {min_cov:>7.4f}  {avg_t1:>7.4f}")
        print()

    # Summary
    print(f"\n  SUMMARY (avg across layers {sample_layers}):")
    print(f"  {'K':>4s}  {'Avg|S|':>7s}  {'AvgExc':>7s}  {'AvgCov':>7s}  "
          f"{'HBM Save':>8s}  {'Verdict':>10s}")
    print(f"  {'-'*52}")

    for K in K_TARGETS:
        all_S = []
        all_exc = []
        all_cov = []
        for li in sample_layers:
            for e in sweep1[K][li]:
                all_S.append(e["final_S"])
                all_exc.append(e["n_exceptions"])
                all_cov.append(e["avg_covered_ratio"])
        if not all_S:
            continue
        avg_S = sum(all_S) / len(all_S)
        avg_exc = sum(all_exc) / len(all_exc)
        avg_cov = sum(all_cov) / len(all_cov)
        hbm = (1 - avg_S / baseline_active) * 100

        if hbm < 5: verdict = "NO GAIN"
        elif hbm < 15: verdict = "MODEST"
        elif hbm < 30: verdict = "GOOD"
        else: verdict = "EXCELLENT"

        print(f"  {K:>4d}  {avg_S:>7.1f}  {avg_exc:>7.1f}  {avg_cov:>7.4f}  "
              f"{hbm:>7.1f}%  {verdict:>10s}")

    # ============================================================
    # SWEEP 2: quality_floor sensitivity (K=150, Layer 9)
    # ============================================================
    print(f"\n{'='*80}")
    print(f"SWEEP 2: quality_floor (K=150, DF=0.5, Layer 9)")
    print(f"{'='*80}")

    QFS = [0.95, 0.85, 0.75, 0.65, 0.50]
    print(f"\n  {'QF':>5s}  {'|S|':>6s}  {'Exc':>5s}  {'AvgCov':>7s}  {'MinCov':>7s}  {'HBM%':>6s}")
    print(f"  {'-'*42}")

    for qf in QFS:
        all_r = []
        for step_data in data:
            if 9 not in step_data:
                continue
            ld = step_data[9]
            r = expert_budgeting_vectorized(
                ld["gate_w"], ld["topk_idx"], ld["topk_w"],
                K_target=150, quality_floor=qf, dominant_floor=0.5)
            all_r.append(r)
        avg_S = sum(r["final_S"] for r in all_r) / len(all_r)
        avg_exc = sum(r["n_exceptions"] for r in all_r) / len(all_r)
        avg_cov = sum(r["avg_covered_ratio"] for r in all_r) / len(all_r)
        min_cov = min(r["min_covered_ratio"] for r in all_r)
        hbm = (1 - avg_S / baseline_active) * 100
        print(f"  {qf:>5.2f}  {avg_S:>6.1f}  {avg_exc:>5.1f}  "
              f"{avg_cov:>7.4f}  {min_cov:>7.4f}  {hbm:>5.1f}%")

    # ============================================================
    # SWEEP 3: dominant_floor sensitivity (K=150, QF=0.85, Layer 9)
    # ============================================================
    print(f"\n{'='*80}")
    print(f"SWEEP 3: dominant_floor (K=150, QF=0.85, Layer 9)")
    print(f"{'='*80}")

    DFS = [0.80, 0.60, 0.50, 0.30, 0.0]
    print(f"\n  {'DF':>5s}  {'|S|':>6s}  {'Exc':>5s}  {'AvgT1':>7s}  {'MinT1':>7s}  {'HBM%':>6s}")
    print(f"  {'-'*42}")

    for df in DFS:
        all_r = []
        for step_data in data:
            if 9 not in step_data:
                continue
            ld = step_data[9]
            r = expert_budgeting_vectorized(
                ld["gate_w"], ld["topk_idx"], ld["topk_w"],
                K_target=150, quality_floor=0.85, dominant_floor=df)
            all_r.append(r)
        avg_S = sum(r["final_S"] for r in all_r) / len(all_r)
        avg_exc = sum(r["n_exceptions"] for r in all_r) / len(all_r)
        avg_t1 = sum(r["avg_top1_ratio"] for r in all_r) / len(all_r)
        min_t1 = min(r["min_top1_ratio"] for r in all_r)
        hbm = (1 - avg_S / baseline_active) * 100
        print(f"  {df:>5.2f}  {avg_S:>6.1f}  {avg_exc:>5.1f}  "
              f"{avg_t1:>7.4f}  {min_t1:>7.4f}  {hbm:>5.1f}%")

    # ============================================================
    # CONCLUSION
    # ============================================================
    print(f"\n{'='*80}")
    print(f"CONCLUSION")
    print(f"{'='*80}")
    ref = []
    for li in sample_layers:
        for e in sweep1[150][li]:
            ref.append(e["final_S"])
    if ref:
        ref_S = sum(ref) / len(ref)
        ref_hbm = (1 - ref_S / baseline_active) * 100
        print(f"  Reference: K_target=150, QF=0.85, DF=0.5")
        print(f"  Avg |S| = {ref_S:.1f}  (baseline {baseline_active:.1f})")
        print(f"  HBM saving = {ref_hbm:.1f}%")
        print(f"  Expert HBM/fwd: {ref_S * EXPERT_TOTAL_MB * 19:.0f} MB "
              f"(baseline: {baseline_active * EXPERT_TOTAL_MB * 19:.0f} MB)")

    # Save
    save_data = {"sweep1": {}, "baseline_active": baseline_active}
    for K in K_TARGETS:
        layer_data = {}
        for li in sample_layers:
            entries = sweep1[K][li]
            if entries:
                layer_data[li] = {
                    "avg_S": sum(e["final_S"] for e in entries) / len(entries),
                    "avg_exc": sum(e["n_exceptions"] for e in entries) / len(entries),
                    "avg_cov": sum(e["avg_covered_ratio"] for e in entries) / len(entries),
                }
        save_data["sweep1"][K] = layer_data
    out_path = REPO_ROOT / "codex_coding" / "results" / "expert_budgeting_feasibility.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
