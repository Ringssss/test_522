#!/usr/bin/env python3
"""
MoE Proxy Analysis & Selection (v0.1.14.3+4)

Analyzes 5681 counterfactual samples to find the best low-cost proxy signals
for predicting MoE skip risk. Produces:
  1. Single-variable analysis (signal vs unsafe rate)
  2. ROC-AUC / PR-AUC ranking
  3. Logistic regression & decision tree
  4. Recommended threshold rules
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
DATA_PATH = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction" / "counterfactual_kvcache_full.json"
RESULTS_DIR = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction"


def load_data():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} samples")
    return raw


# ============================================================
# 1. Basic statistics
# ============================================================
def basic_stats(data):
    print(f"\n{'='*80}")
    print("1. BASIC STATISTICS")
    print(f"{'='*80}")

    total = len(data)
    labels = defaultdict(int)
    for d in data:
        labels[d["risk_label"]] += 1
    for l in ["safe", "borderline", "unsafe"]:
        print(f"  {l:>12s}: {labels[l]:>5d} ({labels[l]/total*100:.1f}%)")

    # Binary: safe vs not-safe
    safe_n = labels["safe"]
    not_safe = total - safe_n
    print(f"\n  Binary: safe={safe_n} ({safe_n/total*100:.1f}%), "
          f"not-safe={not_safe} ({not_safe/total*100:.1f}%)")


# ============================================================
# 2. Single-variable analysis
# ============================================================
PROXY_SIGNALS = [
    "hidden_cos_prev",
    "hidden_rel_l2_prev",
    "gate_cos_prev",
    "gate_rel_l2_prev",
    "gate_topk_overlap_prev",
    "routing_weight_cos_prev",
    "routing_weight_l1_prev",
    "token_confidence",
    "token_margin",
    "stable_len",
    "mask_ratio",
]

CONTEXT_SIGNALS = ["layer", "step"]


def single_variable_analysis(data):
    print(f"\n{'='*80}")
    print("2. SINGLE-VARIABLE ANALYSIS")
    print(f"{'='*80}")

    # For each signal, compute: mean(safe), mean(unsafe), separation
    print(f"\n  {'Signal':<28s} {'mean(safe)':>12s} {'mean(unsafe)':>12s} {'separation':>12s}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*12}")

    for sig in PROXY_SIGNALS:
        safe_vals = [d[sig] for d in data if d["risk_label"] == "safe"]
        unsafe_vals = [d[sig] for d in data if d["risk_label"] == "unsafe"]
        if not safe_vals or not unsafe_vals:
            continue
        ms = np.mean(safe_vals)
        mu = np.mean(unsafe_vals)
        ss = np.std(safe_vals) + 1e-8
        su = np.std(unsafe_vals) + 1e-8
        # Cohen's d as separation measure
        pooled_std = np.sqrt((ss**2 + su**2) / 2)
        cohens_d = abs(ms - mu) / pooled_std
        print(f"  {sig:<28s} {ms:>12.4f} {mu:>12.4f} {cohens_d:>12.3f}")

    # Binned unsafe rate for top signals
    print(f"\n  --- Binned unsafe rate for key signals ---")
    key_signals = ["gate_topk_overlap_prev", "gate_cos_prev", "hidden_cos_prev",
                   "routing_weight_cos_prev", "token_confidence", "stable_len"]

    for sig in key_signals:
        vals = [(d[sig], d["risk_label"]) for d in data if d["risk_label"] != "unknown"]
        if not vals:
            continue
        vals.sort(key=lambda x: x[0])
        # 5 equal-count bins
        n = len(vals)
        bin_size = n // 5
        print(f"\n  {sig}:")
        print(f"    {'Bin range':<30s} {'count':>6s} {'unsafe%':>8s} {'safe%':>8s}")
        for i in range(5):
            start = i * bin_size
            end = (i + 1) * bin_size if i < 4 else n
            bin_data = vals[start:end]
            count = len(bin_data)
            unsafe_r = sum(1 for _, l in bin_data if l == "unsafe") / count * 100
            safe_r = sum(1 for _, l in bin_data if l == "safe") / count * 100
            lo = bin_data[0][0]
            hi = bin_data[-1][0]
            print(f"    [{lo:>8.4f}, {hi:>8.4f}]     {count:>6d} {unsafe_r:>7.1f}% {safe_r:>7.1f}%")


# ============================================================
# 3. ROC-AUC / PR-AUC ranking
# ============================================================
def auc_ranking(data):
    print(f"\n{'='*80}")
    print("3. ROC-AUC / PR-AUC RANKING (safe vs not-safe)")
    print(f"{'='*80}")

    # Binary labels: 1=safe, 0=not-safe
    y = np.array([1 if d["risk_label"] == "safe" else 0 for d in data])

    from sklearn.metrics import roc_auc_score, average_precision_score

    all_signals = PROXY_SIGNALS + CONTEXT_SIGNALS
    results = []

    for sig in all_signals:
        x = np.array([d[sig] for d in data], dtype=float)
        # Handle NaN/inf
        mask = np.isfinite(x)
        if mask.sum() < 100:
            continue
        x_clean = x[mask]
        y_clean = y[mask]

        try:
            roc = roc_auc_score(y_clean, x_clean)
            # If AUC < 0.5, flip (signal is negatively correlated)
            if roc < 0.5:
                roc = 1 - roc
                pr = average_precision_score(y_clean, -x_clean)
                direction = "lower→safe"
            else:
                pr = average_precision_score(y_clean, x_clean)
                direction = "higher→safe"
            results.append((sig, roc, pr, direction))
        except Exception:
            continue

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  {'Rank':<5s} {'Signal':<28s} {'ROC-AUC':>9s} {'PR-AUC':>9s} {'Direction':<14s}")
    print(f"  {'-'*5} {'-'*28} {'-'*9} {'-'*9} {'-'*14}")
    for i, (sig, roc, pr, direction) in enumerate(results):
        print(f"  {i+1:<5d} {sig:<28s} {roc:>9.4f} {pr:>9.4f} {direction:<14s}")

    return results


# ============================================================
# 4. Logistic regression & decision tree
# ============================================================
def ml_analysis(data):
    print(f"\n{'='*80}")
    print("4. LOGISTIC REGRESSION & DECISION TREE")
    print(f"{'='*80}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    # Binary: safe=1, not-safe=0
    y = np.array([1 if d["risk_label"] == "safe" else 0 for d in data])

    # Feature sets to compare
    feature_sets = {
        "token_state_only": ["token_confidence", "token_margin", "stable_len", "mask_ratio"],
        "routing_only": ["gate_cos_prev", "gate_topk_overlap_prev",
                         "routing_weight_cos_prev", "routing_weight_l1_prev"],
        "hidden_only": ["hidden_cos_prev", "hidden_rel_l2_prev"],
        "routing+context": ["gate_cos_prev", "gate_topk_overlap_prev",
                            "routing_weight_cos_prev", "routing_weight_l1_prev",
                            "layer", "step"],
        "routing+token": ["gate_cos_prev", "gate_topk_overlap_prev",
                          "routing_weight_cos_prev", "routing_weight_l1_prev",
                          "token_confidence", "token_margin", "stable_len"],
        "all_signals": PROXY_SIGNALS + CONTEXT_SIGNALS,
    }

    print(f"\n  --- Ablation: which signal group predicts best? ---")
    print(f"  {'Feature set':<24s} {'AUC (5-fold)':>14s} {'#features':>10s}")
    print(f"  {'-'*24} {'-'*14} {'-'*10}")

    best_name = None
    best_auc = 0

    for name, features in feature_sets.items():
        X = np.array([[d[f] for f in features] for d in data], dtype=float)
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(lr, X_scaled, y, cv=5, scoring='roc_auc')
        mean_auc = scores.mean()
        print(f"  {name:<24s} {mean_auc:>10.4f} ± {scores.std():.4f} {len(features):>10d}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_name = name

    print(f"\n  Best: {best_name} (AUC={best_auc:.4f})")

    # Train decision tree on best feature set for interpretable rules
    best_features = feature_sets[best_name]
    X = np.array([[d[f] for f in best_features] for d in data], dtype=float)
    X = np.nan_to_num(X, nan=0.0)

    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)
    dt_scores = cross_val_score(dt, X, y, cv=5, scoring='roc_auc')
    print(f"\n  Decision tree (depth=4) on {best_name}: AUC={dt_scores.mean():.4f}")

    dt.fit(X, y)
    tree_text = export_text(dt, feature_names=best_features, max_depth=4)
    print(f"\n  Decision tree rules:")
    for line in tree_text.split('\n')[:30]:
        print(f"    {line}")

    # Logistic regression coefficients on best set
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y)
    print(f"\n  Logistic regression coefficients ({best_name}):")
    print(f"  {'Feature':<28s} {'Coefficient':>12s}")
    for f, c in sorted(zip(best_features, lr.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {f:<28s} {c:>12.4f}")


# ============================================================
# 5. Threshold rule evaluation
# ============================================================
def threshold_analysis(data):
    print(f"\n{'='*80}")
    print("5. THRESHOLD RULE EVALUATION")
    print(f"{'='*80}")

    total = len(data)

    # Test various threshold rules
    rules = [
        ("baseline: stable_decoded only",
         lambda d: d["token_state"] in ("stable_decoded", "highly_stable")),

        ("gate_topk_overlap >= 1.0",
         lambda d: d["gate_topk_overlap_prev"] >= 1.0),

        ("gate_topk_overlap >= 0.875",
         lambda d: d["gate_topk_overlap_prev"] >= 0.875),

        ("gate_cos >= 0.999",
         lambda d: d["gate_cos_prev"] >= 0.999),

        ("gate_cos >= 0.995",
         lambda d: d["gate_cos_prev"] >= 0.995),

        ("stable + gate_topk >= 1.0",
         lambda d: d["token_state"] in ("stable_decoded", "highly_stable")
                   and d["gate_topk_overlap_prev"] >= 1.0),

        ("stable + gate_cos >= 0.999",
         lambda d: d["token_state"] in ("stable_decoded", "highly_stable")
                   and d["gate_cos_prev"] >= 0.999),

        ("stable + gate_topk >= 0.875 + gate_cos >= 0.995",
         lambda d: d["token_state"] in ("stable_decoded", "highly_stable")
                   and d["gate_topk_overlap_prev"] >= 0.875
                   and d["gate_cos_prev"] >= 0.995),

        ("highly_stable + gate_topk >= 0.875",
         lambda d: d["token_state"] == "highly_stable"
                   and d["gate_topk_overlap_prev"] >= 0.875),

        ("highly_stable + gate_topk >= 1.0",
         lambda d: d["token_state"] == "highly_stable"
                   and d["gate_topk_overlap_prev"] >= 1.0),

        ("stable + gate_topk >= 0.875 + rw_cos >= 0.99",
         lambda d: d["token_state"] in ("stable_decoded", "highly_stable")
                   and d["gate_topk_overlap_prev"] >= 0.875
                   and d["routing_weight_cos_prev"] >= 0.99),

        ("L0-15 + stable + gate_topk >= 0.875",
         lambda d: d["layer"] <= 15
                   and d["token_state"] in ("stable_decoded", "highly_stable")
                   and d["gate_topk_overlap_prev"] >= 0.875),

        ("L0-15 + highly_stable + gate_topk >= 0.875",
         lambda d: d["layer"] <= 15
                   and d["token_state"] == "highly_stable"
                   and d["gate_topk_overlap_prev"] >= 0.875),
    ]

    print(f"\n  {'Rule':<52s} {'reuse_n':>8s} {'reuse%':>7s} "
          f"{'safe_n':>7s} {'unsafe_n':>9s} {'FNR':>6s} {'precision':>10s}")
    print(f"  {'-'*52} {'-'*8} {'-'*7} {'-'*7} {'-'*9} {'-'*6} {'-'*10}")

    for name, rule_fn in rules:
        reused = [d for d in data if rule_fn(d)]
        n_reused = len(reused)
        if n_reused == 0:
            print(f"  {name:<52s} {'0':>8s}")
            continue
        n_safe = sum(1 for d in reused if d["risk_label"] == "safe")
        n_unsafe = sum(1 for d in reused if d["risk_label"] == "unsafe")
        n_border = sum(1 for d in reused if d["risk_label"] == "borderline")
        # FNR = unsafe among reused / total reused (false negative rate)
        fnr = n_unsafe / n_reused * 100
        precision = n_safe / n_reused * 100
        reuse_pct = n_reused / total * 100
        print(f"  {name:<52s} {n_reused:>8d} {reuse_pct:>6.1f}% "
              f"{n_safe:>7d} {n_unsafe:>9d} {fnr:>5.1f}% {precision:>9.1f}%")


# ============================================================
# 6. Layer-wise risk structure
# ============================================================
def layer_analysis(data):
    print(f"\n{'='*80}")
    print("6. LAYER-WISE RISK STRUCTURE")
    print(f"{'='*80}")

    # Per-layer: mean proxy values for safe vs unsafe
    print(f"\n  --- Mean gate_topk_overlap by layer × risk ---")
    print(f"  {'Layer':<8s} {'safe':>10s} {'unsafe':>10s} {'delta':>10s}")
    for layer in range(19):
        safe_vals = [d["gate_topk_overlap_prev"] for d in data
                     if d["layer"] == layer and d["risk_label"] == "safe"]
        unsafe_vals = [d["gate_topk_overlap_prev"] for d in data
                       if d["layer"] == layer and d["risk_label"] == "unsafe"]
        if safe_vals and unsafe_vals:
            ms = np.mean(safe_vals)
            mu = np.mean(unsafe_vals)
            print(f"  {layer:<8d} {ms:>10.4f} {mu:>10.4f} {ms-mu:>10.4f}")

    # Per-layer safe rate for highly_stable tokens with topk_overlap >= 0.875
    print(f"\n  --- Safe rate for highly_stable + topk_overlap >= 0.875, by layer ---")
    print(f"  {'Layer':<8s} {'count':>6s} {'safe%':>8s} {'unsafe%':>8s}")
    for layer in range(19):
        subset = [d for d in data if d["layer"] == layer
                  and d["token_state"] == "highly_stable"
                  and d["gate_topk_overlap_prev"] >= 0.875]
        if not subset:
            continue
        n = len(subset)
        safe_r = sum(1 for d in subset if d["risk_label"] == "safe") / n * 100
        unsafe_r = sum(1 for d in subset if d["risk_label"] == "unsafe") / n * 100
        print(f"  {layer:<8d} {n:>6d} {safe_r:>7.1f}% {unsafe_r:>7.1f}%")


# ============================================================
# Main
# ============================================================
def main():
    data = load_data()

    # Filter out step=0 samples (no previous data to compare)
    data = [d for d in data if d["step"] > 0]
    print(f"After filtering step>0: {len(data)} samples")

    basic_stats(data)
    single_variable_analysis(data)
    auc_results = auc_ranking(data)
    ml_analysis(data)
    threshold_analysis(data)
    layer_analysis(data)

    # Save analysis summary
    summary = {
        "total_samples": len(data),
        "risk_distribution": {l: sum(1 for d in data if d["risk_label"] == l)
                              for l in ["safe", "borderline", "unsafe"]},
        "auc_ranking": [(sig, float(roc), float(pr), direction)
                        for sig, roc, pr, direction in auc_results],
    }
    out_path = RESULTS_DIR / "proxy_analysis_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary to {out_path}")


if __name__ == "__main__":
    main()
