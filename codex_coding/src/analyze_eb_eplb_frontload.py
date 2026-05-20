#!/usr/bin/env python3
"""
EB-aware EPLB front-loaded analysis (offline, trace-driven).

Stages:
1) Skew-latency correlation (using existing per-forward timing reference).
2) Offline placement upper-bound comparison.
3) Replica budget sweep.
4) Global-prior transfer validation on held-out blocks.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

NUM_EXPERTS = 256
EP_SIZE = 8
EXPERTS_PER_EP = NUM_EXPERTS // EP_SIZE


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    # Average ranks for ties, 1-based ranks
    idx = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(a, dtype=np.float64)
    n = a.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and a[idx[j]] == a[idx[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[idx[i:j]] = avg_rank
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson(rx, ry)


def _load_rank_trace(path: Path) -> dict:
    with gzip.open(path, "rt") as f:
        obj = json.load(f)
    out = {
        "rank": int(obj["rank"]),
        "dp_rank": int(obj["dp_rank"]),
        "tp_rank": int(obj["tp_rank"]),
        "stats": obj.get("stats", {}),
        "records": [],
        "grouping_records": obj.get("grouping_records", []),
    }
    for r in obj.get("records", []):
        out["records"].append(
            {
                "rank": int(r["rank"]),
                "dp_rank": int(r["dp_rank"]),
                "tp_rank": int(r["tp_rank"]),
                "layer": int(r["layer"]),
                "block": int(r["block"]),
                "fwd_in_block": int(r["fwd_in_block"]),
                "path": str(r["path"]),
                "n_tokens": int(r["n_tokens"]),
                "s_size": int(r["s_size"]),
                "s_active": np.asarray(r["s_active"], dtype=np.int32),
                "eb_hist4": np.asarray(r["eb_hist4"], dtype=np.int64),
                "noeb_hist4": np.asarray(r["noeb_hist4"], dtype=np.int64),
                "noeb_hist8": np.asarray(r["noeb_hist8"], dtype=np.int64)
                if r.get("noeb_hist8")
                else None,
                "eb_unique4": int(r["eb_unique4"]),
                "noeb_unique4": int(r["noeb_unique4"]),
                "noeb_unique8": int(r["noeb_unique8"])
                if r.get("noeb_unique8") is not None
                else None,
            }
        )
    return out


def _linear_placement() -> np.ndarray:
    return np.asarray(
        [e // EXPERTS_PER_EP for e in range(NUM_EXPERTS)], dtype=np.int32
    )


def _random_balanced_placement(seed: int = 20260429) -> np.ndarray:
    rng = np.random.default_rng(seed)
    experts = np.arange(NUM_EXPERTS, dtype=np.int32)
    rng.shuffle(experts)
    place = np.empty(NUM_EXPERTS, dtype=np.int32)
    for i, eid in enumerate(experts.tolist()):
        place[eid] = i % EP_SIZE
    return place


def _popularity_round_robin(pop: np.ndarray) -> np.ndarray:
    order = np.argsort(-pop, kind="mergesort")
    place = np.empty(NUM_EXPERTS, dtype=np.int32)
    for i, eid in enumerate(order.tolist()):
        place[eid] = i % EP_SIZE
    return place


def _coact_neighbors(
    records: list[dict], route_key: str, top_n: int = 16
) -> list[dict[int, float]]:
    nbr = [defaultdict(float) for _ in range(NUM_EXPERTS)]
    for r in records:
        h = r[route_key]
        nz = np.nonzero(h > 0)[0]
        if nz.size == 0:
            continue
        if nz.size > top_n:
            vals = h[nz]
            idx = np.argpartition(vals, -top_n)[-top_n:]
            cand = nz[idx]
        else:
            cand = nz
        cand = cand[np.argsort(-h[cand], kind="mergesort")]
        for i in range(cand.size):
            ei = int(cand[i])
            ci = float(h[ei])
            for j in range(i + 1, cand.size):
                ej = int(cand[j])
                cj = float(h[ej])
                w = min(ci, cj)
                nbr[ei][ej] += w
                nbr[ej][ei] += w
    return nbr


def _coact_balanced_placement(
    pop: np.ndarray,
    neighbors: list[dict[int, float]],
    lambda_penalty: float = 0.35,
) -> np.ndarray:
    order = np.argsort(-pop, kind="mergesort")
    place = np.full(NUM_EXPERTS, -1, dtype=np.int32)
    gpu_counts = np.zeros(EP_SIZE, dtype=np.int32)
    gpu_pop = np.zeros(EP_SIZE, dtype=np.float64)
    mean_pop = float(pop.sum() / max(EP_SIZE, 1))
    for eid in order.tolist():
        best_g = -1
        best_score = None
        e_pop = float(pop[eid])
        for g in range(EP_SIZE):
            if gpu_counts[g] >= EXPERTS_PER_EP:
                continue
            co_pen = 0.0
            for nb, w in neighbors[eid].items():
                if place[nb] == g:
                    co_pen += float(w)
            load_term = _safe_div(gpu_pop[g], mean_pop + 1e-9)
            pen_term = _safe_div(co_pen, e_pop + 1e-9)
            score = load_term + lambda_penalty * pen_term
            if best_score is None or score < best_score:
                best_score = score
                best_g = g
        if best_g < 0:
            best_g = int(np.argmin(gpu_counts))
        place[eid] = best_g
        gpu_counts[best_g] += 1
        gpu_pop[best_g] += e_pop
    return place


def _ep_loads(hist: np.ndarray, placement: np.ndarray) -> np.ndarray:
    # placement[eid] in [0, EP_SIZE)
    return np.bincount(
        placement, weights=hist.astype(np.float64), minlength=EP_SIZE
    ).astype(np.float64)


def _load_metrics(loads: np.ndarray) -> tuple[float, float, float]:
    mx = float(loads.max())
    mn = float(loads.min())
    sm = float(loads.sum())
    skew = _safe_div(mx, max(mn, 1.0))
    max_frac = _safe_div(mx, sm)
    return mx, skew, max_frac


def _oracle_lpt_max(hist: np.ndarray) -> tuple[float, float]:
    # Per-record optimistic lower proxy:
    # assign each expert count to current least-loaded EP rank (dynamic remap).
    cnts = hist[hist > 0]
    if cnts.size == 0:
        return 0.0, 0.0
    order = np.sort(cnts)[::-1]
    loads = np.zeros(EP_SIZE, dtype=np.float64)
    for c in order.tolist():
        g = int(np.argmin(loads))
        loads[g] += float(c)
    mx = float(loads.max())
    frac = _safe_div(mx, float(order.sum()))
    return mx, frac


def _evaluate_placement(records: list[dict], route_key: str, placement: np.ndarray) -> dict:
    max_loads = []
    skews = []
    max_fracs = []
    for r in records:
        h = r[route_key]
        loads = _ep_loads(h, placement)
        mx, sk, mf = _load_metrics(loads)
        max_loads.append(mx)
        skews.append(sk)
        max_fracs.append(mf)
    return {
        "num_records": int(len(records)),
        "max_load": _stats(max_loads),
        "skew": _stats(skews),
        "max_load_frac": _stats(max_fracs),
    }


def _select_best_placement(stage2: dict) -> str:
    # Primary selector: p95 max_load_frac on EB route.
    candidates = {}
    for name, payload in stage2["placement_eval"]["eb_hist4"].items():
        p95 = payload["max_load_frac"].get("p95", 1e9)
        candidates[name] = p95
    return min(candidates, key=candidates.get)


def _build_replica_map(
    base_placement: np.ndarray,
    pop: np.ndarray,
    pct: float,
) -> dict[int, list[int]]:
    n_rep = max(1, int(round(NUM_EXPERTS * pct)))
    hot = np.argsort(-pop, kind="mergesort")[:n_rep]
    rep_map: dict[int, list[int]] = {}

    # Track global copy-load proxy to choose replica destinations.
    copy_pop = np.zeros(EP_SIZE, dtype=np.float64)
    for eid in range(NUM_EXPERTS):
        copy_pop[base_placement[eid]] += float(pop[eid])

    for eid in hot.tolist():
        g0 = int(base_placement[eid])
        g1 = int(np.argmin(np.where(np.arange(EP_SIZE) == g0, np.inf, copy_pop)))
        rep_map[eid] = [g0, g1]
        copy_pop[g1] += float(pop[eid])
    return rep_map


def _apply_replicas_and_score(
    records: list[dict],
    route_key: str,
    base_placement: np.ndarray,
    rep_map: dict[int, list[int]],
) -> dict:
    max_loads = []
    skews = []
    max_fracs = []
    for r in records:
        h = r[route_key]
        loads = np.zeros(EP_SIZE, dtype=np.float64)
        nz = np.nonzero(h > 0)[0]
        # Process larger experts first to reduce greedy bias.
        nz = nz[np.argsort(-h[nz], kind="mergesort")]
        for eid in nz.tolist():
            c = float(h[eid])
            opts = rep_map.get(eid)
            if not opts:
                loads[int(base_placement[eid])] += c
                continue
            g0, g1 = int(opts[0]), int(opts[1])
            # Two-way water-fill split.
            l0 = loads[g0]
            l1 = loads[g1]
            x = 0.5 * (c + (l1 - l0))
            if x < 0.0:
                x = 0.0
            if x > c:
                x = c
            x0 = float(round(x))
            x1 = c - x0
            loads[g0] += x0
            loads[g1] += x1
        mx, sk, mf = _load_metrics(loads)
        max_loads.append(mx)
        skews.append(sk)
        max_fracs.append(mf)
    return {
        "num_records": int(len(records)),
        "replicated_experts": int(len(rep_map)),
        "max_load": _stats(max_loads),
        "skew": _stats(skews),
        "max_load_frac": _stats(max_fracs),
    }


def _coverage_for_set(hist: np.ndarray, keep: np.ndarray) -> float:
    total = float(hist.sum())
    if total <= 0:
        return 1.0
    return _safe_div(float(hist[keep].sum()), total)


def _stage1(records: list[dict], dp2_block_start: dict | None) -> dict:
    out = {}

    # 1A) Direct timing correlation from existing per-forward timing reference.
    if dp2_block_start and dp2_block_start.get("per_fwd"):
        per = dp2_block_start["per_fwd"]
        x = np.asarray([float(d["skew"]) for d in per], dtype=np.float64)
        y = np.asarray([float(d["moe_ms"]) for d in per], dtype=np.float64)
        mask = (x > 0.0) & np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        out["timing_reference"] = {
            "num_points": int(x.size),
            "pearson_skew_vs_moe_ms": _pearson(x, y),
            "spearman_skew_vs_moe_ms": _spearman(x, y),
            "skew_stats": _stats(x.tolist()),
            "moe_ms_stats": _stats(y.tolist()),
            "block_start_avg_ms": float(dp2_block_start.get("block_start_avg_ms", 0.0)),
            "steady_state_avg_ms": float(dp2_block_start.get("steady_state_avg_ms", 0.0)),
            "block_start_avg_skew": float(dp2_block_start.get("block_start_avg_skew", 0.0)),
            "steady_state_avg_skew": float(dp2_block_start.get("steady_state_avg_skew", 0.0)),
        }
    else:
        out["timing_reference"] = {"warning": "dp2_block_start_timing missing per_fwd data"}

    # 1B) Trace-wide load proxy stats under linear placement.
    linear = _linear_placement()
    eb_skews = []
    eb_max_frac = []
    noeb_skews = []
    noeb_max_frac = []
    by_path = defaultdict(lambda: {"eb_skew": [], "eb_max_frac": []})
    by_layer = defaultdict(lambda: {"eb_skew": [], "eb_max_frac": []})
    for r in records:
        l_eb = _ep_loads(r["eb_hist4"], linear)
        _, sk_eb, mf_eb = _load_metrics(l_eb)
        eb_skews.append(sk_eb)
        eb_max_frac.append(mf_eb)
        by_path[r["path"]]["eb_skew"].append(sk_eb)
        by_path[r["path"]]["eb_max_frac"].append(mf_eb)
        by_layer[r["layer"]]["eb_skew"].append(sk_eb)
        by_layer[r["layer"]]["eb_max_frac"].append(mf_eb)

        l_n4 = _ep_loads(r["noeb_hist4"], linear)
        _, sk_n4, mf_n4 = _load_metrics(l_n4)
        noeb_skews.append(sk_n4)
        noeb_max_frac.append(mf_n4)

    out["trace_proxy"] = {
        "num_records": int(len(records)),
        "eb_skew": _stats(eb_skews),
        "eb_max_load_frac": _stats(eb_max_frac),
        "noeb_skew": _stats(noeb_skews),
        "noeb_max_load_frac": _stats(noeb_max_frac),
        "by_path": {
            k: {
                "eb_skew": _stats(v["eb_skew"]),
                "eb_max_load_frac": _stats(v["eb_max_frac"]),
            }
            for k, v in sorted(by_path.items())
        },
        "by_layer": {
            str(k): {
                "eb_skew": _stats(v["eb_skew"]),
                "eb_max_load_frac": _stats(v["eb_max_frac"]),
            }
            for k, v in sorted(by_layer.items())
        },
    }
    return out


def _stage2(records: list[dict]) -> dict:
    pop_eb = np.zeros(NUM_EXPERTS, dtype=np.float64)
    for r in records:
        pop_eb += r["eb_hist4"].astype(np.float64)

    linear = _linear_placement()
    rnd = _random_balanced_placement(seed=20260429)
    pop_rr = _popularity_round_robin(pop_eb)
    nbr = _coact_neighbors(records, route_key="eb_hist4", top_n=16)
    coact = _coact_balanced_placement(pop_eb, nbr, lambda_penalty=0.35)

    placement_eval = {
        "eb_hist4": {
            "linear": _evaluate_placement(records, "eb_hist4", linear),
            "random_balanced": _evaluate_placement(records, "eb_hist4", rnd),
            "popularity_round_robin": _evaluate_placement(records, "eb_hist4", pop_rr),
            "coactivation_balanced": _evaluate_placement(records, "eb_hist4", coact),
        },
        "noeb_hist4": {
            "linear": _evaluate_placement(records, "noeb_hist4", linear),
            "random_balanced": _evaluate_placement(records, "noeb_hist4", rnd),
            "popularity_round_robin": _evaluate_placement(records, "noeb_hist4", pop_rr),
            "coactivation_balanced": _evaluate_placement(records, "noeb_hist4", coact),
        },
    }

    # Oracle-like dynamic lower proxy on EB route.
    oracle_mx = []
    oracle_frac = []
    for r in records:
        mx, frac = _oracle_lpt_max(r["eb_hist4"])
        oracle_mx.append(mx)
        oracle_frac.append(frac)
    oracle = {
        "num_records": int(len(records)),
        "max_load": _stats(oracle_mx),
        "max_load_frac": _stats(oracle_frac),
    }

    best_name = _select_best_placement({"placement_eval": placement_eval})
    return {
        "placement_eval": placement_eval,
        "oracle_dynamic_lpt": oracle,
        "placement_meta": {
            "best_static_by_eb_p95_max_load_frac": best_name,
            "random_seed": 20260429,
            "coactivation_top_n": 16,
            "coactivation_lambda_penalty": 0.35,
        },
    }


def _stage3(records: list[dict], stage2: dict) -> dict:
    pop_eb = np.zeros(NUM_EXPERTS, dtype=np.float64)
    for r in records:
        pop_eb += r["eb_hist4"].astype(np.float64)

    linear = _linear_placement()
    rnd = _random_balanced_placement(seed=20260429)
    pop_rr = _popularity_round_robin(pop_eb)
    nbr = _coact_neighbors(records, route_key="eb_hist4", top_n=16)
    coact = _coact_balanced_placement(pop_eb, nbr, lambda_penalty=0.35)

    base_name = stage2["placement_meta"]["best_static_by_eb_p95_max_load_frac"]
    placement_map = {
        "linear": linear,
        "random_balanced": rnd,
        "popularity_round_robin": pop_rr,
        "coactivation_balanced": coact,
    }
    base = placement_map[base_name]
    base_eval = _evaluate_placement(records, "eb_hist4", base)

    sweep = {}
    for pct in [0.01, 0.02, 0.05, 0.10]:
        rep_map = _build_replica_map(base, pop_eb, pct=pct)
        rep_eval = _apply_replicas_and_score(
            records, "eb_hist4", base_placement=base, rep_map=rep_map
        )
        p = f"{int(round(pct * 100)):02d}pct"
        sweep[p] = {
            "pct": float(pct),
            "replicated_experts": int(len(rep_map)),
            "expert_copy_overhead_ratio": _safe_div(float(len(rep_map)), float(NUM_EXPERTS)),
            "eval": rep_eval,
            "delta_vs_base": {
                "p95_max_load_frac_abs": float(
                    rep_eval["max_load_frac"]["p95"] - base_eval["max_load_frac"]["p95"]
                ),
                "p95_max_load_abs": float(
                    rep_eval["max_load"]["p95"] - base_eval["max_load"]["p95"]
                ),
                "p95_skew_abs": float(rep_eval["skew"]["p95"] - base_eval["skew"]["p95"]),
            },
        }

    return {
        "base_placement": base_name,
        "base_eval": base_eval,
        "sweep": sweep,
    }


def _stage4(records: list[dict]) -> dict:
    blocks = sorted({int(r["block"]) for r in records})
    if len(blocks) < 2:
        return {"warning": "insufficient blocks for held-out split"}

    cut_idx = max(1, int(math.floor(len(blocks) * 0.6)))
    if cut_idx >= len(blocks):
        cut_idx = len(blocks) - 1
    train_blocks = set(blocks[:cut_idx])
    test_blocks = set(blocks[cut_idx:])

    train = [r for r in records if r["block"] in train_blocks]
    test = [r for r in records if r["block"] in test_blocks]

    # Layer-wise prior by train no-EB demand.
    layer_pop_train = defaultdict(lambda: np.zeros(NUM_EXPERTS, dtype=np.float64))
    layer_pop_all = defaultdict(lambda: np.zeros(NUM_EXPERTS, dtype=np.float64))
    layer_pop_train_eb = defaultdict(lambda: np.zeros(NUM_EXPERTS, dtype=np.float64))
    for r in train:
        layer_pop_train[r["layer"]] += r["noeb_hist4"].astype(np.float64)
        layer_pop_train_eb[r["layer"]] += r["eb_hist4"].astype(np.float64)
    for r in records:
        layer_pop_all[r["layer"]] += r["noeb_hist4"].astype(np.float64)

    cov_eb = []
    cov_prior_train = []
    cov_prior_oracle = []
    for r in test:
        h = r["noeb_hist4"]
        k = int(r["s_size"])
        layer = int(r["layer"])
        eb_keep = r["s_active"]
        k = max(1, min(k, NUM_EXPERTS))

        tr_order = np.argsort(-layer_pop_train[layer], kind="mergesort")[:k]
        or_order = np.argsort(-layer_pop_all[layer], kind="mergesort")[:k]

        cov_eb.append(_coverage_for_set(h, eb_keep))
        cov_prior_train.append(_coverage_for_set(h, tr_order))
        cov_prior_oracle.append(_coverage_for_set(h, or_order))

    # Prior-placement transfer on EB route.
    pop_train_eb = np.zeros(NUM_EXPERTS, dtype=np.float64)
    pop_all_eb = np.zeros(NUM_EXPERTS, dtype=np.float64)
    for r in train:
        pop_train_eb += r["eb_hist4"].astype(np.float64)
    for r in records:
        pop_all_eb += r["eb_hist4"].astype(np.float64)

    placement_linear = _linear_placement()
    placement_train = _popularity_round_robin(pop_train_eb)
    placement_oracle = _popularity_round_robin(pop_all_eb)

    placement_eval = {
        "test_eb_hist4": {
            "linear": _evaluate_placement(test, "eb_hist4", placement_linear),
            "train_prior_pop_rr": _evaluate_placement(test, "eb_hist4", placement_train),
            "oracle_all_pop_rr": _evaluate_placement(test, "eb_hist4", placement_oracle),
        }
    }

    return {
        "split": {
            "all_blocks": blocks,
            "train_blocks": sorted(train_blocks),
            "test_blocks": sorted(test_blocks),
            "num_train_records": int(len(train)),
            "num_test_records": int(len(test)),
        },
        "coverage_transfer_test": {
            "eb_smask": _stats(cov_eb),
            "train_prior_same_size": _stats(cov_prior_train),
            "oracle_all_same_size": _stats(cov_prior_oracle),
            "delta_train_prior_minus_eb_mean": float(
                np.mean(cov_prior_train) - np.mean(cov_eb)
            ),
        },
        "placement_transfer_test": placement_eval,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trace",
        action="append",
        required=True,
        help="Path to eb_heteval512_laws rank json.gz trace. Repeat for multiple ranks.",
    )
    ap.add_argument(
        "--timing-ref",
        type=str,
        default="",
        help="Optional per-forward timing reference json (dp2_block_start_timing.json).",
    )
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    trace_paths = [Path(x) for x in args.trace]
    rank_traces = [_load_rank_trace(p) for p in trace_paths]
    records = []
    for rt in rank_traces:
        records.extend(rt["records"])

    timing_ref = None
    if args.timing_ref:
        tp = Path(args.timing_ref)
        if tp.exists():
            timing_ref = json.loads(tp.read_text())

    stage1 = _stage1(records, timing_ref)
    stage2 = _stage2(records)
    stage3 = _stage3(records, stage2)
    stage4 = _stage4(records)

    out = {
        "meta": {
            "trace_files": [str(p) for p in trace_paths],
            "num_rank_traces": int(len(rank_traces)),
            "num_records": int(len(records)),
            "num_experts": NUM_EXPERTS,
            "ep_size": EP_SIZE,
            "experts_per_ep": EXPERTS_PER_EP,
            "timing_ref": str(args.timing_ref) if args.timing_ref else "",
        },
        "stage1_skew_latency": stage1,
        "stage2_placement_upper_bound": stage2,
        "stage3_replica_budget_sweep": stage3,
        "stage4_global_prior_transfer": stage4,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

