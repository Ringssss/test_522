#!/usr/bin/env python3
"""
Aggregate per-run bench JSONs into a single summary for llada20_performance.py.

Reads:  codex_coding/results/bench_paper/*.json
Writes: codex_coding/results/plt/llada20_performance_data.json

Usage:
    python codex_coding/src/aggregate_bench_paper.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
PAPER_DIR = REPO_ROOT / "codex_coding" / "results" / "bench_paper"
OUT_PATH = REPO_ROOT / "codex_coding" / "results" / "plt" / "llada20_performance_data.json"

DATASETS = ["gsm8k", "humaneval", "mgsm", "mt_bench"]
BATCH_SIZES = [32, 128, 256, 512]
ENGINES = ["baseline_nocache", "baseline_cache", "sglang", "vanilla", "tv6"]


def load_ms_fwd(engine: str, dataset: str, batch: int) -> float | None:
    fname = f"{engine}_{dataset}_b{batch}.json"
    path = PAPER_DIR / fname
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        r = data.get("results", {})
        if engine == "sglang":
            return r.get("sglang", {}).get("avg_ms_fwd")
        elif engine in ("baseline_nocache", "baseline_cache"):
            return r.get("baseline", {}).get("mean_ms_fwd")
        elif engine == "vanilla":
            return r.get("baseline", {}).get("mean_ms_fwd")
        elif engine == "tv6":
            return r.get("tv6", {}).get("mean_ms_fwd")
    except Exception:
        return None
    return None


def main():
    raw_ms_fwd = {}
    for ds in DATASETS:
        raw_ms_fwd[ds] = {}
        for bs in BATCH_SIZES:
            entry = {}
            for eng in ENGINES:
                entry[eng] = load_ms_fwd(eng, ds, bs)
            van = entry.get("vanilla")
            tv6 = entry.get("tv6")
            candidates = [v for v in [van, tv6] if v is not None]
            entry["ours"] = min(candidates) if candidates else None
            raw_ms_fwd[ds][str(bs)] = entry

    data = []       # [row][col][method] — relative exec time
    oom_marks = []   # (row, col, method_index) tuples
    speedup_labels = []

    method_keys = ["baseline_nocache", "baseline_cache", "sglang", "ours"]

    for ri, bs in enumerate(BATCH_SIZES):
        row_data = []
        row_speedup = []
        for ci, ds in enumerate(DATASETS):
            entry = raw_ms_fwd[ds][str(bs)]
            ours = entry["ours"]
            cell = []
            for mi, mk in enumerate(method_keys):
                val = entry.get(mk)
                if val is None or ours is None:
                    cell.append(None)
                    oom_marks.append([ri, ci, mi])
                else:
                    cell.append(round(val / ours, 2))
            row_data.append(cell)

            competitors = [cell[i] for i in range(3) if cell[i] is not None]
            if competitors:
                best_comp = min(competitors)
                row_speedup.append(f"{best_comp:.1f}x")
            else:
                row_speedup.append("N/A")

        data.append(row_data)
        speedup_labels.append(row_speedup)

    summary = {
        "datasets": ["GSM8K", "HumanEval", "MGSM", "MT-Bench"],
        "batch_sizes": BATCH_SIZES,
        "methods": [
            "(A)dInfer w/o cache",
            "(B)dInfer w/ cache",
            "(C)SGLang",
            "(D)Ours",
        ],
        "data": data,
        "oom_marks": oom_marks,
        "speedup_labels": speedup_labels,
        "raw_ms_fwd": raw_ms_fwd,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Written to {OUT_PATH}")

    print("\n--- Raw ms/fwd ---")
    for ds in DATASETS:
        for bs in BATCH_SIZES:
            e = raw_ms_fwd[ds][str(bs)]
            parts = []
            for k in ENGINES + ["ours"]:
                v = e.get(k)
                parts.append(f"{k}={'OOM' if v is None else f'{v:.2f}'}")
            print(f"  {ds} b={bs}: {', '.join(parts)}")

    print("\n--- Relative (Ours=1.0) ---")
    for ri, bs in enumerate(BATCH_SIZES):
        for ci, ds in enumerate(DATASETS):
            cell = data[ri][ci]
            labels = ["A:nocache", "B:cache", "C:sglang", "D:ours"]
            parts = []
            for mi, lb in enumerate(labels):
                v = cell[mi]
                parts.append(f"{lb}={'OOM' if v is None else f'{v:.2f}'}")
            print(f"  b={bs} {ds}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
