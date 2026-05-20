#!/usr/bin/env python3
"""Summarize BSP-MoE Nsight Systems traces.

The script compares one baseline trace and one BSP trace. It filters CUDA work
to the global NVTX generate window, because full-process traces include model
loading and warmup noise that should not be attributed to the measured run.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
RESULTS_DIR = REPO_ROOT / "codex_coding" / "results"

DEFAULT_BASELINE_SQLITE = RESULTS_DIR / "nsys_bsp_A_short_nvtx_8g_20260427.sqlite"
DEFAULT_BSP_SQLITE = RESULTS_DIR / "nsys_bsp_B_short_nvtx_8g_20260427.sqlite"
DEFAULT_BASELINE_LOG = RESULTS_DIR / "nsys_bsp_A_short_nvtx_8g_20260427.log"
DEFAULT_BSP_LOG = RESULTS_DIR / "nsys_bsp_B_short_nvtx_8g_20260427.log"
DEFAULT_COMPONENT_JSON = (
    RESULTS_DIR / "bsp_moe_c12_8g_component_summary_20260427.json"
)
DEFAULT_OUT_JSON = RESULTS_DIR / "nsys_bsp_short_nvtx_analysis_20260427.json"
DEFAULT_OUT_MD = RESULTS_DIR / "nsys_bsp_short_nvtx_analysis_20260427.md"

COMPONENT_NAMES = [
    "moe.bsp_chunk",
    "moe.shared",
    "moe.gate_logits",
    "moe.native_forward",
    "moe.dispatch",
    "moe.quant_apply",
    "moe.combine",
    "moe.tp_all_reduce",
    "moe.tp_all_gather",
]


def ns_to_ms(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return float(value) / 1e6


def pct_delta(base: float, new: float) -> float | None:
    if base == 0:
        return None
    return (new - base) / base * 100.0


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def fmt_pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    run_re = re.compile(
        r"Run\s+(?P<run>\d+):\s+"
        r"(?P<time_s>[\d.]+)s,\s+"
        r"(?P<fwd>\d+)\s+fwd,\s+"
        r"(?P<ms_fwd>[\d.]+)\s+ms/fwd,\s+"
        r"cold=(?P<cold>\d+)\s+hot=(?P<hot>\d+)\s+"
        r"eb_skip=(?P<eb_skip>\d+)\s+path=(?P<path>\{.*?\})"
    )
    runs = []
    for match in run_re.finditer(text):
        path_counts: dict[str, Any]
        try:
            path_counts = ast.literal_eval(match.group("path"))
        except Exception:
            path_counts = {"raw": match.group("path")}
        runs.append(
            {
                "run": int(match.group("run")),
                "time_s": float(match.group("time_s")),
                "fwd": int(match.group("fwd")),
                "ms_fwd": float(match.group("ms_fwd")),
                "cold": int(match.group("cold")),
                "hot": int(match.group("hot")),
                "eb_skip": int(match.group("eb_skip")),
                "path": path_counts,
            }
        )
    target_match = re.search(r"profile_target=([a-z_]+)", text)
    return {
        "path": str(path),
        "profile_target": target_match.group(1) if target_match else None,
        "runs": runs,
        "last_run": runs[-1] if runs else None,
    }


def classify_kernel(name: str) -> str:
    lower = name.lower()
    if lower.startswith("nccldevkernel"):
        return "nccl"
    if "cross_device_reduce" in lower:
        return "vllm_cross_device_reduce"
    if "fused_moe_kernel" in lower:
        return "moe_fused_kernel"
    if "vllm::moe::" in lower:
        return "moe_aux_kernel"
    if "_fused_routing" in lower or "gathertopk" in lower or "topk" in lower:
        return "routing_topk"
    if "flash_fwd" in lower or "fmha" in lower or "attention" in lower:
        return "attention"
    if "gemm" in lower or "cublas" in lower or "cutlass::kernel" in lower:
        return "dense_gemm"
    if "catarraybatchedcopy" in lower:
        return "aten_cat_copy"
    if "copy" in lower:
        return "aten_copy"
    if "argmax" in lower or "softmax" in lower or "where" in lower:
        return "sampling_reduction"
    if "elementwise" in lower or "triton_" in lower:
        return "elementwise_misc"
    return "other"


def classify_collective(name: str) -> str | None:
    if "cross_device_reduce" in name:
        return "vLLM_cross_device_reduce"
    if not name.startswith("ncclDevKernel"):
        return None
    if "AllGather" in name:
        return "NCCL_AllGather"
    if "AllReduce" in name:
        return "NCCL_AllReduce"
    if "Reduce_" in name:
        return "NCCL_Reduce"
    if "Broadcast" in name:
        return "NCCL_Broadcast"
    return "NCCL_Other"


def new_metric() -> dict[str, Any]:
    return {
        "count_total": 0,
        "total_ms_sum": 0.0,
        "by_rank": defaultdict(lambda: {"count": 0, "ms": 0.0}),
    }


def add_metric(bucket: dict[str, Any], rank_key: str, dur_ms: float) -> None:
    bucket["count_total"] += 1
    bucket["total_ms_sum"] += dur_ms
    bucket["by_rank"][rank_key]["count"] += 1
    bucket["by_rank"][rank_key]["ms"] += dur_ms


def finalize_metric(bucket: dict[str, Any], fwd: int | None) -> dict[str, Any]:
    by_rank = {
        key: {"count": int(val["count"]), "ms": val["ms"]}
        for key, val in bucket["by_rank"].items()
    }
    if by_rank:
        rankmax_key, rankmax = max(by_rank.items(), key=lambda item: item[1]["ms"])
    else:
        rankmax_key, rankmax = None, {"count": 0, "ms": 0.0}
    per_fwd = rankmax["ms"] / fwd if fwd else None
    return {
        "count_total": int(bucket["count_total"]),
        "total_ms_sum": bucket["total_ms_sum"],
        "rankmax_key": rankmax_key,
        "rankmax_count": int(rankmax["count"]),
        "rankmax_ms": rankmax["ms"],
        "rankmax_ms_per_fwd": per_fwd,
        "by_rank": by_rank,
    }


def summarize_nvtx_components(
    con: sqlite3.Connection, fwd: int | None
) -> dict[str, Any]:
    raw: dict[str, dict[str, Any]] = {name: new_metric() for name in COMPONENT_NAMES}
    rows = con.execute(
        """
        select text, globalTid, end - start as dur_ns
        from NVTX_EVENTS
        where text in ({})
        """.format(",".join(["?"] * len(COMPONENT_NAMES))),
        COMPONENT_NAMES,
    )
    for text, global_tid, dur_ns in rows:
        add_metric(raw[text], str(global_tid), ns_to_ms(dur_ns))
    return {name: finalize_metric(metric, fwd) for name, metric in raw.items()}


def get_generate_ranges(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        select start, end, text, globalTid
        from NVTX_EVENTS
        where text like '%.generate.run%'
        order by start
        """
    ).fetchall()
    return [
        {
            "start": int(start),
            "end": int(end),
            "text": text,
            "globalTid": int(global_tid),
            "duration_ms": ns_to_ms(end - start),
        }
        for start, end, text, global_tid in rows
    ]


def summarize_generate_window(ranges: list[dict[str, Any]]) -> dict[str, Any]:
    if not ranges:
        raise RuntimeError("No '*.generate.run*' NVTX ranges found.")
    start = min(item["start"] for item in ranges)
    end = max(item["end"] for item in ranges)
    durations = [item["duration_ms"] for item in ranges]
    return {
        "start": start,
        "end": end,
        "num_ranges": len(ranges),
        "global_window_ms": ns_to_ms(end - start),
        "rankmax_ms": max(durations),
        "rankmean_ms": sum(durations) / len(durations),
        "sum_rank_ms": sum(durations),
        "ranges": ranges,
    }


def summarize_kernels(
    con: sqlite3.Connection, start: int, end: int, fwd: int | None
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    category_raw: dict[str, dict[str, Any]] = defaultdict(new_metric)
    collective_raw: dict[str, dict[str, Any]] = defaultdict(new_metric)
    top_raw: dict[str, dict[str, Any]] = defaultdict(new_metric)

    rows = con.execute(
        """
        select k.globalPid, k.deviceId, coalesce(s.value, '<unknown>') as name,
               k.end - k.start as dur_ns
        from CUPTI_ACTIVITY_KIND_KERNEL k
        left join StringIds s on k.demangledName = s.id
        where k.start >= ? and k.end <= ?
        """,
        (start, end),
    )
    for global_pid, device_id, name, dur_ns in rows:
        rank_key = f"{global_pid}/gpu{device_id}"
        dur_ms = ns_to_ms(dur_ns)
        add_metric(category_raw[classify_kernel(name)], rank_key, dur_ms)
        add_metric(top_raw[name], rank_key, dur_ms)
        coll = classify_collective(name)
        if coll is not None:
            add_metric(collective_raw[coll], rank_key, dur_ms)

    categories = {
        key: finalize_metric(metric, fwd)
        for key, metric in sorted(category_raw.items())
    }
    collectives = {
        key: finalize_metric(metric, fwd)
        for key, metric in sorted(collective_raw.items())
    }
    top = []
    for name, metric in top_raw.items():
        item = finalize_metric(metric, fwd)
        item["name"] = name
        top.append(item)
    top.sort(key=lambda item: item["total_ms_sum"], reverse=True)
    return categories, collectives, top


def summarize_memcpy(
    con: sqlite3.Connection, start: int, end: int, fwd: int | None
) -> dict[str, Any]:
    raw: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count_total": 0,
            "total_ms_sum": 0.0,
            "total_mb_sum": 0.0,
            "by_rank": defaultdict(lambda: {"count": 0, "ms": 0.0, "mb": 0.0}),
        }
    )
    rows = con.execute(
        """
        select m.globalPid, m.deviceId, coalesce(e.label, cast(m.copyKind as text)),
               m.bytes, m.end - m.start as dur_ns
        from CUPTI_ACTIVITY_KIND_MEMCPY m
        left join ENUM_CUDA_MEMCPY_OPER e on m.copyKind = e.id
        where m.start >= ? and m.end <= ?
        """,
        (start, end),
    )
    for global_pid, device_id, label, nbytes, dur_ns in rows:
        rank_key = f"{global_pid}/gpu{device_id}"
        dur_ms = ns_to_ms(dur_ns)
        mb = float(nbytes) / 1e6
        item = raw[label]
        item["count_total"] += 1
        item["total_ms_sum"] += dur_ms
        item["total_mb_sum"] += mb
        item["by_rank"][rank_key]["count"] += 1
        item["by_rank"][rank_key]["ms"] += dur_ms
        item["by_rank"][rank_key]["mb"] += mb

    out = {}
    for label, item in sorted(raw.items()):
        by_rank = {
            key: {"count": int(val["count"]), "ms": val["ms"], "mb": val["mb"]}
            for key, val in item["by_rank"].items()
        }
        if by_rank:
            rankmax_key, rankmax = max(by_rank.items(), key=lambda kv: kv[1]["ms"])
        else:
            rankmax_key, rankmax = None, {"count": 0, "ms": 0.0, "mb": 0.0}
        out[label] = {
            "count_total": int(item["count_total"]),
            "total_ms_sum": item["total_ms_sum"],
            "total_mb_sum": item["total_mb_sum"],
            "rankmax_key": rankmax_key,
            "rankmax_count": int(rankmax["count"]),
            "rankmax_ms": rankmax["ms"],
            "rankmax_mb": rankmax["mb"],
            "rankmax_ms_per_fwd": rankmax["ms"] / fwd if fwd else None,
            "rankmax_mb_per_fwd": rankmax["mb"] / fwd if fwd else None,
            "by_rank": by_rank,
        }
    return out


def summarize_runtime(
    con: sqlite3.Connection, start: int, end: int, fwd: int | None
) -> list[dict[str, Any]]:
    raw: dict[str, dict[str, Any]] = defaultdict(new_metric)
    rows = con.execute(
        """
        select r.globalTid, coalesce(s.value, '<unknown>') as name,
               r.end - r.start as dur_ns
        from CUPTI_ACTIVITY_KIND_RUNTIME r
        left join StringIds s on r.nameId = s.id
        where r.start >= ? and r.end <= ?
        """,
        (start, end),
    )
    for global_tid, name, dur_ns in rows:
        add_metric(raw[name], str(global_tid), ns_to_ms(dur_ns))
    out = []
    for name, metric in raw.items():
        item = finalize_metric(metric, fwd)
        item["name"] = name
        out.append(item)
    out.sort(key=lambda item: item["total_ms_sum"], reverse=True)
    return out


def analyze_one(label: str, sqlite_path: Path, log_path: Path) -> dict[str, Any]:
    log_info = parse_log(log_path)
    fwd = log_info["last_run"]["fwd"] if log_info["last_run"] else None
    con = sqlite3.connect(sqlite_path)
    try:
        ranges = get_generate_ranges(con)
        window = summarize_generate_window(ranges)
        start = window["start"]
        end = window["end"]
        components = summarize_nvtx_components(con, fwd)
        categories, collectives, top_kernels = summarize_kernels(con, start, end, fwd)
        memcpy = summarize_memcpy(con, start, end, fwd)
        runtime = summarize_runtime(con, start, end, fwd)
    finally:
        con.close()
    return {
        "label": label,
        "sqlite_path": str(sqlite_path),
        "log": log_info,
        "fwd": fwd,
        "generate_window": window,
        "nvtx_components": components,
        "kernel_categories": categories,
        "collectives": collectives,
        "memcpy": memcpy,
        "runtime_top": runtime[:20],
        "top_kernels": top_kernels[:30],
    }


def load_prior_component_timing(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text())
    return {
        "path": str(path),
        "component_timing": data.get("component_timing", {}),
    }


def compare_metric(
    a: dict[str, Any], b: dict[str, Any], path: list[str]
) -> tuple[Any, Any, float | None]:
    va: Any = a
    vb: Any = b
    for key in path:
        va = va.get(key, {}) if isinstance(va, dict) else None
        vb = vb.get(key, {}) if isinstance(vb, dict) else None
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        return va, vb, pct_delta(float(va), float(vb))
    return va, vb, None


def make_compare_table(
    a: dict[str, Any],
    b: dict[str, Any],
    items: list[tuple[str, list[str], int]],
) -> str:
    rows = []
    for name, path, digits in items:
        va, vb, delta = compare_metric(a, b, path)
        rows.append([name, fmt_num(va, digits), fmt_num(vb, digits), fmt_pct(delta)])
    return markdown_table(["Metric", "A baseline", "B BSP", "B vs A"], rows)


def make_report(result: dict[str, Any]) -> str:
    a = result["baseline"]
    b = result["bsp"]
    lines = [
        "# BSP-MoE Nsight Systems Short Trace Analysis",
        "",
        "## Inputs",
        markdown_table(
            ["Trace", "SQLite", "Log", "Fwd"],
            [
                [
                    "A baseline",
                    a["sqlite_path"],
                    a["log"]["path"],
                    a["fwd"],
                ],
                [
                    "B BSP",
                    b["sqlite_path"],
                    b["log"]["path"],
                    b["fwd"],
                ],
            ],
        ),
        "",
        "## E2E And Generate Window",
        make_compare_table(
            a,
            b,
            [
                ("log time s", ["log", "last_run", "time_s"], 3),
                ("log ms/fwd", ["log", "last_run", "ms_fwd"], 2),
                ("NVTX rankmax generate ms", ["generate_window", "rankmax_ms"], 3),
                ("NVTX global window ms", ["generate_window", "global_window_ms"], 3),
                ("NVTX sum rank ms", ["generate_window", "sum_rank_ms"], 3),
            ],
        ),
        "",
        "## NVTX Components",
    ]

    component_rows = []
    for name in COMPONENT_NAMES:
        ca = a["nvtx_components"].get(name, {})
        cb = b["nvtx_components"].get(name, {})
        va = ca.get("rankmax_ms_per_fwd")
        vb = cb.get("rankmax_ms_per_fwd")
        component_rows.append(
            [
                name,
                fmt_num(ca.get("rankmax_count"), 0),
                fmt_num(va, 3),
                fmt_num(cb.get("rankmax_count"), 0),
                fmt_num(vb, 3),
                fmt_pct(pct_delta(va, vb) if va is not None and vb is not None else None),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Component",
                "A count/rank",
                "A rankmax ms/fwd",
                "B count/rank",
                "B rankmax ms/fwd",
                "B vs A",
            ],
            component_rows,
        )
    )

    lines.extend(["", "## Kernel Categories"])
    categories = sorted(
        set(a["kernel_categories"].keys()) | set(b["kernel_categories"].keys())
    )
    cat_rows = []
    for name in categories:
        ca = a["kernel_categories"].get(name, {})
        cb = b["kernel_categories"].get(name, {})
        va = ca.get("rankmax_ms_per_fwd")
        vb = cb.get("rankmax_ms_per_fwd")
        cat_rows.append(
            [
                name,
                fmt_num(ca.get("count_total"), 0),
                fmt_num(va, 3),
                fmt_num(cb.get("count_total"), 0),
                fmt_num(vb, 3),
                fmt_pct(pct_delta(va, vb) if va is not None and vb is not None else None),
            ]
        )
    cat_rows.sort(key=lambda row: float(row[4]) if row[4] != "n/a" else -1.0, reverse=True)
    lines.append(
        markdown_table(
            [
                "Category",
                "A total count",
                "A rankmax ms/fwd",
                "B total count",
                "B rankmax ms/fwd",
                "B vs A",
            ],
            cat_rows,
        )
    )

    lines.extend(["", "## Collective Split"])
    collectives = sorted(set(a["collectives"].keys()) | set(b["collectives"].keys()))
    coll_rows = []
    for name in collectives:
        ca = a["collectives"].get(name, {})
        cb = b["collectives"].get(name, {})
        va = ca.get("rankmax_ms_per_fwd")
        vb = cb.get("rankmax_ms_per_fwd")
        coll_rows.append(
            [
                name,
                fmt_num(ca.get("count_total"), 0),
                fmt_num(ca.get("total_ms_sum"), 1),
                fmt_num(va, 3),
                fmt_num(cb.get("count_total"), 0),
                fmt_num(cb.get("total_ms_sum"), 1),
                fmt_num(vb, 3),
                fmt_pct(pct_delta(va, vb) if va is not None and vb is not None else None),
            ]
        )
    coll_rows.sort(key=lambda row: float(row[6]) if row[6] != "n/a" else -1.0, reverse=True)
    lines.append(
        markdown_table(
            [
                "Collective",
                "A count",
                "A total ms",
                "A rankmax ms/fwd",
                "B count",
                "B total ms",
                "B rankmax ms/fwd",
                "B vs A",
            ],
            coll_rows,
        )
    )

    lines.extend(["", "## Memcpy"])
    labels = sorted(set(a["memcpy"].keys()) | set(b["memcpy"].keys()))
    memcpy_rows = []
    for label in labels:
        ma = a["memcpy"].get(label, {})
        mb = b["memcpy"].get(label, {})
        memcpy_rows.append(
            [
                label,
                fmt_num(ma.get("total_mb_sum"), 1),
                fmt_num(ma.get("rankmax_ms_per_fwd"), 3),
                fmt_num(mb.get("total_mb_sum"), 1),
                fmt_num(mb.get("rankmax_ms_per_fwd"), 3),
                fmt_pct(
                    pct_delta(ma.get("total_mb_sum", 0.0), mb.get("total_mb_sum", 0.0))
                ),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Memcpy kind",
                "A total MB",
                "A rankmax ms/fwd",
                "B total MB",
                "B rankmax ms/fwd",
                "B MB vs A",
            ],
            memcpy_rows,
        )
    )

    prior = result.get("prior_component_timing")
    if prior:
        lines.extend(["", "## Prior C12 CUDA-Event Component Timing"])
        timing = prior.get("component_timing", {})
        comp_a = timing.get("A) C12-AgRs baseline", {})
        comp_b = timing.get("B) C12-BSP-MoE", {})
        names = sorted(
            set(comp_a.get("components", {}).keys())
            | set(comp_b.get("components", {}).keys())
        )
        rows = []
        for name in names:
            va = comp_a.get("components", {}).get(name)
            vb = comp_b.get("components", {}).get(name)
            delta = (
                pct_delta(va, vb)
                if isinstance(va, (int, float)) and isinstance(vb, (int, float))
                else None
            )
            rows.append([name, fmt_num(va, 3), fmt_num(vb, 3), fmt_pct(delta)])
        lines.append(
            markdown_table(
                ["Component", "A ms/fwd", "B ms/fwd", "B vs A"], rows
            )
        )
        byte_names = sorted(
            set(comp_a.get("bytes", {}).keys()) | set(comp_b.get("bytes", {}).keys())
        )
        if byte_names:
            rows = []
            for name in byte_names:
                va = comp_a.get("bytes", {}).get(name)
                vb = comp_b.get("bytes", {}).get(name)
                delta = (
                    pct_delta(va, vb)
                    if isinstance(va, (int, float))
                    and isinstance(vb, (int, float))
                    else None
                )
                rows.append([name, fmt_num(va, 3), fmt_num(vb, 3), fmt_pct(delta)])
            lines.append("")
            lines.append(
                markdown_table(
                    ["Payload", "A MB/fwd", "B MB/fwd", "B vs A"], rows
                )
            )

    lines.extend(["", "## Top Kernels By Total GPU Time"])
    for title, trace in [("A baseline", a), ("B BSP", b)]:
        rows = []
        for item in trace["top_kernels"][:12]:
            rows.append(
                [
                    item["name"][:120],
                    fmt_num(item["count_total"], 0),
                    fmt_num(item["total_ms_sum"], 1),
                    fmt_num(item["rankmax_ms_per_fwd"], 3),
                ]
            )
        lines.append("")
        lines.append(f"### {title}")
        lines.append(
            markdown_table(
                ["Kernel", "Count", "Total ms", "Rankmax ms/fwd"], rows
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "- CUDA kernel and memcpy rows are filtered to the global `*.generate.run*` NVTX window.",
            "- NVTX component rows use CPU NVTX range durations; use them for relative wrapper-level attribution, not as synchronized CUDA-event timings.",
            "- The short NVTX trace did not enable the timed `forward_impl` patch, so `moe.dispatch`, `moe.quant_apply`, and `moe.combine` sub-ranges are unavailable here; the prior C12 CUDA-event table provides those subcomponent numbers.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-sqlite", type=Path, default=DEFAULT_BASELINE_SQLITE)
    parser.add_argument("--bsp-sqlite", type=Path, default=DEFAULT_BSP_SQLITE)
    parser.add_argument("--baseline-log", type=Path, default=DEFAULT_BASELINE_LOG)
    parser.add_argument("--bsp-log", type=Path, default=DEFAULT_BSP_LOG)
    parser.add_argument("--component-json", type=Path, default=DEFAULT_COMPONENT_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    result = {
        "baseline": analyze_one(
            "A baseline", args.baseline_sqlite, args.baseline_log
        ),
        "bsp": analyze_one("B BSP", args.bsp_sqlite, args.bsp_log),
        "prior_component_timing": load_prior_component_timing(args.component_json),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    args.out_md.write_text(make_report(result))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
