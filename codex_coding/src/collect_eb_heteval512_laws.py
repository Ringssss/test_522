#!/usr/bin/env python3
"""
Collect EB/dLLM-MoE regularity traces on HetEval-512.

This script is observational. It keeps the current EB routing policy intact and
records compressed routing summaries for later policy design.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

if os.environ.get("BSP_DISABLE_DEEP_EP", "1") != "0":
    sys.modules["deep_ep"] = None

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

from baseline_optimizations import apply_all_optimizations
from test_fused_eb_triton import fused_routing
from test_m_skip_sweep import MSkipEBController

MASK_ID = 156895
EOS_ID = 156892
BLOCK_LENGTH = 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
NUM_EXPERTS = 256


def _parse_layers(raw: str) -> set[int]:
    if raw.strip().lower() in {"", "none"}:
        return set()
    return {int(x) for x in raw.split(",") if x.strip()}


def _hist(ids: torch.Tensor) -> list[int]:
    h = torch.bincount(ids.reshape(-1).to(torch.int64), minlength=NUM_EXPERTS)
    return h[:NUM_EXPERTS].cpu().tolist()


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _route_skip_noop_enabled() -> bool:
    raw = os.environ.get("DINF_EPLB_ROUTE_SKIP_NOOP", "0").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _jaccard_from_sets(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def _percentiles(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _collect_eplb_runtime_diag(model) -> dict | None:
    """Collect lightweight EPLB runtime diag for cross-rank load analysis."""
    if not bool(getattr(model, "_eplb_enabled", False)):
        return None
    load_view = getattr(model, "_eplb_expert_load_view", None)
    sparse_ids = getattr(model, "_eplb_sparse_layer_ids", None)
    if load_view is None or sparse_ids is None:
        return None
    try:
        load_cpu = load_view.detach().to(torch.int64).cpu().tolist()
        owned_phys_ids = []
        for layer_id in sparse_ids:
            experts = model.model.layers[layer_id].mlp.experts
            emap = experts.expert_map.detach()
            owned = torch.nonzero(emap >= 0, as_tuple=False).reshape(-1)
            owned_phys_ids.append(owned.to(torch.int64).cpu().tolist())
        diag = {
            "enabled": True,
            "sparse_layer_ids": [int(x) for x in sparse_ids],
            "load_view": load_cpu,
            "owned_phys_ids": owned_phys_ids,
        }
        state = getattr(model, "_eplb_state", None)
        if state is not None:
            diag["state_step"] = {
                "window_step": int(getattr(state, "expert_load_window_step", -1)),
                "window_size": int(getattr(state, "expert_load_window_size", -1)),
                "rearrangement_step": int(getattr(state, "expert_rearrangement_step", -1)),
                "rearrangement_step_interval": int(
                    getattr(state, "expert_rearrangement_step_interval", -1)
                ),
            }
        return diag
    except Exception as e:  # noqa: BLE001
        return {"enabled": False, "error": str(e)}


def _eplb_state_step_snapshot(model) -> dict:
    st = getattr(model, "_eplb_state", None)
    if st is None:
        return {"has_state": False}
    return {
        "has_state": True,
        "window_step": int(getattr(st, "expert_load_window_step", -1)),
        "window_size": int(getattr(st, "expert_load_window_size", -1)),
        "rearrangement_step": int(getattr(st, "expert_rearrangement_step", -1)),
        "rearrangement_step_interval": int(
            getattr(st, "expert_rearrangement_step_interval", -1)
        ),
    }


def _summarize_eplb_load_balance(all_stats: list[dict]) -> dict:
    diags = []
    for st in all_stats or []:
        if not isinstance(st, dict):
            continue
        d = st.get("eplb_runtime_diag")
        if isinstance(d, dict) and d.get("enabled", False):
            diags.append((int(st.get("rank", -1)), d))
    if not diags:
        return {"enabled": False}

    diags.sort(key=lambda x: x[0])
    ranks = [r for r, _ in diags]
    base_sparse = diags[0][1].get("sparse_layer_ids", [])
    base_load = np.asarray(diags[0][1].get("load_view", []), dtype=np.float64)
    if base_load.ndim != 2 or base_load.size == 0:
        return {"enabled": False, "error": "invalid load_view shape"}
    num_layers, num_phys = base_load.shape

    # Aggregate global physical loads across ranks.
    global_load = np.zeros((num_layers, num_phys), dtype=np.float64)
    for _rank, d in diags:
        lv = np.asarray(d.get("load_view", []), dtype=np.float64)
        if lv.shape != (num_layers, num_phys):
            return {
                "enabled": False,
                "error": f"inconsistent load_view shape: {tuple(lv.shape)} vs {(num_layers, num_phys)}",
            }
        global_load += lv

    # Build per-layer owner map from expert_map ownership.
    layer_skews = []
    layer_max_fracs = []
    layer_rank_load_p95 = []
    bad_owner_layers = 0
    missing_total = 0
    dup_total = 0
    rank_to_idx = {r: i for i, r in enumerate(ranks)}
    rank_load_total = np.zeros(len(ranks), dtype=np.float64)

    for li in range(num_layers):
        owner = np.full((num_phys,), -1, dtype=np.int64)
        for r, d in diags:
            owned_layers = d.get("owned_phys_ids", [])
            if li >= len(owned_layers):
                continue
            for pid in owned_layers[li]:
                p = int(pid)
                if p < 0 or p >= num_phys:
                    continue
                if owner[p] == -1:
                    owner[p] = int(r)
                elif owner[p] != int(r):
                    dup_total += 1

        missing = int(np.sum(owner < 0))
        missing_total += missing
        if missing > 0:
            bad_owner_layers += 1

        rank_loads = np.zeros(len(ranks), dtype=np.float64)
        for p in range(num_phys):
            rr = int(owner[p])
            if rr < 0:
                continue
            idx = rank_to_idx.get(rr)
            if idx is None:
                continue
            rank_loads[idx] += float(global_load[li, p])
        rank_load_total += rank_loads

        pos = rank_loads[rank_loads > 0.0]
        if pos.size == 0:
            continue
        mx = float(pos.max())
        mn = float(pos.min())
        sm = float(pos.sum())
        layer_skews.append(mx / max(mn, 1e-12))
        layer_max_fracs.append(mx / max(sm, 1e-12))
        layer_rank_load_p95.append(float(np.percentile(rank_loads, 95)))

    overall_pos = rank_load_total[rank_load_total > 0.0]
    if overall_pos.size > 0:
        overall_skew = float(overall_pos.max() / max(overall_pos.min(), 1e-12))
        overall_max_frac = float(overall_pos.max() / max(overall_pos.sum(), 1e-12))
    else:
        overall_skew = 0.0
        overall_max_frac = 0.0

    return {
        "enabled": True,
        "num_ranks": int(len(ranks)),
        "num_layers": int(num_layers),
        "num_physical_experts": int(num_phys),
        "sparse_layer_ids": [int(x) for x in base_sparse],
        "layer_skew": _percentiles(layer_skews),
        "layer_max_load_frac": _percentiles(layer_max_fracs),
        "layer_rank_load_p95": _percentiles(layer_rank_load_p95),
        "overall_rank_load": {
            "ranks": [int(x) for x in ranks],
            "loads": [float(x) for x in rank_load_total.tolist()],
            "skew": float(overall_skew),
            "max_load_frac": float(overall_max_frac),
        },
        "owner_map_health": {
            "layers_with_missing_owner": int(bad_owner_layers),
            "missing_owner_slots_total": int(missing_total),
            "duplicate_owner_slots_total": int(dup_total),
        },
    }


class TraceMSkipController(MSkipEBController):
    """MSkip EB controller with explicit block clock and path labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_block_id = -1
        self._last_block_id = {}
        self.path_counts = {
            "prefill_fallback": 0,
            "cold": 0,
            "hot_skip": 0,
            "hot_update": 0,
        }
        self.per_layer_cold = {}
        self.last_path = {}
        self.current_block_id = -1

    def note_block_start(self, block_id: int):
        self.current_block_id = int(block_id)

    def reset_block_clock(self):
        self.current_block_id = -1
        self._last_block_id.clear()
        self.path_counts = {
            "prefill_fallback": 0,
            "cold": 0,
            "hot_skip": 0,
            "hot_update": 0,
        }
        self.per_layer_cold.clear()
        self.last_path.clear()

    def _is_new_block_explicit(self, layer_idx: int):
        if self.current_block_id < 0:
            return None
        prev = self._last_block_id.get(layer_idx, -1)
        if prev != self.current_block_id:
            self._last_block_id[layer_idx] = self.current_block_id
            return True
        return False

    def get_s_mask(self, layer_idx, logits, bias):
        explicit_new = self._is_new_block_explicit(layer_idx)
        if explicit_new is None:
            self.path_counts["prefill_fallback"] += 1
            self.last_path[layer_idx] = "prefill_fallback"
            return super().get_s_mask(layer_idx, logits, bias)

        if explicit_new:
            self.path_counts["cold"] += 1
            self.per_layer_cold[layer_idx] = self.per_layer_cold.get(layer_idx, 0) + 1
            self.last_path[layer_idx] = "cold"
            return self.cold_path(layer_idx, logits, bias)

        prev_calls = self.eb_calls
        prev_skips = self.eb_skips
        out = self.hot_path(layer_idx, logits, bias)
        if self.eb_calls > prev_calls:
            path = "hot_update"
        elif self.eb_skips > prev_skips:
            path = "hot_skip"
        else:
            path = "hot_update"
        self.path_counts[path] += 1
        self.last_path[layer_idx] = path
        return out

    def stats(self):
        return {
            "path_counts": dict(self.path_counts),
            "per_layer_cold": {
                str(k): int(v) for k, v in sorted(self.per_layer_cold.items())
            },
            "cold_count": int(self.cold_count),
            "hot_count": int(self.hot_count),
            "eb_calls": int(self.eb_calls),
            "eb_skips": int(self.eb_skips),
        }


class EBTraceCollector:
    def __init__(
        self,
        *,
        rank: int,
        dp_rank: int,
        tp_rank: int,
        ep_size: int,
        trace_full: bool,
        group_layers: set[int],
        group_size: int,
        block_length: int,
    ):
        self.rank = rank
        self.dp_rank = dp_rank
        self.tp_rank = tp_rank
        self.ep_size = ep_size
        self.trace_full = trace_full
        self.group_layers = group_layers
        self.group_size = group_size
        self.block_length = block_length
        self.records: list[dict] = []
        self.grouping_records: list[dict] = []

    def record(
        self,
        *,
        layer: int,
        block: int,
        fwd_in_block: int,
        path: str,
        s_mask: torch.Tensor,
        eb_ids: torch.Tensor,
        noeb4_ids: torch.Tensor,
        noeb8_ids: torch.Tensor | None,
    ):
        if not self.trace_full:
            return
        n_tokens = int(eb_ids.shape[0])
        eb_hist = _hist(eb_ids)
        noeb4_hist = _hist(noeb4_ids)
        noeb8_hist = _hist(noeb8_ids) if noeb8_ids is not None else []
        s_active = torch.nonzero(s_mask.reshape(-1) > 0, as_tuple=False).reshape(-1)
        s_active_list = s_active.cpu().tolist()
        record = {
            "rank": self.rank,
            "dp_rank": self.dp_rank,
            "tp_rank": self.tp_rank,
            "layer": int(layer),
            "block": int(block),
            "fwd_in_block": int(fwd_in_block),
            "path": path,
            "n_tokens": n_tokens,
            "s_size": int(len(s_active_list)),
            "s_active": s_active_list,
            "eb_hist4": eb_hist,
            "noeb_hist4": noeb4_hist,
            "noeb_hist8": noeb8_hist,
            "eb_unique4": int(sum(1 for x in eb_hist if x > 0)),
            "noeb_unique4": int(sum(1 for x in noeb4_hist if x > 0)),
            "noeb_unique8": int(sum(1 for x in noeb8_hist if x > 0)) if noeb8_hist else None,
        }
        self.records.append(record)

        if (
            path == "cold"
            and layer in self.group_layers
            and n_tokens % self.block_length == 0
        ):
            grouping = self._request_grouping_summary(
                layer=layer,
                block=block,
                fwd_in_block=fwd_in_block,
                eb_ids=eb_ids,
                noeb4_ids=noeb4_ids,
            )
            if grouping is not None:
                self.grouping_records.append(grouping)

    def _request_grouping_summary(
        self,
        *,
        layer: int,
        block: int,
        fwd_in_block: int,
        eb_ids: torch.Tensor,
        noeb4_ids: torch.Tensor,
    ) -> dict | None:
        ids = eb_ids.detach().cpu().numpy()
        noeb = noeb4_ids.detach().cpu().numpy()
        bsz = ids.shape[0] // self.block_length
        if bsz <= 0:
            return None

        def build_bitsets(src: np.ndarray) -> np.ndarray:
            req_bits = np.zeros((bsz, NUM_EXPERTS), dtype=np.bool_)
            src = src.reshape(bsz, self.block_length, src.shape[-1])
            for i in range(bsz):
                req_bits[i, np.unique(src[i].reshape(-1))] = True
            return req_bits

        eb_bits = build_bitsets(ids)
        noeb_bits = build_bitsets(noeb)

        def group_union_mean(bits: np.ndarray, order: np.ndarray) -> float:
            vals = []
            for start in range(0, len(order), self.group_size):
                idx = order[start:start + self.group_size]
                if len(idx) == 0:
                    continue
                vals.append(float(bits[idx].any(axis=0).sum()))
            return float(np.mean(vals)) if vals else 0.0

        natural = np.arange(bsz)
        rng = np.random.default_rng(20260428 + int(layer) * 1000 + int(block))
        random_order = rng.permutation(bsz)
        expert_ids = np.arange(NUM_EXPERTS, dtype=np.float64)
        req_sizes = eb_bits.sum(axis=1).astype(np.float64)
        centroid = (eb_bits.astype(np.float64) @ expert_ids) / np.maximum(req_sizes, 1.0)
        centroid_order = np.argsort(centroid)

        return {
            "rank": self.rank,
            "layer": int(layer),
            "block": int(block),
            "fwd_in_block": int(fwd_in_block),
            "batch_seen": int(bsz),
            "group_size": int(self.group_size),
            "eb_req_unique_mean": float(eb_bits.sum(axis=1).mean()),
            "noeb_req_unique_mean": float(noeb_bits.sum(axis=1).mean()),
            "eb_union_all": int(eb_bits.any(axis=0).sum()),
            "noeb_union_all": int(noeb_bits.any(axis=0).sum()),
            "eb_group_union_natural": group_union_mean(eb_bits, natural),
            "eb_group_union_random": group_union_mean(eb_bits, random_order),
            "eb_group_union_centroid": group_union_mean(eb_bits, centroid_order),
            "noeb_group_union_natural": group_union_mean(noeb_bits, natural),
            "noeb_group_union_random": group_union_mean(noeb_bits, random_order),
            "noeb_group_union_centroid": group_union_mean(noeb_bits, centroid_order),
        }


class ColdGatedRearrangeDriver:
    """Experimental cold-gated EPLB rearrange trigger (script layer PoC)."""

    def __init__(self, *, model, enabled: bool, min_block_gap: int = 8):
        self.model = model
        self.enabled = bool(enabled)
        self.min_block_gap = max(1, int(min_block_gap))
        self.last_trigger_block = -10**9
        self.attempts = 0
        self.success = 0
        self.fail = 0
        self.total_ms = 0.0
        self.fail_reasons: list[str] = []
        self._proxy_model = None
        if self.enabled:
            self._proxy_model = self._build_proxy_model()

    def _build_proxy_model(self):
        state = getattr(self.model, "_eplb_state", None)
        if state is None:
            return None
        sparse_ids = getattr(self.model, "_eplb_sparse_layer_ids", None)
        if not sparse_ids:
            return None
        expert_weights = []
        num_local_physical_experts = None
        for lid in sparse_ids:
            mlp = self.model.model.layers[lid].mlp
            if not hasattr(mlp, "experts"):
                return None
            experts = mlp.experts
            if not hasattr(experts, "w13_weight") or not hasattr(experts, "w2_weight"):
                return None
            # vLLM expects: expert_weights[layer] is an iterable of parameter
            # tensors with shape [num_local_physical_experts, ...].
            w13 = experts.w13_weight
            w2 = experts.w2_weight
            if num_local_physical_experts is None:
                num_local_physical_experts = int(w13.shape[0])
            expert_weights.append((w13, w2))
        if not expert_weights:
            return None

        class _Proxy:
            pass

        p = _Proxy()
        p.num_moe_layers = int(len(sparse_ids))
        p.num_logical_experts = int(self.model.config.num_experts)
        p.num_routed_experts = int(self.model.config.num_experts)
        p.num_physical_experts = int(
            getattr(self.model, "_eplb_num_physical_experts", p.num_logical_experts)
        )
        p.num_redundant_experts = int(max(0, p.num_physical_experts - p.num_logical_experts))
        p.num_local_physical_experts = int(
            num_local_physical_experts
            if num_local_physical_experts is not None
            else p.num_physical_experts
        )
        p.num_expert_groups = 1
        p.num_shared_experts = 0
        p.local_physical_experts = None
        p.expert_weights = expert_weights
        return p

    def maybe_rearrange(self, *, path: str, layer_idx: int, block_id: int):
        if not self.enabled:
            return
        if path != "cold":
            return
        if layer_idx != 0:
            return
        if self._proxy_model is None:
            return
        if block_id - self.last_trigger_block < self.min_block_gap:
            return

        state = getattr(self.model, "_eplb_state", None)
        if state is None:
            return

        self.attempts += 1
        self.last_trigger_block = int(block_id)
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            state.rearrange(self._proxy_model)
            torch.cuda.synchronize()
            self.total_ms += (time.perf_counter() - t0) * 1000.0
            self.success += 1
        except Exception as e:  # noqa: BLE001
            self.fail += 1
            msg = (
                f"{type(e).__name__}: {repr(e)}\n"
                f"{traceback.format_exc(limit=8)}"
            )
            if len(msg) > 2048:
                msg = msg[:2048]
            self.fail_reasons.append(msg)

    def stats(self):
        mean_ms = self.total_ms / max(self.success, 1)
        return {
            "enabled": self.enabled,
            "min_block_gap": int(self.min_block_gap),
            "attempts": int(self.attempts),
            "success": int(self.success),
            "fail": int(self.fail),
            "total_ms": float(self.total_ms),
            "mean_ms_per_success": float(mean_ms),
            "fail_reasons": list(self.fail_reasons[:8]),
        }


class NativeRecordGateController:
    """Gate vLLM native EPLB record_enabled tensor by EB path (script-level)."""

    def __init__(self, *, model, mode: str):
        if mode not in {"off", "full", "cold_only"}:
            raise ValueError(f"Unsupported native record gate mode: {mode}")
        self.mode = str(mode)
        self.tensor = None
        self.total_calls = 0
        self.set_true = 0
        self.set_false = 0
        self.path_counts = defaultdict(int)
        state = getattr(model, "_eplb_state", None)
        if state is not None:
            self.tensor = getattr(state, "should_record_tensor", None)
        self.available = self.tensor is not None

    def update_from_path(self, path: str):
        p = str(path)
        self.path_counts[p] += 1
        if not self.available:
            return
        self.total_calls += 1
        if self.mode == "off":
            enabled = False
        elif self.mode == "full":
            enabled = True
        else:
            enabled = (p == "cold")
        self.tensor.fill_(bool(enabled))
        if enabled:
            self.set_true += 1
        else:
            self.set_false += 1

    def stats(self):
        return {
            "mode": self.mode,
            "available": bool(self.available),
            "total_calls": int(self.total_calls),
            "set_true": int(self.set_true),
            "set_false": int(self.set_false),
            "path_counts_seen": {str(k): int(v) for k, v in sorted(self.path_counts.items())},
        }


def _coverage(hist: list[int], expert_set: set[int]) -> float:
    total = sum(hist)
    if total <= 0:
        return 0.0
    return sum(hist[i] for i in expert_set) / total


def _summarize_payloads(payloads: list[dict], args: argparse.Namespace, all_stats: list[dict]):
    records = []
    grouping = []
    for payload in payloads:
        records.extend(payload.get("records", []))
        grouping.extend(payload.get("grouping_records", []))

    valid = [r for r in records if r["block"] >= 0 and r["path"] != "prefill_fallback"]
    by_layer = defaultdict(list)
    by_path = defaultdict(list)
    by_phase = defaultdict(list)
    for r in valid:
        by_layer[int(r["layer"])].append(r)
        by_path[r["path"]].append(r)
        fib = int(r["fwd_in_block"])
        if fib == 0:
            phase = "cold"
        elif fib <= 2:
            phase = "early_hot"
        elif fib <= 8:
            phase = "mid_hot"
        else:
            phase = "late_hot"
        by_phase[phase].append(r)

    def active_summary(rows: list[dict]) -> dict:
        if not rows:
            return {}
        eb = [float(r["eb_unique4"]) for r in rows]
        no4 = [float(r["noeb_unique4"]) for r in rows]
        no8 = [float(r["noeb_unique8"]) for r in rows if r.get("noeb_unique8") is not None]
        return {
            "calls": len(rows),
            "eb_unique4": _percentiles(eb),
            "noeb_unique4": _percentiles(no4),
            "noeb_unique8": _percentiles(no8),
            "mean_reduction_vs_noeb4_pct": (1.0 - _safe_div(float(np.mean(eb)), float(np.mean(no4)))) * 100.0,
            "mean_reduction_vs_noeb8_pct": (1.0 - _safe_div(float(np.mean(eb)), float(np.mean(no8)))) * 100.0 if no8 else None,
            "s_size": _percentiles([float(r["s_size"]) for r in rows]),
        }

    layer_summary = {str(k): active_summary(v) for k, v in sorted(by_layer.items())}
    path_summary = {k: active_summary(v) for k, v in sorted(by_path.items())}
    phase_summary = {k: active_summary(v) for k, v in sorted(by_phase.items())}

    global_pop_by_layer = {}
    for layer, rows in by_layer.items():
        hist = np.zeros(NUM_EXPERTS, dtype=np.int64)
        for r in rows:
            hist += np.asarray(r["noeb_hist4"], dtype=np.int64)
        global_pop_by_layer[layer] = np.argsort(-hist)

    coverage_eb = []
    coverage_global = []
    active_recall_eb = []
    active_recall_global = []
    for r in valid:
        layer = int(r["layer"])
        s_size = int(r["s_size"])
        s_set = set(int(x) for x in r["s_active"])
        global_set = set(int(x) for x in global_pop_by_layer[layer][:s_size])
        noeb_active = {i for i, v in enumerate(r["noeb_hist4"]) if v > 0}
        coverage_eb.append(_coverage(r["noeb_hist4"], s_set))
        coverage_global.append(_coverage(r["noeb_hist4"], global_set))
        active_recall_eb.append(len(noeb_active & s_set) / max(len(noeb_active), 1))
        active_recall_global.append(len(noeb_active & global_set) / max(len(noeb_active), 1))

    temporal_adj = []
    temporal_vs_cold = []
    temporal_by_fib = defaultdict(list)
    cross_block = []
    grouped = defaultdict(list)
    for r in valid:
        grouped[(r["rank"], r["layer"], r["block"])].append(r)
    for (_rank, _layer, _block), rows in grouped.items():
        rows.sort(key=lambda x: x["fwd_in_block"])
        if not rows:
            continue
        cold_set = set(rows[0]["s_active"])
        for i, r in enumerate(rows):
            curr = set(r["s_active"])
            if i > 0:
                prev = set(rows[i - 1]["s_active"])
                j = _jaccard_from_sets(prev, curr)
                temporal_adj.append(j)
                temporal_by_fib[int(r["fwd_in_block"])].append(j)
            temporal_vs_cold.append(_jaccard_from_sets(cold_set, curr))

    grouped_layer = defaultdict(list)
    for r in valid:
        if r["path"] == "cold":
            grouped_layer[(r["rank"], r["layer"])].append(r)
    for (_rank, _layer), rows in grouped_layer.items():
        rows.sort(key=lambda x: x["block"])
        for a, b in zip(rows, rows[1:]):
            cross_block.append(_jaccard_from_sets(set(a["s_active"]), set(b["s_active"])))

    fib_summary = {
        str(k): _percentiles(v)
        for k, v in sorted(temporal_by_fib.items())
        if len(v) >= 3
    }

    grouping_summary = {}
    if grouping:
        def col(name: str) -> list[float]:
            return [float(x[name]) for x in grouping if name in x]
        grouping_summary = {
            "records": len(grouping),
            "eb_req_unique_mean": _percentiles(col("eb_req_unique_mean")),
            "noeb_req_unique_mean": _percentiles(col("noeb_req_unique_mean")),
            "eb_union_all": _percentiles(col("eb_union_all")),
            "noeb_union_all": _percentiles(col("noeb_union_all")),
            "eb_group_union_natural": _percentiles(col("eb_group_union_natural")),
            "eb_group_union_random": _percentiles(col("eb_group_union_random")),
            "eb_group_union_centroid": _percentiles(col("eb_group_union_centroid")),
            "noeb_group_union_natural": _percentiles(col("noeb_group_union_natural")),
            "noeb_group_union_random": _percentiles(col("noeb_group_union_random")),
            "noeb_group_union_centroid": _percentiles(col("noeb_group_union_centroid")),
        }
        nat = np.mean(col("eb_group_union_natural"))
        cen = np.mean(col("eb_group_union_centroid"))
        grouping_summary["eb_centroid_vs_natural_delta_pct"] = (cen / nat - 1.0) * 100.0 if nat else None

    return {
        "config": {
            "batch_size": args.batch_size,
            "gen_length": args.gen_length,
            "block_length": BLOCK_LENGTH,
            "tp_size": args.tp_size,
            "world_size": args.world_size,
            "dp_size": args.world_size // args.tp_size,
            "trace_rank_mode": args.trace_rank_mode,
            "eplb_runtime_record_mode": args.eplb_runtime_record_mode,
            "eplb_runtime_map_impl": args.eplb_runtime_map_impl,
            "eplb_runtime_tensor_cache": args.eplb_runtime_tensor_cache,
            "eplb_cold_rearrange": bool(args.eplb_cold_rearrange),
            "eplb_cold_rearrange_min_gap": int(args.eplb_cold_rearrange_min_gap),
            "group_layers": sorted(_parse_layers(args.group_layers)),
            "eplb_runtime_state": {
                "num_redundant_experts": int(args.eplb_runtime_redundant_experts),
                "window_size": int(args.eplb_runtime_window_size),
                "step_interval": int(args.eplb_runtime_step_interval),
            },
            "native_record_gate_mode": str(args.native_record_gate_mode),
            "eb": {
                "K": 8,
                "routed_topk": 4,
                "K_target": 40,
                "quality_floor": 0.70,
                "q_major": 1.0,
                "per_round_cap": 8,
                "skip_m": 5,
            },
        },
        "controller_all_ranks": all_stats,
        "eplb_load_balance_diag": _summarize_eplb_load_balance(all_stats),
        "num_trace_payloads": len(payloads),
        "num_records": len(records),
        "num_valid_records": len(valid),
        "overall_active_expert": active_summary(valid),
        "by_path": path_summary,
        "by_phase": phase_summary,
        "by_layer": layer_summary,
        "smask_temporal": {
            "adjacent_jaccard": _percentiles(temporal_adj),
            "vs_cold_jaccard": _percentiles(temporal_vs_cold),
            "cross_block_cold_jaccard": _percentiles(cross_block),
            "adjacent_jaccard_by_fwd_in_block": fib_summary,
        },
        "coverage_control": {
            "weighted_coverage_eb_smask": _percentiles(coverage_eb),
            "weighted_coverage_global_pop_same_size": _percentiles(coverage_global),
            "active_recall_eb_smask": _percentiles(active_recall_eb),
            "active_recall_global_pop_same_size": _percentiles(active_recall_global),
            "weighted_coverage_delta_pct": (
                (float(np.mean(coverage_eb)) / max(float(np.mean(coverage_global)), 1e-12) - 1.0) * 100.0
                if coverage_eb and coverage_global else None
            ),
        },
        "request_grouping": grouping_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--trace-rank-mode", choices=["rank0", "tp0", "all"], default="tp0")
    parser.add_argument("--group-layers", default="0,9,18")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--output-prefix", default="eb_heteval512_laws_20260428")
    parser.add_argument("--eplb-runtime-redundant-experts", type=int, default=0)
    parser.add_argument("--eplb-runtime-window-size", type=int, default=16)
    parser.add_argument("--eplb-runtime-step-interval", type=int, default=16)
    parser.add_argument(
        "--eplb-runtime-record-mode",
        choices=["full", "cold_only", "off"],
        default="full",
        help=(
            "Runtime EPLB record policy: full=map+record, "
            "cold_only=record only on cold path, off=map only."
        ),
    )
    parser.add_argument(
        "--eplb-runtime-map-impl",
        choices=["vllm", "flat_eager"],
        default="vllm",
        help=(
            "Runtime EPLB logical->physical map implementation. "
            "vllm=baseline gather path; flat_eager uses flattened-index path."
        ),
    )
    parser.add_argument(
        "--eplb-runtime-tensor-cache",
        choices=["on", "off"],
        default="off",
        help="Enable/disable cached temporary tensors in runtime EPLB map path.",
    )
    parser.add_argument(
        "--skip-runtime-eplb-state",
        action="store_true",
        help=(
            "Keep constructor-time EPLB physical capacity (e.g. E=34), "
            "but skip model.build_and_set_eplb_runtime_state() so runtime "
            "logical-to-physical mapping is disabled."
        ),
    )
    parser.add_argument(
        "--eplb-cold-rearrange",
        action="store_true",
        help=(
            "Experimental: trigger EPLB rearrange only on cold path at layer-0 "
            "with block-gap gate."
        ),
    )
    parser.add_argument(
        "--eplb-cold-rearrange-min-gap",
        type=int,
        default=8,
        help="Minimum block gap between cold-gated rearrange triggers.",
    )
    parser.add_argument(
        "--native-record-gate-mode",
        choices=["off", "full", "cold_only"],
        default="off",
        help=(
            "Gate vLLM native EPLB should_record tensor by EB path. "
            "off=disabled; full=always record; cold_only=record on cold path only."
        ),
    )
    parser.add_argument(
        "--eplb-force-step",
        action="store_true",
        help=(
            "Experimental diagnostics: force one call to model._eplb_state.step() "
            "after generation to validate whether runtime EPLB state is advancing."
        ),
    )
    parser.add_argument(
        "--eplb-step-per-forward",
        action="store_true",
        help=(
            "Experimental diagnostics: call model._eplb_state.step() after each "
            "model forward during generation."
        ),
    )
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size != args.world_size:
        raise RuntimeError(f"Expected world_size={args.world_size}, got {world_size}")
    if world_size % args.tp_size != 0:
        raise RuntimeError("world_size must be divisible by tp_size")

    dp_size = world_size // args.tp_size
    dp_rank = rank // args.tp_size
    tp_rank = rank % args.tp_size
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import prepare_communication_buffer_for_model
    from vllm.forward_context import set_forward_context
    from dinfer import (
        BlockDiffusionLLM,
        BlockIteratorFactory,
        KVCacheFactory,
        ThresholdParallelDecoder,
    )
    from dinfer.model import LLaDA2MoeModelLM
    from dinfer.model.modeling_llada2_moe import (
        set_eplb_runtime_route_path,
        use_eplb_runtime_map_policy,
    )
    from transformers import AutoConfig, AutoTokenizer
    from test_heteval512 import PROMPTS

    pcfg_init = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=True,
    )
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(world_size, rank, "env://", local_rank, "nccl")

    pcfg = ParallelConfig(
        tensor_parallel_size=args.tp_size,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        enable_expert_parallel=True,
    )
    # EPLB must be configured before model/FusedMoE construction so that
    # physical expert capacity can be materialized in layer weights.
    if int(args.eplb_runtime_redundant_experts) > 0:
        pcfg.enable_eplb = True
        pcfg.eplb_config.num_redundant_experts = int(args.eplb_runtime_redundant_experts)
        pcfg.eplb_config.window_size = int(args.eplb_runtime_window_size)
        pcfg.eplb_config.step_interval = int(args.eplb_runtime_step_interval)
    vllm_cfg = VllmConfig(parallel_config=pcfg)

    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(
            tensor_model_parallel_size=args.tp_size,
            backend="nccl",
        )

        if rank == 0:
            print("=" * 80)
            print("EB HetEval512 Law Probe")
            print(f"  batch={args.batch_size}, gen={args.gen_length}, block={BLOCK_LENGTH}")
            print(f"  dp={dp_size}, tp={args.tp_size}, ep={world_size}")
            print(f"  trace_rank_mode={args.trace_rank_mode}")
            print(f"  eplb_runtime_record_mode={args.eplb_runtime_record_mode}")
            print(f"  eplb_runtime_map_impl={args.eplb_runtime_map_impl}")
            print(f"  eplb_runtime_tensor_cache={args.eplb_runtime_tensor_cache}")
            print(
                "  eplb_cold_rearrange="
                f"{bool(args.eplb_cold_rearrange)}"
                f", gap={int(args.eplb_cold_rearrange_min_gap)}"
            )
            print(f"  native_record_gate_mode={args.native_record_gate_mode}")
            print("=" * 80)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        config = AutoConfig.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        if args.eplb_runtime_redundant_experts > 0:
            if args.skip_runtime_eplb_state:
                # Keep constructor-time physical expert capacity (e.g. E=34),
                # but disable runtime EPLB mapping so MoE runs without logical
                # -> physical remap in forward.
                model.set_eplb_runtime_state(enable_eplb=False)
            else:
                model.build_and_set_eplb_runtime_state(
                    num_redundant_experts=int(args.eplb_runtime_redundant_experts),
                    expert_load_window_size=int(args.eplb_runtime_window_size),
                    expert_rearrangement_step_interval=int(args.eplb_runtime_step_interval),
                    device=device,
                )
        if hasattr(model, "set_bsp_sequence_parallel_moe"):
            model.set_bsp_sequence_parallel_moe(False)

        eplb_step_hook_diag = {
            "enabled": bool(args.eplb_step_per_forward),
            "attached": False,
            "calls": 0,
            "errors": 0,
            "last_error": "",
            "last_traceback": "",
            "before": _eplb_state_step_snapshot(model),
            "after": None,
        }
        if args.eplb_step_per_forward:
            if not hasattr(model, "num_moe_layers"):
                sparse_layers = [
                    li for li in range(model.config.num_hidden_layers)
                    if li >= model.config.first_k_dense_replace
                ]
                model.num_moe_layers = len(sparse_layers)
                model.num_expert_groups = 1
                model.num_logical_experts = int(model.config.num_experts)
                model.num_routed_experts = int(model.config.num_experts)
                model.num_shared_experts = 0
                model.num_physical_experts = int(
                    getattr(model, "_eplb_num_physical_experts", model.num_logical_experts)
                )
                model.num_redundant_experts = int(
                    max(0, model.num_physical_experts - model.num_logical_experts)
                )
                example_mlp = model.model.layers[sparse_layers[0]].mlp if sparse_layers else None
                if example_mlp is not None and hasattr(example_mlp, "experts"):
                    model.num_local_physical_experts = int(
                        getattr(example_mlp.experts, "local_num_experts", model.num_physical_experts)
                    )
                else:
                    model.num_local_physical_experts = int(model.num_physical_experts)
                model.moe_layers = []
                model.expert_weights = []
                for layer_id in sparse_layers:
                    mlp = model.model.layers[layer_id].mlp
                    if hasattr(mlp, "experts"):
                        model.moe_layers.append(mlp.experts)

                def _set_eplb_state_for_step(
                    expert_load_view: torch.Tensor,
                    logical_to_physical_map: torch.Tensor,
                    logical_replica_count: torch.Tensor,
                ) -> None:
                    model.expert_weights = []
                    for li, layer in enumerate(model.moe_layers):
                        model.expert_weights.append(layer.get_expert_weights())
                        layer.set_eplb_state(
                            moe_layer_idx=li,
                            expert_load_view=expert_load_view,
                            logical_to_physical_map=logical_to_physical_map,
                            logical_replica_count=logical_replica_count,
                        )

                model.set_eplb_state = _set_eplb_state_for_step
                # Register once before first step() so rearrange path sees
                # consistent expert_weights length.
                model.set_eplb_state(
                    model._eplb_expert_load_view,
                    model._eplb_logical_to_physical_map,
                    model._eplb_logical_replica_count,
                )

            orig_model_forward = model.forward

            def _forward_with_eplb_step(*f_args, **f_kwargs):
                out = orig_model_forward(*f_args, **f_kwargs)
                st = getattr(model, "_eplb_state", None)
                if st is not None:
                    old_eplb_enabled = bool(getattr(model, "_eplb_enabled", False))
                    old_expert_load_view = getattr(model, "_eplb_expert_load_view", None)
                    old_l2p = getattr(model, "_eplb_logical_to_physical_map", None)
                    old_lrc = getattr(model, "_eplb_logical_replica_count", None)
                    try:
                        st.step(model, is_dummy=False, is_profile=False, log_stats=False)
                        eplb_step_hook_diag["calls"] += 1
                    except Exception as e:  # noqa: BLE001
                        eplb_step_hook_diag["errors"] += 1
                        eplb_step_hook_diag["last_error"] = repr(e)
                        eplb_step_hook_diag["last_traceback"] = traceback.format_exc()
                    finally:
                        # Keep runtime map views pointing to current
                        # model._eplb_* tensors after step(), because step may
                        # internally refresh views by calling set_eplb_state.
                        try:
                            model.set_eplb_runtime_state(
                                enable_eplb=old_eplb_enabled,
                                expert_load_view=getattr(model, "_eplb_expert_load_view", old_expert_load_view),
                                logical_to_physical_map=getattr(model, "_eplb_logical_to_physical_map", old_l2p),
                                logical_replica_count=getattr(model, "_eplb_logical_replica_count", old_lrc),
                            )
                        except Exception:  # noqa: BLE001
                            pass
                return out

            model.forward = _forward_with_eplb_step
            eplb_step_hook_diag["attached"] = True

        with torch.inference_mode():
            warmup_tok = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(
                attn_metadata=None,
                vllm_config=vllm_cfg,
                num_tokens=warmup_tok.numel(),
            ):
                _ = model(warmup_tok, use_cache=False)
        apply_all_optimizations(model)
        prepare_communication_buffer_for_model(model)

        local_bs = args.batch_size // dp_size
        all_ids = []
        for i in range(args.batch_size):
            text = PROMPTS[i % len(PROMPTS)]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [
            torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
            if ids.shape[0] < mx else ids
            for ids in all_ids
        ]
        input_ids_full = torch.stack(padded, dim=0)
        my_input = input_ids_full[dp_rank * local_bs:(dp_rank + 1) * local_bs].to(device)

        decoder = ThresholdParallelDecoder(
            temperature=0.0,
            threshold=0.90,
            mask_id=MASK_ID,
            eos_id=EOS_ID,
        )

        def make_dllm():
            return BlockDiffusionLLM(
                model,
                decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True,
                maximum_unroll=4,
                expected_tpf=15,
                backend="vllm",
                lazy_cache_update=True,
                inplace_cache_update=True,
            )

        ctrl = TraceMSkipController(
            num_layers=19,
            K=8,
            M=4,
            K_target=40,
            quality_floor=0.70,
            q_major=1.0,
            per_round_cap=8,
            skip_m=5,
        )
        orig_block_init = decoder.block_init

        def block_init_with_clock(block_x, block_id):
            ctrl.note_block_start(int(block_id))
            return orig_block_init(block_x, block_id)

        decoder.block_init = block_init_with_clock

        if args.trace_rank_mode == "all":
            trace_full = True
        elif args.trace_rank_mode == "rank0":
            trace_full = rank == 0
        else:
            trace_full = tp_rank == 0

        collector = EBTraceCollector(
            rank=rank,
            dp_rank=dp_rank,
            tp_rank=tp_rank,
            ep_size=world_size,
            trace_full=trace_full,
            group_layers=_parse_layers(args.group_layers),
            group_size=args.group_size,
            block_length=BLOCK_LENGTH,
        )
        rearrange_driver = ColdGatedRearrangeDriver(
            model=model,
            enabled=bool(args.eplb_cold_rearrange),
            min_block_gap=int(args.eplb_cold_rearrange_min_gap),
        )
        native_record_gate = NativeRecordGateController(
            model=model,
            mode=args.native_record_gate_mode,
        )
        route_skip_noop = _route_skip_noop_enabled()
        skip_gate_update = route_skip_noop and (args.native_record_gate_mode == "off")
        skip_rearrange = route_skip_noop and (not bool(args.eplb_cold_rearrange))
        set_eplb_runtime_route_path("unknown")

        gate_idx = 0
        for _name, mod in model.named_modules():
            if mod.__class__.__name__ != "LLaDA2MoeGate":
                continue
            bias = mod.expert_bias
            rsf = mod.routed_scaling_factor
            ng = mod.n_group
            tkg = mod.topk_group
            layer_i = gate_idx

            def make_route(bb, rr, nn, gg, layer):
                def route(hidden_states, gating_output, topk, renormalize):
                    s_mask = ctrl.get_s_mask(layer, gating_output, bb)
                    curr_path = ctrl.last_path.get(layer, "unknown")
                    set_eplb_runtime_route_path(curr_path)
                    if not skip_gate_update:
                        native_record_gate.update_from_path(curr_path)
                    if (not skip_rearrange) and ctrl.current_block_id >= 0:
                        rearrange_driver.maybe_rearrange(
                            path=curr_path,
                            layer_idx=layer,
                            block_id=int(ctrl.current_block_id),
                        )
                    eb_w, eb_idx = fused_routing(
                        gating_output, bb, rr, s_mask=s_mask, K=4, ng=nn, tkg=gg)
                    if trace_full and ctrl.current_block_id >= 0:
                        _noeb_w4, noeb_idx4 = fused_routing(
                            gating_output, bb, rr, s_mask=None, K=4, ng=nn, tkg=gg)
                        _noeb_w8, noeb_idx8 = fused_routing(
                            gating_output, bb, rr, s_mask=None, K=8, ng=nn, tkg=gg)
                        collector.record(
                            layer=layer,
                            block=ctrl.current_block_id,
                            fwd_in_block=ctrl._fwd_in_block.get(layer, 0),
                            path=curr_path,
                            s_mask=s_mask,
                            eb_ids=eb_idx,
                            noeb4_ids=noeb_idx4,
                            noeb8_ids=noeb_idx8,
                        )
                    return eb_w.to(gating_output.dtype), eb_idx
                return route

            mod.routing = make_route(bias, rsf, ng, tkg, layer_i)
            gate_idx += 1

        dllm = make_dllm()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        with use_eplb_runtime_map_policy(
            record_mode=args.eplb_runtime_record_mode,
            map_impl=args.eplb_runtime_map_impl,
            tensor_cache=args.eplb_runtime_tensor_cache,
        ):
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(
                    my_input.clone(),
                    gen_length=args.gen_length,
                    block_length=BLOCK_LENGTH,
                )

        eplb_force_step_diag = {"enabled": False}
        if args.eplb_force_step:
            eplb_force_step_diag["enabled"] = True
            st = getattr(model, "_eplb_state", None)
            if st is None:
                eplb_force_step_diag["status"] = "no_state"
            else:
                before = {
                    "window_step": int(getattr(st, "expert_load_window_step", -1)),
                    "rearrangement_step": int(
                        getattr(st, "expert_rearrangement_step", -1)
                    ),
                }
                try:
                    st.step(model, is_dummy=False, is_profile=False, log_stats=False)
                    after = {
                        "window_step": int(getattr(st, "expert_load_window_step", -1)),
                        "rearrangement_step": int(
                            getattr(st, "expert_rearrangement_step", -1)
                        ),
                    }
                    eplb_force_step_diag.update(
                        {
                            "status": "ok",
                            "before": before,
                            "after": after,
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    eplb_force_step_diag.update(
                        {
                            "status": "error",
                            "before": before,
                            "error": str(e),
                        }
                    )
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - t0
        num_fwds = int(dllm.diff_iteration.num_forwards)
        if args.eplb_step_per_forward:
            eplb_step_hook_diag["after"] = _eplb_state_step_snapshot(model)

        stats = {
            "rank": rank,
            "dp_rank": dp_rank,
            "tp_rank": tp_rank,
            "elapsed_s": elapsed,
            "num_forwards": num_fwds,
            "controller": ctrl.stats(),
            "eplb_cold_rearrange": rearrange_driver.stats(),
            "native_record_gate": native_record_gate.stats(),
            "eplb_runtime_diag": _collect_eplb_runtime_diag(model),
            "eplb_force_step_diag": eplb_force_step_diag,
            "eplb_step_hook_diag": eplb_step_hook_diag,
            "traced_records": len(collector.records),
            "grouping_records": len(collector.grouping_records),
        }

        out_dir = REPO_ROOT / "codex_coding" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload_path = out_dir / f"{args.output_prefix}_rank{rank}.json.gz"
        if trace_full:
            payload = {
                "rank": rank,
                "dp_rank": dp_rank,
                "tp_rank": tp_rank,
                "stats": stats,
                "records": collector.records,
                "grouping_records": collector.grouping_records,
            }
            with gzip.open(payload_path, "wt", encoding="utf-8") as f:
                json.dump(payload, f)
        else:
            payload_path = None

        gathered_stats = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(stats, gathered_stats, dst=0)
        dist.barrier()

        if rank == 0:
            payloads = []
            for rr in range(world_size):
                pp = out_dir / f"{args.output_prefix}_rank{rr}.json.gz"
                if not pp.exists():
                    continue
                with gzip.open(pp, "rt", encoding="utf-8") as f:
                    payloads.append(json.load(f))
            summary = _summarize_payloads(payloads, args, gathered_stats)
            summary_path = out_dir / f"{args.output_prefix}_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"  elapsed={elapsed:.3f}s, forwards={num_fwds}, ms/fwd={elapsed * 1000 / max(num_fwds, 1):.3f}")
            print(f"  path_counts={ctrl.stats()['path_counts']}")
            print(f"  trace_payloads={len(payloads)}, valid_records={summary['num_valid_records']}")
            oa = summary["overall_active_expert"]
            print(
                "  active experts mean: "
                f"EB={oa['eb_unique4']['mean']:.2f}, "
                f"noEB4={oa['noeb_unique4']['mean']:.2f}, "
                f"reduction={oa['mean_reduction_vs_noeb4_pct']:.2f}%"
            )
            cov = summary["coverage_control"]
            print(
                "  coverage mean: "
                f"EB_S={cov['weighted_coverage_eb_smask']['mean']:.4f}, "
                f"global_same_size={cov['weighted_coverage_global_pop_same_size']['mean']:.4f}"
            )
            print(f"  saved summary={summary_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
