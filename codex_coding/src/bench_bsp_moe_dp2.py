#!/usr/bin/env python3
"""
BSP-MoE validation for LLaDA2.0-mini on C12-style dp=2, tp=4, ep=8.

BSP-MoE = Block Sequence Parallel MoE:
  - flatten MoE input [bsz, seq_len, hidden] -> [N, hidden]
  - shard N tokens across the TP group before shared/gate/MoE
  - run native vLLM sequence-parallel FusedMoE
  - gather TP token shards back to the original [N, hidden]

The first BSP experiment intentionally does not include Scheme3, compact topk
payloads, active expert pruning, or scheduler changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import triton
import triton.language as tl

# Global kernel phase event timing
_kp_events = []  # list of (start_event, end_event, config_label)
_kp_enabled = bool(os.environ.get("DINF_KERNEL_PHASE_TIMING"))
import types
from collections import Counter
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

if os.environ.get("BSP_DISABLE_DEEP_EP", "1") != "0":
    sys.modules["deep_ep"] = None

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

from test_m_skip_sweep import MSkipEBController
from test_fused_eb_triton import _kernel_A, _kernel_B_v3

MASK_ID = 156895
EOS_ID = 156892
BLOCK_LENGTH = 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"

_TEAM_SKIP_CTRL = None  # set by setup_bsp_team, read by decoder forward


class DecodedSkipController:
    def __init__(self, num_moe_layers: int):
        self.num_moe_layers = num_moe_layers
        self.moe_cache = {}
        self.step_in_block = 0
        self._current_input_ids = None
        self._mask_sp_cache = None
        self.decoded_gathered = None
        self.NULL_EXPERT_ID = None
        self.total_tokens = 0
        self.skipped_tokens = 0

    def on_block_start(self):
        self.moe_cache.clear()
        self.step_in_block = 0
        self._mask_sp_cache = None
        self.decoded_gathered = None

    def set_input_ids(self, input_ids):
        self._current_input_ids = input_ids
        self._mask_sp_cache = None
        self.decoded_gathered = None

    def get_mask_sp(self, tp_rank, tp_size):
        if self._mask_sp_cache is not None:
            return self._mask_sp_cache
        is_mask = (self._current_input_ids == MASK_ID).view(-1)
        N_dp = is_mask.shape[0]
        chunk = N_dp // tp_size
        sp_start = tp_rank * chunk
        self._mask_sp_cache = is_mask[sp_start:sp_start + chunk]
        return self._mask_sp_cache

    def should_skip(self):
        return self.step_in_block > 0 and len(self.moe_cache) > 0

    def after_forward(self):
        self.step_in_block += 1
        self._mask_sp_cache = None
        self.decoded_gathered = None

    def stats(self):
        ratio = self.skipped_tokens / max(self.total_tokens, 1)
        return {"total_tokens": self.total_tokens,
                "skipped_tokens": self.skipped_tokens,
                "skip_ratio": round(ratio, 4)}


def _gather_decoded_mask(skip_ctrl):
    from vllm.distributed import (
        get_ep_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    ep = get_ep_group()
    decoded_sp = ~skip_ctrl.get_mask_sp(tp_rank, tp_size)
    decoded_byte = decoded_sp.to(torch.uint8)
    gathered = [torch.empty_like(decoded_byte) for _ in range(ep.world_size)]
    dist.all_gather(gathered, decoded_byte, group=ep.device_group)
    skip_ctrl.decoded_gathered = torch.cat(gathered, dim=0).bool()


def _wrap_routing_for_team(moe_blocks, skip_ctrl):
    pass  # Disabled: test cache-only without null expert


class ComponentTimer:
    """CUDA event collector for rank-max component timing."""

    def __init__(self):
        self.enabled = False
        self.records = {}
        self.counts = {}
        self.bytes = {}

    def reset(self):
        self.records.clear()
        self.counts.clear()
        self.bytes.clear()

    def time(self, name: str):
        if not self.enabled:
            return PROFILE_RANGES.range(name)
        return _ComponentTimerContext(self, name)

    def add_bytes(self, name: str, nbytes: int):
        if self.enabled:
            self.bytes[name] = self.bytes.get(name, 0) + int(nbytes)

    def _add_record(self, name: str, start: torch.cuda.Event, end: torch.cuda.Event):
        self.records.setdefault(name, []).append((start, end))
        self.counts[name] = self.counts.get(name, 0) + 1

    def summarize_across_ranks(self, fwds: int):
        torch.cuda.synchronize()
        local_ms = {}
        for name, pairs in self.records.items():
            local_ms[name] = sum(start.elapsed_time(end) for start, end in pairs)
        local = {
            "ms": local_ms,
            "counts": dict(self.counts),
            "bytes": dict(self.bytes),
        }
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local)
        if dist.get_rank() != 0:
            return None

        names = sorted({k for item in gathered for k in item["ms"].keys()})
        byte_names = sorted({k for item in gathered for k in item["bytes"].keys()})
        components = {}
        for name in names:
            all_ms = [item["ms"].get(name, 0.0) for item in gathered]
            count = max(item["counts"].get(name, 0) for item in gathered)
            components[name] = {
                "total_ms_rankmax": max(all_ms),
                "total_ms_rankmin": min(all_ms),
                "total_ms_rankmean": sum(all_ms) / len(all_ms),
                "ms_per_fwd_rankmax": max(all_ms) / max(fwds, 1),
                "ms_per_fwd_rankmin": min(all_ms) / max(fwds, 1),
                "ms_per_fwd_rankmean": sum(all_ms) / len(all_ms) / max(fwds, 1),
                "count_rankmax": count,
            }
        byte_components = {}
        for name in byte_names:
            total_bytes = max(item["bytes"].get(name, 0) for item in gathered)
            byte_components[name] = {
                "total_bytes_rankmax": total_bytes,
                "mb_per_fwd_rankmax": total_bytes / max(fwds, 1) / 1e6,
            }
        return {
            "fwds": int(fwds),
            "components": components,
            "byte_components": byte_components,
        }


class _ComponentTimerContext:
    def __init__(self, timer: ComponentTimer, name: str):
        self.timer = timer
        self.name = name
        self.start = None
        self.end = None

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end.record()
        self.timer._add_record(self.name, self.start, self.end)
        return False


COMPONENT_TIMER = ComponentTimer()


class _NvtxRange:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.cuda.nvtx.range_pop()
        return False


class ProfileRanges:
    def __init__(self):
        self.enabled = False

    def range(self, name: str):
        if not self.enabled:
            return nullcontext()
        return _NvtxRange(name)


PROFILE_RANGES = ProfileRanges()


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


class ShapeProbe:
    """Collect a bounded number of shape records per rank."""

    def __init__(self, enabled: bool, limit: int):
        self.enabled = enabled
        self.limit = limit
        self.records = []
        self.summary = Counter()

    def reset(self):
        self.records.clear()
        self.summary.clear()

    def record(self, **kwargs):
        if not self.enabled:
            return
        key = (
            kwargs.get("config"),
            kwargs.get("path_pred"),
            kwargs.get("bsz"),
            kwargs.get("seq_len"),
            kwargs.get("N_dp"),
            kwargs.get("tp_size"),
            kwargs.get("N_sp_expected"),
            kwargs.get("pad_tokens"),
        )
        self.summary[key] += 1
        if len(self.records) < self.limit:
            self.records.append(dict(kwargs))

    def gather(self):
        local = {
            "rank": dist.get_rank(),
            "records": list(self.records),
            "summary": [
                {
                    "config": k[0],
                    "path_pred": k[1],
                    "bsz": k[2],
                    "seq_len": k[3],
                    "N_dp": k[4],
                    "tp_size": k[5],
                    "N_sp_expected": k[6],
                    "pad_tokens": k[7],
                    "count": v,
                }
                for k, v in self.summary.items()
            ],
        }
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local)
        return gathered if dist.get_rank() == 0 else None


SHAPE_PROBE = ShapeProbe(enabled=False, limit=64)


class LayoutDiagnostics:
    """Collect passive SP/TP layout records around attention boundaries."""

    def __init__(self, enabled: bool, limit: int):
        self.enabled = enabled
        self.limit = limit
        self.records = []
        self.summary = Counter()

    def reset(self):
        self.records.clear()
        self.summary.clear()

    def record(self, **kwargs):
        if not self.enabled:
            return
        key = (
            kwargs.get("config"),
            kwargs.get("layer_id"),
            kwargs.get("event"),
            kwargs.get("input_kind"),
            kwargs.get("qkv_tokens"),
            kwargs.get("expected_tokens"),
            kwargs.get("full_qkv_ok"),
            kwargs.get("tp_size"),
        )
        self.summary[key] += 1
        if len(self.records) < self.limit:
            self.records.append(dict(kwargs))

    def gather(self):
        local = {
            "rank": dist.get_rank(),
            "records": list(self.records),
            "summary": [
                {
                    "config": k[0],
                    "layer_id": k[1],
                    "event": k[2],
                    "input_kind": k[3],
                    "qkv_tokens": k[4],
                    "expected_tokens": k[5],
                    "full_qkv_ok": k[6],
                    "tp_size": k[7],
                    "count": v,
                }
                for k, v in self.summary.items()
            ],
        }
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local)
        return gathered if dist.get_rank() == 0 else None


LAYOUT_DIAG = LayoutDiagnostics(enabled=False, limit=128)


def _record_layout(**kwargs):
    LAYOUT_DIAG.record(**kwargs)


def _diff_metrics(local_a: torch.Tensor, local_b: torch.Tensor, device: torch.device):
    diff = (local_a.float() - local_b.float()).abs()
    denom = local_a.float().abs().max().clamp_min(1e-6)
    local = torch.tensor(
        [
            diff.max().item(),
            diff.mean().item(),
            (diff.max() / denom).item(),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return {
        "rankmax_abs_max": float(local[0].item()),
        "rankmax_abs_mean": float(local[1].item()),
        "rankmax_rel_max": float(local[2].item()),
    }


@dataclass
class SPHiddenState:
    """Carry sequence-parallel hidden state across patched decoder layers."""

    hidden_sp: torch.Tensor
    bsz: int
    seq_len: int
    n_tokens: int


@dataclass
class AttnSPResult:
    """Attention output already reduced-scattered across the TP token axis."""

    attn_sp: torch.Tensor
    bsz: int
    seq_len: int
    n_tokens: int


@dataclass
class SPAttentionInput:
    """Normalized attention input still sharded across TP token axis."""

    hidden_norm_sp: torch.Tensor
    bsz: int
    seq_len: int
    n_tokens: int


class BSPMSkipController(MSkipEBController):
    """MSkip controller with explicit block-id clock for robust path counts."""

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

    def _is_new_block_explicit(self, layer_idx: int):
        if self.current_block_id < 0:
            return None
        prev = self._last_block_id.get(layer_idx, -1)
        if prev != self.current_block_id:
            self._last_block_id[layer_idx] = self.current_block_id
            return True
        return False

    def predict_path(self, layer_idx: int) -> str:
        if self.current_block_id < 0:
            return "prefill_fallback"
        if self._last_block_id.get(layer_idx, -1) != self.current_block_id:
            return "cold"
        nxt = self._fwd_in_block.get(layer_idx, 0) + 1
        if layer_idx not in self.s_mask_cache:
            return "hot_update"
        if self.skip_m == float("inf") or nxt % self.skip_m != 0:
            return "hot_skip"
        return "hot_update"

    def get_s_mask(self, layer_idx, logits, bias):
        explicit_new = self._is_new_block_explicit(layer_idx)
        if explicit_new is None:
            self.path_counts["prefill_fallback"] += 1
            return super().get_s_mask(layer_idx, logits, bias)

        if explicit_new:
            self.path_counts["cold"] += 1
            self.per_layer_cold[layer_idx] = self.per_layer_cold.get(layer_idx, 0) + 1
            return self.cold_path(layer_idx, logits, bias)

        prev_calls = self.eb_calls
        prev_skips = self.eb_skips
        out = self.hot_path(layer_idx, logits, bias)
        if self.eb_calls > prev_calls:
            self.path_counts["hot_update"] += 1
        elif self.eb_skips > prev_skips:
            self.path_counts["hot_skip"] += 1
        return out

    def stats(self):
        return {
            "path_counts": dict(self.path_counts),
            "per_layer_cold": {
                str(k): int(v) for k, v in sorted(self.per_layer_cold.items())
            },
        }


class BSPMSkipControllerEPReduce(BSPMSkipController):
    """M3 minimal controller:
    keep cold/hot_skip semantics, replace hot-update pop combine with EP all-reduce."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ep_reduce_calls = 0
        self.ep_reduce_bytes = 0

    def hot_path(self, layer_idx, logits, bias):
        from vllm.distributed import get_ep_group

        N, E = logits.shape
        K_init = self.K_init.get(layer_idx, 103)

        fi = self._fwd_in_block.get(layer_idx, 0) + 1
        self._fwd_in_block[layer_idx] = fi

        if self.skip_m == float("inf") or fi % self.skip_m != 0:
            self.eb_skips += 1
            self.hot_count += 1
            return self.s_mask_cache[layer_idx]

        pop = self.pop_cache[layer_idx]
        lf = logits.float()
        _kernel_A[(N,)](
            lf,
            bias.float(),
            pop,
            N,
            self.rsf,
            lf.stride(0),
            lf.stride(1),
            E=E,
            KEXT=self.K_ext,
            KEXT_PAD=16,
        )
        ep_group = get_ep_group()
        dist.all_reduce(pop, op=dist.ReduceOp.SUM, group=ep_group.device_group)
        self.ep_reduce_calls += 1
        self.ep_reduce_bytes += int(pop.numel() * pop.element_size())

        _kernel_B_v3[(1,)](pop, self.s_mask_cache[layer_idx], K_init, E=E)

        self.eb_calls += 1
        self.hot_count += 1
        return self.s_mask_cache[layer_idx]

    def stats(self):
        out = super().stats()
        out["ep_reduce_calls"] = int(self.ep_reduce_calls)
        out["ep_reduce_mb"] = float(self.ep_reduce_bytes / 1e6)
        return out


class JointStatsCollector:
    """Collect per-layer per-forward (pairs, active_experts, histogram) stats."""

    def __init__(self, num_layers: int, num_local_experts: int, skip_first: int = 20):
        self.num_layers = num_layers
        self.E_local = num_local_experts
        self.skip_first = skip_first
        self.forward_count = 0
        self.records = []
        self._cur = {}

    def record(self, layer_idx: int, topk_ids: torch.Tensor,
               expert_map: torch.Tensor):
        local_ids = expert_map[topk_ids.long()]
        local_mask = (local_ids >= 0)
        pairs = int(local_mask.sum().item())

        valid = local_ids[local_mask]
        hist = torch.zeros(self.E_local, device=topk_ids.device, dtype=torch.int32)
        if valid.numel() > 0:
            hist.scatter_add_(0, valid.long(),
                              torch.ones_like(valid, dtype=torch.int32))
        active = int((hist > 0).sum().item())
        max_tok = int(hist.max().item())
        active_hist = hist[hist > 0]
        min_tok = int(active_hist.min().item()) if active > 0 else 0

        self._cur[layer_idx] = {
            "pairs": pairs,
            "active": active,
            "max_tok": max_tok,
            "min_tok": min_tok,
            "hist": hist.cpu().tolist(),
        }

    def on_forward_end(self):
        if self.forward_count >= self.skip_first and self._cur:
            if COMPONENT_TIMER.enabled:
                torch.cuda.synchronize()
                for l in range(self.num_layers):
                    key = f"moe.quant_apply.L{l}"
                    pairs_list = COMPONENT_TIMER.records.get(key, [])
                    if pairs_list:
                        s, e = pairs_list[-1]
                        self._cur.setdefault(l, {})["kernel_ms"] = s.elapsed_time(e)
            self.records.append(self._cur.copy())
        self._cur = {}
        self.forward_count += 1

    def to_dict(self):
        return {
            "num_layers": self.num_layers,
            "E_local": self.E_local,
            "skip_first": self.skip_first,
            "num_records": len(self.records),
            "records": self.records,
        }


# Global instance, set by main when --collect-joint-stats is used
JOINT_STATS: JointStatsCollector = None
ISOLATE_COMBINE_WAIT: bool = False


def make_timed_forward_impl(experts, layer_id=None):
    """Patch one FusedMoE instance for timing only; behavior stays native.
    When layer_id is provided, uses per-layer timer keys (moe.quant_apply.L0 etc.)."""

    from vllm.distributed import get_ep_group
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEModularKernel
    from vllm.forward_context import get_forward_context

    orig_forward_impl = experts.forward_impl
    _sfx = f".L{layer_id}" if layer_id is not None else ""
    _isolate = ISOLATE_COMBINE_WAIT

    def forward_impl_timed(self, hidden_states, router_logits):
        if not COMPONENT_TIMER.enabled and not _kp_enabled:
            return orig_forward_impl(hidden_states, router_logits)

        if not COMPONENT_TIMER.enabled and _kp_enabled:
            # Kernel phase timing only (no component timing)
            # orig_forward_impl does dispatch+kernel+combine all together
            # We need to intercept at a level where dispatch/combine are separate
            # Fall through to the full timed path below to get event recording
            pass
        else:
            pass

        assert self.quant_method is not None
        self.ensure_moe_quant_config()

        unsupported = (
            self.moe_parallel_config.use_pplx_kernels
            or self.moe_parallel_config.use_deepep_ll_kernels
            or (self.dp_size > 1 and self.use_flashinfer_cutlass_kernels)
            or self.moe_parallel_config.use_deepep_ht_kernels
            or self.moe_config.use_flashinfer_cutlass_kernels
        )
        if unsupported:
            with COMPONENT_TIMER.time("moe.native_forward" + _sfx):
                return orig_forward_impl(hidden_states, router_logits)

        do_naive_dispatch_combine = self.dp_size > 1
        if not do_naive_dispatch_combine:
            with COMPONENT_TIMER.time("moe.native_forward" + _sfx):
                return orig_forward_impl(hidden_states, router_logits)

        if (
            not isinstance(self.quant_method.fused_experts, FusedMoEModularKernel)
            and self.shared_experts is not None
        ):
            shared_output = self.shared_experts(hidden_states)
        else:
            shared_output = None

        ctx = get_forward_context()
        sp_ctx = (
            ctx.dp_metadata.sp_local_sizes(self.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

        with sp_ctx:
            COMPONENT_TIMER.add_bytes(
                "dispatch_payload",
                _tensor_nbytes(hidden_states) + _tensor_nbytes(router_logits),
            )
            with COMPONENT_TIMER.time("moe.dispatch" + _sfx):
                hidden_states, router_logits = get_ep_group().dispatch(
                    hidden_states, router_logits, self.is_sequence_parallel
                )

            if _kp_enabled:
                _kp_s = torch.cuda.Event(enable_timing=True)
                _kp_e = torch.cuda.Event(enable_timing=True)
                _kp_s.record()

            with COMPONENT_TIMER.time("moe.quant_apply" + _sfx):
                final_hidden_states = self.quant_method.apply(
                    layer=self,
                    x=hidden_states,
                    router_logits=router_logits,
                    top_k=self.top_k,
                    renormalize=self.renormalize,
                    use_grouped_topk=self.use_grouped_topk,
                    global_num_experts=self.global_num_experts,
                    expert_map=self.expert_map,
                    topk_group=self.topk_group,
                    num_expert_group=self.num_expert_group,
                    custom_routing_function=self.custom_routing_function,
                    scoring_func=self.scoring_func,
                    routed_scaling_factor=self.routed_scaling_factor,
                    e_score_correction_bias=self.e_score_correction_bias,
                    activation=self.activation,
                    apply_router_weight_on_input=self.apply_router_weight_on_input,
                    enable_eplb=self.enable_eplb,
                    expert_load_view=self.expert_load_view,
                    logical_to_physical_map=self.logical_to_physical_map,
                    logical_replica_count=self.logical_replica_count,
                )

            if _kp_enabled:
                _kp_e.record()
                _kp_events.append((_kp_s, _kp_e, "G"))

            if shared_output is not None:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None
                final_hidden_states = (shared_output, final_hidden_states)
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                final_hidden_states, zero_expert_result = final_hidden_states

            def reduce_output(states: torch.Tensor, do_combine: bool = True):
                if do_naive_dispatch_combine and do_combine:
                    if _isolate and COMPONENT_TIMER.enabled:
                        with COMPONENT_TIMER.time("moe.pre_combine_sync" + _sfx):
                            torch.cuda.synchronize()
                        with COMPONENT_TIMER.time("moe.pre_combine_barrier" + _sfx):
                            dist.barrier()
                    with COMPONENT_TIMER.time("moe.combine" + _sfx):
                        states = get_ep_group().combine(
                            states, self.is_sequence_parallel
                        )

                if (
                    not self.is_sequence_parallel
                    and self.reduce_results
                    and (self.tp_size > 1 or self.ep_size > 1)
                ):
                    with COMPONENT_TIMER.time("moe.tp_all_reduce" + _sfx):
                        states = self.maybe_all_reduce_tensor_model_parallel(states)
                return states

            if self.shared_experts is not None:
                return (
                    reduce_output(final_hidden_states[0], do_combine=False),
                    reduce_output(final_hidden_states[1]),
                )
            if self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, torch.Tensor)
                return reduce_output(final_hidden_states) + zero_expert_result
            return reduce_output(final_hidden_states)

    return types.MethodType(forward_impl_timed, experts)


def make_timed_dense_mlp_forward(mlp):
    """Patch standalone dense MLP timing only; shared experts are timed elsewhere."""

    orig_forward = mlp.forward

    def forward_timed(*args, **kwargs):
        with COMPONENT_TIMER.time("dense.mlp"):
            return orig_forward(*args, **kwargs)

    return forward_timed


def _record_shape(config_label: str, layer_id: int, ctrl: BSPMSkipController,
                  bsz: int, seq_len: int, N: int, h: int, tp_size: int,
                  mode: str, n_sp_actual: int | None = None):
    n_sp_expected = (N + tp_size - 1) // tp_size
    SHAPE_PROBE.record(
        config=config_label,
        mode=mode,
        rank=dist.get_rank(),
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        layer_id=layer_id,
        block_id=int(ctrl.current_block_id),
        fwd_in_block=int(ctrl._fwd_in_block.get(layer_id, 0)),
        path_pred=ctrl.predict_path(layer_id),
        bsz=bsz,
        seq_len=seq_len,
        hidden=h,
        N_dp=N,
        tp_size=tp_size,
        N_sp_expected=n_sp_expected,
        N_sp_actual=n_sp_actual,
        pad_tokens=n_sp_expected * tp_size - N,
    )


def make_baseline_forward(block, layer_id, ctrl, config_label: str):
    """Equivalent to native LLaDA2 MoE forward, with optional timing/probe."""

    from dinfer.model.modeling_llada2_moe import _moe_forward_with_context

    shared_mod = block.shared_experts if block.config.num_shared_experts else None
    experts = block.experts

    def forward(hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        N = hs_flat.shape[0]
        _record_shape(config_label, layer_id, ctrl, bsz, seq_len, N, h,
                      tp_size=1, mode="baseline")

        with COMPONENT_TIMER.time("moe.shared"):
            shared_res = shared_mod(hs_flat) if shared_mod is not None else None
        with COMPONENT_TIMER.time("moe.gate_logits"):
            router_logits = block.gate.get_logits(hs_flat)
        with COMPONENT_TIMER.time("moe.native_forward"):
            y = _moe_forward_with_context(experts, hs_flat, router_logits)
        if shared_res is not None:
            y = y + shared_res
        return y.view(bsz, seq_len, h)

    return forward


def make_bsp_forward(block, layer_id, ctrl, vllm_cfg, config_label: str):
    """BSP-MoE forward using vLLM native sequence-parallel FusedMoE."""

    from vllm.distributed import (
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.utils import sequence_parallel_chunk

    shared_mod = block.shared_experts if block.config.num_shared_experts else None
    experts = block.experts

    def forward(hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        N = hs_flat.shape[0]
        tp_size = get_tensor_model_parallel_world_size()

        with set_forward_context(
            attn_metadata=None,
            vllm_config=vllm_cfg,
            num_tokens=N,
        ):
            with COMPONENT_TIMER.time("moe.bsp_chunk"):
                hs_sp = sequence_parallel_chunk(hs_flat)

            _record_shape(
                config_label,
                layer_id,
                ctrl,
                bsz,
                seq_len,
                N,
                h,
                tp_size=tp_size,
                mode="bsp",
                n_sp_actual=hs_sp.shape[0],
            )

            with COMPONENT_TIMER.time("moe.shared"):
                shared_res = shared_mod(hs_sp) if shared_mod is not None else None
            with COMPONENT_TIMER.time("moe.gate_logits"):
                router_logits = block.gate.get_logits(hs_sp)
            with COMPONENT_TIMER.time("moe.native_forward"):
                y_sp = experts.forward_impl(hs_sp, router_logits)
            if shared_res is not None:
                y_sp = y_sp + shared_res

            COMPONENT_TIMER.add_bytes("tp_gather_payload", _tensor_nbytes(y_sp))
            with COMPONENT_TIMER.time("moe.tp_all_gather"):
                y = tensor_model_parallel_all_gather(y_sp, dim=0)
            y = y[:N]

        return y.view(bsz, seq_len, h)

    return forward


def make_bsp_delay_decoder_forward(decoder_layer, layer_id, ctrl, vllm_cfg,
                                   config_label: str):
    """Conservative delayed-gather BSP:
    keep SP layout through MoE + residual add, gather before layer output."""

    from vllm.distributed import (
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.utils import sequence_parallel_chunk

    mlp = decoder_layer.mlp
    attention = decoder_layer.attention
    input_layernorm = decoder_layer.input_layernorm
    post_attention_layernorm = decoder_layer.post_attention_layernorm

    shared_mod = mlp.shared_experts if mlp.config.num_shared_experts else None
    gate = mlp.gate
    experts = mlp.experts

    def forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        cache_position=None,
        replace_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            replace_position=replace_position,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = post_attention_layernorm(hidden_states)

        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        residual_flat = residual.view(-1, h)
        N = hs_flat.shape[0]
        tp_size = get_tensor_model_parallel_world_size()

        with set_forward_context(
            attn_metadata=None,
            vllm_config=vllm_cfg,
            num_tokens=N,
        ):
            with COMPONENT_TIMER.time("moe.bsp_chunk"):
                hs_sp = sequence_parallel_chunk(hs_flat)
                residual_sp = sequence_parallel_chunk(residual_flat)

            _record_shape(
                config_label,
                layer_id,
                ctrl,
                bsz,
                seq_len,
                N,
                h,
                tp_size=tp_size,
                mode="bsp_delay",
                n_sp_actual=hs_sp.shape[0],
            )

            with COMPONENT_TIMER.time("moe.shared"):
                shared_res = shared_mod(hs_sp) if shared_mod is not None else None
            with COMPONENT_TIMER.time("moe.gate_logits"):
                router_logits = gate.get_logits(hs_sp)
            with COMPONENT_TIMER.time("moe.native_forward"):
                y_sp = experts.forward_impl(hs_sp, router_logits)
            if shared_res is not None:
                y_sp = y_sp + shared_res

            hidden_sp = residual_sp + y_sp.to(residual_sp.device)
            COMPONENT_TIMER.add_bytes("tp_gather_payload", _tensor_nbytes(hidden_sp))
            with COMPONENT_TIMER.time("moe.tp_all_gather"):
                hidden_flat = tensor_model_parallel_all_gather(hidden_sp, dim=0)
            hidden_flat = hidden_flat[:N]

        hidden_states = hidden_flat.view(bsz, seq_len, h)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (None,)
        return outputs

    return forward


def make_bsp_cross_layer_sp_decoder_forward(decoder_layer, layer_id, ctrl, vllm_cfg,
                                            config_label: str, is_last_sparse: bool):
    """C+ path: keep hidden state SP across decoder layer boundaries."""

    from vllm.distributed import (
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.utils import sequence_parallel_chunk

    mlp = decoder_layer.mlp
    attention = decoder_layer.attention
    input_layernorm = decoder_layer.input_layernorm
    post_attention_layernorm = decoder_layer.post_attention_layernorm

    shared_mod = mlp.shared_experts if mlp.config.num_shared_experts else None
    gate = mlp.gate
    experts = mlp.experts

    def forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        cache_position=None,
        replace_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if isinstance(hidden_states, SPHiddenState):
            sp_state = hidden_states
            bsz = sp_state.bsz
            seq_len = sp_state.seq_len
            N = sp_state.n_tokens
            tp_size = get_tensor_model_parallel_world_size()
            pre_gather_tokens = sp_state.hidden_sp.view(-1, sp_state.hidden_sp.shape[-1]).shape[0]
            with COMPONENT_TIMER.time("moe.input_norm_sp"):
                hidden_norm_sp = input_layernorm(
                    sp_state.hidden_sp.view(-1, sp_state.hidden_sp.shape[-1])
                )
            COMPONENT_TIMER.add_bytes("tp_gather_payload", _tensor_nbytes(hidden_norm_sp))
            with COMPONENT_TIMER.time("moe.tp_all_gather"):
                hidden_flat = tensor_model_parallel_all_gather(hidden_norm_sp, dim=0)
            hidden_states = hidden_flat[:N].view(bsz, seq_len, -1)
            residual_sp = sp_state.hidden_sp.view(-1, sp_state.hidden_sp.shape[-1])
            input_kind = "sp_decoder_input_gathered"
        else:
            bsz, seq_len, h = hidden_states.shape
            N = bsz * seq_len
            tp_size = get_tensor_model_parallel_world_size()
            pre_gather_tokens = N
            residual_flat = hidden_states.view(-1, h)
            with COMPONENT_TIMER.time("moe.bsp_chunk"):
                residual_sp = sequence_parallel_chunk(residual_flat)
            hidden_states = input_layernorm(hidden_states)
            input_kind = "full_tensor"

        _record_layout(
            config=config_label,
            layer_id=layer_id,
            event="attention_qkv_input",
            input_kind=input_kind,
            pre_gather_tokens=pre_gather_tokens,
            qkv_tokens=bsz * seq_len,
            expected_tokens=N,
            full_qkv_ok=(bsz * seq_len == N),
            tp_size=tp_size,
            bsz=bsz,
            seq_len=seq_len,
            hidden_size=hidden_states.shape[-1],
        )

        hidden_states, self_attn_weights, present_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            replace_position=replace_position,
        )

        h = hidden_states.shape[-1]
        attn_flat = hidden_states.view(-1, h)
        with set_forward_context(
            attn_metadata=None,
            vllm_config=vllm_cfg,
            num_tokens=N,
        ):
            with COMPONENT_TIMER.time("moe.bsp_chunk"):
                attn_sp = sequence_parallel_chunk(attn_flat)
            hidden_after_attn_sp = residual_sp + attn_sp.to(residual_sp.device)

            with COMPONENT_TIMER.time("moe.post_attn_norm_sp"):
                hs_sp = post_attention_layernorm(hidden_after_attn_sp)

            _record_shape(
                config_label,
                layer_id,
                ctrl,
                bsz,
                seq_len,
                N,
                h,
                tp_size=tp_size,
                mode="bsp_cross_layer_sp",
                n_sp_actual=hs_sp.shape[0],
            )

            with COMPONENT_TIMER.time("moe.shared"):
                shared_res = shared_mod(hs_sp) if shared_mod is not None else None
            with COMPONENT_TIMER.time("moe.gate_logits"):
                router_logits = gate.get_logits(hs_sp)
            with COMPONENT_TIMER.time("moe.native_forward"):
                y_sp = experts.forward_impl(hs_sp, router_logits)
            if shared_res is not None:
                y_sp = y_sp + shared_res

            hidden_sp = hidden_after_attn_sp + y_sp.to(hidden_after_attn_sp.device)

        if is_last_sparse:
            COMPONENT_TIMER.add_bytes("tp_gather_payload", _tensor_nbytes(hidden_sp))
            with COMPONENT_TIMER.time("moe.tp_all_gather"):
                hidden_flat = tensor_model_parallel_all_gather(hidden_sp, dim=0)
            hidden_states_out = hidden_flat[:N].view(bsz, seq_len, h)
        else:
            hidden_states_out = SPHiddenState(hidden_sp, bsz, seq_len, N)

        outputs = (hidden_states_out,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (None,)
        return outputs

    return forward


def make_attention_reduce_scatter_forward(attention, config_label: str = "", layer_id: int = -1):
    """Patch attention o_proj to return TP reduce-scattered token layout.

    This is a BSP-G probe for the vLLM SP compiler-pass idea:
    replace the row-parallel attention output all-reduce with a
    reduce-scatter over the flattened token dimension, so the following
    residual/norm/MoE region can consume SP layout directly.
    """

    import torch.nn.functional as F
    from dinfer.model.modeling_llada2_moe import apply_rotary_pos_emb, repeat_kv
    from vllm.distributed import tensor_model_parallel_reduce_scatter
    from vllm.model_executor.layers.layernorm import rms_norm as vllm_rms_norm

    try:
        from flash_attn import flash_attn_func
    except Exception:
        flash_attn_func = None

    q_norm_w = attention.query_layernorm.weight
    q_norm_eps = attention.query_layernorm.variance_epsilon
    k_norm_w = attention.key_layernorm.weight
    k_norm_eps = attention.key_layernorm.variance_epsilon
    dense = attention.dense

    def forward(
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings=None,
        cache_position=None,
        replace_position=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        n_tokens = bsz * q_len
        num_heads = attention.num_heads
        num_kv_heads = attention.num_key_value_heads
        head_dim = attention.head_dim
        try:
            from vllm.distributed import get_tensor_model_parallel_world_size
            tp_size = get_tensor_model_parallel_world_size()
        except Exception:
            tp_size = 1
        _record_layout(
            config=config_label,
            layer_id=layer_id,
            event="attention_qkv_input",
            input_kind="full_tensor",
            qkv_tokens=n_tokens,
            expected_tokens=n_tokens,
            full_qkv_ok=True,
            tp_size=tp_size,
            bsz=bsz,
            seq_len=q_len,
            hidden_size=hidden_states.shape[-1],
        )

        qkv, _ = attention.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, num_heads + 2 * num_kv_heads, head_dim)
        query_states, key_states, value_states = qkv.split(
            [num_heads, num_kv_heads, num_kv_heads], dim=-2
        )

        query_states = vllm_rms_norm(query_states, q_norm_w, q_norm_eps)
        key_states = vllm_rms_norm(key_states, k_norm_w, k_norm_eps)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, attention.layer_idx, replace_position
            )
        if use_cache:
            past_key_value = (key_states, value_states)

        q_fa = query_states.transpose(1, 2)
        k_fa = key_states.transpose(1, 2)
        v_fa = value_states.transpose(1, 2)

        if attention_mask is not None or flash_attn_func is None:
            num_kv_groups = num_heads // num_kv_heads
            k_exp = repeat_kv(key_states, num_kv_groups).contiguous()
            v_exp = repeat_kv(value_states, num_kv_groups).contiguous()
            q_c = query_states.contiguous()
            am = attention_mask.bool() if attention_mask is not None else None
            if am is not None and am.dim() == 3:
                am = am.unsqueeze(1)
            attn_output = F.scaled_dot_product_attention(
                q_c, k_exp, v_exp, attn_mask=am, dropout_p=0.0, is_causal=False
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attn_output = flash_attn_func(
                q_fa.contiguous(), k_fa.contiguous(), v_fa.contiguous(), causal=False
            )

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if dense.input_is_parallel:
            input_parallel = attn_output
        else:
            chunks = torch.split(
                attn_output,
                attn_output.shape[-1] // dense.tp_size,
                dim=-1,
            )
            input_parallel = chunks[dense.tp_rank].contiguous()

        bias = None if (dense.tp_rank > 0 or dense.skip_bias_add) else dense.bias
        output_parallel = dense.quant_method.apply(dense, input_parallel, bias)
        output_flat = output_parallel.view(-1, output_parallel.shape[-1])
        COMPONENT_TIMER.add_bytes("attn_rs_payload", _tensor_nbytes(output_flat))
        with COMPONENT_TIMER.time("attn.tp_reduce_scatter"):
            output_sp = tensor_model_parallel_reduce_scatter(output_flat, dim=0)

        _record_layout(
            config=config_label,
            layer_id=layer_id,
            event="attention_output",
            output_kind="sp_reduce_scatter",
            output_tokens=output_sp.shape[0],
            expected_sp_tokens=(n_tokens + tp_size - 1) // tp_size,
            expected_tokens=n_tokens,
            tp_size=tp_size,
            bsz=bsz,
            seq_len=q_len,
        )

        return AttnSPResult(output_sp, bsz, q_len, n_tokens), None, past_key_value

    return forward


def make_attention_g2_sp_parity_forward(attention, config_label: str = "", layer_id: int = -1):
    """G2 attention: accept SP-normalized input and return SP output.

    This keeps the SP-to-full gather local to the attention implementation,
    matching the vLLM-style layout-boundary ownership more closely than the
    decoder wrapper doing a materialized gather before calling attention.
    """

    import torch.nn.functional as F
    from dinfer.model.modeling_llada2_moe import apply_rotary_pos_emb, repeat_kv
    from vllm.distributed import (
        tensor_model_parallel_all_gather,
        tensor_model_parallel_reduce_scatter,
    )
    from vllm.model_executor.layers.layernorm import rms_norm as vllm_rms_norm

    try:
        from flash_attn import flash_attn_func
    except Exception:
        flash_attn_func = None

    q_norm_w = attention.query_layernorm.weight
    q_norm_eps = attention.query_layernorm.variance_epsilon
    k_norm_w = attention.key_layernorm.weight
    k_norm_eps = attention.key_layernorm.variance_epsilon
    dense = attention.dense

    def forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings=None,
        cache_position=None,
        replace_position=None,
        **kwargs,
    ):
        if isinstance(hidden_states, SPAttentionInput):
            sp_input = hidden_states
            bsz = sp_input.bsz
            q_len = sp_input.seq_len
            n_tokens = sp_input.n_tokens
            pre_gather_tokens = sp_input.hidden_norm_sp.shape[0]
            COMPONENT_TIMER.add_bytes(
                "attn_input_gather_payload",
                _tensor_nbytes(sp_input.hidden_norm_sp),
            )
            with COMPONENT_TIMER.time("attn.input_all_gather"):
                hidden_flat = tensor_model_parallel_all_gather(
                    sp_input.hidden_norm_sp, dim=0
                )
            hidden_states = hidden_flat[:n_tokens].view(bsz, q_len, -1)
            input_kind = "sp_attention_input_gathered"
        else:
            bsz, q_len, _ = hidden_states.size()
            n_tokens = bsz * q_len
            pre_gather_tokens = n_tokens
            input_kind = "full_tensor"

        num_heads = attention.num_heads
        num_kv_heads = attention.num_key_value_heads
        head_dim = attention.head_dim
        try:
            from vllm.distributed import get_tensor_model_parallel_world_size
            tp_size = get_tensor_model_parallel_world_size()
        except Exception:
            tp_size = 1
        qkv_tokens = bsz * q_len
        _record_layout(
            config=config_label,
            layer_id=layer_id,
            event="attention_qkv_input",
            input_kind=input_kind,
            pre_gather_tokens=pre_gather_tokens,
            qkv_tokens=qkv_tokens,
            expected_tokens=n_tokens,
            full_qkv_ok=(qkv_tokens == n_tokens),
            tp_size=tp_size,
            bsz=bsz,
            seq_len=q_len,
            hidden_size=hidden_states.shape[-1],
        )

        qkv, _ = attention.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, num_heads + 2 * num_kv_heads, head_dim)
        query_states, key_states, value_states = qkv.split(
            [num_heads, num_kv_heads, num_kv_heads], dim=-2
        )

        query_states = vllm_rms_norm(query_states, q_norm_w, q_norm_eps)
        key_states = vllm_rms_norm(key_states, k_norm_w, k_norm_eps)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, attention.layer_idx, replace_position
            )
        if use_cache:
            past_key_value = (key_states, value_states)

        q_fa = query_states.transpose(1, 2)
        k_fa = key_states.transpose(1, 2)
        v_fa = value_states.transpose(1, 2)

        if attention_mask is not None or flash_attn_func is None:
            num_kv_groups = num_heads // num_kv_heads
            k_exp = repeat_kv(key_states, num_kv_groups).contiguous()
            v_exp = repeat_kv(value_states, num_kv_groups).contiguous()
            q_c = query_states.contiguous()
            am = attention_mask.bool() if attention_mask is not None else None
            if am is not None and am.dim() == 3:
                am = am.unsqueeze(1)
            attn_output = F.scaled_dot_product_attention(
                q_c, k_exp, v_exp, attn_mask=am, dropout_p=0.0, is_causal=False
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            attn_output = flash_attn_func(
                q_fa.contiguous(), k_fa.contiguous(), v_fa.contiguous(), causal=False
            )

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if dense.input_is_parallel:
            input_parallel = attn_output
        else:
            chunks = torch.split(
                attn_output,
                attn_output.shape[-1] // dense.tp_size,
                dim=-1,
            )
            input_parallel = chunks[dense.tp_rank].contiguous()

        bias = None if (dense.tp_rank > 0 or dense.skip_bias_add) else dense.bias
        output_parallel = dense.quant_method.apply(dense, input_parallel, bias)
        output_flat = output_parallel.view(-1, output_parallel.shape[-1])
        COMPONENT_TIMER.add_bytes("attn_rs_payload", _tensor_nbytes(output_flat))
        with COMPONENT_TIMER.time("attn.tp_reduce_scatter"):
            output_sp = tensor_model_parallel_reduce_scatter(output_flat, dim=0)
        _record_layout(
            config=config_label,
            layer_id=layer_id,
            event="attention_output",
            output_kind="sp_reduce_scatter",
            output_tokens=output_sp.shape[0],
            expected_sp_tokens=(n_tokens + tp_size - 1) // tp_size,
            expected_tokens=n_tokens,
            tp_size=tp_size,
            bsz=bsz,
            seq_len=q_len,
        )

        return AttnSPResult(output_sp, bsz, q_len, n_tokens), None, past_key_value

    return forward


def make_bsp_g_attn_rs_decoder_forward(decoder_layer, layer_id, ctrl, vllm_cfg,
                                       config_label: str, is_last_sparse: bool):
    """BSP-G: attention o_proj reduce-scatters directly to SP layout."""

    from vllm.distributed import (
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.utils import sequence_parallel_chunk

    mlp = decoder_layer.mlp
    attention = decoder_layer.attention
    input_layernorm = decoder_layer.input_layernorm
    post_attention_layernorm = decoder_layer.post_attention_layernorm

    shared_mod = mlp.shared_experts if mlp.config.num_shared_experts else None
    gate = mlp.gate
    experts = mlp.experts

    def forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        cache_position=None,
        replace_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if isinstance(hidden_states, SPHiddenState):
            sp_state = hidden_states
            bsz = sp_state.bsz
            seq_len = sp_state.seq_len
            N = sp_state.n_tokens
            tp_size = get_tensor_model_parallel_world_size()
            residual_sp = sp_state.hidden_sp.view(-1, sp_state.hidden_sp.shape[-1])
            pre_gather_tokens = residual_sp.shape[0]
            with COMPONENT_TIMER.time("moe.input_norm_sp"):
                hidden_norm_sp = input_layernorm(residual_sp)
            COMPONENT_TIMER.add_bytes("tp_gather_payload", _tensor_nbytes(hidden_norm_sp))
            with COMPONENT_TIMER.time("moe.tp_all_gather"):
                hidden_flat = tensor_model_parallel_all_gather(hidden_norm_sp, dim=0)
            hidden_states = hidden_flat[:N].view(bsz, seq_len, -1)
            input_kind = "sp_decoder_input_gathered"
        else:
            bsz, seq_len, h = hidden_states.shape
            N = bsz * seq_len
            tp_size = get_tensor_model_parallel_world_size()
            pre_gather_tokens = N
            residual_flat = hidden_states.view(-1, h)
            with COMPONENT_TIMER.time("moe.bsp_chunk"):
                residual_sp = sequence_parallel_chunk(residual_flat)
            hidden_states = input_layernorm(hidden_states)
            input_kind = "full_tensor"

        _record_layout(
            config=config_label,
            layer_id=layer_id,
            event="attention_qkv_input",
            input_kind=input_kind,
            pre_gather_tokens=pre_gather_tokens,
            qkv_tokens=bsz * seq_len,
            expected_tokens=N,
            full_qkv_ok=(bsz * seq_len == N),
            tp_size=tp_size,
            bsz=bsz,
            seq_len=seq_len,
            hidden_size=hidden_states.shape[-1],
        )

        attn_out, self_attn_weights, present_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            replace_position=replace_position,
        )
        assert isinstance(attn_out, AttnSPResult), type(attn_out)

        with set_forward_context(
            attn_metadata=None,
            vllm_config=vllm_cfg,
            num_tokens=N,
        ):
            hidden_after_attn_sp = residual_sp + attn_out.attn_sp.to(residual_sp.device)

            with COMPONENT_TIMER.time("moe.post_attn_norm_sp"):
                hs_sp = post_attention_layernorm(hidden_after_attn_sp)

            _record_shape(
                config_label,
                layer_id,
                ctrl,
                bsz,
                seq_len,
                N,
                hidden_after_attn_sp.shape[-1],
                tp_size=tp_size,
                mode="bsp_g_attn_rs_sp",
                n_sp_actual=hs_sp.shape[0],
            )

            with COMPONENT_TIMER.time("moe.shared"):
                shared_res = shared_mod(hs_sp) if shared_mod is not None else None
            with COMPONENT_TIMER.time("moe.gate_logits"):
                router_logits = gate.get_logits(hs_sp)
            with COMPONENT_TIMER.time("moe.native_forward"):
                y_sp = experts.forward_impl(hs_sp, router_logits)
            if shared_res is not None:
                y_sp = y_sp + shared_res

            hidden_sp = hidden_after_attn_sp + y_sp.to(hidden_after_attn_sp.device)

        if is_last_sparse:
            if os.environ.get("DINF_SP_LM_HEAD"):
                from dinfer.model.modeling_llada2_moe import _BSPGSPHiddenState
                hidden_states_out = _BSPGSPHiddenState(hidden_sp, bsz, seq_len, N)
            else:
                COMPONENT_TIMER.add_bytes("tp_gather_payload", _tensor_nbytes(hidden_sp))
                with COMPONENT_TIMER.time("moe.tp_all_gather"):
                    hidden_flat = tensor_model_parallel_all_gather(hidden_sp, dim=0)
                hidden_states_out = hidden_flat[:N].view(bsz, seq_len, -1)
        else:
            hidden_states_out = SPHiddenState(hidden_sp, bsz, seq_len, N)

        outputs = (hidden_states_out,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (None,)
        return outputs

    return forward


def make_bsp_h_decoder_forward(decoder_layer, layer_id, ctrl, vllm_cfg,
                               config_label: str, dp_rank: int,
                               is_last_sparse: bool):
    """BSP-H: BSP-G attention RS + EP AllReduce replacing combine+TP-gather."""

    from contextlib import nullcontext
    from vllm.distributed import (
        get_ep_group,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.model_executor.models.utils import sequence_parallel_chunk

    mlp = decoder_layer.mlp
    attention = decoder_layer.attention
    input_layernorm = decoder_layer.input_layernorm
    post_attention_layernorm = decoder_layer.post_attention_layernorm

    shared_mod = mlp.shared_experts if mlp.config.num_shared_experts else None
    gate = mlp.gate
    experts = mlp.experts

    def forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        cache_position=None,
        replace_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        tp_size = get_tensor_model_parallel_world_size()

        if isinstance(hidden_states, SPHiddenState):
            sp_state = hidden_states
            bsz = sp_state.bsz
            seq_len = sp_state.seq_len
            N = sp_state.n_tokens
            residual_sp = sp_state.hidden_sp.view(-1, sp_state.hidden_sp.shape[-1])
            with COMPONENT_TIMER.time("moe.input_norm_sp"):
                hidden_norm_sp = input_layernorm(residual_sp)
            COMPONENT_TIMER.add_bytes("tp_gather_payload", _tensor_nbytes(hidden_norm_sp))
            with COMPONENT_TIMER.time("moe.tp_all_gather"):
                hidden_flat = tensor_model_parallel_all_gather(hidden_norm_sp, dim=0)
            hidden_states = hidden_flat[:N].view(bsz, seq_len, -1)
        else:
            bsz, seq_len, h = hidden_states.shape
            N = bsz * seq_len
            residual_flat = hidden_states.view(-1, h)
            with COMPONENT_TIMER.time("moe.bsp_chunk"):
                residual_sp = sequence_parallel_chunk(residual_flat)
            hidden_states = input_layernorm(hidden_states)

        attn_out, self_attn_weights, present_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            replace_position=replace_position,
        )
        assert isinstance(attn_out, AttnSPResult), type(attn_out)

        with set_forward_context(
            attn_metadata=None,
            vllm_config=vllm_cfg,
            num_tokens=N,
        ):
            hidden_after_attn_sp = residual_sp + attn_out.attn_sp.to(residual_sp.device)

            with COMPONENT_TIMER.time("moe.post_attn_norm_sp"):
                hs_sp = post_attention_layernorm(hidden_after_attn_sp)

            with COMPONENT_TIMER.time("moe.shared"):
                shared_res = shared_mod(hs_sp) if shared_mod is not None else None
            with COMPONENT_TIMER.time("moe.gate_logits"):
                router_logits = gate.get_logits(hs_sp)

            experts.ensure_moe_quant_config()
            ctx = get_forward_context()
            sp_ctx = (
                ctx.dp_metadata.sp_local_sizes(experts.sp_size)
                if ctx.dp_metadata
                else nullcontext()
            )
            with sp_ctx:
                sizes = ctx.dp_metadata.get_chunk_sizes_across_dp_rank()

                COMPONENT_TIMER.add_bytes(
                    "dispatch_payload",
                    _tensor_nbytes(hs_sp) + _tensor_nbytes(router_logits),
                )
                with COMPONENT_TIMER.time("moe.dispatch"):
                    hs_gathered, lg_gathered = get_ep_group().dispatch(
                        hs_sp, router_logits, True
                    )

                with COMPONENT_TIMER.time("moe.quant_apply"):
                    partial_moe = experts.quant_method.apply(
                        layer=experts,
                        x=hs_gathered,
                        router_logits=lg_gathered,
                        top_k=experts.top_k,
                        renormalize=experts.renormalize,
                        use_grouped_topk=experts.use_grouped_topk,
                        global_num_experts=experts.global_num_experts,
                        expert_map=experts.expert_map,
                        topk_group=experts.topk_group,
                        num_expert_group=experts.num_expert_group,
                        custom_routing_function=experts.custom_routing_function,
                        scoring_func=experts.scoring_func,
                        routed_scaling_factor=experts.routed_scaling_factor,
                        e_score_correction_bias=experts.e_score_correction_bias,
                        activation=experts.activation,
                        apply_router_weight_on_input=experts.apply_router_weight_on_input,
                        enable_eplb=experts.enable_eplb,
                        expert_load_view=experts.expert_load_view,
                        logical_to_physical_map=experts.logical_to_physical_map,
                        logical_replica_count=experts.logical_replica_count,
                    )

                if isinstance(partial_moe, tuple):
                    partial_moe = partial_moe[0]

                ep_rank = get_ep_group().rank_in_group
                local_start = sum(sizes[:ep_rank])
                local_size = sizes[ep_rank]
                fused_sp = hidden_after_attn_sp
                if shared_res is not None:
                    fused_sp = fused_sp + shared_res.to(fused_sp.device)
                partial_moe[local_start:local_start + local_size].add_(fused_sp)

                COMPONENT_TIMER.add_bytes("ep_full_allreduce_payload", _tensor_nbytes(partial_moe))
                with COMPONENT_TIMER.time("moe.ep_full_all_reduce"):
                    partial_moe = get_ep_group().all_reduce(partial_moe)

                dp_group_start = dp_rank * tp_size
                dp_group_end = dp_group_start + tp_size
                dp_start = sum(sizes[:dp_group_start])
                dp_padded = sum(sizes[dp_group_start:dp_group_end])
                hidden_flat = partial_moe[dp_start:dp_start + dp_padded][:N]

        if is_last_sparse and os.environ.get("DINF_SP_LM_HEAD"):
            from dinfer.model.modeling_llada2_moe import _BSPGSPHiddenState
            hidden_sp_out = sequence_parallel_chunk(hidden_flat)
            hidden_states_out = _BSPGSPHiddenState(hidden_sp_out, bsz, seq_len, N)
        elif is_last_sparse:
            hidden_states_out = hidden_flat.view(bsz, seq_len, -1)
        else:
            hidden_states_out = hidden_flat.view(bsz, seq_len, -1)

        outputs = (hidden_states_out,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (None,)
        return outputs

    return forward


def make_bsp_team_decoder_forward(decoder_layer, layer_id, ctrl, vllm_cfg,
                                  config_label: str, is_last_sparse: bool,
                                  skip_ctrl: 'DecodedSkipController'):
    """BSP-G + TEAM decoded-token skip: skip MoE for decoded positions."""

    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.utils import sequence_parallel_chunk

    mlp = decoder_layer.mlp
    attention = decoder_layer.attention
    input_layernorm = decoder_layer.input_layernorm
    post_attention_layernorm = decoder_layer.post_attention_layernorm

    shared_mod = mlp.shared_experts if mlp.config.num_shared_experts else None
    gate = mlp.gate
    experts = mlp.experts

    def forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        cache_position=None,
        replace_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        # === Attention path (identical to BSP-G) ===
        if isinstance(hidden_states, SPHiddenState):
            sp_state = hidden_states
            bsz = sp_state.bsz
            seq_len = sp_state.seq_len
            N = sp_state.n_tokens
            residual_sp = sp_state.hidden_sp.view(-1, sp_state.hidden_sp.shape[-1])
            with COMPONENT_TIMER.time("moe.input_norm_sp"):
                hidden_norm_sp = input_layernorm(residual_sp)
            COMPONENT_TIMER.add_bytes("tp_gather_payload", _tensor_nbytes(hidden_norm_sp))
            with COMPONENT_TIMER.time("moe.tp_all_gather"):
                hidden_flat = tensor_model_parallel_all_gather(hidden_norm_sp, dim=0)
            hidden_states = hidden_flat[:N].view(bsz, seq_len, -1)
        else:
            bsz, seq_len, h = hidden_states.shape
            N = bsz * seq_len
            residual_flat = hidden_states.view(-1, h)
            with COMPONENT_TIMER.time("moe.bsp_chunk"):
                residual_sp = sequence_parallel_chunk(residual_flat)
            hidden_states = input_layernorm(hidden_states)

        attn_out, self_attn_weights, present_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            replace_position=replace_position,
        )
        assert isinstance(attn_out, AttnSPResult), type(attn_out)

        with set_forward_context(
            attn_metadata=None,
            vllm_config=vllm_cfg,
            num_tokens=N,
        ):
            hidden_after_attn_sp = residual_sp + attn_out.attn_sp.to(residual_sp.device)

            with COMPONENT_TIMER.time("moe.post_attn_norm_sp"):
                hs_sp = post_attention_layernorm(hidden_after_attn_sp)

            # === shared expert on ALL tokens (always fresh) ===
            with COMPONENT_TIMER.time("moe.shared"):
                shared_res = shared_mod(hs_sp) if shared_mod is not None else None

            # === gate on ALL tokens (EB s_mask quality) ===
            with COMPONENT_TIMER.time("moe.gate_logits"):
                router_logits = gate.get_logits(hs_sp)

            # === routed MoE with decoded-token skip ===
            N_sp = hs_sp.shape[0]

            if (skip_ctrl.should_skip() and skip_ctrl._current_input_ids is not None
                    and layer_id in skip_ctrl.moe_cache
                    and skip_ctrl.moe_cache[layer_id].shape[0] == N_sp):
                mask_sp = skip_ctrl.get_mask_sp(tp_rank, tp_size)
                mask_idx = mask_sp.nonzero(as_tuple=True)[0]
                n_mask = mask_idx.numel()

                import torch.distributed as _dist
                from vllm.distributed import get_ep_group as _get_ep
                _ep = _get_ep()
                _n_mask_t = torch.tensor([n_mask], device=hs_sp.device, dtype=torch.long)
                _dist.all_reduce(_n_mask_t, op=_dist.ReduceOp.MAX, group=_ep.device_group)
                max_n_mask = int(_n_mask_t.item())

                if max_n_mask == 0:
                    with COMPONENT_TIMER.time("moe.cache_hit"):
                        y_routed_sp = skip_ctrl.moe_cache[layer_id]
                    skip_ctrl.skipped_tokens += N_sp
                else:
                    hs_padded = torch.zeros(max_n_mask, hs_sp.shape[1],
                                            dtype=hs_sp.dtype, device=hs_sp.device)
                    lg_padded = torch.zeros(max_n_mask, router_logits.shape[1],
                                            dtype=router_logits.dtype, device=router_logits.device)
                    if n_mask > 0:
                        hs_padded[:n_mask] = hs_sp[mask_idx]
                        lg_padded[:n_mask] = router_logits[mask_idx]

                    n_mask_dp = max_n_mask * tp_size
                    with set_forward_context(
                        attn_metadata=None, vllm_config=vllm_cfg,
                        num_tokens=n_mask_dp,
                    ):
                        with COMPONENT_TIMER.time("moe.native_forward"):
                            y_padded = experts.forward_impl(hs_padded, lg_padded)
                    if isinstance(y_padded, tuple):
                        y_padded = y_padded[1]

                    if n_mask == 0:
                        y_routed_sp = skip_ctrl.moe_cache[layer_id]
                    elif n_mask < N_sp:
                        with COMPONENT_TIMER.time("moe.cache_merge"):
                            y_routed_sp = skip_ctrl.moe_cache[layer_id].clone()
                            y_routed_sp[mask_idx] = y_padded[:n_mask]
                    else:
                        y_routed_sp = y_padded[:n_mask]

                    skip_ctrl.skipped_tokens += max(N_sp - n_mask, 0)
                    skip_ctrl.total_tokens += N_sp
            else:
                with COMPONENT_TIMER.time("moe.native_forward"):
                    y_routed_sp = experts.forward_impl(hs_sp, router_logits)
                if isinstance(y_routed_sp, tuple):
                    y_routed_sp = y_routed_sp[1]
                skip_ctrl.total_tokens += N_sp

            skip_ctrl.moe_cache[layer_id] = y_routed_sp.detach().clone()

            y_sp = y_routed_sp
            if shared_res is not None:
                y_sp = y_sp + shared_res
            hidden_sp = hidden_after_attn_sp + y_sp.to(hidden_after_attn_sp.device)

        # === Output (identical to BSP-G) ===
        if is_last_sparse:
            if os.environ.get("DINF_SP_LM_HEAD"):
                from dinfer.model.modeling_llada2_moe import _BSPGSPHiddenState
                hidden_states_out = _BSPGSPHiddenState(hidden_sp, bsz, seq_len, N)
            else:
                COMPONENT_TIMER.add_bytes("tp_gather_payload", _tensor_nbytes(hidden_sp))
                with COMPONENT_TIMER.time("moe.tp_all_gather"):
                    hidden_flat = tensor_model_parallel_all_gather(hidden_sp, dim=0)
                hidden_states_out = hidden_flat[:N].view(bsz, seq_len, -1)
        else:
            hidden_states_out = SPHiddenState(hidden_sp, bsz, seq_len, N)

        outputs = (hidden_states_out,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (None,)
        return outputs

    return forward


def make_bsp_team_v2_decoder_forward(decoder_layer, layer_id, ctrl, vllm_cfg,
                                     config_label: str, is_last_sparse: bool,
                                     skip_ctrl: 'DecodedSkipController'):
    """BSP-G + TEAM null-expert: identical to BSP-G except cache read/write."""

    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.utils import sequence_parallel_chunk

    mlp = decoder_layer.mlp
    attention = decoder_layer.attention
    input_layernorm = decoder_layer.input_layernorm
    post_attention_layernorm = decoder_layer.post_attention_layernorm

    shared_mod = mlp.shared_experts if mlp.config.num_shared_experts else None
    gate = mlp.gate
    experts = mlp.experts

    def forward(
        hidden_states, attention_mask=None, position_ids=None,
        past_key_value=None, output_attentions=False,
        output_router_logits=False, use_cache=False,
        cache_position=None, replace_position=None,
        position_embeddings=None, **kwargs,
    ):
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        if isinstance(hidden_states, SPHiddenState):
            sp_state = hidden_states
            bsz = sp_state.bsz
            seq_len = sp_state.seq_len
            N = sp_state.n_tokens
            residual_sp = sp_state.hidden_sp.view(-1, sp_state.hidden_sp.shape[-1])
            with COMPONENT_TIMER.time("moe.input_norm_sp"):
                hidden_norm_sp = input_layernorm(residual_sp)
            COMPONENT_TIMER.add_bytes("tp_gather_payload", _tensor_nbytes(hidden_norm_sp))
            with COMPONENT_TIMER.time("moe.tp_all_gather"):
                hidden_flat = tensor_model_parallel_all_gather(hidden_norm_sp, dim=0)
            hidden_states = hidden_flat[:N].view(bsz, seq_len, -1)
        else:
            bsz, seq_len, h = hidden_states.shape
            N = bsz * seq_len
            residual_flat = hidden_states.view(-1, h)
            with COMPONENT_TIMER.time("moe.bsp_chunk"):
                residual_sp = sequence_parallel_chunk(residual_flat)
            hidden_states = input_layernorm(hidden_states)

        attn_out, self_attn_weights, present_key_value = attention(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache,
            cache_position=cache_position, position_embeddings=position_embeddings,
            replace_position=replace_position,
        )
        assert isinstance(attn_out, AttnSPResult), type(attn_out)

        with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg, num_tokens=N):
            hidden_after_attn_sp = residual_sp + attn_out.attn_sp.to(residual_sp.device)

            with COMPONENT_TIMER.time("moe.post_attn_norm_sp"):
                hs_sp = post_attention_layernorm(hidden_after_attn_sp)

            with COMPONENT_TIMER.time("moe.shared"):
                shared_res = shared_mod(hs_sp) if shared_mod is not None else None
            with COMPONENT_TIMER.time("moe.gate_logits"):
                router_logits = gate.get_logits(hs_sp)

            with COMPONENT_TIMER.time("moe.native_forward"):
                y_sp = experts.forward_impl(hs_sp, router_logits)
            if isinstance(y_sp, tuple):
                y_sp = y_sp[1]

            N_sp = hs_sp.shape[0]
            if (skip_ctrl.should_skip()
                    and layer_id in skip_ctrl.moe_cache
                    and skip_ctrl.moe_cache[layer_id].shape[0] == N_sp):
                decoded_sp = ~skip_ctrl.get_mask_sp(tp_rank, tp_size)
                with COMPONENT_TIMER.time("moe.cache_merge"):
                    y_sp[decoded_sp] = skip_ctrl.moe_cache[layer_id][decoded_sp]
                skip_ctrl.skipped_tokens += int(decoded_sp.sum().item())
                skip_ctrl.total_tokens += N_sp

            skip_ctrl.moe_cache[layer_id] = y_sp.detach().clone()

            if shared_res is not None:
                y_sp = y_sp + shared_res
            hidden_sp = hidden_after_attn_sp + y_sp.to(hidden_after_attn_sp.device)

        if is_last_sparse:
            if os.environ.get("DINF_SP_LM_HEAD"):
                from dinfer.model.modeling_llada2_moe import _BSPGSPHiddenState
                hidden_states_out = _BSPGSPHiddenState(hidden_sp, bsz, seq_len, N)
            else:
                COMPONENT_TIMER.add_bytes("tp_gather_payload", _tensor_nbytes(hidden_sp))
                with COMPONENT_TIMER.time("moe.tp_all_gather"):
                    hidden_flat = tensor_model_parallel_all_gather(hidden_sp, dim=0)
                hidden_states_out = hidden_flat[:N].view(bsz, seq_len, -1)
        else:
            hidden_states_out = SPHiddenState(hidden_sp, bsz, seq_len, N)

        outputs = (hidden_states_out,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (None,)
        return outputs

    return forward


def make_bsp_g2_sp_parity_decoder_forward(decoder_layer, layer_id, ctrl, vllm_cfg,
                                          config_label: str, is_last_sparse: bool):
    """BSP-G2: own SP/full attention boundaries inside the attention path."""

    from vllm.distributed import (
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.utils import sequence_parallel_chunk

    mlp = decoder_layer.mlp
    attention = decoder_layer.attention
    input_layernorm = decoder_layer.input_layernorm
    post_attention_layernorm = decoder_layer.post_attention_layernorm

    shared_mod = mlp.shared_experts if mlp.config.num_shared_experts else None
    gate = mlp.gate
    experts = mlp.experts

    def forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        cache_position=None,
        replace_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        if isinstance(hidden_states, SPHiddenState):
            sp_state = hidden_states
            bsz = sp_state.bsz
            seq_len = sp_state.seq_len
            N = sp_state.n_tokens
            tp_size = get_tensor_model_parallel_world_size()
            residual_sp = sp_state.hidden_sp.view(-1, sp_state.hidden_sp.shape[-1])
            with COMPONENT_TIMER.time("moe.input_norm_sp"):
                hidden_norm_sp = input_layernorm(residual_sp)
            attn_input = SPAttentionInput(hidden_norm_sp, bsz, seq_len, N)
        else:
            bsz, seq_len, h = hidden_states.shape
            N = bsz * seq_len
            tp_size = get_tensor_model_parallel_world_size()
            residual_flat = hidden_states.view(-1, h)
            with COMPONENT_TIMER.time("moe.bsp_chunk"):
                residual_sp = sequence_parallel_chunk(residual_flat)
            attn_input = input_layernorm(hidden_states)

        attn_out, self_attn_weights, present_key_value = attention(
            hidden_states=attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            replace_position=replace_position,
        )
        assert isinstance(attn_out, AttnSPResult), type(attn_out)

        with set_forward_context(
            attn_metadata=None,
            vllm_config=vllm_cfg,
            num_tokens=N,
        ):
            hidden_after_attn_sp = residual_sp + attn_out.attn_sp.to(residual_sp.device)

            with COMPONENT_TIMER.time("moe.post_attn_norm_sp"):
                hs_sp = post_attention_layernorm(hidden_after_attn_sp)

            _record_shape(
                config_label,
                layer_id,
                ctrl,
                bsz,
                seq_len,
                N,
                hidden_after_attn_sp.shape[-1],
                tp_size=tp_size,
                mode="bsp_g2_sp_parity",
                n_sp_actual=hs_sp.shape[0],
            )

            with COMPONENT_TIMER.time("moe.shared"):
                shared_res = shared_mod(hs_sp) if shared_mod is not None else None
            with COMPONENT_TIMER.time("moe.gate_logits"):
                router_logits = gate.get_logits(hs_sp)
            with COMPONENT_TIMER.time("moe.native_forward"):
                y_sp = experts.forward_impl(hs_sp, router_logits)
            if shared_res is not None:
                y_sp = y_sp + shared_res

            hidden_sp = hidden_after_attn_sp + y_sp.to(hidden_after_attn_sp.device)

        if is_last_sparse:
            if os.environ.get("DINF_SP_LM_HEAD"):
                from dinfer.model.modeling_llada2_moe import _BSPGSPHiddenState
                hidden_states_out = _BSPGSPHiddenState(hidden_sp, bsz, seq_len, N)
            else:
                COMPONENT_TIMER.add_bytes("tp_gather_payload", _tensor_nbytes(hidden_sp))
                with COMPONENT_TIMER.time("moe.tp_all_gather"):
                    hidden_flat = tensor_model_parallel_all_gather(hidden_sp, dim=0)
                hidden_states_out = hidden_flat[:N].view(bsz, seq_len, -1)
        else:
            hidden_states_out = SPHiddenState(hidden_sp, bsz, seq_len, N)

        outputs = (hidden_states_out,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (None,)
        return outputs

    return forward


def make_bsp_allreduce_full_decoder_forward(decoder_layer, layer_id, ctrl, vllm_cfg,
                                            config_label: str, dp_rank: int):
    """C++ P0 probe: replace combine+TP-gather with one EP all-reduce over full tokens."""

    from vllm.distributed import (
        get_ep_group,
        get_tensor_model_parallel_world_size,
    )
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.model_executor.models.utils import sequence_parallel_chunk

    mlp = decoder_layer.mlp
    attention = decoder_layer.attention
    input_layernorm = decoder_layer.input_layernorm
    post_attention_layernorm = decoder_layer.post_attention_layernorm

    shared_mod = mlp.shared_experts if mlp.config.num_shared_experts else None
    gate = mlp.gate
    experts = mlp.experts

    def routed_partial_global(hidden_states_sp, router_logits_sp):
        """Run native EP dispatch + local expert compute, but skip combine."""
        assert experts.quant_method is not None
        experts.ensure_moe_quant_config()

        ctx = get_forward_context()
        sp_ctx = (
            ctx.dp_metadata.sp_local_sizes(experts.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )
        with sp_ctx:
            sizes = ctx.dp_metadata.get_chunk_sizes_across_dp_rank()
            COMPONENT_TIMER.add_bytes(
                "dispatch_payload",
                _tensor_nbytes(hidden_states_sp) + _tensor_nbytes(router_logits_sp),
            )
            with COMPONENT_TIMER.time("moe.dispatch"):
                hidden_states_global, router_logits_global = get_ep_group().dispatch(
                    hidden_states_sp, router_logits_sp, True
                )
            with COMPONENT_TIMER.time("moe.quant_apply"):
                final_hidden_states = experts.quant_method.apply(
                    layer=experts,
                    x=hidden_states_global,
                    router_logits=router_logits_global,
                    top_k=experts.top_k,
                    renormalize=experts.renormalize,
                    use_grouped_topk=experts.use_grouped_topk,
                    global_num_experts=experts.global_num_experts,
                    expert_map=experts.expert_map,
                    topk_group=experts.topk_group,
                    num_expert_group=experts.num_expert_group,
                    custom_routing_function=experts.custom_routing_function,
                    scoring_func=experts.scoring_func,
                    routed_scaling_factor=experts.routed_scaling_factor,
                    e_score_correction_bias=experts.e_score_correction_bias,
                    activation=experts.activation,
                    apply_router_weight_on_input=experts.apply_router_weight_on_input,
                    enable_eplb=experts.enable_eplb,
                    expert_load_view=experts.expert_load_view,
                    logical_to_physical_map=experts.logical_to_physical_map,
                    logical_replica_count=experts.logical_replica_count,
                )
        if isinstance(final_hidden_states, tuple):
            final_hidden_states = final_hidden_states[0]
        return final_hidden_states, sizes

    def forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        cache_position=None,
        replace_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            replace_position=replace_position,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = post_attention_layernorm(hidden_states)

        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        residual_flat = residual.view(-1, h)
        N = hs_flat.shape[0]
        tp_size = get_tensor_model_parallel_world_size()

        with set_forward_context(
            attn_metadata=None,
            vllm_config=vllm_cfg,
            num_tokens=N,
        ):
            with COMPONENT_TIMER.time("moe.bsp_chunk"):
                hs_sp = sequence_parallel_chunk(hs_flat)
                residual_sp = sequence_parallel_chunk(residual_flat)

            _record_shape(
                config_label,
                layer_id,
                ctrl,
                bsz,
                seq_len,
                N,
                h,
                tp_size=tp_size,
                mode="bsp_allreduce_full_probe",
                n_sp_actual=hs_sp.shape[0],
            )

            with COMPONENT_TIMER.time("moe.shared"):
                shared_res = shared_mod(hs_sp) if shared_mod is not None else None
            with COMPONENT_TIMER.time("moe.gate_logits"):
                router_logits = gate.get_logits(hs_sp)
            with COMPONENT_TIMER.time("moe.native_forward"):
                y_global_partial, sizes = routed_partial_global(hs_sp, router_logits)

            if shared_res is not None:
                local_sp = residual_sp + shared_res.to(residual_sp.device)
            else:
                local_sp = residual_sp

            full_buffer = y_global_partial
            ep_rank = get_ep_group().rank_in_group
            local_start = sum(sizes[:ep_rank])
            local_size = sizes[ep_rank]
            assert local_size == local_sp.shape[0], (
                f"local_size mismatch: {local_size} vs {local_sp.shape[0]}"
            )
            full_buffer[local_start:local_start + local_size].add_(local_sp)

            COMPONENT_TIMER.add_bytes("ep_full_allreduce_payload", _tensor_nbytes(full_buffer))
            with COMPONENT_TIMER.time("moe.ep_full_all_reduce"):
                full_buffer = get_ep_group().all_reduce(full_buffer)

            dp_group_start = dp_rank * tp_size
            dp_group_end = dp_group_start + tp_size
            dp_start = sum(sizes[:dp_group_start])
            dp_padded = sum(sizes[dp_group_start:dp_group_end])
            hidden_flat = full_buffer[dp_start:dp_start + dp_padded][:N]

        hidden_states = hidden_flat.view(bsz, seq_len, h)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (None,)
        return outputs

    return forward


def _set_experts_sequence_parallel(experts, enabled: bool, tp_size: int):
    experts.is_sequence_parallel = bool(enabled)
    experts.sp_size = int(tp_size) if enabled else 1


def _print_timing_summary(label: str, timing_summary):
    if dist.get_rank() != 0 or timing_summary is None:
        return
    components = timing_summary["components"]
    bytes_info = timing_summary["byte_components"]
    print(f"  Component timing ({label}, rank-max ms/fwd):")
    ordered = [
        "global.embedding",
        "dense.mlp",
        "attn.input_all_gather",
        "attn.qkv_proj",
        "attn.kv_cache_update",
        "attn.flash_compute",
        "attn.tp_reduce_scatter",
        "moe.bsp_chunk",
        "moe.input_norm_sp",
        "moe.shared",
        "moe.gate_logits",
        "moe.post_attn_norm_sp",
        "moe.native_forward",
        "moe.dispatch",
        "moe.quant_apply",
        "moe.pre_combine_sync",
        "moe.pre_combine_barrier",
        "moe.combine",
        "moe.tv3_extract",
        "moe.tv3_disp_alloc",
        "moe.tv3_disp_post",
        "moe.tv3_comb_pre",
        "moe.tv3_comb_post",
        "moe.td4_cache_merge",
        "moe.td4_cache_write",
        "model.mask_compute",
        "model.mask_allgather",
        "model.mask_postproc",
        "moe.ep_full_all_reduce",
        "moe.cache_hit",
        "moe.cache_merge",
        "moe.tp_all_reduce",
        "moe.tp_all_gather",
        "global.lm_head",
        "global.logits_float",
    ]
    for name in ordered:
        if name not in components:
            continue
        item = components[name]
        rmax = item['ms_per_fwd_rankmax']
        rmean = item.get('ms_per_fwd_rankmean', rmax)
        rmin = item.get('ms_per_fwd_rankmin', rmax)
        tail = rmax - rmean
        print(
            f"    {name:<22} max={rmax:>7.3f} mean={rmean:>7.3f} "
            f"tail={tail:>7.3f} count_max={item['count_rankmax']}"
        )
    for name in [
        "attn_input_gather_payload",
        "attn_rs_payload",
        "dispatch_payload",
        "tp_gather_payload",
        "ep_full_allreduce_payload",
    ]:
        if name in bytes_info:
            mb = bytes_info[name]["mb_per_fwd_rankmax"]
            print(f"    {name:<22} {mb:>8.3f} MB/fwd")


def _print_shape_summary(label: str, gathered):
    if dist.get_rank() != 0 or gathered is None:
        return
    merged = Counter()
    records = []
    for item in gathered:
        records.extend(item["records"])
        for row in item["summary"]:
            key = (
                row["config"],
                row["path_pred"],
                row["bsz"],
                row["seq_len"],
                row["N_dp"],
                row["tp_size"],
                row["N_sp_expected"],
                row["pad_tokens"],
            )
            merged[key] += row["count"]

    print(f"  Shape summary ({label}):")
    print(
        f"    {'path':<16} {'bsz':>5} {'seq':>5} {'N_dp':>8} "
        f"{'tp':>3} {'N_sp':>8} {'pad':>5} {'count':>8}"
    )
    for key, count in merged.most_common(12):
        _cfg, path, bsz, seq_len, n_dp, tp_size, n_sp, pad = key
        print(
            f"    {path:<16} {bsz:>5} {seq_len:>5} {n_dp:>8} "
            f"{tp_size:>3} {n_sp:>8} {pad:>5} {count:>8}"
        )
    print(f"  Shape records sample ({label}, first {min(8, len(records))}):")
    for row in records[:8]:
        print(f"    {row}")


def _print_layout_summary(label: str, gathered):
    if dist.get_rank() != 0 or not gathered:
        return
    bad = []
    qkv_records = 0
    for item in gathered:
        for row in item.get("records", []):
            if row.get("event") != "attention_qkv_input":
                continue
            qkv_records += 1
            if row.get("full_qkv_ok") is False:
                bad.append(row)
    if qkv_records == 0:
        print(f"  Layout diagnostics ({label}): no attention_qkv_input records")
        return
    if bad:
        print(
            f"  Layout diagnostics ({label}): INVALID "
            f"{len(bad)}/{qkv_records} sampled QKV inputs are not full-token"
        )
        for row in bad[:5]:
            print(f"    bad: {row}")
    else:
        print(
            f"  Layout diagnostics ({label}): OK "
            f"{qkv_records} sampled QKV inputs are full-token"
        )


def _gather_controller_stats(ctrl):
    local = {
        "rank": dist.get_rank(),
        "stats": ctrl.stats(),
    }
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    return gathered if dist.get_rank() == 0 else []


def _print_controller_rank_summary(label: str, gathered):
    if dist.get_rank() != 0 or not gathered:
        return

    path_variants = {}
    ep_reduce_calls = []
    ep_reduce_mb = []
    for item in gathered:
        stats = item["stats"]
        key = json.dumps(stats.get("path_counts", {}), sort_keys=True)
        path_variants.setdefault(key, []).append(item["rank"])
        if "ep_reduce_calls" in stats:
            ep_reduce_calls.append(int(stats["ep_reduce_calls"]))
            ep_reduce_mb.append(float(stats.get("ep_reduce_mb", 0.0)))

    if len(path_variants) == 1:
        ranks = next(iter(path_variants.values()))
        print(f"  Controller rank check ({label}): path_counts identical on {len(ranks)} ranks")
    else:
        print(f"  Controller rank check ({label}): path_counts differ across ranks")
        for key, ranks in sorted(path_variants.items()):
            print(f"    ranks={ranks}: {key}")

    if ep_reduce_calls:
        print(
            f"    ep_reduce_calls min/max={min(ep_reduce_calls)}/{max(ep_reduce_calls)}, "
            f"ep_reduce_mb min/max={min(ep_reduce_mb):.3f}/{max(ep_reduce_mb):.3f}"
        )


def _manual_quality_payload(
    tokenizer,
    out,
    prompt_len: int,
    full_output: bool = False,
    local_bs: int = 0,
    dp_rank: int = 0,
):
    from test_heteval128 import VERIFIABLE

    gen = out[:, prompt_len:]
    local_snippets = {}
    global_start = int(dp_rank) * int(local_bs)
    global_end = global_start + gen.shape[0]
    for global_bi in sorted(VERIFIABLE.keys()):
        if global_bi < global_start or global_bi >= global_end:
            continue
        local_bi = global_bi - global_start
        toks = gen[local_bi]
        valid = toks[(toks != 0) & (toks != EOS_ID) & (toks != MASK_ID)]
        text = tokenizer.decode(valid, skip_special_tokens=True)
        local_snippets[str(global_bi)] = text if full_output else text[:1200]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_snippets)
    if dist.get_rank() != 0:
        return {}
    snippets = {}
    for item in gathered:
        for key, value in sorted((item or {}).items(), key=lambda kv: int(kv[0])):
            snippets.setdefault(key, value)
    return snippets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--num-runs", type=int, default=2)
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help=(
            "Override tensor parallel size. Default keeps the historical "
            "world_size=8 -> tp=4, world_size=4 -> tp=2 mapping."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["shape", "compare", "forward-check", "moe-internal-check"],
        default="compare",
    )
    parser.add_argument("--component-timing", action="store_true")
    parser.add_argument("--shape-limit", type=int, default=64)
    parser.add_argument(
        "--profile-target",
        choices=["both", "baseline", "bsp"],
        default="both",
        help="Limit compare mode to one config for nsys-friendly profiling.",
    )
    parser.add_argument(
        "--config-set",
        choices=["all", "bspg", "bspg2", "aeg2f", "aeggf", "bspg_source", "bspg_h", "bspg_team", "team_debug", "team_g_only", "team_td4_only", "team_tv4", "team_tv5", "team_tv4m", "team_tv4m_v2", "team_tv6", "team_kp"],
        default="all",
        help=(
            "Limit compare matrix. all=A/B/C/D/E/G/F, "
            "bspg=A/E/G/F, bspg2=A/E/G/G2/F, aeg2f=A/E/G2/F, "
            "aeggf=A/E/G/G2/F, bspg_source=A/E/G/GS, "
            "team_debug=G/TD1/TD2 (null expert quality ablation)."
        ),
    )
    parser.add_argument(
        "--cuda-profiler-capture",
        action="store_true",
        help="Wrap the measured generate with cudaProfilerStart/Stop for nsys.",
    )
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip quality snippet decoding to keep profiling traces clean.",
    )
    parser.add_argument(
        "--collect-joint-stats",
        action="store_true",
        help="Collect per-layer per-GPU (pairs, active_experts, histogram) stats.",
    )
    parser.add_argument(
        "--per-layer-timing",
        action="store_true",
        help="Use per-layer timer keys (moe.quant_apply.L0 etc.) for component timing.",
    )
    parser.add_argument(
        "--isolate-combine-wait",
        action="store_true",
        help="Insert cuda.sync + dist.barrier before combine to isolate straggler wait.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Apply torch.compile(mode='reduce-overhead') to model.forward.",
    )
    parser.add_argument(
        "--full-quality-output",
        action="store_true",
        help="Save full decoded verifiable outputs instead of truncated snippets.",
    )
    parser.add_argument(
        "--layout-diagnostics",
        action="store_true",
        help="Collect passive attention-boundary SP/TP layout diagnostics.",
    )
    parser.add_argument(
        "--layout-limit",
        type=int,
        default=256,
        help="Maximum layout diagnostic records to keep per rank.",
    )
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if args.tp_size is not None:
        tp_size = args.tp_size
    elif world_size == 8:
        tp_size = 4
    elif world_size == 4:
        tp_size = 2
    else:
        raise AssertionError(f"Requires 8 GPUs primary or 4 GPUs fallback, got {world_size}")

    assert world_size % tp_size == 0, (
        f"world_size={world_size} must be divisible by tp_size={tp_size}"
    )
    dp_size = world_size // tp_size
    dp_rank = rank // tp_size
    tp_rank_local = rank % tp_size
    assert args.batch_size % dp_size == 0
    local_bs = args.batch_size // dp_size

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    alltoall_backend = os.environ.get("VLLM_ALL2ALL_BACKEND", "allgather_reducescatter")

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (
        BlockDiffusionLLM,
        BlockIteratorFactory,
        KVCacheFactory,
        ThresholdParallelDecoder,
    )
    from dinfer.model import LLaDA2MoeModelLM
    from dinfer.model.modeling_llada2_moe import (
        LLaDA2MoeMLP,
        LLaDA2MoeSparseMoeBlock,
    )
    from transformers import AutoConfig, AutoTokenizer
    from test_heteval512 import PROMPTS
    from test_fused_eb_triton import fused_routing
    from baseline_optimizations import apply_all_optimizations

    pcfg_init = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=True,
    )
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(
            world_size, rank, "env://", local_rank, "nccl"
        )

    pcfg = ParallelConfig(
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        enable_expert_parallel=True,
    )
    vllm_cfg = VllmConfig(parallel_config=pcfg)

    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            backend="nccl",
        )

        from vllm.distributed import get_ep_group, prepare_communication_buffer_for_model
        from vllm.forward_context import set_forward_context

        if rank == 0:
            print("=" * 80)
            print(f"BSP-MoE Benchmark — dp={dp_size} tp={tp_size} ep={world_size}")
            print(f"  batch={args.batch_size} local_bs={local_bs}")
            print(f"  gen={args.gen_length}, block={BLOCK_LENGTH}")
            print(f"  mode={args.mode}, component_timing={args.component_timing}")
            print(f"  profile_target={args.profile_target}, cuda_capture={args.cuda_profiler_capture}")
            print(f"  AllToAll backend: {alltoall_backend}")
            print("=" * 80)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        )
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        )

        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

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

        # OPT-4: fp8 communication compression
        if os.environ.get("DINF_FP8_COMM"):
            from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
            _orig_dispatch = AgRsAll2AllManager.dispatch
            _orig_combine = AgRsAll2AllManager.combine

            def _fp8_dispatch(self, hidden_states, router_logits,
                              is_sequence_parallel=False):
                hs_fp8 = hidden_states.to(torch.float8_e4m3fn).view(torch.uint8)
                lg_fp8 = router_logits.to(torch.float8_e4m3fn).view(torch.uint8)
                hs_out_u8, lg_out_u8 = _orig_dispatch(
                    self, hs_fp8, lg_fp8, is_sequence_parallel)
                hs_out = hs_out_u8.view(torch.float8_e4m3fn).to(torch.bfloat16)
                lg_out = lg_out_u8.view(torch.float8_e4m3fn).to(torch.bfloat16)
                return hs_out, lg_out

            AgRsAll2AllManager.dispatch = _fp8_dispatch
            if rank == 0:
                print("  [FP8-COMM] Enabled fp8 dispatch only (combine stays bf16)")

        if getattr(args, 'torch_compile', False):
            if rank == 0:
                print("  [COMPILE] Applying torch.compile(mode='reduce-overhead')...")
            try:
                model.forward = torch.compile(
                    model.forward, mode='reduce-overhead',
                    fullgraph=False, dynamic=True)
                with torch.inference_mode():
                    dummy = torch.randint(0, 1000, (2, 64), device=device)
                    _ = model(dummy, use_cache=False)
                torch.cuda.synchronize()
                if rank == 0:
                    print("  [COMPILE] Success! Compiled forward works.")
            except Exception as e:
                if rank == 0:
                    print(f"  [COMPILE] FAILED: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()

        moe_blocks = [
            m for _n, m in model.named_modules()
            if isinstance(m, LLaDA2MoeSparseMoeBlock)
        ]
        gate_params = []
        for _n, m in model.named_modules():
            if m.__class__.__name__ == "LLaDA2MoeGate":
                gate_params.append(
                    {
                        "bias": m.expert_bias,
                        "rsf": m.routed_scaling_factor,
                        "ng": m.n_group,
                        "tkg": m.topk_group,
                    }
                )

        orig_block_forwards = [b.forward for b in moe_blocks]
        orig_expert_forward_impls = [b.experts.forward_impl for b in moe_blocks]
        orig_expert_sp = [
            (getattr(b.experts, "is_sequence_parallel", False),
             getattr(b.experts, "sp_size", 1))
            for b in moe_blocks
        ]

        global JOINT_STATS
        global ISOLATE_COMBINE_WAIT
        ISOLATE_COMBINE_WAIT = bool(args.isolate_combine_wait)
        if args.collect_joint_stats:
            e_local = moe_blocks[0].experts.w2_weight.size(0) if moe_blocks else 32
            skip_n = 5 if args.gen_length <= 64 else 20
            JOINT_STATS = JointStatsCollector(len(moe_blocks), e_local, skip_first=skip_n)
            if rank == 0:
                print(f"  [JOINT-STATS] Enabled: {len(moe_blocks)} layers, "
                      f"E_local={e_local}, skip_first={skip_n}")
        sparse_decoder_layers = []
        dense_mlp_layers = []
        for dl in model.model.layers:
            mlp = getattr(dl, "mlp", None)
            if isinstance(mlp, LLaDA2MoeSparseMoeBlock):
                sparse_decoder_layers.append(dl)
            elif isinstance(mlp, LLaDA2MoeMLP):
                dense_mlp_layers.append(dl)
        assert len(sparse_decoder_layers) == len(moe_blocks), (
            f"Expected sparse decoder layers == moe blocks, got "
            f"{len(sparse_decoder_layers)} vs {len(moe_blocks)}"
        )
        orig_decoder_forwards = [dl.forward for dl in sparse_decoder_layers]
        orig_attention_forwards = [dl.attention.forward for dl in sparse_decoder_layers]
        orig_dense_mlp_forwards = [dl.mlp.forward for dl in dense_mlp_layers]
        orig_model_forward = model.forward

        def set_source_bsp_g(enabled: bool):
            if hasattr(model, "set_bsp_sequence_parallel_moe"):
                model.set_bsp_sequence_parallel_moe(enabled)
            try:
                from dinfer.model import modeling_llada2_moe as llada2_moe
                llada2_moe.set_bsp_g_component_timer(
                    COMPONENT_TIMER if enabled else None)
                llada2_moe.set_bsp_g_layout_recorder(
                    _record_layout if (enabled and args.layout_diagnostics) else None)
            except Exception:
                pass

        if rank == 0:
            print(f"  MoE blocks: {len(moe_blocks)}, EP size: {get_ep_group().world_size}")
            print(f"  GPU memory: {torch.cuda.memory_allocated(device) / 1e9:.1f} GB")
            if moe_blocks and getattr(moe_blocks[0], "shared_experts", None) is not None:
                shared0 = moe_blocks[0].shared_experts
                print(
                    "  Shared expert[0]: "
                    f"{shared0.__class__.__name__}, "
                    f"gate={shared0.gate_proj.__class__.__name__}, "
                    f"up={shared0.up_proj.__class__.__name__}, "
                    f"down={shared0.down_proj.__class__.__name__}"
                )

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
            torch.cat([
                torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype),
                ids,
            ])
            if ids.shape[0] < mx
            else ids
            for ids in all_ids
        ]
        input_ids_full = torch.stack(padded, dim=0)
        my_input = input_ids_full[dp_rank * local_bs: (dp_rank + 1) * local_bs].to(device)
        prompt_len = my_input.shape[1]

        if rank == 0:
            print(f"  Full input: {tuple(input_ids_full.shape)}, local: {tuple(my_input.shape)}")

        decoder = ThresholdParallelDecoder(
            temperature=0.0,
            threshold=0.90,
            mask_id=MASK_ID,
            eos_id=EOS_ID,
        )
        orig_block_init = decoder.block_init
        _gate_modules = [m for _n, m in model.named_modules()
                         if m.__class__.__name__ == "LLaDA2MoeGate"]
        _orig_gate_get_logits = [g.get_logits for g in _gate_modules]

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

        orig_block_init = decoder.block_init

        def setup_block_clock(ctrl_ref):
            def block_init_with_clock(block_x, block_id):
                ctrl_ref.note_block_start(int(block_id))
                return orig_block_init(block_x, block_id)
            decoder.block_init = block_init_with_clock

        def restore_blocks_and_experts():
            global _TEAM_SKIP_CTRL
            set_source_bsp_g(False)
            model.forward = orig_model_forward
            decoder.block_init = orig_block_init
            for gm, ogl in zip(_gate_modules, _orig_gate_get_logits):
                gm.get_logits = ogl
            _TEAM_SKIP_CTRL = None
            for dl, of in zip(sparse_decoder_layers, orig_decoder_forwards):
                dl.forward = of
            for dl, of in zip(sparse_decoder_layers, orig_attention_forwards):
                dl.attention.forward = of
            for dl, of in zip(dense_mlp_layers, orig_dense_mlp_forwards):
                dl.mlp.forward = of
            for blk, of, ofi, sp_state in zip(
                moe_blocks,
                orig_block_forwards,
                orig_expert_forward_impls,
                orig_expert_sp,
            ):
                blk.forward = of
                blk.experts.forward_impl = ofi
                blk.experts.is_sequence_parallel = sp_state[0]
                blk.experts.sp_size = sp_state[1]

        def setup_dense_mlp_timing():
            for dl in dense_mlp_layers:
                dl.mlp.forward = make_timed_dense_mlp_forward(dl.mlp)

        def setup_routing(ctrl_ref):
            gi = 0
            for _n, m in model.named_modules():
                if m.__class__.__name__ == "LLaDA2MoeGate":
                    b = m.expert_bias
                    r = m.routed_scaling_factor
                    ng = m.n_group
                    tkg = m.topk_group
                    li = gi

                    def mk(bb, rr, nn, gg, layer_i, cc):
                        def fn(hs, go, topk, renorm):
                            sm = cc.get_s_mask(layer_i, go, bb)
                            w, idx = fused_routing(
                                go,
                                bb,
                                rr,
                                s_mask=sm,
                                K=4,
                                ng=nn,
                                tkg=gg,
                            )
                            if JOINT_STATS is not None and layer_i < len(moe_blocks):
                                em = moe_blocks[layer_i].experts.expert_map
                                if em is not None:
                                    JOINT_STATS.record(layer_i, idx, em)
                                if layer_i == len(moe_blocks) - 1:
                                    JOINT_STATS.on_forward_end()
                            return w.to(go.dtype), idx
                        return fn

                    m.routing = mk(b, r, ng, tkg, li, ctrl_ref)
                    gi += 1

        def _timed_fwd_impl(experts_obj, layer_idx):
            lid = layer_idx if args.per_layer_timing else None
            return make_timed_forward_impl(experts_obj, layer_id=lid)

        def setup_baseline(ctrl_ref, label: str):
            restore_blocks_and_experts()
            setup_dense_mlp_timing()
            setup_block_clock(ctrl_ref)
            setup_routing(ctrl_ref)
            for i, blk in enumerate(moe_blocks):
                blk.forward = make_baseline_forward(blk, i, ctrl_ref, label)
                blk.experts.forward_impl = _timed_fwd_impl(blk.experts, i)

        def setup_bsp(ctrl_ref, label: str):
            restore_blocks_and_experts()
            setup_dense_mlp_timing()
            setup_block_clock(ctrl_ref)
            setup_routing(ctrl_ref)
            for i, blk in enumerate(moe_blocks):
                _set_experts_sequence_parallel(blk.experts, True, tp_size)
                blk.experts.forward_impl = _timed_fwd_impl(blk.experts, i)
                blk.forward = make_bsp_forward(blk, i, ctrl_ref, vllm_cfg, label)

        def setup_bsp_delay(ctrl_ref, label: str):
            restore_blocks_and_experts()
            setup_dense_mlp_timing()
            setup_block_clock(ctrl_ref)
            setup_routing(ctrl_ref)
            for i, blk in enumerate(moe_blocks):
                _set_experts_sequence_parallel(blk.experts, True, tp_size)
                blk.experts.forward_impl = _timed_fwd_impl(blk.experts, i)
                blk.forward = make_bsp_forward(blk, i, ctrl_ref, vllm_cfg, label)
            for i, dl in enumerate(sparse_decoder_layers):
                dl.forward = make_bsp_delay_decoder_forward(
                    dl, i, ctrl_ref, vllm_cfg, label
                )

        def setup_bsp_delay_m3(ctrl_ref, label: str):
            setup_bsp_delay(ctrl_ref, label)

        def setup_bsp_cross_layer_sp(ctrl_ref, label: str):
            restore_blocks_and_experts()
            setup_dense_mlp_timing()
            setup_block_clock(ctrl_ref)
            setup_routing(ctrl_ref)
            for i, blk in enumerate(moe_blocks):
                _set_experts_sequence_parallel(blk.experts, True, tp_size)
                blk.experts.forward_impl = _timed_fwd_impl(blk.experts, i)
            last_sparse_idx = len(sparse_decoder_layers) - 1
            for i, dl in enumerate(sparse_decoder_layers):
                dl.forward = make_bsp_cross_layer_sp_decoder_forward(
                    dl,
                    i,
                    ctrl_ref,
                    vllm_cfg,
                    label,
                    is_last_sparse=(i == last_sparse_idx),
                )

        def setup_bsp_g_attn_rs(ctrl_ref, label: str):
            restore_blocks_and_experts()
            setup_dense_mlp_timing()
            setup_block_clock(ctrl_ref)
            setup_routing(ctrl_ref)
            for i, blk in enumerate(moe_blocks):
                _set_experts_sequence_parallel(blk.experts, True, tp_size)
                blk.experts.forward_impl = _timed_fwd_impl(blk.experts, i)
            last_sparse_idx = len(sparse_decoder_layers) - 1
            for i, dl in enumerate(sparse_decoder_layers):
                dl.attention.forward = make_attention_reduce_scatter_forward(
                    dl.attention, label, i)
                dl.forward = make_bsp_g_attn_rs_decoder_forward(
                    dl,
                    i,
                    ctrl_ref,
                    vllm_cfg,
                    label,
                    is_last_sparse=(i == last_sparse_idx),
                )

        def setup_bsp_g2_sp_parity(ctrl_ref, label: str):
            restore_blocks_and_experts()
            setup_dense_mlp_timing()
            setup_block_clock(ctrl_ref)
            setup_routing(ctrl_ref)
            for i, blk in enumerate(moe_blocks):
                _set_experts_sequence_parallel(blk.experts, True, tp_size)
                blk.experts.forward_impl = _timed_fwd_impl(blk.experts, i)
            last_sparse_idx = len(sparse_decoder_layers) - 1
            for i, dl in enumerate(sparse_decoder_layers):
                dl.attention.forward = make_attention_g2_sp_parity_forward(
                    dl.attention, label, i)
                dl.forward = make_bsp_g2_sp_parity_decoder_forward(
                    dl,
                    i,
                    ctrl_ref,
                    vllm_cfg,
                    label,
                    is_last_sparse=(i == last_sparse_idx),
                )

        def setup_bsp_g_source(ctrl_ref, label: str):
            restore_blocks_and_experts()
            setup_dense_mlp_timing()
            setup_block_clock(ctrl_ref)
            setup_routing(ctrl_ref)
            for i, blk in enumerate(moe_blocks):
                blk.experts.forward_impl = _timed_fwd_impl(blk.experts, i)
            set_source_bsp_g(True)

        def setup_bsp_allreduce_full(ctrl_ref, label: str):
            restore_blocks_and_experts()
            setup_dense_mlp_timing()
            setup_block_clock(ctrl_ref)
            setup_routing(ctrl_ref)
            for blk in moe_blocks:
                _set_experts_sequence_parallel(blk.experts, True, tp_size)
            for i, dl in enumerate(sparse_decoder_layers):
                dl.forward = make_bsp_allreduce_full_decoder_forward(
                    dl,
                    i,
                    ctrl_ref,
                    vllm_cfg,
                    label,
                    dp_rank=dp_rank,
                )

        def setup_bsp_h(ctrl_ref, label: str):
            restore_blocks_and_experts()
            setup_dense_mlp_timing()
            setup_block_clock(ctrl_ref)
            setup_routing(ctrl_ref)
            for blk in moe_blocks:
                _set_experts_sequence_parallel(blk.experts, True, tp_size)
            last_sparse_idx = len(sparse_decoder_layers) - 1
            for i, dl in enumerate(sparse_decoder_layers):
                dl.attention.forward = make_attention_reduce_scatter_forward(
                    dl.attention, label, i)
                dl.forward = make_bsp_h_decoder_forward(
                    dl,
                    i,
                    ctrl_ref,
                    vllm_cfg,
                    label,
                    dp_rank=dp_rank,
                    is_last_sparse=(i == last_sparse_idx),
                )

        def setup_bsp_team(ctrl_ref, label: str):
            global _TEAM_SKIP_CTRL
            restore_blocks_and_experts()
            setup_dense_mlp_timing()
            setup_block_clock(ctrl_ref)
            setup_routing(ctrl_ref)

            skip_ctrl = DecodedSkipController(num_moe_layers=len(moe_blocks))
            _TEAM_SKIP_CTRL = skip_ctrl

            for i, blk in enumerate(moe_blocks):
                _set_experts_sequence_parallel(blk.experts, True, tp_size)
                blk.experts.forward_impl = _timed_fwd_impl(blk.experts, i)
                orig_ne = blk.experts.global_num_experts
                skip_ctrl.NULL_EXPERT_ID = orig_ne
                blk.experts.global_num_experts = orig_ne + 1
                if blk.experts.expert_map is not None:
                    blk.experts.expert_map = torch.cat([
                        blk.experts.expert_map,
                        torch.tensor([-1], device=blk.experts.expert_map.device,
                                     dtype=blk.experts.expert_map.dtype)
                    ])

            _wrap_routing_for_team(moe_blocks, skip_ctrl)

            last_sparse_idx = len(sparse_decoder_layers) - 1
            for i, dl in enumerate(sparse_decoder_layers):
                dl.attention.forward = make_attention_reduce_scatter_forward(
                    dl.attention, label, i)
                dl.forward = make_bsp_team_v2_decoder_forward(
                    dl, i, ctrl_ref, vllm_cfg, label,
                    is_last_sparse=(i == last_sparse_idx),
                    skip_ctrl=skip_ctrl,
                )

            orig_bi = decoder.block_init
            def block_init_team(block_x, block_id):
                skip_ctrl.on_block_start()
                return orig_bi(block_x, block_id)
            decoder.block_init = block_init_team

            def model_forward_team(input_ids=None, *args, **kwargs):
                if input_ids is not None:
                    skip_ctrl.set_input_ids(input_ids)
                    if skip_ctrl.should_skip():
                        _gather_decoded_mask(skip_ctrl)
                result = orig_model_forward(input_ids, *args, **kwargs)
                skip_ctrl.after_forward()
                return result
            model.forward = model_forward_team

        # ---- team_debug ablation setups (quality debug for null expert) ----
        _orig_global_num_experts = [blk.experts.global_num_experts for blk in moe_blocks]
        _orig_expert_maps = [blk.experts.expert_map.clone() if blk.experts.expert_map is not None else None
                             for blk in moe_blocks]
        _orig_custom_routing_fns = [blk.experts.custom_routing_function for blk in moe_blocks]

        def _restore_expert_attrs():
            for blk, gne, em, crf in zip(moe_blocks, _orig_global_num_experts,
                                         _orig_expert_maps, _orig_custom_routing_fns):
                blk.experts.global_num_experts = gne
                if em is not None:
                    blk.experts.expert_map = em.clone()
                blk.experts.custom_routing_function = crf

        def setup_td1(ctrl_ref, label: str):
            """G + V1: only global_num_experts=257, nothing else changed."""
            setup_bsp_g_attn_rs(ctrl_ref, label)
            _restore_expert_attrs()
            for blk in moe_blocks:
                blk.experts.global_num_experts += 1

        def setup_td2(ctrl_ref, label: str):
            """G + V1 + V2: global_num_experts=257 + expert_map[256]=-1."""
            setup_bsp_g_attn_rs(ctrl_ref, label)
            _restore_expert_attrs()
            for blk in moe_blocks:
                blk.experts.global_num_experts += 1
                if blk.experts.expert_map is not None:
                    blk.experts.expert_map = torch.cat([
                        blk.experts.expert_map,
                        torch.tensor([-1], device=blk.experts.expert_map.device,
                                     dtype=blk.experts.expert_map.dtype)
                    ])

        _td3_decoded_gathered = [None]  # mutable container for gathered decoded mask

        def setup_td3(ctrl_ref, label: str):
            """G + V1 + V2 + V3: routing wrap sets topk_ids[decoded]=256."""
            setup_bsp_g_attn_rs(ctrl_ref, label)
            _restore_expert_attrs()
            NULL_ID = _orig_global_num_experts[0]  # = 256
            for blk in moe_blocks:
                blk.experts.global_num_experts += 1
                if blk.experts.expert_map is not None:
                    blk.experts.expert_map = torch.cat([
                        blk.experts.expert_map,
                        torch.tensor([-1], device=blk.experts.expert_map.device,
                                     dtype=blk.experts.expert_map.dtype)
                    ])
                orig_rf = blk.experts.custom_routing_function

                def _make_wrap(orig_fn, null_id):
                    def wrapped(hidden_states, gating_output, topk, renormalize):
                        w, idx = orig_fn(hidden_states=hidden_states,
                                         gating_output=gating_output,
                                         topk=topk, renormalize=renormalize)
                        dec = _td3_decoded_gathered[0]
                        if dec is not None and dec.shape[0] == idx.shape[0]:
                            idx[dec] = null_id
                        return w, idx
                    return wrapped
                blk.experts.custom_routing_function = _make_wrap(orig_rf, NULL_ID)

            from vllm.distributed import (
                get_ep_group,
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            orig_mf = model.forward
            def model_forward_td3(input_ids=None, *args, **kwargs):
                if input_ids is not None:
                    tp_rank = get_tensor_model_parallel_rank()
                    tp_size = get_tensor_model_parallel_world_size()
                    is_mask = (input_ids == MASK_ID).view(-1)
                    N_dp = is_mask.shape[0]
                    chunk = N_dp // tp_size
                    sp_start = tp_rank * chunk
                    decoded_sp = ~is_mask[sp_start:sp_start + chunk]
                    decoded_byte = decoded_sp.to(torch.uint8)
                    ep = get_ep_group()
                    gathered = [torch.empty_like(decoded_byte) for _ in range(ep.world_size)]
                    dist.all_gather(gathered, decoded_byte, group=ep.device_group)
                    _td3_decoded_gathered[0] = torch.cat(gathered, dim=0).bool()
                else:
                    _td3_decoded_gathered[0] = None
                return orig_mf(input_ids, *args, **kwargs)
            model.forward = model_forward_td3

        _td4_moe_cache = {}
        _td4_null_mask_sp = [None]       # READ positions (SP-local)
        _td4_null_mask_gathered = [None]  # READ positions (gathered, extracted from dispatch)
        _td4_prev_decoded_sp = [None]
        _td4_step_in_block = [0]

        def setup_td4(ctrl_ref, label: str):
            """TEAM null expert: decoded mask piggy-backed on dispatch AllGatherV.
            No separate AllGather. prev_decoded state machine + M=5 refresh."""
            from vllm.distributed import (
                get_ep_group,
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            _restore_expert_attrs()
            setup_bsp_g_attn_rs(ctrl_ref, label)
            _td4_moe_cache.clear()
            _td4_step_in_block[0] = 0
            _td4_prev_decoded_sp[0] = None
            _td4_null_mask_sp[0] = None
            _td4_null_mask_gathered[0] = None
            NULL_ID = _orig_global_num_experts[0]
            _REFRESH_M = 5
            _tp_size = get_tensor_model_parallel_world_size()
            _tp_rank = get_tensor_model_parallel_rank()

            for blk in moe_blocks:
                blk.experts.global_num_experts += 1
                if blk.experts.expert_map is not None:
                    blk.experts.expert_map = torch.cat([
                        blk.experts.expert_map,
                        torch.tensor([-1], device=blk.experts.expert_map.device,
                                     dtype=blk.experts.expert_map.dtype)
                    ])
                orig_rf = blk.experts.custom_routing_function

                def _make_routing_wrap(orig_fn, null_id):
                    def wrapped(hidden_states, gating_output, topk, renormalize):
                        null_col = gating_output[:, -1]
                        gating_clean = gating_output[:, :-1]
                        w, idx = orig_fn(hidden_states=hidden_states,
                                         gating_output=gating_clean,
                                         topk=topk, renormalize=renormalize)
                        null_g = (null_col > 0.5)
                        _td4_null_mask_gathered[0] = null_g if null_g.any() else None
                        if null_g.any():
                            idx[null_g] = null_id
                        return w, idx
                    return wrapped
                blk.experts.custom_routing_function = _make_routing_wrap(orig_rf, NULL_ID)

            for gi, gate_mod in enumerate(g for _n, g in model.named_modules()
                                         if g.__class__.__name__ == "LLaDA2MoeGate"):
                orig_get_logits = gate_mod.get_logits
                def _make_gate_wrap(orig_gl):
                    def get_logits_with_mask(hs_sp):
                        logits = orig_gl(hs_sp)
                        null_sp = _td4_null_mask_sp[0]
                        if null_sp is not None and null_sp.shape[0] == logits.shape[0]:
                            mask_col = null_sp.to(logits.dtype).unsqueeze(1)
                        else:
                            mask_col = torch.zeros(logits.shape[0], 1, device=logits.device, dtype=logits.dtype)
                        return torch.cat([logits, mask_col], dim=1)
                    return get_logits_with_mask
                gate_mod.get_logits = _make_gate_wrap(orig_get_logits)

            for i, blk in enumerate(moe_blocks):
                prev_fwd_impl = blk.experts.forward_impl
                layer_id = i

                def _make_cached_impl(orig_impl, lid):
                    def cached_forward_impl(self, hidden_states, router_logits):
                        y = orig_impl(hidden_states, router_logits)
                        if isinstance(y, tuple):
                            y = y[1]
                        null_sp = _td4_null_mask_sp[0]
                        N_sp = y.shape[0]
                        if null_sp is None or null_sp.shape[0] != N_sp:
                            # Refresh step / prefill / cross_block / new-decoded:
                            # full MoE computed, write fresh result to cache
                            with COMPONENT_TIMER.time("moe.td4_cache_write"):
                                _td4_moe_cache[lid] = y.detach().clone()
                            return y
                        # Skip step: null expert produced 0 for decoded positions,
                        # replace with cached fresh values, do NOT overwrite cache
                        with COMPONENT_TIMER.time("moe.td4_cache_merge"):
                            if lid in _td4_moe_cache and _td4_moe_cache[lid].shape[0] == N_sp:
                                y[null_sp] = _td4_moe_cache[lid][null_sp]
                        return y
                    return types.MethodType(cached_forward_impl, blk.experts)
                blk.experts.forward_impl = _make_cached_impl(prev_fwd_impl, layer_id)

            orig_bi = decoder.block_init
            def block_init_td4(block_x, block_id):
                _td4_moe_cache.clear()
                _td4_step_in_block[0] = 0
                _td4_prev_decoded_sp[0] = None
                _td4_null_mask_sp[0] = None
                _td4_null_mask_gathered[0] = None
                return orig_bi(block_x, block_id)
            decoder.block_init = block_init_td4

            orig_mf = model.forward
            def model_forward_td4(input_ids=None, *args, **kwargs):
                if input_ids is not None:
                    step = _td4_step_in_block[0]
                    is_mask = (input_ids == MASK_ID).view(-1)
                    chunk = is_mask.shape[0] // _tp_size
                    sp_start = _tp_rank * chunk
                    decoded_sp = ~is_mask[sp_start:sp_start + chunk]

                    prev_dec = _td4_prev_decoded_sp[0]
                    if (prev_dec is not None
                            and prev_dec.shape[0] == decoded_sp.shape[0]
                            and step % _REFRESH_M != 1):
                        _td4_null_mask_sp[0] = decoded_sp & prev_dec
                    else:
                        _td4_null_mask_sp[0] = None
                else:
                    _td4_null_mask_sp[0] = None

                result = orig_mf(input_ids, *args, **kwargs)

                if input_ids is not None:
                    _td4_prev_decoded_sp[0] = decoded_sp
                    _td4_step_in_block[0] += 1
                return result
            model.forward = model_forward_td4

        # ---- TD8: null expert + periodic refresh, NO cache ----
        _td8_null_mask_sp = [None]
        _td8_null_mask_gathered = [None]
        _td8_prev_decoded_sp = [None]
        _td8_step_in_block = [0]

        def setup_td8(ctrl_ref, label: str):
            """TEAM null expert with periodic refresh, NO cache.
            Tests whether refresh-step residual correction alone is sufficient
            (without cache merge on skip steps)."""
            from vllm.distributed import (
                get_ep_group,
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            _restore_expert_attrs()
            setup_bsp_g_attn_rs(ctrl_ref, label)
            _td8_step_in_block[0] = 0
            _td8_prev_decoded_sp[0] = None
            _td8_null_mask_sp[0] = None
            _td8_null_mask_gathered[0] = None
            NULL_ID = _orig_global_num_experts[0]
            _REFRESH_M = 5
            _tp_size = get_tensor_model_parallel_world_size()
            _tp_rank = get_tensor_model_parallel_rank()

            for blk in moe_blocks:
                blk.experts.global_num_experts += 1
                if blk.experts.expert_map is not None:
                    blk.experts.expert_map = torch.cat([
                        blk.experts.expert_map,
                        torch.tensor([-1], device=blk.experts.expert_map.device,
                                     dtype=blk.experts.expert_map.dtype)
                    ])
                orig_rf = blk.experts.custom_routing_function

                def _make_routing_wrap(orig_fn, null_id):
                    def wrapped(hidden_states, gating_output, topk, renormalize):
                        null_col = gating_output[:, -1]
                        gating_clean = gating_output[:, :-1]
                        w, idx = orig_fn(hidden_states=hidden_states,
                                         gating_output=gating_clean,
                                         topk=topk, renormalize=renormalize)
                        null_g = (null_col > 0.5)
                        _td8_null_mask_gathered[0] = null_g if null_g.any() else None
                        if null_g.any():
                            idx[null_g] = null_id
                        return w, idx
                    return wrapped
                blk.experts.custom_routing_function = _make_routing_wrap(orig_rf, NULL_ID)

            for gi, gate_mod in enumerate(g for _n, g in model.named_modules()
                                         if g.__class__.__name__ == "LLaDA2MoeGate"):
                orig_get_logits = gate_mod.get_logits
                def _make_gate_wrap(orig_gl):
                    def get_logits_with_mask(hs_sp):
                        logits = orig_gl(hs_sp)
                        null_sp = _td8_null_mask_sp[0]
                        if null_sp is not None and null_sp.shape[0] == logits.shape[0]:
                            mask_col = null_sp.to(logits.dtype).unsqueeze(1)
                        else:
                            mask_col = torch.zeros(logits.shape[0], 1, device=logits.device, dtype=logits.dtype)
                        return torch.cat([logits, mask_col], dim=1)
                    return get_logits_with_mask
                gate_mod.get_logits = _make_gate_wrap(orig_get_logits)

            # NO forward_impl wrapper — no cache operations

            orig_bi = decoder.block_init
            def block_init_td8(block_x, block_id):
                _td8_step_in_block[0] = 0
                _td8_prev_decoded_sp[0] = None
                _td8_null_mask_sp[0] = None
                _td8_null_mask_gathered[0] = None
                return orig_bi(block_x, block_id)
            decoder.block_init = block_init_td8

            orig_mf = model.forward
            def model_forward_td8(input_ids=None, *args, **kwargs):
                if input_ids is not None:
                    step = _td8_step_in_block[0]
                    is_mask = (input_ids == MASK_ID).view(-1)
                    chunk = is_mask.shape[0] // _tp_size
                    sp_start = _tp_rank * chunk
                    decoded_sp = ~is_mask[sp_start:sp_start + chunk]

                    prev_dec = _td8_prev_decoded_sp[0]
                    if (prev_dec is not None
                            and prev_dec.shape[0] == decoded_sp.shape[0]
                            and step % _REFRESH_M != 1):
                        _td8_null_mask_sp[0] = decoded_sp & prev_dec
                    else:
                        _td8_null_mask_sp[0] = None
                else:
                    _td8_null_mask_sp[0] = None

                result = orig_mf(input_ids, *args, **kwargs)

                if input_ids is not None:
                    _td8_prev_decoded_sp[0] = decoded_sp
                    _td8_step_in_block[0] += 1
                return result
            model.forward = model_forward_td8

        # ---- Triton gather/scatter kernels for TV4 ----
        @triton.jit
        def _triton_gather_rows(
            src_ptr, indices_ptr, dst_ptr,
            N_rows, D: tl.constexpr,
            BLOCK_D: tl.constexpr,
        ):
            """dst[i, :] = src[indices[i], :] for i in [0, N_rows)"""
            row_id = tl.program_id(0)
            if row_id >= N_rows:
                return
            src_row = tl.load(indices_ptr + row_id)
            col_offs = tl.arange(0, BLOCK_D)
            mask = col_offs < D
            vals = tl.load(src_ptr + src_row.to(tl.int64) * D + col_offs, mask=mask)
            tl.store(dst_ptr + row_id.to(tl.int64) * D + col_offs, vals, mask=mask)

        @triton.jit
        def _triton_scatter_rows(
            src_ptr, indices_ptr, dst_ptr,
            N_rows, D: tl.constexpr,
            BLOCK_D: tl.constexpr,
        ):
            """dst[indices[i], :] = src[i, :] for i in [0, N_rows)"""
            row_id = tl.program_id(0)
            if row_id >= N_rows:
                return
            dst_row = tl.load(indices_ptr + row_id)
            col_offs = tl.arange(0, BLOCK_D)
            mask = col_offs < D
            vals = tl.load(src_ptr + row_id.to(tl.int64) * D + col_offs, mask=mask)
            tl.store(dst_ptr + dst_row.to(tl.int64) * D + col_offs, vals, mask=mask)

        def triton_gather(src, indices, dst):
            """dst[i] = src[indices[i]] — single kernel launch"""
            N = indices.shape[0]
            D = src.shape[1]
            BLOCK_D = triton.next_power_of_2(D)
            _triton_gather_rows[(N,)](src, indices, dst, N, D, BLOCK_D)

        def triton_scatter(src, indices, dst):
            """dst[indices[i]] = src[i] — single kernel launch"""
            N = indices.shape[0]
            D = dst.shape[1]
            BLOCK_D = triton.next_power_of_2(D)
            _triton_scatter_rows[(N,)](src, indices, dst, N, D, BLOCK_D)

        # ---- Mapped fused_moe_kernel: reads A via input_map indirection ----
        @triton.jit
        def _write_zeros(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token,
                         token_mask, BLOCK_SIZE_M: tl.constexpr,
                         BLOCK_SIZE_N: tl.constexpr, compute_type: tl.constexpr):
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
            c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
            tl.store(c_ptrs, acc, mask=c_mask)

        @triton.jit
        def _fused_moe_kernel_mapped(
            a_ptr, b_ptr, c_ptr,
            input_map_ptr,
            topk_weights_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
            N, K, EM, num_valid_tokens,
            stride_am, stride_ak,
            stride_be, stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
            MUL_ROUTED_WEIGHT: tl.constexpr,
            top_k: tl.constexpr, compute_type: tl.constexpr,
        ):
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
            if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
                return
            offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
            token_mask = offs_token < num_valid_tokens

            off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
            if off_experts == -1:
                _write_zeros(c_ptr, stride_cm, stride_cn, pid_n, N,
                             offs_token, token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N,
                             compute_type)
                return

            # Read A rows: use input_map for indirection if available
            # Read A rows: always use input_map for indirection
            max_row_idx = tl.maximum(num_valid_tokens // top_k - 1, 0)
            compact_tok = tl.minimum(offs_token // top_k, max_row_idx)
            a_row_idx = tl.load(input_map_ptr + compact_tok.to(tl.int64),
                                mask=token_mask,
                                other=0).to(tl.int64)

            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (a_row_idx[:, None] * stride_am +
                              offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                        offs_bn[None, :] * stride_bn)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                a = tl.load(a_ptrs,
                            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                            other=0.0)
                b = tl.load(b_ptrs,
                            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                            other=0.0)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            if MUL_ROUTED_WEIGHT:
                moe_weight = tl.load(topk_weights_ptr + offs_token,
                                     mask=token_mask, other=0)
                accumulator = accumulator * moe_weight[:, None]
            accumulator = accumulator.to(compute_type)

            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
            c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
            tl.store(c_ptrs, accumulator, mask=c_mask)

        def invoke_mapped_kernel(A, B, C, input_map, topk_weights,
                                 sorted_token_ids, expert_ids,
                                 num_tokens_post_padded,
                                 mul_routed_weight, top_k,
                                 num_valid_tokens,
                                 config, compute_type):
            EM = sorted_token_ids.size(0)
            grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
                B.size(1), META['BLOCK_SIZE_N']), )
            # C may be 3D (M, top_k, N): flatten to 2D for correct strides
            # sorted_token_ids indexes flat M*top_k dimension
            C_flat = C.view(-1, C.size(-1)) if C.ndim == 3 else C
            _fused_moe_kernel_mapped[grid](
                A, B, C_flat,
                input_map,
                topk_weights,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                B.size(1), A.size(1), EM, num_valid_tokens,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(2), B.stride(1),
                C_flat.stride(0), C_flat.stride(1),
                MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k,
                compute_type=compute_type,
                **config,
            )

        # ---- TV4m: mapped kernel (read FULL A, compact cache) ----
        _tv4m_moe_cache = {}
        _tv4m_null_mask_sp = [None]
        _tv4m_compute_indices = [None]
        _tv4m_null_indices_sp = [None]
        _tv4m_prev_decoded_sp = [None]
        _tv4m_step_in_block = [0]
        _tv4m_buf_cached = [None]
        _tv4m_bufs = {}  # pre-allocated buffers, keyed by N_compute

        def setup_tv4m(ctrl_ref, label: str):
            """TV4m: mapped kernel reads FULL A via input_map, compact intermediate cache."""
            from vllm.distributed import (
                get_ep_group,
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            from vllm.forward_context import get_forward_context, set_forward_context
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE
            from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
            from vllm import _custom_ops as vllm_ops

            _restore_expert_attrs()
            setup_bsp_g_attn_rs(ctrl_ref, label)
            _tv4m_moe_cache.clear()
            _tv4m_step_in_block[0] = 0
            _tv4m_prev_decoded_sp[0] = None
            _tv4m_null_mask_sp[0] = None
            _tv4m_compute_indices[0] = None
            _tv4m_null_indices_sp[0] = None
            _tv4m_bufs.clear()
            _REFRESH_M = 5
            _tp_size = get_tensor_model_parallel_world_size()
            _tp_rank = get_tensor_model_parallel_rank()
            _ep_group = get_ep_group()
            _ep_world = _ep_group.world_size

            # Pre-allocate fixed-size buffers
            _N_total = _ep_world * (model.config.hidden_size * 2 // _tp_size)  # typical gathered size
            # Will be sized on first use
            _hidden_dim = model.config.hidden_size
            _tv4m_bufs['hidden_dim'] = _hidden_dim
            _tv4m_bufs['buf_out_g'] = None  # allocated on first forward (need N_total from dispatch)
            _tv4m_bufs['config'] = None  # cached config from try_get_optimal_moe_config

            for i, blk in enumerate(moe_blocks):
                layer_id = i
                experts_obj = blk.experts

                def _make_tv4m_impl(orig_impl, lid, exp_obj):
                    def tv4m_forward_impl(self, hidden_states, router_logits):
                        compute_indices = _tv4m_compute_indices[0]

                        if compute_indices is None:
                            y = orig_impl(hidden_states, router_logits)
                            if isinstance(y, tuple):
                                y = y[1]
                            _tv4m_moe_cache[lid] = y.detach().clone()
                            return y

                        # --- SPARSE MAPPED PATH ---
                        ctx = get_forward_context()
                        sp_ctx = (
                            ctx.dp_metadata.sp_local_sizes(self.sp_size)
                            if ctx.dp_metadata
                            else nullcontext()
                        )

                        with sp_ctx:
                            hs_g, rl_g = _ep_group.dispatch(
                                hidden_states, router_logits,
                                self.is_sequence_parallel)

                            N_total = hs_g.shape[0]

                            topk_weights, topk_ids, _ = FusedMoE.select_experts(
                                hidden_states=hs_g,
                                router_logits=rl_g,
                                use_grouped_topk=self.use_grouped_topk,
                                top_k=self.top_k,
                                renormalize=self.renormalize,
                                topk_group=self.topk_group,
                                num_expert_group=self.num_expert_group,
                                custom_routing_function=self.custom_routing_function,
                                scoring_func=self.scoring_func,
                                routed_scaling_factor=self.routed_scaling_factor,
                                e_score_correction_bias=self.e_score_correction_bias,
                                indices_type=getattr(self, 'topk_indices_dtype', None),
                            )

                            # Extract compact topk_ids/weights (small tensors)
                            c_ids = topk_ids[compute_indices]
                            c_wts = topk_weights[compute_indices]
                            N_compute = compute_indices.shape[0]
                            top_k_num = c_ids.shape[1]
                            hidden_dim = hs_g.shape[1]
                            N = exp_obj.w13_weight.shape[1]  # intermediate size * 2 (output dim)
                            K = hidden_dim

                            # --- Use pre-allocated buffers ---
                            bufs = _tv4m_bufs
                            n_valid = N_compute * top_k_num
                            n_pairs = n_valid

                            # Config: cache once
                            if bufs.get('config') is None:
                                from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
                                bufs['config'] = try_get_optimal_moe_config(
                                    exp_obj.w13_weight.size(),
                                    exp_obj.w2_weight.size(),
                                    top_k_num,
                                    hs_g.dtype,
                                    N_compute,
                                    block_shape=None,
                                )
                            config = bufs['config']

                            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                                moe_align_block_size(
                                    c_ids, config['BLOCK_SIZE_M'],
                                    self.global_num_experts, self.expert_map))

                            # input_map: compact token → FULL A row
                            input_map = compute_indices  # [N_compute], int64

                            # Pre-allocate or reuse compact caches
                            buf_key = n_pairs
                            if bufs.get('_last_n_pairs') != buf_key:
                                bufs['_last_n_pairs'] = buf_key
                                bufs['cache13'] = torch.empty(
                                    n_pairs * max(N, K),
                                    device=hs_g.device, dtype=hs_g.dtype)
                                bufs['cache2'] = torch.empty(
                                    n_pairs, N // 2,
                                    device=hs_g.device, dtype=hs_g.dtype)
                                bufs['compact_output'] = torch.empty(
                                    N_compute, K,
                                    device=hs_g.device, dtype=hs_g.dtype)
                                bufs['identity_map_2nd'] = torch.arange(
                                    n_pairs, device=hs_g.device, dtype=torch.int64)

                            # Reuse fixed-size buf_out_g
                            if bufs.get('buf_out_g') is None or bufs['buf_out_g'].shape[0] != N_total:
                                bufs['buf_out_g'] = torch.empty(
                                    N_total, K, device=hs_g.device, dtype=hs_g.dtype)

                            cache13 = bufs['cache13']
                            intermediate_cache1 = cache13[:n_pairs * N].view(
                                N_compute, top_k_num, N)
                            intermediate_cache3 = cache13[:n_pairs * K].view(
                                N_compute, top_k_num, K)
                            intermediate_cache2 = bufs['cache2']

                            ct = tl.bfloat16

                            # First GEMM: read FULL A via input_map → compact cache1
                            invoke_mapped_kernel(
                                hs_g, exp_obj.w13_weight, intermediate_cache1,
                                input_map, c_wts.view(-1),
                                sorted_token_ids, expert_ids, num_tokens_post_padded,
                                self.apply_router_weight_on_input, top_k_num,
                                n_valid, config, ct)

                            # Activation (compact)
                            torch.ops._C.silu_and_mul(
                                intermediate_cache2,
                                intermediate_cache1.view(-1, N))

                            # Second GEMM: compact cache2 → compact cache3 (identity map)
                            invoke_mapped_kernel(
                                intermediate_cache2, exp_obj.w2_weight,
                                intermediate_cache3,
                                bufs['identity_map_2nd'], c_wts,
                                sorted_token_ids, expert_ids, num_tokens_post_padded,
                                not self.apply_router_weight_on_input, 1,
                                n_valid, config, ct)

                            # moe_sum (compact)
                            compact_output = bufs['compact_output']
                            vllm_ops.moe_sum(
                                intermediate_cache3.view(N_compute, top_k_num, K),
                                compact_output)

                            # Scatter back to FULL (zero + copy)
                            buf_out_g = bufs['buf_out_g']
                            buf_out_g.zero_()
                            buf_out_g.index_copy_(0, compute_indices, compact_output)

                            y_sp = _ep_group.combine(
                                buf_out_g, self.is_sequence_parallel)

                        null_idx_sp = _tv4m_null_indices_sp[0]
                        if (null_idx_sp is not None
                                and lid in _tv4m_moe_cache
                                and _tv4m_moe_cache[lid].shape[0] == y_sp.shape[0]):
                            buf_cached = _tv4m_buf_cached[0]
                            torch.index_select(_tv4m_moe_cache[lid], 0,
                                               null_idx_sp, out=buf_cached)
                            y_sp.index_copy_(0, null_idx_sp, buf_cached)

                        return y_sp
                    return types.MethodType(tv4m_forward_impl, blk.experts)
                blk.experts.forward_impl = _make_tv4m_impl(
                    blk.experts.forward_impl, layer_id, experts_obj)

            orig_bi = decoder.block_init
            def block_init_tv4m(block_x, block_id):
                _tv4m_moe_cache.clear()
                _tv4m_step_in_block[0] = 0
                _tv4m_prev_decoded_sp[0] = None
                _tv4m_null_mask_sp[0] = None
                _tv4m_compute_indices[0] = None
                _tv4m_null_indices_sp[0] = None
                return orig_bi(block_x, block_id)
            decoder.block_init = block_init_tv4m

            orig_mf = model.forward
            def model_forward_tv4m(input_ids=None, *args, **kwargs):
                if input_ids is not None:
                    step = _tv4m_step_in_block[0]
                    is_mask = (input_ids == MASK_ID).view(-1)
                    chunk = is_mask.shape[0] // _tp_size
                    sp_start = _tp_rank * chunk
                    decoded_sp = ~is_mask[sp_start:sp_start + chunk]

                    prev_dec = _tv4m_prev_decoded_sp[0]
                    if (prev_dec is not None
                            and prev_dec.shape[0] == decoded_sp.shape[0]
                            and step % _REFRESH_M != 1):
                        null_mask_sp = decoded_sp & prev_dec
                    else:
                        null_mask_sp = None

                    if null_mask_sp is not None and null_mask_sp.any():
                        null_byte = null_mask_sp.to(torch.uint8)
                        gathered = [torch.empty(chunk, dtype=torch.uint8,
                                                device=null_byte.device)
                                    for _ in range(_ep_world)]
                        dist.all_gather(gathered, null_byte,
                                        group=_ep_group.device_group)
                        null_gathered = torch.cat(gathered).bool()
                        _tv4m_compute_indices[0] = (~null_gathered).nonzero(
                            as_tuple=True)[0]
                        _tv4m_null_mask_sp[0] = null_mask_sp
                        _tv4m_null_indices_sp[0] = null_mask_sp.nonzero(
                            as_tuple=True)[0]

                        _hidden_dim = model.config.hidden_size
                        _N_null_sp = _tv4m_null_indices_sp[0].shape[0]
                        _dev = _tv4m_compute_indices[0].device
                        _tv4m_buf_cached[0] = torch.empty(
                            _N_null_sp, _hidden_dim, device=_dev,
                            dtype=torch.bfloat16)
                    else:
                        _tv4m_compute_indices[0] = None
                        _tv4m_null_mask_sp[0] = None
                        _tv4m_null_indices_sp[0] = None
                else:
                    _tv4m_compute_indices[0] = None
                    _tv4m_null_mask_sp[0] = None
                    _tv4m_null_indices_sp[0] = None

                result = orig_mf(input_ids, *args, **kwargs)

                if input_ids is not None:
                    _tv4m_prev_decoded_sp[0] = decoded_sp
                    _tv4m_step_in_block[0] += 1
                return result
            model.forward = model_forward_tv4m

        # ---- TV4m-v2: full-layout moe_align + token_remap kernel ----
        # Eliminates c_ids/c_wts Python gathers by using full-space sorted_token_ids
        # and a token_remap (full→compact) for output writes.

        @triton.jit
        def _fused_moe_kernel_v2(
            a_ptr, b_ptr, c_ptr,
            token_remap_ptr,
            topk_weights_ptr,
            sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
            N, K, EM, num_valid_tokens,
            stride_am, stride_ak,
            stride_be, stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
            BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
            MUL_ROUTED_WEIGHT: tl.constexpr,
            top_k: tl.constexpr, compute_type: tl.constexpr,
            USE_REMAP_WRITE: tl.constexpr,
            USE_REMAP_READ: tl.constexpr,
        ):
            """
            Fused MoE kernel with token_remap for compact output.
            - USE_REMAP_WRITE: remap offs_token → compact index for C write
            - USE_REMAP_READ: remap offs_token → compact index for A read
              (used by 2nd GEMM where A=compact cache)
            When both are False, behaves like standard fused_moe_kernel.
            """
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
            num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
            if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
                return
            offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
            token_mask = offs_token < num_valid_tokens
            # Clamp padding sentinels to prevent OOB on token_remap and topk_weights
            offs_token = tl.where(token_mask, offs_token, num_valid_tokens - 1)

            off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
            if off_experts == -1:
                # Expert not on this rank — write zeros to compact C
                if USE_REMAP_WRITE:
                    max_tok = num_valid_tokens // top_k - 1
                    full_token = tl.minimum(offs_token // top_k, max_tok)
                    compact_token = tl.load(token_remap_ptr + full_token.to(tl.int64),
                                            mask=token_mask, other=0)
                    compact_pair = compact_token * top_k + offs_token % top_k
                    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    c_ptrs = c_ptr + stride_cm * compact_pair[:, None] + stride_cn * offs_cn[None, :]
                    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
                    tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask)
                else:
                    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
                    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
                    tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask)
                return

            # A read index
            # Clamp full_token to prevent padding sentinel OOB on token_remap
            max_token_idx = num_valid_tokens // top_k - 1
            if USE_REMAP_READ:
                full_token_r = tl.minimum(offs_token // top_k, max_token_idx)
                compact_token_r = tl.load(token_remap_ptr + full_token_r.to(tl.int64),
                                          mask=token_mask, other=0)
                a_row_idx = (compact_token_r * top_k + offs_token % top_k).to(tl.int64)
            else:
                a_row_idx = (offs_token // top_k).to(tl.int64)

            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (a_row_idx[:, None] * stride_am +
                              offs_k[None, :] * stride_ak)
            b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                        offs_bn[None, :] * stride_bn)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                a = tl.load(a_ptrs,
                            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                            other=0.0)
                b = tl.load(b_ptrs,
                            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                            other=0.0)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            if MUL_ROUTED_WEIGHT:
                moe_weight = tl.load(topk_weights_ptr + offs_token,
                                     mask=token_mask, other=0)
                accumulator = accumulator * moe_weight[:, None]
            accumulator = accumulator.to(compute_type)

            # C write index
            if USE_REMAP_WRITE:
                full_token_w = tl.minimum(offs_token // top_k, max_token_idx)
                compact_token_w = tl.load(token_remap_ptr + full_token_w.to(tl.int64),
                                          mask=token_mask, other=0)
                write_idx = compact_token_w * top_k + offs_token % top_k
            else:
                write_idx = offs_token

            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * write_idx[:, None] + stride_cn * offs_cn[None, :]
            c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
            tl.store(c_ptrs, accumulator, mask=c_mask)

        def invoke_v2_kernel(A, B, C, token_remap, topk_weights,
                             sorted_token_ids, expert_ids,
                             num_tokens_post_padded,
                             mul_routed_weight, top_k,
                             num_valid_tokens, config,
                             use_remap_write, use_remap_read):
            EM = sorted_token_ids.size(0)
            grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
                B.size(1), META['BLOCK_SIZE_N']), )
            C_flat = C.view(-1, C.size(-1)) if C.ndim == 3 else C
            _fused_moe_kernel_v2[grid](
                A, B, C_flat,
                token_remap,
                topk_weights,
                sorted_token_ids, expert_ids, num_tokens_post_padded,
                B.size(1), A.size(1), EM, num_valid_tokens,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(2), B.stride(1),
                C_flat.stride(0), C_flat.stride(1),
                MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k,
                compute_type=tl.bfloat16,
                USE_REMAP_WRITE=use_remap_write,
                USE_REMAP_READ=use_remap_read,
                **config,
            )

        # ---- TV4m-v2 state ----
        _tv4mv2_moe_cache = {}
        _tv4mv2_null_mask_sp = [None]
        _tv4mv2_compute_indices = [None]
        _tv4mv2_null_indices_sp = [None]
        _tv4mv2_prev_decoded_sp = [None]
        _tv4mv2_step_in_block = [0]
        _tv4mv2_buf_cached = [None]
        _tv4mv2_bufs = {}

        def setup_tv4m_v2(ctrl_ref, label: str):
            """TV4m-v2: full-layout moe_align + token_remap. No Python gathers."""
            from vllm.distributed import (
                get_ep_group,
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            from vllm.forward_context import get_forward_context, set_forward_context
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE
            from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
            from vllm import _custom_ops as vllm_ops

            _restore_expert_attrs()
            setup_bsp_g_attn_rs(ctrl_ref, label)
            _tv4mv2_moe_cache.clear()
            _tv4mv2_step_in_block[0] = 0
            _tv4mv2_prev_decoded_sp[0] = None
            _tv4mv2_null_mask_sp[0] = None
            _tv4mv2_compute_indices[0] = None
            _tv4mv2_null_indices_sp[0] = None
            _tv4mv2_bufs.clear()
            _REFRESH_M = 5
            _tp_size = get_tensor_model_parallel_world_size()
            _tp_rank = get_tensor_model_parallel_rank()
            _ep_group = get_ep_group()
            _ep_world = _ep_group.world_size

            for i, blk in enumerate(moe_blocks):
                layer_id = i
                experts_obj = blk.experts

                def _make_tv4mv2_impl(orig_impl, lid, exp_obj):
                    def tv4mv2_forward_impl(self, hidden_states, router_logits):
                        compute_indices = _tv4mv2_compute_indices[0]

                        if compute_indices is None:
                            y = orig_impl(hidden_states, router_logits)
                            if isinstance(y, tuple):
                                y = y[1]
                            _tv4mv2_moe_cache[lid] = y.detach().clone()
                            return y

                        # --- SPARSE PATH (v2: full-layout moe_align) ---
                        ctx = get_forward_context()
                        sp_ctx = (
                            ctx.dp_metadata.sp_local_sizes(self.sp_size)
                            if ctx.dp_metadata
                            else nullcontext()
                        )

                        with sp_ctx:
                            hs_g, rl_g = _ep_group.dispatch(
                                hidden_states, router_logits,
                                self.is_sequence_parallel)

                            N_total = hs_g.shape[0]
                            top_k_num = self.top_k
                            hidden_dim = hs_g.shape[1]
                            N = exp_obj.w13_weight.shape[1]
                            K = hidden_dim
                            N_compute = compute_indices.shape[0]

                            topk_weights, topk_ids, _ = FusedMoE.select_experts(
                                hidden_states=hs_g,
                                router_logits=rl_g,
                                use_grouped_topk=self.use_grouped_topk,
                                top_k=self.top_k,
                                renormalize=self.renormalize,
                                topk_group=self.topk_group,
                                num_expert_group=self.num_expert_group,
                                custom_routing_function=self.custom_routing_function,
                                scoring_func=self.scoring_func,
                                routed_scaling_factor=self.routed_scaling_factor,
                                e_score_correction_bias=self.e_score_correction_bias,
                                indices_type=getattr(self, 'topk_indices_dtype', None),
                            )

                            # Mark null tokens with invalid expert (skip in moe_align)
                            null_mask_gathered = _tv4mv2_bufs.get('null_mask_gathered')
                            topk_ids.masked_fill_(
                                null_mask_gathered.unsqueeze(1), self.global_num_experts)

                            # --- Pre-allocated buffers ---
                            bufs = _tv4mv2_bufs
                            n_valid = N_total * top_k_num  # full space num_valid

                            if bufs.get('config') is None:
                                from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
                                bufs['config'] = try_get_optimal_moe_config(
                                    exp_obj.w13_weight.size(),
                                    exp_obj.w2_weight.size(),
                                    top_k_num,
                                    hs_g.dtype,
                                    N_total,
                                    block_shape=None,
                                )
                            config = bufs['config']

                            # moe_align on FULL topk_ids (null tokens skipped naturally)
                            sorted_token_ids, expert_ids, num_tokens_post_padded = (
                                moe_align_block_size(
                                    topk_ids, config['BLOCK_SIZE_M'],
                                    self.global_num_experts, self.expert_map))

                            # token_remap: full_token → compact_token (pre-computed per forward)
                            token_remap = bufs['token_remap']

                            # Allocate compact caches (reuse if size unchanged)
                            n_compact_pairs = N_compute * top_k_num
                            if bufs.get('_n_compact') != n_compact_pairs:
                                bufs['_n_compact'] = n_compact_pairs
                                bufs['cache13'] = torch.empty(
                                    n_compact_pairs * max(N, K),
                                    device=hs_g.device, dtype=hs_g.dtype)
                                bufs['cache2'] = torch.empty(
                                    n_compact_pairs, N // 2,
                                    device=hs_g.device, dtype=hs_g.dtype)
                                bufs['compact_output'] = torch.empty(
                                    N_compute, K, device=hs_g.device, dtype=hs_g.dtype)
                            if bufs.get('buf_out_g') is None or bufs['buf_out_g'].shape[0] != N_total:
                                bufs['buf_out_g'] = torch.empty(
                                    N_total, K, device=hs_g.device, dtype=hs_g.dtype)

                            cache13 = bufs['cache13']
                            intermediate_cache1 = cache13[:n_compact_pairs * N].view(
                                N_compute, top_k_num, N)
                            intermediate_cache3 = cache13[:n_compact_pairs * K].view(
                                N_compute, top_k_num, K)
                            intermediate_cache2 = bufs['cache2']

                            ct = tl.bfloat16

                            # First GEMM: read A directly from hs_g, write compact C via remap
                            invoke_v2_kernel(
                                hs_g, exp_obj.w13_weight, intermediate_cache1,
                                token_remap, topk_weights.view(-1),
                                sorted_token_ids, expert_ids, num_tokens_post_padded,
                                self.apply_router_weight_on_input, top_k_num,
                                n_valid, config,
                                use_remap_write=True, use_remap_read=False)

                            # Activation (compact)
                            torch.ops._C.silu_and_mul(
                                intermediate_cache2,
                                intermediate_cache1.view(-1, N))

                            # Second GEMM: read compact cache2 via remap, write compact cache3
                            invoke_v2_kernel(
                                intermediate_cache2, exp_obj.w2_weight,
                                intermediate_cache3,
                                token_remap, topk_weights.view(-1),
                                sorted_token_ids, expert_ids, num_tokens_post_padded,
                                not self.apply_router_weight_on_input, top_k_num,
                                n_valid, config,
                                use_remap_write=True, use_remap_read=True)

                            # moe_sum (compact)
                            compact_output = bufs['compact_output']
                            vllm_ops.moe_sum(
                                intermediate_cache3.view(N_compute, top_k_num, K),
                                compact_output)

                            # Scatter back to FULL
                            buf_out_g = bufs['buf_out_g']
                            buf_out_g.zero_()
                            buf_out_g.index_copy_(0, compute_indices, compact_output)

                            y_sp = _ep_group.combine(
                                buf_out_g, self.is_sequence_parallel)

                        null_idx_sp = _tv4mv2_null_indices_sp[0]
                        if (null_idx_sp is not None
                                and lid in _tv4mv2_moe_cache
                                and _tv4mv2_moe_cache[lid].shape[0] == y_sp.shape[0]):
                            buf_cached = _tv4mv2_buf_cached[0]
                            torch.index_select(_tv4mv2_moe_cache[lid], 0,
                                               null_idx_sp, out=buf_cached)
                            y_sp.index_copy_(0, null_idx_sp, buf_cached)

                        return y_sp
                    return types.MethodType(tv4mv2_forward_impl, blk.experts)
                blk.experts.forward_impl = _make_tv4mv2_impl(
                    blk.experts.forward_impl, layer_id, experts_obj)

            orig_bi = decoder.block_init
            def block_init_tv4mv2(block_x, block_id):
                _tv4mv2_moe_cache.clear()
                _tv4mv2_step_in_block[0] = 0
                _tv4mv2_prev_decoded_sp[0] = None
                _tv4mv2_null_mask_sp[0] = None
                _tv4mv2_compute_indices[0] = None
                _tv4mv2_null_indices_sp[0] = None
                return orig_bi(block_x, block_id)
            decoder.block_init = block_init_tv4mv2

            orig_mf = model.forward
            def model_forward_tv4mv2(input_ids=None, *args, **kwargs):
                if input_ids is not None:
                    step = _tv4mv2_step_in_block[0]
                    is_mask = (input_ids == MASK_ID).view(-1)
                    chunk = is_mask.shape[0] // _tp_size
                    sp_start = _tp_rank * chunk
                    decoded_sp = ~is_mask[sp_start:sp_start + chunk]

                    prev_dec = _tv4mv2_prev_decoded_sp[0]
                    if (prev_dec is not None
                            and prev_dec.shape[0] == decoded_sp.shape[0]
                            and step % _REFRESH_M != 1):
                        null_mask_sp = decoded_sp & prev_dec
                    else:
                        null_mask_sp = None

                    if null_mask_sp is not None and null_mask_sp.any():
                        null_byte = null_mask_sp.to(torch.uint8)
                        gathered = [torch.empty(chunk, dtype=torch.uint8,
                                                device=null_byte.device)
                                    for _ in range(_ep_world)]
                        dist.all_gather(gathered, null_byte,
                                        group=_ep_group.device_group)
                        null_gathered = torch.cat(gathered).bool()
                        compute_indices = (~null_gathered).nonzero(
                            as_tuple=True)[0]
                        N_total = null_gathered.shape[0]
                        N_compute = compute_indices.shape[0]

                        _tv4mv2_compute_indices[0] = compute_indices
                        _tv4mv2_null_mask_sp[0] = null_mask_sp
                        _tv4mv2_null_indices_sp[0] = null_mask_sp.nonzero(
                            as_tuple=True)[0]

                        # Pre-compute token_remap (full→compact), shared by all 19 layers
                        token_remap = torch.zeros(N_total, dtype=torch.int32,
                                                  device=compute_indices.device)
                        token_remap[compute_indices] = torch.arange(
                            N_compute, device=compute_indices.device, dtype=torch.int32)
                        _tv4mv2_bufs['token_remap'] = token_remap
                        _tv4mv2_bufs['null_mask_gathered'] = null_gathered

                        _hidden_dim = model.config.hidden_size
                        _N_null_sp = _tv4mv2_null_indices_sp[0].shape[0]
                        _dev = compute_indices.device
                        _tv4mv2_buf_cached[0] = torch.empty(
                            _N_null_sp, _hidden_dim, device=_dev,
                            dtype=torch.bfloat16)
                    else:
                        _tv4mv2_compute_indices[0] = None
                        _tv4mv2_null_mask_sp[0] = None
                        _tv4mv2_null_indices_sp[0] = None
                else:
                    _tv4mv2_compute_indices[0] = None
                    _tv4mv2_null_mask_sp[0] = None
                    _tv4mv2_null_indices_sp[0] = None

                result = orig_mf(input_ids, *args, **kwargs)

                if input_ids is not None:
                    _tv4mv2_prev_decoded_sp[0] = decoded_sp
                    _tv4mv2_step_in_block[0] += 1
                return result
            model.forward = model_forward_tv4mv2

        # ---- TV6: compact dispatch/combine + vllm native kernel ----
        # Based on TV4m architecture but with compact AllGatherV/ReduceScatterV.
        # No custom Triton kernel — uses vllm's quant_method.apply directly.
        _tv6_moe_cache = {}
        _tv6_compute_mask_sp = [None]
        _tv6_null_mask_sp = [None]
        _tv6_null_indices_sp = [None]
        _tv6_compute_indices_sp = [None]
        _tv6_compact_sizes = [None]
        _tv6_prev_decoded_sp = [None]
        _tv6_step_in_block = [0]
        _tv6_buf_cached = [None]
        _tv6_bufs = {}

        def setup_tv6(ctrl_ref, label: str):
            """TV6: compact dispatch/combine + direct vllm kernel calls."""
            from vllm.distributed import (
                get_ep_group,
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            from vllm.forward_context import get_forward_context
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE
            from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                invoke_fused_moe_kernel, try_get_optimal_moe_config,
            )
            from vllm import _custom_ops as vllm_ops

            _restore_expert_attrs()
            setup_bsp_g_attn_rs(ctrl_ref, label)
            _tv6_moe_cache.clear()
            _tv6_step_in_block[0] = 0
            _tv6_prev_decoded_sp[0] = None
            _tv6_compute_mask_sp[0] = None
            _tv6_null_mask_sp[0] = None
            _tv6_null_indices_sp[0] = None
            _tv6_compute_indices_sp[0] = None
            _tv6_compact_sizes[0] = None
            _tv6_bufs.clear()
            _REFRESH_M = 5
            _tp_size = get_tensor_model_parallel_world_size()
            _tp_rank = get_tensor_model_parallel_rank()
            _ep_group = get_ep_group()
            _ep_world = _ep_group.world_size

            for i, blk in enumerate(moe_blocks):
                layer_id = i
                experts_obj = blk.experts

                def _make_tv6_impl(orig_impl, lid, exp_obj):
                    def tv6_forward_impl(self, hidden_states, router_logits):
                        compute_mask_sp = _tv6_compute_mask_sp[0]

                        if compute_mask_sp is None:
                            y = orig_impl(hidden_states, router_logits)
                            if isinstance(y, tuple):
                                y = y[1]
                            _tv6_moe_cache[lid] = y.detach().clone()
                            return y

                        # --- TV6 SPARSE PATH (direct API) ---
                        ctx = get_forward_context()
                        N_sp = hidden_states.shape[0]
                        hidden_dim = hidden_states.shape[1]

                        # ① Extract compact tokens (SP local, integer indexing to avoid sync)
                        compute_idx_sp = _tv6_compute_indices_sp[0]
                        hs_compute_sp = hidden_states[compute_idx_sp]
                        rl_compute_sp = router_logits[compute_idx_sp]

                        # ② Set compact sizes + dispatch
                        ctx.dp_metadata.local_sizes = _tv6_compact_sizes[0]
                        hs_g, rl_g = _ep_group.dispatch(
                            hs_compute_sp, rl_compute_sp,
                            self.is_sequence_parallel)

                        N_ct = hs_g.shape[0]  # N_compute_total

                        # ③ Routing (EB on compact gathered data)
                        topk_weights, topk_ids, _ = FusedMoE.select_experts(
                            hidden_states=hs_g,
                            router_logits=rl_g,
                            use_grouped_topk=self.use_grouped_topk,
                            top_k=self.top_k,
                            renormalize=self.renormalize,
                            topk_group=self.topk_group,
                            num_expert_group=self.num_expert_group,
                            custom_routing_function=self.custom_routing_function,
                            scoring_func=self.scoring_func,
                            routed_scaling_factor=self.routed_scaling_factor,
                            e_score_correction_bias=self.e_score_correction_bias,
                            indices_type=getattr(self, 'topk_indices_dtype', None),
                        )

                        top_k_num = topk_ids.shape[1]
                        N = exp_obj.w13_weight.shape[1]
                        K = hidden_dim
                        n_pairs = N_ct * top_k_num

                        # ④ Pre-allocate / reuse buffers
                        bufs = _tv6_bufs
                        if bufs.get('config') is None:
                            bufs['config'] = try_get_optimal_moe_config(
                                exp_obj.w13_weight.size(),
                                exp_obj.w2_weight.size(),
                                top_k_num, hs_g.dtype, N_ct,
                                block_shape=None)
                        config = bufs['config']

                        if bufs.get('_n_pairs') != n_pairs:
                            bufs['_n_pairs'] = n_pairs
                            bufs['cache13'] = torch.empty(
                                n_pairs * max(N, K),
                                device=hs_g.device, dtype=hs_g.dtype)
                            bufs['cache2'] = torch.empty(
                                n_pairs, N // 2,
                                device=hs_g.device, dtype=hs_g.dtype)
                            bufs['compact_output'] = torch.empty(
                                N_ct, K, device=hs_g.device, dtype=hs_g.dtype)

                        cache13 = bufs['cache13']
                        cache1 = cache13[:n_pairs * N].view(N_ct, top_k_num, N)
                        cache3 = cache13[:n_pairs * K].view(N_ct, top_k_num, K)
                        cache2 = bufs['cache2']

                        # ⑤ moe_align
                        sorted_token_ids, expert_ids, ntp = moe_align_block_size(
                            topk_ids, config['BLOCK_SIZE_M'],
                            self.global_num_experts, self.expert_map)

                        ct = tl.bfloat16

                        # ⑥ 1st GEMM (standard vllm kernel, A=compact hs_g)
                        invoke_fused_moe_kernel(
                            hs_g, exp_obj.w13_weight, cache1,
                            None, None, None, topk_weights,
                            sorted_token_ids, expert_ids, ntp,
                            self.apply_router_weight_on_input, top_k_num,
                            config, compute_type=ct,
                            use_fp8_w8a8=False, use_int8_w8a8=False,
                            use_int8_w8a16=False, use_int4_w4a16=False,
                            per_channel_quant=False)

                        # ⑦ silu
                        torch.ops._C.silu_and_mul(cache2, cache1.view(-1, N))

                        # ⑧ 2nd GEMM
                        invoke_fused_moe_kernel(
                            cache2, exp_obj.w2_weight, cache3,
                            None, None, None, topk_weights,
                            sorted_token_ids, expert_ids, ntp,
                            not self.apply_router_weight_on_input, 1,
                            config, compute_type=ct,
                            use_fp8_w8a8=False, use_int8_w8a8=False,
                            use_int8_w8a16=False, use_int4_w4a16=False,
                            per_channel_quant=False)

                        # ⑨ moe_sum
                        compact_output = bufs['compact_output']
                        vllm_ops.moe_sum(
                            cache3.view(N_ct, top_k_num, K),
                            compact_output)

                        # ⑩ Compact combine
                        y_compute_sp = _ep_group.combine(
                            compact_output, self.is_sequence_parallel)

                        # ⑪ Restore local_sizes
                        ctx.dp_metadata.local_sizes = None

                        # ⑫ In-place cache update (null positions retained, compute overwritten)
                        if lid in _tv6_moe_cache and _tv6_moe_cache[lid].shape[0] == N_sp:
                            y_sp = _tv6_moe_cache[lid]
                        else:
                            y_sp = torch.zeros(N_sp, hidden_dim,
                                               device=hidden_states.device,
                                               dtype=hidden_states.dtype)
                            _tv6_moe_cache[lid] = y_sp

                        compute_idx_sp = _tv6_compute_indices_sp[0]
                        y_sp.index_copy_(0, compute_idx_sp, y_compute_sp)

                        return y_sp

                    return types.MethodType(tv6_forward_impl, blk.experts)
                blk.experts.forward_impl = _make_tv6_impl(
                    blk.experts.forward_impl, layer_id, experts_obj)

            orig_bi = decoder.block_init
            def block_init_tv6(block_x, block_id):
                _tv6_moe_cache.clear()
                _tv6_step_in_block[0] = 0
                _tv6_prev_decoded_sp[0] = None
                _tv6_compute_mask_sp[0] = None
                _tv6_null_mask_sp[0] = None
                _tv6_null_indices_sp[0] = None
                _tv6_compute_indices_sp[0] = None
                _tv6_compact_sizes[0] = None
                return orig_bi(block_x, block_id)
            decoder.block_init = block_init_tv6

            orig_mf = model.forward
            def model_forward_tv6(input_ids=None, *args, **kwargs):
                if input_ids is not None:
                    step = _tv6_step_in_block[0]
                    is_mask = (input_ids == MASK_ID).view(-1)
                    chunk = is_mask.shape[0] // _tp_size
                    sp_start = _tp_rank * chunk
                    decoded_sp = ~is_mask[sp_start:sp_start + chunk]

                    prev_dec = _tv6_prev_decoded_sp[0]
                    if (prev_dec is not None
                            and prev_dec.shape[0] == decoded_sp.shape[0]
                            and step % _REFRESH_M != 1):
                        null_mask_sp = decoded_sp & prev_dec
                    else:
                        null_mask_sp = None

                    if null_mask_sp is not None and null_mask_sp.any():
                        compute_mask_sp = ~null_mask_sp
                        _tv6_compute_mask_sp[0] = compute_mask_sp
                        _tv6_null_mask_sp[0] = null_mask_sp
                        _tv6_compute_indices_sp[0] = compute_mask_sp.nonzero(
                            as_tuple=True)[0]
                        _tv6_null_indices_sp[0] = null_mask_sp.nonzero(
                            as_tuple=True)[0]

                        # Compute compact_sizes: per-rank N_compute_sp
                        n_compute_sp = _tv6_compute_indices_sp[0].shape[0]
                        n_tensor = torch.tensor(
                            [n_compute_sp], device=decoded_sp.device, dtype=torch.int64)
                        gathered_counts = [torch.empty(1, device=decoded_sp.device,
                                                       dtype=torch.int64)
                                           for _ in range(_ep_world)]
                        dist.all_gather(gathered_counts, n_tensor,
                                        group=_ep_group.device_group)
                        _tv6_compact_sizes[0] = torch.cat(gathered_counts).tolist()

                        # Pre-allocate cache merge buffer
                        _hidden_dim = model.config.hidden_size
                        _N_null_sp = _tv6_null_indices_sp[0].shape[0]
                        _tv6_buf_cached[0] = torch.empty(
                            _N_null_sp, _hidden_dim, device=decoded_sp.device,
                            dtype=torch.bfloat16)
                    else:
                        _tv6_compute_mask_sp[0] = None
                        _tv6_null_mask_sp[0] = None
                        _tv6_compute_indices_sp[0] = None
                        _tv6_null_indices_sp[0] = None
                        _tv6_compact_sizes[0] = None
                else:
                    _tv6_compute_mask_sp[0] = None
                    _tv6_null_mask_sp[0] = None
                    _tv6_compute_indices_sp[0] = None
                    _tv6_null_indices_sp[0] = None
                    _tv6_compact_sizes[0] = None

                result = orig_mf(input_ids, *args, **kwargs)

                if input_ids is not None:
                    _tv6_prev_decoded_sp[0] = decoded_sp
                    _tv6_step_in_block[0] += 1
                return result
            model.forward = model_forward_tv6

        # ---- TV4: sparse kernel (no null expert, extract compute tokens) ----
        _tv4_moe_cache = {}
        _tv4_null_mask_sp = [None]
        _tv4_compute_indices = [None]   # [N_compute] int64, gathered space
        _tv4_null_indices_sp = [None]   # [N_null_sp] int64, SP space (for cache merge)
        _tv4_prev_decoded_sp = [None]
        _tv4_step_in_block = [0]
        # Pre-allocated buffers (set per-forward in model_forward_tv4)
        _tv4_buf_hs = [None]      # [N_compute, hidden_dim]
        _tv4_buf_ids = [None]     # [N_compute, K]
        _tv4_buf_wts = [None]     # [N_compute, K]
        _tv4_buf_out_g = [None]   # [N_total, hidden_dim] pre-zeroed
        _tv4_buf_cached = [None]  # [N_null_sp, hidden_dim]

        def setup_tv4(ctrl_ref, label: str):
            """Sparse kernel: extract compute-only tokens, run MoE with 256 experts,
            scatter back. No null expert, no expert_map change, no gate wrap."""
            from vllm.distributed import (
                get_ep_group,
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            from vllm.forward_context import get_forward_context, set_forward_context
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE

            _restore_expert_attrs()
            setup_bsp_g_attn_rs(ctrl_ref, label)
            _tv4_moe_cache.clear()
            _tv4_step_in_block[0] = 0
            _tv4_prev_decoded_sp[0] = None
            _tv4_null_mask_sp[0] = None
            _tv4_compute_indices[0] = None
            _REFRESH_M = 5
            _tp_size = get_tensor_model_parallel_world_size()
            _tp_rank = get_tensor_model_parallel_rank()
            _ep_group = get_ep_group()
            _ep_world = _ep_group.world_size

            for i, blk in enumerate(moe_blocks):
                layer_id = i
                experts_obj = blk.experts

                def _make_tv4_impl(orig_impl, lid, exp_obj):
                    def tv4_forward_impl(self, hidden_states, router_logits):
                        compute_indices = _tv4_compute_indices[0]

                        if compute_indices is None:
                            y = orig_impl(hidden_states, router_logits)
                            if isinstance(y, tuple):
                                y = y[1]
                            with COMPONENT_TIMER.time("moe.td4_cache_write"):
                                _tv4_moe_cache[lid] = y.detach().clone()
                            return y

                        # --- SPARSE PATH ---
                        ctx = get_forward_context()
                        sp_ctx = (
                            ctx.dp_metadata.sp_local_sizes(self.sp_size)
                            if ctx.dp_metadata
                            else nullcontext()
                        )

                        with sp_ctx:
                            with COMPONENT_TIMER.time("moe.dispatch"):
                                hs_g, rl_g = _ep_group.dispatch(
                                    hidden_states, router_logits,
                                    self.is_sequence_parallel)

                            if _kp_enabled:
                                _kp_s = torch.cuda.Event(enable_timing=True)
                                _kp_e = torch.cuda.Event(enable_timing=True)
                                _kp_s.record()

                            with COMPONENT_TIMER.time("moe.quant_apply"):
                                topk_weights, topk_ids, _ = FusedMoE.select_experts(
                                    hidden_states=hs_g,
                                    router_logits=rl_g,
                                    use_grouped_topk=self.use_grouped_topk,
                                    top_k=self.top_k,
                                    renormalize=self.renormalize,
                                    topk_group=self.topk_group,
                                    num_expert_group=self.num_expert_group,
                                    custom_routing_function=self.custom_routing_function,
                                    scoring_func=self.scoring_func,
                                    routed_scaling_factor=self.routed_scaling_factor,
                                    e_score_correction_bias=self.e_score_correction_bias,
                                    indices_type=getattr(self, 'topk_indices_dtype', None),
                                )

                                buf_hs = _tv4_buf_hs[0]
                                buf_ids = _tv4_buf_ids[0]
                                buf_wts = _tv4_buf_wts[0]
                                buf_out_g = _tv4_buf_out_g[0]

                                torch.index_select(hs_g, 0, compute_indices, out=buf_hs)
                                torch.index_select(topk_ids, 0, compute_indices, out=buf_ids)
                                torch.index_select(topk_weights, 0, compute_indices, out=buf_wts)

                                from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
                                c_out = fused_experts(
                                    hidden_states=buf_hs,
                                    w1=exp_obj.w13_weight,
                                    w2=exp_obj.w2_weight,
                                    topk_weights=buf_wts,
                                    topk_ids=buf_ids,
                                    inplace=False,
                                    activation=self.activation,
                                    apply_router_weight_on_input=self.apply_router_weight_on_input,
                                    global_num_experts=self.global_num_experts,
                                    expert_map=self.expert_map,
                                )

                                buf_out_g.index_copy_(0, compute_indices, c_out)

                            if _kp_enabled:
                                _kp_e.record()
                                _kp_events.append((_kp_s, _kp_e, "TV4"))

                            with COMPONENT_TIMER.time("moe.combine"):
                                y_sp = _ep_group.combine(
                                    buf_out_g, self.is_sequence_parallel)

                        null_idx_sp = _tv4_null_indices_sp[0]
                        with COMPONENT_TIMER.time("moe.td4_cache_merge"):
                            if (null_idx_sp is not None
                                    and lid in _tv4_moe_cache
                                    and _tv4_moe_cache[lid].shape[0] == y_sp.shape[0]):
                                buf_cached = _tv4_buf_cached[0]
                                torch.index_select(_tv4_moe_cache[lid], 0, null_idx_sp,
                                                   out=buf_cached)
                                y_sp.index_copy_(0, null_idx_sp, buf_cached)

                        return y_sp
                    return types.MethodType(tv4_forward_impl, blk.experts)
                blk.experts.forward_impl = _make_tv4_impl(
                    blk.experts.forward_impl, layer_id, experts_obj)

            orig_bi = decoder.block_init
            def block_init_tv4(block_x, block_id):
                _tv4_moe_cache.clear()
                _tv4_step_in_block[0] = 0
                _tv4_prev_decoded_sp[0] = None
                _tv4_null_mask_sp[0] = None
                _tv4_compute_indices[0] = None
                _tv4_null_indices_sp[0] = None
                return orig_bi(block_x, block_id)
            decoder.block_init = block_init_tv4

            orig_mf = model.forward
            def model_forward_tv4(input_ids=None, *args, **kwargs):
                if input_ids is not None:
                    step = _tv4_step_in_block[0]
                    is_mask = (input_ids == MASK_ID).view(-1)
                    chunk = is_mask.shape[0] // _tp_size
                    sp_start = _tp_rank * chunk
                    decoded_sp = ~is_mask[sp_start:sp_start + chunk]

                    prev_dec = _tv4_prev_decoded_sp[0]
                    if (prev_dec is not None
                            and prev_dec.shape[0] == decoded_sp.shape[0]
                            and step % _REFRESH_M != 1):
                        null_mask_sp = decoded_sp & prev_dec
                    else:
                        null_mask_sp = None

                    if null_mask_sp is not None and null_mask_sp.any():
                        null_byte = null_mask_sp.to(torch.uint8)
                        gathered = [torch.empty(chunk, dtype=torch.uint8,
                                                device=null_byte.device)
                                    for _ in range(_ep_world)]
                        dist.all_gather(gathered, null_byte,
                                        group=_ep_group.device_group)
                        null_gathered = torch.cat(gathered).bool()
                        compute_indices = (~null_gathered).nonzero(
                            as_tuple=True)[0]
                        _tv4_compute_indices[0] = compute_indices
                        _tv4_null_mask_sp[0] = null_mask_sp
                        _tv4_null_indices_sp[0] = null_mask_sp.nonzero(
                            as_tuple=True)[0]
                        _hidden_dim = model.config.hidden_size
                        _K = moe_blocks[0].experts.top_k
                        _N_total = chunk * _ep_world
                        _N_compute = compute_indices.shape[0]
                        _N_null_sp = _tv4_null_indices_sp[0].shape[0]
                        _dev = compute_indices.device

                        _tv4_buf_hs[0] = torch.empty(
                            _N_compute, _hidden_dim, device=_dev, dtype=torch.bfloat16)
                        _tv4_buf_ids[0] = torch.empty(
                            _N_compute, _K, device=_dev,
                            dtype=torch.int32)
                        _tv4_buf_wts[0] = torch.empty(
                            _N_compute, _K, device=_dev, dtype=torch.float32)
                        _tv4_buf_out_g[0] = torch.zeros(
                            _N_total, _hidden_dim, device=_dev, dtype=torch.bfloat16)
                        _tv4_buf_cached[0] = torch.empty(
                            _N_null_sp, _hidden_dim, device=_dev, dtype=torch.bfloat16)
                    else:
                        _tv4_compute_indices[0] = None
                        _tv4_null_mask_sp[0] = None
                        _tv4_null_indices_sp[0] = None
                else:
                    _tv4_compute_indices[0] = None
                    _tv4_null_mask_sp[0] = None
                    _tv4_null_indices_sp[0] = None

                result = orig_mf(input_ids, *args, **kwargs)

                if input_ids is not None:
                    _tv4_prev_decoded_sp[0] = decoded_sp
                    _tv4_step_in_block[0] += 1
                return result
            model.forward = model_forward_tv4

        # ---- TV5: topk_ids >= num_experts skip (no extract/scatter) ----
        _tv5_moe_cache = {}
        _tv5_null_mask_sp = [None]
        _tv5_null_indices_gathered = [None]  # [N_null] int64, gathered space
        _tv5_null_indices_sp = [None]        # [N_null_sp] int64, SP space
        _tv5_prev_decoded_sp = [None]
        _tv5_step_in_block = [0]
        _tv5_buf_cached = [None]             # [N_null_sp, hidden_dim]
        _tv5_is_sparse = [False]             # whether current forward is sparse

        def setup_tv5(ctrl_ref, label: str):
            """topk_ids >= num_experts skip: kernel auto-skips null tokens.
            No extract/scatter, no null expert, no expert_map change."""
            from vllm.distributed import (
                get_ep_group,
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            from vllm.forward_context import get_forward_context, set_forward_context
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE

            _restore_expert_attrs()
            setup_bsp_g_attn_rs(ctrl_ref, label)
            _tv5_moe_cache.clear()
            _tv5_step_in_block[0] = 0
            _tv5_prev_decoded_sp[0] = None
            _tv5_null_mask_sp[0] = None
            _tv5_null_indices_gathered[0] = None
            _tv5_null_indices_sp[0] = None
            _tv5_is_sparse[0] = False
            _REFRESH_M = 5
            _tp_size = get_tensor_model_parallel_world_size()
            _tp_rank = get_tensor_model_parallel_rank()
            _ep_group = get_ep_group()
            _ep_world = _ep_group.world_size

            for i, blk in enumerate(moe_blocks):
                layer_id = i
                experts_obj = blk.experts

                def _make_tv5_impl(orig_impl, lid, exp_obj):
                    def tv5_forward_impl(self, hidden_states, router_logits):
                        if not _tv5_is_sparse[0]:
                            y = orig_impl(hidden_states, router_logits)
                            if isinstance(y, tuple):
                                y = y[1]
                            with COMPONENT_TIMER.time("moe.td4_cache_write"):
                                _tv5_moe_cache[lid] = y.detach().clone()
                            return y

                        # --- SPARSE PATH ---
                        ctx = get_forward_context()
                        sp_ctx = (
                            ctx.dp_metadata.sp_local_sizes(self.sp_size)
                            if ctx.dp_metadata
                            else nullcontext()
                        )

                        null_idx_g = _tv5_null_indices_gathered[0]
                        null_idx_sp = _tv5_null_indices_sp[0]

                        with sp_ctx:
                            with COMPONENT_TIMER.time("moe.dispatch"):
                                hs_g, rl_g = _ep_group.dispatch(
                                    hidden_states, router_logits,
                                    self.is_sequence_parallel)

                            with COMPONENT_TIMER.time("moe.quant_apply"):
                                topk_weights, topk_ids, _ = FusedMoE.select_experts(
                                    hidden_states=hs_g,
                                    router_logits=rl_g,
                                    use_grouped_topk=self.use_grouped_topk,
                                    top_k=self.top_k,
                                    renormalize=self.renormalize,
                                    topk_group=self.topk_group,
                                    num_expert_group=self.num_expert_group,
                                    custom_routing_function=self.custom_routing_function,
                                    scoring_func=self.scoring_func,
                                    routed_scaling_factor=self.routed_scaling_factor,
                                    e_score_correction_bias=self.e_score_correction_bias,
                                    indices_type=getattr(self, 'topk_indices_dtype', None),
                                )

                                topk_ids.index_fill_(0, null_idx_g,
                                                     self.global_num_experts)

                                from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
                                c_out = fused_experts(
                                    hidden_states=hs_g,
                                    w1=exp_obj.w13_weight,
                                    w2=exp_obj.w2_weight,
                                    topk_weights=topk_weights,
                                    topk_ids=topk_ids,
                                    inplace=False,
                                    activation=self.activation,
                                    apply_router_weight_on_input=self.apply_router_weight_on_input,
                                    global_num_experts=self.global_num_experts,
                                    expert_map=self.expert_map,
                                )

                                c_out.index_fill_(0, null_idx_g, 0)

                            with COMPONENT_TIMER.time("moe.combine"):
                                y_sp = _ep_group.combine(
                                    c_out, self.is_sequence_parallel)

                        with COMPONENT_TIMER.time("moe.td4_cache_merge"):
                            if (null_idx_sp is not None
                                    and lid in _tv5_moe_cache
                                    and _tv5_moe_cache[lid].shape[0] == y_sp.shape[0]):
                                buf_cached = _tv5_buf_cached[0]
                                torch.index_select(_tv5_moe_cache[lid], 0,
                                                   null_idx_sp, out=buf_cached)
                                y_sp.index_copy_(0, null_idx_sp, buf_cached)

                        return y_sp
                    return types.MethodType(tv5_forward_impl, blk.experts)
                blk.experts.forward_impl = _make_tv5_impl(
                    blk.experts.forward_impl, layer_id, experts_obj)

            orig_bi = decoder.block_init
            def block_init_tv5(block_x, block_id):
                _tv5_moe_cache.clear()
                _tv5_step_in_block[0] = 0
                _tv5_prev_decoded_sp[0] = None
                _tv5_null_mask_sp[0] = None
                _tv5_null_indices_gathered[0] = None
                _tv5_null_indices_sp[0] = None
                _tv5_is_sparse[0] = False
                return orig_bi(block_x, block_id)
            decoder.block_init = block_init_tv5

            orig_mf = model.forward
            def model_forward_tv5(input_ids=None, *args, **kwargs):
                if input_ids is not None:
                    step = _tv5_step_in_block[0]
                    is_mask = (input_ids == MASK_ID).view(-1)
                    chunk = is_mask.shape[0] // _tp_size
                    sp_start = _tp_rank * chunk
                    decoded_sp = ~is_mask[sp_start:sp_start + chunk]

                    prev_dec = _tv5_prev_decoded_sp[0]
                    if (prev_dec is not None
                            and prev_dec.shape[0] == decoded_sp.shape[0]
                            and step % _REFRESH_M != 1):
                        null_mask_sp = decoded_sp & prev_dec
                    else:
                        null_mask_sp = None

                    if null_mask_sp is not None and null_mask_sp.any():
                        null_byte = null_mask_sp.to(torch.uint8)
                        gathered = [torch.empty(chunk, dtype=torch.uint8,
                                                device=null_byte.device)
                                    for _ in range(_ep_world)]
                        dist.all_gather(gathered, null_byte,
                                        group=_ep_group.device_group)
                        null_gathered = torch.cat(gathered).bool()
                        _tv5_null_indices_gathered[0] = null_gathered.nonzero(
                            as_tuple=True)[0]
                        _tv5_null_indices_sp[0] = null_mask_sp.nonzero(
                            as_tuple=True)[0]
                        _tv5_null_mask_sp[0] = null_mask_sp
                        _tv5_is_sparse[0] = True

                        _hidden_dim = model.config.hidden_size
                        _N_null_sp = _tv5_null_indices_sp[0].shape[0]
                        _dev = _tv5_null_indices_gathered[0].device
                        _tv5_buf_cached[0] = torch.empty(
                            _N_null_sp, _hidden_dim, device=_dev,
                            dtype=torch.bfloat16)
                    else:
                        _tv5_null_indices_gathered[0] = None
                        _tv5_null_indices_sp[0] = None
                        _tv5_null_mask_sp[0] = None
                        _tv5_is_sparse[0] = False
                else:
                    _tv5_is_sparse[0] = False

                result = orig_mf(input_ids, *args, **kwargs)

                if input_ids is not None:
                    _tv5_prev_decoded_sp[0] = decoded_sp
                    _tv5_step_in_block[0] += 1
                return result
            model.forward = model_forward_tv5

        # ---- TEAMv3: sparse dispatch + null expert + sparse combine + cache ----
        _tv3_moe_cache = {}
        _tv3_decoded_sp = [None]
        _tv3_mask_sp = [None]
        _tv3_decoded_gathered = [None]
        _tv3_mask_gathered = [None]
        _tv3_mask_indices_sp = [None]
        _tv3_mask_indices_gathered = [None]
        _tv3_step_in_block = [0]
        _tv3_sizes = [None]

        def setup_tv3(ctrl_ref, label: str):
            """BSP-TEAMv3: sparse dispatch + null expert kernel + sparse combine + cache."""
            from vllm.distributed import (
                get_ep_group,
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            from vllm.forward_context import get_forward_context, set_forward_context

            setup_bsp_g_attn_rs(ctrl_ref, label)
            _restore_expert_attrs()
            _tv3_moe_cache.clear()
            _tv3_step_in_block[0] = 0
            NULL_ID = _orig_global_num_experts[0]
            _tp_size = get_tensor_model_parallel_world_size()
            _tp_rank = get_tensor_model_parallel_rank()
            _ep_group = get_ep_group()
            _ep_world = _ep_group.world_size
            _dp_size_tv3 = _ep_world // _tp_size
            _block_length_tv3 = 32
            HOT_N_SP = (args.batch_size // _dp_size_tv3) * _block_length_tv3 // _tp_size
            HOT_N_TOTAL = HOT_N_SP * _ep_world
            _hidden_dim = model.config.hidden_size   # 2048

            for blk in moe_blocks:
                blk.experts.global_num_experts += 1
                if blk.experts.expert_map is not None:
                    blk.experts.expert_map = torch.cat([
                        blk.experts.expert_map,
                        torch.tensor([-1], device=blk.experts.expert_map.device,
                                     dtype=blk.experts.expert_map.dtype)
                    ])
                orig_rf = blk.experts.custom_routing_function

                def _make_routing_wrap(orig_fn, null_id):
                    def wrapped(hidden_states, gating_output, topk, renormalize):
                        w, idx = orig_fn(hidden_states=hidden_states,
                                         gating_output=gating_output,
                                         topk=topk, renormalize=renormalize)
                        N = idx.shape[0]
                        if N == HOT_N_TOTAL and _tv3_step_in_block[0] > 0:
                            dec = _tv3_decoded_gathered[0]
                            if dec is not None and dec.shape[0] == N:
                                idx[dec] = null_id
                        return w, idx
                    return wrapped
                blk.experts.custom_routing_function = _make_routing_wrap(orig_rf, NULL_ID)

            _buf_hs = torch.empty(HOT_N_TOTAL, _hidden_dim,
                                  device=moe_blocks[0].experts.w13_weight.device,
                                  dtype=torch.bfloat16)
            _num_gate_experts = moe_blocks[0].experts.w13_weight.shape[0] * _ep_world
            _buf_rl = torch.empty(HOT_N_TOTAL, _num_gate_experts,
                                  device=_buf_hs.device, dtype=torch.bfloat16)

            # Pre-allocated buffers for sparse dispatch/combine
            _MAX_MASK_PER_RANK = HOT_N_SP  # max possible = all tokens are MASK
            _tv3_max_sz = [_MAX_MASK_PER_RANK]
            _tv3_n_mask_total = [0]
            _tv3_disp_hs_send = torch.empty(_MAX_MASK_PER_RANK, _hidden_dim, device=_buf_hs.device, dtype=torch.bfloat16)
            _tv3_disp_rl_send = torch.empty(_MAX_MASK_PER_RANK, _num_gate_experts, device=_buf_hs.device, dtype=torch.bfloat16)
            _tv3_disp_hs_recv = [torch.empty(_MAX_MASK_PER_RANK, _hidden_dim, device=_buf_hs.device, dtype=torch.bfloat16)
                                 for _ in range(_ep_world)]
            _tv3_disp_rl_recv = [torch.empty(_MAX_MASK_PER_RANK, _num_gate_experts, device=_buf_hs.device, dtype=torch.bfloat16)
                                 for _ in range(_ep_world)]
            _tv3_comb_send = [torch.empty(_MAX_MASK_PER_RANK, _hidden_dim, device=_buf_hs.device, dtype=torch.bfloat16)
                              for _ in range(_ep_world)]
            _tv3_comb_recv = torch.empty(_MAX_MASK_PER_RANK, _hidden_dim, device=_buf_hs.device, dtype=torch.bfloat16)

            for i, blk in enumerate(moe_blocks):
                prev_fwd_impl = blk.experts.forward_impl
                layer_id = i

                def _make_tv3_impl(orig_impl, lid, ep_grp):
                    def tv3_forward_impl(self, hidden_states, router_logits):
                        N_sp = hidden_states.shape[0]
                        if lid == 0 and rank == 0 and not _tv3_moe_cache.get('_shape_log', False):
                            print(f"  [TV3-DBG] N_sp={N_sp} HOT_N_SP={HOT_N_SP} step={_tv3_step_in_block[0]} match={N_sp==HOT_N_SP}")
                            if _tv3_step_in_block[0] > 0:
                                _tv3_moe_cache['_shape_log'] = True
                        if N_sp != HOT_N_SP or _tv3_step_in_block[0] == 0:
                            y = orig_impl(hidden_states, router_logits)
                            if isinstance(y, tuple):
                                y = y[1]
                            if N_sp == HOT_N_SP:
                                _tv3_moe_cache[lid] = y.detach().clone()
                            return y

                        mask_sp = _tv3_mask_sp[0]
                        mask_idx_sp = _tv3_mask_indices_sp[0]
                        mask_idx_gathered = _tv3_mask_indices_gathered[0]
                        sizes = _tv3_sizes[0]
                        if mask_sp is None or sizes is None:
                            y = orig_impl(hidden_states, router_logits)
                            if isinstance(y, tuple):
                                y = y[1]
                            if N_sp == HOT_N_SP:
                                _tv3_moe_cache[lid] = y.detach().clone()
                            return y

                        my_rank = ep_grp.rank_in_group
                        my_sz = sizes[my_rank]
                        max_sz = _tv3_max_sz[0]
                        n_mask_total = _tv3_n_mask_total[0]

                        # --- DISPATCH: extract + AllGather (pre-allocated) ---
                        hs_compact = hidden_states.index_select(0, mask_idx_sp)
                        rl_compact = router_logits.index_select(0, mask_idx_sp)

                        COMPONENT_TIMER.add_bytes(
                            "dispatch_payload",
                            _tensor_nbytes(hs_compact) + _tensor_nbytes(rl_compact),
                        )

                        _tv3_disp_hs_send[:my_sz] = hs_compact
                        _tv3_disp_rl_send[:my_sz] = rl_compact

                        with COMPONENT_TIMER.time("moe.dispatch"):
                            dist.all_gather(_tv3_disp_hs_recv, _tv3_disp_hs_send,
                                            group=ep_grp.device_group)
                            dist.all_gather(_tv3_disp_rl_recv, _tv3_disp_rl_send,
                                            group=ep_grp.device_group)

                        # scatter gathered MASK data into full buffer
                        offset = 0
                        for r in range(_ep_world):
                            sz = sizes[r]
                            if sz > 0:
                                src_hs = _tv3_disp_hs_recv[r][:sz]
                                src_rl = _tv3_disp_rl_recv[r][:sz]
                                idx = mask_idx_gathered[offset:offset + sz]
                                _buf_hs.index_copy_(0, idx, src_hs)
                                _buf_rl.index_copy_(0, idx, src_rl)
                                offset += sz

                        # --- KERNEL ---
                        with COMPONENT_TIMER.time("moe.quant_apply"):
                            final_hs = self.quant_method.apply(
                                layer=self,
                                x=_buf_hs,
                                router_logits=_buf_rl,
                                top_k=self.top_k,
                                renormalize=self.renormalize,
                                use_grouped_topk=self.use_grouped_topk,
                                global_num_experts=self.global_num_experts,
                                expert_map=self.expert_map,
                                topk_group=self.topk_group,
                                num_expert_group=self.num_expert_group,
                                custom_routing_function=self.custom_routing_function,
                                scoring_func=self.scoring_func,
                                routed_scaling_factor=self.routed_scaling_factor,
                                e_score_correction_bias=self.e_score_correction_bias,
                                activation=self.activation,
                                apply_router_weight_on_input=self.apply_router_weight_on_input,
                                enable_eplb=self.enable_eplb,
                                expert_load_view=self.expert_load_view,
                                logical_to_physical_map=self.logical_to_physical_map,
                                logical_replica_count=self.logical_replica_count,
                            )
                        if isinstance(final_hs, tuple):
                            final_hs = final_hs[1] if len(final_hs) == 2 else final_hs[0]

                        # --- COMBINE: extract + ReduceScatter (pre-allocated) ---
                        offset = 0
                        for r in range(_ep_world):
                            sz = sizes[r]
                            if sz > 0:
                                idx = mask_idx_gathered[offset:offset + sz]
                                _tv3_comb_send[r][:sz] = final_hs.index_select(0, idx)
                                offset += sz

                        with COMPONENT_TIMER.time("moe.combine"):
                            dist.reduce_scatter(_tv3_comb_recv, _tv3_comb_send,
                                                group=ep_grp.device_group)

                        out_sp_mask = _tv3_comb_recv[:my_sz]

                        # --- Assemble y_sp from cache + fresh MASK ---
                        with COMPONENT_TIMER.time("moe.td4_cache_merge"):
                            if lid in _tv3_moe_cache:
                                y_sp = _tv3_moe_cache[lid]
                            else:
                                y_sp = torch.zeros(HOT_N_SP, _hidden_dim,
                                                   device=hidden_states.device, dtype=hidden_states.dtype)
                            y_sp.index_copy_(0, mask_idx_sp, out_sp_mask)
                        with COMPONENT_TIMER.time("moe.td4_cache_write"):
                            _tv3_moe_cache[lid] = y_sp
                        return y_sp
                    return types.MethodType(tv3_forward_impl, blk.experts)
                blk.experts.forward_impl = _make_tv3_impl(prev_fwd_impl, layer_id, _ep_group)

            orig_bi = decoder.block_init
            def block_init_tv3(block_x, block_id):
                _tv3_moe_cache.clear()
                _tv3_step_in_block[0] = 0
                return orig_bi(block_x, block_id)
            decoder.block_init = block_init_tv3

            # Pre-compute DP group for lightweight cross-DP mask sync
            from vllm.distributed import get_dp_group
            _dp_group = get_dp_group()
            _dp_world = _dp_group.world_size  # 2
            _dp_rank = _dp_group.rank_in_group
            _my_dp_mask_buf = torch.empty(HOT_N_SP * _tp_size, dtype=torch.uint8, device=_buf_hs.device)
            _all_dp_mask_bufs = [torch.empty_like(_my_dp_mask_buf) for _ in range(_dp_world)]

            orig_mf = model.forward
            def model_forward_tv3(input_ids=None, *args, **kwargs):
                is_hot = (input_ids is not None and input_ids.numel() // _tp_size == HOT_N_SP)
                if input_ids is not None and is_hot:
                    with COMPONENT_TIMER.time("model.mask_compute"):
                        is_mask = (input_ids == MASK_ID).view(-1)
                        N_dp = is_mask.shape[0]
                        chunk = N_dp // _tp_size
                        sp_start = _tp_rank * chunk
                        mask_sp = is_mask[sp_start:sp_start + chunk]
                        _tv3_mask_sp[0] = mask_sp
                        _tv3_decoded_sp[0] = ~mask_sp
                        _tv3_mask_indices_sp[0] = mask_sp.nonzero(as_tuple=True)[0]

                    with COMPONENT_TIMER.time("model.mask_allgather"):
                        my_dp_decoded_byte = (~is_mask).to(torch.uint8)
                        _my_dp_mask_buf[:N_dp] = my_dp_decoded_byte
                        dist.all_gather(_all_dp_mask_bufs, _my_dp_mask_buf,
                                        group=_dp_group.device_group)

                    with COMPONENT_TIMER.time("model.mask_postproc"):
                        decoded_gathered = torch.cat([
                            _all_dp_mask_bufs[r][:N_dp] for r in range(_dp_world)
                        ], dim=0).bool()
                        mask_gathered = ~decoded_gathered
                        _tv3_decoded_gathered[0] = decoded_gathered
                        _tv3_mask_gathered[0] = mask_gathered
                        _tv3_mask_indices_gathered[0] = mask_gathered.nonzero(as_tuple=True)[0]

                        per_rank_sizes = []
                        for r in range(_ep_world):
                            r_start = r * HOT_N_SP
                            r_end = r_start + HOT_N_SP
                            per_rank_sizes.append(int(mask_gathered[r_start:r_end].sum().item()))
                        _tv3_sizes[0] = per_rank_sizes
                        _tv3_max_sz[0] = max(per_rank_sizes)
                        _tv3_n_mask_total[0] = sum(per_rank_sizes)
                else:
                    _tv3_mask_sp[0] = None
                    _tv3_decoded_sp[0] = None
                    _tv3_decoded_gathered[0] = None
                    _tv3_mask_gathered[0] = None
                    _tv3_mask_indices_sp[0] = None
                    _tv3_mask_indices_gathered[0] = None
                    _tv3_sizes[0] = None
                result = orig_mf(input_ids, *args, **kwargs)
                if is_hot:
                    _tv3_step_in_block[0] += 1
                return result
            model.forward = model_forward_tv3

        _td5_moe_cache = {}
        _td5_decoded_sp = [None]
        _td5_step_in_block = [0]

        def setup_td5(ctrl_ref, label: str):
            """G + V4 only (cache merge, NO null expert). Control experiment."""
            setup_bsp_g_attn_rs(ctrl_ref, label)
            _restore_expert_attrs()
            _td5_moe_cache.clear()
            _td5_step_in_block[0] = 0
            _td5_decoded_sp[0] = None

            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            for i, blk in enumerate(moe_blocks):
                prev_fwd_impl = blk.experts.forward_impl
                layer_id = i

                def _make_cached_impl5(orig_impl, lid):
                    def cached_forward_impl(self, hidden_states, router_logits):
                        return orig_impl(hidden_states, router_logits)
                    return types.MethodType(cached_forward_impl, blk.experts)
                blk.experts.forward_impl = _make_cached_impl5(prev_fwd_impl, layer_id)

            orig_bi = decoder.block_init
            def block_init_td5(block_x, block_id):
                _td5_moe_cache.clear()
                _td5_step_in_block[0] = 0
                return orig_bi(block_x, block_id)
            decoder.block_init = block_init_td5

            orig_mf = model.forward
            def model_forward_td5(input_ids=None, *args, **kwargs):
                if input_ids is not None:
                    tp_rank = get_tensor_model_parallel_rank()
                    tp_size = get_tensor_model_parallel_world_size()
                    is_mask = (input_ids == MASK_ID).view(-1)
                    N_dp = is_mask.shape[0]
                    chunk = N_dp // tp_size
                    sp_start = tp_rank * chunk
                    _td5_decoded_sp[0] = ~is_mask[sp_start:sp_start + chunk]
                result = orig_mf(input_ids, *args, **kwargs)
                _td5_step_in_block[0] += 1
                return result
            model.forward = model_forward_td5

        # ---- TD6: cache merge only (no null expert, no V1/V2/V3) ----
        _td6_moe_cache = {}
        _td6_decoded_sp = [None]
        _td6_step_in_block = [0]

        def setup_td6(ctrl_ref, label: str):
            """G + V4 only: cache merge for decoded, NO null expert routing."""
            setup_bsp_g_attn_rs(ctrl_ref, label)
            _restore_expert_attrs()
            _td6_moe_cache.clear()
            _td6_step_in_block[0] = 0

            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            for i, blk in enumerate(moe_blocks):
                prev_fwd_impl = blk.experts.forward_impl
                layer_id = i

                def _make_cached6(orig_impl, lid):
                    def cached_fwd(self, hidden_states, router_logits):
                        y = orig_impl(hidden_states, router_logits)
                        if isinstance(y, tuple):
                            y = y[1]
                        N_sp = y.shape[0]
                        if _td6_step_in_block[0] > 0 and lid in _td6_moe_cache and _td6_moe_cache[lid].shape[0] == N_sp:
                            dec_sp = _td6_decoded_sp[0]
                            if dec_sp is not None and dec_sp.shape[0] == N_sp:
                                y[dec_sp] = _td6_moe_cache[lid][dec_sp]
                        _td6_moe_cache[lid] = y.detach().clone()
                        return y
                    return types.MethodType(cached_fwd, blk.experts)
                blk.experts.forward_impl = _make_cached6(prev_fwd_impl, layer_id)

            orig_bi = decoder.block_init
            def block_init_td6(block_x, block_id):
                _td6_moe_cache.clear()
                _td6_step_in_block[0] = 0
                return orig_bi(block_x, block_id)
            decoder.block_init = block_init_td6

            _td6_diag_count = [0]
            orig_mf = model.forward
            def model_forward_td6(input_ids=None, *args, **kwargs):
                if input_ids is not None:
                    tp_rank = get_tensor_model_parallel_rank()
                    tp_size = get_tensor_model_parallel_world_size()
                    is_mask = (input_ids == MASK_ID).view(-1)
                    N_dp = is_mask.shape[0]
                    chunk = N_dp // tp_size
                    sp_start = tp_rank * chunk
                    _td6_decoded_sp[0] = ~is_mask[sp_start:sp_start + chunk]
                    if rank == 0 and _td6_diag_count[0] < 40:
                        n_mask = is_mask.sum().item()
                        print(f"  [TD6-shape] call={_td6_diag_count[0]} step={_td6_step_in_block[0]} "
                              f"input_ids.shape={list(input_ids.shape)} N_dp={N_dp} N_sp={chunk} "
                              f"n_mask={n_mask} n_dec={N_dp-n_mask}")
                    _td6_diag_count[0] += 1
                result = orig_mf(input_ids, *args, **kwargs)
                _td6_step_in_block[0] += 1
                return result
            model.forward = model_forward_td6

        # ---- TD7c: null expert + zero weights for decoded ----
        _td7c_decoded_gathered = [None]

        def setup_td7c(ctrl_ref, label: str):
            """G + V1 + V2 + V3 + zero weights: topk_ids[dec]=256 AND topk_weights[dec]=0."""
            setup_bsp_g_attn_rs(ctrl_ref, label)
            _restore_expert_attrs()
            NULL_ID = _orig_global_num_experts[0]

            from vllm.distributed import (
                get_ep_group,
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            for blk in moe_blocks:
                blk.experts.global_num_experts += 1
                if blk.experts.expert_map is not None:
                    blk.experts.expert_map = torch.cat([
                        blk.experts.expert_map,
                        torch.tensor([-1], device=blk.experts.expert_map.device,
                                     dtype=blk.experts.expert_map.dtype)
                    ])
                orig_rf = blk.experts.custom_routing_function

                def _make_wrap7c(orig_fn, null_id):
                    def wrapped(hidden_states, gating_output, topk, renormalize):
                        w, idx = orig_fn(hidden_states=hidden_states,
                                         gating_output=gating_output,
                                         topk=topk, renormalize=renormalize)
                        dec = _td7c_decoded_gathered[0]
                        if dec is not None and dec.shape[0] == idx.shape[0]:
                            idx[dec] = null_id
                            w[dec] = 0.0
                        return w, idx
                    return wrapped
                blk.experts.custom_routing_function = _make_wrap7c(orig_rf, NULL_ID)

            orig_mf = model.forward
            def model_forward_td7c(input_ids=None, *args, **kwargs):
                if input_ids is not None:
                    tp_rank = get_tensor_model_parallel_rank()
                    tp_size = get_tensor_model_parallel_world_size()
                    is_mask = (input_ids == MASK_ID).view(-1)
                    N_dp = is_mask.shape[0]
                    chunk = N_dp // tp_size
                    sp_start = tp_rank * chunk
                    decoded_sp = ~is_mask[sp_start:sp_start + chunk]
                    decoded_byte = decoded_sp.to(torch.uint8)
                    ep = get_ep_group()
                    gathered = [torch.empty_like(decoded_byte) for _ in range(ep.world_size)]
                    dist.all_gather(gathered, decoded_byte, group=ep.device_group)
                    _td7c_decoded_gathered[0] = torch.cat(gathered, dim=0).bool()
                else:
                    _td7c_decoded_gathered[0] = None
                return orig_mf(input_ids, *args, **kwargs)
            model.forward = model_forward_td7c

        # ---- TD7b: diagnostic — compare MASK output with/without null expert ----
        _td7b_decoded_gathered = [None]
        _td7b_diag_done = [False]

        def setup_td7b(ctrl_ref, label: str):
            """Diagnostic: run kernel twice (normal vs null expert), compare MASK output."""
            setup_bsp_g_attn_rs(ctrl_ref, label)
            _restore_expert_attrs()
            _td7b_diag_done[0] = False
            _td7b_decoded_gathered[0] = None
            NULL_ID = _orig_global_num_experts[0]  # 256

            from vllm.distributed import (
                get_ep_group,
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

            for blk in moe_blocks:
                blk.experts.global_num_experts += 1
                if blk.experts.expert_map is not None:
                    blk.experts.expert_map = torch.cat([
                        blk.experts.expert_map,
                        torch.tensor([-1], device=blk.experts.expert_map.device,
                                     dtype=blk.experts.expert_map.dtype)
                    ])
                orig_rf = blk.experts.custom_routing_function

                def _make_diag_wrap(orig_fn, null_id, experts_obj, layer_idx):
                    def wrapped(hidden_states, gating_output, topk, renormalize):
                        w, idx = orig_fn(hidden_states=hidden_states,
                                         gating_output=gating_output,
                                         topk=topk, renormalize=renormalize)
                        dec = _td7b_decoded_gathered[0]
                        if dec is not None and not _td7b_diag_done[0] and layer_idx == 0 and dec.shape[0] == idx.shape[0] and dec.sum().item() > 100 and (~dec).sum().item() > 100:
                            _td7b_diag_done[0] = True
                            mask_positions = ~dec
                            n_mask = mask_positions.sum().item()
                            n_dec = dec.sum().item()
                            # Normal topk_ids — call kernel directly
                            from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
                            emap = experts_obj.expert_map
                            gne = experts_obj.global_num_experts
                            out_normal = fused_experts(
                                hidden_states=hidden_states,
                                w1=experts_obj.w13_weight,
                                w2=experts_obj.w2_weight,
                                topk_weights=w,
                                topk_ids=idx,
                                activation=experts_obj.activation,
                                apply_router_weight_on_input=experts_obj.apply_router_weight_on_input,
                                global_num_experts=gne,
                                expert_map=emap,
                            )
                            # Null expert topk_ids
                            idx_null = idx.clone()
                            w_null = w.clone()
                            idx_null[dec] = null_id
                            out_null = fused_experts(
                                hidden_states=hidden_states,
                                w1=experts_obj.w13_weight,
                                w2=experts_obj.w2_weight,
                                topk_weights=w_null,
                                topk_ids=idx_null,
                                activation=experts_obj.activation,
                                apply_router_weight_on_input=experts_obj.apply_router_weight_on_input,
                                global_num_experts=gne,
                                expert_map=emap,
                            )
                            # Compare MASK positions
                            mask_out_normal = out_normal[mask_positions]
                            mask_out_null = out_null[mask_positions]
                            diff = (mask_out_normal - mask_out_null).abs()
                            if rank == 0:
                                print(f"\n  [TD7b-DIAG] Layer 0, N={idx.shape[0]}, n_mask={n_mask}, n_dec={n_dec}")
                                print(f"  [TD7b-DIAG] MASK output normal norm: {mask_out_normal.norm().item():.2f}")
                                print(f"  [TD7b-DIAG] MASK output null   norm: {mask_out_null.norm().item():.2f}")
                                print(f"  [TD7b-DIAG] MASK diff: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")
                                print(f"  [TD7b-DIAG] Decoded output normal norm: {out_normal[dec].norm().item():.2f}")
                                print(f"  [TD7b-DIAG] Decoded output null   norm: {out_null[dec].norm().item():.2f}")
                                if diff.max().item() > 1e-5:
                                    print(f"  [TD7b-DIAG] *** MASK OUTPUTS DIFFER! Null expert corrupts MASK! ***")
                                else:
                                    print(f"  [TD7b-DIAG] MASK outputs IDENTICAL. Null expert does NOT corrupt MASK.")
                        return w, idx  # return NORMAL routing (don't apply null expert)
                    return wrapped
                blk.experts.custom_routing_function = _make_diag_wrap(orig_rf, NULL_ID, blk.experts, moe_blocks.index(blk))

            orig_mf = model.forward
            def model_forward_td7b(input_ids=None, *args, **kwargs):
                if input_ids is not None:
                    tp_rank = get_tensor_model_parallel_rank()
                    tp_size = get_tensor_model_parallel_world_size()
                    is_mask = (input_ids == MASK_ID).view(-1)
                    N_dp = is_mask.shape[0]
                    chunk = N_dp // tp_size
                    sp_start = tp_rank * chunk
                    decoded_sp = ~is_mask[sp_start:sp_start + chunk]
                    decoded_byte = decoded_sp.to(torch.uint8)
                    ep = get_ep_group()
                    gathered = [torch.empty_like(decoded_byte) for _ in range(ep.world_size)]
                    dist.all_gather(gathered, decoded_byte, group=ep.device_group)
                    _td7b_decoded_gathered[0] = torch.cat(gathered, dim=0).bool()
                else:
                    _td7b_decoded_gathered[0] = None
                return orig_mf(input_ids, *args, **kwargs)
            model.forward = model_forward_td7b

        def reset_controller(ctrl_ref):
            ctrl_ref.prev_N.clear()
            ctrl_ref.K_init.clear()
            ctrl_ref.cold_count = 0
            ctrl_ref.hot_count = 0
            ctrl_ref.eb_calls = 0
            ctrl_ref.eb_skips = 0
            ctrl_ref._bufs.clear()
            ctrl_ref.k_init_history.clear()
            ctrl_ref.s_mask_cache.clear()
            ctrl_ref.pop_cache.clear()
            ctrl_ref._fwd_in_block.clear()
            ctrl_ref._block_idx.clear()
            ctrl_ref.reset_block_clock()

        def new_ctrl(ep_reduce_hot_update: bool = False):
            ctrl_cls = BSPMSkipControllerEPReduce if ep_reduce_hot_update else BSPMSkipController
            return ctrl_cls(
                num_layers=19,
                K=8,
                M=4,
                K_target=40,
                quality_floor=0.70,
                q_major=1.0,
                per_round_cap=8,
                skip_m=5,
            )

        def run_config(label, setup_fn, num_runs: int, do_warmup: bool,
                       ep_reduce_hot_update: bool = False):
            if rank == 0:
                print(f"\n--- {label} ---")
            ctrl = new_ctrl(ep_reduce_hot_update=ep_reduce_hot_update)
            setup_fn(ctrl, label)

            if do_warmup:
                reset_controller(ctrl)
                dllm = make_dllm()
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    _ = dllm.generate(
                        my_input.clone(),
                        gen_length=args.gen_length,
                        block_length=BLOCK_LENGTH,
                    )
                torch.cuda.synchronize()
                dist.barrier()
                if rank == 0:
                    print(
                        f"  Warmup: {dllm.diff_iteration.num_forwards} fwd, "
                        f"cold={ctrl.cold_count} hot={ctrl.hot_count}"
                    )

            times = []
            fwds = []
            timing_runs = []
            shape_runs = []
            layout_runs = []
            quality = {}
            out = None
            for ri in range(num_runs):
                reset_controller(ctrl)
                COMPONENT_TIMER.enabled = bool(args.component_timing)
                PROFILE_RANGES.enabled = bool(args.cuda_profiler_capture)
                COMPONENT_TIMER.reset()
                SHAPE_PROBE.enabled = args.mode == "shape"
                SHAPE_PROBE.limit = args.shape_limit
                SHAPE_PROBE.reset()
                LAYOUT_DIAG.enabled = bool(args.layout_diagnostics)
                LAYOUT_DIAG.limit = args.layout_limit
                LAYOUT_DIAG.reset()

                dllm = make_dllm()
                torch.cuda.synchronize()
                dist.barrier()
                if args.cuda_profiler_capture:
                    torch.cuda.profiler.start()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    with PROFILE_RANGES.range(f"{label}.generate.run{ri + 1}"):
                        out = dllm.generate(
                            my_input.clone(),
                            gen_length=args.gen_length,
                            block_length=BLOCK_LENGTH,
                        )
                torch.cuda.synchronize()
                if args.cuda_profiler_capture:
                    torch.cuda.profiler.stop()
                dist.barrier()
                dt = time.perf_counter() - t0
                nf = dllm.diff_iteration.num_forwards

                timing_summary = (
                    COMPONENT_TIMER.summarize_across_ranks(nf)
                    if args.component_timing
                    else None
                )
                shape_summary = SHAPE_PROBE.gather() if args.mode == "shape" else None
                layout_summary = LAYOUT_DIAG.gather() if args.layout_diagnostics else None
                COMPONENT_TIMER.enabled = False
                PROFILE_RANGES.enabled = False
                LAYOUT_DIAG.enabled = False

                times.append(dt)
                fwds.append(nf)
                if rank == 0 and timing_summary is not None:
                    timing_runs.append(timing_summary)
                if rank == 0 and shape_summary is not None:
                    shape_runs.append(shape_summary)
                if rank == 0 and layout_summary is not None:
                    layout_runs.append(layout_summary)

                if rank == 0:
                    st = ctrl.stats()
                    print(
                        f"  Run {ri + 1}: {dt:.3f}s, {nf} fwd, "
                        f"{dt * 1000 / nf:.2f} ms/fwd, "
                        f"cold={ctrl.cold_count} hot={ctrl.hot_count} "
                        f"eb_skip={ctrl.eb_skips} path={st['path_counts']}"
                    )
                    _print_timing_summary(label, timing_summary)
                    _print_shape_summary(label, shape_summary)
                    _print_layout_summary(label, layout_summary)

            quality = {}
            if out is not None:
                quality = _manual_quality_payload(
                    tokenizer,
                    out,
                    prompt_len,
                    full_output=args.full_quality_output,
                    local_bs=local_bs,
                    dp_rank=dp_rank,
                )
                if rank == 0 and not args.no_quality:
                    print(f"  Verifiable Quality ({label}):")
                    for bi, text in quality.items():
                        print(f"    #{bi}: {text[:800]}")

            avg_time = sum(times) / len(times)
            avg_fwd = sum(fwds) / len(fwds)
            controller_all_ranks = _gather_controller_stats(ctrl)
            _print_controller_rank_summary(label, controller_all_ranks)
            result = {
                "config": label,
                "time_s": avg_time,
                "fwd": avg_fwd,
                "ms_fwd": avg_time / avg_fwd * 1000,
                "times": times,
                "fwds": fwds,
                "controller": ctrl.stats(),
                "controller_all_ranks": controller_all_ranks if rank == 0 else [],
                "component_timing_runs": timing_runs if rank == 0 else [],
                "shape_runs": shape_runs if rank == 0 else [],
                "layout_diagnostic_runs": layout_runs if rank == 0 else [],
                "quality_snippets": quality if rank == 0 else {},
                "quality_is_full_output": bool(args.full_quality_output),
            }
            # Avoid cumulative VRAM growth across A/B/C runs in the same process.
            del dllm
            if out is not None:
                del out
            torch.cuda.empty_cache()
            dist.barrier()
            return result

        def run_forward_check():
            hidden_size = int(config.hidden_size)
            check_seq = BLOCK_LENGTH
            layer_ids = [0, len(moe_blocks) // 2, len(moe_blocks) - 1]
            results = []

            for layer_id in layer_ids:
                gen = torch.Generator(device=device)
                gen.manual_seed(20260427 + dp_rank * 1000 + layer_id)
                inputs = [
                    torch.randn(
                        (local_bs, check_seq, hidden_size),
                        device=device,
                        dtype=torch.float32,
                        generator=gen,
                    ).to(torch.bfloat16)
                    for _ in range(6)
                ]

                def run_sequence(label, setup_fn):
                    ctrl = new_ctrl()
                    setup_fn(ctrl, label)
                    ctrl.note_block_start(0)
                    outs = []
                    with torch.inference_mode():
                        for x in inputs:
                            y = moe_blocks[layer_id](x.clone())
                            outs.append(y.detach().clone())
                    torch.cuda.synchronize()
                    dist.barrier()
                    return outs, ctrl.stats()

                base_outs, base_stats = run_sequence(
                    f"FWDCHK baseline layer{layer_id}", setup_baseline
                )
                bsp_outs, bsp_stats = run_sequence(
                    f"FWDCHK bsp layer{layer_id}", setup_bsp
                )

                for step, (a, b) in enumerate(zip(base_outs, bsp_outs)):
                    diff = (a.float() - b.float()).abs()
                    denom = a.float().abs().max().clamp_min(1e-6)
                    local = torch.tensor(
                        [
                            diff.max().item(),
                            diff.mean().item(),
                            (diff.max() / denom).item(),
                        ],
                        dtype=torch.float64,
                        device=device,
                    )
                    dist.all_reduce(local, op=dist.ReduceOp.MAX)
                    if rank == 0:
                        results.append(
                            {
                                "layer": layer_id,
                                "step": step,
                                "path_hint": (
                                    "cold" if step == 0
                                    else "hot_update" if step == 5
                                    else "hot_skip"
                                ),
                                "rankmax_abs_max": float(local[0].item()),
                                "rankmax_abs_mean": float(local[1].item()),
                                "rankmax_rel_max": float(local[2].item()),
                            }
                        )

                if rank == 0:
                    print(f"  Forward-check layer {layer_id}:")
                    print(f"    baseline paths: {base_stats['path_counts']}")
                    print(f"    bsp paths:      {bsp_stats['path_counts']}")
                    for row in results[-6:]:
                        print(
                            f"    step={row['step']} {row['path_hint']:<10} "
                            f"abs_max={row['rankmax_abs_max']:.6e} "
                            f"abs_mean={row['rankmax_abs_mean']:.6e} "
                            f"rel_max={row['rankmax_rel_max']:.6e}"
                        )

            return {
                "config": "forward-check BSP vs baseline MoE blocks",
                "local_bs": local_bs,
                "seq_len": check_seq,
                "hidden_size": hidden_size,
                "layers": layer_ids,
                "results": results if rank == 0 else [],
            }

        def run_moe_internal_check():
            from vllm.distributed import tensor_model_parallel_all_gather
            from vllm.forward_context import DPMetadata, set_forward_context
            from vllm.model_executor.models.utils import sequence_parallel_chunk

            hidden_size = int(config.hidden_size)
            check_seq = BLOCK_LENGTH
            layer_ids = [0, len(moe_blocks) // 2, len(moe_blocks) - 1]
            results = []

            def make_ctrl_for_compare(layer_id: int):
                ctrl = new_ctrl()
                ctrl.note_block_start(0)
                return ctrl

            def apply_routing_for_layer(block, layer_id: int, ctrl_ref):
                b = block.gate.expert_bias
                r = block.gate.routed_scaling_factor
                ng = block.gate.n_group
                tkg = block.gate.topk_group

                def fn(hidden_states=None,
                       gating_output=None,
                       topk=None,
                       renormalize=None,
                       **kwargs):
                    go = gating_output
                    if go is None:
                        go = kwargs.get("router_logits")
                    if go is None:
                        raise TypeError("diagnostic routing requires gating_output")
                    sm = ctrl_ref.get_s_mask(layer_id, go, b)
                    w, idx = fused_routing(
                        go,
                        b,
                        r,
                        s_mask=sm,
                        K=4,
                        ng=ng,
                        tkg=tkg,
                    )
                    return w.to(go.dtype), idx

                block.gate.routing = fn
                block.experts.custom_routing_function = fn

            def context_for_mode(mode: str, n_tokens: int, local_n: int):
                if mode == "full":
                    return set_forward_context(
                        attn_metadata=None,
                        vllm_config=vllm_cfg,
                        num_tokens=n_tokens,
                    )
                if mode == "sp":
                    if dp_size > 1:
                        return set_forward_context(
                            attn_metadata=None,
                            vllm_config=vllm_cfg,
                            num_tokens=n_tokens,
                        )
                    assert dp_size == 1
                    # vLLM does not build DPMetadata when dp_size==1, but EP
                    # sequence-parallel dispatch/combine still needs per-SP
                    # chunk sizes. This diagnostic injects the equivalent
                    # metadata without changing production paths.
                    num_tokens_cpu = torch.tensor(
                        [n_tokens],
                        dtype=torch.int32,
                        device="cpu",
                    )
                    dp_meta = DPMetadata(
                        max_tokens_across_dp_cpu=torch.tensor(
                            n_tokens,
                            dtype=torch.int32,
                            device="cpu",
                        ),
                        num_tokens_across_dp_cpu=num_tokens_cpu,
                        local_sizes=[local_n for _ in range(tp_size)],
                    )
                    return set_forward_context(
                        attn_metadata=None,
                        vllm_config=vllm_cfg,
                        num_tokens=n_tokens,
                        num_tokens_across_dp=num_tokens_cpu,
                    ) if False else _InjectedDPMetadataContext(vllm_cfg, dp_meta)
                raise ValueError(mode)

            class _InjectedDPMetadataContext:
                def __init__(self, cfg, dp_meta):
                    self.cfg = cfg
                    self.dp_meta = dp_meta
                    self.ctx = None

                def __enter__(self):
                    from vllm.forward_context import create_forward_context
                    from vllm.forward_context import override_forward_context

                    self.ctx = override_forward_context(
                        create_forward_context(
                            attn_metadata=None,
                            vllm_config=self.cfg,
                            virtual_engine=0,
                            dp_metadata=self.dp_meta,
                        )
                    )
                    return self.ctx.__enter__()

                def __exit__(self, exc_type, exc, tb):
                    return self.ctx.__exit__(exc_type, exc, tb)

            for layer_id in layer_ids:
                block = moe_blocks[layer_id]
                shared_mod = block.shared_experts if block.config.num_shared_experts else None
                experts = block.experts
                orig_gate_routing = block.gate.routing
                orig_custom_routing = experts.custom_routing_function
                orig_sp_state = (
                    getattr(experts, "is_sequence_parallel", False),
                    getattr(experts, "sp_size", 1),
                )

                gen = torch.Generator(device=device)
                gen.manual_seed(20260428 + dp_rank * 1000 + layer_id)
                hs_full = torch.randn(
                    (local_bs * check_seq, hidden_size),
                    device=device,
                    dtype=torch.float32,
                    generator=gen,
                ).to(torch.bfloat16)
                n_tokens = hs_full.shape[0]

                with torch.inference_mode():
                    hs_sp = sequence_parallel_chunk(hs_full)
                    hs_gather = tensor_model_parallel_all_gather(hs_sp, dim=0)[:n_tokens]
                    local_rows = hs_sp.shape[0]
                    chunk_start = tp_rank_local * local_rows
                    chunk_end = min(chunk_start + local_rows, n_tokens)
                    valid_rows = max(chunk_end - chunk_start, 0)

                    layer_rows = []
                    layer_rows.append({
                        "layer": layer_id,
                        "stage": "chunk_gather_identity",
                        "tp_rank": tp_rank_local,
                        "valid_rows": int(valid_rows),
                        **_diff_metrics(hs_full, hs_gather, device),
                    })

                    shared_full = shared_mod(hs_full) if shared_mod is not None else torch.zeros_like(hs_full)
                    shared_sp = shared_mod(hs_sp) if shared_mod is not None else torch.zeros_like(hs_sp)
                    shared_gather = tensor_model_parallel_all_gather(shared_sp, dim=0)[:n_tokens]
                    layer_rows.append({
                        "layer": layer_id,
                        "stage": "shared_gather_equiv",
                        **_diff_metrics(shared_full, shared_gather, device),
                    })

                    logits_full = block.gate.get_logits(hs_full)
                    logits_sp = block.gate.get_logits(hs_sp)
                    logits_gather = tensor_model_parallel_all_gather(logits_sp, dim=0)[:n_tokens]
                    layer_rows.append({
                        "layer": layer_id,
                        "stage": "gate_logits_gather_equiv",
                        **_diff_metrics(logits_full, logits_gather, device),
                    })

                    try:
                        ctrl_full = make_ctrl_for_compare(layer_id)
                        apply_routing_for_layer(block, layer_id, ctrl_full)
                        _set_experts_sequence_parallel(experts, False, tp_size)
                        with context_for_mode("full", n_tokens, local_rows):
                            routed_full = experts.forward_impl(hs_full, logits_full)

                        ctrl_sp = make_ctrl_for_compare(layer_id)
                        apply_routing_for_layer(block, layer_id, ctrl_sp)
                        _set_experts_sequence_parallel(experts, True, tp_size)
                        with context_for_mode("sp", n_tokens, local_rows):
                            routed_sp = experts.forward_impl(hs_sp, logits_sp)
                        routed_gather = tensor_model_parallel_all_gather(
                            routed_sp, dim=0)[:n_tokens]

                        layer_rows.append({
                            "layer": layer_id,
                            "stage": "routed_fused_moe_gather_equiv",
                            "full_shape": list(routed_full.shape),
                            "sp_shape": list(routed_sp.shape),
                            **_diff_metrics(routed_full, routed_gather, device),
                        })

                        total_full = routed_full + shared_full
                        total_gather = routed_gather + shared_gather
                        layer_rows.append({
                            "layer": layer_id,
                            "stage": "total_moe_gather_equiv",
                            **_diff_metrics(total_full, total_gather, device),
                        })
                    finally:
                        block.gate.routing = orig_gate_routing
                        experts.custom_routing_function = orig_custom_routing
                        experts.is_sequence_parallel = orig_sp_state[0]
                        experts.sp_size = orig_sp_state[1]

                    results.extend(layer_rows)

                if rank == 0:
                    print(f"  MoE-internal layer {layer_id}:")
                    for row in layer_rows:
                        print(
                            f"    {row['stage']:<30} "
                            f"abs_max={row['rankmax_abs_max']:.6e} "
                            f"abs_mean={row['rankmax_abs_mean']:.6e} "
                            f"rel_max={row['rankmax_rel_max']:.6e}"
                        )

            return {
                "config": "MoE internal full-vs-SP equivalence check",
                "local_bs": local_bs,
                "seq_len": check_seq,
                "hidden_size": hidden_size,
                "layers": layer_ids,
                "results": results if rank == 0 else [],
            }

        if args.mode == "forward-check":
            results = {
                "batch_size": args.batch_size,
                "local_bs": local_bs,
                "gen_length": args.gen_length,
                "block_length": BLOCK_LENGTH,
                "tp_size": tp_size,
                "dp_size": dp_size,
                "ep_size": world_size,
                "backend": alltoall_backend,
                "forward_check": run_forward_check(),
            }
            out_path = REPO_ROOT / "codex_coding" / "results" / "bsp_moe_forward_check.json"
        elif args.mode == "moe-internal-check":
            results = {
                "batch_size": args.batch_size,
                "local_bs": local_bs,
                "gen_length": args.gen_length,
                "block_length": BLOCK_LENGTH,
                "tp_size": tp_size,
                "dp_size": dp_size,
                "ep_size": world_size,
                "backend": alltoall_backend,
                "moe_internal_check": run_moe_internal_check(),
            }
            out_path = REPO_ROOT / "codex_coding" / "results" / "bsp_moe_moe_internal_check.json"
        elif args.mode == "shape":
            results = {
                "batch_size": args.batch_size,
                "local_bs": local_bs,
                "gen_length": args.gen_length,
                "block_length": BLOCK_LENGTH,
                "tp_size": tp_size,
                "dp_size": dp_size,
                "ep_size": world_size,
                "backend": alltoall_backend,
                "shape_baseline": run_config(
                    "P) C12 shape-probe baseline",
                    setup_baseline,
                    num_runs=args.num_runs,
                    do_warmup=False,
                ),
                "shape_bsp": run_config(
                    "P2) C12 shape-probe BSP-MoE",
                    setup_bsp,
                    num_runs=args.num_runs,
                    do_warmup=False,
                ),
            }
            out_path = REPO_ROOT / "codex_coding" / "results" / "bsp_moe_shape_probe.json"
        else:
            ra = None
            rb = None
            rc = None
            rd = None
            re = None
            rf = None
            rg = None
            rg2 = None
            rgs = None
            selected_configs = {
                "all": {"A", "B", "C", "D", "E", "G", "F"},
                "bspg": {"A", "E", "G", "F"},
                "bspg2": {"A", "E", "G", "G2", "F"},
                "aeg2f": {"A", "E", "G2", "F"},
                "aeggf": {"A", "E", "G", "G2", "F"},
                "bspg_source": {"A", "E", "G", "GS"},
                "bspg_h": {"GS", "H"},
                "bspg_team": {"G", "TEAM"},
                "team_debug": {"G", "TD4", "TD8"},  # M=5 cache refresh quality test
                "team_kp": {"G", "TV4"},  # kernel phase timing experiment
                "team_g_only": {"G"},
                "team_td4_only": {"TD4"},
                "team_tv4": {"TV4"},
                "team_tv5": {"TV5"},
                "team_tv4m": {"TV4m"},
                "team_tv4m_v2": {"TV4m-v2"},
                "team_tv6": {"TV6"},
            }[args.config_set]
            if args.profile_target == "baseline":
                selected_configs = selected_configs & {"A"}
            elif args.profile_target == "bsp":
                selected_configs = selected_configs - {"A"}

            if "A" in selected_configs:
                ra = run_config(
                    "A) C12-AgRs baseline",
                    setup_baseline,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "B" in selected_configs:
                rb = run_config(
                    "B) C12-BSP-MoE",
                    setup_bsp,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "C" in selected_configs:
                rc = run_config(
                    "C) C12-BSP-DelayGather",
                    setup_bsp_delay,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "D" in selected_configs:
                rd = run_config(
                    "D) C12-BSP-DelayGather-M3EPReduce",
                    setup_bsp_delay_m3,
                    num_runs=args.num_runs,
                    do_warmup=True,
                    ep_reduce_hot_update=True,
                )
            if "E" in selected_configs:
                re = run_config(
                    "E) C12-BSP-CrossLayerSP",
                    setup_bsp_cross_layer_sp,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "G" in selected_configs:
                rg = run_config(
                    "G) C12-BSP-G-AttnReduceScatterSP",
                    setup_bsp_g_attn_rs,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "G2" in selected_configs:
                rg2 = run_config(
                    "G2) C12-BSP-G2-SPParityBundle",
                    setup_bsp_g2_sp_parity,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "GS" in selected_configs:
                rgs = run_config(
                    "GS) C12-BSP-G-SourcePath",
                    setup_bsp_g_source,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "F" in selected_configs:
                rf = run_config(
                    "F) C12-BSP-AllReduceFullProbe",
                    setup_bsp_allreduce_full,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "H" in selected_configs:
                rh = run_config(
                    "H) C12-BSP-H-AllReduce",
                    setup_bsp_h,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "TEAM" in selected_configs:
                r_team = run_config(
                    "TEAM) C12-BSP-G-TEAM-DecodedSkip",
                    setup_bsp_team,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
                if rank == 0 and _TEAM_SKIP_CTRL is not None:
                    print(f"  TEAM skip stats: {_TEAM_SKIP_CTRL.stats()}")
            if "TD1" in selected_configs:
                r_td1 = run_config(
                    "TD1) G+V1(num_experts=257)",
                    setup_td1,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "TD2" in selected_configs:
                r_td2 = run_config(
                    "TD2) G+V1+V2(expert_map[256]=-1)",
                    setup_td2,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "TD3" in selected_configs:
                r_td3 = run_config(
                    "TD3) G+V1+V2+V3(routing_wrap)",
                    setup_td3,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "TD4" in selected_configs:
                r_td4 = run_config(
                    "TD4) G+V1+V2+V3+V4(cache_merge)",
                    setup_td4,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "TD8" in selected_configs:
                r_td8 = run_config(
                    "TD8) G+null_expert+refresh(no_cache)",
                    setup_td8,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "TV4" in selected_configs:
                r_tv4 = run_config(
                    "TV4) sparse_kernel(no_null_expert)",
                    setup_tv4,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "TV5" in selected_configs:
                r_tv5 = run_config(
                    "TV5) topk_skip(no_extract_scatter)",
                    setup_tv5,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "TV4m" in selected_configs:
                r_tv4m = run_config(
                    "TV4m) mapped_kernel(compact_cache)",
                    setup_tv4m,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "TV4m-v2" in selected_configs:
                r_tv4m_v2 = run_config(
                    "TV4m-v2) full_align+remap(compact_cache)",
                    setup_tv4m_v2,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "TV6" in selected_configs:
                r_tv6 = run_config(
                    "TV6) sparse_dispatch(compact_comm)",
                    setup_tv6,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "TD5" in selected_configs:
                r_td5 = run_config(
                    "TD5) G+V4_only(cache_no_null)",
                    setup_td5,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "TD6" in selected_configs:
                r_td6 = run_config(
                    "TD6) G+cache_merge_only",
                    setup_td6,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "TD7c" in selected_configs:
                r_td7c = run_config(
                    "TD7c) G+null_expert+zero_weights",
                    setup_td7c,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "TV3" in selected_configs:
                r_tv3 = run_config(
                    "TV3) BSP-TEAMv3-SparseDispatch",
                    setup_tv3,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            if "TD7b" in selected_configs:
                r_td7b = run_config(
                    "TD7b) DIAG:MASK_output_compare",
                    setup_td7b,
                    num_runs=args.num_runs,
                    do_warmup=True,
                )
            delta_b_pct = (
                (rb["ms_fwd"] / ra["ms_fwd"] - 1.0) * 100.0
                if ra is not None and rb is not None
                else None
            )
            delta_c_pct = (
                (rc["ms_fwd"] / ra["ms_fwd"] - 1.0) * 100.0
                if ra is not None and rc is not None
                else None
            )
            delta_d_pct = (
                (rd["ms_fwd"] / ra["ms_fwd"] - 1.0) * 100.0
                if ra is not None and rd is not None
                else None
            )
            delta_e_pct = (
                (re["ms_fwd"] / ra["ms_fwd"] - 1.0) * 100.0
                if ra is not None and re is not None
                else None
            )
            delta_f_pct = (
                (rf["ms_fwd"] / ra["ms_fwd"] - 1.0) * 100.0
                if ra is not None and rf is not None
                else None
            )
            delta_g_pct = (
                (rg["ms_fwd"] / ra["ms_fwd"] - 1.0) * 100.0
                if ra is not None and rg is not None
                else None
            )
            delta_g2_pct = (
                (rg2["ms_fwd"] / ra["ms_fwd"] - 1.0) * 100.0
                if ra is not None and rg2 is not None
                else None
            )
            delta_gs_pct = (
                (rgs["ms_fwd"] / ra["ms_fwd"] - 1.0) * 100.0
                if ra is not None and rgs is not None
                else None
            )
            if rank == 0 and _kp_enabled and _kp_events:
                torch.cuda.synchronize()
                from collections import defaultdict
                kp_by_config = defaultdict(list)
                for s, e, label in _kp_events:
                    kp_by_config[label].append(s.elapsed_time(e))
                print(f"\n{'=' * 70}")
                print(f"KERNEL PHASE GPU WALL-CLOCK (dispatch→combine, per MoE layer call)")
                print(f"{'=' * 70}")
                for label, times in sorted(kp_by_config.items()):
                    import numpy as np
                    t = np.array(times)
                    print(f"  {label}: n={len(t)}, mean={t.mean():.3f} ms, "
                          f"median={np.median(t):.3f} ms, p95={np.percentile(t,95):.3f} ms, "
                          f"total={t.sum():.0f} ms")
                _kp_events.clear()

            if rank == 0:
                print(f"\n{'=' * 70}")
                print(f"SUMMARY — batch={args.batch_size}, gen={args.gen_length}")
                print(f"{'=' * 70}")
                print(f"{'Config':<28} {'Time':>8} {'Fwd':>6} {'ms/fwd':>9} {'vs A':>8}")
                print(f"{'-' * 68}")
                if ra is not None:
                    print(f"{ra['config']:<28} {ra['time_s']:>7.2f}s {ra['fwd']:>6.0f} {ra['ms_fwd']:>8.2f}    —")
                    print(f"  A paths: {ra['controller']['path_counts']}")
                if rb is not None:
                    vs_a = f"{delta_b_pct:+7.1f}%" if delta_b_pct is not None else "n/a"
                    print(f"{rb['config']:<28} {rb['time_s']:>7.2f}s {rb['fwd']:>6.0f} {rb['ms_fwd']:>8.2f} {vs_a:>8}")
                    print(f"  B paths: {rb['controller']['path_counts']}")
                if rc is not None:
                    vs_a = f"{delta_c_pct:+7.1f}%" if delta_c_pct is not None else "n/a"
                    print(f"{rc['config']:<28} {rc['time_s']:>7.2f}s {rc['fwd']:>6.0f} {rc['ms_fwd']:>8.2f} {vs_a:>8}")
                    print(f"  C paths: {rc['controller']['path_counts']}")
                if rd is not None:
                    vs_a = f"{delta_d_pct:+7.1f}%" if delta_d_pct is not None else "n/a"
                    print(f"{rd['config']:<28} {rd['time_s']:>7.2f}s {rd['fwd']:>6.0f} {rd['ms_fwd']:>8.2f} {vs_a:>8}")
                    print(f"  D paths: {rd['controller']['path_counts']}")
                if re is not None:
                    vs_a = f"{delta_e_pct:+7.1f}%" if delta_e_pct is not None else "n/a"
                    print(f"{re['config']:<28} {re['time_s']:>7.2f}s {re['fwd']:>6.0f} {re['ms_fwd']:>8.2f} {vs_a:>8}")
                    print(f"  E paths: {re['controller']['path_counts']}")
                if rg is not None:
                    vs_a = f"{delta_g_pct:+7.1f}%" if delta_g_pct is not None else "n/a"
                    print(f"{rg['config']:<28} {rg['time_s']:>7.2f}s {rg['fwd']:>6.0f} {rg['ms_fwd']:>8.2f} {vs_a:>8}")
                    print(f"  G paths: {rg['controller']['path_counts']}")
                if rg2 is not None:
                    vs_a = f"{delta_g2_pct:+7.1f}%" if delta_g2_pct is not None else "n/a"
                    print(f"{rg2['config']:<28} {rg2['time_s']:>7.2f}s {rg2['fwd']:>6.0f} {rg2['ms_fwd']:>8.2f} {vs_a:>8}")
                    print(f"  G2 paths: {rg2['controller']['path_counts']}")
                if rgs is not None:
                    vs_a = f"{delta_gs_pct:+7.1f}%" if delta_gs_pct is not None else "n/a"
                    print(f"{rgs['config']:<28} {rgs['time_s']:>7.2f}s {rgs['fwd']:>6.0f} {rgs['ms_fwd']:>8.2f} {vs_a:>8}")
                    print(f"  GS paths: {rgs['controller']['path_counts']}")
                if rf is not None:
                    vs_a = f"{delta_f_pct:+7.1f}%" if delta_f_pct is not None else "n/a"
                    print(f"{rf['config']:<28} {rf['time_s']:>7.2f}s {rf['fwd']:>6.0f} {rf['ms_fwd']:>8.2f} {vs_a:>8}")
                    print(f"  F paths: {rf['controller']['path_counts']}")
            results = {
                "batch_size": args.batch_size,
                "local_bs": local_bs,
                "gen_length": args.gen_length,
                "block_length": BLOCK_LENGTH,
                "tp_size": tp_size,
                "dp_size": dp_size,
                "ep_size": world_size,
                "backend": alltoall_backend,
                "profile_target": args.profile_target,
                "config_set": args.config_set,
                "A_baseline": ra,
                "B_bsp_moe": rb,
                "C_bsp_delay_gather": rc,
                "D_bsp_delay_gather_m3_epreduce": rd,
                "E_bsp_cross_layer_sp": re,
                "G_bsp_g_attn_reduce_scatter_sp": rg,
                "G2_bsp_g2_sp_parity_bundle": rg2,
                "GS_bsp_g_source_path": rgs,
                "F_bsp_allreduce_full_probe": rf,
                "delta_B_pct": delta_b_pct,
                "delta_C_pct": delta_c_pct,
                "delta_D_pct": delta_d_pct,
                "delta_E_pct": delta_e_pct,
                "delta_G_pct": delta_g_pct,
                "delta_G2_pct": delta_g2_pct,
                "delta_GS_pct": delta_gs_pct,
                "delta_F_pct": delta_f_pct,
            }
            out_path = REPO_ROOT / "codex_coding" / "results" / "bsp_moe_dp2_results.json"

        if rank == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Saved to {out_path}")

        if JOINT_STATS is not None and JOINT_STATS.records:
            js_path = (
                REPO_ROOT / "codex_coding" / "results"
                / f"joint_stats_rank{rank}.json"
            )
            with open(js_path, "w") as f:
                json.dump(JOINT_STATS.to_dict(), f)
            if rank == 0:
                print(f"  [JOINT-STATS] Saved {len(JOINT_STATS.records)} records "
                      f"to joint_stats_rank*.json")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
