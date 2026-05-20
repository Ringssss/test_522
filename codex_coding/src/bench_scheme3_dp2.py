#!/usr/bin/env python3
"""
Scheme 3: S_mask-Aware Route-Before-Dispatch for C12 (dp=2, tp=4, ep=8).

Skip path (77%): route locally on 256 tokens → all_gatherv(hidden + topk) → experts
Cold/Update (23%): unchanged — dispatch(hidden + logits) → route on 512 → experts

Two configs:
  A) C12-AgRs baseline (vllm 0.11.0, allgather_reducescatter)
  B) C12-Scheme3 (skip path: route-before-dispatch)

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter \
    torchrun --nproc_per_node=8 codex_coding/src/bench_scheme3_dp2.py
"""

from __future__ import annotations
import sys; sys.modules['deep_ep'] = None
import os, time, json, argparse, types
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.distributed as dist

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))
from test_m_skip_sweep import MSkipEBController

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
SCHEME3_TOPK_PAYLOADS = {}


class ComponentTimer:
    """Low-overhead CUDA event collector for rank-max component timing."""

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
            return nullcontext()
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
            total_ms = max(item["ms"].get(name, 0.0) for item in gathered)
            count = max(item["counts"].get(name, 0) for item in gathered)
            components[name] = {
                "total_ms_rankmax": total_ms,
                "ms_per_fwd_rankmax": total_ms / max(fwds, 1),
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


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


class Scheme3MSkipController(MSkipEBController):
    """MSkip controller with explicit block-id clock for robust cold/hot split."""

    def __init__(self, *args, guard_branches=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.guard_branches = guard_branches
        self.current_block_id = -1
        self._last_block_id = {}
        self.guard_fallback_count = 0
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
        self.guard_fallback_count = 0
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

    def predict_skip(self, layer_idx: int) -> bool:
        """Predict skip path without mutating controller state."""
        if self.current_block_id < 0:
            return False
        if self._last_block_id.get(layer_idx, -1) != self.current_block_id:
            return False
        nxt = self._fwd_in_block.get(layer_idx, 0) + 1
        if self.skip_m == float('inf'):
            return layer_idx in self.s_mask_cache
        return (nxt % self.skip_m != 0) and (layer_idx in self.s_mask_cache)

    def mark_guard_fallback(self):
        self.guard_fallback_count += 1

    def get_s_mask(self, layer_idx, logits, bias):
        # During prefill, block_id is not available. Fall back to legacy behavior.
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
            "guard_fallback_count": int(self.guard_fallback_count),
            "per_layer_cold": {
                str(k): int(v) for k, v in sorted(self.per_layer_cold.items())
            },
        }


# ========================================================================
# Scheme 3: skip path forward
# ========================================================================

def make_scheme3_forward(block, layer_id, gp, ctrl, vllm_cfg):
    """
    Patched SparseMoeBlock.forward:
      - skip path: route locally → all_gatherv(hidden + topk) → fused_experts → combine
      - cold/update: fall through to original _moe_forward_with_context
    """
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.distributed import (get_dp_group, get_ep_group,
                                  tensor_model_parallel_all_reduce)
    from vllm.forward_context import set_forward_context, get_forward_context
    from dinfer.model.modeling_llada2_moe import _moe_forward_with_context
    from test_fused_eb_triton import fused_routing

    shared_mod = block.shared_experts if block.config.num_shared_experts else None
    experts = block.experts

    def _sync_skip_decision(local_skip: bool) -> bool:
        if not getattr(ctrl, "guard_branches", False):
            return local_skip
        val = torch.tensor(1 if local_skip else 0,
                           dtype=torch.int32,
                           device=next(block.parameters()).device)
        dp_group = get_dp_group().device_group
        vmin = val.clone()
        vmax = val.clone()
        dist.all_reduce(vmin, op=dist.ReduceOp.MIN, group=dp_group)
        dist.all_reduce(vmax, op=dist.ReduceOp.MAX, group=dp_group)
        if vmin.item() != vmax.item():
            ctrl.mark_guard_fallback()
            return False
        return bool(vmin.item())

    def forward(hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        N = hs_flat.shape[0]

        # Shared expert (always before routed — fused_experts is inplace)
        with COMPONENT_TIMER.time("moe.shared"):
            shared_res = shared_mod(hs_flat) if shared_mod is not None else None

        # Gate logits (local, 256 tokens)
        with COMPONENT_TIMER.time("moe.gate_logits"):
            router_logits = block.gate.get_logits(hs_flat)

        with COMPONENT_TIMER.time("moe.skip_decision"):
            want_skip = ctrl.predict_skip(layer_id)
            do_skip = _sync_skip_decision(want_skip)

        if do_skip:
            # ========== SKIP PATH: route-before-dispatch ==========

            # ① get_s_mask: returns cached s_mask, updates fwd_in_block counter
            with COMPONENT_TIMER.time("moe.get_s_mask"):
                s_mask = ctrl.get_s_mask(layer_id, router_logits, gp["bias"])

            # ② Route locally on 256 tokens (instead of 512)
            with COMPONENT_TIMER.time("moe.local_routing"):
                topk_w, topk_ids = fused_routing(
                    router_logits, gp["bias"], gp["rsf"],
                    s_mask=s_mask, K=4, ng=gp["ng"], tkg=gp["tkg"])
                topk_w = topk_w.to(router_logits.dtype)

            # ③ Dispatch: all_gatherv(hidden + topk_w + topk_ids)
            #    Communication: [256,2048] + [256,4] + [256,4]
            #    vs original:   [256,2048] + [256,256]  → saves 122 KB/rank/layer
            with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                     num_tokens=N):
                ctx = get_forward_context()
                sp_ctx = (ctx.dp_metadata.sp_local_sizes(1)
                          if ctx.dp_metadata else nullcontext())
                with sp_ctx:
                    dp = get_dp_group()
                    sizes = ctx.dp_metadata.get_chunk_sizes_across_dp_rank()

                    topk_ids_payload = topk_ids.to(torch.bfloat16)
                    COMPONENT_TIMER.add_bytes(
                        "dispatch_payload",
                        _tensor_nbytes(hs_flat) + _tensor_nbytes(topk_w)
                        + _tensor_nbytes(topk_ids_payload))
                    with COMPONENT_TIMER.time("moe.dispatch"):
                        g_hs, g_tw, g_ti = dp.all_gatherv(
                            [hs_flat, topk_w, topk_ids_payload],
                            dim=0, sizes=sizes)
                    with COMPONENT_TIMER.time("moe.unpack_topk"):
                        g_ti = g_ti.to(topk_ids.dtype)

                    # ④ fused_experts (same as original path)
                    with COMPONENT_TIMER.time("moe.fused_experts"):
                        y = fused_experts(
                            g_hs, experts.w13_weight, experts.w2_weight,
                            g_tw, g_ti, inplace=True,
                            global_num_experts=experts.global_num_experts,
                            expert_map=experts.expert_map)

                    # ⑤ Combine: reduce_scatterv (same as AgRs combine)
                    with COMPONENT_TIMER.time("moe.combine"):
                        y = dp.reduce_scatterv(y, dim=0, sizes=sizes)

            # ⑥ TP all_reduce (same as FusedMoE reduce_output)
            with COMPONENT_TIMER.time("moe.tp_all_reduce"):
                y = tensor_model_parallel_all_reduce(y)

        else:
            # ========== COLD / HOT-UPDATE: original path ==========
            with COMPONENT_TIMER.time("moe.native_forward"):
                y = _moe_forward_with_context(experts, hs_flat, router_logits)

        if shared_res is not None:
            y = y + shared_res
        return y.view(bsz, seq_len, h)

    return forward


def _pack_topk_payload(topk_w: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    topk_ids_as_w = topk_ids.to(torch.float32).to(topk_w.dtype)
    return torch.cat([topk_w, topk_ids_as_w], dim=-1).contiguous()


def _unpack_topk_payload(payload: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    actual_top_k = payload.shape[-1] // 2
    topk_w = payload[:, :actual_top_k].contiguous()
    topk_ids = payload[:, actual_top_k:actual_top_k * 2].to(torch.int32).contiguous()
    return topk_w, topk_ids


def ensure_native_topk_select_patch():
    """Patch FusedMoE.select_experts so compact topk payloads bypass routing."""
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    if getattr(FusedMoE, "_scheme3_select_patched", False):
        return

    orig_select = FusedMoE.select_experts

    def select_experts_with_precomputed(*args, **kwargs):
        router_logits = kwargs.get("router_logits")
        indices_type = kwargs.get("indices_type")
        if router_logits is None and len(args) >= 2:
            router_logits = args[1]

        payload = SCHEME3_TOPK_PAYLOADS.pop(id(router_logits), None)
        if payload is not None:
            with COMPONENT_TIMER.time("moe.unpack_topk"):
                topk_w, topk_ids = _unpack_topk_payload(payload)
            if indices_type is not None:
                with COMPONENT_TIMER.time("moe.unpack_topk"):
                    topk_ids = topk_ids.to(dtype=indices_type)
            return topk_w, topk_ids, None

        with COMPONENT_TIMER.time("moe.select_experts"):
            return orig_select(*args, **kwargs)

    FusedMoE._scheme3_orig_select_experts = orig_select
    FusedMoE.select_experts = staticmethod(select_experts_with_precomputed)
    FusedMoE._scheme3_select_patched = True


def make_native_topk_forward_impl(experts):
    """
    Patch one FusedMoE instance: skip path dispatches hidden + compact topk,
    then reuses the native quant/expert/combine path.
    """
    from vllm.distributed import get_ep_group
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEModularKernel
    from vllm.forward_context import get_forward_context

    orig_forward_impl = experts.forward_impl

    def forward_impl_native_topk(self, hidden_states, router_logits):
        payload_info = getattr(self, "_scheme3_pending_topk", None)
        if payload_info is None:
            return orig_forward_impl(hidden_states, router_logits)

        topk_w, topk_ids = payload_info
        self._scheme3_pending_topk = None

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
            raise RuntimeError("Scheme3 native-topk only supports the native "
                               "allgather/reducescatter MoE path")

        do_naive_dispatch_combine = self.dp_size > 1
        if not do_naive_dispatch_combine:
            raise RuntimeError("Scheme3 native-topk requires DP dispatch/combine")

        if (not isinstance(self.quant_method.fused_experts,
                           FusedMoEModularKernel)
                and self.shared_experts is not None):
            shared_output = self.shared_experts(hidden_states)
        else:
            shared_output = None

        with COMPONENT_TIMER.time("moe.pack_topk"):
            compact = _pack_topk_payload(topk_w, topk_ids)
        ctx = get_forward_context()
        sp_ctx = ctx.dp_metadata.sp_local_sizes(
            self.sp_size) if ctx.dp_metadata else nullcontext()

        with sp_ctx:
            COMPONENT_TIMER.add_bytes(
                "dispatch_payload",
                _tensor_nbytes(hidden_states) + _tensor_nbytes(compact))
            with COMPONENT_TIMER.time("moe.dispatch"):
                hidden_states, compact = get_ep_group().dispatch(
                    hidden_states, compact, self.is_sequence_parallel)
            SCHEME3_TOPK_PAYLOADS[id(compact)] = compact

            with COMPONENT_TIMER.time("moe.quant_apply"):
                final_hidden_states = self.quant_method.apply(
                    layer=self,
                    x=hidden_states,
                    router_logits=compact,
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

            if shared_output is not None:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None
                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                final_hidden_states, zero_expert_result = final_hidden_states

            def reduce_output(states: torch.Tensor,
                              do_combine: bool = True) -> torch.Tensor:
                if do_naive_dispatch_combine and do_combine:
                    with COMPONENT_TIMER.time("moe.combine"):
                        states = get_ep_group().combine(
                            states, self.is_sequence_parallel)

                if (not self.is_sequence_parallel and self.reduce_results
                        and (self.tp_size > 1 or self.ep_size > 1)):
                    with COMPONENT_TIMER.time("moe.tp_all_reduce"):
                        states = self.maybe_all_reduce_tensor_model_parallel(
                            states)

                return states

            if self.shared_experts is not None:
                return (
                    reduce_output(final_hidden_states[0], do_combine=False),
                    reduce_output(final_hidden_states[1]),
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, torch.Tensor)
                return reduce_output(final_hidden_states) + zero_expert_result
            else:
                return reduce_output(final_hidden_states)

    return types.MethodType(forward_impl_native_topk, experts)


def make_timed_forward_impl(experts):
    """Patch one FusedMoE instance for timing only; behavior stays native."""
    from vllm.distributed import get_ep_group
    from vllm.model_executor.layers.fused_moe.layer import FusedMoEModularKernel
    from vllm.forward_context import get_forward_context

    orig_forward_impl = experts.forward_impl

    def forward_impl_timed(self, hidden_states, router_logits):
        if not COMPONENT_TIMER.enabled:
            return orig_forward_impl(hidden_states, router_logits)

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
            with COMPONENT_TIMER.time("moe.native_forward"):
                return orig_forward_impl(hidden_states, router_logits)

        do_naive_dispatch_combine = self.dp_size > 1
        if not do_naive_dispatch_combine:
            with COMPONENT_TIMER.time("moe.native_forward"):
                return orig_forward_impl(hidden_states, router_logits)

        if (not isinstance(self.quant_method.fused_experts,
                           FusedMoEModularKernel)
                and self.shared_experts is not None):
            shared_output = self.shared_experts(hidden_states)
        else:
            shared_output = None

        ctx = get_forward_context()
        sp_ctx = ctx.dp_metadata.sp_local_sizes(
            self.sp_size) if ctx.dp_metadata else nullcontext()

        with sp_ctx:
            COMPONENT_TIMER.add_bytes(
                "dispatch_payload",
                _tensor_nbytes(hidden_states) + _tensor_nbytes(router_logits))
            with COMPONENT_TIMER.time("moe.dispatch"):
                hidden_states, router_logits = get_ep_group().dispatch(
                    hidden_states, router_logits, self.is_sequence_parallel)

            with COMPONENT_TIMER.time("moe.quant_apply"):
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

            if shared_output is not None:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None
                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                final_hidden_states, zero_expert_result = final_hidden_states

            def reduce_output(states: torch.Tensor,
                              do_combine: bool = True) -> torch.Tensor:
                if do_naive_dispatch_combine and do_combine:
                    with COMPONENT_TIMER.time("moe.combine"):
                        states = get_ep_group().combine(
                            states, self.is_sequence_parallel)

                if (not self.is_sequence_parallel and self.reduce_results
                        and (self.tp_size > 1 or self.ep_size > 1)):
                    with COMPONENT_TIMER.time("moe.tp_all_reduce"):
                        states = self.maybe_all_reduce_tensor_model_parallel(
                            states)

                return states

            if self.shared_experts is not None:
                return (
                    reduce_output(final_hidden_states[0], do_combine=False),
                    reduce_output(final_hidden_states[1]),
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, torch.Tensor)
                return reduce_output(final_hidden_states) + zero_expert_result
            else:
                return reduce_output(final_hidden_states)

    return types.MethodType(forward_impl_timed, experts)


def make_scheme3_native_topk_forward(block, layer_id, gp, ctrl, vllm_cfg):
    """
    Patched SparseMoeBlock.forward:
      - skip path computes local topk and lets FusedMoE.forward_impl dispatch it
      - cold/update path remains the original vLLM MoE path
    """
    from vllm.distributed import get_dp_group
    from vllm.forward_context import set_forward_context
    from test_fused_eb_triton import fused_routing

    shared_mod = block.shared_experts if block.config.num_shared_experts else None
    experts = block.experts

    def _sync_skip_decision(local_skip: bool) -> bool:
        if not getattr(ctrl, "guard_branches", False):
            return local_skip
        val = torch.tensor(1 if local_skip else 0,
                           dtype=torch.int32,
                           device=next(block.parameters()).device)
        dp_group = get_dp_group().device_group
        vmin = val.clone()
        vmax = val.clone()
        dist.all_reduce(vmin, op=dist.ReduceOp.MIN, group=dp_group)
        dist.all_reduce(vmax, op=dist.ReduceOp.MAX, group=dp_group)
        if vmin.item() != vmax.item():
            ctrl.mark_guard_fallback()
            return False
        return bool(vmin.item())

    def forward(hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)

        # Shared expert must run before routed experts because fused_moe is inplace.
        with COMPONENT_TIMER.time("moe.shared"):
            shared_res = shared_mod(hs_flat) if shared_mod is not None else None
        with COMPONENT_TIMER.time("moe.gate_logits"):
            router_logits = block.gate.get_logits(hs_flat)

        with COMPONENT_TIMER.time("moe.skip_decision"):
            want_skip = ctrl.predict_skip(layer_id)
            do_skip = _sync_skip_decision(want_skip)

        if do_skip:
            with COMPONENT_TIMER.time("moe.get_s_mask"):
                s_mask = ctrl.get_s_mask(layer_id, router_logits, gp["bias"])
            with COMPONENT_TIMER.time("moe.local_routing"):
                topk_w, topk_ids = fused_routing(
                    router_logits, gp["bias"], gp["rsf"],
                    s_mask=s_mask, K=4, ng=gp["ng"], tkg=gp["tkg"])
            experts._scheme3_pending_topk = (
                topk_w.to(router_logits.dtype),
                topk_ids,
            )
        else:
            pass

        with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                 num_tokens=hs_flat.shape[0]):
            with COMPONENT_TIMER.time("moe.native_forward"):
                y = experts.forward_impl(
                    hidden_states=hs_flat, router_logits=router_logits)
        if shared_res is not None:
            y = y + shared_res
        return y.view(bsz, seq_len, h)

    return forward


# ========================================================================
# Main
# ========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--num-runs", type=int, default=2)
    parser.add_argument("--component-timing", action="store_true")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size == 8:
        TP_SIZE = 4
    elif world_size == 4:
        # Quality fallback mode when 8-GPU run is blocked by external occupancy.
        TP_SIZE = 2
    else:
        raise AssertionError(f"Requires 8 GPUs (primary) or 4 GPUs (quality fallback), got {world_size}")

    dp_size = world_size // TP_SIZE
    dp_rank = rank // TP_SIZE
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeSparseMoeBlock
    from transformers import AutoTokenizer, AutoConfig
    from test_heteval512 import PROMPTS
    from test_heteval128 import VERIFIABLE
    from test_fused_eb_triton import fused_routing
    from baseline_optimizations import apply_all_optimizations

    # --- Distributed init ---
    pcfg_init = ParallelConfig(tensor_parallel_size=1, data_parallel_size=1,
                               enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(world_size, rank, "env://", local_rank, "nccl")

    pcfg = ParallelConfig(tensor_parallel_size=TP_SIZE, data_parallel_size=dp_size,
                          data_parallel_rank=dp_rank, enable_expert_parallel=True)
    vllm_cfg = VllmConfig(parallel_config=pcfg)

    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(
            tensor_model_parallel_size=TP_SIZE, backend="nccl")

        from vllm.distributed import get_dp_group, get_ep_group

        if rank == 0:
            print("=" * 80)
            print(f"Scheme 3 Benchmark — dp={dp_size} tp={TP_SIZE} ep={world_size}, {world_size} GPUs")
            print(f"  batch={args.batch_size}, gen={args.gen_length}")
            print(f"  component_timing={args.component_timing}, num_runs={args.num_runs}")
            print("=" * 80)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)

        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        from vllm.forward_context import set_forward_context
        with torch.inference_mode():
            warmup_tok = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                     num_tokens=warmup_tok.numel()):
                _ = model(warmup_tok, use_cache=False)

        apply_all_optimizations(model)
        from vllm.distributed import prepare_communication_buffer_for_model
        prepare_communication_buffer_for_model(model)

        if rank == 0:
            print(f"  GPU memory: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

        # --- Collect MoE blocks and gate params ---
        moe_blocks = []
        gate_params = []
        for n, m in model.named_modules():
            if isinstance(m, LLaDA2MoeSparseMoeBlock):
                moe_blocks.append(m)
        for n, m in model.named_modules():
            if m.__class__.__name__ == "LLaDA2MoeGate":
                gate_params.append({"bias": m.expert_bias, "rsf": m.routed_scaling_factor,
                                    "ng": m.n_group, "tkg": m.topk_group})

        # Save original forwards
        orig_forwards = [b.forward for b in moe_blocks]
        orig_expert_forward_impls = [b.experts.forward_impl for b in moe_blocks]

        # --- Build input ---
        local_bs = args.batch_size // dp_size
        all_ids = []
        for i in range(args.batch_size):
            text = PROMPTS[i % len(PROMPTS)]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=False)
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                  if ids.shape[0] < mx else ids for ids in all_ids]
        input_ids_full = torch.stack(padded, dim=0)
        my_input = input_ids_full[dp_rank * local_bs : (dp_rank + 1) * local_bs].to(device)
        prompt_len = my_input.shape[1]

        decoder = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # --- Routing patches ---
        orig_block_init = decoder.block_init

        def setup_block_clock(ctrl_ref):
            def block_init_with_clock(block_x, block_id):
                ctrl_ref.note_block_start(int(block_id))
                return orig_block_init(block_x, block_id)
            decoder.block_init = block_init_with_clock

        def setup_baseline_routing(ctrl_ref):
            setup_block_clock(ctrl_ref)
            gi = 0
            for n, m in model.named_modules():
                if m.__class__.__name__ == "LLaDA2MoeGate":
                    b, r, ng, tkg = m.expert_bias, m.routed_scaling_factor, m.n_group, m.topk_group
                    li = gi
                    def mk(bb, rr, nn, gg, layer_i, cc):
                        def fn(hs, go, topk, renorm):
                            sm = cc.get_s_mask(layer_i, go, bb)
                            w, idx = fused_routing(go, bb, rr, s_mask=sm, K=4, ng=nn, tkg=gg)
                            return w.to(go.dtype), idx
                        return fn
                    m.routing = mk(b, r, ng, tkg, li, ctrl_ref)
                    gi += 1
            for blk, of in zip(moe_blocks, orig_forwards):
                blk.forward = of
            for blk, ofi in zip(moe_blocks, orig_expert_forward_impls):
                blk.experts.forward_impl = ofi
                blk.experts._scheme3_pending_topk = None
            for blk in moe_blocks:
                blk.experts.forward_impl = make_timed_forward_impl(blk.experts)

        def setup_scheme3(ctrl_ref):
            # Gate routing patch still needed for cold/update path
            setup_baseline_routing(ctrl_ref)
            # Override SparseMoeBlock.forward with scheme3 version
            for i, blk in enumerate(moe_blocks):
                blk.forward = make_scheme3_forward(
                    blk, i, gate_params[i], ctrl_ref, vllm_cfg)

        def setup_scheme3_native_topk(ctrl_ref):
            # Gate routing patch still needed for cold/update path
            setup_baseline_routing(ctrl_ref)
            ensure_native_topk_select_patch()
            for i, blk in enumerate(moe_blocks):
                blk.experts.forward_impl = make_native_topk_forward_impl(
                    blk.experts)
                blk.forward = make_scheme3_native_topk_forward(
                    blk, i, gate_params[i], ctrl_ref, vllm_cfg)

        def reset(ctrl_ref):
            ctrl_ref.prev_N.clear(); ctrl_ref.K_init.clear()
            ctrl_ref.cold_count = 0; ctrl_ref.hot_count = 0
            ctrl_ref.eb_calls = 0; ctrl_ref.eb_skips = 0
            ctrl_ref._bufs.clear(); ctrl_ref.k_init_history.clear()
            ctrl_ref.s_mask_cache.clear(); ctrl_ref.pop_cache.clear()
            ctrl_ref._fwd_in_block.clear(); ctrl_ref._block_idx.clear()
            ctrl_ref.reset_block_clock()

        def _print_timing_summary(label, timing_summary):
            if rank != 0 or timing_summary is None:
                return
            components = timing_summary["components"]
            bytes_info = timing_summary["byte_components"]
            print(f"  Component timing ({label}, rank-max ms/fwd):")
            ordered = [
                "moe.shared", "moe.gate_logits", "moe.skip_decision",
                "moe.get_s_mask", "moe.local_routing", "moe.pack_topk",
                "moe.dispatch", "moe.select_experts", "moe.unpack_topk",
                "moe.quant_apply", "moe.fused_experts", "moe.combine",
                "moe.tp_all_reduce", "moe.native_forward",
            ]
            for name in ordered:
                if name not in components:
                    continue
                item = components[name]
                print(f"    {name:<22} {item['ms_per_fwd_rankmax']:>8.3f} "
                      f"ms/fwd  count={item['count_rankmax']}")
            if "dispatch_payload" in bytes_info:
                mb = bytes_info["dispatch_payload"]["mb_per_fwd_rankmax"]
                print(f"    {'dispatch_payload':<22} {mb:>8.3f} MB/fwd")

        def run_config(label, ctrl_ref, setup_fn, num_runs=2):
            if rank == 0:
                print(f"\n--- {label} ---")
            setup_fn(ctrl_ref)

            # Warmup
            reset(ctrl_ref)
            dllm = make_dllm()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(my_input.clone(), gen_length=args.gen_length,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize(); dist.barrier()
            if rank == 0:
                print(f"  Warmup: {dllm.diff_iteration.num_forwards} fwd, "
                      f"cold={ctrl_ref.cold_count} hot={ctrl_ref.hot_count}")

            # Timed
            times, fwds, timing_runs = [], [], []
            for ri in range(num_runs):
                reset(ctrl_ref)
                COMPONENT_TIMER.enabled = bool(args.component_timing)
                COMPONENT_TIMER.reset()
                dllm = make_dllm()
                torch.cuda.synchronize(); dist.barrier()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    out = dllm.generate(my_input.clone(), gen_length=args.gen_length,
                                        block_length=BLOCK_LENGTH)
                torch.cuda.synchronize(); dist.barrier()
                dt = time.perf_counter() - t0
                nf = dllm.diff_iteration.num_forwards
                timing_summary = (COMPONENT_TIMER.summarize_across_ranks(nf)
                                  if args.component_timing else None)
                COMPONENT_TIMER.enabled = False
                times.append(dt); fwds.append(nf)
                if rank == 0 and timing_summary is not None:
                    timing_runs.append(timing_summary)
                if rank == 0:
                    st = ctrl_ref.stats()
                    print(f"  Run {ri+1}: {dt:.3f}s, {nf} fwd, {dt*1000/nf:.2f} ms/fwd, "
                          f"cold={ctrl_ref.cold_count} hot={ctrl_ref.hot_count} "
                          f"eb_skip={ctrl_ref.eb_skips} "
                          f"path={st['path_counts']} guard={st['guard_fallback_count']}")
                    _print_timing_summary(label, timing_summary)

            # Quality: print first 3 outputs
            if rank == 0:
                gen = out[:, prompt_len:]
                print(f"  Quality ({label}):")
                for bi in range(min(3, gen.shape[0])):
                    toks = gen[bi]
                    valid = toks[(toks != 0) & (toks != EOS_ID) & (toks != MASK_ID)]
                    text = tokenizer.decode(valid, skip_special_tokens=True)
                    print(f"    #{bi}: {text[:200]}")

                # Quality (verifiable 5 prompts): PASS/FAIL + text snippet
                print(f"  Verifiable Quality ({label}):")
                for bi in sorted(VERIFIABLE.keys()):
                    if bi >= gen.shape[0]:
                        continue
                    toks = gen[bi]
                    valid = toks[(toks != 0) & (toks != EOS_ID) & (toks != MASK_ID)]
                    text = tokenizer.decode(valid, skip_special_tokens=True)
                    print(f"    #{bi}: {text[:800]}")

            t = sum(times) / len(times)
            f = sum(fwds) / len(fwds)
            return {
                "config": label,
                "time_s": t,
                "fwd": f,
                "ms_fwd": t / f * 1000,
                "controller": ctrl_ref.stats(),
                "component_timing_runs": timing_runs if rank == 0 else [],
            }

        # --- Run both configs ---
        ctrl_a = Scheme3MSkipController(num_layers=19, K=8, M=4, K_target=40,
                                        quality_floor=0.70, q_major=1.0,
                                        per_round_cap=8, skip_m=5, guard_branches=True)
        ctrl_b = Scheme3MSkipController(num_layers=19, K=8, M=4, K_target=40,
                                        quality_floor=0.70, q_major=1.0,
                                        per_round_cap=8, skip_m=5, guard_branches=True)
        ctrl_b2 = Scheme3MSkipController(num_layers=19, K=8, M=4, K_target=40,
                                         quality_floor=0.70, q_major=1.0,
                                         per_round_cap=8, skip_m=5, guard_branches=True)

        ra = run_config("A) C12-AgRs baseline", ctrl_a, setup_baseline_routing,
                        num_runs=args.num_runs)
        rb = run_config("B) C12-Scheme3 (route-before-dispatch)", ctrl_b, setup_scheme3,
                        num_runs=args.num_runs)
        rb2 = run_config("B2) C12-Scheme3-native-topk", ctrl_b2, setup_scheme3_native_topk,
                         num_runs=args.num_runs)

        # --- Summary ---
        if rank == 0:
            ms_a, ms_b, ms_b2 = ra["ms_fwd"], rb["ms_fwd"], rb2["ms_fwd"]
            delta = (ms_b / ms_a - 1) * 100
            delta_b2 = (ms_b2 / ms_a - 1) * 100

            print(f"\n{'='*70}")
            print(f"SUMMARY — batch={args.batch_size}, gen={args.gen_length}")
            print(f"{'='*70}")
            print(f"{'Config':<40} {'Time':>7} {'Fwd':>5} {'ms/fwd':>8} {'vs A':>8}")
            print(f"{'-'*68}")
            print(f"{ra['config']:<40} {ra['time_s']:>6.2f}s {ra['fwd']:>5.0f} {ms_a:>7.2f}    —")
            print(f"{rb['config']:<40} {rb['time_s']:>6.2f}s {rb['fwd']:>5.0f} {ms_b:>7.2f} {delta:>+7.1f}%")
            print(f"{rb2['config']:<40} {rb2['time_s']:>6.2f}s {rb2['fwd']:>5.0f} {ms_b2:>7.2f} {delta_b2:>+7.1f}%")
            print(f"  A paths: {ra['controller']['path_counts']} guard={ra['controller']['guard_fallback_count']}")
            print(f"  B paths: {rb['controller']['path_counts']} guard={rb['controller']['guard_fallback_count']}")
            print(f"  B2 paths: {rb2['controller']['path_counts']} guard={rb2['controller']['guard_fallback_count']}")
            print(f"\n  Ref: C12 original (naive, vllm 0.10.2) = 74.76 ms/fwd")

            results = {
                "batch_size": args.batch_size, "gen_length": args.gen_length,
                "A_baseline": ra, "B_scheme3": rb,
                "B2_scheme3_native_topk": rb2,
                "delta_pct": delta,
                "delta_b2_pct": delta_b2,
            }
            out_path = REPO_ROOT / "codex_coding" / "results" / "scheme3_dp2_results.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Saved to {out_path}")

    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
