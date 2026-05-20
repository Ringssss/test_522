# coding=utf-8
# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BailingMoE model."""

import math
import warnings
from contextlib import contextmanager, nullcontext
from typing import List, Optional, Tuple, Union, Literal
import os

import torch
import torch.nn.functional as F
from torch import nn
import tqdm

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
try:
    from transformers.utils.import_utils import is_torch_fx_available
except ImportError:
    def is_torch_fx_available():
        return hasattr(torch, "fx")
from .configuration_llada2_moe import LLaDA2MoeConfig
from torch.nn.modules.normalization import RMSNorm
import torch.distributed as dist
from ..decoding.utils import KVCache
from transformers.generation.utils import GenerationMixin
from dataclasses import dataclass
from transformers.utils import ModelOutput

from pathlib import Path
import json
from safetensors.torch import load_file
from functools import partial
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.distributed.eplb.eplb_state import EplbState
import re
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                        ReplicatedLinear,
                        QKVParallelLinear,
                        RowParallelLinear)

# NOTE: shared expert overlap was attempted (v0.1.15.10) but is a negative
# optimization on 4-GPU NVSwitch (MK async overhead > overlap gain).
# FusedMoE.shared_experts property patch is disabled.
# See progress_diff_summary.md for details.
def torch_all_reduce(tensor):
    torch.distributed.all_reduce(tensor)
    return tensor
import vllm.distributed as vllm_distributed
# Only monkey-patch TP all_reduce when tp_size > 1 (AllReduce EP mode).
# With dp > 1 (AllToAll EP), tp_size=1 and the patch would use the wrong group.
def _maybe_patch_all_reduce():
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
        tp_ws = get_tensor_model_parallel_world_size()
        if tp_ws > 1:
            vllm_distributed.tensor_model_parallel_all_reduce = torch_all_reduce
    except Exception as e:
        # If distributed is not initialized, do NOT patch — it's safer to leave
        # the default (which uses TP group and is no-op when tp=1).
        import logging
        logging.getLogger(__name__).warning(
            f"_maybe_patch_all_reduce: skipping patch due to: {e}")

from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)


_BSP_G_COMPONENT_TIMER = None
_BSP_G_LAYOUT_RECORDER = None
_EPLB_RUNTIME_MAP_PATCH_STATE = {
    "installed": False,
    "orig_fn": None,
    "record_mode": "full",
    "map_impl": "vllm",
    "tensor_cache": "on",
    "path": "unknown",
    "path_is_cold": False,
    "path_is_hot_skip": False,
    # Cache for identity-mapping passthrough detection:
    # key=(map_sig, lrc_sig) -> bool
    "identity_passthrough_cache": {},
    "diag_identity_checks": 0,
    "diag_identity_hits": 0,
    "diag_identity_cache_hit": 0,
    "diag_identity_cache_miss": 0,
    "diag_identity_true": 0,
    "diag_identity_false": 0,
    "diag_map_only_calls": 0,
    "diag_record_calls": 0,
    "diag_twochoice_total": 0,
    "diag_twochoice_multi": 0,
    "diag_twochoice_single": 0,
    "diag_twochoice_lb_applied": 0,
    "diag_twochoice_decay_calls": 0,
    "diag_twochoice_update_calls": 0,
}
_EPLB_RUNTIME_TENSOR_CACHE = {
    "pos_indices": {},
    "ones": {},
    "twochoice_state": {},
}


def _eplb_runtime_env_int(name: str, default: int) -> int:
    try:
        raw = os.environ.get(name, "").strip()
        if raw == "":
            return int(default)
        return int(raw)
    except Exception:
        return int(default)


def _eplb_runtime_fastpath_enabled() -> bool:
    raw = os.environ.get("DINF_EPLB_RUNTIME_FASTPATH", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _eplb_runtime_identity_fastpath_enabled() -> bool:
    raw = os.environ.get("DINF_EPLB_RUNTIME_IDENTITY_FASTPATH", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _eplb_runtime_current_path() -> str:
    return str(_EPLB_RUNTIME_MAP_PATCH_STATE.get("path", "unknown"))


def get_eplb_runtime_map_diag(reset: bool = False) -> dict:
    st = _EPLB_RUNTIME_MAP_PATCH_STATE
    out = {
        "identity_checks": int(st.get("diag_identity_checks", 0)),
        "identity_hits": int(st.get("diag_identity_hits", 0)),
        "identity_cache_hit": int(st.get("diag_identity_cache_hit", 0)),
        "identity_cache_miss": int(st.get("diag_identity_cache_miss", 0)),
        "identity_true": int(st.get("diag_identity_true", 0)),
        "identity_false": int(st.get("diag_identity_false", 0)),
        "map_only_calls": int(st.get("diag_map_only_calls", 0)),
        "record_calls": int(st.get("diag_record_calls", 0)),
        "twochoice_total": int(st.get("diag_twochoice_total", 0)),
        "twochoice_multi": int(st.get("diag_twochoice_multi", 0)),
        "twochoice_single": int(st.get("diag_twochoice_single", 0)),
        "twochoice_lb_applied": int(st.get("diag_twochoice_lb_applied", 0)),
        "twochoice_decay_calls": int(st.get("diag_twochoice_decay_calls", 0)),
        "twochoice_update_calls": int(st.get("diag_twochoice_update_calls", 0)),
        "cache_entries": int(len(st.get("identity_passthrough_cache", {}))),
    }
    if reset:
        st["diag_identity_checks"] = 0
        st["diag_identity_hits"] = 0
        st["diag_identity_cache_hit"] = 0
        st["diag_identity_cache_miss"] = 0
        st["diag_identity_true"] = 0
        st["diag_identity_false"] = 0
        st["diag_map_only_calls"] = 0
        st["diag_record_calls"] = 0
        st["diag_twochoice_total"] = 0
        st["diag_twochoice_multi"] = 0
        st["diag_twochoice_single"] = 0
        st["diag_twochoice_lb_applied"] = 0
        st["diag_twochoice_decay_calls"] = 0
        st["diag_twochoice_update_calls"] = 0
        st["identity_passthrough_cache"].clear()
    return out


def set_eplb_runtime_route_path(path: str):
    p = str(path)
    st = _EPLB_RUNTIME_MAP_PATCH_STATE
    st["path"] = p
    st["path_is_cold"] = (p == "cold")
    st["path_is_hot_skip"] = (p == "hot_skip")


def set_eplb_runtime_tensor_cache(mode: str = "on"):
    mode_s = str(mode).strip().lower()
    if mode_s not in {"on", "off"}:
        raise ValueError(f"Unsupported EPLB tensor_cache mode: {mode}")
    _EPLB_RUNTIME_MAP_PATCH_STATE["tensor_cache"] = mode_s
    # Avoid stale tensors when switching modes or across large-shape changes.
    _EPLB_RUNTIME_TENSOR_CACHE["pos_indices"].clear()
    _EPLB_RUNTIME_TENSOR_CACHE["ones"].clear()
    _EPLB_RUNTIME_MAP_PATCH_STATE["identity_passthrough_cache"].clear()


def _eplb_runtime_tensor_cache_enabled() -> bool:
    return _EPLB_RUNTIME_MAP_PATCH_STATE.get("tensor_cache", "on") == "on"


def _runtime_cache_device_key(device: torch.device) -> tuple[str, int]:
    if device.type == "cuda":
        idx = int(device.index if device.index is not None else torch.cuda.current_device())
        return ("cuda", idx)
    return (str(device.type), -1)


def _runtime_tensor_signature(t: torch.Tensor) -> tuple:
    """Build a cheap tensor signature for runtime cache invalidation."""
    return (
        _runtime_cache_device_key(t.device),
        str(t.dtype),
        tuple(int(x) for x in t.shape),
        int(t.data_ptr()),
        int(getattr(t, "_version", -1)),
    )


def _eplb_runtime_can_passthrough_identity_map(
    *,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
) -> bool:
    """Return True when runtime map is effectively identity with replica_count=1.

    In this case map(topk_ids) == topk_ids, so map computation can be bypassed.
    The expensive value check is cached by tensor signature and only recomputed
    when map/replica tensors are updated in place.
    """
    if not _eplb_runtime_identity_fastpath_enabled():
        return False

    st = _EPLB_RUNTIME_MAP_PATCH_STATE
    st["diag_identity_checks"] = int(st.get("diag_identity_checks", 0)) + 1
    cache = st.get("identity_passthrough_cache")
    if not isinstance(cache, dict):
        cache = {}
        st["identity_passthrough_cache"] = cache

    key = (
        _runtime_tensor_signature(logical_to_physical_map),
        _runtime_tensor_signature(logical_replica_count),
    )
    hit = cache.get(key)
    if hit is not None:
        st["diag_identity_cache_hit"] = int(st.get("diag_identity_cache_hit", 0)) + 1
        if hit:
            st["diag_identity_true"] = int(st.get("diag_identity_true", 0)) + 1
        else:
            st["diag_identity_false"] = int(st.get("diag_identity_false", 0)) + 1
        return bool(hit)

    st["diag_identity_cache_miss"] = int(st.get("diag_identity_cache_miss", 0)) + 1
    enabled = False
    try:
        # Support either per-layer views [E, slots]/[E] or layered
        # tensors [L, E, slots]/[L, E].
        if logical_to_physical_map.ndim == 2:
            l2p = logical_to_physical_map.unsqueeze(0)
        elif logical_to_physical_map.ndim == 3:
            l2p = logical_to_physical_map
        else:
            l2p = None

        if logical_replica_count.ndim == 1:
            lrc = logical_replica_count.unsqueeze(0)
        elif logical_replica_count.ndim == 2:
            lrc = logical_replica_count
        else:
            lrc = None

        if (
            l2p is not None
            and lrc is not None
            and int(l2p.shape[-2]) == int(lrc.shape[-1])
        ):
            num_layers = int(l2p.shape[0])
            num_logical = int(l2p.shape[-2])
            if int(lrc.shape[0]) == num_layers and num_logical > 0:
                ref = torch.arange(
                    num_logical, device=l2p.device, dtype=l2p.dtype
                ).view(1, -1).expand(num_layers, -1)
                map_ok = torch.equal(l2p[..., 0], ref)
                lrc_ok = bool((lrc == 1).all().item())
                enabled = bool(map_ok and lrc_ok)
    except Exception:
        enabled = False

    if len(cache) >= 512:
        cache.clear()
    cache[key] = bool(enabled)
    if enabled:
        st["diag_identity_true"] = int(st.get("diag_identity_true", 0)) + 1
    else:
        st["diag_identity_false"] = int(st.get("diag_identity_false", 0)) + 1
    return bool(enabled)


def _eplb_runtime_maybe_identity_passthrough(
    *,
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    indices_type: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    """Return mapped ids when identity fast path can bypass runtime map."""
    if not _eplb_runtime_can_passthrough_identity_map(
        logical_to_physical_map=logical_to_physical_map,
        logical_replica_count=logical_replica_count,
    ):
        return None
    st = _EPLB_RUNTIME_MAP_PATCH_STATE
    st["diag_identity_hits"] = int(st.get("diag_identity_hits", 0)) + 1
    if indices_type is None or topk_ids.dtype == indices_type:
        return topk_ids
    return topk_ids.to(dtype=indices_type)


def _get_cached_pos_indices_like(ref: torch.Tensor) -> torch.Tensor:
    if not _eplb_runtime_tensor_cache_enabled():
        return torch.arange(ref.numel(), device=ref.device, dtype=torch.long).reshape_as(ref)
    key = (_runtime_cache_device_key(ref.device), int(ref.numel()))
    cache = _EPLB_RUNTIME_TENSOR_CACHE["pos_indices"].get(key)
    if cache is None or cache.device != ref.device or cache.numel() != ref.numel():
        cache = torch.arange(ref.numel(), device=ref.device, dtype=torch.long)
        _EPLB_RUNTIME_TENSOR_CACHE["pos_indices"][key] = cache
    return cache.reshape_as(ref)


def _get_cached_ones(
    *,
    numel: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not _eplb_runtime_tensor_cache_enabled():
        return torch.ones((numel,), device=device, dtype=dtype)
    key = (_runtime_cache_device_key(device), int(numel), str(dtype))
    cache = _EPLB_RUNTIME_TENSOR_CACHE["ones"].get(key)
    if cache is None or cache.device != device or cache.numel() != numel or cache.dtype != dtype:
        cache = torch.ones((numel,), device=device, dtype=dtype)
        _EPLB_RUNTIME_TENSOR_CACHE["ones"][key] = cache
    return cache


def _eplb_runtime_map_core_vllm_eager(
    *,
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
) -> torch.Tensor:
    topk_ids_long = topk_ids.long()
    replica_count = logical_replica_count[topk_ids_long]
    replica_count = torch.clamp(replica_count, min=1)
    pos_indices = _get_cached_pos_indices_like(topk_ids_long)
    replica_indices = torch.remainder(pos_indices, replica_count).unsqueeze(-1)
    return logical_to_physical_map[topk_ids_long].gather(-1, replica_indices).squeeze(-1)


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _eplb_triton_map_kernel(
        topk_ids_ptr,
        logical_replica_count_ptr,
        logical_to_physical_ptr,
        out_ids_ptr,
        out_ptr,
        record_enabled_val,
        num_logical_experts,
        map_slots,
        out_size,
        numel,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < numel
        expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int64)
        valid_expert = (expert_id >= 0) & (expert_id < num_logical_experts)
        safe_expert_id = tl.where(valid_expert, expert_id, 0)
        replica_count = tl.load(
            logical_replica_count_ptr + safe_expert_id,
            mask=mask & valid_expert, other=1,
        )
        replica_count = tl.maximum(replica_count, 1)
        replica_idx = offs % replica_count
        map_index = safe_expert_id * map_slots + replica_idx
        physical_id = tl.load(
            logical_to_physical_ptr + map_index,
            mask=mask & valid_expert, other=-1,
        )
        tl.store(out_ids_ptr + offs, physical_id, mask=mask)
        if record_enabled_val > 0:
            valid_rec = mask & (physical_id >= 0) & (physical_id < out_size)
            safe_pid = tl.where(physical_id >= 0, physical_id, 0)
            tl.atomic_add(out_ptr + safe_pid, 1, mask=valid_rec)

    _EPLB_TRITON_AVAILABLE = True
except ImportError:
    _EPLB_TRITON_AVAILABLE = False


def _eplb_runtime_map_core_triton_fused(
    *,
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    expert_load_view: Optional[torch.Tensor] = None,
    record_enabled: bool = False,
) -> torch.Tensor:
    """Fused Triton kernel for logical->physical mapping + optional recording.
    Adapted from vllm/model_executor/layers/fused_moe/router/base_router.py."""
    import triton
    import triton.language as tl

    topk_ids_in = topk_ids.contiguous().to(dtype=torch.int32)
    numel = topk_ids_in.numel()
    if numel == 0:
        return topk_ids
    out_flat = torch.empty((numel,), device=topk_ids.device, dtype=topk_ids.dtype)
    if expert_load_view is None:
        expert_load_view = torch.zeros(
            (int(logical_to_physical_map.max().item()) + 1,),
            device=topk_ids.device, dtype=torch.int32,
        )
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    _eplb_triton_map_kernel[grid](
        topk_ids_in,
        logical_replica_count.contiguous(),
        logical_to_physical_map.contiguous(),
        out_flat,
        expert_load_view.contiguous(),
        1 if record_enabled else 0,
        int(logical_replica_count.shape[0]),
        int(logical_to_physical_map.shape[1]),
        int(expert_load_view.shape[0]),
        numel,
        BLOCK_SIZE=256,
    )
    return out_flat.reshape(topk_ids.shape)


class _EplbNativeMapper:
    """Lightweight callable for EPLB logical→physical mapping.

    Scheme B: for replica_count==1 experts (majority), physical_id == logical_id
    (identity, no map needed). Only multi-replica experts need the modulo lookup.
    Uses a precomputed slot-0 table for the fast path.
    """
    __slots__ = (
        "_st", "_use_triton", "_slot0_map", "_has_multi_replica",
        "_compute_calls",
    )

    def __init__(self, st: dict, use_triton: bool):
        self._st = st
        self._use_triton = use_triton and _EPLB_TRITON_AVAILABLE
        self._slot0_map: Optional[torch.Tensor] = None
        self._has_multi_replica: Optional[bool] = None
        self._compute_calls = 0

    def _ensure_slot0_map(
        self, logical_to_physical_map: torch.Tensor, logical_replica_count: torch.Tensor
    ):
        if self._slot0_map is not None:
            return
        self._slot0_map = logical_to_physical_map[:, 0].contiguous()
        self._has_multi_replica = bool((logical_replica_count > 1).any().item())

    def __call__(
        self,
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        indices_type: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        st = self._st
        is_cold = st["path_is_cold"]
        self._compute_calls += 1

        self._ensure_slot0_map(logical_to_physical_map, logical_replica_count)

        if not self._has_multi_replica:
            pids = self._slot0_map[topk_ids.long()]
        elif self._use_triton:
            pids = _eplb_runtime_map_core_triton_fused(
                topk_ids=topk_ids,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
                expert_load_view=expert_load_view,
                record_enabled=is_cold,
            )
        else:
            pids = _eplb_runtime_map_core_flat_eager(
                topk_ids=topk_ids,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

        if is_cold and not (self._use_triton and self._has_multi_replica):
            st["diag_record_calls"] = int(st.get("diag_record_calls", 0)) + 1
            _eplb_runtime_record_load(
                expert_load_view=expert_load_view, physical_ids=pids
            )

        if indices_type is not None:
            pids = pids.to(dtype=indices_type)
        return pids


def _eplb_runtime_map_core_flat_eager(
    *,
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
) -> torch.Tensor:
    topk_ids_long = topk_ids.long()
    replica_count = logical_replica_count[topk_ids_long]
    replica_count = torch.clamp(replica_count, min=1)
    pos_indices = _get_cached_pos_indices_like(topk_ids_long)
    replica_indices = torch.remainder(pos_indices, replica_count)
    slots = int(logical_to_physical_map.shape[-1])
    flat_map = logical_to_physical_map.reshape(-1)
    flat_idx = topk_ids_long * slots + replica_indices
    return flat_map[flat_idx]


_FLAT_EAGER_TIMER: dict = {"total_us": 0.0, "calls": 0, "enabled": False}


def _eplb_runtime_map_core_flat_eager_timed(
    *,
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
) -> torch.Tensor:
    import time
    t0 = time.perf_counter_ns()
    result = _eplb_runtime_map_core_flat_eager(
        topk_ids=topk_ids,
        logical_to_physical_map=logical_to_physical_map,
        logical_replica_count=logical_replica_count,
    )
    t1 = time.perf_counter_ns()
    _FLAT_EAGER_TIMER["total_us"] += (t1 - t0) / 1000.0
    _FLAT_EAGER_TIMER["calls"] += 1
    return result


def _twochoice_state_key(
    *,
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
) -> tuple:
    return (
        _runtime_cache_device_key(topk_ids.device),
        tuple(int(x) for x in topk_ids.shape),
        int(logical_to_physical_map.shape[0]),
        int(logical_to_physical_map.shape[-1]),
        int(logical_to_physical_map.data_ptr()),
    )


def _get_twochoice_state(
    *,
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
) -> dict:
    cache = _EPLB_RUNTIME_TENSOR_CACHE["twochoice_state"]
    key = _twochoice_state_key(
        topk_ids=topk_ids, logical_to_physical_map=logical_to_physical_map
    )
    st = cache.get(key)
    if st is not None:
        return st

    num_physical = int(logical_to_physical_map.max().item()) + 1
    nr_env = os.environ.get("DINF_EPLB_TWOCHOICE_NUM_RANKS", "").strip()
    try:
        num_ranks = int(nr_env) if nr_env else 8
    except Exception:
        num_ranks = 8
    num_ranks = max(1, num_ranks)
    rank_load = torch.zeros((num_ranks,), device=topk_ids.device, dtype=torch.float32)
    replica_load = torch.zeros((num_physical,), device=topk_ids.device, dtype=torch.float32)
    phys2rank = _physical_to_rank_ids(
        num_physical=num_physical, num_ranks=num_ranks, device=topk_ids.device
    )
    st = {
        "replica_load": replica_load,
        "rank_load": rank_load,
        "num_physical": num_physical,
        "num_ranks": num_ranks,
        "phys2rank": phys2rank,
    }
    cache[key] = st
    return st


def _infer_num_ranks_from_l2p(
    *,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
) -> int:
    # For our current ON32 path (EP=8, local physical slots contiguous),
    # num_ranks can be inferred from max replica count + base slots relation.
    # Fallback conservatively to EP=8-like guess when ambiguous.
    try:
        num_physical = int(logical_to_physical_map.max().item()) + 1
        max_rep = int(torch.clamp(logical_replica_count, min=1).max().item())
        if max_rep <= 0:
            return 8
        # Keep a bounded reasonable guess.
        for cand in (8, 4, 16, 2):
            if num_physical % cand == 0:
                return cand
        return 8
    except Exception:
        return 8


def _physical_to_rank_ids(
    *,
    num_physical: int,
    num_ranks: int,
    device: torch.device,
) -> torch.Tensor:
    num_ranks = max(1, int(num_ranks))
    base = num_physical // num_ranks
    rem = num_physical % num_ranks
    out = torch.empty((num_physical,), device=device, dtype=torch.long)
    start = 0
    for r in range(num_ranks):
        count = base + (1 if r < rem else 0)
        end = start + count
        out[start:end] = int(r)
        start = end
    return out


def _eplb_runtime_map_core_twochoice_lb(
    *,
    topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    path_is_cold: bool,
) -> torch.Tensor:
    # Leverage dLLM temporal locality: keep hot path minimal and do heavier
    # load-aware selection only on cold boundaries by default.
    if (
        _eplb_runtime_env_int("DINF_EPLB_TWOCHOICE_COLD_ONLY", 1) != 0
        and not path_is_cold
    ):
        return _eplb_runtime_map_core_flat_eager(
            topk_ids=topk_ids,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
        )

    topk_ids_long = topk_ids.long()
    ids_flat = topk_ids_long.reshape(-1)
    replica_count = torch.clamp(logical_replica_count[ids_flat], min=1)
    l2p_flat = logical_to_physical_map[ids_flat]
    # Fast path for the majority with single replica.
    phys_flat = l2p_flat[:, 0].clone()
    multi_mask = replica_count > 1
    st_diag = _EPLB_RUNTIME_MAP_PATCH_STATE
    total_n = int(ids_flat.numel())
    multi_n = int(multi_mask.sum().item())
    st_diag["diag_twochoice_total"] = int(st_diag.get("diag_twochoice_total", 0)) + total_n
    st_diag["diag_twochoice_multi"] = int(st_diag.get("diag_twochoice_multi", 0)) + multi_n
    st_diag["diag_twochoice_single"] = int(st_diag.get("diag_twochoice_single", 0)) + (total_n - multi_n)
    if bool(torch.any(multi_mask)):
        pos_flat = torch.arange(ids_flat.numel(), device=ids_flat.device, dtype=torch.long)
        rc_multi = replica_count[multi_mask]
        pos_multi = pos_flat[multi_mask]
        l2p_multi = l2p_flat[multi_mask]
        h1 = torch.remainder(pos_multi, rc_multi)
        h2 = torch.remainder(
            (pos_multi * 1315423911 + 2654435761) ^ (pos_multi >> 1), rc_multi
        )
        row_idx = torch.arange(l2p_multi.shape[0], device=l2p_multi.device, dtype=torch.long)
        p1 = l2p_multi[row_idx, h1]
        p2 = l2p_multi[row_idx, h2]

        state = _get_twochoice_state(topk_ids=topk_ids, logical_to_physical_map=logical_to_physical_map)
        replica_load = state["replica_load"]
        rank_load = state["rank_load"]
        phys2rank = state["phys2rank"]
        r1 = phys2rank[p1.long()]
        r2 = phys2rank[p2.long()]
        # For hot path, keep a cheap and deterministic select; only use load-aware
        # decision and counter update on cold boundaries to leverage temporal locality.
        use_lb_on_hot = _eplb_runtime_env_int("DINF_EPLB_TWOCHOICE_HOT_USE_LB", 0) != 0
        if use_lb_on_hot or path_is_cold:
            c1 = replica_load[p1.long()] + rank_load[r1.long()]
            c2 = replica_load[p2.long()] + rank_load[r2.long()]
            choose_p2 = c2 < c1
            phys_multi = torch.where(choose_p2, p2, p1)
            _EPLB_RUNTIME_MAP_PATCH_STATE["diag_twochoice_lb_applied"] = int(
                _EPLB_RUNTIME_MAP_PATCH_STATE.get("diag_twochoice_lb_applied", 0)
            ) + int(multi_n)
        else:
            choose_p2 = h2 < h1
            phys_multi = torch.where(choose_p2, p2, p1)
        phys_flat[multi_mask] = phys_multi

        # Decay + counter update on cold path (default), or always when explicitly enabled.
        update_on_hot = _eplb_runtime_env_int("DINF_EPLB_TWOCHOICE_HOT_UPDATE", 0) != 0
        if path_is_cold:
            decay = float(os.environ.get("DINF_EPLB_TWOCHOICE_DECAY", "0.5"))
            if decay <= 0.0 or decay > 1.0:
                decay = 0.5
            replica_load.mul_(decay)
            rank_load.mul_(decay)
            _EPLB_RUNTIME_MAP_PATCH_STATE["diag_twochoice_decay_calls"] = int(
                _EPLB_RUNTIME_MAP_PATCH_STATE.get("diag_twochoice_decay_calls", 0)
            ) + 1
        if path_is_cold or update_on_hot:
            ones_rep = torch.ones_like(
                phys_multi, dtype=replica_load.dtype, device=phys_multi.device
            )
            replica_load.scatter_add_(0, phys_multi.long(), ones_rep)
            rank_multi = phys2rank[phys_multi.long()]
            ones_rank = torch.ones_like(
                rank_multi, dtype=rank_load.dtype, device=rank_multi.device
            )
            rank_load.scatter_add_(0, rank_multi.long(), ones_rank)
            _EPLB_RUNTIME_MAP_PATCH_STATE["diag_twochoice_update_calls"] = int(
                _EPLB_RUNTIME_MAP_PATCH_STATE.get("diag_twochoice_update_calls", 0)
            ) + 1
    else:
        # No multi-replica token this call: only decay on cold for stability.
        state = _get_twochoice_state(topk_ids=topk_ids, logical_to_physical_map=logical_to_physical_map)
        if path_is_cold:
            decay = float(os.environ.get("DINF_EPLB_TWOCHOICE_DECAY", "0.5"))
            if decay <= 0.0 or decay > 1.0:
                decay = 0.5
            state["replica_load"].mul_(decay)
            state["rank_load"].mul_(decay)
            _EPLB_RUNTIME_MAP_PATCH_STATE["diag_twochoice_decay_calls"] = int(
                _EPLB_RUNTIME_MAP_PATCH_STATE.get("diag_twochoice_decay_calls", 0)
            ) + 1

    physical_ids = phys_flat.reshape_as(topk_ids_long)
    return physical_ids


def _eplb_runtime_record_load(
    *,
    expert_load_view: torch.Tensor,
    physical_ids: torch.Tensor,
):
    ids_flat = physical_ids.flatten()
    ones = _get_cached_ones(
        numel=int(ids_flat.numel()),
        device=ids_flat.device,
        dtype=expert_load_view.dtype,
    )
    expert_load_view.scatter_add_(
        dim=0,
        index=ids_flat.long(),
        src=ones,
    )


def configure_eplb_runtime_map_policy(
    *,
    record_mode: str = "full",
    map_impl: str = "vllm",
    tensor_cache: str = "off",
):
    """Configure EPLB runtime map/record behavior on top of vLLM FusedMoE.

    record_mode:
      - full: map + record (baseline)
      - cold_only: record only when current path is cold
      - off: map only, no record
    map_impl:
      - vllm: baseline gather-style eager map
      - flat_eager: flattened-index eager map
    """
    if record_mode not in {"full", "cold_only", "off"}:
        raise ValueError(f"Unsupported EPLB record_mode: {record_mode}")
    if map_impl not in {"vllm", "flat_eager", "triton_fused", "native_class", "two_choice_lb"}:
        raise ValueError(f"Unsupported EPLB map_impl: {map_impl}")
    if tensor_cache not in {"on", "off"}:
        raise ValueError(f"Unsupported EPLB tensor_cache mode: {tensor_cache}")

    from vllm.model_executor.layers.fused_moe import layer as fused_layer

    st = _EPLB_RUNTIME_MAP_PATCH_STATE
    if st["installed"] and st["orig_fn"] is not None:
        fused_layer.eplb_map_to_physical_and_record = st["orig_fn"]
        st["installed"] = False
        st["orig_fn"] = None

    st["record_mode"] = str(record_mode)
    st["map_impl"] = str(map_impl)
    set_eplb_runtime_tensor_cache(str(tensor_cache))

    # Full + vllm is exactly baseline; no monkey patch needed.
    if record_mode == "full" and map_impl == "vllm":
        return

    orig = fused_layer.eplb_map_to_physical_and_record

    def _map_only(
        *,
        topk_ids: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        indices_type: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        st["diag_map_only_calls"] = int(st.get("diag_map_only_calls", 0)) + 1
        impl = st["map_impl"]
        if impl == "vllm":
            physical_ids = _eplb_runtime_map_core_vllm_eager(
                topk_ids=topk_ids,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )
        elif impl == "flat_eager":
            physical_ids = _eplb_runtime_map_core_flat_eager(
                topk_ids=topk_ids,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )
        elif impl == "two_choice_lb":
            physical_ids = _eplb_runtime_map_core_twochoice_lb(
                topk_ids=topk_ids,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
                path_is_cold=bool(st.get("path_is_cold", False)),
            )
        else:
            raise ValueError(f"Unknown EPLB map_impl at runtime: {impl}")
        if indices_type is not None:
            physical_ids = physical_ids.to(dtype=indices_type)
        return physical_ids

    # Native class mapper: no closure, no identity check, hot_skip cache built in.
    if (
        _eplb_runtime_fastpath_enabled()
        and record_mode == "cold_only"
        and map_impl == "native_class"
        and tensor_cache == "off"
    ):
        use_triton = _EPLB_TRITON_AVAILABLE
        mapper = _EplbNativeMapper(st=st, use_triton=use_triton)
        fused_layer.eplb_map_to_physical_and_record = mapper
        st["installed"] = True
        st["orig_fn"] = orig
        return

    # Noop map for overhead isolation profiling.
    if os.environ.get("DINF_EPLB_MAP_NOOP", "").strip() == "1":
        def _patched_noop(
            topk_ids: torch.Tensor,
            expert_load_view: torch.Tensor,
            logical_to_physical_map: torch.Tensor,
            logical_replica_count: torch.Tensor,
            indices_type: Optional[torch.dtype] = None,
        ) -> torch.Tensor:
            return topk_ids if indices_type is None else topk_ids.to(dtype=indices_type)
        fused_layer.eplb_map_to_physical_and_record = _patched_noop
        st["installed"] = True
        st["orig_fn"] = orig
        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            print("[EPLB] MAP NOOP mode enabled (overhead isolation)")
        return

    # Timed map for profiling per-call cost.
    _use_timed = os.environ.get("DINF_EPLB_MAP_TIMED", "").strip() == "1"
    if _use_timed:
        _FLAT_EAGER_TIMER["enabled"] = True

    # Fast path for triton_fused + cold_only: single kernel does map+record.
    if (
        _eplb_runtime_fastpath_enabled()
        and record_mode == "cold_only"
        and map_impl == "triton_fused"
        and tensor_cache == "off"
    ):
        def _patched_triton(
            topk_ids: torch.Tensor,
            expert_load_view: torch.Tensor,
            logical_to_physical_map: torch.Tensor,
            logical_replica_count: torch.Tensor,
            indices_type: Optional[torch.dtype] = None,
        ) -> torch.Tensor:
            physical_ids = _eplb_runtime_map_core_triton_fused(
                topk_ids=topk_ids,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
                expert_load_view=expert_load_view,
                record_enabled=bool(st.get("path_is_cold", False)),
            )
            if indices_type is not None:
                physical_ids = physical_ids.to(dtype=indices_type)
            return physical_ids

        fused_layer.eplb_map_to_physical_and_record = _patched_triton
        st["installed"] = True
        st["orig_fn"] = orig
        return

    # Fast path for the current best runtime policy:
    # cold_only + flat_eager + tensor_cache=off.
    if (
        _eplb_runtime_fastpath_enabled()
        and record_mode == "cold_only"
        and map_impl == "flat_eager"
        and tensor_cache == "off"
    ):
        def _patched_fast(
            topk_ids: torch.Tensor,
            expert_load_view: torch.Tensor,
            logical_to_physical_map: torch.Tensor,
            logical_replica_count: torch.Tensor,
            indices_type: Optional[torch.dtype] = None,
        ) -> torch.Tensor:
            passthrough = _eplb_runtime_maybe_identity_passthrough(
                topk_ids=topk_ids,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
                indices_type=indices_type,
            )
            if passthrough is not None:
                if st["path_is_cold"]:
                    st["diag_record_calls"] = int(st.get("diag_record_calls", 0)) + 1
                    _eplb_runtime_record_load(
                        expert_load_view=expert_load_view, physical_ids=passthrough
                    )
                return passthrough

            physical_ids = _eplb_runtime_map_core_flat_eager(
                topk_ids=topk_ids,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )
            if st["path_is_cold"]:
                st["diag_record_calls"] = int(st.get("diag_record_calls", 0)) + 1
                _eplb_runtime_record_load(
                    expert_load_view=expert_load_view, physical_ids=physical_ids
                )
            if indices_type is not None:
                physical_ids = physical_ids.to(dtype=indices_type)
            return physical_ids
            _skip_cache_fe["compute_calls"] += 1
            return physical_ids

        fused_layer.eplb_map_to_physical_and_record = _patched_fast
        st["installed"] = True
        st["orig_fn"] = orig
        return

    # Fast path for map-only runtime policy:
    # off + flat_eager/two_choice_lb + tensor_cache=off.
    if (
        _eplb_runtime_fastpath_enabled()
        and record_mode == "off"
        and map_impl in {"flat_eager", "two_choice_lb"}
        and tensor_cache == "off"
    ):
        # Even in record=off mode, two-choice may rely on cold/hot signal.
        # Keep route-path setter active unless explicitly disabled.
        if map_impl == "two_choice_lb":
            _EPLB_RUNTIME_MAP_PATCH_STATE["record_mode"] = "cold_only"

        def _patched_off_flat(
            topk_ids: torch.Tensor,
            expert_load_view: torch.Tensor,
            logical_to_physical_map: torch.Tensor,
            logical_replica_count: torch.Tensor,
            indices_type: Optional[torch.dtype] = None,
        ) -> torch.Tensor:
            del expert_load_view
            passthrough = _eplb_runtime_maybe_identity_passthrough(
                topk_ids=topk_ids,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
                indices_type=indices_type,
            )
            if passthrough is not None:
                return passthrough
            st["diag_map_only_calls"] = int(st.get("diag_map_only_calls", 0)) + 1
            if st["map_impl"] == "flat_eager":
                physical_ids = _eplb_runtime_map_core_flat_eager(
                    topk_ids=topk_ids,
                    logical_to_physical_map=logical_to_physical_map,
                    logical_replica_count=logical_replica_count,
                )
            else:
                physical_ids = _eplb_runtime_map_core_twochoice_lb(
                    topk_ids=topk_ids,
                    logical_to_physical_map=logical_to_physical_map,
                    logical_replica_count=logical_replica_count,
                    path_is_cold=bool(st.get("path_is_cold", False)),
                )
            if indices_type is not None:
                physical_ids = physical_ids.to(dtype=indices_type)
            return physical_ids

        fused_layer.eplb_map_to_physical_and_record = _patched_off_flat
        st["installed"] = True
        st["orig_fn"] = orig
        return

    def _patched(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
        indices_type: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        mode = st["record_mode"]
        impl = st["map_impl"]
        if mode == "off":
            return _map_only(
                topk_ids=topk_ids,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
                indices_type=indices_type,
            )

        if mode == "full" and impl == "vllm":
            return orig(
                topk_ids=topk_ids,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
                indices_type=indices_type,
            )

        need_record = (mode == "full") or (mode == "cold_only" and st["path_is_cold"])
        if need_record and impl == "vllm":
            return orig(
                topk_ids=topk_ids,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
                indices_type=indices_type,
            )

        physical_ids = _map_only(
            topk_ids=topk_ids,
            logical_to_physical_map=logical_to_physical_map,
            logical_replica_count=logical_replica_count,
            indices_type=None,
        )
        if need_record:
            st["diag_record_calls"] = int(st.get("diag_record_calls", 0)) + 1
            _eplb_runtime_record_load(
                expert_load_view=expert_load_view, physical_ids=physical_ids
            )
        if indices_type is not None:
            physical_ids = physical_ids.to(dtype=indices_type)
        return physical_ids

    fused_layer.eplb_map_to_physical_and_record = _patched
    st["installed"] = True
    st["orig_fn"] = orig


def reset_eplb_runtime_map_policy():
    from vllm.model_executor.layers.fused_moe import layer as fused_layer

    st = _EPLB_RUNTIME_MAP_PATCH_STATE
    if st["installed"] and st["orig_fn"] is not None:
        fused_layer.eplb_map_to_physical_and_record = st["orig_fn"]
    st["installed"] = False
    st["orig_fn"] = None
    st["record_mode"] = "full"
    st["map_impl"] = "vllm"
    st["tensor_cache"] = "on"
    st["path"] = "unknown"
    st["path_is_cold"] = False
    # Keep diag counters for per-run collection. Bench resets via
    # get_eplb_runtime_map_diag(reset=True) at run start.
    st["identity_passthrough_cache"].clear()
    _EPLB_RUNTIME_TENSOR_CACHE["pos_indices"].clear()
    _EPLB_RUNTIME_TENSOR_CACHE["ones"].clear()
    _EPLB_RUNTIME_TENSOR_CACHE["twochoice_state"].clear()


@contextmanager
def use_eplb_runtime_map_policy(
    *,
    record_mode: str = "full",
    map_impl: str = "vllm",
    tensor_cache: str = "off",
):
    configure_eplb_runtime_map_policy(
        record_mode=record_mode, map_impl=map_impl, tensor_cache=tensor_cache
    )
    try:
        yield
    finally:
        reset_eplb_runtime_map_policy()


def set_bsp_g_component_timer(timer):
    """Install an optional benchmark-only timer for BSP-G source experiments."""
    global _BSP_G_COMPONENT_TIMER
    _BSP_G_COMPONENT_TIMER = timer


def set_bsp_g_layout_recorder(recorder):
    """Install an optional benchmark-only SP/TP layout recorder."""
    global _BSP_G_LAYOUT_RECORDER
    _BSP_G_LAYOUT_RECORDER = recorder


def _bsp_g_record_layout(**kwargs):
    if _BSP_G_LAYOUT_RECORDER is not None:
        try:
            _BSP_G_LAYOUT_RECORDER(**kwargs)
        except Exception:
            pass


def _bsp_g_timer_enabled() -> bool:
    return bool(
        _BSP_G_COMPONENT_TIMER is not None
        and getattr(_BSP_G_COMPONENT_TIMER, "enabled", False)
    )


def _bsp_g_time(name: str):
    if _bsp_g_timer_enabled():
        return _BSP_G_COMPONENT_TIMER.time(name)
    return nullcontext()


# OPT-1: Cached ForwardContext to eliminate per-layer DPMetadata AllReduce
_CACHED_FORWARD_CONTEXT = {}


def _get_cached_forward_context(cfg, n_tokens):
    """Return cached ForwardContext if DINF_CACHE_FWD_CONTEXT is set."""
    if not os.environ.get("DINF_CACHE_FWD_CONTEXT"):
        return None
    key = n_tokens
    if key not in _CACHED_FORWARD_CONTEXT:
        import torch
        from vllm.forward_context import DPMetadata, create_forward_context
        dp_size = cfg.parallel_config.data_parallel_size
        if dp_size > 1:
            num_tokens_cpu = torch.full((dp_size,), n_tokens, dtype=torch.int64)
            max_tokens_cpu = torch.tensor(n_tokens, dtype=torch.int64)
            dp_metadata = DPMetadata(max_tokens_cpu, num_tokens_cpu)
        else:
            dp_metadata = None
        _CACHED_FORWARD_CONTEXT[key] = create_forward_context(
            attn_metadata=None, vllm_config=cfg,
            virtual_engine=0, dp_metadata=dp_metadata)
    return _CACHED_FORWARD_CONTEXT[key]


def _bsp_g_add_bytes(name: str, tensor: torch.Tensor):
    if _bsp_g_timer_enabled():
        _BSP_G_COMPONENT_TIMER.add_bytes(
            name, int(tensor.numel() * tensor.element_size()))


def _get_ep_rank_and_size():
    """Get expert-parallel rank and size, preferring EP group over TP group."""
    try:
        from vllm.distributed import get_ep_group
        ep = get_ep_group()
        if ep is not None and ep.world_size > 1:
            return ep.rank_in_group, ep.world_size
    except Exception:
        pass
    # Fallback: use TP group (legacy AllReduce EP mode)
    return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()


def _get_eplb_constructor_hint() -> tuple[bool, int]:
    """Read current vLLM EPLB config for MoE layer construction.

    Returns:
        (enable_eplb, num_redundant_experts)
    """
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        if cfg is None:
            return False, 0
        pcfg = cfg.parallel_config
        enable_eplb = bool(getattr(pcfg, "enable_eplb", False))
        eplb_cfg = getattr(pcfg, "eplb_config", None)
        num_redundant = int(getattr(eplb_cfg, "num_redundant_experts", 0)) if eplb_cfg is not None else 0
        if not enable_eplb:
            num_redundant = 0
        return enable_eplb, max(0, num_redundant)
    except Exception:
        return False, 0


def _align_redundant_for_ep(num_logical_experts: int, num_redundant_experts: int, ep_size: int) -> int:
    """Round up redundant experts so (logical + redundant) is divisible by EP size."""
    if ep_size <= 1:
        return max(0, int(num_redundant_experts))
    logical = int(num_logical_experts)
    redundant = max(0, int(num_redundant_experts))
    total = logical + redundant
    rem = total % int(ep_size)
    if rem == 0:
        return redundant
    return redundant + (int(ep_size) - rem)


def _load_custom_redundant_logical_ids(
    *,
    num_logical_experts: int,
    num_redundant_experts: int,
) -> Optional[list[int]]:
    """Load optional logical ids used by redundant physical experts.

    Env:
        DINF_EPLB_REPLICA_LOGICAL_IDS_PATH

    File payload can be tensor/list/tuple. Values must be in [0, num_logical).
    If payload length is shorter than required redundant count, repeat cyclically.
    """
    if int(num_redundant_experts) <= 0:
        return None
    env_path = os.environ.get("DINF_EPLB_REPLICA_LOGICAL_IDS_PATH", "").strip()
    if not env_path:
        return None

    p = Path(env_path).expanduser()
    if not p.exists():
        print(f"[EPLB] custom replica ids file not found: {p}")
        return None

    try:
        raw = torch.load(p, map_location="cpu")
        if isinstance(raw, torch.Tensor):
            arr = raw.to(torch.int64).reshape(-1).tolist()
        elif isinstance(raw, (list, tuple)):
            arr = [int(x) for x in raw]
        else:
            print(f"[EPLB] ignore unsupported replica ids payload type: {type(raw)}")
            return None

        clean = []
        for x in arr:
            xi = int(x)
            if 0 <= xi < int(num_logical_experts):
                clean.append(xi)
        if not clean:
            print(f"[EPLB] no valid replica ids in file: {p}")
            return None

        need = int(num_redundant_experts)
        if len(clean) >= need:
            out = clean[:need]
        else:
            out = [clean[i % len(clean)] for i in range(need)]

        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                print(
                    "[EPLB] custom redundant logical ids enabled: "
                    f"path={p}, input={len(clean)}, used={len(out)}, "
                    f"head={out[:min(16, len(out))]}"
                )
        else:
            print(
                "[EPLB] custom redundant logical ids enabled: "
                f"path={p}, input={len(clean)}, used={len(out)}, "
                f"head={out[:min(16, len(out))]}"
            )
        return out
    except Exception as e:
        print(f"[EPLB] failed loading custom replica ids from {p}: {e}")
        return None


def _build_global_physical_to_logical_map(
    *,
    num_logical_experts: int,
    num_redundant_experts: int,
) -> tuple[list[int], bool]:
    """Build physical->logical mapping with optional custom redundant ids."""
    base = list(range(int(num_logical_experts)))
    num_redundant = max(0, int(num_redundant_experts))
    if num_redundant <= 0:
        return base, False

    custom = _load_custom_redundant_logical_ids(
        num_logical_experts=int(num_logical_experts),
        num_redundant_experts=num_redundant,
    )
    if custom is not None:
        return base + [int(x) for x in custom], True

    default = EplbState.build_initial_global_physical_to_logical_map(
        num_routed_experts=int(num_logical_experts),
        num_redundant_experts=num_redundant,
    )
    return [int(x) for x in default], False


def _build_layered_l2p_lrc_from_p2l(
    *,
    physical_to_logical_map: torch.Tensor,
    num_logical_experts: int,
    max_slots_per_logical: int,
    l2p_dtype: torch.dtype,
    lrc_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild logical->physical map and replica count from layered p2l mapping."""
    if physical_to_logical_map.ndim != 2:
        raise RuntimeError(
            f"[EPLB] physical_to_logical_map must be 2D, got {tuple(physical_to_logical_map.shape)}"
        )
    num_layers, num_physical = physical_to_logical_map.shape
    dev = physical_to_logical_map.device
    l2p = torch.full(
        (num_layers, int(num_logical_experts), int(max_slots_per_logical)),
        -1,
        dtype=l2p_dtype,
        device=dev,
    )
    lrc = torch.zeros(
        (num_layers, int(num_logical_experts)),
        dtype=lrc_dtype,
        device=dev,
    )
    for li in range(int(num_layers)):
        for pid in range(int(num_physical)):
            lid = int(physical_to_logical_map[li, pid].item())
            if lid < 0 or lid >= int(num_logical_experts):
                raise RuntimeError(
                    f"[EPLB] layer {li}: logical id out of range in p2l (pid={pid}, lid={lid})"
                )
            slot = int(lrc[li, lid].item())
            if slot >= int(max_slots_per_logical):
                raise RuntimeError(
                    f"[EPLB] layer {li}: replica slots overflow for logical={lid}, "
                    f"slot={slot}, max={int(max_slots_per_logical)}"
                )
            l2p[li, lid, slot] = int(pid)
            lrc[li, lid] += 1
    return l2p, lrc


def _is_identity_linear_ep_map(map_tensor: torch.Tensor, num_experts: int, ep_size: int) -> bool:
    """Check whether expert->ep map is the default linear split."""
    if map_tensor.ndim != 2 or map_tensor.shape[1] != num_experts:
        return False
    chunk = num_experts // ep_size
    if chunk <= 0:
        return False
    ref = (torch.arange(num_experts, dtype=torch.int64) // chunk).clamp_max(ep_size - 1)
    return bool(torch.all(map_tensor.to(torch.int64) == ref.unsqueeze(0)))


def _build_rank_to_local_phys_ids(global_num_physical_experts: int, ep_size: int) -> dict[int, list[int]]:
    """Build per-rank local physical id lists with linear even split."""
    base = int(global_num_physical_experts) // int(ep_size)
    rem = int(global_num_physical_experts) % int(ep_size)
    out: dict[int, list[int]] = {}
    for r in range(int(ep_size)):
        start = r * base + min(r, rem)
        end = start + base + (1 if r < rem else 0)
        out[r] = list(range(start, end))
    return out


def _load_vllm_balanced_phy2log() -> Optional[torch.Tensor]:
    """Load pre-computed per-layer phy2log from DINF_EPLB_VLLM_PHY2LOG_PATH."""
    path = os.environ.get("DINF_EPLB_VLLM_PHY2LOG_PATH", "").strip()
    if not path:
        return None
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        print(f"[EPLB] vllm_balanced phy2log not found: {p}")
        return None
    try:
        t = torch.load(p, map_location="cpu").to(torch.int64)
        print(f"[EPLB] loaded vllm balanced phy2log: {p} shape={tuple(t.shape)}")
        return t
    except Exception as e:
        print(f"[EPLB] failed loading vllm phy2log: {e}")
        return None


def _build_vllm_balanced_local_maps(
    *,
    phy2log: torch.Tensor,
    layer_id: int,
    global_num_physical_experts: int,
    ep_size: int,
) -> dict[int, list[int]]:
    """Build per-rank local physical ids from vllm balanced phy2log."""
    phy_per_gpu = int(global_num_physical_experts) // int(ep_size)
    row = phy2log[layer_id].tolist()
    per_rank: dict[int, list[int]] = {r: [] for r in range(int(ep_size))}
    for gpu in range(int(ep_size)):
        start = gpu * phy_per_gpu
        end = start + phy_per_gpu
        for local_slot, logical_id in enumerate(row[start:end]):
            per_rank[gpu].append(start + local_slot)
    return per_rank


def _build_joint_external_replica_local_maps(
    *,
    external_map: torch.Tensor,
    layer_id: int,
    num_experts: int,
    global_num_physical_experts: int,
    ep_size: int,
) -> dict[int, list[int]]:
    """Build per-rank local physical ids under joint P1/P5 policy.

    Policy:
    - Keep canonical physical->logical relation unchanged.
    - Reassign physical-id ownership to satisfy P1 for base experts:
      pid in [0, num_experts) is owned by rank external_map[layer_id, pid].
    - For redundant physical ids pid in [num_experts, global_num_physical_experts),
      assign in round-robin over ranks.
    """
    if external_map.ndim != 2:
        raise RuntimeError(f"[EPLB] external map must be 2D, got {tuple(external_map.shape)}")
    if layer_id < 0 or layer_id >= int(external_map.shape[0]):
        raise RuntimeError(f"[EPLB] invalid layer_id={layer_id} for external map")
    if int(external_map.shape[1]) != int(num_experts):
        raise RuntimeError(
            f"[EPLB] external map width mismatch: got {int(external_map.shape[1])}, "
            f"expect {int(num_experts)}"
        )
    if global_num_physical_experts < num_experts:
        raise RuntimeError(
            f"[EPLB] invalid physical experts: global={global_num_physical_experts} < logical={num_experts}"
        )

    local_cap = int(global_num_physical_experts) // int(ep_size)
    if int(global_num_physical_experts) % int(ep_size) != 0:
        raise RuntimeError(
            f"[EPLB] global_num_physical_experts={global_num_physical_experts} "
            f"is not divisible by ep_size={ep_size}"
        )

    per_rank: dict[int, list[int]] = {r: [] for r in range(int(ep_size))}
    layer_map = external_map[layer_id].to(torch.int64)

    # Base physical experts follow P1 map.
    for lid in range(int(num_experts)):
        r = int(layer_map[lid].item())
        if r < 0 or r >= int(ep_size):
            raise RuntimeError(
                f"[EPLB] layer {layer_id}: rank out of range for logical={lid}, rank={r}"
            )
        per_rank[r].append(lid)

    # Redundant physical experts are assigned round-robin across ranks.
    for pid in range(int(num_experts), int(global_num_physical_experts)):
        r = int((pid - int(num_experts)) % int(ep_size))
        per_rank[r].append(pid)

    # Invariant: exact local capacity per rank.
    for r in range(int(ep_size)):
        c = len(per_rank[r])
        if c != local_cap:
            raise RuntimeError(
                f"[EPLB] layer {layer_id}: joint mapping local capacity mismatch "
                f"rank={r}, got={c}, expect={local_cap}"
            )
    return per_rank


def _get_eplb_init_placement_mode() -> str:
    """Init placement mode for external-map + redundant experts."""
    raw = os.environ.get("DINF_EPLB_INIT_PLACEMENT_MODE", "").strip().lower()
    if raw in {"", "joint_p1_p5", "joint", "legacy"}:
        return "joint_p1_p5"
    if raw in {"weight_balance", "wb"}:
        return "weight_balance"
    if raw in {"vllm_balanced", "vllm"}:
        return "vllm_balanced"
    print(
        f"[EPLB] unknown DINF_EPLB_INIT_PLACEMENT_MODE='{raw}', "
        "fallback to 'joint_p1_p5'"
    )
    return "joint_p1_p5"


def _aggregate_routing_payload_to_expert_load(
    *,
    payload: dict,
    num_layers: int,
    num_experts: int,
) -> Optional[torch.Tensor]:
    """Aggregate routing trace payload (expert_budgeting_routing_data.pt style)."""
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    out = torch.zeros((num_layers, num_experts), dtype=torch.float64)
    valid_steps = 0
    for step in data:
        if not isinstance(step, dict):
            continue
        valid_steps += 1
        for layer_key, layer_payload in step.items():
            if not isinstance(layer_payload, dict):
                continue
            try:
                layer_id = int(layer_key)
            except Exception:
                continue
            if layer_id < 0 or layer_id >= num_layers:
                continue
            topk_idx = layer_payload.get("topk_idx")
            if not isinstance(topk_idx, torch.Tensor):
                continue
            ids = topk_idx.to(torch.long).reshape(-1)
            if ids.numel() == 0:
                continue
            in_range = (ids >= 0) & (ids < num_experts)
            if not bool(in_range.all()):
                ids = ids[in_range]
            if ids.numel() == 0:
                continue
            out[layer_id] += torch.bincount(ids, minlength=num_experts).to(torch.float64)
    if valid_steps <= 0:
        return None
    return out


def _load_eplb_init_global_expert_load(
    *,
    num_layers: int,
    num_experts: int,
    sparse_layer_ids: list[int],
) -> Optional[torch.Tensor]:
    """Load optional expert-load prior for init weight-balance placement.

    Env:
      - DINF_EPLB_INIT_GLOBAL_EXPERT_LOAD_PATH

    Accepted payload:
      1) Tensor [num_layers, num_experts]
      2) Tensor [num_sparse_layers, num_experts]
      3) Tensor [num_experts] (broadcast to sparse layers)
      4) Dict with one of:
         - key 'global_expert_load' / 'expert_load' / 'weight' -> Tensor
         - key 'data' routing payload (expert_budgeting_routing_data.pt style)
    """
    env_path = os.environ.get("DINF_EPLB_INIT_GLOBAL_EXPERT_LOAD_PATH", "").strip()
    if not env_path:
        return None

    p = Path(env_path).expanduser()
    if not p.exists():
        print(f"[EPLB] init global expert load file not found: {p}")
        return None

    try:
        raw = torch.load(p, map_location="cpu")
    except Exception as e:
        print(f"[EPLB] failed loading init global expert load {p}: {e}")
        return None

    t: Optional[torch.Tensor] = None
    if isinstance(raw, torch.Tensor):
        t = raw
    elif isinstance(raw, dict):
        for k in ("global_expert_load", "expert_load", "weight"):
            v = raw.get(k)
            if isinstance(v, torch.Tensor):
                t = v
                break
        if t is None:
            t = _aggregate_routing_payload_to_expert_load(
                payload=raw, num_layers=num_layers, num_experts=num_experts
            )
    if t is None:
        print(f"[EPLB] unsupported init global expert load payload: {type(raw)}")
        return None

    t = t.to(torch.float64).cpu()
    if t.ndim == 1:
        if int(t.numel()) != int(num_experts):
            print(
                f"[EPLB] invalid 1D init load shape={tuple(t.shape)}, "
                f"expect ({num_experts},)"
            )
            return None
        t = t.unsqueeze(0).repeat(len(sparse_layer_ids), 1)
    elif t.ndim == 2:
        if tuple(t.shape) == (num_layers, num_experts):
            t = t[sparse_layer_ids]
        elif tuple(t.shape) == (len(sparse_layer_ids), num_experts):
            pass
        else:
            print(
                f"[EPLB] invalid 2D init load shape={tuple(t.shape)}, "
                f"expect ({num_layers},{num_experts}) or "
                f"({len(sparse_layer_ids)},{num_experts})"
            )
            return None
    else:
        print(f"[EPLB] invalid init load ndim={t.ndim}, shape={tuple(t.shape)}")
        return None

    t = torch.clamp(t, min=0)
    if float(t.sum().item()) <= 0.0:
        print("[EPLB] init load sum<=0, ignore init placement load prior.")
        return None
    print(
        "[EPLB] loaded init global expert load prior: "
        f"path={p}, shape={tuple(t.shape)}"
    )
    return t


def _build_weight_balanced_external_replica_local_maps(
    *,
    external_map: torch.Tensor,
    layer_id: int,
    num_experts: int,
    global_num_physical_experts: int,
    ep_size: int,
    physical_to_logical: list[int],
    layer_expert_load: torch.Tensor,
    pin_base_experts: bool = True,
) -> dict[int, list[int]]:
    """Build per-rank local physical ids with load-aware placement.

    Keep physical->logical mapping unchanged; only rebalance ownership of
    physical ids across ranks for this layer.
    """
    if external_map.ndim != 2:
        raise RuntimeError(f"[EPLB] external map must be 2D, got {tuple(external_map.shape)}")
    if layer_id < 0 or layer_id >= int(external_map.shape[0]):
        raise RuntimeError(f"[EPLB] invalid layer_id={layer_id} for external map")
    if int(external_map.shape[1]) != int(num_experts):
        raise RuntimeError(
            f"[EPLB] external map width mismatch: got {int(external_map.shape[1])}, "
            f"expect {int(num_experts)}"
        )
    if len(physical_to_logical) != int(global_num_physical_experts):
        raise RuntimeError(
            "[EPLB] physical_to_logical length mismatch: "
            f"got={len(physical_to_logical)}, expect={int(global_num_physical_experts)}"
        )
    if layer_expert_load.ndim != 1 or int(layer_expert_load.numel()) != int(num_experts):
        raise RuntimeError(
            f"[EPLB] layer_expert_load shape mismatch: got {tuple(layer_expert_load.shape)}, "
            f"expect ({int(num_experts)},)"
        )

    local_cap = int(global_num_physical_experts) // int(ep_size)
    if int(global_num_physical_experts) % int(ep_size) != 0:
        raise RuntimeError(
            f"[EPLB] global_num_physical_experts={global_num_physical_experts} "
            f"is not divisible by ep_size={ep_size}"
        )

    # Replica count by logical expert from canonical physical->logical map.
    replica_count = [0 for _ in range(int(num_experts))]
    for pid in range(int(global_num_physical_experts)):
        lid = int(physical_to_logical[pid])
        if lid < 0 or lid >= int(num_experts):
            raise RuntimeError(
                f"[EPLB] invalid logical id in physical_to_logical: pid={pid}, lid={lid}"
            )
        replica_count[lid] += 1
    for lid, c in enumerate(replica_count):
        if c <= 0:
            raise RuntimeError(f"[EPLB] zero replica count for logical expert lid={lid}")

    layer_map = external_map[layer_id].to(torch.int64)
    per_rank: dict[int, list[int]] = {r: [] for r in range(int(ep_size))}
    rank_load = [0.0 for _ in range(int(ep_size))]

    # Step1: optionally pin base experts to P1 rank.
    assigned = [False for _ in range(int(global_num_physical_experts))]
    if pin_base_experts:
        for lid in range(int(num_experts)):
            r = int(layer_map[lid].item())
            if r < 0 or r >= int(ep_size):
                raise RuntimeError(
                    f"[EPLB] layer {layer_id}: rank out of range for logical={lid}, rank={r}"
                )
            pid = lid  # canonical base physical ids
            per_rank[r].append(pid)
            assigned[pid] = True
            rank_load[r] += float(layer_expert_load[lid].item()) / float(replica_count[lid])

    # Step2: assign remaining physical ids with load-aware greedy packing.
    remaining = [pid for pid in range(int(global_num_physical_experts)) if not assigned[pid]]
    pid_weight = []
    for pid in remaining:
        lid = int(physical_to_logical[pid])
        w = float(layer_expert_load[lid].item()) / float(replica_count[lid])
        pid_weight.append((pid, w))
    pid_weight.sort(key=lambda x: (-x[1], x[0]))

    for pid, w in pid_weight:
        candidates = [r for r in range(int(ep_size)) if len(per_rank[r]) < local_cap]
        if not candidates:
            raise RuntimeError("[EPLB] no capacity left while assigning physical ids")
        # choose the currently lightest rank, then smaller count, then rank id
        best = min(candidates, key=lambda r: (rank_load[r], len(per_rank[r]), r))
        per_rank[best].append(int(pid))
        rank_load[best] += float(w)

    # Invariants.
    flat = []
    for r in range(int(ep_size)):
        if len(per_rank[r]) != local_cap:
            raise RuntimeError(
                f"[EPLB] layer {layer_id}: weight-balance local capacity mismatch "
                f"rank={r}, got={len(per_rank[r])}, expect={local_cap}"
            )
        flat.extend(per_rank[r])
    if len(flat) != int(global_num_physical_experts) or len(set(flat)) != int(global_num_physical_experts):
        raise RuntimeError(
            f"[EPLB] layer {layer_id}: weight-balance global physical assignment is not a bijection"
        )
    return per_rank


def _moe_forward_with_context(experts, hidden_states_flat, router_logits):
    """Call experts.forward_impl with ForwardContext when vLLM config exists."""
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
        if cfg is not None:
            from vllm.forward_context import set_forward_context
            num_tokens = hidden_states_flat.shape[0]
            with set_forward_context(
                attn_metadata=None, vllm_config=cfg, num_tokens=num_tokens,
            ):
                return experts.forward_impl(
                    hidden_states=hidden_states_flat, router_logits=router_logits)
    except Exception:
        pass
    return experts.forward_impl(
        hidden_states=hidden_states_flat, router_logits=router_logits)


@dataclass
class _BSPGSPHiddenState:
    hidden_sp: torch.Tensor
    bsz: int
    seq_len: int
    n_tokens: int


@dataclass
class _BSPGAttnSPResult:
    attn_sp: torch.Tensor
    bsz: int
    seq_len: int
    n_tokens: int


def _bsp_g_current_vllm_config():
    from vllm.config import get_current_vllm_config

    cfg = get_current_vllm_config()
    if cfg is None:
        raise RuntimeError("BSP-G source path requires current vLLM config.")
    return cfg


def _bsp_g_gather_sp_hidden(sp_state: _BSPGSPHiddenState) -> torch.Tensor:
    from vllm.distributed import tensor_model_parallel_all_gather

    hidden_flat = tensor_model_parallel_all_gather(sp_state.hidden_sp, dim=0)
    return hidden_flat[:sp_state.n_tokens].view(
        sp_state.bsz, sp_state.seq_len, -1)


if is_flash_attn_2_available():
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    except Exception:
        flash_attn_func = None
        flash_attn_varlen_func = None
        index_first_axis = None
        pad_input = None
        unpad_input = None


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LLaDA2MoeConfig"


def _compute_default_rope_parameters(config, device=None):
    base = getattr(config, "rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    attention_factor = 1.0
    return inv_freq, attention_factor


def roll_tensor(tensor, shifts=-1, dims=-1, fill_value=0):
    """Roll the tensor input along the given dimension(s).
    Inserted elements are set to be 0.0.
    """
    rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
    rolled_tensor.select(dims, shifts).fill_(fill_value)
    return rolled_tensor, rolled_tensor.sum()

def replace_linear_class(
    linear: nn.Linear, style: Literal["colwise", "rowwise", "qkv"],
    quant_config, model_config
) -> Union[ColumnParallelLinear, RowParallelLinear]:
    """
    Replace nn.Linear with one of vLLM's tensor parallel linear classes.

    Args:
        linear (nn.Linear): `nn.Linear` to be replaced.
        style (str): Tensor parallel style of the new linear, e.g. "colwise".
        quant_config (QuantConfig): Quantization config for the new linear.
    Returns:
        Union[ColumnParallelLinear, RowParallelLinear]: The new linear.
    """

    if not isinstance(style, str):
        raise ValueError(
            f"Unsupported parallel style type {type(style)}, expected str")

    vllm_linear_cls = {
        "colwise": ColumnParallelLinear,
        "rowwise": RowParallelLinear,
        "qkv": QKVParallelLinear
    }.get(style, ReplicatedLinear)
    if style != "qkv":
        return vllm_linear_cls(
            input_size=linear.in_features,
            output_size=linear.out_features,
            bias=linear.bias is not None,
            quant_config=quant_config,
            return_bias=False,
        )
    else:
        return QKVParallelLinear(
            hidden_size = model_config.hidden_size,
            head_size=model_config.head_dim,
            total_num_heads=model_config.num_attention_heads,
            total_num_kv_heads=model_config.num_key_value_heads,
            bias=linear.bias is not None,
            quant_config=quant_config,
            return_bias=False,
        )     
def _all_gather_cat(
    tensor: torch.Tensor,
    dim: int = 1,
    group: Optional[dist.ProcessGroup] = None,
    normal_len: int = 0,
    last_len: int = 0,
) -> torch.Tensor:
    """
    Gather tensors along `dim` from all ranks and concatenate them.
    Only the last chunk may be shorter than `normal_len`; all others are exactly `normal_len`.

    Args:
        tensor: local tensor on current rank
        dim: dimension along which to concatenate
        normal_len: length of the first (world_size-1) ranks along `dim`
        last_len: length of the last rank along `dim`

    Returns:
        Concatenated tensor of shape [total_len, ...] along `dim`
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if world_size == 1:
        return tensor

    # 1. Move the concatenation dimension to 0 for easier all_gather
    tensor = tensor.movedim(dim, 0)          # [L_local, ...]
    L_local = tensor.size(0)

    # 2. Compute global length across all ranks
    total_len = normal_len * (world_size - 1) + last_len

    # 3. Pre-allocate receive buffers (same shape for all ranks, sized for the largest chunk)
    max_len = max(normal_len, last_len)
    gather_list = [
        torch.empty([max_len] + list(tensor.shape[1:]),
                   dtype=tensor.dtype,
                   device=tensor.device)
        for _ in range(world_size)
    ]

    # 4. Copy local data into the corresponding buffer (only first L_local rows are valid)
    gather_list[rank][:L_local] = tensor

    # 5. All-gather (communicate only valid parts)
    dist.all_gather(gather_list, gather_list[rank], group=group)

    # 6. Trim padding and concatenate
    gathered = torch.cat(gather_list, dim=0)[:total_len]

    # 7. Move dimension back to original position
    return gathered.movedim(0, dim)    

class H2Embed:
    def __init__(self, embedding: nn.Embedding, tau: float = 1.0):
        """
        W_e : token embedding weights [V, d]
        tau : temperature; lower values yield sharper distributions
        """
        self.embedding = embedding
        self.W_e = embedding.weight
        self.tau = tau
        self.sp_size = 1  # no sequence parallel by default

    def __call__(
        self,
        x: torch.Tensor,
        mask_index: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        iter_cont_weight: float = 0.0
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L] token ids
            mask_index: [B, L] bool tensor, True where continuous embedding should be used
            logits: [B, L, V] logits used to produce continuous embeddings
            iter_cont_weight: blending weight between continuous and discrete embeddings

        Returns:
            Embedded representations [B, L, d]
        """
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        seq_len = x.shape[1]

        # If sequence parallel is enabled, each rank handles a slice of the sequence
        if self.sp_size > 1:
            normal_seq_len = (seq_len + self.sp_size - 1) // self.sp_size
            last_seq_len = seq_len - normal_seq_len * (self.sp_size - 1)

            part_start = normal_seq_len * rank
            part_end = min(normal_seq_len * (rank + 1), seq_len)
            x_part = x[:, part_start:part_end]

            if mask_index is not None:
                mask_part = mask_index[:, part_start:part_end]
                logits_part = logits[:, part_start:part_end] if logits is not None else None
            else:
                mask_part = None
                logits_part = None
        else:
            x_part = x
            mask_part = mask_index
            logits_part = logits

        # Base discrete embedding
        result_part = self.embedding(x_part)

        # Replace selected positions with continuous embeddings
        if mask_part is not None and logits_part is not None:
            prob = torch.softmax(logits_part / self.tau, dim=-1)  # [B, L_part, V]
            input_embeds_h = prob.to(self.W_e.dtype) @ self.W_e  # [B, L_part, d]

            # Blend continuous and discrete embeddings
            result_part = torch.where(
                mask_part.unsqueeze(-1),
                iter_cont_weight * input_embeds_h + 1 * result_part,
                result_part
            )

        # 4. Gather and concatenate sequence slices across ranks
        if self.sp_size > 1:
            out = _all_gather_cat(
                result_part,
                dim=1,
                group=None,
                normal_len=normal_seq_len,
                last_len=last_seq_len
            )
        else:
            out = result_part

        return out


@dataclass
class MoEV2CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs as well as Mixture of Expert's router hidden
    states terms, to train a MoE model.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        z_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            z_loss for the sparse modules.
        aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            aux_loss for the sparse modules.
        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Router logits of the encoder model, useful to compute the auxiliary loss and the z_loss for the sparse
            modules.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    z_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None
    mtp_loss: Optional[torch.FloatTensor] = None
    mtp_logits: Optional[tuple[torch.FloatTensor, ...]] = None


class MoeV2ModelOutputWithPast(MoeModelOutputWithPast):

    def __init__(self, mtp_hidden_states=None, **kwargs):
        super().__init__(**kwargs)
        self.mtp_hidden_states = mtp_hidden_states


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.LLaDA2Moe.modeling_LLaDA2Moe._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.LLaDA2Moe.modeling_LLaDA2Moe._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.LLaDA2Moe.modeling_LLaDA2Moe.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


class LLaDA2MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LLaDA2MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LLaDA2MoeRMSNorm)


class LLaDA2MoeRotaryEmbedding(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type, _compute_default_rope_parameters)

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class LLaDA2MoeMLP(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LLaDA2MoeGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts

        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.routed_scaling_factor = config.routed_scaling_factor

        self.register_buffer("expert_bias", torch.zeros((self.num_experts)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def group_limited_topk(
        self,
        scores: torch.Tensor,
    ):
        num_tokens, _ = scores.size()
        # Organize the experts into groups
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Mask the experts based on selection groups
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )

        masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
        probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)

        return probs, top_indices

    def forward(self, hidden_states):
        # compute gating score
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

        scores = torch.sigmoid(logits.float()).type_as(logits)

        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)

        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)

        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight, logits

    def get_logits(self, hidden_states):
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return logits

    def routing(self, hidden_states, gating_output, topk, renormalize):
        scores = torch.sigmoid(gating_output.float()).type_as(gating_output)

        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)

        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(gating_output)

        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_weight, topk_idx

def static_routing_function(gate, hidden_states, gating_output, topk, renormalize):
    return gate.routing(hidden_states, gating_output, topk, renormalize)
class LLaDA2MoeSparseMoeBlock(nn.Module):
    """A tensor-parallel MoE implementation for Olmoe that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(self,
                 config,
                 prefix: str = "",
                 use_padding_free: bool = False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.use_padding_free = use_padding_free

        # Gate always runs at half / full precision for now.
        self.gate = LLaDA2MoeGate(config)
        # print('config.num_shared_experts', config.num_shared_experts)
        if config.num_shared_experts is not None:
            # print('config.num_shared_experts is not None!')
            self.shared_experts = LLaDA2MoeMLP(
                config=config, intermediate_size=config.moe_intermediate_size * config.num_shared_experts
            )
        # custom_routing = partial(custom_routing_function, gate=self.gate)
        enable_eplb, num_redundant_experts = _get_eplb_constructor_hint()
        ep_rank, ep_size = _get_ep_rank_and_size()
        aligned_redundant = _align_redundant_for_ep(
            num_logical_experts=self.num_experts,
            num_redundant_experts=num_redundant_experts,
            ep_size=ep_size,
        )
        if enable_eplb and aligned_redundant != num_redundant_experts and ep_rank == 0:
            print(
                "[EPLB] align redundant experts for EP divisibility: "
                f"request={num_redundant_experts}, aligned={aligned_redundant}, ep_size={ep_size}"
            )
        self.experts = FusedMoE(num_experts=self.num_experts,
                                top_k=self.top_k,
                                hidden_size=self.hidden_size,
                                intermediate_size=config.moe_intermediate_size,
                                reduce_results=True,
                                quant_config=None,
                                tp_size=None,
                                custom_routing_function=partial(static_routing_function, self.gate),
                                prefix=f"{prefix}.experts",
                                enable_eplb=enable_eplb,
                                num_redundant_experts=aligned_redundant)
        # This is a hack. expert_map in FusedMoE isn't moved to GPU by default.
        # We have to register it explicitly so that it can be moved to GPU with FusedMoE
        expert_map = self.experts.expert_map
        del self.experts.expert_map
        self.experts.register_buffer('expert_map', expert_map)

        # C11: shared expert overlap is enabled externally after
        # prepare_communication_buffer_for_model creates the MK.
        # See bench_multi_gpu.py: attach_shared_experts_for_overlap(model)

    def set_sequence_parallel(self, enabled: bool, tp_size: Optional[int] = None):
        tp = tp_size or get_tensor_model_parallel_world_size()
        self.experts.is_sequence_parallel = bool(enabled)
        self.experts.sp_size = tp if enabled else 1

    def forward_sp(self, hidden_states_sp: torch.Tensor) -> torch.Tensor:
        if os.environ.get("DINF_OVERLAP_SHARED") and self.config.num_shared_experts is not None:
            # OPT-5: shared expert runs on side stream, overlapping with gate+dispatch
            if not hasattr(self, '_shared_stream'):
                self._shared_stream = torch.cuda.Stream()
            with torch.cuda.stream(self._shared_stream):
                shared_res = self.shared_experts(hidden_states_sp)
            with _bsp_g_time("moe.gate_logits"):
                router_logits = self.gate.get_logits(hidden_states_sp)
            with _bsp_g_time("moe.native_forward"):
                y_sp = self.experts.forward_impl(hidden_states_sp, router_logits)
            torch.cuda.current_stream().wait_stream(self._shared_stream)
            y_sp = y_sp + shared_res
        else:
            with _bsp_g_time("moe.shared"):
                shared_res = (
                    self.shared_experts(hidden_states_sp)
                    if self.config.num_shared_experts is not None else None
                )
            with _bsp_g_time("moe.gate_logits"):
                router_logits = self.gate.get_logits(hidden_states_sp)
            with _bsp_g_time("moe.native_forward"):
                y_sp = self.experts.forward_impl(hidden_states_sp, router_logits)
            if shared_res is not None:
                y_sp = y_sp + shared_res
        return y_sp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, h = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, h)

        if self.use_padding_free:
            from dinfer.decoding.padding_free_moe import padding_free_moe
            shared_res = self.shared_experts(hidden_states_flat) if self.config.num_shared_experts is not None else None
            topk_idx, topk_weight, _ = self.gate(hidden_states_flat)
            y = padding_free_moe(
                hidden_states_flat,
                self.experts.w13_weight,
                self.experts.w2_weight,
                topk_weight,
                topk_idx,
                num_experts=self.num_experts,
                top_k=self.top_k,
            )
            if shared_res is not None:
                y = y + shared_res
        else:
            # Shared expert MUST run before routed experts (fused_moe inplace)
            shared_res = self.shared_experts(hidden_states_flat) if self.config.num_shared_experts is not None else None
            router_logits = self.gate.get_logits(hidden_states_flat)
            y = _moe_forward_with_context(
                self.experts, hidden_states_flat, router_logits)
            if shared_res is not None:
                y = y + shared_res

        y = y.view(bsz, seq_len, h)
        return y

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->LLaDA2Moe
class LLaDA2MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LLaDA2MoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        # TP-aware head partitioning (C11: Qwen3Moe-style TP attention)
        self.total_num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or self.hidden_size // self.total_num_heads
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        self.rope_dim = int(self.head_dim * partial_rotary_factor)
        self.total_num_kv_heads = config.num_key_value_heads

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.num_heads = self.total_num_heads // self.tp_size
        self.num_key_value_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = False

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.use_qkv_bias,
        )

        # if self.config.use_qk_norm:
        self.query_layernorm = LLaDA2MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = LLaDA2MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _dense_reduce_scatter_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_reduce_scatter

        dense = self.dense
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
        _bsp_g_add_bytes("attn_rs_payload", output_flat)
        with _bsp_g_time("attn.tp_reduce_scatter"):
            return tensor_model_parallel_reduce_scatter(output_flat, dim=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        qkv, _ = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # if self.config.use_qk_norm:
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        kv_seq_len = key_states.shape[-2]
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output, _ = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def forward_bsp_g_sp(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[_BSPGAttnSPResult, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        from vllm.model_executor.layers.layernorm import rms_norm as vllm_rms_norm

        bsz, q_len, _ = hidden_states.size()
        n_tokens = bsz * q_len
        try:
            tp_size = get_tensor_model_parallel_world_size()
        except Exception:
            tp_size = 1
        _bsp_g_record_layout(
            config="GS_source",
            layer_id=self.layer_idx,
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

        with _bsp_g_time("attn.qkv_proj"):
            qkv, _ = self.query_key_value(hidden_states)
            qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
            query_states, key_states, value_states = qkv.split(
                [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
            )

            query_states = vllm_rms_norm(
                query_states,
                self.query_layernorm.weight,
                self.query_layernorm.variance_epsilon,
            )
            key_states = vllm_rms_norm(
                key_states,
                self.key_layernorm.weight,
                self.key_layernorm.variance_epsilon,
            )

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        with _bsp_g_time("attn.kv_cache_update"):
            if past_key_value is not None:
                if self.layer_idx is None:
                    raise ValueError("BSP-G attention requires layer_idx when cache is used.")
                replace_position = kwargs.get("replace_position", None)
                try:
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, self.layer_idx, replace_position)
                except TypeError:
                    cache_kwargs = {"sin": sin, "cos": cos}
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, self.layer_idx, cache_kwargs)

        if use_cache:
            past_key_value = (key_states, value_states)

        with _bsp_g_time("attn.flash_compute"):
            flash_func = globals().get("flash_attn_func", None)
            if attention_mask is None and flash_func is not None:
                attn_output = flash_func(
                    query_states.transpose(1, 2).contiguous(),
                    key_states.transpose(1, 2).contiguous(),
                    value_states.transpose(1, 2).contiguous(),
                    causal=False,
                )
            else:
                if attention_mask is not None:
                    kv_seq_len = key_states.shape[-2]
                    if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                        attention_mask = attention_mask.unsqueeze(1)
                    attention_mask = attention_mask.bool()

                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                if query_states.device.type == "cuda" and attention_mask is not None:
                    query_states = query_states.contiguous()
                    key_states = key_states.contiguous()
                    value_states = value_states.contiguous()

                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=attention_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=self.is_causal and attention_mask is None and q_len > 1,
                )
                attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)

        output_sp = self._dense_reduce_scatter_output(attn_output)
        _bsp_g_record_layout(
            config="GS_source",
            layer_id=self.layer_idx,
            event="attention_output",
            output_kind="sp_reduce_scatter",
            output_tokens=output_sp.shape[0],
            expected_sp_tokens=(n_tokens + tp_size - 1) // tp_size,
            expected_tokens=n_tokens,
            tp_size=tp_size,
            bsz=bsz,
            seq_len=q_len,
        )
        attn_weights = None
        return _BSPGAttnSPResult(output_sp, bsz, q_len, bsz * q_len), attn_weights, past_key_value


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->LLaDA2Moe
class LLaDA2MoeFlashAttention2(LLaDA2MoeAttention):
    """
    LLaDA2Moe flash attention module. This module inherits from `LLaDA2MoeAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # LLaDA2MoeFlashAttention2 attention does not support output_attentions
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape

        qkv, _ = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # if self.config.use_qk_norm:
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently cast in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slow down training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LLaDA2MoeRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = self.query_key_value.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output, _ = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            query_length (`int`):
                The length of the query sequence in terms of tokens. This represents the number of tokens in the
                `query_states` tensor along the sequence dimension. It is used to determine the effective sequence
                length for attention computations.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LLaDA2MoeFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention with Llama->LLaDA2Moe
class LLaDA2MoeSdpaAttention(LLaDA2MoeAttention):
    """
    LLaDA2Moe attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LLaDA2MoeAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LLaDA2MoeAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        cache_position: Optional[torch.LongTensor] = None,
        replace_position= None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LLaDA2MoeModel is using LLaDA2MoeSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        # vanilla version
        # qkv = self.query_key_value(hidden_states)
        # qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        # query_states, key_states, value_states = qkv.split(
        #     [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        # )

        #tp version (C11: num_heads is already local after TP partition)
        qkv, _ = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # if self.config.use_qk_norm:
        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings
        # print('shape in sdpa:', query_states.shape, key_states.shape, cos.shape, sin.shape)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # if past_key_value is not None:
        #     cache_kwargs = {"sin": sin, "cos": cos}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, replace_position)
        
        if use_cache:
            past_key_value = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            kv_seq_len = key_states.shape[-2]
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
              attention_mask = attention_mask.unsqueeze(1)
                # raise ValueError(
                #     f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                # )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attention_mask = attention_mask.bool() if attention_mask is not None else None

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output, _ = self.dense(attn_output)

        return attn_output, None, past_key_value


ATTENTION_CLASSES = {
    "eager": LLaDA2MoeAttention,
    "flash_attention_2": LLaDA2MoeFlashAttention2,
    "sdpa": LLaDA2MoeSdpaAttention,
}


class LLaDA2MoeMTPLayer(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.enorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.post_attention_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = LLaDA2MoeSparseMoeBlock(config)

        self.hnorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embeds,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        input_embeds = self.enorm(input_embeds)
        hidden_states = self.hnorm(hidden_states)
        hidden_states = self.eh_proj(torch.cat([input_embeds, hidden_states], dim=-1))
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.to(residual.device)
        hidden_states = self.final_layernorm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class LLaDA2MoeDecoderLayer(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention = ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = (
            LLaDA2MoeSparseMoeBlock(config, prefix=f"model.layers.{layer_idx}.mlp")
            if (config.num_experts is not None and layer_idx >= config.first_k_dense_replace)
            else LLaDA2MoeMLP(config=config, intermediate_size=config.intermediate_size)
        )
        self.input_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_bsp_sequence_parallel_moe = False
        self.bsp_is_last_sparse = False

    def set_bsp_sequence_parallel_moe(self, enabled: bool, is_last_sparse: bool = False):
        self.use_bsp_sequence_parallel_moe = bool(enabled)
        self.bsp_is_last_sparse = bool(is_last_sparse)
        if isinstance(self.mlp, LLaDA2MoeSparseMoeBlock):
            self.mlp.set_sequence_parallel(
                bool(enabled), get_tensor_model_parallel_world_size())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Tuple[torch.Tensor]] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        replace_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*):
                cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        if self.use_bsp_sequence_parallel_moe and isinstance(self.mlp, LLaDA2MoeSparseMoeBlock):
            return self._forward_bsp_g(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                cache_position=cache_position,
                replace_position=replace_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # print(hidden_states.shape)
        # print("attn_mask")
        # print(attention_mask.shape)
        # print("position_ids")
        # print(position_ids.shape)
        hidden_states, self_attn_weights, present_key_value = self.attention(
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

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.to(residual.device)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs

    def _forward_bsp_g(
        self,
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
        from vllm.distributed import (
            get_tensor_model_parallel_world_size,
            tensor_model_parallel_all_gather,
        )
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.models.utils import sequence_parallel_chunk

        if isinstance(hidden_states, _BSPGSPHiddenState):
            sp_state = hidden_states
            bsz = sp_state.bsz
            seq_len = sp_state.seq_len
            n_tokens = sp_state.n_tokens
            residual_sp = sp_state.hidden_sp.view(-1, sp_state.hidden_sp.shape[-1])
            with _bsp_g_time("moe.input_norm_sp"):
                hidden_norm_sp = self.input_layernorm(residual_sp)
            _bsp_g_add_bytes("tp_gather_payload", hidden_norm_sp)
            with _bsp_g_time("moe.tp_all_gather"):
                hidden_flat = tensor_model_parallel_all_gather(hidden_norm_sp, dim=0)
            hidden_states = hidden_flat[:n_tokens].view(bsz, seq_len, -1)
        else:
            bsz, seq_len, h = hidden_states.shape
            n_tokens = bsz * seq_len
            residual_flat = hidden_states.view(-1, h)
            with _bsp_g_time("moe.bsp_chunk"):
                residual_sp = sequence_parallel_chunk(residual_flat)
            hidden_states = self.input_layernorm(hidden_states)

        attn_out, self_attn_weights, present_key_value = self.attention.forward_bsp_g_sp(
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
        if not isinstance(attn_out, _BSPGAttnSPResult):
            raise TypeError(f"Expected _BSPGAttnSPResult, got {type(attn_out)}")

        cfg = _bsp_g_current_vllm_config()
        tp_size = get_tensor_model_parallel_world_size()
        _cached_ctx = _get_cached_forward_context(cfg, n_tokens)
        if _cached_ctx is not None:
            from vllm.forward_context import override_forward_context
            _ctx_mgr = override_forward_context(_cached_ctx)
        else:
            _ctx_mgr = set_forward_context(
                attn_metadata=None,
                vllm_config=cfg,
                num_tokens=n_tokens,
            )
        with _ctx_mgr:
            hidden_after_attn_sp = residual_sp + attn_out.attn_sp.to(residual_sp.device)
            with _bsp_g_time("moe.post_attn_norm_sp"):
                hs_sp = self.post_attention_layernorm(hidden_after_attn_sp)
            y_sp = self.mlp.forward_sp(hs_sp)
            hidden_sp = hidden_after_attn_sp + y_sp.to(hidden_after_attn_sp.device)

        if self.bsp_is_last_sparse:
            if os.environ.get("DINF_SP_LM_HEAD"):
                hidden_states_out = _BSPGSPHiddenState(hidden_sp, bsz, seq_len, n_tokens)
            else:
                _bsp_g_add_bytes("tp_gather_payload", hidden_sp)
                with _bsp_g_time("moe.tp_all_gather"):
                    hidden_flat = tensor_model_parallel_all_gather(hidden_sp, dim=0)
                hidden_states_out = hidden_flat[:n_tokens].view(bsz, seq_len, -1)
        else:
            hidden_states_out = _BSPGSPHiddenState(hidden_sp, bsz, seq_len, n_tokens)

        outputs = (hidden_states_out,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (None,)
        return outputs


LLaDA2Moe_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LLaDA2MoeConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaDA2Moe Model outputting raw hidden-states without any specific head on top.",
    LLaDA2Moe_START_DOCSTRING,
)
class LLaDA2MoePreTrainedModel(PreTrainedModel):
    config_class = LLaDA2MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LLaDA2MoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLaDA2Moe_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaDA2Moe Model outputting raw hidden-states without any specific head on top.",
    LLaDA2Moe_START_DOCSTRING,
)
class LLaDA2MoeModel(LLaDA2MoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LLaDA2MoeDecoderLayer`]

    Args:
        config: LLaDA2MoeConfig
    """

    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_nextn_predict_layers = 0

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = []
        for layer_idx in range(config.num_hidden_layers + self.num_nextn_predict_layers):
            layer_cls = LLaDA2MoeDecoderLayer if layer_idx < config.num_hidden_layers else LLaDA2MoeMTPLayer
            self.layers.append(layer_cls(config, layer_idx))

        self.layers = nn.ModuleList(self.layers)

        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LLaDA2MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def set_bsp_sequence_parallel_moe(self, enabled: bool):
        sparse_layers = [
            layer for layer in self.layers
            if isinstance(getattr(layer, "mlp", None), LLaDA2MoeSparseMoeBlock)
        ]
        last_sparse = sparse_layers[-1] if sparse_layers else None
        for layer in self.layers:
            if hasattr(layer, "set_bsp_sequence_parallel_moe"):
                layer.set_bsp_sequence_parallel_moe(
                    enabled,
                    is_last_sparse=(layer is last_sparse),
                )

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    @add_start_docstrings_to_model_forward(LLaDA2Moe_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None, # add extra cache_position / replace position for dInfer kvcache managerment
        replace_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, MoeV2ModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            with _bsp_g_time("global.embedding"):
                inputs_embeds = self.word_embeddings(input_ids)

        # past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        # if position_ids is None:
            # position_ids = torch.arange(
            #     past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            # )
            # position_ids = position_ids.unsqueeze(0)
        
        # 
        if position_ids is None:
            if replace_position is not None:
                position_ids = torch.arange(replace_position[0], replace_position[1], device=inputs_embeds.device, dtype=torch.long).unsqueeze(0)
            else:
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device, dtype=torch.long).unsqueeze(0)


        # if self._use_flash_attention_2:
        #     # 2d mask is passed through the layers
        #     attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        # elif self._use_sdpa and not output_attentions:
        #     # output_attentions=True can not be supported when using SDPA, and we fall back on
        #     # the manual implementation that requires a 4D causal mask in all cases.
        #     attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
        #         attention_mask,
        #         (batch_size, seq_length),
        #         inputs_embeds,
        #         past_seen_tokens,
        #     )
        # else:
        #     # 4d mask is passed through the layers
        #     attention_mask = _prepare_4d_causal_attention_mask(
        #         attention_mask, (batch_size, seq_length), inputs_embeds, past_seen_tokens
        #     )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        # next_decoder_cache = None
        next_decoder_cache = []
        layers = self.layers[: -self.num_nextn_predict_layers] if self.num_nextn_predict_layers > 0 else self.layers
        mtp_layers = self.layers[-self.num_nextn_predict_layers :] if self.num_nextn_predict_layers > 0 else None

        for decoder_layer in layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    cache_position=cache_position, # add 2 extra cache args
                    replace_position=replace_position,

                )
            hidden_states = layer_outputs[0]

            if use_cache:
                # next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                next_decoder_cache.extend(layer_outputs[2 if output_attentions else 1])


            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        if isinstance(hidden_states, _BSPGSPHiddenState):
            if os.environ.get("DINF_SP_LM_HEAD"):
                hidden_sp = hidden_states.hidden_sp
                hidden_sp = self.norm(hidden_sp)
                main_hidden_states = _BSPGSPHiddenState(
                    hidden_sp, hidden_states.bsz, hidden_states.seq_len,
                    hidden_states.n_tokens)
            else:
                hidden_states = _bsp_g_gather_sp_hidden(hidden_states)
                hidden_states = self.norm(hidden_states)
                main_hidden_states = hidden_states
        else:
            if os.environ.get("DINF_SP_LM_HEAD") and not getattr(self, '_norm_sp_warned', False):
                import torch.distributed as _d
                if _d.is_initialized() and _d.get_rank() == 0:
                    print(f"  [SP-LM] norm: hidden is {type(hidden_states).__name__} shape={getattr(hidden_states,'shape','?')}")
                self._norm_sp_warned = True
            hidden_states = self.norm(hidden_states)
            main_hidden_states = hidden_states

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (main_hidden_states,)


        mtp_hidden_states = None

        if mtp_layers:
            for decoder_layer in mtp_layers:
                input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1)
                inputs_embeds = self.word_embeddings(input_ids)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        inputs_embeds,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        output_router_logits,
                        use_cache,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        inputs_embeds,
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                    )
                if mtp_hidden_states is None:
                    mtp_hidden_states = []
                hidden_states = layer_outputs[0]
                mtp_hidden_states.append(hidden_states)

                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                if output_router_logits and layer_outputs[-1] is not None:
                    all_router_logits += (layer_outputs[-1],)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache

        if not return_dict:
            return tuple(
                v
                for v in [main_hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeV2ModelOutputWithPast(
            last_hidden_state=main_hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            mtp_hidden_states=mtp_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class LLaDA2MoeModelLM(LLaDA2MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__(config)
        self.model = LLaDA2MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_nextn_predict_layers = 0
        self.mtp_loss_scaling_factor = 0

        # # Initialize weights and apply final processing
        # self.post_init()
        # Initialize weights and apply final processing
        self._tp_size = 1
        self.post_init()
        self._tp_plan = {
            "layers.*.attention.query_key_value": "qkv",
            "layers.*.attention.dense": "rowwise",
        }
        # Optional EPLB runtime state (disabled by default).
        self._eplb_enabled = False
        self._eplb_expert_load_view: Optional[torch.Tensor] = None
        self._eplb_logical_to_physical_map: Optional[torch.Tensor] = None
        self._eplb_logical_replica_count: Optional[torch.Tensor] = None
        self.init_h2e_module()

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def set_bsp_sequence_parallel_moe(self, enabled: bool):
        self.model.set_bsp_sequence_parallel_moe(enabled)

    def set_eplb_runtime_state(
        self,
        *,
        enable_eplb: bool,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ):
        """Set EPLB runtime state for all sparse MoE layers.

        The tensors follow vLLM FusedMoE set_eplb_state convention:
        - expert_load_view: [num_sparse_layers, ...]
        - logical_to_physical_map: [num_sparse_layers, ...]
        - logical_replica_count: [num_sparse_layers, ...]
        """
        self._eplb_enabled = bool(enable_eplb)
        self._eplb_expert_load_view = expert_load_view
        self._eplb_logical_to_physical_map = logical_to_physical_map
        self._eplb_logical_replica_count = logical_replica_count

        sparse_layers = [
            li for li in range(self.config.num_hidden_layers)
            if li >= self.config.first_k_dense_replace
        ]

        for local_i, layer_id in enumerate(sparse_layers):
            mlp = self.model.layers[layer_id].mlp
            if not isinstance(mlp, LLaDA2MoeSparseMoeBlock):
                continue
            experts = mlp.experts
            experts.enable_eplb = self._eplb_enabled
            if not self._eplb_enabled:
                continue
            if (
                expert_load_view is None
                or logical_to_physical_map is None
                or logical_replica_count is None
            ):
                raise ValueError(
                    "enable_eplb=True requires expert_load_view, "
                    "logical_to_physical_map, logical_replica_count"
                )
            experts.set_eplb_state(
                moe_layer_idx=local_i,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def build_and_set_eplb_runtime_state(
        self,
        *,
        num_redundant_experts: int = 0,
        expert_load_window_size: int = 16,
        expert_rearrangement_step_interval: int = 16,
        device: Optional[torch.device] = None,
    ):
        """Build initial EPLB tensors via vLLM EplbState and register them."""
        if device is None:
            device = next(self.parameters()).device

        sparse_layer_ids = [
            li for li in range(self.config.num_hidden_layers)
            if li >= self.config.first_k_dense_replace
        ]
        num_sparse_layers = len(sparse_layer_ids)
        num_logical_experts = int(self.config.num_experts)
        _, ep_size = _get_ep_rank_and_size()
        aligned_redundant = _align_redundant_for_ep(
            num_logical_experts=num_logical_experts,
            num_redundant_experts=int(num_redundant_experts),
            ep_size=ep_size,
        )
        if int(num_redundant_experts) != aligned_redundant:
            if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                print(
                    "[EPLB] align runtime redundant experts for EP divisibility: "
                    f"request={int(num_redundant_experts)}, aligned={aligned_redundant}, ep_size={ep_size}"
                )
        num_physical_experts = num_logical_experts + aligned_redundant

        class _EplbModelProxy:
            def set_eplb_state(
                self,
                expert_load_view: torch.Tensor,
                logical_to_physical_map: torch.Tensor,
                logical_replica_count: torch.Tensor,
            ):
                self.expert_load_view = expert_load_view
                self.logical_to_physical_map = logical_to_physical_map
                self.logical_replica_count = logical_replica_count

        proxy = _EplbModelProxy()
        proxy.num_moe_layers = num_sparse_layers
        proxy.num_logical_experts = num_logical_experts
        proxy.num_routed_experts = num_logical_experts
        proxy.num_redundant_experts = aligned_redundant
        proxy.num_physical_experts = num_physical_experts
        proxy.num_local_physical_experts = num_physical_experts
        proxy.num_expert_groups = 1
        proxy.num_shared_experts = 0
        proxy.local_physical_experts = None

        from vllm.config import get_current_vllm_config
        from vllm.distributed.eplb.eplb_state import EplbState

        cfg = get_current_vllm_config()
        if cfg is None:
            raise RuntimeError(
                "build_and_set_eplb_runtime_state requires current vLLM config."
            )

        # Keep EPLB state build consistent with dInfer-side runtime knobs.
        cfg.parallel_config.eplb_config.window_size = int(expert_load_window_size)
        cfg.parallel_config.eplb_config.step_interval = int(
            expert_rearrangement_step_interval
        )

        state = EplbState.build(
            model=proxy,
            device=device,
            parallel_config=cfg.parallel_config,
        )

        def _normalize_layered_tensor(
            t: torch.Tensor, *, target_layers: int, name: str
        ) -> torch.Tensor:
            if t.dim() == 1:
                return t.unsqueeze(0).repeat(target_layers, 1)
            if t.dim() == 2:
                if t.shape[0] == target_layers:
                    return t
                if t.shape[0] == 1:
                    return t.repeat(target_layers, 1)
                return t.unsqueeze(0).repeat(target_layers, 1, 1)
            if t.dim() == 3:
                if t.shape[0] == target_layers:
                    return t
                if t.shape[0] == 1:
                    return t.repeat(target_layers, 1, 1)
            raise RuntimeError(
                f"Unexpected EPLB tensor shape for {name}: {tuple(t.shape)}"
            )

        l2p = _normalize_layered_tensor(
            state.logical_to_physical_map,
            target_layers=num_sparse_layers,
            name="logical_to_physical_map",
        )
        lrc = _normalize_layered_tensor(
            state.logical_replica_count,
            target_layers=num_sparse_layers,
            name="logical_replica_count",
        )
        elv = _normalize_layered_tensor(
            state.expert_load_pass,
            target_layers=num_sparse_layers,
            name="expert_load_pass",
        )

        # Optional: override redundant logical ids (for pid>=num_logical) using
        # offline hot-expert prior; keep base pid[0:num_logical) unchanged.
        num_redundant_effective = int(max(0, num_physical_experts - num_logical_experts))
        custom_replica_ids = _load_custom_redundant_logical_ids(
            num_logical_experts=num_logical_experts,
            num_redundant_experts=num_redundant_effective,
        )
        if custom_replica_ids is not None and num_redundant_effective > 0:
            p2l = _normalize_layered_tensor(
                state.physical_to_logical_map,
                target_layers=num_sparse_layers,
                name="physical_to_logical_map",
            ).to(torch.long)
            p2l = p2l.clone()
            tail = torch.as_tensor(
                custom_replica_ids,
                dtype=p2l.dtype,
                device=p2l.device,
            ).view(1, -1).expand(num_sparse_layers, -1)
            p2l[:, num_logical_experts:num_logical_experts + num_redundant_effective] = tail
            l2p, lrc = _build_layered_l2p_lrc_from_p2l(
                physical_to_logical_map=p2l,
                num_logical_experts=num_logical_experts,
                max_slots_per_logical=int(l2p.shape[-1]),
                l2p_dtype=l2p.dtype,
                lrc_dtype=lrc.dtype,
            )
            state.physical_to_logical_map = p2l
            state.logical_to_physical_map = l2p
            state.logical_replica_count = lrc

        # dInfer currently loads routed-expert weights for logical experts only.
        # If EPLB asks for extra physical expert ids (redundant replicas) without
        # corresponding weights materialized in FusedMoE, collapse maps to the
        # in-range base experts to preserve correctness and avoid OOB routing.
        layer_global_caps = []
        for layer_id in sparse_layer_ids:
            mlp = self.model.layers[layer_id].mlp
            if isinstance(mlp, LLaDA2MoeSparseMoeBlock):
                layer_global_caps.append(
                    int(getattr(mlp.experts, "global_num_experts", num_logical_experts))
                )
        max_supported_physical = (
            min(layer_global_caps) if layer_global_caps else num_logical_experts
        )
        self._eplb_runtime_redundant_effective = bool(
            num_physical_experts <= max_supported_physical
        )
        if num_physical_experts > max_supported_physical:
            if max_supported_physical <= 0:
                raise RuntimeError(
                    f"Invalid max_supported_physical={max_supported_physical}"
                )
            logical_ids = torch.arange(
                l2p.shape[1], device=l2p.device, dtype=l2p.dtype
            ).unsqueeze(0).expand(l2p.shape[0], -1)
            fallback_phys = logical_ids.clamp_max(max_supported_physical - 1)

            valid = (l2p >= 0) & (l2p < max_supported_physical)
            l2p = torch.where(
                valid,
                l2p,
                fallback_phys.unsqueeze(-1).expand_as(l2p),
            )
            lrc = torch.ones_like(lrc)
            if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                print(
                    "[EPLB] fallback: requested redundant experts exceed loaded "
                    f"physical capacity ({num_physical_experts} > {max_supported_physical}); "
                    "collapse to in-range logical experts."
                )

        self._eplb_expert_load_view = elv
        self._eplb_logical_to_physical_map = l2p
        self._eplb_logical_replica_count = lrc
        self._eplb_state = state
        self._eplb_sparse_layer_ids = sparse_layer_ids
        self._eplb_num_physical_experts = num_physical_experts
        self._eplb_num_redundant_experts_aligned = aligned_redundant
        self._eplb_expert_load_window_size = int(expert_load_window_size)
        self._eplb_expert_rearrangement_step_interval = int(
            expert_rearrangement_step_interval
        )

        self.set_eplb_runtime_state(
            enable_eplb=True,
            expert_load_view=elv,
            logical_to_physical_map=l2p,
            logical_replica_count=lrc,
        )

    def get_eplb_runtime_state(self):
        if not self._eplb_enabled:
            return None
        return {
            "expert_load_view": self._eplb_expert_load_view,
            "logical_to_physical_map": self._eplb_logical_to_physical_map,
            "logical_replica_count": self._eplb_logical_replica_count,
            "sparse_layer_ids": getattr(self, "_eplb_sparse_layer_ids", None),
            "num_physical_experts": getattr(self, "_eplb_num_physical_experts", None),
            "num_redundant_experts_aligned": getattr(
                self, "_eplb_num_redundant_experts_aligned", None
            ),
        }

    def _load_external_expert_map(
        self,
        *,
        model_dir: str,
        num_layers: int,
        num_experts: int,
        ep_size: int,
        sparse_layer_ids: list[int],
    ) -> Optional[torch.Tensor]:
        """Load optional per-layer expert->ep_rank map.

        Priority:
        1) DINF_EPLB_EXPERT_MAP_PATH (absolute/relative file path)
        2) <model_dir>/expert_map_ep{ep_size}.pt
        3) <model_dir>/expert_map.pt

        Returns:
            Tensor[int64] with shape [num_layers, num_experts], or None if not found/invalid.
            For sparse-only map files of shape [num_sparse_layers, num_experts], rows
            are mapped onto sparse_layer_ids.
        """
        env_path = os.environ.get("DINF_EPLB_EXPERT_MAP_PATH", "").strip()
        candidates = []
        if env_path:
            candidates.append(Path(env_path))
        model_root = Path(model_dir)
        candidates.append(model_root / f"expert_map_ep{ep_size}.pt")
        candidates.append(model_root / "expert_map.pt")

        for p in candidates:
            if not p.exists():
                continue
            try:
                raw = torch.load(p, map_location="cpu")
                if not isinstance(raw, torch.Tensor):
                    print(f"[EPLB] ignore non-tensor expert map: {p}")
                    continue
                emap = raw.to(torch.int64)
                if emap.ndim == 1 and emap.numel() == num_experts:
                    emap = emap.unsqueeze(0).repeat(num_layers, 1)
                if emap.shape != (num_layers, num_experts):
                    if emap.shape == (len(sparse_layer_ids), num_experts):
                        full = torch.zeros(
                            (num_layers, num_experts), dtype=torch.int64
                        )
                        # Dense layers are ignored by MoE path; fill with linear
                        # placeholder so shape stays uniform.
                        for e in range(num_experts):
                            full[:, e] = e // (num_experts // ep_size)
                        for row_i, layer_id in enumerate(sparse_layer_ids):
                            full[layer_id] = emap[row_i]
                        emap = full
                    else:
                        print(
                            f"[EPLB] ignore map with invalid shape {tuple(emap.shape)}; "
                            f"expect ({num_layers}, {num_experts}) or "
                            f"({len(sparse_layer_ids)}, {num_experts}): {p}"
                        )
                        continue
                mn = int(emap.min().item())
                mx = int(emap.max().item())
                if mn < 0 or mx >= ep_size:
                    print(
                        f"[EPLB] ignore map out of ep range [0,{ep_size-1}] "
                        f"(min={mn}, max={mx}): {p}"
                    )
                    continue
                print(f"[EPLB] loaded expert map: {p}")
                return emap
            except Exception as e:
                print(f"[EPLB] failed loading map {p}: {e}")
        return None
        
    def load_state_dict(self, model_dir, strict=True, dtype=torch.bfloat16, device=None):
        num_experts = self.config.num_experts
        moe_intermediate_size = self.config.moe_intermediate_size
        num_layers = self.config.num_hidden_layers
        ep_rank, ep_size = _get_ep_rank_and_size()
        example_sparse_layer = self.model.layers[self.config.first_k_dense_replace].mlp.experts
        num_local_physical_experts = int(getattr(example_sparse_layer, "local_num_experts", num_experts // ep_size))
        global_num_physical_experts = int(getattr(example_sparse_layer, "global_num_experts", num_experts))
        if global_num_physical_experts < num_experts:
            raise RuntimeError(
                f"Invalid global_num_physical_experts={global_num_physical_experts} < logical={num_experts}"
            )
        physical_to_logical, custom_replica_enabled = _build_global_physical_to_logical_map(
            num_logical_experts=num_experts,
            num_redundant_experts=global_num_physical_experts - num_experts,
        )
        if global_num_physical_experts % ep_size != 0:
            raise RuntimeError(
                f"[EPLB] global_num_physical_experts={global_num_physical_experts} "
                f"must be divisible by ep_size={ep_size}"
            )
        rank_to_linear_local_phys = _build_rank_to_local_phys_ids(
            global_num_physical_experts=global_num_physical_experts,
            ep_size=ep_size,
        )
        local_physical_ids = list(rank_to_linear_local_phys[int(ep_rank)])
        if len(local_physical_ids) != int(num_local_physical_experts):
            raise RuntimeError(
                f"[EPLB] local physical slot mismatch: got={len(local_physical_ids)}, "
                f"expect={num_local_physical_experts}"
            )

        sparse_layer_ids = [
            li for li in range(num_layers) if li >= self.config.first_k_dense_replace
        ]

        # Optional EPLB map. None => default linear placement.
        external_map = self._load_external_expert_map(
            model_dir=model_dir,
            num_layers=num_layers,
            num_experts=num_experts,
            ep_size=ep_size,
            sparse_layer_ids=sparse_layer_ids,
        )
        init_placement_mode = _get_eplb_init_placement_mode()
        init_global_expert_load = None
        vllm_phy2log = None
        if (
            external_map is not None
            and global_num_physical_experts > num_experts
            and init_placement_mode == "vllm_balanced"
        ):
            vllm_phy2log = _load_vllm_balanced_phy2log()
            if vllm_phy2log is None:
                if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                    print(
                        "[EPLB] init placement mode=vllm_balanced but no phy2log found; "
                        "fallback to joint_p1_p5."
                    )
                init_placement_mode = "joint_p1_p5"
            elif tuple(vllm_phy2log.shape) != (num_layers, global_num_physical_experts):
                if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                    print(
                        f"[EPLB] vllm phy2log shape mismatch: got {tuple(vllm_phy2log.shape)}, "
                        f"expect ({num_layers}, {global_num_physical_experts}); "
                        "fallback to joint_p1_p5."
                    )
                vllm_phy2log = None
                init_placement_mode = "joint_p1_p5"
        if (
            external_map is not None
            and global_num_physical_experts > num_experts
            and init_placement_mode == "weight_balance"
        ):
            init_global_expert_load = _load_eplb_init_global_expert_load(
                num_layers=num_layers,
                num_experts=num_experts,
                sparse_layer_ids=sparse_layer_ids,
            )
            if init_global_expert_load is None:
                if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                    print(
                        "[EPLB] init placement mode=weight_balance but no valid "
                        "global expert load prior found; fallback to joint_p1_p5."
                    )
                init_placement_mode = "joint_p1_p5"
        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
            if external_map is not None and global_num_physical_experts > num_experts:
                print(f"[EPLB] init placement mode: {init_placement_mode}")
        if external_map is None:
            per_layer_global_physical = [list(local_physical_ids) for _ in range(num_layers)]
            per_layer_logical_for_local = [
                [int(physical_to_logical[pid]) for pid in local_physical_ids]
                for _ in range(num_layers)
            ]
        else:
            per_layer_global_physical = []
            per_layer_logical_for_local = []
            for layer_id in range(num_layers):
                if layer_id in sparse_layer_ids:
                    if global_num_physical_experts > num_experts:
                        if init_placement_mode == "vllm_balanced" and vllm_phy2log is not None:
                            rank_to_phys = _build_vllm_balanced_local_maps(
                                phy2log=vllm_phy2log,
                                layer_id=layer_id,
                                global_num_physical_experts=global_num_physical_experts,
                                ep_size=ep_size,
                            )
                        elif init_placement_mode == "weight_balance":
                            sparse_idx = sparse_layer_ids.index(layer_id)
                            rank_to_phys = _build_weight_balanced_external_replica_local_maps(
                                external_map=external_map,
                                layer_id=layer_id,
                                num_experts=num_experts,
                                global_num_physical_experts=global_num_physical_experts,
                                ep_size=ep_size,
                                physical_to_logical=physical_to_logical,
                                layer_expert_load=init_global_expert_load[sparse_idx],
                                pin_base_experts=True,
                            )
                        else:
                            rank_to_phys = _build_joint_external_replica_local_maps(
                                external_map=external_map,
                                layer_id=layer_id,
                                num_experts=num_experts,
                                global_num_physical_experts=global_num_physical_experts,
                                ep_size=ep_size,
                            )
                    else:
                        # No redundant physical experts: follow exact P1 assignment.
                        assigned = torch.nonzero(
                            external_map[layer_id].to(torch.int64) == int(ep_rank),
                            as_tuple=False,
                        ).reshape(-1).tolist()
                        assigned = sorted(int(x) for x in assigned)
                        if len(assigned) != int(num_local_physical_experts):
                            raise RuntimeError(
                                f"[EPLB] layer {layer_id}: local expert count mismatch "
                                f"(got {len(assigned)}, expect {num_local_physical_experts})"
                            )
                        rank_to_phys = {r: [] for r in range(int(ep_size))}
                        rank_to_phys[int(ep_rank)] = assigned
                    layer_global_phys = list(rank_to_phys[int(ep_rank)])
                else:
                    # Dense layers keep linear physical placement.
                    layer_global_phys = list(local_physical_ids)

                if len(layer_global_phys) != int(num_local_physical_experts):
                    raise RuntimeError(
                        f"[EPLB] layer {layer_id}: physical slot count mismatch "
                        f"(got {len(layer_global_phys)}, expect {num_local_physical_experts})"
                    )
                if len(set(layer_global_phys)) != len(layer_global_phys):
                    raise RuntimeError(
                        f"[EPLB] layer {layer_id}: duplicated physical ids in local slots"
                    )
                if any((pid < 0 or pid >= global_num_physical_experts) for pid in layer_global_phys):
                    raise RuntimeError(
                        f"[EPLB] layer {layer_id}: out-of-range physical ids detected"
                    )
                per_layer_global_physical.append(layer_global_phys)
                if init_placement_mode == "vllm_balanced" and vllm_phy2log is not None:
                    per_layer_logical_for_local.append(
                        [int(vllm_phy2log[layer_id, pid].item()) for pid in layer_global_phys]
                    )
                else:
                    per_layer_logical_for_local.append(
                        [int(physical_to_logical[pid]) for pid in layer_global_phys]
                    )
            # Joint-map invariants: no rank loses all replicas of any logical expert.
            if global_num_physical_experts > num_experts:
                logical_replica_total = {}
                for pid in range(global_num_physical_experts):
                    lid = int(physical_to_logical[pid])
                    logical_replica_total[lid] = logical_replica_total.get(lid, 0) + 1
                for layer_id in sparse_layer_ids:
                    # Per-rank local physical ids for this layer.
                    rank_to_layer_phys: dict[int, list[int]] = {r: [] for r in range(int(ep_size))}
                    sparse_idx = sparse_layer_ids.index(layer_id)
                    for r in range(int(ep_size)):
                        if init_placement_mode == "weight_balance":
                            rank_to_layer_phys[r] = _build_weight_balanced_external_replica_local_maps(
                                external_map=external_map,
                                layer_id=layer_id,
                                num_experts=num_experts,
                                global_num_physical_experts=global_num_physical_experts,
                                ep_size=ep_size,
                                physical_to_logical=physical_to_logical,
                                layer_expert_load=init_global_expert_load[sparse_idx],
                                pin_base_experts=True,
                            )[r]
                        else:
                            rank_to_layer_phys[r] = _build_joint_external_replica_local_maps(
                                external_map=external_map,
                                layer_id=layer_id,
                                num_experts=num_experts,
                                global_num_physical_experts=global_num_physical_experts,
                                ep_size=ep_size,
                            )[r]
                    # Check all physical ids are assigned exactly once globally.
                    flat = [pid for r in range(int(ep_size)) for pid in rank_to_layer_phys[r]]
                    if len(flat) != global_num_physical_experts or len(set(flat)) != global_num_physical_experts:
                        raise RuntimeError(
                            f"[EPLB] layer {layer_id}: global physical assignment is not a bijection"
                        )
                    # Check each logical expert has expected replica count.
                    logical_seen = {}
                    for pid in flat:
                        lid = int(physical_to_logical[pid])
                        logical_seen[lid] = logical_seen.get(lid, 0) + 1
                    for lid in range(num_experts):
                        if logical_seen.get(lid, 0) != logical_replica_total.get(lid, 0):
                            raise RuntimeError(
                                f"[EPLB] layer {layer_id}: logical replica count mismatch for lid={lid}, "
                                f"seen={logical_seen.get(lid, 0)}, expect={logical_replica_total.get(lid, 0)}"
                            )
        # Index by logical expert id for checkpoint filtering.
        per_layer_logical_set = [set(x) for x in per_layer_logical_for_local]
        index_path = Path(model_dir) / "model.safetensors.index.json"
        with open(index_path, "r") as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        shard_files = {v for v in weight_map.values()}

        state_dict = {}
        # print(shard_files)
        for shard in tqdm.tqdm(sorted(shard_files)):
            shard_path = Path(model_dir) / shard
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard: {shard_path}")
            
            with torch.inference_mode():
                file_state_dict = load_file(str(shard_path))
                filtered_file_state_dict = {}
                for key, value in file_state_dict.items():
                    if ".mlp.experts." in key:
                        layer_id = int(key.split(".mlp.experts.")[0].split(".")[-1])
                        expert_id = int(key.split(".mlp.experts.")[1].split(".")[0])
                        if expert_id in per_layer_logical_set[layer_id]:
                            filtered_file_state_dict[key] = value
                    else:
                        filtered_file_state_dict[key] = value
                            
                        
                state_dict.update(filtered_file_state_dict)

        new_state_dict = {}
        gate_projs = [{} for _ in range(num_layers)]
        up_projs = [{} for _ in range(num_layers)]
        down_projs = [{} for _ in range(num_layers)]
        for key, value in tqdm.tqdm(state_dict.items()):
            if ".mlp.experts." in key:
                layer_id = int(key.split(".mlp.experts.")[0].split(".")[-1])
                expert_id = int(key.split(".mlp.experts.")[1].split(".")[0])
                if layer_id < num_layers:
                    if expert_id not in per_layer_logical_set[layer_id]:
                        continue
                    if "gate_proj" in key:
                        gate_projs[layer_id][expert_id] = value
                    elif "up_proj" in key:
                        up_projs[layer_id][expert_id] = value
                    elif "down_proj" in key:
                        down_projs[layer_id][expert_id] = value
            else:
                new_state_dict[key] = value

        del state_dict
        sparse_layer_set = set(sparse_layer_ids)
        for layer_id in tqdm.trange(num_layers):
            if f"model.layers.{layer_id}.mlp.w1" in new_state_dict.keys():
                ep_rank, ep_size = _get_ep_rank_and_size()
                size = divide(new_state_dict[f"model.layers.{layer_id}.mlp.w1"].shape[0], ep_size)
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight"] = new_state_dict[f"model.layers.{layer_id}.mlp.w1"][ep_rank*size:(ep_rank+1)*size]
                new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight"] = new_state_dict[f"model.layers.{layer_id}.mlp.w2"][ep_rank*size:(ep_rank+1)*size]
                del new_state_dict[f"model.layers.{layer_id}.mlp.w1"]
                del new_state_dict[f"model.layers.{layer_id}.mlp.w2"]
            else:
                if layer_id not in sparse_layer_set:
                    # Dense MLP layers do not use routed experts.
                    continue
                w13_weight = []
                w2_weight = []
                if per_layer_logical_for_local[layer_id]:
                    missing = [
                        lid for lid in set(per_layer_logical_for_local[layer_id])
                        if lid not in gate_projs[layer_id]
                        or lid not in up_projs[layer_id]
                        or lid not in down_projs[layer_id]
                    ]
                    if missing:
                        raise RuntimeError(
                            f"[EPLB] layer {layer_id}: missing logical expert weights for ids={sorted(missing)[:8]}"
                        )
                    for local_idx in range(num_local_physical_experts):
                        logical_id = int(per_layer_logical_for_local[layer_id][local_idx])
                        gate_proj = gate_projs[layer_id][logical_id].to(device)
                        up_proj = up_projs[layer_id][logical_id].to(device)
                        down_proj = down_projs[layer_id][logical_id].to(device)
                        w13_weight.append(torch.cat([gate_proj, up_proj], dim=0))
                        w2_weight.append(down_proj)
                    w13_weight = torch.stack(w13_weight, dim=0)
                    w2_weight = torch.stack(w2_weight, dim=0)
                    new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight"] = w13_weight.contiguous().to(device)
                    new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight"] = w2_weight.contiguous().to(device)
                    del w13_weight, w2_weight

        # Materialize per-layer expert_map directly into live modules.
        # FusedMoE expert_map is a buffer under layer.mlp.experts.expert_map
        for layer_id in sparse_layer_ids:
            emap = torch.full(
                (global_num_physical_experts,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            for local_idx, global_phys_id in enumerate(per_layer_global_physical[layer_id]):
                emap[int(global_phys_id)] = int(local_idx)
            self.model.layers[layer_id].mlp.experts.expert_map.data.copy_(emap)


        # print("====new_state_dict")
        # for key, value in new_state_dict.items():
        #     # if int(key.split(".")[3])<num_layers:
        #     print(key, value.shape, value.dtype)

        # print("====self.state_dict")
        # for key, value in self.state_dict().items():
        #     print(key, value.shape, value.dtype)

        new_state_dict_keys = new_state_dict.keys()
        self_state_dict_keys = self.state_dict().keys()
        
        if os.environ.get("DINF_LOAD_DEBUG", "0") == "1":
            unused_keys = []
            for key in new_state_dict_keys:
                if key not in self_state_dict_keys:
                    unused_keys.append(key)

            not_inited_keys = []
            for key in self_state_dict_keys:
                if key not in new_state_dict_keys:
                    not_inited_keys.append(key)

            print("unused_keys", unused_keys)
            print("not_inited_keys", not_inited_keys)
        
        

        # 调用父类方法加载
        # super().load_state_dict(new_state_dict, strict=strict)
        for key, value in tqdm.tqdm(new_state_dict.items()):
            new_state_dict[key] = value.to(device)
        params_dict = dict(self.named_parameters())
        buffer_dict = dict(self.named_buffers())

        # C11: TP attention weight sharding
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        head_dim = self.config.head_dim or (self.config.hidden_size // self.config.num_attention_heads)
        total_q_dim = self.config.num_attention_heads * head_dim
        total_kv_dim = self.config.num_key_value_heads * head_dim
        local_q_dim = total_q_dim // tp_size
        if tp_size >= self.config.num_key_value_heads:
            num_kv_head_replicas = tp_size // self.config.num_key_value_heads
            kv_shard_rank = tp_rank // num_kv_head_replicas
            local_kv_heads = 1
        else:
            num_kv_head_replicas = 1
            kv_shard_rank = tp_rank
            local_kv_heads = self.config.num_key_value_heads // tp_size
        local_kv_dim = local_kv_heads * head_dim

        for name, loaded_weight in new_state_dict.items():
            if name in params_dict:
                param = params_dict[name]

                if tp_size > 1 and '.query_key_value.weight' in name:
                    # Combined QKV [total_qkv, hidden] -> TP shard
                    q_full = loaded_weight[:total_q_dim]
                    k_full = loaded_weight[total_q_dim : total_q_dim + total_kv_dim]
                    v_full = loaded_weight[total_q_dim + total_kv_dim :]
                    q_local = q_full[tp_rank * local_q_dim : (tp_rank + 1) * local_q_dim]
                    k_local = k_full[kv_shard_rank * local_kv_dim : (kv_shard_rank + 1) * local_kv_dim]
                    v_local = v_full[kv_shard_rank * local_kv_dim : (kv_shard_rank + 1) * local_kv_dim]
                    param.data.copy_(torch.cat([q_local, k_local, v_local], dim=0))
                elif tp_size > 1 and '.dense.weight' in name and '.attention.' in name:
                    # O projection [hidden, total_q] -> shard columns
                    param.data.copy_(loaded_weight[:, tp_rank * local_q_dim : (tp_rank + 1) * local_q_dim])
                else:
                    param.data = loaded_weight

            elif name in buffer_dict:
                buffer = buffer_dict[name]
                buffer.data = loaded_weight
            else:
                print('params not matching:', name)
        # super().load_state_dict(new_state_dict, strict=strict)
        for name, param in self.named_parameters():
            if '.mlp.gate.expert_bias' in name:
                param.data = param.data.to(torch.float32)
            else:
                param.data = param.data.to(dtype)
    
            


    def load_sharded_safetensors(self, model_dir):
        index_path = Path(model_dir) / "model.safetensors.index.json"
        with open(index_path, "r") as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        shard_files = {v for v in weight_map.values()}

        state_dict = {}
        for shard in sorted(shard_files):  
            shard_path = Path(model_dir) / shard
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard: {shard_path}")
            
            with torch.inference_mode():
                state_dict.update(load_file(str(shard_path)))

        return state_dict

    def init_h2e_module(self):
        self.h2e = H2Embed(self.model.word_embeddings, tau=1.0)

    def load_weights(self, model_path, torch_dtype = torch.bfloat16, device=None):
        self.load_state_dict(model_path, strict=False, dtype=torch_dtype, device=device)
        self.init_h2e_module()


    @add_start_docstrings_to_model_forward(LLaDA2Moe_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoEV2CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        replace_position: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[KVCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, MoEV2CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer

        >>> model = LLaDA2MoeForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists`` of (dec_features, layer_state, dec_hidden, dec_attn)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            replace_position=replace_position,
            **kwargs,
        )

        loss = None
        all_mtp_loss = None
        aux_loss = None
        hidden_states = outputs[0]

        # SP LM Head path: compute lm_head on SP layout, local decode, all-gather tokens
        _sp_lm_env = os.environ.get("DINF_SP_LM_HEAD")
        _is_sp = isinstance(hidden_states, _BSPGSPHiddenState)
        if _sp_lm_env and not _is_sp and not getattr(self, '_sp_lm_warned', False):
            import torch.distributed as _dist
            if _dist.is_initialized() and _dist.get_rank() == 0:
                print(f"  [SP-LM] WARNING: hidden_states is {type(hidden_states).__name__}, not _BSPGSPHiddenState")
            self._sp_lm_warned = True
        if _is_sp and _sp_lm_env:
            sp_state = hidden_states
            hidden_sp = sp_state.hidden_sp  # [N_sp, hidden]

            with _bsp_g_time("global.lm_head_sp"):
                logits_sp = self.lm_head(hidden_sp)  # [N_sp, vocab]

            # Return SP logits + metadata. Decoder handles local decode + gather.
            logits = (logits_sp, sp_state)
        else:
            _selective_lm = os.environ.get("DINF_SELECTIVE_LM_HEAD")
            if _selective_lm and input_ids is not None:
                _mask_id = int(_selective_lm) if _selective_lm != "1" else 156895
                mask_pos = (input_ids == _mask_id)
                num_mask = mask_pos.sum().item()
                total_tokens = input_ids.numel()
                if 0 < num_mask < total_tokens:
                    with _bsp_g_time("global.lm_head"):
                        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
                        mask_idx = mask_pos.view(-1).nonzero(as_tuple=True)[0]
                        logits_mask = self.lm_head(hidden_flat[mask_idx])
                        logits = hidden_flat.new_zeros(hidden_flat.shape[0], logits_mask.shape[-1])
                        logits[mask_idx] = logits_mask
                        logits = logits.view(*hidden_states.shape[:-1], -1)
                else:
                    with _bsp_g_time("global.lm_head"):
                        logits = self.lm_head(hidden_states)
            else:
                with _bsp_g_time("global.lm_head"):
                    logits = self.lm_head(hidden_states)
            if not os.environ.get("DINF_SKIP_LOGITS_FLOAT"):
                with _bsp_g_time("global.logits_float"):
                    logits = logits.float()

        if labels is not None:
            loss = self.loss_function(logits, labels, self.config.vocab_size, **kwargs)

        all_mtp_logits = None
        if self.num_nextn_predict_layers > 0:
            mtp_hidden_states = outputs.mtp_hidden_states
            shift_labels_mtp = None
            for i in range(self.num_nextn_predict_layers):
                mtp_hidden_states = mtp_hidden_states[i]
                mtp_logits = self.lm_head(mtp_hidden_states).float()
                if all_mtp_logits is None:
                    all_mtp_logits = []
                all_mtp_logits.append(mtp_logits)
                if labels is not None:
                    if shift_labels_mtp is None:
                        shift_labels_mtp = labels.clone()
                    shift_labels_mtp, _ = roll_tensor(shift_labels_mtp, shifts=-1, dims=-1, fill_value=-100)
                    mtp_logits_ = mtp_logits.view(-1, self.config.vocab_size)
                    mtp_loss = self.loss_function(mtp_logits_, shift_labels_mtp.to(mtp_logits_.device).view(-1), self.config.vocab_size, **kwargs)
                    if loss is not None:
                        loss += self.mtp_loss_scaling_factor * mtp_loss
                    else:
                        loss = self.mtp_loss_scaling_factor * mtp_loss

                    if all_mtp_loss is None:
                        all_mtp_loss = []
                    all_mtp_loss.append(mtp_loss)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        past_key_values = KVCache(outputs.past_key_values) if outputs.past_key_values is not None else None

        return MoEV2CausalLMOutputWithPast(
            loss=loss,
            mtp_loss=all_mtp_loss,
            aux_loss=aux_loss,
            logits=logits,
            mtp_logits=all_mtp_logits,
            # past_key_values=outputs.past_key_values,
            past_key_values=past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
    def tensor_parallel(self, tp_size):
        """
        Apply the model's tensor parallelization plan.
        Currently only supports linear layers.
        """
        tp_plan = self._tp_plan
        self._tp_size = tp_size

        def _tensor_parallel(module: nn.Module, prefix: str = ""):
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)
                # print(qual_name)
                for pattern, style in tp_plan.items():
                    if re.match(pattern, qual_name) and isinstance(
                            child_module, nn.Linear):
                        new_module = replace_linear_class(
                            child_module, style, None, self.config)
                        dtype = child_module.weight.dtype
                        new_module.weight_loader(new_module.weight, child_module.weight)
                        new_module.weight.data = new_module.weight.data.to(dtype)
                        setattr(module, child_name, new_module)
                        break
                    else:
                        _tensor_parallel(child_module, prefix=qual_name)
                if '.attention' in qual_name and len(qual_name.split('.'))==3:
                    child_module.tp_size = tp_size
            self.h2e.sp_size = tp_size
                # if qual_name == "transformer.ff_out":
                #     new_module = ColumnParallelLinear(child_module.in_features, child_module.out_features, False, True, return_bias=False)
                #     new_module.weight_loader(new_module.weight, child_module.weight)
                #     setattr(module, child_name, new_module)
                    
        _tensor_parallel(self.model)
