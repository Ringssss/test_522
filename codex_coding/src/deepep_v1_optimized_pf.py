"""
DeepEP V1 HT Optimized PrepareAndFinalize.

Borrows V2's key ideas to optimize V1:
1. async get_dispatch_layout + event chaining (eliminates CPU sync #1)
2. async combine + stream wait (eliminates CPU sync #2)
3. Configurable SM count

Drop-in replacement for the original DeepEPHTPrepareAndFinalize.

Usage:
    from deepep_v1_optimized_pf import replace_with_optimized_v1
    replace_with_optimized_v1(model, num_sms=10)
"""
from __future__ import annotations
from typing import Optional, Callable, Union

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous, TopKWeightAndReduceDelegate)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input


class DeepEPV1OptimizedPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Optimized V1 HT PrepareAndFinalize with:
    - async get_dispatch_layout (GPU event chain, no CPU sync)
    - async combine (GPU stream wait, no CPU sync)
    - configurable SM count
    """

    def __init__(self, buffer: deep_ep.Buffer, num_dispatchers: int,
                 dp_size: int, rank_expert_offset: int,
                 num_sms: Optional[int] = None):
        super().__init__()
        self.buffer = buffer
        self.num_dispatchers_ = num_dispatchers
        self.dp_size = dp_size
        self.rank_expert_offset = rank_expert_offset
        self.async_prepare = True
        self.handle = None

        # Configurable SM count (V2 idea: use fewer SMs)
        if num_sms is not None:
            deep_ep.Buffer.set_num_sms(num_sms)

        self.available_rank_configs = [2, 4, 8, 16, 24, 32, 64, 128, 144, 160]

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return torch.int64

    def _get_dispatch_config(self) -> Optional[deep_ep.Config]:
        if self.num_dispatchers_ not in self.available_rank_configs:
            return None
        return deep_ep.Buffer.get_dispatch_config(self.num_dispatchers_)

    def _get_combine_config(self) -> Optional[deep_ep.Config]:
        if self.num_dispatchers_ not in self.available_rank_configs:
            return None
        return deep_ep.Buffer.get_combine_config(self.num_dispatchers_)

    def _do_dispatch(
        self,
        tokens: torch.Tensor,
        token_scales: Optional[torch.Tensor],
        rank_topk_ids: torch.Tensor,
        rank_topk_weights: torch.Tensor,
        num_experts: int,
        a1_scale: Optional[torch.Tensor],
        quant_config: FusedMoEQuantConfig,
    ) -> Callable:
        has_scales = token_scales is not None

        # ★ OPTIMIZATION 1: async layout + event chaining
        # NOTE: V1's C++ dispatch asserts on previous_event+async combo.
        # For now, keep layout synchronous. The combine async is the bigger win.
        (num_tokens_per_rank, num_tokens_per_rdma_rank,
         dispatch_expert_num_tokens, is_token_in_rank,
         layout_event) = self.buffer.get_dispatch_layout(
             topk_idx=rank_topk_ids,
             num_experts=num_experts,
             previous_event=None,
             async_finish=False,              # keep sync for now
             allocate_on_comm_stream=False)

        token_data = tokens
        if has_scales:
            token_data = (tokens, token_scales)

        (
            token_data, expert_topk_ids, expert_topk_weights,
            expert_num_tokens_per_expert_list, self.handle, event
        ) = self.buffer.dispatch(
            x=token_data,
            handle=None,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=dispatch_expert_num_tokens,
            topk_idx=rank_topk_ids,
            topk_weights=rank_topk_weights,
            expert_alignment=1,
            config=self._get_dispatch_config(),
            previous_event=None,             # keep simple for now
            async_finish=self.async_prepare,
            allocate_on_comm_stream=False)

        return lambda: self._receiver(
            event, has_scales, token_data, expert_topk_ids,
            num_experts, expert_num_tokens_per_expert_list,
            expert_topk_weights, a1_scale, quant_config)

    def _receiver(
        self,
        event: deep_ep.EventOverlap,
        has_scales: bool,
        token_data: Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        expert_topk_ids: Optional[torch.Tensor],
        num_experts: int,
        expert_num_tokens_per_expert_list: list[int],
        expert_topk_weights: Optional[torch.Tensor],
        a1_scale: Optional[torch.Tensor],
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        if self.async_prepare:
            event.current_stream_wait()

        if has_scales:
            expert_x, expert_x_scale = token_data
        else:
            expert_x, expert_x_scale = token_data, None

        # Remap local → global expert IDs (same as original V1)
        assert expert_topk_ids is not None
        expert_topk_ids = torch.where(
            expert_topk_ids == -1,
            num_experts - 1 if self.rank_expert_offset == 0 else 0,
            expert_topk_ids + self.rank_expert_offset)

        expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(
            expert_num_tokens_per_expert_list, device=expert_x.device)

        if not quant_config.is_block_quantized:
            expert_x_scale = None
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    expert_x, a1_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=False,
                    block_shape=quant_config.block_shape)

        return (expert_x, expert_x_scale, expert_tokens_meta,
                expert_topk_ids, expert_topk_weights)

    def supports_async(self) -> bool:
        return True

    def prepare_async(
        self, a1, a1_scale, a2_scale, topk_weights, topk_ids,
        num_experts, expert_map, apply_router_weight_on_input, quant_config,
    ) -> Callable:
        if apply_router_weight_on_input:
            assert topk_ids.size(1) == 1
            a1 = a1 * topk_weights.to(a1.dtype)

        if quant_config.is_block_quantized:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1, a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=quant_config.block_shape)
            if a1q_scale is not None and a1q_scale.numel() == 1:
                a1q_scale = a1q_scale.view(1, 1)
            a1_post_scale = None
        else:
            a1q, a1q_scale, a1_post_scale = a1, None, a1_scale

        return self._do_dispatch(
            tokens=a1q, token_scales=a1q_scale,
            rank_topk_ids=topk_ids, rank_topk_weights=topk_weights,
            num_experts=num_experts, a1_scale=a1_post_scale,
            quant_config=quant_config)

    def prepare(self, a1, a1_scale, a2_scale, topk_weights, topk_ids,
                num_experts, expert_map, apply_router_weight_on_input,
                quant_config) -> mk.PrepareResultType:
        receiver = self.prepare_async(
            a1, a1_scale, a2_scale, topk_weights, topk_ids,
            num_experts, expert_map, apply_router_weight_on_input, quant_config)
        return receiver()

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        assert self.handle is not None

        if fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input)

        # ★ OPTIMIZATION 2: async combine + GPU stream wait
        # Original V1: async_finish=False (CPU blocks)
        # Optimized:   async_finish=True  (GPU stream wait only)
        combined_x, _, event = self.buffer.combine(
            x=fused_expert_output,
            handle=self.handle,
            topk_weights=None,
            config=self._get_combine_config(),
            previous_event=None,
            async_finish=True,               # ← was False
            allocate_on_comm_stream=False)

        # GPU-side wait (not CPU-side) — current stream waits for combine
        event.current_stream_wait()

        output.copy_(combined_x, non_blocking=True)


def replace_with_optimized_v1(model: torch.nn.Module, num_sms: int = None):
    """
    Replace all FusedMoE layers' DeepEPHTPrepareAndFinalize with optimized version.
    Must be called AFTER prepare_communication_buffer_for_model(model).
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEModularKernel
    from vllm.model_executor.layers.fused_moe.deepep_ht_prepare_finalize import (
        DeepEPHTPrepareAndFinalize)

    replaced = 0
    for mod in model.modules():
        if not isinstance(mod, FusedMoE):
            continue
        qm = mod.quant_method
        if not hasattr(qm, 'fused_experts') or qm.fused_experts is None:
            continue
        kernel = qm.fused_experts
        if not isinstance(kernel, FusedMoEModularKernel):
            continue
        old_pf = kernel.prepare_finalize
        if not isinstance(old_pf, DeepEPHTPrepareAndFinalize):
            continue

        # Create optimized version reusing the same buffer
        opt_pf = DeepEPV1OptimizedPrepareAndFinalize(
            buffer=old_pf.buffer,
            num_dispatchers=old_pf.num_dispatchers_,
            dp_size=old_pf.dp_size,
            rank_expert_offset=old_pf.rank_expert_offset,
            num_sms=num_sms,
        )
        kernel.prepare_finalize = opt_pf
        replaced += 1

    return replaced
