"""
DeepEP V2 ElasticBuffer PrepareAndFinalize for vllm's modular kernel framework.

Replaces V1's DeepEPHTPrepareAndFinalize with V2's ElasticBuffer,
providing: unified dispatch/combine API, fewer SMs, analytical SM calculation.

Usage:
    from deepep_v2_pf import replace_with_deepep_v2
    replace_with_deepep_v2(model, ep_group, num_local_experts=64, num_experts=256, top_k=8)
"""
from __future__ import annotations
from typing import Optional, Callable, Union

import torch
from deep_ep import ElasticBuffer, EPHandle, EventOverlap

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous, TopKWeightAndReduceDelegate)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

# Global V2 buffer singleton
_elastic_buffer: Optional[ElasticBuffer] = None
_num_comm_sms: int = 0


def get_or_create_elastic_buffer(
    ep_cpu_group,
    num_max_tokens_per_rank: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    use_fp8_dispatch: bool = False,
) -> tuple[ElasticBuffer, int]:
    """Create or reuse the global ElasticBuffer singleton."""
    global _elastic_buffer, _num_comm_sms
    if _elastic_buffer is None:
        _elastic_buffer = ElasticBuffer(
            ep_cpu_group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden,
            num_topk=num_topk,
            use_fp8_dispatch=use_fp8_dispatch,
        )
        _num_comm_sms = _elastic_buffer.get_theoretical_num_sms(
            num_experts, num_topk)
    return _elastic_buffer, _num_comm_sms


class DeepEPV2PrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    PrepareAndFinalize using DeepEP V2 ElasticBuffer.
    Drop-in replacement for DeepEPHTPrepareAndFinalize (V1 Buffer).
    """

    def __init__(self, buffer: ElasticBuffer, num_sms: int,
                 dp_size: int, rank_expert_offset: int):
        super().__init__()
        self.buffer = buffer
        self.num_sms = num_sms
        self.dp_size = dp_size
        self.rank_expert_offset = rank_expert_offset
        self.handle: Optional[EPHandle] = None

    def num_dispatchers(self) -> int:
        return self.dp_size

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return torch.int64

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input only implemented for topk=1")
            a1 = a1 * topk_weights.to(a1.dtype)

        # ★ V2 dispatch: single call, no separate get_dispatch_layout
        recv_x, recv_topk_idx, recv_topk_weights, handle, event = \
            self.buffer.dispatch(
                a1,
                topk_idx=topk_ids,
                topk_weights=topk_weights,
                num_experts=num_experts,
                num_max_tokens_per_rank=a1.shape[0],
                expert_alignment=1,
                num_sms=self.num_sms,
                async_with_compute_stream=False,
                allocate_on_comm_stream=False,
            )
        if event.event is not None:
            event.current_stream_wait()
        self.handle = handle

        expert_x = recv_x
        expert_x_scale = None

        # Remap expert IDs: local → global (same logic as V1)
        # DeepEP dispatch returns local expert indices; offset to global space
        # for compatibility with vllm's expert_map
        assert recv_topk_idx is not None
        recv_topk_idx = torch.where(
            recv_topk_idx == -1,
            num_experts - 1 if self.rank_expert_offset == 0 else 0,
            recv_topk_idx + self.rank_expert_offset)

        # ExpertTokensMetadata from per-expert token counts
        expert_tokens_meta = mk.ExpertTokensMetadata.make_from_list(
            handle.num_recv_tokens_per_expert_list, device=expert_x.device)

        # Quantization (same as V1 — bf16 model, no block quant)
        if not quant_config.is_block_quantized:
            expert_x_scale = None
            if expert_x.numel() != 0:
                expert_x, expert_x_scale = moe_kernel_quantize_input(
                    expert_x,
                    a1_scale,
                    quant_dtype=quant_config.quant_dtype,
                    per_act_token_quant=False,
                    block_shape=quant_config.block_shape)

        return (expert_x, expert_x_scale, expert_tokens_meta,
                recv_topk_idx, recv_topk_weights)

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

        # weight_and_reduce (same as V1)
        if fused_expert_output.numel() != 0:
            if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
                weight_and_reduce_impl = TopKWeightAndReduceContiguous()
            fused_expert_output = weight_and_reduce_impl.apply(
                output=None,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )

        # ★ V2 combine
        combined_x, _, event = self.buffer.combine(
            fused_expert_output,
            handle=self.handle,
            topk_weights=None,  # weights already applied above
            num_sms=self.num_sms,
            async_with_compute_stream=False,
            allocate_on_comm_stream=False,
        )
        if event.event is not None:
            event.current_stream_wait()

        # Write result to output tensor
        output.copy_(combined_x, non_blocking=True)


def replace_with_deepep_v2(
    model: torch.nn.Module,
    ep_cpu_group,
    num_local_experts: int,
    num_experts: int = 256,
    top_k: int = 8,
    hidden: int = 2048,
    max_tokens_per_rank: int = 10000,
):
    """
    Replace all FusedMoE layers' PrepareAndFinalize with V2 ElasticBuffer version.

    Must be called AFTER prepare_communication_buffer_for_model(model)
    so that the FusedMoEModularKernel and expert GEMM are already initialized.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEModularKernel
    from vllm.distributed import get_ep_group

    buffer, num_sms = get_or_create_elastic_buffer(
        ep_cpu_group,
        num_max_tokens_per_rank=max_tokens_per_rank,
        hidden=hidden,
        num_topk=top_k,
        num_experts=num_experts,
    )

    ep = get_ep_group()
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

        v2_pf = DeepEPV2PrepareAndFinalize(
            buffer=buffer,
            num_sms=num_sms,
            dp_size=ep.world_size,
            rank_expert_offset=ep.rank_in_group * num_local_experts,
        )
        kernel.prepare_finalize = v2_pf
        replaced += 1

    return replaced, num_sms
