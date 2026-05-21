"""
Fused MoE execution for Epoch-Spark.

Uses vLLM's production fused_experts_impl (Triton kernel with token sorting,
fused GEMM, SiLU activation) — no forward context required.

Also provides grouped_topk routing matching LLaDA2's gate logic.
"""

import torch
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl


def fused_experts(
    hidden_states: torch.Tensor,  # [M, K]
    w1: torch.Tensor,             # [E, 2*N, K] — gate_up stacked
    w2: torch.Tensor,             # [E, K, N]
    topk_weights: torch.Tensor,   # [M, top_k] float32
    topk_ids: torch.Tensor,       # [M, top_k] int
) -> torch.Tensor:
    """Fused MoE using vLLM's production Triton kernel."""
    topk_weights = topk_weights.float()
    return fused_experts_impl(
        hidden_states, w1, w2, topk_weights, topk_ids,
        inplace=False, activation="silu",
    )


def grouped_topk(router_logits, num_experts=256, top_k=8, n_group=8, topk_group=4):
    """LLaDA2-style grouped routing: select top groups then top experts within."""
    scores = torch.sigmoid(router_logits.float())
    N = scores.shape[0]
    device = scores.device
    group_size = num_experts // n_group

    grouped = scores.view(N, n_group, group_size)
    group_max = grouped.max(dim=2).values
    _, top_groups = group_max.topk(topk_group, dim=1)  # [N, topk_group]

    # Build group mask vectorized
    group_mask = torch.zeros(N, n_group, device=device, dtype=scores.dtype)
    group_mask.scatter_(1, top_groups, 1.0)
    group_mask = group_mask.unsqueeze(2).expand(-1, -1, group_size).reshape(N, num_experts)

    masked_scores = scores * group_mask
    topk_weights, topk_ids = masked_scores.topk(top_k, dim=1)
    topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-8)
    return topk_weights.float(), topk_ids
