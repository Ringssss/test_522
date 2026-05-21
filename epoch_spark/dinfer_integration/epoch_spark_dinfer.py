"""
Epoch-Spark integration for dInfer.

Patches dInfer's MoE forward with:
  1. Fused Triton routing (vectorized grouped_topk, no Python loops)
  2. Block-scoped decoded-token MoE output cache with bounded staleness
  3. Adaptive dense/sparse execution based on live token fraction

Usage:
    from epoch_spark_dinfer import EpochSparkController, patch_dinfer_model
    controller = EpochSparkController()
    patch_dinfer_model(model, controller)
    # Then use dInfer's normal generation pipeline
"""

import torch
from collections import defaultdict
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl


class EpochSparkController:
    """Block-scoped MoE execution controller for dInfer integration."""

    def __init__(self, mask_id=156895, refresh_m=5, sparse_threshold=0.7):
        self.mask_id = mask_id
        self.refresh_m = refresh_m
        self.sparse_threshold = sparse_threshold

        self.current_block_id = -1
        self.current_iter = 0
        self.token_mask_state = None

        self.decoded_cache = {}   # layer_idx -> tensor [N, H]
        self.cache_age = {}       # layer_idx -> int

        self.stats = defaultdict(int)

    def on_block_start(self, block_id, x_data):
        """Call at beginning of each diffusion block."""
        self.current_block_id = block_id
        self.current_iter = 0
        self.decoded_cache.clear()
        self.cache_age.clear()
        if x_data is not None:
            self.token_mask_state = (x_data == self.mask_id)

    def on_iter_start(self, x_data):
        """Call before each diffusion iteration forward."""
        self.current_iter += 1
        if x_data is not None:
            self.token_mask_state = (x_data == self.mask_id)

    def on_iter_end(self):
        for lid in self.cache_age:
            self.cache_age[lid] += 1

    def should_use_cache(self, layer_idx, N):
        if self.token_mask_state is None:
            return False
        if layer_idx not in self.decoded_cache:
            return False
        if self.cache_age.get(layer_idx, self.refresh_m + 1) >= self.refresh_m:
            return False
        flat = self.token_mask_state.view(-1)
        if flat.shape[0] < N:
            return False
        live_frac = flat[:N].float().mean().item()
        return live_frac < self.sparse_threshold

    def update_cache(self, layer_idx, output):
        self.decoded_cache[layer_idx] = output.detach().clone()
        self.cache_age[layer_idx] = 0

    def get_summary(self):
        total = self.stats.get("total_tokens", 0)
        cached = self.stats.get("cached_tokens", 0)
        return {
            "total_tokens": total,
            "live_tokens": self.stats.get("live_tokens", 0),
            "cached_tokens": cached,
            "cached_pct": cached / total * 100 if total > 0 else 0,
            "dense_calls": self.stats.get("dense_calls", 0),
            "sparse_calls": self.stats.get("sparse_calls", 0),
        }


def _vectorized_grouped_topk(gate, hidden_states_flat):
    """Fast grouped routing using vectorized scatter (no Python loops)."""
    gating_output = gate.get_logits(hidden_states_flat)
    scores = torch.sigmoid(gating_output.float())
    scores_for_routing = scores + gate.expert_bias

    n_group = gate.n_group
    topk_group = gate.topk_group
    num_experts = gate.num_experts
    top_k = gate.top_k
    group_size = num_experts // n_group

    grouped = scores_for_routing.view(-1, n_group, group_size)
    group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
    top_group_idx = group_scores.topk(topk_group, dim=-1).indices  # [N, topk_group]

    group_mask = torch.zeros(scores.shape[0], n_group, device=scores.device, dtype=scores.dtype)
    group_mask.scatter_(1, top_group_idx, 1.0)
    group_mask = group_mask.unsqueeze(2).expand(-1, -1, group_size).reshape(-1, num_experts)

    masked = scores_for_routing * group_mask
    _, topk_idx = masked.topk(top_k, dim=1)

    topk_weight = torch.gather(scores, 1, topk_idx)
    topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * gate.routed_scaling_factor

    return topk_weight, topk_idx


def _make_epoch_spark_moe_forward(moe_block, layer_idx, controller):
    """Create patched MoE forward for a single LLaDA2MoeSparseMoeBlock."""

    original_gate = moe_block.gate
    experts_module = moe_block.experts
    shared_experts = getattr(moe_block, 'shared_experts', None)
    config = moe_block.config

    def patched_forward(hidden_states):
        # Shared expert (always runs on all tokens)
        res = shared_experts(hidden_states) if shared_experts is not None else 0

        bsz, seq_len, h = hidden_states.shape
        N = bsz * seq_len
        hidden_flat = hidden_states.view(N, h)

        # Fast vectorized routing
        topk_weight, topk_idx = _vectorized_grouped_topk(original_gate, hidden_flat)

        # Check if we should use decoded-token cache
        use_cache = controller.should_use_cache(layer_idx, N)

        if not use_cache:
            # Dense path: full fused kernel on all tokens
            controller.stats["total_tokens"] += N
            controller.stats["live_tokens"] += N
            controller.stats["dense_calls"] += 1

            y = fused_experts_impl(
                hidden_flat, experts_module.w13_weight, experts_module.w2_weight,
                topk_weight.float(), topk_idx, inplace=False, activation="silu",
            )

            controller.update_cache(layer_idx, y)
        else:
            # Sparse path: only compute live tokens, use cache for decoded
            mask_flat = controller.token_mask_state.view(-1)[:N]
            live_idx = mask_flat.nonzero(as_tuple=True)[0]
            decoded_idx = (~mask_flat).nonzero(as_tuple=True)[0]

            controller.stats["total_tokens"] += N
            controller.stats["live_tokens"] += len(live_idx)
            controller.stats["cached_tokens"] += len(decoded_idx)
            controller.stats["sparse_calls"] += 1

            y = torch.zeros(N, h, device=hidden_flat.device, dtype=hidden_flat.dtype)

            # Fill decoded from cache
            cached = controller.decoded_cache[layer_idx]
            if cached.shape[0] == N:
                y[decoded_idx] = cached[decoded_idx]

            # Compute live tokens via fused kernel
            if len(live_idx) > 0:
                live_hs = hidden_flat[live_idx].contiguous()
                live_tw = topk_weight[live_idx].contiguous().float()
                live_ti = topk_idx[live_idx].contiguous()

                live_out = fused_experts_impl(
                    live_hs, experts_module.w13_weight, experts_module.w2_weight,
                    live_tw, live_ti, inplace=False, activation="silu",
                )
                y[live_idx] = live_out

            # Check if we should refresh the cache
            if controller.cache_age.get(layer_idx, controller.refresh_m + 1) >= controller.refresh_m:
                controller.update_cache(layer_idx, y)

        y = y.view(bsz, seq_len, h)
        if shared_experts is not None:
            y = y + res
        return y

    return patched_forward


def patch_dinfer_model(model, controller):
    """Patch all MoE blocks in a dInfer LLaDA2 model with Epoch-Spark forward.

    Returns the number of patched layers.
    """
    count = 0
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "LLaDA2MoeSparseMoeBlock":
            layer_idx = count + 1  # MoE layers start at 1
            mod.forward = _make_epoch_spark_moe_forward(mod, layer_idx, controller)
            count += 1

    print(f"[epoch-spark] Patched {count} MoE layers in dInfer model")
    return count


def unpatch_dinfer_model(model):
    """Restore to baseline fused forward (not original vLLM forward)."""
    patch_dinfer_baseline(model)


def patch_dinfer_baseline(model):
    """Patch MoE blocks with fused Triton forward (no block-scoped opts).

    This fixes the vLLM forward_context issue for standalone usage
    while providing a fair baseline comparison.
    """
    count = 0
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "LLaDA2MoeSparseMoeBlock":
            mod.forward = _make_baseline_moe_forward(mod)
            count += 1
    print(f"[epoch-spark] Patched {count} MoE layers with fused baseline forward")
    return count


def _make_baseline_moe_forward(moe_block):
    """Baseline fused MoE forward — same kernel, no block-scoped optimizations."""
    gate = moe_block.gate
    experts = moe_block.experts
    shared = getattr(moe_block, 'shared_experts', None)

    def fwd(hidden_states):
        res = shared(hidden_states) if shared is not None else 0
        bsz, seq_len, h = hidden_states.shape
        flat = hidden_states.view(-1, h)

        topk_w, topk_idx = _vectorized_grouped_topk(gate, flat)

        y = fused_experts_impl(
            flat, experts.w13_weight, experts.w2_weight,
            topk_w.float(), topk_idx, inplace=False, activation="silu",
        )
        y = y.view(bsz, seq_len, h)
        if shared is not None:
            y = y + res
        return y

    return fwd
