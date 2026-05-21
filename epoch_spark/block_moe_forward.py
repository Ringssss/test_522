"""
Phase 3: Block-Scoped MoE Forward with Fused Triton Execution.

Replaces standard MoE forward with:
  1. Block-boundary routing profiling → residency planning
  2. Live/decoded token classification → compact execution
  3. Fused Triton kernel for live tokens via fused_experts_impl
  4. Decoded-token MoE output cache with bounded staleness
"""

import torch
from collections import defaultdict

from config import (
    NUM_EXPERTS, TOP_K, N_GROUP, TOPK_GROUP, MASK_ID,
    MOE_INTERMEDIATE_SIZE, DEFAULT_DECODED_CACHE_REFRESH_M,
)


class BlockMoEController:
    """Controls block-scoped MoE execution with residency manager integration."""

    def __init__(self, residency_mgr=None, refresh_m=DEFAULT_DECODED_CACHE_REFRESH_M):
        self.rmgr = residency_mgr
        self.refresh_m = refresh_m
        self.current_block_id = -1
        self.current_iter = 0
        self.token_mask_state = None

        self.decoded_cache = {}
        self.cache_age = {}
        self.stats = defaultdict(int)

    def on_block_start(self, block_id, x_tokens, routing_profiles=None):
        self.current_block_id = block_id
        self.current_iter = 0
        self.token_mask_state = (x_tokens == MASK_ID)
        self.decoded_cache.clear()
        self.cache_age.clear()

        if routing_profiles is not None and self.rmgr is not None:
            self.rmgr.plan_block(routing_profiles)
            self.rmgr.sync_transfers()

    def on_iter_start(self, iter_idx, x_tokens):
        self.current_iter = iter_idx
        self.token_mask_state = (x_tokens == MASK_ID)

    def should_refresh_cache(self, layer_id):
        return self.cache_age.get(layer_id, self.refresh_m + 1) >= self.refresh_m

    def update_cache(self, layer_id, moe_output, full_seq_len):
        self.decoded_cache[layer_id] = moe_output.detach().clone()
        self.cache_age[layer_id] = 0

    def age_caches(self):
        for lid in self.cache_age:
            self.cache_age[lid] += 1


def make_block_moe_forward(fused_moe_module, layer_id, controller):
    """Create block-aware MoE forward using fused Triton kernel for live tokens."""
    from triton_moe import fused_experts, grouped_topk

    def block_moe_forward(*args, **kwargs):
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        router_logits = kwargs.get("router_logits", args[1] if len(args) > 1 else None)

        N, H = hidden_states.shape
        device = hidden_states.device

        # Always recompute routing
        topk_weights, topk_ids = grouped_topk(router_logits)
        fused_moe_module._last_topk_ids = topk_ids
        fused_moe_module._last_topk_weights = topk_weights

        # Determine live vs decoded tokens
        mask_state = controller.token_mask_state
        if mask_state is not None and mask_state.ndim > 1:
            mask_state = mask_state.view(-1)

        use_cache = (
            mask_state is not None and
            layer_id in controller.decoded_cache and
            not controller.should_refresh_cache(layer_id) and
            mask_state.shape[0] >= N
        )

        # Only do sparse execution if >30% tokens are decoded (otherwise overhead > savings)
        n_live = N
        if use_cache:
            n_live = mask_state[:N].sum().item()
        live_fraction = n_live / N if N > 0 else 1.0

        if not use_cache or live_fraction > 0.7 or N <= 512:
            # Dense path: run full fused kernel (faster for small N or high live fraction)
            controller.stats["total_tokens"] += N
            controller.stats["live_tokens"] += N
            output = fused_experts(
                hidden_states, fused_moe_module.w13_weight, fused_moe_module.w2_weight,
                topk_weights, topk_ids,
            )
            if controller.should_refresh_cache(layer_id) or layer_id not in controller.decoded_cache:
                controller.update_cache(layer_id, output, N)
            return output

        # Sparse path: only compute live tokens, use cache for decoded
        live_idx = mask_state[:N].nonzero(as_tuple=True)[0]
        decoded_idx = (~mask_state[:N]).nonzero(as_tuple=True)[0]

        controller.stats["total_tokens"] += N
        controller.stats["live_tokens"] += len(live_idx)
        controller.stats["cached_tokens"] += len(decoded_idx)

        output = torch.zeros(N, H, device=device, dtype=hidden_states.dtype)

        # Fill decoded from cache
        cached = controller.decoded_cache[layer_id]
        if cached.shape[0] == N:
            output[decoded_idx] = cached[decoded_idx].to(output.dtype)

        # Compute live tokens via fused kernel
        if len(live_idx) > 0:
            live_hs = hidden_states[live_idx].contiguous()
            live_topk_w = topk_weights[live_idx].contiguous()
            live_topk_ids = topk_ids[live_idx].contiguous()

            live_output = fused_experts(
                live_hs, fused_moe_module.w13_weight, fused_moe_module.w2_weight,
                live_topk_w, live_topk_ids,
            )
            output[live_idx] = live_output

        if controller.should_refresh_cache(layer_id) or layer_id not in controller.decoded_cache:
            controller.update_cache(layer_id, output, N)

        return output

    return block_moe_forward


def patch_model_with_block_moe(model, controller):
    """Replace all FusedMoE forwards with block-scoped versions."""
    from vllm.model_executor.layers.fused_moe import FusedMoE

    layer_id = 0
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "LLaDA2MoeSparseMoeBlock":
            layer_id += 1
            experts = mod.experts
            if isinstance(experts, FusedMoE):
                experts.forward = make_block_moe_forward(experts, layer_id, controller)

    print(f"[block_moe] Patched {layer_id} MoE layers with block-scoped fused forward")
    return layer_id


def profile_routing(model, x, position_ids=None):
    """Run one forward pass and collect active expert sets per layer."""
    from vllm.model_executor.layers.fused_moe import FusedMoE

    profiles = {}
    with torch.no_grad():
        _ = model(input_ids=x, position_ids=position_ids, use_cache=False, return_dict=True)

    layer_id = 0
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "LLaDA2MoeSparseMoeBlock":
            layer_id += 1
            experts = mod.experts
            topk_ids = getattr(experts, "_last_topk_ids", None)
            if topk_ids is not None:
                profiles[layer_id] = set(topk_ids.flatten().tolist())

    return profiles
