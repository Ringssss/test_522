"""
Phase 3: Block-Scoped MoE Forward with Hybrid GPU-CPU Execution.

Replaces standard MoE forward with:
  1. Block-boundary routing profiling → residency planning
  2. Live/decoded token classification → compact execution
  3. GPU-resident experts: direct GEMM
  4. CPU-fallback experts: sync transfer (rare after planning)
  5. Decoded-token MoE output cache with bounded staleness
"""

import torch
from collections import defaultdict

from config import (
    NUM_EXPERTS, TOP_K, N_GROUP, TOPK_GROUP, MASK_ID,
    MOE_INTERMEDIATE_SIZE, DEFAULT_DECODED_CACHE_REFRESH_M,
)


class BlockMoEController:
    """Controls block-scoped MoE execution with residency manager integration."""

    def __init__(self, residency_mgr, refresh_m=DEFAULT_DECODED_CACHE_REFRESH_M):
        self.rmgr = residency_mgr
        self.refresh_m = refresh_m

        # Block state
        self.current_block_id = -1
        self.current_iter = 0
        self.token_mask_state = None  # [S] bool — True if position is still MASK

        # Per-layer decoded-token MoE output cache
        # decoded_cache[layer_id] = tensor [S, H] — cached MoE outputs
        self.decoded_cache = {}
        self.cache_age = {}  # layer_id -> iterations since last refresh

        # Stats
        self.stats = defaultdict(int)

    def on_block_start(self, block_id, x_tokens, routing_profiles=None):
        """Called at block boundary. Plans residency and resets caches."""
        self.current_block_id = block_id
        self.current_iter = 0
        self.token_mask_state = (x_tokens == MASK_ID)  # [B, S] or [S]

        # Clear decoded caches for new block
        self.decoded_cache.clear()
        self.cache_age.clear()

        # Plan expert residency
        if routing_profiles is not None and self.rmgr is not None:
            self.rmgr.plan_block(routing_profiles)
            self.rmgr.sync_transfers()

    def on_iter_start(self, iter_idx, x_tokens):
        """Called before each iteration. Updates token liveness."""
        self.current_iter = iter_idx
        self.token_mask_state = (x_tokens == MASK_ID)

    def should_refresh_cache(self, layer_id):
        age = self.cache_age.get(layer_id, self.refresh_m + 1)
        return age >= self.refresh_m

    def update_cache(self, layer_id, moe_output, full_seq_len):
        """Store full MoE output for decoded-token cache."""
        self.decoded_cache[layer_id] = moe_output.detach().clone()
        self.cache_age[layer_id] = 0

    def age_caches(self):
        for lid in self.cache_age:
            self.cache_age[lid] += 1


def make_block_moe_forward(fused_moe_module, layer_id, controller):
    """Create a block-aware MoE forward function for a specific layer.

    This replaces the standard FusedMoE forward with:
      - Residency-aware expert execution
      - Decoded-token skip with bounded-staleness cache
      - Compact live-token execution
    """
    rmgr = controller.rmgr

    def block_moe_forward(*args, **kwargs):
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        router_logits = kwargs.get("router_logits", args[1] if len(args) > 1 else None)

        N, H = hidden_states.shape
        device = hidden_states.device

        # Routing (always recomputed)
        scores = torch.sigmoid(router_logits.float())
        group_size = NUM_EXPERTS // N_GROUP
        grouped = scores.view(N, N_GROUP, group_size)
        group_max = grouped.max(dim=2).values
        _, top_groups = group_max.topk(TOPK_GROUP, dim=1)

        group_mask = torch.zeros(N, NUM_EXPERTS, device=device)
        for g in range(TOPK_GROUP):
            g_idx = top_groups[:, g]
            starts = g_idx * group_size
            for offset in range(group_size):
                group_mask[torch.arange(N, device=device), starts + offset] = 1.0
        masked_scores = scores * group_mask
        topk_weights, topk_ids = masked_scores.topk(TOP_K, dim=1)
        topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Store for collector/profiling access
        fused_moe_module._last_topk_ids = topk_ids
        fused_moe_module._last_topk_weights = topk_weights

        active_experts = topk_ids.unique().tolist()

        # Determine live vs decoded tokens
        mask_state = controller.token_mask_state
        if mask_state is not None and mask_state.ndim > 1:
            mask_state = mask_state.view(-1)

        use_cache = (
            mask_state is not None and
            layer_id in controller.decoded_cache and
            not controller.should_refresh_cache(layer_id) and
            len(mask_state) == N
        )

        if use_cache:
            live_idx = mask_state[:N].nonzero(as_tuple=True)[0]
            decoded_idx = (~mask_state[:N]).nonzero(as_tuple=True)[0]
        else:
            live_idx = torch.arange(N, device=device)
            decoded_idx = torch.tensor([], dtype=torch.long, device=device)

        controller.stats["total_tokens"] += N
        controller.stats["live_tokens"] += len(live_idx)
        controller.stats["cached_tokens"] += len(decoded_idx)

        # Initialize output
        output = torch.zeros(N, H, device=device, dtype=hidden_states.dtype)

        # Fill decoded positions from cache
        if len(decoded_idx) > 0 and use_cache:
            cached = controller.decoded_cache[layer_id]
            if cached.shape[0] == N:
                output[decoded_idx] = cached[decoded_idx].to(output.dtype)

        # Compute live tokens through active experts
        if len(live_idx) > 0:
            live_hs = hidden_states[live_idx]
            live_topk_ids = topk_ids[live_idx]
            live_topk_weights = topk_weights[live_idx]
            live_output = torch.zeros(len(live_idx), H, device=device, dtype=hidden_states.dtype)

            live_active = live_topk_ids.unique().tolist()
            I = MOE_INTERMEDIATE_SIZE

            for eid in live_active:
                token_mask = (live_topk_ids == eid)
                token_has = token_mask.any(dim=1)
                if not token_has.any():
                    continue

                idx = token_has.nonzero(as_tuple=True)[0]
                hs_e = live_hs[idx]

                # Get weights from residency manager or fallback
                if rmgr is not None:
                    w13, w2 = rmgr.get_expert_weights(layer_id, eid)
                else:
                    w13 = fused_moe_module.w13_weight[eid]
                    w2 = fused_moe_module.w2_weight[eid]

                gate_up = hs_e @ w13.T.to(hs_e.dtype)
                gate_out = torch.nn.functional.silu(gate_up[:, :I])
                up_out = gate_up[:, I:]
                intermediate = gate_out * up_out
                expert_out = intermediate @ w2.T.to(hs_e.dtype)

                weights_for = live_topk_weights[idx]
                pos_mask = token_mask[idx]
                ew = (weights_for * pos_mask.float()).sum(dim=1, keepdim=True)
                live_output[idx] += expert_out * ew.to(expert_out.dtype)

            output[live_idx] = live_output

        # Update decoded-token cache
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

    print(f"[block_moe] Patched {layer_id} MoE layers with block-scoped forward")
    return layer_id


def profile_routing(model, x, position_ids=None):
    """Run one forward pass and collect active expert sets per layer.

    Returns dict {layer_id: set of active expert ids}.
    """
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
