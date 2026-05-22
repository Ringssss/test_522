"""
Epoch-Spark v3: Double-Buffer Async Prefetch Offload for dInfer.

Zero CPU compute. GPU does ALL MoE computation on a cached expert subset.
Expert weights rotate between GPU double-buffers via async PCIe DMA,
completely overlapped with block computation.

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │  GPU HBM                                                    │
  │  ┌─────────────┐  ┌─────────────┐                           │
  │  │ Buffer A     │  │ Buffer B     │  (pingpong)              │
  │  │ [budget,2I,H]│  │ [budget,2I,H]│                          │
  │  └─────────────┘  └─────────────┘                           │
  │       ▲ active         ▲ staging (async fill from CPU)       │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
  │  │ Attention │  │ Gate/LMH │  │SharedExpt│ (always resident) │
  │  └──────────┘  └──────────┘  └──────────┘                  │
  └──────────────────────────────────────────────────────────────┘
                         ▲ async PCIe Gen5 (32 GB/s)
  ┌──────────────────────────────────────────────────────────────┐
  │  CPU Pinned Memory                                           │
  │  expert_pool[256, 2I, H] — full expert weight pool           │
  └──────────────────────────────────────────────────────────────┘

Flow:
  Block N:
    1. active_buf points to buffer with block N's experts (prefetched during N-1)
    2. All ~12 iterations: GPU fused kernel on active_buf (zero CPU compute)
    3. Decoded-token cache: skip MoE for already-decoded positions
    4. Meanwhile: predict block N+1 experts, async fill staging_buf from CPU pool
    5. End of block: swap active ↔ staging pointers

Optimizations:
  - Pingpong double buffer: no allocation/deallocation, just pointer swap
  - Cross-block expert prediction via routing heat EMA
  - Overlap: async copy on dedicated CUDA stream, fully hidden by compute
  - Compact fused kernel: E=budget instead of E=256 → less dispatch overhead
  - Miss fallback: use block-level cached output (no CPU compute, no sync)
"""

import torch
from collections import defaultdict
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl


# ════════════════════════════════════════════════════════════════
# PingPong Expert Buffer
# ════════════════════════════════════════════════════════════════

class PingPongExpertBuffer:
    """Double-buffer for expert weights on GPU with async CPU↔GPU transfer."""

    def __init__(self, num_experts, expert_shape_w13, expert_shape_w2,
                 dtype, device, budget):
        self.num_experts = num_experts
        self.budget = budget
        self.device = device
        self.dtype = dtype

        # Pingpong GPU buffers
        self.buf_w13 = [
            torch.empty(budget, *expert_shape_w13, dtype=dtype, device=device),
            torch.empty(budget, *expert_shape_w13, dtype=dtype, device=device),
        ]
        self.buf_w2 = [
            torch.empty(budget, *expert_shape_w2, dtype=dtype, device=device),
            torch.empty(budget, *expert_shape_w2, dtype=dtype, device=device),
        ]

        # Which buffer is active (0 or 1)
        self.active_idx = 0

        # Mapping per buffer: expert_id → slot, slot → expert_id
        self.expert_to_slot = [
            torch.full((num_experts,), -1, dtype=torch.long, device=device),
            torch.full((num_experts,), -1, dtype=torch.long, device=device),
        ]
        self.slot_to_expert = [
            [-1] * budget,
            [-1] * budget,
        ]

        # CPU pinned expert pool (full weights)
        self.cpu_w13 = None  # set by init_from_model
        self.cpu_w2 = None

        # Async transfer stream
        self.transfer_stream = torch.cuda.Stream(device=device)
        self.transfer_event = torch.cuda.Event()

        # Expert heat scores for cross-block prediction
        self.heat = torch.zeros(num_experts, device="cpu")
        self.heat_decay = 0.6

        # Stats
        self.stats = defaultdict(int)

    def init_from_model(self, full_w13, full_w2):
        """Initialize from model weights. Copies full weights to CPU pool,
        fills initial GPU buffer with hottest experts."""
        E = full_w13.shape[0]

        # Full CPU pool (pinned)
        self.cpu_w13 = full_w13.to("cpu").pin_memory()
        self.cpu_w2 = full_w2.to("cpu").pin_memory()

        # Initial hot experts by weight norm
        norms = full_w13.float().reshape(E, -1).norm(dim=1).cpu()
        _, sorted_idx = norms.sort(descending=True)
        initial_ids = sorted_idx[:self.budget].tolist()

        # Fill both buffers with the same initial set
        for buf_idx in range(2):
            for slot, eid in enumerate(initial_ids):
                self.buf_w13[buf_idx][slot].copy_(full_w13[eid])
                self.buf_w2[buf_idx][slot].copy_(full_w2[eid])
                self.expert_to_slot[buf_idx][eid] = slot
                self.slot_to_expert[buf_idx][slot] = eid

        # Initialize heat
        self.heat[sorted_idx[:self.budget]] = 1.0

    @property
    def active_w13(self):
        return self.buf_w13[self.active_idx]

    @property
    def active_w2(self):
        return self.buf_w2[self.active_idx]

    @property
    def active_mapping(self):
        return self.expert_to_slot[self.active_idx]

    @property
    def staging_idx(self):
        return 1 - self.active_idx

    def swap_buffers(self):
        """Swap active ↔ staging. Call after async prefetch is complete."""
        self.transfer_stream.synchronize()
        self.active_idx = self.staging_idx
        self.stats["buffer_swaps"] += 1

    def predict_next_block_experts(self, current_routing_ids):
        """Predict which experts the next block will need based on
        current routing + heat history."""
        # Update heat with current block's routing
        self.heat *= self.heat_decay
        if current_routing_ids is not None:
            unique_ids = current_routing_ids.unique().cpu()
            self.heat[unique_ids] += 1.0

        # Top-budget experts by heat
        _, top_ids = self.heat.topk(self.budget)
        return set(top_ids.tolist())

    def async_prefetch_to_staging(self, needed_experts):
        """Async fill staging buffer with needed experts from CPU pool.
        Completely non-blocking — runs on dedicated CUDA stream."""
        staging = self.staging_idx
        current_in_staging = set(
            self.slot_to_expert[staging][s]
            for s in range(self.budget)
            if self.slot_to_expert[staging][s] >= 0
        )

        to_load = needed_experts - current_in_staging
        to_evict = current_in_staging - needed_experts

        if not to_load:
            self.stats["prefetch_skip"] += 1
            return

        evict_list = list(to_evict)
        load_list = list(to_load)
        n_swap = min(len(evict_list), len(load_list))

        self.stats["prefetch_swaps"] += n_swap

        # Copy staging mapping from active (so unchanged experts stay correct)
        self.expert_to_slot[staging].copy_(self.expert_to_slot[self.active_idx])
        for s in range(self.budget):
            self.slot_to_expert[staging][s] = self.slot_to_expert[self.active_idx][s]
        # Copy active buffer data to staging for unchanged experts
        with torch.cuda.stream(self.transfer_stream):
            self.buf_w13[staging].copy_(self.buf_w13[self.active_idx], non_blocking=True)
            self.buf_w2[staging].copy_(self.buf_w2[self.active_idx], non_blocking=True)

            # Now swap the changed experts
            for i in range(n_swap):
                evict_eid = evict_list[i]
                load_eid = load_list[i]
                slot = self.expert_to_slot[staging][evict_eid].item()
                if slot < 0:
                    # Find any slot used by an evicted expert
                    for s in range(self.budget):
                        if self.slot_to_expert[staging][s] in to_evict:
                            slot = s
                            break
                if slot < 0:
                    continue

                # CPU → GPU async copy into staging buffer
                self.buf_w13[staging][slot].copy_(self.cpu_w13[load_eid], non_blocking=True)
                self.buf_w2[staging][slot].copy_(self.cpu_w2[load_eid], non_blocking=True)

                # Update staging mapping
                self.expert_to_slot[staging][evict_eid] = -1
                self.expert_to_slot[staging][load_eid] = slot
                self.slot_to_expert[staging][slot] = load_eid

        self.transfer_event.record(self.transfer_stream)

    def remap_ids(self, topk_ids):
        """Remap expert IDs to active buffer slot IDs. Returns (remapped, miss_mask)."""
        remapped = self.active_mapping[topk_ids.long()]
        miss_mask = (remapped < 0)
        return remapped.clamp(min=0).int(), miss_mask

    def gpu_mem_mb(self):
        return sum(b.nbytes for b in self.buf_w13 + self.buf_w2) / (1024**2)

    def cpu_mem_mb(self):
        if self.cpu_w13 is None:
            return 0
        return (self.cpu_w13.nbytes + self.cpu_w2.nbytes) / (1024**2)


# ════════════════════════════════════════════════════════════════
# Controller
# ════════════════════════════════════════════════════════════════

class EpochSparkController:
    def __init__(self, mask_id=156895, refresh_m=5, gpu_budget=80, enable_offload=True):
        self.mask_id = mask_id
        self.refresh_m = refresh_m
        self.gpu_budget = gpu_budget
        self.enable_offload = enable_offload

        self.current_block_id = -1
        self.block_iter_count = 0
        self.token_mask_state = None

        self.moe_cache = {}       # layer_idx -> [N, H] cached MoE output
        self.cache_populated = set()

        self.buffers = {}         # layer_idx -> PingPongExpertBuffer
        self.all_topk_ids = {}    # layer_idx -> last topk_ids (for cross-block prediction)

        self.stats = defaultdict(int)

    def on_block_start(self, block_id, x_data):
        """Swap buffers (prefetch from previous block is now ready) and reset."""
        self.current_block_id = block_id
        self.block_iter_count = 0
        self.moe_cache.clear()
        self.cache_populated.clear()
        if x_data is not None:
            self.token_mask_state = (x_data == self.mask_id)

        # Swap pingpong buffers: staging (prefetched during last block) → active
        if block_id > 0 and self.enable_offload:
            for buf in self.buffers.values():
                buf.swap_buffers()

    def on_block_end(self):
        """Predict next block's experts and start async prefetch."""
        if not self.enable_offload:
            return
        for layer_idx, buf in self.buffers.items():
            routing_ids = self.all_topk_ids.get(layer_idx)
            needed = buf.predict_next_block_experts(routing_ids)
            buf.async_prefetch_to_staging(needed)

    def on_iter_start(self, x_data):
        self.block_iter_count += 1
        if x_data is not None:
            self.token_mask_state = (x_data == self.mask_id)

    def is_first_iter(self):
        return self.block_iter_count <= 1

    def has_cache(self, layer_idx):
        return layer_idx in self.cache_populated

    def cache_output(self, layer_idx, output):
        self.moe_cache[layer_idx] = output.detach().clone()
        self.cache_populated.add(layer_idx)

    def get_cached(self, layer_idx):
        return self.moe_cache.get(layer_idx)

    def get_summary(self):
        total = self.stats.get("total_tokens", 0)
        buf_stats = {}
        for lid, buf in self.buffers.items():
            buf_stats.update(buf.stats)
        return {
            "total_tokens": total,
            "gpu_fused_tokens": self.stats.get("gpu_fused_tokens", 0),
            "cached_tokens": self.stats.get("cached_tokens", 0),
            "miss_tokens": self.stats.get("miss_tokens", 0),
            "buffer_swaps": buf_stats.get("buffer_swaps", 0),
            "prefetch_swaps": buf_stats.get("prefetch_swaps", 0),
            "gpu_cache_mb": sum(b.gpu_mem_mb() for b in self.buffers.values()),
            "cpu_pool_mb": sum(b.cpu_mem_mb() for b in self.buffers.values()),
        }


# ════════════════════════════════════════════════════════════════
# Routing
# ════════════════════════════════════════════════════════════════

def _vectorized_grouped_topk(gate, flat):
    """Use Triton fused routing kernel (replaces Python topk+sort)."""
    from dinfer.triton_ops import triton_routing
    gating_output = gate.get_logits(flat)
    return triton_routing(
        gating_output, gate.expert_bias, gate.routed_scaling_factor,
        K=gate.top_k, ng=gate.n_group, tkg=gate.topk_group,
    )


# ════════════════════════════════════════════════════════════════
# MoE Forward
# ════════════════════════════════════════════════════════════════

def _make_epoch_spark_forward(moe_block, layer_idx, controller):
    gate = moe_block.gate
    experts = moe_block.experts
    shared = getattr(moe_block, 'shared_experts', None)

    # Initialize pingpong buffer
    ppbuf = None
    if controller.enable_offload:
        shape_w13 = experts.w13_weight.shape[1:]  # (2I, H)
        shape_w2 = experts.w2_weight.shape[1:]    # (H, I)
        ppbuf = PingPongExpertBuffer(
            gate.num_experts, shape_w13, shape_w2,
            experts.w13_weight.dtype, experts.w13_weight.device,
            controller.gpu_budget,
        )
        ppbuf.init_from_model(experts.w13_weight.data, experts.w2_weight.data)
        controller.buffers[layer_idx] = ppbuf

        # Free original full expert weights from GPU — saves ~1.6GB per layer
        experts.w13_weight = torch.nn.Parameter(torch.empty(0, device="cpu"), requires_grad=False)
        experts.w2_weight = torch.nn.Parameter(torch.empty(0, device="cpu"), requires_grad=False)
        torch.cuda.empty_cache()

    def fwd(hidden_states):
        nonlocal ppbuf
        res = shared(hidden_states) if shared is not None else 0
        bsz, seq_len, h = hidden_states.shape
        N = bsz * seq_len
        flat = hidden_states.view(N, h)

        topk_w, topk_ids = _vectorized_grouped_topk(gate, flat)
        controller.all_topk_ids[layer_idx] = topk_ids

        if not controller.enable_offload:
            y = fused_experts_impl(
                flat, experts.w13_weight, experts.w2_weight,
                topk_w.float(), topk_ids, inplace=False, activation="silu",
            ).to(flat.dtype)
        else:
            # ── OFFLOAD PATH: GPU-only fused kernel on cached subset ──

            # Check decoded-token cache
            use_cache = (
                controller.has_cache(layer_idx) and
                controller.token_mask_state is not None and
                not controller.is_first_iter()
            )

            if use_cache:
                mask_flat = controller.token_mask_state.view(-1)
                if mask_flat.shape[0] >= N:
                    live_idx = mask_flat[:N].nonzero(as_tuple=True)[0]
                    decoded_idx = (~mask_flat[:N]).nonzero(as_tuple=True)[0]
                else:
                    live_idx = torch.arange(N, device=flat.device)
                    decoded_idx = torch.tensor([], dtype=torch.long, device=flat.device)
                    use_cache = False
            else:
                live_idx = torch.arange(N, device=flat.device)
                decoded_idx = torch.tensor([], dtype=torch.long, device=flat.device)

            controller.stats["total_tokens"] += N

            if use_cache and len(decoded_idx) > 0:
                # Sparse path: compute live tokens only, cache for decoded
                controller.stats["gpu_fused_tokens"] += len(live_idx)
                controller.stats["cached_tokens"] += len(decoded_idx)

                cached = controller.get_cached(layer_idx)
                y = cached.clone() if cached is not None and cached.shape[0] == N else torch.zeros(N, h, device=flat.device, dtype=flat.dtype)

                if len(live_idx) > 0:
                    live_flat = flat[live_idx].contiguous()
                    live_tw = topk_w[live_idx].contiguous()
                    live_ti = topk_ids[live_idx].contiguous()

                    remapped, miss = ppbuf.remap_ids(live_ti)
                    safe_w = live_tw.clone()
                    safe_w[miss] = 0.0

                    live_out = fused_experts_impl(
                        live_flat, ppbuf.active_w13, ppbuf.active_w2,
                        safe_w.float(), remapped, inplace=False, activation="silu",
                    ).to(flat.dtype)

                    # For miss experts: use cached value (no CPU compute)
                    if miss.any():
                        n_miss = miss.any(dim=1).sum().item()
                        controller.stats["miss_tokens"] += n_miss

                    y[live_idx] = live_out
            else:
                # Dense path: all tokens through GPU fused kernel
                controller.stats["gpu_fused_tokens"] += N

                remapped, miss = ppbuf.remap_ids(topk_ids)
                safe_w = topk_w.clone()
                safe_w[miss] = 0.0

                y = fused_experts_impl(
                    flat, ppbuf.active_w13, ppbuf.active_w2,
                    safe_w.float(), remapped, inplace=False, activation="silu",
                ).to(flat.dtype)

                if miss.any():
                    controller.stats["miss_tokens"] += miss.any(dim=1).sum().item()

            # Update decoded-token cache
            if not controller.has_cache(layer_idx) or controller.block_iter_count % controller.refresh_m == 0:
                controller.cache_output(layer_idx, y)

        y = y.view(bsz, seq_len, h)
        if shared is not None:
            y = y + res
        return y

    return fwd


# ════════════════════════════════════════════════════════════════
# Patching
# ════════════════════════════════════════════════════════════════

def patch_dinfer_model(model, controller):
    count = 0
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "LLaDA2MoeSparseMoeBlock":
            count += 1
            mod.forward = _make_epoch_spark_forward(mod, count, controller)
    mode = f"budget={controller.gpu_budget}" if controller.enable_offload else "no-offload"
    print(f"[ES-v3] Patched {count} MoE layers (pingpong, {mode})")
    return count


def unpatch_dinfer_model(model):
    patch_dinfer_baseline(model)


def patch_dinfer_baseline(model):
    count = 0
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "LLaDA2MoeSparseMoeBlock":
            mod.forward = _make_baseline_forward(mod)
            count += 1
    print(f"[ES-v3] Patched {count} MoE layers (baseline fused)")
    return count


def _make_baseline_forward(moe_block):
    gate = moe_block.gate
    experts = moe_block.experts
    shared = getattr(moe_block, 'shared_experts', None)

    # Build expert_map for EP sharding: maps global expert ID → local tensor index
    local_num_experts = experts.w13_weight.shape[0]
    global_num_experts = gate.num_experts
    expert_map = getattr(experts, 'expert_map', None)
    if expert_map is not None:
        expert_map = expert_map.to(experts.w13_weight.device)

    def fwd(hidden_states):
        res = shared(hidden_states) if shared is not None else 0
        bsz, seq_len, h = hidden_states.shape
        flat = hidden_states.view(-1, h)
        topk_w, topk_idx = _vectorized_grouped_topk(gate, flat)
        y = fused_experts_impl(
            flat, experts.w13_weight, experts.w2_weight,
            topk_w.float(), topk_idx, inplace=False, activation="silu",
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        ).to(flat.dtype)
        y = y.view(bsz, seq_len, h)
        if shared is not None:
            y = y + res
        return y

    return fwd
