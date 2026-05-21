"""
Phase 2: Block-Scoped Weight Residency Manager.

Manages expert weight placement between GPU and CPU at block granularity.
Inspired by SparkInfer's DFR (Dynamic Frequency-based Reloading) but operates
at block scope instead of per-forward.

Key design:
  - At model init: split expert weights into GPU cache (hot) + CPU pinned pool (cold)
  - At block boundary: profile routing → plan which experts to stage
  - Between blocks: async CPU→GPU prefetch on dedicated CUDA stream
  - Within block: all ~12 iterations use the same resident set (no CPU sync on hot path)
"""

import torch
import torch.cuda
import time
from collections import defaultdict

from config import (
    NUM_EXPERTS, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE,
    DEFAULT_GPU_EXPERT_BUDGET, DEFAULT_DFR_DECAY, MOE_LAYERS,
)


class BlockResidencyManager:
    """Manages expert weight residency between GPU HBM and CPU pinned memory."""

    def __init__(self, model, device="cuda:0",
                 gpu_budget_per_layer=DEFAULT_GPU_EXPERT_BUDGET,
                 dfr_decay=DEFAULT_DFR_DECAY):
        self.device = torch.device(device)
        self.gpu_budget = gpu_budget_per_layer
        self.dfr_decay = dfr_decay
        self.n_moe_layers = 0

        # Per-layer state
        self.gpu_experts = {}      # layer_id -> set of expert ids on GPU
        self.cpu_experts = {}      # layer_id -> set of expert ids on CPU
        self.heat_scores = {}      # layer_id -> tensor [NUM_EXPERTS] tracking usage frequency
        self.gpu_w13 = {}          # layer_id -> dict {eid: tensor [2*I, H] on GPU}
        self.gpu_w2 = {}           # layer_id -> dict {eid: tensor [H, I] on GPU}
        self.cpu_w13 = {}          # layer_id -> dict {eid: tensor [2*I, H] on CPU pinned}
        self.cpu_w2 = {}           # layer_id -> dict {eid: tensor [H, I] on CPU pinned}

        # Async transfer
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self.pending_transfers = []

        # Stats
        self.stats = defaultdict(int)
        self.block_count = 0

        self._init_from_model(model)

    def _init_from_model(self, model):
        """Extract expert weights from FusedMoE modules and split to GPU/CPU."""
        from vllm.model_executor.layers.fused_moe import FusedMoE

        layer_id = 0
        for name, mod in model.named_modules():
            if not isinstance(mod, FusedMoE):
                continue
            layer_id += 1

            w13 = mod.w13_weight.data  # [E, 2*I, H] on GPU
            w2 = mod.w2_weight.data    # [E, H, I] on GPU

            # Initially: top experts by weight norm go to GPU cache, rest to CPU
            norms = w13.float().norm(dim=(1, 2))  # [E]
            _, sorted_idx = norms.sort(descending=True)

            gpu_ids = set(sorted_idx[:self.gpu_budget].tolist())
            cpu_ids = set(sorted_idx[self.gpu_budget:].tolist())

            self.gpu_experts[layer_id] = gpu_ids
            self.cpu_experts[layer_id] = cpu_ids
            self.heat_scores[layer_id] = torch.zeros(NUM_EXPERTS, device="cpu")

            # Copy GPU experts to cache (already on GPU)
            self.gpu_w13[layer_id] = {}
            self.gpu_w2[layer_id] = {}
            for eid in gpu_ids:
                self.gpu_w13[layer_id][eid] = w13[eid].clone()
                self.gpu_w2[layer_id][eid] = w2[eid].clone()

            # Move CPU experts to pinned memory
            self.cpu_w13[layer_id] = {}
            self.cpu_w2[layer_id] = {}
            for eid in cpu_ids:
                t13 = w13[eid].to("cpu").pin_memory()
                t2 = w2[eid].to("cpu").pin_memory()
                self.cpu_w13[layer_id][eid] = t13
                self.cpu_w2[layer_id][eid] = t2

            self.n_moe_layers += 1

        gpu_mb = sum(
            sum(t.nbytes for t in self.gpu_w13[lid].values()) +
            sum(t.nbytes for t in self.gpu_w2[lid].values())
            for lid in self.gpu_w13
        ) / (1024 * 1024)
        cpu_mb = sum(
            sum(t.nbytes for t in self.cpu_w13[lid].values()) +
            sum(t.nbytes for t in self.cpu_w2[lid].values())
            for lid in self.cpu_w13
        ) / (1024 * 1024)

        print(f"[residency] Initialized: {self.n_moe_layers} MoE layers, "
              f"GPU budget={self.gpu_budget}/layer")
        print(f"[residency] GPU cache: {gpu_mb:.1f} MB, CPU pool: {cpu_mb:.1f} MB")

    def plan_block(self, routing_profiles):
        """Plan expert residency for the upcoming block.

        Args:
            routing_profiles: dict {layer_id: active_expert_set} from first iteration routing
        """
        self.block_count += 1
        transfers_needed = []

        for layer_id, needed_experts in routing_profiles.items():
            if layer_id not in self.gpu_experts:
                continue

            needed = set(needed_experts)
            current_gpu = self.gpu_experts[layer_id]

            # Update heat scores with EMA
            hs = self.heat_scores[layer_id]
            hs *= self.dfr_decay
            for eid in needed:
                hs[eid] += 1.0

            # Determine what's missing on GPU
            missing_on_gpu = needed - current_gpu
            if not missing_on_gpu:
                self.stats["no_swap_blocks"] += 1
                continue

            # Evict coldest GPU experts to make room
            n_to_evict = len(missing_on_gpu)
            evict_candidates = current_gpu - needed
            if len(evict_candidates) < n_to_evict:
                # Not enough room — only stage what we can
                missing_on_gpu = set(list(missing_on_gpu)[:len(evict_candidates)])
                n_to_evict = len(missing_on_gpu)

            # Sort evict candidates by heat score (coldest first)
            evict_list = sorted(evict_candidates, key=lambda e: hs[e].item())[:n_to_evict]
            stage_list = list(missing_on_gpu)[:n_to_evict]

            for evict_eid, stage_eid in zip(evict_list, stage_list):
                transfers_needed.append((layer_id, evict_eid, stage_eid))

            self.stats["total_swaps"] += len(transfers_needed)

        # Execute transfers asynchronously
        if transfers_needed:
            self._async_stage(transfers_needed)

    def _async_stage(self, transfers):
        """Async CPU→GPU and GPU→CPU weight transfers on dedicated stream."""
        with torch.cuda.stream(self.transfer_stream):
            for layer_id, evict_eid, stage_eid in transfers:
                # GPU → CPU (evict)
                self.cpu_w13[layer_id][evict_eid] = (
                    self.gpu_w13[layer_id][evict_eid].to("cpu", non_blocking=True).pin_memory()
                )
                self.cpu_w2[layer_id][evict_eid] = (
                    self.gpu_w2[layer_id][evict_eid].to("cpu", non_blocking=True).pin_memory()
                )
                del self.gpu_w13[layer_id][evict_eid]
                del self.gpu_w2[layer_id][evict_eid]

                # CPU → GPU (stage)
                self.gpu_w13[layer_id][stage_eid] = (
                    self.cpu_w13[layer_id][stage_eid].to(self.device, non_blocking=True)
                )
                self.gpu_w2[layer_id][stage_eid] = (
                    self.cpu_w2[layer_id][stage_eid].to(self.device, non_blocking=True)
                )
                del self.cpu_w13[layer_id][stage_eid]
                del self.cpu_w2[layer_id][stage_eid]

                # Update tracking
                self.gpu_experts[layer_id].discard(evict_eid)
                self.gpu_experts[layer_id].add(stage_eid)
                self.cpu_experts[layer_id].discard(stage_eid)
                self.cpu_experts[layer_id].add(evict_eid)

    def sync_transfers(self):
        """Wait for all async transfers to complete. Call after plan_block, before iteration loop."""
        self.transfer_stream.synchronize()

    def get_expert_weights(self, layer_id, expert_id):
        """Get weights for a specific expert. Returns (w13, w2) tensors.

        If expert is on GPU, returns immediately.
        If on CPU, does sync transfer (fallback path).
        """
        if expert_id in self.gpu_w13.get(layer_id, {}):
            self.stats["gpu_hits"] += 1
            return self.gpu_w13[layer_id][expert_id], self.gpu_w2[layer_id][expert_id]

        # CPU fallback — sync transfer
        self.stats["cpu_fallbacks"] += 1
        w13 = self.cpu_w13[layer_id][expert_id].to(self.device)
        w2 = self.cpu_w2[layer_id][expert_id].to(self.device)
        return w13, w2

    def is_on_gpu(self, layer_id, expert_id):
        return expert_id in self.gpu_experts.get(layer_id, set())

    def get_gpu_hit_rate(self):
        total = self.stats["gpu_hits"] + self.stats["cpu_fallbacks"]
        return self.stats["gpu_hits"] / total if total > 0 else 0.0

    def get_stats(self):
        return dict(self.stats)

    def gpu_cache_mb(self):
        total = 0
        for lid in self.gpu_w13:
            total += sum(t.nbytes for t in self.gpu_w13[lid].values())
            total += sum(t.nbytes for t in self.gpu_w2[lid].values())
        return total / (1024 * 1024)

    def cpu_pool_mb(self):
        total = 0
        for lid in self.cpu_w13:
            total += sum(t.nbytes for t in self.cpu_w13[lid].values())
            total += sum(t.nbytes for t in self.cpu_w2[lid].values())
        return total / (1024 * 1024)
