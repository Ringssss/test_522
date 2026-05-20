#!/usr/bin/env python3
"""
Collect MoE routing stability data for paper motivation figures.

Panel (a): Pairwise Jaccard heatmap of active expert sets across iterations
           within a block, for Layer 4, 10, 18.
Panel (b): Expert routing concentration CDF (cumulative coverage curve).

Uses K=8 native routing (NO EB, NO s_mask, NO fused_routing).

Launch:
    cd /home/wuhang/wuhang/dllm_wh && \\
    torchrun --nproc_per_node=8 codex_coding/src/collect_routing_stability.py \\
        --batch-size 32 --gen-length 256 --block-length 32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID = 156895
EOS_ID = 156892
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
BLOCK_LENGTH = 32
TARGET_LAYERS = [4, 10, 18]
N_EXPERTS = 256


# =====================================================================
# RoutingCollector
# =====================================================================

class RoutingCollector:
    """Collects per-layer per-iteration routing data (active expert set + histogram)."""

    def __init__(self, target_layers: list[int], n_experts: int = 256):
        self.target_layers = set(target_layers)
        self.n_experts = n_experts
        self._current_block_id = -1
        self._iter_in_block = 0
        self._pending: dict[int, dict] = {}
        # blocks[layer_id] = list of blocks, each block = list of iter records
        self.blocks: dict[int, list[list[dict]]] = {
            lid: [] for lid in target_layers
        }

    def on_block_init(self, block_id: int):
        self._current_block_id = int(block_id)
        self._iter_in_block = 0
        for lid in self.target_layers:
            self.blocks[lid].append([])

    def on_forward_start(self):
        self._pending = {}

    def on_forward_end(self):
        if self._current_block_id < 0:
            return
        for lid in self.target_layers:
            if lid in self._pending and self.blocks[lid]:
                self.blocks[lid][-1].append(self._pending[lid])
        self._iter_in_block += 1
        self._pending = {}

    def record(self, layer_id: int, topk_ids: torch.Tensor):
        if layer_id not in self.target_layers:
            return
        flat = topk_ids.flatten().long()
        active_set = frozenset(flat.tolist())
        histogram = torch.bincount(flat, minlength=self.n_experts).tolist()
        self._pending[layer_id] = {
            "active_set": active_set,
            "histogram": histogram,
        }


# =====================================================================
# Hook installation
# =====================================================================

def install_routing_hooks(model, collector: RoutingCollector):
    installed = []
    gate_idx = 0
    for _name, mod in model.named_modules():
        if mod.__class__.__name__ != "LLaDA2MoeGate":
            continue
        layer_id = gate_idx
        gate_idx += 1
        if layer_id not in collector.target_layers:
            continue

        orig_routing = mod.routing

        def _make_wrapper(lid, orig_fn, coll):
            def wrapper(hidden_states, gating_output, topk, renormalize):
                w, ids = orig_fn(hidden_states, gating_output, topk, renormalize)
                coll.record(lid, ids.cpu())
                return w, ids
            return wrapper

        mod.routing = _make_wrapper(layer_id, orig_routing, collector)
        installed.append((mod, layer_id, orig_routing))
    return installed


def remove_routing_hooks(installed):
    for gate, _lid, orig_routing in installed:
        gate.routing = orig_routing


def install_lifecycle_hooks(dllm, collector: RoutingCollector):
    decoder = dllm.decoder
    model = dllm.model
    orig_block_init = decoder.block_init
    orig_model_forward = model.forward

    def block_init_wrapper(block_x, block_id):
        collector.on_block_init(int(block_id))
        return orig_block_init(block_x, block_id)

    decoder.block_init = block_init_wrapper

    def model_forward_wrapper(input_ids=None, *args, **kwargs):
        collector.on_forward_start()
        result = orig_model_forward(input_ids, *args, **kwargs)
        collector.on_forward_end()
        return result

    model.forward = model_forward_wrapper
    return {
        "orig_block_init": orig_block_init,
        "orig_model_forward": orig_model_forward,
        "decoder": decoder,
        "model": model,
    }


def remove_lifecycle_hooks(originals):
    originals["decoder"].block_init = originals["orig_block_init"]
    originals["model"].forward = originals["orig_model_forward"]


# =====================================================================
# Post-processing
# =====================================================================

def compute_jaccard_matrix(collector: RoutingCollector):
    results = {}
    for lid in sorted(collector.target_layers):
        all_blocks = [b for b in collector.blocks[lid] if len(b) >= 2]
        if not all_blocks:
            results[lid] = {
                "jaccard_matrix": [], "mean_jaccard": 0.0,
                "iter_counts_per_block": [], "T_min": 0, "n_blocks_used": 0,
            }
            continue

        iter_counts = [len(b) for b in all_blocks]
        T = min(iter_counts)
        sum_matrix = np.zeros((T, T))
        n_blocks = 0

        for block in all_blocks:
            seqs = block[:T]
            for i in range(T):
                for j in range(T):
                    if i == j:
                        sum_matrix[i][j] += 1.0
                    else:
                        si = seqs[i]["active_set"]
                        sj = seqs[j]["active_set"]
                        inter = len(si & sj)
                        union = len(si | sj)
                        sum_matrix[i][j] += inter / union if union > 0 else 1.0
            n_blocks += 1

        avg_matrix = (sum_matrix / n_blocks).tolist()
        off_diag = [avg_matrix[i][j] for i in range(T) for j in range(T) if i != j]
        mean_j = float(np.mean(off_diag)) if off_diag else 0.0

        results[lid] = {
            "jaccard_matrix": avg_matrix,
            "mean_jaccard": round(mean_j, 4),
            "iter_counts_per_block": iter_counts,
            "T_min": T,
            "n_blocks_used": n_blocks,
        }
    return results


def compute_concentration_cdf(collector: RoutingCollector):
    results = {}
    for lid in sorted(collector.target_layers):
        agg_hist = np.zeros(N_EXPERTS, dtype=np.int64)
        for block in collector.blocks[lid]:
            for rec in block:
                agg_hist += np.array(rec["histogram"], dtype=np.int64)

        total = int(agg_hist.sum())
        sorted_counts = np.sort(agg_hist)[::-1]
        cumsum = np.cumsum(sorted_counts)
        cdf = (100.0 * cumsum / total).tolist() if total > 0 else [0.0] * N_EXPERTS

        results[lid] = {
            "expert_rank": list(range(1, N_EXPERTS + 1)),
            "cumulative_pct": [round(v, 2) for v in cdf],
            "total_tokens": total,
            "sorted_counts": sorted_counts.tolist(),
        }
    return results


def compute_active_expert_stats(collector: RoutingCollector):
    results = {}
    for lid in sorted(collector.target_layers):
        all_blocks = [b for b in collector.blocks[lid] if len(b) >= 2]
        if not all_blocks:
            results[lid] = {"per_iter_mean": [], "per_iter_std": []}
            continue

        T = min(len(b) for b in all_blocks)
        counts_per_iter = [[] for _ in range(T)]
        for block in all_blocks:
            for i in range(T):
                counts_per_iter[i].append(len(block[i]["active_set"]))

        results[lid] = {
            "per_iter_mean": [round(float(np.mean(c)), 1) for c in counts_per_iter],
            "per_iter_std": [round(float(np.std(c)), 1) for c in counts_per_iter],
        }
    return results


# =====================================================================
# Cross-rank aggregation
# =====================================================================

def gather_and_merge(collector: RoutingCollector, rank: int, world_size: int):
    local_data = {}
    for lid in sorted(collector.target_layers):
        serialized_blocks = []
        for block in collector.blocks[lid]:
            serialized_iters = []
            for rec in block:
                serialized_iters.append({
                    "active_set": list(rec["active_set"]),
                    "histogram": rec["histogram"],
                })
            serialized_blocks.append(serialized_iters)
        local_data[lid] = serialized_blocks

    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_data)

    if rank != 0:
        return None

    merged = RoutingCollector(list(collector.target_layers), N_EXPERTS)
    ref = gathered[0]
    for lid in sorted(collector.target_layers):
        n_blocks = len(ref[lid])
        for b_idx in range(n_blocks):
            merged.blocks[lid].append([])
            valid_ranks = [gd for gd in gathered if b_idx < len(gd[lid])]
            n_iters = min(len(gd[lid][b_idx]) for gd in valid_ranks)
            for i_idx in range(n_iters):
                merged_active = set()
                merged_hist = np.zeros(N_EXPERTS, dtype=np.int64)
                for gd in valid_ranks:
                    if i_idx < len(gd[lid][b_idx]):
                        rec = gd[lid][b_idx][i_idx]
                        merged_active.update(rec["active_set"])
                        merged_hist += np.array(rec["histogram"], dtype=np.int64)
                merged.blocks[lid][-1].append({
                    "active_set": frozenset(merged_active),
                    "histogram": merged_hist.tolist(),
                })
    return merged


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Collect routing stability data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=BLOCK_LENGTH)
    parser.add_argument("--tp-size", type=int, default=None)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if args.tp_size is not None:
        tp_size = args.tp_size
    elif world_size == 8:
        tp_size = 4
    elif world_size == 4:
        tp_size = 2
    else:
        raise AssertionError(f"Requires 8 or 4 GPUs, got {world_size}")

    dp_size = world_size // tp_size
    dp_rank = rank // tp_size
    local_bs = args.batch_size // dp_size
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # --- distributed init ---
    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    pcfg_init = ParallelConfig(
        tensor_parallel_size=1, data_parallel_size=1,
        enable_expert_parallel=True,
    )
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(
            world_size, rank, "env://", local_rank, "nccl"
        )

    pcfg = ParallelConfig(
        tensor_parallel_size=tp_size, data_parallel_size=dp_size,
        data_parallel_rank=dp_rank, enable_expert_parallel=True,
    )
    vllm_cfg = VllmConfig(parallel_config=pcfg)

    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, backend="nccl",
        )

        from vllm.distributed import prepare_communication_buffer_for_model
        from vllm.forward_context import set_forward_context
        from dinfer import (
            BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
            ThresholdParallelDecoder,
        )
        from dinfer.model import LLaDA2MoeModelLM
        from transformers import AutoConfig, AutoTokenizer
        from test_heteval512 import PROMPTS
        from baseline_optimizations import apply_all_optimizations

        if rank == 0:
            print("=" * 70)
            print("MoE Routing Stability Data Collection (K=8 native, no EB)")
            print(f"  batch={args.batch_size} gen={args.gen_length} block={args.block_length}")
            print(f"  tp={tp_size} dp={dp_size} ep={world_size}")
            print(f"  target_layers={TARGET_LAYERS}")
            print("=" * 70)

        # --- model loading ---
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)

        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        with torch.inference_mode():
            warmup_tok = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(
                attn_metadata=None, vllm_config=vllm_cfg,
                num_tokens=warmup_tok.numel(),
            ):
                _ = model(warmup_tok, use_cache=False)

        apply_all_optimizations(model)
        prepare_communication_buffer_for_model(model)

        # --- tokenization ---
        all_ids = []
        for i in range(args.batch_size):
            text = PROMPTS[i % len(PROMPTS)]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=False,
                )
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [
            torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
            if ids.shape[0] < mx else ids
            for ids in all_ids
        ]
        input_ids_full = torch.stack(padded, dim=0)
        my_input = input_ids_full[dp_rank * local_bs: (dp_rank + 1) * local_bs].to(device)

        if rank == 0:
            print(f"  Input shape (local): {tuple(my_input.shape)}")

        # --- decoder + dllm factory ---
        decoder = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90,
            mask_id=MASK_ID, eos_id=EOS_ID,
        )

        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend="vllm", lazy_cache_update=True, inplace_cache_update=True,
            )

        # --- warmup run (no hooks) ---
        if rank == 0:
            print("  Warmup run...")
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.iter_no = 0
            _ = dllm.generate(
                my_input.clone(), gen_length=args.gen_length,
                block_length=args.block_length,
            )
        torch.cuda.synchronize()
        dist.barrier()

        # --- collection run ---
        if rank == 0:
            print("  Collection run...")
        collector = RoutingCollector(TARGET_LAYERS, N_EXPERTS)
        dllm = make_dllm()
        hooks = install_routing_hooks(model, collector)
        lifecycle = install_lifecycle_hooks(dllm, collector)

        t0 = time.time()
        try:
            with torch.inference_mode():
                dllm.diff_iteration.iter_no = 0
                out = dllm.generate(
                    my_input.clone(), gen_length=args.gen_length,
                    block_length=args.block_length,
                )
            torch.cuda.synchronize()
            dist.barrier()
            total_fwd = dllm.diff_iteration.num_forwards
        finally:
            remove_routing_hooks(hooks)
            remove_lifecycle_hooks(lifecycle)

        elapsed = time.time() - t0
        if rank == 0:
            print(f"  Collection done: {total_fwd} forwards in {elapsed:.1f}s")
            for lid in TARGET_LAYERS:
                n_blk = len(collector.blocks[lid])
                n_rec = sum(len(b) for b in collector.blocks[lid])
                print(f"    Layer {lid}: {n_blk} blocks, {n_rec} iter records")

        # --- gather across ranks ---
        if rank == 0:
            print("  Gathering data across ranks...")
        merged = gather_and_merge(collector, rank, world_size)

        # --- post-processing + save (rank 0 only) ---
        if rank == 0:
            print("  Computing Jaccard matrices...")
            jaccard_results = compute_jaccard_matrix(merged)
            print("  Computing CDFs...")
            cdf_results = compute_concentration_cdf(merged)
            print("  Computing active expert stats...")
            active_stats = compute_active_expert_stats(merged)

            for lid in TARGET_LAYERS:
                jr = jaccard_results[lid]
                cr = cdf_results[lid]
                ar = active_stats[lid]
                print(f"\n  Layer {lid}:")
                print(f"    Jaccard: mean={jr['mean_jaccard']}, T_min={jr['T_min']}, "
                      f"n_blocks={jr['n_blocks_used']}")
                print(f"    CDF: total_tokens={cr['total_tokens']}")
                if cr["cumulative_pct"]:
                    idx_50 = next((i for i, v in enumerate(cr["cumulative_pct"]) if v >= 50), -1)
                    idx_90 = next((i for i, v in enumerate(cr["cumulative_pct"]) if v >= 90), -1)
                    print(f"    CDF: top-{idx_50+1} experts cover 50%, "
                          f"top-{idx_90+1} experts cover 90%")
                if ar["per_iter_mean"]:
                    print(f"    Active experts: mean={ar['per_iter_mean'][0]:.0f}-"
                          f"{ar['per_iter_mean'][-1]:.0f} / {N_EXPERTS}")

            meta = {
                "batch_size": args.batch_size,
                "gen_length": args.gen_length,
                "block_length": args.block_length,
                "tp_size": tp_size,
                "dp_size": dp_size,
                "world_size": world_size,
                "target_layers": TARGET_LAYERS,
                "n_experts": N_EXPERTS,
                "topk": 8,
                "eb_enabled": False,
                "total_fwd": total_fwd,
                "elapsed_sec": round(elapsed, 1),
            }

            output = {
                "meta": meta,
                "jaccard": {f"layer_{lid}": jaccard_results[lid] for lid in TARGET_LAYERS},
                "cdf": {f"layer_{lid}": cdf_results[lid] for lid in TARGET_LAYERS},
                "active_expert_count": {
                    f"layer_{lid}": active_stats[lid] for lid in TARGET_LAYERS
                },
            }

            out_path = REPO_ROOT / "codex_coding" / "results" / "plt" / "routing_stability_data.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\n  Saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
            print("Done.")

        dist.barrier()


if __name__ == "__main__":
    main()
