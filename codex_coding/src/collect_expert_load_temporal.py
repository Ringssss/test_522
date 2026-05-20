#!/usr/bin/env python3
"""
Expert load temporal analysis: per-forward, per-block dynamics.

Collects topk_ids for every forward and analyzes:
1. Per-GPU token-expert pairs count per forward (time series)
2. Block boundary detection (N jumps up = new block)
3. Intra-block convergence (N decreases as tokens are decoded)
4. Cross-block load shift

Usage:
  CUDA_VISIBLE_DEVICES=4 python codex_coding/src/collect_expert_load_temporal.py
"""

from __future__ import annotations
import os, sys, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
NUM_EXPERTS = 256
EP_SIZE = 4
EXPERTS_PER_GPU = NUM_EXPERTS // EP_SIZE


class TemporalCollector:
    """Collect per-forward, per-layer routing data."""

    def __init__(self):
        self.fwd_idx = 0
        # records[fwd_idx] = {layer_idx: topk_ids_numpy}
        self.records = defaultdict(dict)

    def record(self, layer_idx, topk_ids):
        self.records[self.fwd_idx][layer_idx] = topk_ids.cpu().numpy().copy()

    def next_fwd(self):
        self.fwd_idx += 1


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")

    BATCH_SIZE = 128
    GEN_LENGTH = 256

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoTokenizer, AutoConfig
    from test_fused_eb_triton import fused_routing
    from test_m_skip_sweep import MSkipEBController
    from baseline_optimizations import apply_all_optimizations
    from test_heteval128 import PROMPTS

    vllm_dist.init_distributed_environment(1, 0, "env://", 0, "nccl")
    vllm_dist.initialize_model_parallel(1, backend="nccl")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=DEVICE)
        model = model.to(DEVICE)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=DEVICE).unsqueeze(0),
                      use_cache=False)
        apply_all_optimizations(model)

    # Build input
    all_ids = []
    for i in range(BATCH_SIZE):
        text = PROMPTS[i % len(PROMPTS)]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=False)
        all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
    mx = max(x.shape[0] for x in all_ids)
    pad_id = tokenizer.pad_token_id or 0
    padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
              if ids.shape[0] < mx else ids for ids in all_ids]
    input_ids = torch.stack(padded, dim=0).to(DEVICE)
    print(f"Input shape: {input_ids.shape}")

    decoder = ThresholdParallelDecoder(
        temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

    # --- C11-M5-K4 with temporal collection ---
    collector = TemporalCollector()

    ctrl = MSkipEBController(
        num_layers=19, K=8, M=4, K_target=40,
        quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)

    # Count forwards per layer to detect fwd boundaries
    layer_fwd_counter = [0]

    gate_idx = 0
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "LLaDA2MoeGate":
            b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                 mod.top_k, mod.n_group, mod.topk_group)
            li = gate_idx
            def mk(bb, rr, nn, gg, layer_i, cc, coll):
                def fn(hs, go, topk, renorm):
                    s_mask = cc.get_s_mask(layer_i, go, bb)
                    w, idx = fused_routing(go, bb, rr, s_mask=s_mask, K=4, ng=nn, tkg=gg)
                    coll.record(layer_i, idx)
                    # Detect new forward: layer 0 resets counter
                    if layer_i == 0:
                        if layer_fwd_counter[0] > 0:
                            coll.next_fwd()
                        layer_fwd_counter[0] += 1
                    return w.to(go.dtype), idx
                return fn
            mod.routing = mk(b, r, ng, tkg, li, ctrl, collector)
            gate_idx += 1

    ctrl.prev_N.clear(); ctrl.K_init.clear()
    ctrl.cold_count = 0; ctrl.hot_count = 0
    ctrl.eb_calls = 0; ctrl.eb_skips = 0
    ctrl._bufs.clear(); ctrl.k_init_history.clear()
    ctrl.s_mask_cache.clear(); ctrl.pop_cache.clear()
    ctrl._fwd_in_block.clear(); ctrl._block_idx.clear()

    dllm = BlockDiffusionLLM(
        model, decoder, BlockIteratorFactory(use_block_diffusion=True),
        cache_factory=KVCacheFactory("prefix", is_bd_model=True),
        early_stop=True, maximum_unroll=4, expected_tpf=15,
        backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

    with torch.inference_mode():
        dllm.diff_iteration.num_forwards = 0
        _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)

    total_fwd = dllm.diff_iteration.num_forwards
    print(f"\nC11-M5-K4: {total_fwd} forwards, "
          f"cold={ctrl.cold_count}, hot={ctrl.hot_count}")
    print(f"Collected data for {len(collector.records)} forwards")

    # ========== Analysis ==========
    # Use layer 1 (first MoE layer, index 0) as representative
    ANALYSIS_LAYER = 0  # layer index in MoE layers (L1 in model)

    print(f"\n{'='*80}")
    print(f"TEMPORAL ANALYSIS — Layer {ANALYSIS_LAYER+1} (MoE layer {ANALYSIS_LAYER})")
    print(f"  EP={EP_SIZE}, {EXPERTS_PER_GPU} experts/GPU, K=4")
    print(f"{'='*80}")

    # Per-forward time series
    fwd_data = []
    for fi in range(len(collector.records)):
        if ANALYSIS_LAYER not in collector.records[fi]:
            continue
        topk_ids = collector.records[fi][ANALYSIS_LAYER]
        N, K = topk_ids.shape
        flat = topk_ids.flatten()
        gpu_counts = []
        for g in range(EP_SIZE):
            start = g * EXPERTS_PER_GPU
            end = (g + 1) * EXPERTS_PER_GPU
            gpu_counts.append(int(np.sum((flat >= start) & (flat < end))))
        total = sum(gpu_counts)
        skew = max(gpu_counts) / max(min(gpu_counts), 1)
        fwd_data.append({
            'fwd': fi, 'N': N, 'K': K, 'total_pairs': total,
            'gpu_counts': gpu_counts, 'skew': skew,
        })

    # Detect block boundaries: N jumps up
    block_boundaries = [0]
    for i in range(1, len(fwd_data)):
        if fwd_data[i]['N'] > fwd_data[i-1]['N']:
            block_boundaries.append(i)

    num_blocks = len(block_boundaries)
    print(f"\n  Total forwards: {len(fwd_data)}")
    print(f"  Detected blocks: {num_blocks}")
    print(f"  Block boundaries at forwards: {block_boundaries[:10]}...")

    # Print per-forward details (first 3 blocks + last block)
    blocks_to_show = list(range(min(3, num_blocks))) + [num_blocks - 1]
    blocks_to_show = sorted(set(blocks_to_show))

    for bi in blocks_to_show:
        block_start = block_boundaries[bi]
        block_end = block_boundaries[bi + 1] if bi + 1 < num_blocks else len(fwd_data)
        block_fwds = fwd_data[block_start:block_end]

        print(f"\n  --- Block {bi} (fwd {block_start}-{block_end-1}, {len(block_fwds)} forwards) ---")
        print(f"  {'Fwd':>5s} {'N':>6s} {'Total':>7s} "
              + " ".join(f"{'GPU'+str(g):>7s}" for g in range(EP_SIZE))
              + f" {'Skew':>6s}")

        # Show first 5, last 2 of each block
        indices = list(range(min(5, len(block_fwds))))
        if len(block_fwds) > 7:
            indices.append(-1)  # separator
            indices.extend(range(len(block_fwds)-2, len(block_fwds)))

        for idx in indices:
            if idx == -1:
                print(f"  {'...':>5s}")
                continue
            d = block_fwds[idx]
            gpu_str = " ".join(f"{c:>7d}" for c in d['gpu_counts'])
            print(f"  {d['fwd']:>5d} {d['N']:>6d} {d['total_pairs']:>7d} "
                  f"{gpu_str} {d['skew']:>6.2f}")

    # Cross-block summary
    print(f"\n  --- Cross-block load summary (layer {ANALYSIS_LAYER+1}) ---")
    print(f"  {'Block':>6s} {'Fwds':>5s} {'AvgN':>7s} "
          + " ".join(f"{'GPU'+str(g):>7s}" for g in range(EP_SIZE))
          + f" {'AvgSkew':>8s} {'MaxSkew':>8s}")

    block_summaries = []
    for bi in range(num_blocks):
        block_start = block_boundaries[bi]
        block_end = block_boundaries[bi + 1] if bi + 1 < num_blocks else len(fwd_data)
        block_fwds = fwd_data[block_start:block_end]

        avg_n = np.mean([d['N'] for d in block_fwds])
        avg_gpu = [np.mean([d['gpu_counts'][g] for d in block_fwds]) for g in range(EP_SIZE)]
        avg_skew = np.mean([d['skew'] for d in block_fwds])
        max_skew = max(d['skew'] for d in block_fwds)

        gpu_str = " ".join(f"{c:>7.0f}" for c in avg_gpu)
        print(f"  {bi:>6d} {len(block_fwds):>5d} {avg_n:>7.0f} "
              f"{gpu_str} {avg_skew:>8.2f} {max_skew:>8.2f}")

        block_summaries.append({
            'block': bi, 'num_fwds': len(block_fwds),
            'avg_N': float(avg_n),
            'avg_gpu_load': [float(x) for x in avg_gpu],
            'avg_skew': float(avg_skew),
            'max_skew': float(max_skew),
        })

    # Multi-layer comparison (show all 19 layers for one block)
    print(f"\n  --- Multi-layer load for Block 0 ---")
    print(f"  {'Layer':>6s} "
          + " ".join(f"{'GPU'+str(g):>7s}" for g in range(EP_SIZE))
          + f" {'AvgSkew':>8s}")

    block0_end = block_boundaries[1] if num_blocks > 1 else len(fwd_data)
    for li in range(19):
        gpu_avg = np.zeros(EP_SIZE)
        count = 0
        for fi in range(block0_end):
            if li in collector.records[fi]:
                topk_ids = collector.records[fi][li]
                flat = topk_ids.flatten()
                for g in range(EP_SIZE):
                    start = g * EXPERTS_PER_GPU
                    end = (g + 1) * EXPERTS_PER_GPU
                    gpu_avg[g] += np.sum((flat >= start) & (flat < end))
                count += 1
        if count > 0:
            gpu_avg /= count
        skew = gpu_avg.max() / max(gpu_avg.min(), 1)
        gpu_str = " ".join(f"{c:>7.0f}" for c in gpu_avg)
        print(f"  L{li+1:>4d} {gpu_str} {skew:>8.2f}")

    # Save
    results = {
        'config': 'C11-M5-K4',
        'ep_size': EP_SIZE,
        'total_fwd': total_fwd,
        'num_blocks': num_blocks,
        'block_summaries': block_summaries,
        'per_fwd_data': [
            {'fwd': d['fwd'], 'N': d['N'], 'total_pairs': d['total_pairs'],
             'gpu_counts': d['gpu_counts'], 'skew': d['skew']}
            for d in fwd_data
        ],
    }
    out_path = REPO_ROOT / "codex_coding" / "results" / "expert_load_temporal.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
