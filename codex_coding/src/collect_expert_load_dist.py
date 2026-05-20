#!/usr/bin/env python3
"""
Expert load distribution analysis for EPLB feasibility.

Collects per-forward, per-layer routing decisions (topk_ids) and analyzes:
1. Per-GPU partition load (token-expert pairs per GPU)
2. Load skew ratio (max/min across GPUs)
3. Per-layer skew variation
4. EB S_mask impact on load distribution

Runs single-GPU (no torchrun needed), analyzes routing as if EP=4.

Usage:
  CUDA_VISIBLE_DEVICES=4 python codex_coding/src/collect_expert_load_dist.py
"""

from __future__ import annotations
import os, sys, json, time
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
EP_SIZES = [4, 8]  # Analyze for both 4-GPU and 8-GPU partitions


class LoadDistCollector:
    """Hook into gate.routing to collect topk_ids per layer per forward."""

    def __init__(self, num_experts=256):
        self.num_experts = num_experts
        self.fwd_count = 0
        # per_layer_data[layer_idx] = list of topk_ids arrays (one per fwd)
        self.per_layer_data = defaultdict(list)
        self.s_mask_data = defaultdict(list)  # S_mask per layer per fwd

    def record(self, layer_idx, topk_ids, s_mask=None):
        """Record topk_ids [N, K] for one layer in one forward."""
        self.per_layer_data[layer_idx].append(topk_ids.cpu().numpy().copy())
        if s_mask is not None:
            self.s_mask_data[layer_idx].append(s_mask.cpu().numpy().copy())

    def next_fwd(self):
        self.fwd_count += 1


def analyze_load_distribution(collector, ep_size, config_name):
    """Analyze load distribution across EP partitions."""
    num_experts = collector.num_experts
    experts_per_gpu = num_experts // ep_size
    num_layers = len(collector.per_layer_data)
    num_fwds = collector.fwd_count

    print(f"\n{'='*80}")
    print(f"EXPERT LOAD DISTRIBUTION — {config_name} (EP={ep_size})")
    print(f"  {num_experts} experts, {experts_per_gpu} per GPU, "
          f"{num_layers} MoE layers, {num_fwds} forwards")
    print(f"{'='*80}")

    # Per-layer analysis
    layer_skews = []
    layer_gpu_loads = []
    all_layer_stats = []

    for li in sorted(collector.per_layer_data.keys()):
        fwd_list = collector.per_layer_data[li]
        gpu_loads_all_fwds = []  # [num_fwds, ep_size]

        for topk_ids in fwd_list:
            # topk_ids: [N, K] — count token-expert pairs per GPU partition
            flat = topk_ids.flatten()
            gpu_counts = np.zeros(ep_size, dtype=np.int64)
            for g in range(ep_size):
                start = g * experts_per_gpu
                end = (g + 1) * experts_per_gpu
                gpu_counts[g] = np.sum((flat >= start) & (flat < end))
            gpu_loads_all_fwds.append(gpu_counts)

        gpu_loads = np.array(gpu_loads_all_fwds)  # [num_fwds, ep_size]
        avg_load = gpu_loads.mean(axis=0)  # [ep_size]
        max_load = avg_load.max()
        min_load = avg_load.min()
        skew = max_load / min_load if min_load > 0 else float('inf')

        # Per-forward skew
        per_fwd_skew = gpu_loads.max(axis=1) / np.maximum(gpu_loads.min(axis=1), 1)
        avg_skew = per_fwd_skew.mean()
        max_skew = per_fwd_skew.max()

        layer_skews.append(avg_skew)
        layer_gpu_loads.append(avg_load)

        all_layer_stats.append({
            'layer': li,
            'avg_load_per_gpu': avg_load.tolist(),
            'avg_skew': float(avg_skew),
            'max_skew': float(max_skew),
            'total_pairs': float(gpu_loads.sum(axis=1).mean()),
        })

    # Print per-layer summary
    print(f"\n  {'Layer':<8s} ", end="")
    for g in range(ep_size):
        print(f"{'GPU'+str(g):>10s}", end="")
    print(f"  {'AvgSkew':>8s} {'MaxSkew':>8s}")
    print(f"  {'-'*(8 + ep_size*10 + 20)}")

    for stats in all_layer_stats:
        li = stats['layer']
        print(f"  L{li:<6d} ", end="")
        for load in stats['avg_load_per_gpu']:
            print(f"{load:>10.0f}", end="")
        print(f"  {stats['avg_skew']:>8.3f} {stats['max_skew']:>8.3f}")

    # Global summary
    all_gpu_loads = np.array(layer_gpu_loads)  # [num_layers, ep_size]
    global_avg = all_gpu_loads.mean(axis=0)
    global_skew = global_avg.max() / global_avg.min()
    worst_layer = max(all_layer_stats, key=lambda x: x['avg_skew'])
    best_layer = min(all_layer_stats, key=lambda x: x['avg_skew'])

    print(f"\n  Global avg load per GPU: {global_avg.tolist()}")
    print(f"  Global skew (max/min):   {global_skew:.3f}")
    print(f"  Worst layer: L{worst_layer['layer']} (skew={worst_layer['avg_skew']:.3f})")
    print(f"  Best layer:  L{best_layer['layer']} (skew={best_layer['avg_skew']:.3f})")

    # Ideal vs actual: if perfectly balanced, each GPU gets total/ep_size pairs
    total_pairs = all_layer_stats[0]['total_pairs']
    ideal_per_gpu = total_pairs / ep_size
    actual_max = max(global_avg)
    overhead_pct = (actual_max / ideal_per_gpu - 1) * 100
    print(f"\n  Ideal load per GPU:   {ideal_per_gpu:.0f} pairs")
    print(f"  Actual max GPU load:  {actual_max:.0f} pairs")
    print(f"  Load overhead:        {overhead_pct:.1f}% "
          f"({'significant' if overhead_pct > 10 else 'moderate' if overhead_pct > 5 else 'small'})")

    # S_mask analysis (if EB was used)
    if collector.s_mask_data:
        print(f"\n  --- S_mask distribution across GPUs ---")
        for li in sorted(collector.s_mask_data.keys())[:3]:  # Show first 3 layers
            masks = collector.s_mask_data[li]
            gpu_expert_counts = np.zeros(ep_size)
            for mask in masks:
                active_experts = np.where(mask > 0)[0]
                for g in range(ep_size):
                    start = g * experts_per_gpu
                    end = (g + 1) * experts_per_gpu
                    gpu_expert_counts[g] += np.sum((active_experts >= start) & (active_experts < end))
            avg_per_gpu = gpu_expert_counts / len(masks)
            total_active = avg_per_gpu.sum()
            print(f"    L{li}: |S|={total_active:.0f}, per-GPU: "
                  f"{[f'{x:.1f}' for x in avg_per_gpu]}, "
                  f"skew={avg_per_gpu.max()/max(avg_per_gpu.min(),1):.2f}")

    return {
        'config': config_name,
        'ep_size': ep_size,
        'global_skew': float(global_skew),
        'global_avg_load': global_avg.tolist(),
        'overhead_pct': float(overhead_pct),
        'worst_layer': worst_layer,
        'best_layer': best_layer,
        'per_layer': all_layer_stats,
    }


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    BATCH_SIZE = 128
    GEN_LENGTH = 256  # Full generation for representative routing stats

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

    # Single-GPU setup with vllm distributed (needed for FusedMoE)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")
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

    results = {}

    # ========== Config 1: C5 (fused routing, no EB) ==========
    print("\n\nCollecting C5 routing data...")
    collector_c5 = LoadDistCollector()

    # Patch routing to collect topk_ids
    for li_idx, (name, mod) in enumerate(
            [(n, m) for n, m in model.named_modules() if m.__class__.__name__ == "LLaDA2MoeGate"]):
        b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                             mod.top_k, mod.n_group, mod.topk_group)
        layer_i = li_idx
        def mk_c5(bb, rr, tt, nn, gg, li, coll):
            def fn(hs, go, topk, renorm):
                w, idx = fused_routing(go, bb, rr, s_mask=None, K=tt, ng=nn, tkg=gg)
                coll.record(li, idx)
                return w.to(go.dtype), idx
            return fn
        mod.routing = mk_c5(b, r, tk, ng, tkg, layer_i, collector_c5)

    dllm = BlockDiffusionLLM(
        model, decoder, BlockIteratorFactory(use_block_diffusion=True),
        cache_factory=KVCacheFactory("prefix", is_bd_model=True),
        early_stop=True, maximum_unroll=4, expected_tpf=15,
        backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

    with torch.inference_mode():
        dllm.diff_iteration.num_forwards = 0
        _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
    collector_c5.fwd_count = dllm.diff_iteration.num_forwards
    print(f"  C5: {collector_c5.fwd_count} forwards")

    for ep in EP_SIZES:
        r = analyze_load_distribution(collector_c5, ep, f"C5_K8")
        results[f'C5_K8_EP{ep}'] = r

    # ========== Config 2: C11-M5-K4 (EB + topk=4) ==========
    print("\n\nCollecting C11-M5-K4 routing data...")
    collector_eb = LoadDistCollector()

    ctrl = MSkipEBController(
        num_layers=19, K=8, M=4, K_target=40,
        quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)

    gate_idx = 0
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "LLaDA2MoeGate":
            b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                 mod.top_k, mod.n_group, mod.topk_group)
            li = gate_idx
            def mk_eb(bb, rr, tt, nn, gg, layer_i, cc, coll):
                def fn(hs, go, topk, renorm):
                    s_mask = cc.get_s_mask(layer_i, go, bb)
                    w, idx = fused_routing(go, bb, rr, s_mask=s_mask, K=4, ng=nn, tkg=gg)
                    coll.record(layer_i, idx, s_mask=s_mask)
                    return w.to(go.dtype), idx
                return fn
            mod.routing = mk_eb(b, r, tk, ng, tkg, li, ctrl, collector_eb)
            gate_idx += 1

    # Reset EB controller
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
    collector_eb.fwd_count = dllm.diff_iteration.num_forwards
    print(f"  C11-M5-K4: {collector_eb.fwd_count} forwards, "
          f"cold={ctrl.cold_count}, hot={ctrl.hot_count}")

    for ep in EP_SIZES:
        r = analyze_load_distribution(collector_eb, ep, f"C11-M5-K4")
        results[f'C11_M5_K4_EP{ep}'] = r

    # Save
    out_path = REPO_ROOT / "codex_coding" / "results" / "expert_load_distribution.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
