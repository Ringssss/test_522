#!/usr/bin/env python3
"""
EB impact on expert load distribution: C5 (no EB, K=8) vs C11-M5-K4 (EB, K=4).
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
EPG = NUM_EXPERTS // EP_SIZE


class Collector:
    def __init__(self):
        self.fwd_idx = 0
        self.records = defaultdict(dict)
        self._layer_seen = set()

    def record(self, layer_idx, topk_ids):
        self.records[self.fwd_idx][layer_idx] = topk_ids.cpu().numpy().copy()
        if layer_idx == 0 and 0 in self._layer_seen:
            self.fwd_idx += 1
            self._layer_seen.clear()
        self._layer_seen.add(layer_idx)

    def gpu_load(self, fwd, layer):
        """Return [EP_SIZE] array of token-expert pairs per GPU."""
        ids = self.records[fwd][layer].flatten()
        return np.array([np.sum((ids >= g*EPG) & (ids < (g+1)*EPG)) for g in range(EP_SIZE)])


def compare_configs(c5: Collector, eb: Collector):
    """Compare C5 vs C11-M5-K4 load distributions."""
    num_fwd_c5 = len(c5.records)
    num_fwd_eb = len(eb.records)
    layers = sorted(c5.records[0].keys())

    print(f"\n{'='*80}")
    print(f"EB IMPACT ON LOAD DISTRIBUTION (EP={EP_SIZE})")
    print(f"  C5 K8: {num_fwd_c5} fwd | C11-M5-K4: {num_fwd_eb} fwd")
    print(f"{'='*80}")

    # Per-layer comparison
    print(f"\n  --- Per-layer avg skew ---")
    print(f"  {'Layer':>6s} {'C5_skew':>9s} {'EB_skew':>9s} {'Δ':>8s}  {'C5 hot GPU':>12s} {'EB hot GPU':>12s}")

    for li in layers:
        # C5
        c5_skews = []
        c5_gpu_avg = np.zeros(EP_SIZE)
        for fi in range(num_fwd_c5):
            if li in c5.records[fi]:
                load = c5.gpu_load(fi, li)
                c5_skews.append(load.max() / max(load.min(), 1))
                c5_gpu_avg += load
        c5_gpu_avg /= max(len(c5_skews), 1)
        c5_avg_skew = np.mean(c5_skews) if c5_skews else 0

        # EB
        eb_skews = []
        eb_gpu_avg = np.zeros(EP_SIZE)
        for fi in range(num_fwd_eb):
            if li in eb.records[fi]:
                load = eb.gpu_load(fi, li)
                eb_skews.append(load.max() / max(load.min(), 1))
                eb_gpu_avg += load
        eb_gpu_avg /= max(len(eb_skews), 1)
        eb_avg_skew = np.mean(eb_skews) if eb_skews else 0

        delta = eb_avg_skew - c5_avg_skew
        c5_hot = f"GPU{c5_gpu_avg.argmax()}({c5_gpu_avg.max():.0f})"
        eb_hot = f"GPU{eb_gpu_avg.argmax()}({eb_gpu_avg.max():.0f})"
        marker = " ★" if abs(delta) > 0.5 else ""
        print(f"  L{li+1:>4d} {c5_avg_skew:>9.2f} {eb_avg_skew:>9.2f} {delta:>+8.2f}{marker}  {c5_hot:>12s} {eb_hot:>12s}")

    # Block-level: first 3 forwards of each block (where skew is worst)
    print(f"\n  --- Block-start skew (first 3 fwds of each block, Layer 1) ---")
    print(f"  {'Config':>12s} {'Blk':>4s} {'Fwd':>4s} {'N':>6s} "
          + " ".join(f"{'GPU'+str(g):>7s}" for g in range(EP_SIZE))
          + f" {'Skew':>7s}")

    for label, coll in [("C5_K8", c5), ("C11-M5-K4", eb)]:
        # Detect block boundaries for this collector
        fwd_ns = []
        for fi in range(len(coll.records)):
            if 0 in coll.records[fi]:
                N = coll.records[fi][0].shape[0]
                fwd_ns.append((fi, N))
        boundaries = [0]
        for i in range(1, len(fwd_ns)):
            if fwd_ns[i][1] > fwd_ns[i-1][1]:
                boundaries.append(i)

        for bi, bs in enumerate(boundaries[:4]):
            for offset in range(min(3, len(fwd_ns) - bs)):
                fi, N = fwd_ns[bs + offset]
                load = coll.gpu_load(fi, 0)  # layer 0
                skew = load.max() / max(load.min(), 1)
                gpu_str = " ".join(f"{c:>7d}" for c in load)
                print(f"  {label:>12s} {bi:>4d} {fi:>4d} {N:>6d} {gpu_str} {skew:>7.2f}")
        print()

    # Total pairs comparison (K=8 vs K=4)
    print(f"  --- Total token-expert pairs per forward ---")
    for label, coll in [("C5_K8", c5), ("C11-M5-K4", eb)]:
        totals = []
        for fi in range(len(coll.records)):
            if 0 in coll.records[fi]:
                totals.append(coll.records[fi][0].size)  # N*K
        avg_total = np.mean(totals)
        print(f"  {label}: avg {avg_total:.0f} pairs/fwd (K={'8' if 'K8' in label else '4'})")


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=DEVICE)
        model = model.to(DEVICE)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=DEVICE).unsqueeze(0), use_cache=False)
        apply_all_optimizations(model)

    all_ids = []
    for i in range(128):
        text = PROMPTS[i]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                  add_generation_prompt=True, tokenize=False)
        all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
    mx = max(x.shape[0] for x in all_ids)
    pad_id = tokenizer.pad_token_id or 0
    padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
              if ids.shape[0] < mx else ids for ids in all_ids]
    input_ids = torch.stack(padded, dim=0).to(DEVICE)

    decoder = ThresholdParallelDecoder(temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

    def make_dllm():
        return BlockDiffusionLLM(
            model, decoder, BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=4, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

    orig_routings = {}
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "LLaDA2MoeGate":
            orig_routings[name] = mod.routing

    # ========== C5 K8 ==========
    print("Collecting C5 K8...")
    c5_coll = Collector()
    gate_idx = 0
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "LLaDA2MoeGate":
            b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                 mod.top_k, mod.n_group, mod.topk_group)
            li = gate_idx
            def mk(bb, rr, tt, nn, gg, layer_i, coll):
                def fn(hs, go, topk, renorm):
                    w, idx = fused_routing(go, bb, rr, s_mask=None, K=tt, ng=nn, tkg=gg)
                    coll.record(layer_i, idx)
                    return w.to(go.dtype), idx
                return fn
            mod.routing = mk(b, r, tk, ng, tkg, li, c5_coll)
            gate_idx += 1

    dllm = make_dllm()
    with torch.inference_mode():
        dllm.diff_iteration.num_forwards = 0
        _ = dllm.generate(input_ids.clone(), gen_length=256, block_length=BLOCK_LENGTH)
    print(f"  C5: {dllm.diff_iteration.num_forwards} fwd, collected {len(c5_coll.records)}")

    # ========== C11-M5-K4 ==========
    print("Collecting C11-M5-K4...")
    eb_coll = Collector()
    ctrl = MSkipEBController(num_layers=19, K=8, M=4, K_target=40,
                              quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)

    gate_idx = 0
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "LLaDA2MoeGate":
            b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                 mod.top_k, mod.n_group, mod.topk_group)
            li = gate_idx
            def mk(bb, rr, nn, gg, layer_i, cc, coll):
                def fn(hs, go, topk, renorm):
                    sm = cc.get_s_mask(layer_i, go, bb)
                    w, idx = fused_routing(go, bb, rr, s_mask=sm, K=4, ng=nn, tkg=gg)
                    coll.record(layer_i, idx)
                    return w.to(go.dtype), idx
                return fn
            mod.routing = mk(b, r, ng, tkg, li, ctrl, eb_coll)
            gate_idx += 1

    ctrl.prev_N.clear(); ctrl.K_init.clear()
    ctrl.cold_count = 0; ctrl.hot_count = 0
    ctrl.eb_calls = 0; ctrl.eb_skips = 0
    ctrl._bufs.clear(); ctrl.k_init_history.clear()
    ctrl.s_mask_cache.clear(); ctrl.pop_cache.clear()
    ctrl._fwd_in_block.clear(); ctrl._block_idx.clear()

    dllm = make_dllm()
    with torch.inference_mode():
        dllm.diff_iteration.num_forwards = 0
        _ = dllm.generate(input_ids.clone(), gen_length=256, block_length=BLOCK_LENGTH)
    print(f"  C11-M5-K4: {dllm.diff_iteration.num_forwards} fwd, collected {len(eb_coll.records)}")

    compare_configs(c5_coll, eb_coll)


if __name__ == "__main__":
    main()
