#!/usr/bin/env python3
"""
Insight validation experiments for dLLM MoE paper.

E1: Cross-dataset stability (3 datasets × batch=128)
E2: Block-start routing determinism (cross-block + cross-dataset)
E3: Cross-batch-size stability (batch=32/128)
E4: Full 19-layer temporal profile
E5: Communication envelope tightness (S_mask prediction vs actual)

Usage:
  CUDA_VISIBLE_DEVICES=4 python codex_coding/src/validate_insights.py
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
EP_SIZE = 4
EPG = NUM_EXPERTS // EP_SIZE  # 64


# ================================================================
# Data collector
# ================================================================
class InsightCollector:
    """Collect per-forward, per-layer routing + S_mask data."""

    def __init__(self):
        self.fwd_idx = 0
        # topk_data[fwd_idx][layer_idx] = topk_ids numpy [N, K]
        self.topk_data = defaultdict(dict)
        # smask_data[fwd_idx][layer_idx] = s_mask numpy [256] (bool/int)
        self.smask_data = defaultdict(dict)
        self._seen_l0 = False

    def record(self, layer_idx, topk_ids, s_mask=None):
        if layer_idx == 0:
            if self._seen_l0:
                self.fwd_idx += 1
            self._seen_l0 = True
        self.topk_data[self.fwd_idx][layer_idx] = topk_ids.cpu().numpy().copy()
        if s_mask is not None:
            self.smask_data[self.fwd_idx][layer_idx] = s_mask.cpu().numpy().copy()

    def gpu_load(self, fwd, layer):
        ids = self.topk_data[fwd][layer].flatten()
        return np.array([np.sum((ids >= g*EPG) & (ids < (g+1)*EPG)) for g in range(EP_SIZE)])

    def num_fwds(self):
        return self.fwd_idx + 1 if self.topk_data else 0


# ================================================================
# Model setup (reusable)
# ================================================================
def setup_model():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    try:
        vllm_dist.init_distributed_environment(1, 0, "env://", 0, "nccl")
        vllm_dist.initialize_model_parallel(1, backend="nccl")
    except Exception:
        pass  # Already initialized

    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoTokenizer, AutoConfig
    from baseline_optimizations import apply_all_optimizations

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    vllm_cfg = VllmConfig(parallel_config=pcfg)

    from vllm.config import set_current_vllm_config
    with set_current_vllm_config(vllm_cfg):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=DEVICE)
        model = model.to(DEVICE)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=DEVICE).unsqueeze(0), use_cache=False)
        apply_all_optimizations(model)

    return model, tokenizer, vllm_cfg


def build_input(tokenizer, prompts, batch_size):
    all_ids = []
    for i in range(batch_size):
        text = prompts[i % len(prompts)]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=False)
        all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
    mx = max(x.shape[0] for x in all_ids)
    pad_id = tokenizer.pad_token_id or 0
    padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
              if ids.shape[0] < mx else ids for ids in all_ids]
    return torch.stack(padded, dim=0).to(DEVICE)


def run_collection(model, input_ids, gen_length=256):
    """Run generation with C11-M5-K4 config, collecting routing data."""
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from test_fused_eb_triton import fused_routing
    from test_m_skip_sweep import MSkipEBController

    collector = InsightCollector()
    ctrl = MSkipEBController(
        num_layers=19, K=8, M=4, K_target=40,
        quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)

    # Patch routing
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
                    coll.record(layer_i, idx, s_mask=sm)
                    return w.to(go.dtype), idx
                return fn
            mod.routing = mk(b, r, ng, tkg, li, ctrl, collector)
            gate_idx += 1

    # Reset
    ctrl.prev_N.clear(); ctrl.K_init.clear()
    ctrl.cold_count = 0; ctrl.hot_count = 0
    ctrl.eb_calls = 0; ctrl.eb_skips = 0
    ctrl._bufs.clear(); ctrl.k_init_history.clear()
    ctrl.s_mask_cache.clear(); ctrl.pop_cache.clear()
    ctrl._fwd_in_block.clear(); ctrl._block_idx.clear()

    decoder = ThresholdParallelDecoder(
        temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
    dllm = BlockDiffusionLLM(
        model, decoder, BlockIteratorFactory(use_block_diffusion=True),
        cache_factory=KVCacheFactory("prefix", is_bd_model=True),
        early_stop=True, maximum_unroll=4, expected_tpf=15,
        backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

    with torch.inference_mode():
        dllm.diff_iteration.num_forwards = 0
        _ = dllm.generate(input_ids.clone(), gen_length=gen_length, block_length=BLOCK_LENGTH)

    return collector, ctrl


# ================================================================
# Analysis functions
# ================================================================
def detect_blocks(collector, layer=0):
    """Detect block boundaries by N jumps."""
    boundaries = [0]
    prev_n = None
    for fi in range(collector.num_fwds()):
        if layer in collector.topk_data[fi]:
            n = collector.topk_data[fi][layer].shape[0]
            if prev_n is not None and n > prev_n:
                boundaries.append(fi)
            prev_n = n
    return boundaries


def per_layer_hot_gpu(collector):
    """Return {layer: hot_gpu_index} averaged across all forwards."""
    result = {}
    for li in range(19):
        gpu_avg = np.zeros(EP_SIZE)
        count = 0
        for fi in range(collector.num_fwds()):
            if li in collector.topk_data[fi]:
                gpu_avg += collector.gpu_load(fi, li)
                count += 1
        if count > 0:
            gpu_avg /= count
            result[li] = int(gpu_avg.argmax())
    return result


def block_start_routing(collector, layer=0):
    """Extract topk_ids of the SECOND forward in each block (first decode, not prefill)."""
    boundaries = detect_blocks(collector, layer)
    routings = []
    for bi, bs in enumerate(boundaries):
        # Second fwd in block (first is often prefill with 2N tokens)
        fi = bs + 1 if bs + 1 < collector.num_fwds() else bs
        if layer in collector.topk_data[fi]:
            routings.append(collector.topk_data[fi][layer])
    return routings


def smask_stability_within_block(collector, layer=0):
    """Measure S_mask Jaccard within each block."""
    boundaries = detect_blocks(collector, layer)
    block_jaccards = []
    for bi in range(len(boundaries)):
        bs = boundaries[bi]
        be = boundaries[bi+1] if bi+1 < len(boundaries) else collector.num_fwds()
        masks_in_block = []
        for fi in range(bs, be):
            if layer in collector.smask_data[fi]:
                m = collector.smask_data[fi][layer]
                masks_in_block.append(set(np.where(m > 0)[0]))
        if len(masks_in_block) < 2:
            continue
        jaccards = []
        for i in range(1, len(masks_in_block)):
            a, b = masks_in_block[i-1], masks_in_block[i]
            if len(a | b) > 0:
                jaccards.append(len(a & b) / len(a | b))
        if jaccards:
            block_jaccards.append(np.mean(jaccards))
    return block_jaccards


def communication_envelope(collector, layer=0):
    """Compare S_mask predicted max per-GPU load vs actual."""
    boundaries = detect_blocks(collector, layer)
    ratios = []  # actual_max / predicted_max per block
    for bi in range(len(boundaries)):
        bs = boundaries[bi]
        be = boundaries[bi+1] if bi+1 < len(boundaries) else collector.num_fwds()

        # Get S_mask at cold path (first forward of block)
        if layer not in collector.smask_data[bs]:
            continue
        smask = collector.smask_data[bs][layer]
        active = np.where(smask > 0)[0]
        # Predicted: max per-GPU active experts
        predicted_gpu_experts = np.zeros(EP_SIZE)
        for eid in active:
            g = eid // EPG
            predicted_gpu_experts[g] += 1

        # Actual: max per-GPU pairs across all forwards in block
        actual_max_per_gpu = np.zeros(EP_SIZE)
        for fi in range(bs, be):
            if layer in collector.topk_data[fi]:
                load = collector.gpu_load(fi, layer)
                actual_max_per_gpu = np.maximum(actual_max_per_gpu, load)

        # Predicted max pairs ≈ predicted_gpu_experts / total_active_experts * total_pairs
        total_active = len(active)
        if total_active > 0:
            N = collector.topk_data[bs][layer].shape[0]
            K = collector.topk_data[bs][layer].shape[1]
            total_pairs = N * K
            predicted_max_pairs = predicted_gpu_experts.max() / total_active * total_pairs
            actual_max_pairs = actual_max_per_gpu.max()
            if predicted_max_pairs > 0:
                ratios.append(actual_max_pairs / predicted_max_pairs)
    return ratios


# ================================================================
# Main
# ================================================================
def main():
    print("Setting up model...")
    model, tokenizer, vllm_cfg = setup_model()

    # Load datasets
    from test_heteval128 import PROMPTS as het128

    gsm8k = []
    with open(REPO_ROOT / "data" / "gsm8k.jsonl") as f:
        for line in f:
            gsm8k.append(json.loads(line)["question"])
    gsm8k_128 = gsm8k[:128]

    mt_bench = []
    with open(REPO_ROOT / "data" / "mt_bench.jsonl") as f:
        for line in f:
            mt_bench.append(json.loads(line)["turns"][0])

    datasets = {
        "heteval128": het128,
        "gsm8k128": gsm8k_128,
        "mt_bench80": mt_bench[:80],
    }

    from vllm.config import set_current_vllm_config

    # ========== Collect data for all conditions ==========
    collectors = {}

    with set_current_vllm_config(vllm_cfg):
        # E1 + E2 + E4 + E5: 3 datasets × batch=128
        for ds_name, prompts in datasets.items():
            bs = min(128, len(prompts))
            print(f"\n  Collecting {ds_name} batch={bs}...")
            inp = build_input(tokenizer, prompts, bs)
            t0 = time.perf_counter()
            coll, ctrl = run_collection(model, inp, gen_length=256)
            t1 = time.perf_counter()
            print(f"    {coll.num_fwds()} fwd in {t1-t0:.1f}s")
            collectors[(ds_name, bs)] = coll

        # E3: heteval128 batch=32
        print(f"\n  Collecting heteval128 batch=32...")
        inp32 = build_input(tokenizer, het128, 32)
        t0 = time.perf_counter()
        coll32, _ = run_collection(model, inp32, gen_length=256)
        t1 = time.perf_counter()
        print(f"    {coll32.num_fwds()} fwd in {t1-t0:.1f}s")
        collectors[("heteval128", 32)] = coll32

    # ========== E1: Cross-dataset hot GPU stability ==========
    print(f"\n{'='*80}")
    print("E1: CROSS-DATASET HOT GPU STABILITY")
    print(f"{'='*80}")

    hot_gpus = {}
    for (ds, bs), coll in collectors.items():
        if bs != 128 and ds != "heteval128":
            continue
        if bs == 32:
            continue
        hot_gpus[(ds, bs)] = per_layer_hot_gpu(coll)

    print(f"\n  {'Layer':<8s}", end="")
    for ds in datasets:
        print(f"  {ds:>12s}", end="")
    print(f"  {'Consistent?':>12s}")

    e1_consistent = 0
    e1_total = 0
    for li in range(19):
        print(f"  L{li+1:<6d}", end="")
        gpus = []
        for ds in datasets:
            bs = min(128, len(datasets[ds]))
            g = hot_gpus.get((ds, bs), {}).get(li, -1)
            gpus.append(g)
            print(f"  {'GPU'+str(g):>12s}", end="")
        consistent = len(set(gpus)) == 1
        e1_consistent += int(consistent)
        e1_total += 1
        print(f"  {'YES' if consistent else 'NO ★':>12s}")

    print(f"\n  Consistent: {e1_consistent}/{e1_total} layers "
          f"({e1_consistent/e1_total*100:.0f}%)")

    # ========== E2: Block-start routing determinism ==========
    print(f"\n{'='*80}")
    print("E2: BLOCK-START ROUTING DETERMINISM")
    print(f"{'='*80}")

    print(f"\n  Cross-block Jaccard (within each dataset, Layer 1):")
    for (ds, bs), coll in collectors.items():
        if bs == 32:
            continue
        routings = block_start_routing(coll, layer=0)
        if len(routings) < 2:
            continue
        jaccards = []
        for i in range(1, len(routings)):
            a_set = set(routings[i-1].flatten())
            b_set = set(routings[i].flatten())
            if len(a_set | b_set) > 0:
                jaccards.append(len(a_set & b_set) / len(a_set | b_set))
        print(f"    {ds}: {len(routings)} blocks, "
              f"adj Jaccard = {np.mean(jaccards):.4f} ± {np.std(jaccards):.4f}")

    # Cross-dataset: compare block-start routing between datasets
    print(f"\n  Cross-dataset block-start Jaccard (Layer 1, first block):")
    ds_first_routing = {}
    for (ds, bs), coll in collectors.items():
        if bs == 32:
            continue
        routings = block_start_routing(coll, layer=0)
        if routings:
            ds_first_routing[ds] = set(routings[0].flatten())

    ds_names = list(ds_first_routing.keys())
    for i in range(len(ds_names)):
        for j in range(i+1, len(ds_names)):
            a, b = ds_first_routing[ds_names[i]], ds_first_routing[ds_names[j]]
            jac = len(a & b) / len(a | b) if len(a | b) > 0 else 0
            print(f"    {ds_names[i]} vs {ds_names[j]}: Jaccard = {jac:.4f}")

    # Per-GPU load at block start across datasets
    print(f"\n  Block-start per-GPU load (Layer 1, 2nd fwd of block 1):")
    for (ds, bs), coll in collectors.items():
        if bs == 32:
            continue
        boundaries = detect_blocks(coll, 0)
        if len(boundaries) > 1:
            fi = boundaries[1] + 1  # 2nd fwd of block 1
            if fi < coll.num_fwds() and 0 in coll.topk_data[fi]:
                load = coll.gpu_load(fi, 0)
                skew = load.max() / max(load.min(), 1)
                print(f"    {ds}: {load.tolist()}, skew={skew:.2f}")

    # ========== E3: Cross-batch-size stability ==========
    print(f"\n{'='*80}")
    print("E3: CROSS-BATCH-SIZE STABILITY (heteval128)")
    print(f"{'='*80}")

    for bs in [32, 128]:
        coll = collectors.get(("heteval128", bs))
        if coll is None:
            continue
        hot = per_layer_hot_gpu(coll)
        print(f"\n  batch={bs}: hot GPUs = ", end="")
        for li in range(19):
            print(f"L{li+1}:GPU{hot.get(li,-1)} ", end="")
        print()

    # Compare hot GPUs
    h32 = per_layer_hot_gpu(collectors[("heteval128", 32)])
    h128 = per_layer_hot_gpu(collectors[("heteval128", 128)])
    match = sum(1 for li in range(19) if h32.get(li) == h128.get(li))
    print(f"\n  Hot GPU match (b32 vs b128): {match}/19 ({match/19*100:.0f}%)")

    # S_mask Jaccard comparison
    print(f"\n  S_mask within-block Jaccard (Layer 1):")
    for bs in [32, 128]:
        coll = collectors.get(("heteval128", bs))
        if coll is None:
            continue
        jacs = smask_stability_within_block(coll, layer=0)
        if jacs:
            print(f"    batch={bs}: {np.mean(jacs):.4f} ± {np.std(jacs):.4f} "
                  f"(over {len(jacs)} blocks)")

    # ========== E4: Full 19-layer temporal profile ==========
    print(f"\n{'='*80}")
    print("E4: FULL 19-LAYER TEMPORAL PROFILE (heteval128 b=128)")
    print(f"{'='*80}")

    coll = collectors[("heteval128", 128)]
    boundaries = detect_blocks(coll, 0)

    # For each layer: show block-start skew and steady-state skew
    print(f"\n  {'Layer':<8s} {'BlockStart':>11s} {'Steady':>11s} {'Converge?':>10s}")
    print(f"         {'avgSkew':>11s} {'avgSkew':>11s}")

    for li in range(19):
        block_start_skews = []
        steady_skews = []
        for bi in range(len(boundaries)):
            bs_idx = boundaries[bi]
            be_idx = boundaries[bi+1] if bi+1 < len(boundaries) else coll.num_fwds()

            for fi in range(bs_idx, be_idx):
                if li not in coll.topk_data[fi]:
                    continue
                load = coll.gpu_load(fi, li)
                skew = load.max() / max(load.min(), 1)
                pos_in_block = fi - bs_idx
                if pos_in_block <= 3:
                    block_start_skews.append(skew)
                elif pos_in_block >= 20:
                    steady_skews.append(skew)

        bs_avg = np.mean(block_start_skews) if block_start_skews else 0
        ss_avg = np.mean(steady_skews) if steady_skews else 0
        converges = bs_avg > ss_avg * 1.5 if ss_avg > 0 else False
        print(f"  L{li+1:<6d} {bs_avg:>11.2f} {ss_avg:>11.2f} {'YES' if converges else 'NO':>10s}")

    # ========== E5: Communication envelope tightness ==========
    print(f"\n{'='*80}")
    print("E5: COMMUNICATION ENVELOPE TIGHTNESS (heteval128 b=128)")
    print(f"{'='*80}")

    print(f"\n  Ratio = actual_max_gpu_pairs / predicted_from_smask")
    print(f"  (>1 = prediction too low, <1 = prediction conservative)")
    print(f"\n  {'Layer':<8s} {'AvgRatio':>10s} {'MaxRatio':>10s} {'MinRatio':>10s}")

    for li in range(19):
        ratios = communication_envelope(coll, layer=li)
        if ratios:
            print(f"  L{li+1:<6d} {np.mean(ratios):>10.3f} {np.max(ratios):>10.3f} "
                  f"{np.min(ratios):>10.3f}")

    # ========== Summary ==========
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  E1 Cross-dataset hot GPU consistency: {e1_consistent}/{e1_total}")
    print(f"  E2 Block-start routing: see Jaccard above")
    print(f"  E3 Cross-batch hot GPU match: {match}/19")
    print(f"  E4 19-layer convergence: see table above")
    print(f"  E5 Envelope tightness: see ratios above")

    # Save raw results
    results = {
        'e1_consistent': e1_consistent,
        'e1_total': e1_total,
        'e3_match': match,
        'conditions': [(ds, bs) for (ds, bs) in collectors.keys()],
    }
    out_path = REPO_ROOT / "codex_coding" / "results" / "insight_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
