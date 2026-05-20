#!/usr/bin/env python3
"""
Expert usage pattern collection for fused_experts optimization.

Hooks fused_routing output (topk_ids) per layer per forward, collects:
  - n_unique_experts: how many experts are actually used
  - expert_token_counts[256]: per-expert token count
  - top10_concentration: fraction of pairs in top-10 busiest experts
  - adj_expert_set_jaccard: Jaccard vs previous forward's active set
  - adj_token_count_correlation: Pearson corr of per-expert counts vs prev
  - adj_top1_overlap: fraction of tokens keeping same top-1 expert
  - cross_window_expert_jaccard: Jaccard across M-skip window boundaries

Config: batch=128, gen_length=256, M=5, q_major=1.0, temp=0
"""

from __future__ import annotations
import os, sys, time, socket, json, math
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

from test_fused_eb_triton import (
    fused_routing, FusedEBController,
    _kernel_A, _kernel_A_cold, _kernel_B_v3,
    _kernel_C, _kernel_D_v2, _kernel_zero_init,
)
from baseline_optimizations import apply_all_optimizations
from test_heteval128 import PROMPTS

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
GEN_LENGTH = 256
BATCH_SIZE = 128


# ================================================================
# M-skip EB Controller (same as quality check)
# ================================================================
class MSkipEBController(FusedEBController):
    def __init__(self, *args, skip_m=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_m = skip_m
        self.k_init_history = []
        self.s_mask_cache = {}
        self.pop_cache = {}
        self._fwd_in_block = {}
        self._block_idx = {}
        self.eb_calls = 0
        self.eb_skips = 0

    def cold_path(self, layer_idx, logits, bias):
        N, E = logits.shape
        b = self._get_bufs(N, E, logits.device)
        _kernel_zero_init[(1,)](b['pop'], b['sat_flag'], b['sat_count'],
                                b['G'], b['H'], E=E)
        lf = logits.float(); bf = bias.float()
        _kernel_A_cold[(N,)](lf, bf, b['pop'], b['topkm_idx'], b['topkm_w'], b['r'],
                             N, self.rsf, self.quality_floor, lf.stride(0), lf.stride(1),
                             b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                             E=E, KEXT=self.K_ext, KEXT_PAD=16, K=self.K)
        _kernel_B_v3[(1,)](b['pop'], b['s_mask'], self.K_target, E=E)
        q_major_x1000 = int(self.q_major * 1000)
        for _ in range(self.MAX_ROUNDS):
            _kernel_C[(N,)](b['topkm_idx'], b['topkm_w'], b['r'],
                           b['s_mask'], b['sat_flag'], b['sat_count'], b['G'], b['H'],
                           N, b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                           E=E, KEXT=self.K_ext, KEXT_PAD=16)
            _kernel_D_v2[(1,)](b['s_mask'], b['sat_flag'], b['sat_count'],
                              b['G'], b['H'], N, q_major_x1000, E=E, CAP=self.cap)
        actual_s = int(b['s_mask'].sum().item())
        self.K_init[layer_idx] = actual_s
        self.k_init_history.append(actual_s)
        if layer_idx not in self.s_mask_cache:
            self.s_mask_cache[layer_idx] = torch.zeros(256, device=logits.device, dtype=torch.int32)
            self.pop_cache[layer_idx] = torch.zeros(256, device=logits.device, dtype=torch.float32)
        self.s_mask_cache[layer_idx].copy_(b['s_mask'])
        bi = self._block_idx.get(layer_idx, -1) + 1
        self._block_idx[layer_idx] = bi
        self._fwd_in_block[layer_idx] = 0
        self.cold_count += 1
        return b['s_mask']

    def hot_path(self, layer_idx, logits, bias):
        N, E = logits.shape
        K_init = self.K_init.get(layer_idx, 103)
        fi = self._fwd_in_block.get(layer_idx, 0) + 1
        self._fwd_in_block[layer_idx] = fi
        if self.skip_m == float('inf') or fi % self.skip_m != 0:
            self.eb_skips += 1
            self.hot_count += 1
            return self.s_mask_cache[layer_idx]
        pop = self.pop_cache[layer_idx]
        lf = logits.float()
        _kernel_A[(N,)](lf, bias.float(), pop, N, self.rsf,
                        lf.stride(0), lf.stride(1), E=E, KEXT=self.K_ext, KEXT_PAD=16)
        _kernel_B_v3[(1,)](pop, self.s_mask_cache[layer_idx], K_init, E=E)
        self.eb_calls += 1
        self.hot_count += 1
        return self.s_mask_cache[layer_idx]

    def reset(self):
        self.prev_N.clear(); self.K_init.clear()
        self.cold_count = 0; self.hot_count = 0
        self._bufs.clear(); self.k_init_history.clear()
        self.s_mask_cache.clear(); self.pop_cache.clear()
        self._fwd_in_block.clear(); self._block_idx.clear()
        self.eb_calls = 0; self.eb_skips = 0


# ================================================================
# Recording infrastructure
# ================================================================
class ExpertUsageRecorder:
    """Records per-layer per-forward expert usage from topk_ids."""

    def __init__(self, num_experts=256):
        self.E = num_experts
        # per-layer state for adjacent comparison
        self._prev_active_set = {}   # layer_idx -> set of expert ids
        self._prev_counts = {}       # layer_idx -> np.array [E]
        self._prev_top1 = {}         # layer_idx -> np.array [N] (top-1 expert per token)
        self._prev_fwd_in_block = {} # layer_idx -> int
        # collected records
        self.records = []
        self._global_fwd = 0

    def record(self, layer_idx, topk_ids_cpu, fwd_in_block, is_cold, block_idx, s_mask_size):
        """
        topk_ids_cpu: [N, K] int numpy array
        """
        N, K = topk_ids_cpu.shape
        E = self.E

        # --- Expert usage stats ---
        flat_ids = topk_ids_cpu.flatten()
        counts = np.bincount(flat_ids, minlength=E)  # [E]
        active_set = set(np.where(counts > 0)[0].tolist())
        n_unique = len(active_set)

        # Top-10 concentration
        sorted_counts = np.sort(counts)[::-1]
        total_pairs = N * K
        top10_conc = float(sorted_counts[:10].sum()) / total_pairs if total_pairs > 0 else 0

        # Token count distribution stats
        active_counts = counts[counts > 0]
        count_mean = float(active_counts.mean()) if len(active_counts) > 0 else 0
        count_std = float(active_counts.std()) if len(active_counts) > 0 else 0
        count_max = int(active_counts.max()) if len(active_counts) > 0 else 0
        count_min = int(active_counts.min()) if len(active_counts) > 0 else 0

        # Top-1 expert per token
        top1 = topk_ids_cpu[:, 0]  # [N]

        # --- Adjacent forward comparison ---
        adj_jaccard = None
        adj_corr = None
        adj_top1_overlap = None
        cross_window_jaccard = None

        prev_set = self._prev_active_set.get(layer_idx)
        prev_counts = self._prev_counts.get(layer_idx)
        prev_top1 = self._prev_top1.get(layer_idx)
        prev_fib = self._prev_fwd_in_block.get(layer_idx)

        if prev_set is not None:
            # Jaccard of active expert sets
            inter = len(active_set & prev_set)
            union = len(active_set | prev_set)
            adj_jaccard = inter / union if union > 0 else 1.0

            # Pearson correlation of per-expert counts
            if prev_counts is not None:
                c1 = counts.astype(np.float64)
                c2 = prev_counts.astype(np.float64)
                m1, m2 = c1.mean(), c2.mean()
                d1, d2 = c1 - m1, c2 - m2
                denom = math.sqrt((d1**2).sum() * (d2**2).sum())
                adj_corr = float((d1 * d2).sum() / denom) if denom > 0 else 0.0

            # Top-1 overlap
            if prev_top1 is not None and len(prev_top1) == len(top1):
                adj_top1_overlap = float((top1 == prev_top1).sum()) / len(top1)

            # Cross-window: is_cold means S_mask just changed
            if is_cold and prev_fib is not None and prev_fib > 0:
                cross_window_jaccard = adj_jaccard

        # Update state
        self._prev_active_set[layer_idx] = active_set
        self._prev_counts[layer_idx] = counts.copy()
        self._prev_top1[layer_idx] = top1.copy()
        self._prev_fwd_in_block[layer_idx] = fwd_in_block

        self.records.append({
            'layer': layer_idx,
            'global_fwd': self._global_fwd,
            'block_idx': block_idx,
            'fwd_in_block': fwd_in_block,
            'is_cold': is_cold,
            's_mask_size': s_mask_size,
            'n_unique': n_unique,
            'top10_concentration': round(top10_conc, 4),
            'count_mean': round(count_mean, 2),
            'count_std': round(count_std, 2),
            'count_max': count_max,
            'count_min': count_min,
            'adj_expert_jaccard': round(adj_jaccard, 4) if adj_jaccard is not None else None,
            'adj_count_corr': round(adj_corr, 4) if adj_corr is not None else None,
            'adj_top1_overlap': round(adj_top1_overlap, 4) if adj_top1_overlap is not None else None,
            'cross_window_jaccard': round(cross_window_jaccard, 4) if cross_window_jaccard is not None else None,
        })

    def inc_fwd(self):
        self._global_fwd += 1


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoTokenizer, AutoConfig

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print(f"Expert Usage Pattern Collection")
    print(f"  batch={BATCH_SIZE}, gen_length={GEN_LENGTH}, M=5, q_major=1.0")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)

        print("\nApplying baseline optimizations...")
        n_rms, n_fa = apply_all_optimizations(model)
        print(f"  Fused RMSNorm: {n_rms}, Flash-attn: {n_fa}")

        # Build input
        all_ids = []
        for i in range(BATCH_SIZE):
            text = PROMPTS[i]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                  if ids.shape[0] < mx else ids for ids in all_ids]
        input_ids = torch.stack(padded, dim=0).to(device)
        print(f"  Input shape: {input_ids.shape}")

        decoder = ThresholdParallelDecoder(temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Create EB controller and recorder
        ctrl = MSkipEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)
        recorder = ExpertUsageRecorder(num_experts=256)

        # Patch routing to record topk_ids
        orig_routings = {}
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                orig_routings[name] = mod.routing

        # We need to track block_idx and fwd_in_block per layer from the controller
        layer_idx_counter = [0]  # mutable counter for closure

        def patch_recording():
            idx = 0
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate":
                    b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                         mod.top_k, mod.n_group, mod.topk_group)
                    li = idx
                    def mk(bb, rr, tt, nn, gg, layer_i, cc, rec):
                        def fn(hs, go, topk, renorm):
                            N = go.shape[0]
                            # Peek cold/hot BEFORE get_s_mask (without modifying state)
                            prev = cc.prev_N.get(layer_i, -1)
                            is_cold = (prev == -1) or (N > prev)

                            sm = cc.get_s_mask(layer_i, go, bb)
                            w, ids = fused_routing(go, bb, rr, s_mask=sm, K=tt, ng=nn, tkg=gg)

                            # Record after routing
                            s_mask_size = int(sm.sum().item())

                            ids_cpu = ids.cpu().numpy()
                            rec.record(layer_i, ids_cpu,
                                       fwd_in_block=cc._fwd_in_block.get(layer_i, 0),
                                       is_cold=is_cold,
                                       block_idx=cc._block_idx.get(layer_i, 0),
                                       s_mask_size=s_mask_size)

                            # Track global fwd (only once per forward, at layer 1)
                            if layer_i == 1:
                                rec.inc_fwd()

                            return w.to(go.dtype), ids
                        return fn
                    mod.routing = mk(b, r, tk, ng, tkg, li, ctrl, recorder)
                    idx += 1

        patch_recording()

        # Warmup (don't record)
        print("\nWarmup...")
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd")

        # Reset everything for actual collection
        ctrl.reset()
        recorder = ExpertUsageRecorder(num_experts=256)
        # Re-patch with fresh recorder
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate" and name in orig_routings:
                mod.routing = orig_routings[name]
        patch_recording()

        # Collection run
        print("\nCollecting expert usage data...")
        dllm = make_dllm()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        total_fwd = dllm.diff_iteration.num_forwards
        print(f"  Done: {total_fwd} fwd, {t1-t0:.3f}s")
        print(f"  Records: {len(recorder.records)}")

        # ============================================================
        # Aggregate analysis
        # ============================================================
        recs = recorder.records
        print(f"\n{'='*80}")
        print(f"EXPERT USAGE ANALYSIS (batch={BATCH_SIZE}, M=5)")
        print(f"{'='*80}")

        # --- Overall stats ---
        all_unique = [r['n_unique'] for r in recs]
        all_smask = [r['s_mask_size'] for r in recs if r['s_mask_size'] > 0]
        print(f"\n  Overall ({len(recs)} records, {total_fwd} fwd × 19 layers):")
        print(f"    n_unique_experts:  avg={np.mean(all_unique):.1f}, "
              f"min={np.min(all_unique)}, max={np.max(all_unique)}")
        print(f"    |S_mask|:          avg={np.mean(all_smask):.1f}, "
              f"min={np.min(all_smask)}, max={np.max(all_smask)}")
        print(f"    unique/|S_mask|:   avg={np.mean(all_unique)/np.mean(all_smask)*100:.1f}%")

        all_top10 = [r['top10_concentration'] for r in recs]
        print(f"    top10_concentration: avg={np.mean(all_top10):.4f}")

        all_cmean = [r['count_mean'] for r in recs]
        all_cstd = [r['count_std'] for r in recs]
        all_cmax = [r['count_max'] for r in recs]
        print(f"    count per expert:  avg={np.mean(all_cmean):.1f}, "
              f"std={np.mean(all_cstd):.1f}, max={np.mean(all_cmax):.0f}")

        # --- Adjacent forward stats ---
        adj_j = [r['adj_expert_jaccard'] for r in recs if r['adj_expert_jaccard'] is not None]
        adj_c = [r['adj_count_corr'] for r in recs if r['adj_count_corr'] is not None]
        adj_t = [r['adj_top1_overlap'] for r in recs if r['adj_top1_overlap'] is not None]
        cw_j = [r['cross_window_jaccard'] for r in recs if r['cross_window_jaccard'] is not None]

        print(f"\n  Adjacent forward comparison:")
        if adj_j:
            print(f"    adj_expert_jaccard:    avg={np.mean(adj_j):.4f}, "
                  f"min={np.min(adj_j):.4f}, max={np.max(adj_j):.4f}")
        if adj_c:
            print(f"    adj_count_correlation: avg={np.mean(adj_c):.4f}, "
                  f"min={np.min(adj_c):.4f}, max={np.max(adj_c):.4f}")
        if adj_t:
            print(f"    adj_top1_overlap:      avg={np.mean(adj_t):.4f}, "
                  f"min={np.min(adj_t):.4f}, max={np.max(adj_t):.4f}")
        if cw_j:
            print(f"    cross_window_jaccard:  avg={np.mean(cw_j):.4f}, "
                  f"min={np.min(cw_j):.4f}, max={np.max(cw_j):.4f} "
                  f"({len(cw_j)} transitions)")

        # --- By fwd_in_block ---
        print(f"\n  By fwd_in_block (M=5 windows):")
        print(f"    {'fib':>4s} {'n_unique':>10s} {'adj_J':>8s} {'adj_corr':>10s} "
              f"{'top1_ovlp':>10s} {'top10_conc':>10s} {'count':>6s}")
        print(f"    {'-'*60}")
        fib_groups = defaultdict(list)
        for r in recs:
            fib_groups[r['fwd_in_block']].append(r)
        for fib in sorted(fib_groups.keys())[:35]:
            grp = fib_groups[fib]
            nu = np.mean([r['n_unique'] for r in grp])
            aj = [r['adj_expert_jaccard'] for r in grp if r['adj_expert_jaccard'] is not None]
            ac = [r['adj_count_corr'] for r in grp if r['adj_count_corr'] is not None]
            at = [r['adj_top1_overlap'] for r in grp if r['adj_top1_overlap'] is not None]
            t10 = np.mean([r['top10_concentration'] for r in grp])
            aj_s = f"{np.mean(aj):.4f}" if aj else "—"
            ac_s = f"{np.mean(ac):.4f}" if ac else "—"
            at_s = f"{np.mean(at):.4f}" if at else "—"
            print(f"    {fib:>4d} {nu:>10.1f} {aj_s:>8s} {ac_s:>10s} "
                  f"{at_s:>10s} {t10:>10.4f} {len(grp):>6d}")

        # --- By layer ---
        print(f"\n  By layer:")
        print(f"    {'layer':>5s} {'n_unique':>10s} {'adj_J':>8s} {'adj_corr':>10s} "
              f"{'top1_ovlp':>10s} {'top10_conc':>10s}")
        print(f"    {'-'*55}")
        layer_groups = defaultdict(list)
        for r in recs:
            layer_groups[r['layer']].append(r)
        for li in sorted(layer_groups.keys()):
            grp = layer_groups[li]
            nu = np.mean([r['n_unique'] for r in grp])
            aj = [r['adj_expert_jaccard'] for r in grp if r['adj_expert_jaccard'] is not None]
            ac = [r['adj_count_corr'] for r in grp if r['adj_count_corr'] is not None]
            at = [r['adj_top1_overlap'] for r in grp if r['adj_top1_overlap'] is not None]
            t10 = np.mean([r['top10_concentration'] for r in grp])
            aj_s = f"{np.mean(aj):.4f}" if aj else "—"
            ac_s = f"{np.mean(ac):.4f}" if ac else "—"
            at_s = f"{np.mean(at):.4f}" if at else "—"
            print(f"    L{li:>3d} {nu:>10.1f} {aj_s:>8s} {ac_s:>10s} "
                  f"{at_s:>10s} {t10:>10.4f}")

        # Save raw data (records without full counts to keep size manageable)
        out_path = REPO_ROOT / "codex_coding" / "results" / "expert_usage_patterns_b128.json"
        save_data = {
            'config': {'batch': BATCH_SIZE, 'gen_length': GEN_LENGTH, 'block': BLOCK_LENGTH,
                       'M': 5, 'q_major': 1.0, 'total_fwd': total_fwd},
            'summary': {
                'n_unique_avg': float(np.mean(all_unique)),
                'n_unique_min': int(np.min(all_unique)),
                'n_unique_max': int(np.max(all_unique)),
                's_mask_avg': float(np.mean(all_smask)),
                'top10_conc_avg': float(np.mean(all_top10)),
                'adj_expert_jaccard_avg': float(np.mean(adj_j)) if adj_j else None,
                'adj_count_corr_avg': float(np.mean(adj_c)) if adj_c else None,
                'adj_top1_overlap_avg': float(np.mean(adj_t)) if adj_t else None,
                'cross_window_jaccard_avg': float(np.mean(cw_j)) if cw_j else None,
            },
            'records': recs,
        }
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\n  Saved to {out_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
