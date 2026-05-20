#!/usr/bin/env python3
"""
D1 Token Skip → Expert Elimination Analysis

Collects token-level confidence (from LM head output) + per-layer expert assignments,
then analyzes: if we skip high-confidence tokens, how many experts become unused?

Uses previous forward's confidence as the predictor (realistic mechanism).
All analysis done online per forward — no large arrays stored.
"""

from __future__ import annotations
import os, sys, time, socket, json
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
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]


class MSkipEBController(FusedEBController):
    def __init__(self, *args, skip_m=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_m = skip_m
        self.k_init_history = []; self.s_mask_cache = {}; self.pop_cache = {}
        self._fwd_in_block = {}; self._block_idx = {}
        self.eb_calls = 0; self.eb_skips = 0

    def cold_path(self, layer_idx, logits, bias):
        N, E = logits.shape; b = self._get_bufs(N, E, logits.device)
        _kernel_zero_init[(1,)](b['pop'], b['sat_flag'], b['sat_count'], b['G'], b['H'], E=E)
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
        self.K_init[layer_idx] = actual_s; self.k_init_history.append(actual_s)
        if layer_idx not in self.s_mask_cache:
            self.s_mask_cache[layer_idx] = torch.zeros(256, device=logits.device, dtype=torch.int32)
            self.pop_cache[layer_idx] = torch.zeros(256, device=logits.device, dtype=torch.float32)
        self.s_mask_cache[layer_idx].copy_(b['s_mask'])
        bi = self._block_idx.get(layer_idx, -1) + 1
        self._block_idx[layer_idx] = bi; self._fwd_in_block[layer_idx] = 0
        self.cold_count += 1; return b['s_mask']

    def hot_path(self, layer_idx, logits, bias):
        N, E = logits.shape; K_init = self.K_init.get(layer_idx, 103)
        fi = self._fwd_in_block.get(layer_idx, 0) + 1; self._fwd_in_block[layer_idx] = fi
        if self.skip_m == float('inf') or fi % self.skip_m != 0:
            self.eb_skips += 1; self.hot_count += 1; return self.s_mask_cache[layer_idx]
        pop = self.pop_cache[layer_idx]; lf = logits.float()
        _kernel_A[(N,)](lf, bias.float(), pop, N, self.rsf,
                        lf.stride(0), lf.stride(1), E=E, KEXT=self.K_ext, KEXT_PAD=16)
        _kernel_B_v3[(1,)](pop, self.s_mask_cache[layer_idx], K_init, E=E)
        self.eb_calls += 1; self.hot_count += 1; return self.s_mask_cache[layer_idx]

    def reset(self):
        self.prev_N.clear(); self.K_init.clear(); self.cold_count = 0; self.hot_count = 0
        self._bufs.clear(); self.k_init_history.clear()
        self.s_mask_cache.clear(); self.pop_cache.clear()
        self._fwd_in_block.clear(); self._block_idx.clear()
        self.eb_calls = 0; self.eb_skips = 0


class D1ExpertAnalyzer:
    """Online analysis: token skip → expert elimination."""

    def __init__(self, thresholds, num_experts=256):
        self.thresholds = thresholds
        self.E = num_experts
        self.prev_confidence = None  # [N] from previous forward
        self.records = []            # aggregated per-forward records
        self.correlation_data = []   # stable_fraction per expert hotness band
        self._global_fwd = 0

    def analyze_forward(self, all_layer_ids, confidence, fwd_in_block, block_idx, is_cold, s_mask_sizes):
        """
        all_layer_ids: dict layer_idx -> [N, 8] numpy (topk_ids)
        confidence: [N] numpy (max softmax prob per token)
        """
        N = confidence.shape[0]

        if self.prev_confidence is not None and self.prev_confidence.shape[0] == N:
            prev_conf = self.prev_confidence
        else:
            prev_conf = None

        for th in self.thresholds:
            if prev_conf is not None:
                stable_mask = prev_conf > th  # [N] bool
            else:
                stable_mask = np.zeros(N, dtype=bool)  # first forward: nothing to skip

            skip_rate = float(stable_mask.mean())
            n_skip = int(stable_mask.sum())

            for layer_idx, ids in all_layer_ids.items():
                # Before skip
                flat_before = ids.flatten()
                counts_before = np.bincount(flat_before, minlength=self.E)
                unique_before = int((counts_before > 0).sum())

                # After skip: remove stable tokens
                if n_skip > 0 and n_skip < N:
                    active_ids = ids[~stable_mask].flatten()
                    counts_after = np.bincount(active_ids, minlength=self.E)
                elif n_skip == 0:
                    counts_after = counts_before.copy()
                else:  # all skipped
                    counts_after = np.zeros(self.E, dtype=int)

                unique_after = int((counts_after > 0).sum())
                eliminated = unique_before - unique_after

                self.records.append({
                    'global_fwd': self._global_fwd,
                    'block_idx': block_idx,
                    'fwd_in_block': fwd_in_block,
                    'is_cold': is_cold,
                    'layer': layer_idx,
                    'threshold': th,
                    'N': N,
                    'skip_rate': round(skip_rate, 4),
                    'n_skip': n_skip,
                    'unique_before': unique_before,
                    'unique_after': unique_after,
                    'eliminated': eliminated,
                    's_mask_size': s_mask_sizes.get(layer_idx, 0),
                })

        # Correlation analysis: for threshold=0.90, check stable fraction by expert hotness
        if prev_conf is not None:
            stable_090 = prev_conf > 0.90
            if stable_090.any():
                for layer_idx, ids in all_layer_ids.items():
                    flat = ids.flatten()
                    counts = np.bincount(flat, minlength=self.E)

                    # For each expert, what fraction of its tokens are stable?
                    for expert_id in range(self.E):
                        if counts[expert_id] == 0:
                            continue
                        # Tokens using this expert (any of K=8 slots)
                        token_uses_expert = (ids == expert_id).any(axis=1)  # [N]
                        n_total = int(token_uses_expert.sum())
                        n_stable = int((token_uses_expert & stable_090).sum())
                        if n_total > 0:
                            self.correlation_data.append({
                                'global_fwd': self._global_fwd,
                                'layer': layer_idx,
                                'expert_id': expert_id,
                                'token_count': n_total,
                                'stable_count': n_stable,
                                'stable_frac': round(n_stable / n_total, 4),
                            })

        self.prev_confidence = confidence.copy()
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
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print("D1 Token Skip → Expert Elimination Analysis")
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

        all_ids_list = []
        for i in range(BATCH_SIZE):
            text = PROMPTS[i]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
            all_ids_list.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids_list)
        pad_id = tokenizer.pad_token_id or 0
        padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                  if ids.shape[0] < mx else ids for ids in all_ids_list]
        input_ids = torch.stack(padded, dim=0).to(device)
        print(f"  Input shape: {input_ids.shape}")

        decoder = ThresholdParallelDecoder(temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Setup EB controller and analyzer
        ctrl = MSkipEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)
        analyzer = D1ExpertAnalyzer(thresholds=THRESHOLDS, num_experts=256)

        # Storage for per-forward per-layer topk_ids
        fwd_layer_ids = {}      # layer_idx -> [N, 8] numpy
        fwd_s_mask_sizes = {}   # layer_idx -> int
        fwd_metadata = {}       # block_idx, fwd_in_block, is_cold

        orig_routings = {}
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                orig_routings[name] = mod.routing

        def patch_recording():
            fwd_layer_ids.clear(); fwd_s_mask_sizes.clear(); fwd_metadata.clear()
            idx = 0
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate":
                    b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                         mod.top_k, mod.n_group, mod.topk_group)
                    li = idx
                    def mk(bb, rr, tt, nn, gg, layer_i, cc):
                        def fn(hs, go, topk, renorm):
                            N = go.shape[0]
                            prev = cc.prev_N.get(layer_i, -1)
                            is_cold = (prev == -1) or (N > prev)
                            sm = cc.get_s_mask(layer_i, go, bb)
                            w, ids = fused_routing(go, bb, rr, s_mask=sm, K=tt, ng=nn, tkg=gg)
                            # Store for analysis after forward completes
                            fwd_layer_ids[layer_i] = ids.cpu().numpy()
                            fwd_s_mask_sizes[layer_i] = int(sm.sum().item())
                            if layer_i == 0:
                                fwd_metadata['fwd_in_block'] = cc._fwd_in_block.get(layer_i, 0)
                                fwd_metadata['block_idx'] = cc._block_idx.get(layer_i, 0)
                                fwd_metadata['is_cold'] = is_cold
                            return w.to(go.dtype), ids
                        return fn
                    mod.routing = mk(b, r, tk, ng, tkg, li, ctrl)
                    idx += 1

        # Hook LM head to capture logits for confidence
        lm_head_output = {}

        def lm_head_hook(mod, inp, out):
            lm_head_output['logits'] = out.detach()

        lm_head = model.lm_head
        hook_handle = lm_head.register_forward_hook(lm_head_hook)

        # We need to intercept each forward pass to analyze after it completes.
        # Hook into the block diffusion iteration's forward method.
        from dinfer.decoding.generate_uniform import BlockDiffusionIteration

        orig_iter_fwd = BlockDiffusionIteration.forward

        def patched_iter_fwd(self_iter, *args, **kwargs):
            fwd_layer_ids.clear(); fwd_s_mask_sizes.clear(); fwd_metadata.clear()
            result = orig_iter_fwd(self_iter, *args, **kwargs)

            # After forward: compute confidence from LM head output
            if 'logits' in lm_head_output and fwd_layer_ids:
                logits = lm_head_output['logits']  # [batch, seq_len, vocab]
                # Flatten to match MoE's N = batch * seq_len
                logits_flat = logits.view(-1, logits.shape[-1])  # [N, vocab]
                probs = torch.softmax(logits_flat.float(), dim=-1)
                confidence = probs.max(dim=-1).values.cpu().numpy()  # [N]

                fib = fwd_metadata.get('fwd_in_block', 0)
                bi = fwd_metadata.get('block_idx', 0)
                ic = fwd_metadata.get('is_cold', False)

                analyzer.analyze_forward(
                    fwd_layer_ids, confidence, fib, bi, ic, fwd_s_mask_sizes)

            return result

        BlockDiffusionIteration.forward = patched_iter_fwd

        patch_recording()

        # Warmup
        print("\nWarmup...")
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd")

        # Reset for collection
        ctrl.reset()
        analyzer = D1ExpertAnalyzer(thresholds=THRESHOLDS, num_experts=256)

        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate" and name in orig_routings:
                mod.routing = orig_routings[name]
        patch_recording()

        # Collection run
        print("\nCollecting data...")
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
        print(f"  Records: {len(analyzer.records)}, Correlation entries: {len(analyzer.correlation_data)}")

        # Restore
        BlockDiffusionIteration.forward = orig_iter_fwd
        hook_handle.remove()

        # ============================================================
        # Analysis
        # ============================================================
        recs = analyzer.records
        corr = analyzer.correlation_data

        print(f"\n{'='*80}")
        print("D1 TOKEN SKIP → EXPERT ELIMINATION")
        print(f"{'='*80}")

        # --- Main table: threshold vs expert elimination ---
        print(f"\n  {'Threshold':>10s} {'SkipRate':>10s} {'Unique(B)':>10s} {'Unique(A)':>10s} "
              f"{'Eliminated':>11s} {'HBM_saved':>10s}")
        print(f"  {'-'*63}")

        for th in THRESHOLDS:
            th_recs = [r for r in recs if r['threshold'] == th and r['n_skip'] > 0]
            if not th_recs:
                print(f"  {th:>10.2f}     (no data with prev_confidence)")
                continue
            avg_skip = np.mean([r['skip_rate'] for r in th_recs])
            avg_ub = np.mean([r['unique_before'] for r in th_recs])
            avg_ua = np.mean([r['unique_after'] for r in th_recs])
            avg_elim = np.mean([r['eliminated'] for r in th_recs])
            hbm_pct = avg_elim / avg_ub * 100 if avg_ub > 0 else 0
            print(f"  {th:>10.2f} {avg_skip:>9.1%} {avg_ub:>10.1f} {avg_ua:>10.1f} "
                  f"{avg_elim:>10.1f} {hbm_pct:>9.1f}%")

        # --- By fwd_in_block for threshold=0.90 ---
        print(f"\n  By fwd_in_block (threshold=0.90):")
        print(f"    {'fib':>4s} {'SkipRate':>10s} {'Eliminated':>11s} {'UniqueA':>10s} {'count':>6s}")
        print(f"    {'-'*45}")
        fib_recs = [r for r in recs if r['threshold'] == 0.90 and r['n_skip'] > 0]
        fib_groups = defaultdict(list)
        for r in fib_recs:
            fib_groups[r['fwd_in_block']].append(r)
        for fib in sorted(fib_groups.keys())[:32]:
            grp = fib_groups[fib]
            sr = np.mean([r['skip_rate'] for r in grp])
            el = np.mean([r['eliminated'] for r in grp])
            ua = np.mean([r['unique_after'] for r in grp])
            print(f"    {fib:>4d} {sr:>9.1%} {el:>10.1f} {ua:>10.1f} {len(grp):>6d}")

        # --- By layer for threshold=0.90 ---
        print(f"\n  By layer (threshold=0.90):")
        print(f"    {'layer':>5s} {'SkipRate':>10s} {'Eliminated':>11s} {'UniqueA':>10s}")
        print(f"    {'-'*40}")
        layer_recs = [r for r in recs if r['threshold'] == 0.90 and r['n_skip'] > 0]
        layer_groups = defaultdict(list)
        for r in layer_recs:
            layer_groups[r['layer']].append(r)
        for li in sorted(layer_groups.keys()):
            grp = layer_groups[li]
            sr = np.mean([r['skip_rate'] for r in grp])
            el = np.mean([r['eliminated'] for r in grp])
            ua = np.mean([r['unique_after'] for r in grp])
            print(f"    L{li:>3d} {sr:>9.1%} {el:>10.1f} {ua:>10.1f}")

        # --- Correlation: stable token ↔ expert hotness ---
        if corr:
            print(f"\n  Stable Token ↔ Expert Hotness (threshold=0.90):")
            # Bin experts by token count
            bins = [(0, 5, "<5"), (5, 20, "5-20"), (20, 50, "20-50"),
                    (50, 200, "50-200"), (200, 10000, ">200")]
            print(f"    {'Band':>10s} {'#experts':>10s} {'avg_stable_frac':>16s} {'avg_count':>10s}")
            print(f"    {'-'*50}")
            for lo, hi, label in bins:
                band = [c for c in corr if lo <= c['token_count'] < hi]
                if band:
                    avg_sf = np.mean([c['stable_frac'] for c in band])
                    avg_ct = np.mean([c['token_count'] for c in band])
                    print(f"    {label:>10s} {len(band):>10d} {avg_sf:>15.4f} {avg_ct:>10.1f}")

            # Cold experts (<10 tokens): what fraction of their tokens are stable?
            cold = [c for c in corr if c['token_count'] < 10]
            hot = [c for c in corr if c['token_count'] > 100]
            if cold and hot:
                cold_sf = np.mean([c['stable_frac'] for c in cold])
                hot_sf = np.mean([c['stable_frac'] for c in hot])
                print(f"\n    Cold experts (<10 tokens): avg stable_fraction = {cold_sf:.4f}")
                print(f"    Hot experts (>100 tokens):  avg stable_fraction = {hot_sf:.4f}")
                if cold_sf > hot_sf:
                    print(f"    → Stable tokens PREFER cold experts (good for elimination)")
                elif cold_sf < hot_sf:
                    print(f"    → Stable tokens PREFER hot experts (bad for elimination)")
                else:
                    print(f"    → No preference")

        # Save
        save_data = {
            'config': {'batch': BATCH_SIZE, 'gen_length': GEN_LENGTH, 'M': 5,
                       'q_major': 1.0, 'total_fwd': total_fwd, 'thresholds': THRESHOLDS},
            'summary': {},
            'records': recs,
        }
        # Compact summary
        for th in THRESHOLDS:
            th_recs = [r for r in recs if r['threshold'] == th and r['n_skip'] > 0]
            if th_recs:
                save_data['summary'][str(th)] = {
                    'avg_skip_rate': float(np.mean([r['skip_rate'] for r in th_recs])),
                    'avg_eliminated': float(np.mean([r['eliminated'] for r in th_recs])),
                    'avg_unique_before': float(np.mean([r['unique_before'] for r in th_recs])),
                    'avg_unique_after': float(np.mean([r['unique_after'] for r in th_recs])),
                }

        out_path = REPO_ROOT / "codex_coding" / "results" / "d1_expert_elimination_analysis.json"
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\n  Saved to {out_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
