#!/usr/bin/env python3
"""
Dump per-layer per-cold-start |S_mask| and actual rounds needed.
Quick diagnostic: how many rounds does each cold start actually need?
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path

import torch

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

from test_fused_eb_triton import (
    fused_routing, FusedEBController,
    _kernel_A_cold, _kernel_B_v3,
    _kernel_C, _kernel_D_v2, _kernel_zero_init,
)
from baseline_optimizations import apply_all_optimizations
from test_heteval128 import PROMPTS

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
GEN_LENGTH = 256
BATCH_SIZE = 128


class DiagnosticEBController(FusedEBController):
    """Records per-cold-start: layer, |S| after K_B, |S| after each round, actual rounds."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cold_records = []
        self.s_mask_cache = {}

    def cold_path(self, layer_idx, logits, bias):
        N, E = logits.shape
        b = self._get_bufs(N, E, logits.device)

        _kernel_zero_init[(1,)](b['pop'], b['sat_flag'], b['sat_count'],
                                b['G'], b['H'], E=E)
        lf = logits.float(); bf = bias.float()
        _kernel_A_cold[(N,)](lf, bf, b['pop'], b['topkm_idx'], b['topkm_w'], b['r'],
                             N, self.rsf, self.quality_floor,
                             lf.stride(0), lf.stride(1),
                             b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                             E=E, KEXT=self.K_ext, KEXT_PAD=16, K=self.K)
        _kernel_B_v3[(1,)](b['pop'], b['s_mask'], self.K_target, E=E)

        s_after_kb = int(b['s_mask'].sum().item())

        q_major_x1000 = int(self.q_major * 1000)
        round_sizes = []
        actual_rounds = 0
        for r in range(self.MAX_ROUNDS):
            _kernel_C[(N,)](b['topkm_idx'], b['topkm_w'], b['r'],
                           b['s_mask'], b['sat_flag'], b['sat_count'], b['G'], b['H'],
                           N, b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                           E=E, KEXT=self.K_ext, KEXT_PAD=16)
            _kernel_D_v2[(1,)](b['s_mask'], b['sat_flag'], b['sat_count'],
                              b['G'], b['H'], N, q_major_x1000, E=E, CAP=self.cap)
            s_now = int(b['s_mask'].sum().item())
            sat = int(b['sat_flag'].item())
            round_sizes.append(s_now)
            if sat == 1 and actual_rounds == 0:
                actual_rounds = r + 1

        if actual_rounds == 0:
            actual_rounds = self.MAX_ROUNDS  # never satisfied within MAX_ROUNDS

        final_s = int(b['s_mask'].sum().item())

        self.cold_records.append({
            'layer': layer_idx,
            'cold_idx': self.cold_count,
            'N': N,
            's_after_kb': s_after_kb,
            'final_s': final_s,
            'actual_rounds': actual_rounds,
            'round_sizes': round_sizes,
        })

        self.K_init[layer_idx] = final_s
        if layer_idx not in self.s_mask_cache:
            self.s_mask_cache[layer_idx] = torch.zeros(E, device=logits.device, dtype=torch.int32)
        self.s_mask_cache[layer_idx].copy_(b['s_mask'])
        self.cold_count += 1
        return b['s_mask']

    def hot_path(self, layer_idx, logits, bias):
        self.hot_count += 1
        return self.s_mask_cache[layer_idx]


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

    print("S_mask Diagnostic: per-layer per-cold-start")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)
        apply_all_optimizations(model)

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

        ctrl = DiagnosticEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8)

        # Patch routing
        idx = 0
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                     mod.top_k, mod.n_group, mod.topk_group)
                li = idx
                def mk(bb, rr, tt, nn, gg, layer_i, cc):
                    def fn(hs, go, topk, renorm):
                        sm = cc.get_s_mask(layer_i, go, bb)
                        w, i = fused_routing(go, bb, rr, s_mask=sm, K=tt, ng=nn, tkg=gg)
                        return w.to(go.dtype), i
                    return fn
                mod.routing = mk(b, r, tk, ng, tkg, li, ctrl)
                idx += 1

        decoder = ThresholdParallelDecoder(temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm = BlockDiffusionLLM(
            model, decoder, BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=4, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Warmup
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        # Reset and collect
        ctrl.prev_N.clear(); ctrl.K_init.clear()
        ctrl.cold_count = 0; ctrl.hot_count = 0
        ctrl._bufs.clear(); ctrl.s_mask_cache.clear()
        ctrl.cold_records.clear()

        dllm = BlockDiffusionLLM(
            model, decoder, BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=4, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        total_fwd = dllm.diff_iteration.num_forwards

        recs = ctrl.cold_records
        print(f"\nTotal: {total_fwd} fwd, {len(recs)} cold starts")

        # By layer: avg |S|, avg rounds, min/max
        print(f"\n  Per-Layer Summary:")
        print(f"    {'Layer':>5s} {'|S|_avg':>8s} {'|S|_min':>8s} {'|S|_max':>8s} "
              f"{'Rounds_avg':>11s} {'Rounds_max':>11s} {'S_after_KB':>11s} {'Count':>6s}")
        print(f"    {'-'*72}")
        import numpy as np
        from collections import defaultdict
        by_layer = defaultdict(list)
        for r in recs:
            by_layer[r['layer']].append(r)
        for li in sorted(by_layer.keys()):
            grp = by_layer[li]
            ss = [r['final_s'] for r in grp]
            rds = [r['actual_rounds'] for r in grp]
            kb = [r['s_after_kb'] for r in grp]
            print(f"    L{li:>3d} {np.mean(ss):>8.1f} {np.min(ss):>8d} {np.max(ss):>8d} "
                  f"{np.mean(rds):>11.1f} {np.max(rds):>11d} {np.mean(kb):>11.1f} {len(grp):>6d}")

        # Show a few example cold starts with round-by-round |S|
        print(f"\n  Example cold starts (round-by-round |S|):")
        for r in recs[:6]:  # first 6
            sizes = r['round_sizes']
            sizes_str = " ".join(f"{s:>3d}" for s in sizes[:r['actual_rounds']+2])
            if r['actual_rounds'] < len(sizes):
                sizes_str += " [sat]"
            print(f"    L{r['layer']:>2d} cold#{r['cold_idx']:>3d} N={r['N']:>5d}: "
                  f"KB={r['s_after_kb']:>3d} → {sizes_str} final={r['final_s']}")

        # Distribution of actual rounds
        all_rounds = [r['actual_rounds'] for r in recs]
        print(f"\n  Actual rounds distribution:")
        from collections import Counter
        cnt = Counter(all_rounds)
        for rd in sorted(cnt.keys()):
            bar = "█" * (cnt[rd] // 2)
            print(f"    Round {rd:>2d}: {cnt[rd]:>4d} ({cnt[rd]/len(recs)*100:>5.1f}%) {bar}")

        print(f"\n  Overall: avg_rounds={np.mean(all_rounds):.1f}, "
              f"max={np.max(all_rounds)}, never_satisfied={sum(1 for r in all_rounds if r == ctrl.MAX_ROUNDS)}")


if __name__ == "__main__":
    main()
