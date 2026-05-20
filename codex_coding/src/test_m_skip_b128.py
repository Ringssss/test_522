#!/usr/bin/env python3
"""
v0.1.15.8l — M-skip sweep at batch=128

M-sweep (1,2,3,5,10,inf) with timing + quality, HetEval-128 (batch=128, block=32).
q_major=1.0.
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path
from collections import OrderedDict

import torch
import triton
import triton.language as tl

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

# Reuse HetEval-128 prompts
from test_heteval128 import PROMPTS, VERIFIABLE

from test_fused_eb_triton import (
    fused_routing, FusedEBController,
    _kernel_A, _kernel_A_cold, _kernel_B_v2, _kernel_B_v3,
    _kernel_C, _kernel_D_v2, _kernel_zero_init,
)
from baseline_optimizations import apply_all_optimizations

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
GEN_LENGTH = 256
BATCH_SIZE = 128


class MSkipEBController128(FusedEBController):
    def __init__(self, *args, skip_m=1, **kwargs):
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
    print(f"M-skip sweep at batch={BATCH_SIZE}, block={BLOCK_LENGTH}, q_major=1.0")
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
        prompt_len = input_ids.shape[1]
        print(f"  Input shape: {input_ids.shape}")

        decoder_t0 = ThresholdParallelDecoder(temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        decoder_t7 = ThresholdParallelDecoder(temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm(decoder):
            return BlockDiffusionLLM(
                model, decoder, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        orig_routings = {}
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                orig_routings[name] = mod.routing

        def restore():
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate" and name in orig_routings:
                    mod.routing = orig_routings[name]

        def patch_c5():
            restore()
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate":
                    b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                         mod.top_k, mod.n_group, mod.topk_group)
                    def mk(bb, rr, tt, nn, gg):
                        def fn(hs, go, topk, renorm):
                            w, i = fused_routing(go, bb, rr, s_mask=None, K=tt, ng=nn, tkg=gg)
                            return w.to(go.dtype), i
                        return fn
                    mod.routing = mk(b, r, tk, ng, tkg)

        def patch_eb(ctrl):
            restore()
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

        def reset_ctrl(ctrl):
            ctrl.prev_N.clear(); ctrl.K_init.clear()
            ctrl.cold_count = 0; ctrl.hot_count = 0
            ctrl._bufs.clear(); ctrl.k_init_history.clear()
            ctrl.s_mask_cache.clear(); ctrl.pop_cache.clear()
            ctrl._fwd_in_block.clear(); ctrl._block_idx.clear()
            ctrl.eb_calls = 0; ctrl.eb_skips = 0

        results = OrderedDict()

        # C5
        print(f"\n{'='*60}")
        print("C5: Fused routing only")
        print(f"{'='*60}")
        patch_c5()
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        c5_times, c5_fwds = [], []
        for ri in range(2):
            dllm = make_dllm(decoder_t0); torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            c5_times.append(t1-t0); c5_fwds.append(dllm.diff_iteration.num_forwards)
            print(f"    Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd")
        results['C5'] = {'avg_time': sum(c5_times)/2, 'avg_fwd': sum(c5_fwds)/2,
                         'ms_per_fwd': sum(c5_times)/2 / (sum(c5_fwds)/2) * 1000}

        # M-sweep
        M_VALUES = [1, 2, 3, 5, 10, float('inf')]
        for M in M_VALUES:
            m_label = f"M={M}" if M != float('inf') else "M=inf"
            print(f"\n{'='*60}")
            print(f"C10 {m_label}")
            print(f"{'='*60}")

            ctrl = MSkipEBController128(
                num_layers=19, K=8, M=4, K_target=40,
                quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=M)
            patch_eb(ctrl)

            # Warmup
            dllm = make_dllm(decoder_t0)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()

            # Timing
            reset_ctrl(ctrl)
            times, fwds = [], []
            for ri in range(2):
                reset_ctrl(ctrl)
                dllm = make_dllm(decoder_t0); torch.cuda.synchronize(); t0 = time.perf_counter()
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                times.append(t1-t0); fwds.append(dllm.diff_iteration.num_forwards)
                eb_rate = ctrl.eb_calls / max(ctrl.eb_calls + ctrl.eb_skips, 1) * 100
                print(f"    Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd "
                      f"| EB rate={eb_rate:.0f}%")

            # Quality
            print(f"  Quality (temp=0.7):")
            reset_ctrl(ctrl)
            dllm_q = make_dllm(decoder_t7)
            with torch.inference_mode():
                _ = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            reset_ctrl(ctrl)
            dllm_q = make_dllm(decoder_t7)
            with torch.inference_mode():
                out = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            gen_tokens = out[:, prompt_len:]
            quality = {}
            for bi in sorted(VERIFIABLE.keys()):
                gt = gen_tokens[bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                quality[bi] = text[:200]
                print(f"    #{bi}: {text[:120]}")

            avg_t = sum(times)/2; avg_f = sum(fwds)/2
            eb_rate = ctrl.eb_calls / max(ctrl.eb_calls + ctrl.eb_skips, 1) * 100
            results[m_label] = {
                'avg_time': avg_t, 'avg_fwd': avg_f,
                'ms_per_fwd': avg_t / avg_f * 1000,
                'eb_rate_pct': eb_rate, 'quality': quality,
            }

        # Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY (batch={BATCH_SIZE}, block={BLOCK_LENGTH})")
        print(f"{'='*80}")
        c5_t = results['C5']['avg_time']
        c5_f = results['C5']['avg_fwd']
        print(f"  {'Config':<15s} {'Time(s)':>8s} {'Fwd':>5s} {'ms/fwd':>8s} "
              f"{'vs C5':>8s} {'dFwd':>6s} {'EB rate':>8s}")
        print(f"  {'-'*62}")
        for cn, r in results.items():
            d = (r['avg_time'] - c5_t) / c5_t * 100
            df = r['avg_fwd'] - c5_f
            eb = f"{r.get('eb_rate_pct', 100):.0f}%" if 'eb_rate_pct' in r else "—"
            print(f"  {cn:<15s} {r['avg_time']:>8.3f} {r['avg_fwd']:>5.0f} "
                  f"{r['ms_per_fwd']:>8.2f} {d:>+7.1f}% {df:>+5.0f} {eb:>8s}")

        out_path = REPO_ROOT / "codex_coding" / "results" / "m_skip_sweep_b128_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
