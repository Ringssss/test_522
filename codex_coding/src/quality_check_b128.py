#!/usr/bin/env python3
"""
Quality check: C5 vs C10-M5, all 128 prompts, temp=0.

Saves full decoded text for all prompts, runs automated pre-screening,
and outputs a quality report.
"""

from __future__ import annotations
import os, sys, time, socket, json, re
from pathlib import Path
from collections import OrderedDict

import torch

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

from test_fused_eb_triton import (
    fused_routing, FusedEBController,
    _kernel_A, _kernel_A_cold, _kernel_B_v3,
    _kernel_C, _kernel_D_v2, _kernel_zero_init,
)
from baseline_optimizations import apply_all_optimizations
from test_heteval128 import PROMPTS, VERIFIABLE

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
GEN_LENGTH = 256
BATCH_SIZE = 128


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


def detect_repetition(text, ngram_size=10, threshold=5):
    """Detect excessive n-gram repetition. Returns (is_repetitive, details)."""
    words = text.split()
    if len(words) < ngram_size * 2:
        return False, ""
    ngrams = {}
    for i in range(len(words) - ngram_size + 1):
        ng = " ".join(words[i:i+ngram_size])
        ngrams[ng] = ngrams.get(ng, 0) + 1
    worst = max(ngrams.values()) if ngrams else 0
    if worst >= threshold:
        worst_ng = [k for k, v in ngrams.items() if v == worst][0]
        return True, f"'{worst_ng[:60]}...' repeated {worst}x"
    return False, ""


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
    print(f"Quality Check: C5 vs C10-M5 (batch={BATCH_SIZE}, temp=0)")
    print(f"  gen_length={GEN_LENGTH}, block={BLOCK_LENGTH}")
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
        prompt_len = input_ids.shape[1]
        print(f"  Input shape: {input_ids.shape}")

        decoder_t0 = ThresholdParallelDecoder(temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder_t0, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Save original routing
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

        def decode_outputs(gen_tokens):
            """Decode generated tokens for all batch entries."""
            texts = []
            for bi in range(gen_tokens.shape[0]):
                gt = gen_tokens[bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                texts.append(text)
            return texts

        results = {}

        # ============================================================
        # C5
        # ============================================================
        print(f"\n{'='*60}")
        print("Generating C5 outputs (temp=0)...")
        print(f"{'='*60}")
        patch_c5()

        # Warmup
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        # Actual run
        dllm = make_dllm()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            out_c5 = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd_c5 = dllm.diff_iteration.num_forwards
        print(f"  Done: {fwd_c5} fwd, {t1-t0:.3f}s")

        gen_c5 = out_c5[:, prompt_len:]
        texts_c5 = decode_outputs(gen_c5)
        c5_lens = [len(t.split()) for t in texts_c5]
        print(f"  Avg output length: {sum(c5_lens)/len(c5_lens):.0f} words")

        # ============================================================
        # C10-M5
        # ============================================================
        print(f"\n{'='*60}")
        print("Generating C10-M5 outputs (temp=0, q_major=1.0)...")
        print(f"{'='*60}")
        ctrl = MSkipEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)
        patch_eb(ctrl)

        # Warmup
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        # Actual run
        ctrl.reset()
        dllm = make_dllm()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            out_c10 = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd_c10 = dllm.diff_iteration.num_forwards
        print(f"  Done: {fwd_c10} fwd, {t1-t0:.3f}s")

        gen_c10 = out_c10[:, prompt_len:]
        texts_c10 = decode_outputs(gen_c10)
        c10_lens = [len(t.split()) for t in texts_c10]
        print(f"  Avg output length: {sum(c10_lens)/len(c10_lens):.0f} words")

        # ============================================================
        # Quality Analysis
        # ============================================================
        print(f"\n{'='*80}")
        print("QUALITY ANALYSIS")
        print(f"{'='*80}")

        issues = []
        per_prompt = []

        for i in range(BATCH_SIZE):
            t5 = texts_c5[i]
            t10 = texts_c10[i]
            l5 = len(t5.split())
            l10 = len(t10.split())

            flags = []

            # Check 1: Empty or very short
            if l10 < 20:
                flags.append(f"VERY_SHORT (C10={l10} words)")
            elif l10 < 50:
                flags.append(f"SHORT (C10={l10} words)")

            # Check 2: Length ratio
            if l5 > 20:
                ratio = l10 / l5
                if ratio < 0.5:
                    flags.append(f"LENGTH_DROP (ratio={ratio:.2f})")
                elif ratio > 2.0:
                    flags.append(f"LENGTH_BLOAT (ratio={ratio:.2f})")

            # Check 3: Repetition
            rep, rep_detail = detect_repetition(t10)
            if rep:
                flags.append(f"REPETITION: {rep_detail}")

            # Check 4: Verifiable prompts
            if i in VERIFIABLE:
                flags.append(f"[VERIFIABLE: {VERIFIABLE[i]}]")

            status = "FLAG" if any(f for f in flags if not f.startswith("[VERIFIABLE")) else "OK"

            per_prompt.append({
                'idx': i,
                'c5_words': l5, 'c10_words': l10,
                'status': status,
                'flags': flags,
                'c5_text': t5, 'c10_text': t10,
            })

            if flags:
                issues.append((i, flags))

        # Print flagged prompts
        n_ok = sum(1 for p in per_prompt if p['status'] == 'OK')
        n_flag = sum(1 for p in per_prompt if p['status'] == 'FLAG')
        print(f"\n  Automated screening: {n_ok} OK, {n_flag} FLAGGED")

        if n_flag > 0:
            print(f"\n  FLAGGED prompts:")
            for i, flags in issues:
                if per_prompt[i]['status'] == 'FLAG':
                    print(f"    #{i}: {', '.join(flags)}")
                    print(f"      C5  ({per_prompt[i]['c5_words']} words): {texts_c5[i][:100]}...")
                    print(f"      C10 ({per_prompt[i]['c10_words']} words): {texts_c10[i][:100]}...")

        # Print verifiable prompts
        print(f"\n  Verifiable prompts (5):")
        for i in sorted(VERIFIABLE.keys()):
            print(f"    #{i} [{VERIFIABLE[i]}]:")
            print(f"      C5:  {texts_c5[i][:150]}")
            print(f"      C10: {texts_c10[i][:150]}")

        # Print sample comparison (20 random non-verifiable)
        import random
        random.seed(42)
        non_verifiable = [i for i in range(BATCH_SIZE) if i not in VERIFIABLE]
        sample = sorted(random.sample(non_verifiable, min(20, len(non_verifiable))))
        print(f"\n  Sample comparison (20 prompts, first 150 chars):")
        for i in sample:
            p = per_prompt[i]
            print(f"\n    #{i} (C5={p['c5_words']}w, C10={p['c10_words']}w) "
                  f"{'[' + p['status'] + ']' if p['status'] != 'OK' else ''}")
            print(f"      C5:  {texts_c5[i][:150]}")
            print(f"      C10: {texts_c10[i][:150]}")

        # Length statistics
        ratios = [c10_lens[i] / max(c5_lens[i], 1) for i in range(BATCH_SIZE)]
        print(f"\n  Length statistics:")
        print(f"    C5  avg: {sum(c5_lens)/len(c5_lens):.0f} words")
        print(f"    C10 avg: {sum(c10_lens)/len(c10_lens):.0f} words")
        print(f"    Ratio avg: {sum(ratios)/len(ratios):.3f}")
        print(f"    Ratio min: #{min(range(len(ratios)), key=lambda i: ratios[i])} = {min(ratios):.3f}")
        print(f"    Ratio max: #{max(range(len(ratios)), key=lambda i: ratios[i])} = {max(ratios):.3f}")

        # Save full results
        save_data = {
            'config': {'batch': BATCH_SIZE, 'gen_length': GEN_LENGTH, 'block': BLOCK_LENGTH,
                       'M': 5, 'q_major': 1.0, 'temp': 0.0},
            'c5_fwd': fwd_c5, 'c10_fwd': fwd_c10,
            'n_ok': n_ok, 'n_flag': n_flag,
            'prompts': [{
                'idx': p['idx'], 'c5_words': p['c5_words'], 'c10_words': p['c10_words'],
                'status': p['status'], 'flags': p['flags'],
                'c5_text': p['c5_text'], 'c10_text': p['c10_text'],
            } for p in per_prompt],
        }
        out_path = REPO_ROOT / "codex_coding" / "results" / "quality_check_c5_vs_c10m5.json"
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"\n  Full results saved to {out_path}")

        print("\nDone.")


if __name__ == "__main__":
    main()
