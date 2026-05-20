#!/usr/bin/env python3
"""
mask_chaos Experiment 1: Random Routing for MASK tokens

Hypothesis: MASK tokens with low confidence don't need precise expert routing.
Replacing their topk_ids with random experts from S_mask should not affect quality.

Sweep: confidence_threshold = [0.0, 0.3, 0.5, 0.7, 0.9]
  - 0.0: ALL tokens get random routing (most aggressive)
  - 0.9: Only tokens with confidence < 0.9 get random routing (least aggressive)

For each config: measure fwd count + quality check.
Reference: C10-M5 with normal routing = ~282 fwd.
"""

from __future__ import annotations
import os, sys, time, socket, json, random
from pathlib import Path
from collections import OrderedDict

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
from test_heteval128 import PROMPTS, VERIFIABLE

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
GEN_LENGTH = 256
BATCH_SIZE = 128

# Confidence thresholds to sweep
# Below this confidence → random routing applied
CHAOS_THRESHOLDS = [0.0, 0.3, 0.5, 0.7, 0.9]


class MSkipEBController(FusedEBController):
    def __init__(self, *args, skip_m=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_m = skip_m
        self.s_mask_cache = {}; self.pop_cache = {}
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
        self.K_init[layer_idx] = actual_s
        if layer_idx not in self.s_mask_cache:
            self.s_mask_cache[layer_idx] = torch.zeros(256, device=logits.device, dtype=torch.int32)
            self.pop_cache[layer_idx] = torch.zeros(256, device=logits.device, dtype=torch.float32)
        self.s_mask_cache[layer_idx].copy_(b['s_mask'])
        self._block_idx[layer_idx] = self._block_idx.get(layer_idx, -1) + 1
        self._fwd_in_block[layer_idx] = 0
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
        self._bufs.clear(); self.s_mask_cache.clear(); self.pop_cache.clear()
        self._fwd_in_block.clear(); self._block_idx.clear()
        self.eb_calls = 0; self.eb_skips = 0


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
    print("mask_chaos Experiment 1: Random Routing for low-confidence tokens")
    print(f"  batch={BATCH_SIZE}, gen_length={GEN_LENGTH}, M=5")
    print(f"  Thresholds: {CHAOS_THRESHOLDS}")
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

        # Save original routing
        orig_routings = {}
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                orig_routings[name] = mod.routing

        def restore():
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate" and name in orig_routings:
                    mod.routing = orig_routings[name]

        # We need access to current input_ids to know which tokens are MASK
        # Hook into the model forward to capture current input
        current_input = {}

        def input_hook(mod, args, kwargs):
            if args:
                current_input['input_ids'] = args[0]
            elif 'input_ids' in kwargs:
                current_input['input_ids'] = kwargs['input_ids']
            return None

        # Hook LM head to get confidence from previous forward
        prev_confidence = {}  # {layer_0_N: confidence_array}
        lm_head_logits = {}

        def lm_head_hook(mod, inp, out):
            lm_head_logits['logits'] = out.detach()

        lm_hook = model.lm_head.register_forward_hook(lm_head_hook)

        # Track which tokens are MASK in current forward
        # In block diffusion, input_ids contains MASK_ID for unresolved positions
        from dinfer.decoding.generate_uniform import BlockDiffusionIteration

        orig_iter_fwd = BlockDiffusionIteration.forward

        def make_chaos_iter_fwd(chaos_threshold):
            """Create patched iteration forward that updates prev_confidence after each fwd."""
            def patched_fwd(self_iter, *args, **kwargs):
                result = orig_iter_fwd(self_iter, *args, **kwargs)
                # After forward: compute confidence
                if 'logits' in lm_head_logits:
                    logits = lm_head_logits['logits']
                    probs = torch.softmax(logits.float(), dim=-1)
                    conf = probs.max(dim=-1).values  # [batch, seq]
                    prev_confidence['conf'] = conf.detach()
                return result
            return patched_fwd

        def patch_chaos_routing(ctrl, chaos_threshold):
            """Patch routing: for MASK tokens with prev_confidence < chaos_threshold → random expert IDs."""
            restore()
            idx = 0
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate":
                    b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                         mod.top_k, mod.n_group, mod.topk_group)
                    li = idx
                    def mk(bb, rr, tt, nn, gg, layer_i, cc, ch_th):
                        def fn(hs, go, topk, renorm):
                            N = go.shape[0]
                            prev = cc.prev_N.get(layer_i, -1)
                            is_cold = (prev == -1) or (N > prev)
                            sm = cc.get_s_mask(layer_i, go, bb)

                            # Normal routing first
                            w, ids = fused_routing(go, bb, rr, s_mask=sm, K=tt, ng=nn, tkg=gg)

                            # Apply chaos: randomize routing for low-confidence tokens
                            if ch_th > 0.0 and 'conf' in prev_confidence:
                                conf = prev_confidence['conf']  # [batch, seq]
                                conf_flat = conf.reshape(-1)    # [N]

                                if conf_flat.shape[0] == N:
                                    # Get active experts from S_mask
                                    active_experts = sm.nonzero(as_tuple=True)[0]  # [|S|]
                                    n_active = active_experts.shape[0]

                                    # Identify low-confidence tokens
                                    low_conf_mask = conf_flat < ch_th  # [N]
                                    n_chaos = int(low_conf_mask.sum().item())

                                    if n_chaos > 0:
                                        # Generate random expert IDs from active set
                                        rand_idx = torch.randint(0, n_active, (n_chaos, tt),
                                                                 device=ids.device)
                                        rand_ids = active_experts[rand_idx]  # [n_chaos, K]

                                        # Replace routing for low-confidence tokens
                                        ids[low_conf_mask] = rand_ids.to(ids.dtype)

                                        # Uniform weights for randomized tokens
                                        w[low_conf_mask] = (1.0 / tt) * rr

                            return w.to(go.dtype), ids
                        return fn
                    mod.routing = mk(b, r, tk, ng, tkg, li, ctrl, chaos_threshold)
                    idx += 1

        results = OrderedDict()

        # ============================================================
        # Baseline: C10-M5 normal routing
        # ============================================================
        print(f"\n{'='*60}")
        print("Baseline: C10-M5 (normal routing)")
        print(f"{'='*60}")
        ctrl = MSkipEBController(num_layers=19, K=8, M=4, K_target=40,
                                  quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)
        patch_chaos_routing(ctrl, chaos_threshold=0.0)  # 0.0 = no chaos
        BlockDiffusionIteration.forward = make_chaos_iter_fwd(0.0)

        # Warmup
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        # Timing
        ctrl.reset(); prev_confidence.clear()
        dllm = make_dllm(decoder_t0); torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        baseline_fwd = dllm.diff_iteration.num_forwards
        baseline_time = t1 - t0
        print(f"  {baseline_fwd} fwd, {baseline_time:.3f}s")
        results['baseline'] = {'fwd': baseline_fwd, 'time': baseline_time}

        # ============================================================
        # Chaos sweep
        # ============================================================
        for ch_th in CHAOS_THRESHOLDS:
            print(f"\n{'='*60}")
            print(f"Chaos threshold: {ch_th} (randomize tokens with confidence < {ch_th})")
            print(f"{'='*60}")

            ctrl.reset(); prev_confidence.clear()
            patch_chaos_routing(ctrl, chaos_threshold=ch_th)
            BlockDiffusionIteration.forward = make_chaos_iter_fwd(ch_th)

            # Warmup
            dllm = make_dllm(decoder_t0)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()

            # Timing run
            ctrl.reset(); prev_confidence.clear()
            dllm = make_dllm(decoder_t0); torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                out = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            fwd = dllm.diff_iteration.num_forwards
            dfwd = fwd - baseline_fwd
            print(f"  {fwd} fwd (Δ{dfwd:+d}), {t1-t0:.3f}s")

            # Quality check
            ctrl.reset(); prev_confidence.clear()
            dllm_q = make_dllm(decoder_t7)
            with torch.inference_mode():
                dllm_q.diff_iteration.num_forwards = 0
                _ = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            ctrl.reset(); prev_confidence.clear()
            dllm_q = make_dllm(decoder_t7)
            with torch.inference_mode():
                out_q = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            gen_tokens = out_q[:, prompt_len:]
            quality = {}
            print(f"  Quality (temp=0.7):")
            for bi in sorted(VERIFIABLE.keys()):
                gt = gen_tokens[bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                quality[bi] = text[:200]
                print(f"    #{bi}: {text[:100]}")

            results[f'chaos_{ch_th}'] = {
                'threshold': ch_th, 'fwd': fwd, 'delta_fwd': dfwd,
                'time': t1-t0, 'quality': quality,
            }

        # ============================================================
        # Summary
        # ============================================================
        BlockDiffusionIteration.forward = orig_iter_fwd
        lm_hook.remove()

        print(f"\n{'='*80}")
        print("SUMMARY: mask_chaos Random Routing")
        print(f"{'='*80}")
        print(f"  {'Config':<25s} {'Fwd':>5s} {'ΔFwd':>6s} {'Time':>8s} {'Verdict':>10s}")
        print(f"  {'-'*58}")
        print(f"  {'Baseline (no chaos)':<25s} {baseline_fwd:>5d} {0:>+5d} {baseline_time:>7.3f}s {'—':>10s}")
        for ch_th in CHAOS_THRESHOLDS:
            key = f'chaos_{ch_th}'
            r = results[key]
            verdict = "SAFE" if abs(r['delta_fwd']) <= 5 else ("RISKY" if abs(r['delta_fwd']) <= 15 else "FAIL")
            print(f"  {'chaos<'+str(ch_th):<25s} {r['fwd']:>5d} {r['delta_fwd']:>+5d} "
                  f"{r['time']:>7.3f}s {verdict:>10s}")

        out_path = REPO_ROOT / "codex_coding" / "results" / "mask_chaos_random_routing.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
