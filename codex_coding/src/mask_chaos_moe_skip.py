#!/usr/bin/env python3
"""
mask_chaos Experiment 3: MoE Skip for Low-Confidence Tokens

For tokens with prev_confidence < threshold:
  - shared_experts: computed normally
  - routed_experts: replaced with hs * sum(routing_weights) (identity pass-through)

Architecture: patch gate.routing (saves topk_w) + post-hook on MoE block.
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
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


class MSkipEBController(FusedEBController):
    def __init__(self, *args, skip_m=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_m = skip_m
        self.k_init_history = []; self.s_mask_cache = {}; self.pop_cache = {}
        self._fwd_in_block = {}; self._block_idx = {}

    def cold_path(self, layer_idx, logits, bias):
        N, E = logits.shape; b = self._get_bufs(N, E, logits.device)
        _kernel_zero_init[(1,)](b['pop'], b['sat_flag'], b['sat_count'], b['G'], b['H'], E=E)
        lf = logits.float(); bf = bias.float()
        _kernel_A_cold[(N,)](lf, bf, b['pop'], b['topkm_idx'], b['topkm_w'], b['r'],
                             N, self.rsf, self.quality_floor, lf.stride(0), lf.stride(1),
                             b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                             E=E, KEXT=self.K_ext, KEXT_PAD=16, K=self.K)
        _kernel_B_v3[(1,)](b['pop'], b['s_mask'], self.K_target, E=E)
        q = int(self.q_major * 1000)
        for _ in range(self.MAX_ROUNDS):
            _kernel_C[(N,)](b['topkm_idx'], b['topkm_w'], b['r'],
                           b['s_mask'], b['sat_flag'], b['sat_count'], b['G'], b['H'],
                           N, b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                           E=E, KEXT=self.K_ext, KEXT_PAD=16)
            _kernel_D_v2[(1,)](b['s_mask'], b['sat_flag'], b['sat_count'],
                              b['G'], b['H'], N, q, E=E, CAP=self.cap)
        actual_s = int(b['s_mask'].sum().item())
        self.K_init[layer_idx] = actual_s; self.k_init_history.append(actual_s)
        if layer_idx not in self.s_mask_cache:
            self.s_mask_cache[layer_idx] = torch.zeros(256, device=logits.device, dtype=torch.int32)
            self.pop_cache[layer_idx] = torch.zeros(256, device=logits.device, dtype=torch.float32)
        self.s_mask_cache[layer_idx].copy_(b['s_mask'])
        self._block_idx[layer_idx] = self._block_idx.get(layer_idx, -1) + 1
        self._fwd_in_block[layer_idx] = 0
        self.cold_count += 1; return b['s_mask']

    def hot_path(self, layer_idx, logits, bias):
        fi = self._fwd_in_block.get(layer_idx, 0) + 1; self._fwd_in_block[layer_idx] = fi
        if self.skip_m == float('inf') or fi % self.skip_m != 0:
            self.hot_count += 1; return self.s_mask_cache[layer_idx]
        N, E = logits.shape; K_init = self.K_init.get(layer_idx, 103)
        pop = self.pop_cache[layer_idx]; lf = logits.float()
        _kernel_A[(N,)](lf, bias.float(), pop, N, self.rsf,
                        lf.stride(0), lf.stride(1), E=E, KEXT=self.K_ext, KEXT_PAD=16)
        _kernel_B_v3[(1,)](pop, self.s_mask_cache[layer_idx], K_init, E=E)
        self.hot_count += 1; return self.s_mask_cache[layer_idx]

    def reset(self):
        self.prev_N.clear(); self.K_init.clear(); self.cold_count = 0; self.hot_count = 0
        self._bufs.clear(); self.k_init_history.clear()
        self.s_mask_cache.clear(); self.pop_cache.clear()
        self._fwd_in_block.clear(); self._block_idx.clear()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration
    from transformers import AutoTokenizer, AutoConfig

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print("mask_chaos Exp3: MoE Skip (identity pass-through for low-conf tokens)")
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

        # EB controller
        ctrl = MSkipEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)

        # State shared across closures
        state = {
            'prev_confidence': None,
            'chaos_threshold': 0.0,
            'skip_counts': [],
            'total_counts': [],
            'saved_topk_w': {},   # layer_idx -> topk_w [N, K]
            'saved_hs_flat': {},  # layer_idx -> hs_flat [N, h]
        }

        # Save original gate.routing functions
        orig_routings = {}
        moe_layer_map = {}  # li (model layer idx) -> eb_layer_idx
        idx = 0
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer.mlp, 'gate'):
                orig_routings[li] = layer.mlp.gate.routing
                moe_layer_map[li] = idx
                idx += 1

        def patch_all():
            """Patch gate.routing + MoE forward for all MoE layers."""
            hooks = []

            for li, layer in enumerate(model.model.layers):
                if not hasattr(layer.mlp, 'gate'):
                    continue
                moe = layer.mlp
                gate = moe.gate
                eb_li = moe_layer_map[li]
                b, r = gate.expert_bias, gate.routed_scaling_factor
                tk, ng, tkg = gate.top_k, gate.n_group, gate.topk_group

                # Patch gate.routing: use EB + fused_routing + save topk_w
                def mk_routing(bb, rr, tt, nn, gg, layer_i, cc):
                    def fn(hs, go, topk, renorm):
                        N = go.shape[0]
                        prev = cc.prev_N.get(layer_i, -1)
                        is_cold = (prev == -1) or (N > prev)
                        sm = cc.get_s_mask(layer_i, go, bb)
                        w, ids = fused_routing(go, bb, rr, s_mask=sm, K=tt, ng=nn, tkg=gg)
                        state['saved_topk_w'][layer_i] = w.detach()
                        return w.to(go.dtype), ids
                    return fn
                gate.routing = mk_routing(b, r, tk, ng, tkg, eb_li, ctrl)

                # Pre-hook on MoE block: save hs_flat for identity computation
                def mk_pre(layer_i):
                    def pre_hook(mod, inp):
                        hs = inp[0]  # [batch, seq, hidden]
                        state['saved_hs_flat'][layer_i] = hs.view(-1, hs.shape[-1]).detach()
                    return pre_hook

                # Post-hook on MoE block: apply skip for low-conf tokens
                def mk_post(layer_i):
                    def post_hook(mod, inp, out):
                        th = state['chaos_threshold']
                        prev_conf = state['prev_confidence']
                        if th <= 0.0 or prev_conf is None:
                            return out

                        hs_flat = state['saved_hs_flat'].get(layer_i)
                        topk_w = state['saved_topk_w'].get(layer_i)
                        if hs_flat is None or topk_w is None:
                            return out

                        N = hs_flat.shape[0]
                        if prev_conf.shape[0] != N:
                            return out

                        low_conf = prev_conf < th  # [N] bool, CPU
                        n_skip = int(low_conf.sum())
                        if n_skip == 0:
                            return out

                        # Record stats (once per forward at first MoE layer)
                        if layer_i == 0:
                            state['skip_counts'].append(n_skip)
                            state['total_counts'].append(N)

                        low_conf_gpu = low_conf.to(device)

                        # out = shared(hs) + routed(hs), shape [batch, seq, hidden]
                        bsz, seq_len, h = out.shape
                        out_flat = out.view(-1, h)

                        # shared(hs) is the same for skip and non-skip tokens
                        # For skip tokens, replace routed(hs) with hs * sum(topk_w)
                        # routed(hs) = out - shared(hs), but we don't have shared separately
                        # Instead: skip_out = shared(hs) + hs * sum(topk_w)
                        #        = (out - routed(hs)) + hs * sum(topk_w)
                        # We can compute: out[skip] = out[skip] - routed[skip] + identity[skip]
                        # But we don't have routed[skip] separately.

                        # Simpler approach: recompute shared for skip tokens
                        # Actually simplest: compute full skip output and blend
                        # skip_token_output = shared(hs) + hs * weight_sum
                        # But shared(hs) was already computed inside MoE forward...

                        # Best approach: compute identity contribution and use it to
                        # reconstruct what the output SHOULD be for skip tokens.
                        # We know: out = shared + routed
                        # We want: out_new = shared + identity (for skip tokens)
                        # So: out_new = out - routed + identity
                        # = out - (out - shared) + identity
                        # = shared + identity
                        # But we need 'shared' which we don't have separately.

                        # Practical workaround: recompute shared_experts for the blend
                        shared_out = mod.shared_experts(inp[0])  # [bsz, seq, h]
                        shared_flat = shared_out.view(-1, h)

                        weight_sum = topk_w.sum(dim=1, keepdim=True).to(hs_flat.dtype)  # [N, 1]
                        identity_routed = hs_flat * weight_sum  # [N, h]
                        skip_output = shared_flat + identity_routed  # [N, h]

                        # Blend: use skip_output for low-conf, original out for high-conf
                        new_out = torch.where(
                            low_conf_gpu.unsqueeze(1),
                            skip_output,
                            out_flat
                        )
                        return new_out.view(bsz, seq_len, h)

                    return post_hook

                hooks.append(moe.register_forward_pre_hook(mk_pre(eb_li)))
                hooks.append(moe.register_forward_hook(mk_post(eb_li)))

            return hooks

        def restore_all():
            for li in orig_routings:
                model.model.layers[li].mlp.gate.routing = orig_routings[li]

        # Hook LM head for confidence
        lm_output = {}
        def lm_hook(mod, inp, out):
            lm_output['logits'] = out.detach()
        lm_handle = model.lm_head.register_forward_hook(lm_hook)

        # Patch iteration forward to update confidence
        orig_iter_fwd = BlockDiffusionIteration.forward
        def patched_iter_fwd(self_iter, *args, **kwargs):
            result = orig_iter_fwd(self_iter, *args, **kwargs)
            if 'logits' in lm_output:
                logits = lm_output['logits']
                probs = torch.softmax(logits.view(-1, logits.shape[-1]).float(), dim=-1)
                state['prev_confidence'] = probs.max(dim=-1).values.cpu()
            return result
        BlockDiffusionIteration.forward = patched_iter_fwd

        decoder = ThresholdParallelDecoder(temperature=0.0, threshold=0.90,
                                           mask_id=MASK_ID, eos_id=EOS_ID)
        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        def check_quality(output_ids, label):
            checks = {
                0: ["480/7", "68.57", "68.6", "average speed"],
                8: ["x = 2", "x = 3", "x=2", "x=3"],
                13: ["B", "C", "D"],
                19: ["55", "fibonacci", "fib"],
                28: ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"],
            }
            results = {}
            for idx, keywords in checks.items():
                if idx >= output_ids.shape[0]: continue
                tokens = output_ids[idx]
                text = tokenizer.decode(tokens[tokens != MASK_ID], skip_special_tokens=True)
                found = sum(1 for kw in keywords if kw.lower() in text.lower())
                results[idx] = found > 0
            n_pass = sum(results.values())
            n_total = len(results)
            status = 'PASS' if n_pass == n_total else 'PARTIAL' if n_pass > 0 else 'FAIL'
            print(f"    Quality [{label}]: {n_pass}/{n_total} {status}")
            return n_pass, n_total

        # Warmup with patched routing (no skip)
        print("\nWarmup...")
        moe_hooks = patch_all()
        state['chaos_threshold'] = 0.0
        state['prev_confidence'] = None
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd")

        # Run experiments
        all_results = {}
        for th in THRESHOLDS:
            label = f"skip<{th:.1f}" if th > 0 else "baseline"
            print(f"\n--- {label} ---")

            # Reset state
            ctrl.reset()
            state['prev_confidence'] = None
            state['chaos_threshold'] = th
            state['skip_counts'] = []
            state['total_counts'] = []
            state['saved_topk_w'].clear()
            state['saved_hs_flat'].clear()

            dllm = make_dllm()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                output_ids = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                            block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            total_fwd = dllm.diff_iteration.num_forwards
            total_time = t1 - t0
            avg_skip = 0.0
            if state['skip_counts']:
                rates = [s / t for s, t in zip(state['skip_counts'], state['total_counts'])]
                avg_skip = sum(rates) / len(rates)

            print(f"    Time: {total_time:.3f}s | Fwd: {total_fwd} | ms/fwd: {total_time*1000/total_fwd:.2f}")
            print(f"    Avg skip rate: {avg_skip:.1%}")
            qp, qt = check_quality(output_ids, label)

            all_results[label] = {
                'threshold': th, 'time_s': total_time, 'fwd': total_fwd,
                'ms_per_fwd': total_time * 1000 / total_fwd,
                'avg_skip_rate': avg_skip, 'quality_pass': qp, 'quality_total': qt,
            }

        # Cleanup
        for h in moe_hooks: h.remove()
        lm_handle.remove()
        BlockDiffusionIteration.forward = orig_iter_fwd
        restore_all()

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY: MoE Skip (identity pass-through)")
        print(f"{'='*80}")
        print(f"  {'Config':<15s} {'Time(s)':>8s} {'Fwd':>5s} {'ms/fwd':>8s} "
              f"{'SkipRate':>9s} {'Quality':>8s} {'ΔFwd':>6s}")
        print(f"  {'-'*62}")
        base_fwd = all_results.get('baseline', {}).get('fwd', 0)
        for label in all_results:
            r = all_results[label]
            dfwd = r['fwd'] - base_fwd if base_fwd else 0
            q_str = f"{r['quality_pass']}/{r['quality_total']}"
            print(f"  {label:<15s} {r['time_s']:>8.3f} {r['fwd']:>5d} {r['ms_per_fwd']:>8.2f} "
                  f"{r['avg_skip_rate']:>8.1%} {q_str:>8s} {dfwd:>+6d}")

        # Save
        out_path = REPO_ROOT / "codex_coding" / "results" / "mask_chaos_moe_skip.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
