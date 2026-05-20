#!/usr/bin/env python3
"""
mask_chaos Exp5: Top-1 Concentration for low-confidence tokens.

Low-conf tokens: topk_ids[:, 1:] = topk_ids[:, 0] (all 8 slots → top-1 expert)
All tokens still go through fused_experts. No identity, no skip.

Measures: quality, fwd, unique expert count reduction.
Sweep: [0.0, 0.3, 0.5, 0.6]
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

from test_fused_eb_triton import (
    fused_routing, FusedEBController,
    _kernel_A_cold, _kernel_B_v3,
    _kernel_C, _kernel_D_v2, _kernel_zero_init,
)
from baseline_optimizations import apply_all_optimizations
from test_heteval128 import PROMPTS, ColdOnlyEBController

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
GEN_LENGTH = 256
BATCH_SIZE = 128
THRESHOLDS = [0.0, 0.3, 0.5, 0.6]

VERIFY_INDICES = [0, 8, 13, 19, 28]
VERIFY_LABELS = {0: "Math", 8: "QuadEq", 13: "Logic", 19: "Fib", 28: "Planets"}


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
    print("mask_chaos Exp5: Top-1 Concentration for low-conf tokens")
    print(f"  batch={BATCH_SIZE}, gen={GEN_LENGTH}, M=inf, q_major=1.0")
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
        prompt_len = input_ids.shape[1]

        decoder_t0 = ThresholdParallelDecoder(temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        decoder_t7 = ThresholdParallelDecoder(temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm(dec):
            return BlockDiffusionLLM(
                model, dec, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        state = {
            'prev_confidence': None,
            'chaos_threshold': 0.0,
            'override_counts': [],
            'total_counts': [],
            'unique_expert_counts': [],  # per-forward, per-layer
        }

        orig_routings = {}
        moe_layer_map = {}
        eb_idx = 0
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer.mlp, 'gate'):
                orig_routings[li] = layer.mlp.gate.routing
                moe_layer_map[li] = eb_idx
                eb_idx += 1

        def restore_all():
            for li in orig_routings:
                model.model.layers[li].mlp.gate.routing = orig_routings[li]

        def patch_top1_routing(ctrl):
            restore_all()
            for li, layer in enumerate(model.model.layers):
                if not hasattr(layer.mlp, 'gate'):
                    continue
                gate = layer.mlp.gate
                eb_li = moe_layer_map[li]
                b, r = gate.expert_bias, gate.routed_scaling_factor
                tk, ng, tkg = gate.top_k, gate.n_group, gate.topk_group

                def mk_routing(bb, rr, tt, nn, gg, layer_i, cc):
                    def fn(hs, go, topk, renorm):
                        N = go.shape[0]
                        sm = cc.get_s_mask(layer_i, go, bb)
                        w, ids = fused_routing(go, bb, rr, s_mask=sm, K=tt, ng=nn, tkg=gg)

                        th = state['chaos_threshold']
                        prev_conf = state['prev_confidence']

                        if th > 0 and prev_conf is not None and prev_conf.shape[0] == N:
                            low_conf = prev_conf < th  # [N] CPU bool
                            n_low = int(low_conf.sum())

                            if n_low > 0:
                                low_conf_gpu = low_conf.to(ids.device)
                                # Top-1 concentration: all 8 slots → top-1 expert
                                top1 = ids[low_conf_gpu, 0:1]  # [n_low, 1]
                                ids[low_conf_gpu] = top1.expand(-1, ids.shape[1])  # broadcast to K=8

                                if layer_i == 0:
                                    state['override_counts'].append(n_low)
                                    state['total_counts'].append(N)

                            # Record unique experts (all layers)
                            unique_count = int(ids.unique().numel())
                            state['unique_expert_counts'].append({
                                'layer': layer_i, 'unique': unique_count
                            })

                        return w.to(go.dtype), ids
                    return fn

                gate.routing = mk_routing(b, r, tk, ng, tkg, eb_li, ctrl)

        # LM head hook
        lm_output = {}
        def lm_hook(mod, inp, out): lm_output['logits'] = out.detach()
        lm_handle = model.lm_head.register_forward_hook(lm_hook)

        orig_iter_fwd = BlockDiffusionIteration.forward
        def patched_iter_fwd(self_iter, *args, **kwargs):
            result = orig_iter_fwd(self_iter, *args, **kwargs)
            if 'logits' in lm_output:
                logits = lm_output['logits']
                probs = torch.softmax(logits.view(-1, logits.shape[-1]).float(), dim=-1)
                state['prev_confidence'] = probs.max(dim=-1).values.cpu()
            return result
        BlockDiffusionIteration.forward = patched_iter_fwd

        def reset_state(ctrl):
            ctrl.prev_N.clear(); ctrl.K_init.clear()
            ctrl.cold_count = 0; ctrl.hot_count = 0
            ctrl._bufs.clear(); ctrl.k_init_history.clear()
            ctrl.s_mask_cache.clear()
            state['prev_confidence'] = None
            state['override_counts'] = []; state['total_counts'] = []
            state['unique_expert_counts'] = []

        # Warmup
        print("\nWarmup...")
        ctrl_w = ColdOnlyEBController(num_layers=19, K=8, M=4, K_target=40,
                                       quality_floor=0.70, q_major=1.0, per_round_cap=8)
        state['chaos_threshold'] = 0.0
        patch_top1_routing(ctrl_w)
        reset_state(ctrl_w)
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Done: {dllm.diff_iteration.num_forwards} fwd")

        results = OrderedDict()
        for th in THRESHOLDS:
            label = f"top1<{th:.1f}" if th > 0 else "baseline"
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")

            state['chaos_threshold'] = th
            ctrl = ColdOnlyEBController(num_layers=19, K=8, M=4, K_target=40,
                                         quality_floor=0.70, q_major=1.0, per_round_cap=8)
            patch_top1_routing(ctrl)

            # Timing (temp=0)
            reset_state(ctrl)
            dllm = make_dllm(decoder_t0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fwd_t0 = dllm.diff_iteration.num_forwards
            time_t0 = t1 - t0

            override_rate = 0.0
            if state['override_counts']:
                rates = [o / t for o, t in zip(state['override_counts'], state['total_counts'])]
                override_rate = sum(rates) / len(rates)

            # Compute per-layer unique expert stats
            ue_by_layer = {}
            for rec in state['unique_expert_counts']:
                li = rec['layer']
                if li not in ue_by_layer:
                    ue_by_layer[li] = []
                ue_by_layer[li].append(rec['unique'])
            avg_unique_all = np.mean([r['unique'] for r in state['unique_expert_counts']]) if state['unique_expert_counts'] else 0

            print(f"  temp=0: {time_t0:.3f}s, {fwd_t0} fwd, {time_t0*1000/fwd_t0:.2f} ms/fwd")
            print(f"  Override rate: {override_rate:.1%}")
            print(f"  Avg unique experts (all layers): {avg_unique_all:.1f}")
            if ue_by_layer:
                print(f"  By layer:")
                for li in sorted(ue_by_layer.keys()):
                    vals = ue_by_layer[li]
                    print(f"    L{li:>2d}: avg={np.mean(vals):.1f}, min={min(vals)}, max={max(vals)}")

            # Quality (temp=0.7)
            reset_state(ctrl)
            dllm = make_dllm(decoder_t7)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                out = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            fwd_t7 = dllm.diff_iteration.num_forwards

            gen_tokens = out[:, prompt_len:]
            print(f"  temp=0.7: {fwd_t7} fwd")
            print(f"  --- Visual Quality ---")
            for bi in VERIFY_INDICES:
                gt = gen_tokens[bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                print(f"    #{bi} [{VERIFY_LABELS[bi]}]: {text[:200]}")

            results[label] = {
                'threshold': th, 'time_s': time_t0, 'fwd_t0': fwd_t0,
                'ms_per_fwd': time_t0 * 1000 / fwd_t0,
                'override_rate': override_rate,
                'avg_unique_experts': avg_unique_all,
                'unique_by_layer': {str(li): {'avg': float(np.mean(v)), 'min': min(v), 'max': max(v)}
                                    for li, v in ue_by_layer.items()},
                'fwd_t7': fwd_t7,
            }

        # Cleanup
        lm_handle.remove()
        BlockDiffusionIteration.forward = orig_iter_fwd
        restore_all()

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY — Top-1 Concentration")
        print(f"{'='*80}")
        print(f"  {'Config':<12s} {'Time':>7s} {'Fwd0':>5s} {'ms/f':>6s} "
              f"{'Over%':>6s} {'UniqE':>6s} {'Fwd7':>5s}")
        print(f"  {'-'*50}")
        for label, r in results.items():
            print(f"  {label:<12s} {r['time_s']:>7.3f} {r['fwd_t0']:>5d} "
                  f"{r['ms_per_fwd']:>6.2f} {r['override_rate']:>5.1%} "
                  f"{r['avg_unique_experts']:>6.1f} {r['fwd_t7']:>5d}")

        out_path = REPO_ROOT / "codex_coding" / "results" / "mask_chaos_top1_exp5.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
