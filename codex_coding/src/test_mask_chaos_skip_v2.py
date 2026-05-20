#!/usr/bin/env python3
"""
mask_chaos Exp3v2 — MoE Skip sweep with visual quality inspection.

Based on test_heteval128.py architecture.
Sweep: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
Quality: print first 200 chars for visual inspection (same as test_heteval128.py).
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path
from collections import OrderedDict

import torch

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
THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

VERIFY_INDICES = [0, 8, 13, 19, 28]
VERIFY_LABELS = {
    0: "Math: avg speed", 8: "Quadratic eq",
    13: "Logic puzzle", 19: "Fibonacci",
    28: "Planets",
}


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
    print("mask_chaos Exp3v2: MoE Skip Sweep + Visual Quality")
    print(f"  batch={BATCH_SIZE}, gen={GEN_LENGTH}, M=inf, q_major=1.0")
    print(f"  Thresholds: {THRESHOLDS}")
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

        decoder_t0 = ThresholdParallelDecoder(temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        decoder_t7 = ThresholdParallelDecoder(temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm(dec):
            return BlockDiffusionLLM(
                model, dec, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Shared state
        state = {
            'chaos_threshold': 0.0,
            'prev_confidence': None,
            'saved_topk_w': {},
            'skip_counts': [],
            'total_counts': [],
        }

        # Save originals
        orig_routings = {}
        orig_moe_fwds = {}
        moe_layer_map = {}
        eb_idx = 0
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer.mlp, 'gate'):
                orig_routings[li] = layer.mlp.gate.routing
                orig_moe_fwds[li] = layer.mlp.forward
                moe_layer_map[li] = eb_idx
                eb_idx += 1

        def restore_all():
            for li in orig_routings:
                model.model.layers[li].mlp.gate.routing = orig_routings[li]
            for li in orig_moe_fwds:
                model.model.layers[li].mlp.forward = orig_moe_fwds[li]

        def patch_eb_skip(ctrl):
            restore_all()
            for li, layer in enumerate(model.model.layers):
                if not hasattr(layer.mlp, 'gate'): continue
                moe = layer.mlp; gate = moe.gate; eb_li = moe_layer_map[li]
                b, r = gate.expert_bias, gate.routed_scaling_factor
                tk, ng, tkg = gate.top_k, gate.n_group, gate.topk_group

                def mk_routing(bb, rr, tt, nn, gg, layer_i, cc):
                    def fn(hs, go, topk, renorm):
                        sm = cc.get_s_mask(layer_i, go, bb)
                        w, ids = fused_routing(go, bb, rr, s_mask=sm, K=tt, ng=nn, tkg=gg)
                        state['saved_topk_w'][layer_i] = w.detach()
                        return w.to(go.dtype), ids
                    return fn
                gate.routing = mk_routing(b, r, tk, ng, tkg, eb_li, ctrl)

                def mk_moe_fwd(blk, layer_i):
                    def fwd(hidden_states):
                        shared_res = blk.shared_experts(hidden_states)
                        bsz, seq_len, h = hidden_states.shape
                        hs_flat = hidden_states.view(-1, h)
                        N = hs_flat.shape[0]
                        logits = blk.gate.get_logits(hs_flat)
                        routed = blk.experts.forward_impl(
                            hidden_states=hs_flat, router_logits=logits)
                        th = state['chaos_threshold']
                        prev_conf = state['prev_confidence']
                        if th > 0 and prev_conf is not None and prev_conf.shape[0] == N:
                            low_conf = prev_conf < th
                            n_skip = int(low_conf.sum())
                            if n_skip > 0:
                                topk_w = state['saved_topk_w'].get(layer_i)
                                if topk_w is not None:
                                    low_conf_gpu = low_conf.to(device)
                                    weight_sum = topk_w.sum(dim=1, keepdim=True).to(hs_flat.dtype)
                                    identity = hs_flat * weight_sum
                                    routed = torch.where(low_conf_gpu.unsqueeze(1), identity, routed)
                                if layer_i == 0:
                                    state['skip_counts'].append(n_skip)
                                    state['total_counts'].append(N)
                        routed = routed.view(bsz, seq_len, h)
                        return shared_res + routed
                    return fwd
                moe.forward = mk_moe_fwd(moe, eb_li)

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
            state['saved_topk_w'].clear()
            state['skip_counts'] = []
            state['total_counts'] = []

        # Global warmup
        print("\nGlobal warmup...")
        ctrl_w = ColdOnlyEBController(num_layers=19, K=8, M=4, K_target=40,
                                       quality_floor=0.70, q_major=1.0, per_round_cap=8)
        state['chaos_threshold'] = 0.0
        patch_eb_skip(ctrl_w)
        reset_state(ctrl_w)
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Done: {dllm.diff_iteration.num_forwards} fwd")

        # Run all thresholds
        results = OrderedDict()

        for chaos_th in THRESHOLDS:
            label = f"skip<{chaos_th:.1f}" if chaos_th > 0 else "baseline"
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")

            state['chaos_threshold'] = chaos_th
            ctrl = ColdOnlyEBController(num_layers=19, K=8, M=4, K_target=40,
                                         quality_floor=0.70, q_major=1.0, per_round_cap=8)
            patch_eb_skip(ctrl)

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
            skip_t0 = 0.0
            if state['skip_counts']:
                rates = [s / t for s, t in zip(state['skip_counts'], state['total_counts'])]
                skip_t0 = sum(rates) / len(rates)
            print(f"  temp=0: {time_t0:.3f}s, {fwd_t0} fwd, {time_t0*1000/fwd_t0:.2f} ms/fwd, skip={skip_t0:.1%}")

            # Quality (temp=0.7)
            reset_state(ctrl)
            dllm = make_dllm(decoder_t7)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                out = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            fwd_t7 = dllm.diff_iteration.num_forwards
            skip_t7 = 0.0
            if state['skip_counts']:
                rates = [s / t for s, t in zip(state['skip_counts'], state['total_counts'])]
                skip_t7 = sum(rates) / len(rates)

            gen_tokens = out[:, prompt_len:]
            print(f"  temp=0.7: {fwd_t7} fwd, skip={skip_t7:.1%}")
            print(f"  --- Visual Quality Check (5 verifiable prompts) ---")
            for bi in VERIFY_INDICES:
                gt = gen_tokens[bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                print(f"    #{bi} [{VERIFY_LABELS[bi]}]: {text[:200]}")

            results[label] = {
                'chaos_threshold': chaos_th,
                'time_s': time_t0, 'fwd_t0': fwd_t0,
                'ms_per_fwd': time_t0 * 1000 / fwd_t0,
                'skip_rate_t0': skip_t0,
                'fwd_t7': fwd_t7, 'skip_rate_t7': skip_t7,
            }

        # Cleanup
        lm_handle.remove()
        BlockDiffusionIteration.forward = orig_iter_fwd
        restore_all()

        # Summary table
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Config':<12s} {'Time(s)':>8s} {'Fwd(t0)':>8s} {'ms/fwd':>8s} "
              f"{'Skip%(t0)':>10s} {'Fwd(t7)':>8s} {'Skip%(t7)':>10s}")
        print(f"  {'-'*66}")
        for label, r in results.items():
            print(f"  {label:<12s} {r['time_s']:>8.3f} {r['fwd_t0']:>8d} {r['ms_per_fwd']:>8.2f} "
                  f"{r['skip_rate_t0']:>9.1%} {r['fwd_t7']:>8d} {r['skip_rate_t7']:>9.1%}")

        out_path = REPO_ROOT / "codex_coding" / "results" / "mask_chaos_skip_v2.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
