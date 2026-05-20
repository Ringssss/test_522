#!/usr/bin/env python3
"""
mask_chaos Exp3 — MoE Skip Quality Verification (reliable architecture)

Based on test_heteval128.py's proven pattern:
  - gate.routing patch for EB + fused_routing (saves topk_w)
  - MoE forward patch for skip logic (shared + identity for low-conf tokens)
  - LM head hook for confidence tracking
  - Quality check: gen_tokens = out[:, prompt_len:], decode valid tokens

Sweep: baseline (no skip) + skip<0.3 + skip<0.5
Both timing (temp=0) and quality (temp=0.7).
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

VERIFIABLE = {
    0: ["480/7", "68.57", "68.6", "average speed"],
    8: ["x = 2", "x = 3", "x=2", "x=3", "2 and 3", "2, 3"],
    13: ["B is true", "C is true", "D is true"],
    19: ["55"],
    28: ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"],
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
    print("mask_chaos Exp3: MoE Skip — Reliable Architecture")
    print(f"  batch={BATCH_SIZE}, gen_length={GEN_LENGTH}, M=inf, q_major=1.0")
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
        print(f"  Input shape: {input_ids.shape}, prompt_len={prompt_len}")

        decoder_t0 = ThresholdParallelDecoder(temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        decoder_t7 = ThresholdParallelDecoder(temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm(dec):
            return BlockDiffusionLLM(
                model, dec, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Shared mutable state
        state = {
            'chaos_threshold': 0.0,
            'prev_confidence': None,
            'saved_topk_w': {},   # eb_layer_idx -> [N, K] topk_weights
            'skip_counts': [],
            'total_counts': [],
        }

        # Save originals
        orig_routings = {}
        orig_moe_fwds = {}
        moe_layer_map = {}  # model_layer_idx -> eb_layer_idx
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
            """Patch gate.routing (EB + save topk_w) + MoE forward (skip logic)."""
            restore_all()

            for li, layer in enumerate(model.model.layers):
                if not hasattr(layer.mlp, 'gate'):
                    continue
                moe = layer.mlp
                gate = moe.gate
                eb_li = moe_layer_map[li]
                b, r = gate.expert_bias, gate.routed_scaling_factor
                tk, ng, tkg = gate.top_k, gate.n_group, gate.topk_group

                # Patch gate.routing: EB + fused_routing + save topk_w
                def mk_routing(bb, rr, tt, nn, gg, layer_i, cc):
                    def fn(hs, go, topk, renorm):
                        sm = cc.get_s_mask(layer_i, go, bb)
                        w, ids = fused_routing(go, bb, rr, s_mask=sm, K=tt, ng=nn, tkg=gg)
                        state['saved_topk_w'][layer_i] = w.detach()
                        return w.to(go.dtype), ids
                    return fn
                gate.routing = mk_routing(b, r, tk, ng, tkg, eb_li, ctrl)

                # Patch MoE forward: shared + routed (with skip for low-conf tokens)
                def mk_moe_fwd(blk, layer_i):
                    def fwd(hidden_states):
                        # ① shared_experts (all tokens, always)
                        shared_res = blk.shared_experts(hidden_states)

                        bsz, seq_len, h = hidden_states.shape
                        hs_flat = hidden_states.view(-1, h)
                        N = hs_flat.shape[0]

                        # ② gate + routing + fused_experts (via forward_impl)
                        logits = blk.gate.get_logits(hs_flat)
                        routed = blk.experts.forward_impl(
                            hidden_states=hs_flat, router_logits=logits)

                        # ③ MoE skip: replace routed with identity for low-conf tokens
                        th = state['chaos_threshold']
                        prev_conf = state['prev_confidence']
                        if th > 0 and prev_conf is not None and prev_conf.shape[0] == N:
                            low_conf = prev_conf < th  # [N] bool, CPU
                            n_skip = int(low_conf.sum())
                            if n_skip > 0:
                                topk_w = state['saved_topk_w'].get(layer_i)
                                if topk_w is not None:
                                    low_conf_gpu = low_conf.to(device)
                                    weight_sum = topk_w.sum(dim=1, keepdim=True).to(hs_flat.dtype)
                                    identity = hs_flat * weight_sum
                                    routed = torch.where(
                                        low_conf_gpu.unsqueeze(1), identity, routed)

                                # Record stats (once per forward at first MoE layer)
                                if layer_i == 0:
                                    state['skip_counts'].append(n_skip)
                                    state['total_counts'].append(N)

                        routed = routed.view(bsz, seq_len, h)
                        return shared_res + routed
                    return fwd

                moe.forward = mk_moe_fwd(moe, eb_li)

        # LM head hook for confidence
        lm_output = {}
        def lm_hook(mod, inp, out):
            lm_output['logits'] = out.detach()
        lm_handle = model.lm_head.register_forward_hook(lm_hook)

        # Patch iteration forward for confidence tracking
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

        def check_quality(gen_tokens, label):
            n_pass = 0
            for bi, keywords in VERIFIABLE.items():
                gt = gen_tokens[bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                found = any(kw.lower() in text.lower() for kw in keywords)
                n_pass += found
                status = "PASS" if found else "FAIL"
                print(f"    #{bi}: {status} — {text[:150]}")
            total = len(VERIFIABLE)
            print(f"  Quality: {n_pass}/{total} {'PASS' if n_pass == total else 'PARTIAL'}")
            return n_pass, total

        # ============================================================
        # Run experiments
        # ============================================================
        results = OrderedDict()

        for chaos_th in [0.0, 0.3, 0.5]:
            label = f"skip<{chaos_th:.1f}" if chaos_th > 0 else "C10-M∞ baseline"
            print(f"\n{'='*60}")
            print(f"{label}")
            print(f"{'='*60}")

            state['chaos_threshold'] = chaos_th

            # Fresh controller for each config
            ctrl = ColdOnlyEBController(
                num_layers=19, K=8, M=4, K_target=40,
                quality_floor=0.70, q_major=1.0, per_round_cap=8)
            patch_eb_skip(ctrl)

            # Warmup
            reset_state(ctrl)
            dllm = make_dllm(decoder_t0)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            print(f"  Warmup: {dllm.diff_iteration.num_forwards} fwd")

            # Timing run (temp=0)
            reset_state(ctrl)
            dllm = make_dllm(decoder_t0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                out_t0 = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fwd_t0 = dllm.diff_iteration.num_forwards
            time_t0 = t1 - t0
            avg_skip = 0.0
            if state['skip_counts']:
                rates = [s / t for s, t in zip(state['skip_counts'], state['total_counts'])]
                avg_skip = sum(rates) / len(rates)
            print(f"  Timing (temp=0): {time_t0:.3f}s, {fwd_t0} fwd, "
                  f"{time_t0*1000/fwd_t0:.2f} ms/fwd, skip={avg_skip:.1%}")

            # Quality run (temp=0.7)
            reset_state(ctrl)
            dllm = make_dllm(decoder_t7)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                out_t7 = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            fwd_t7 = dllm.diff_iteration.num_forwards
            avg_skip_q = 0.0
            if state['skip_counts']:
                rates = [s / t for s, t in zip(state['skip_counts'], state['total_counts'])]
                avg_skip_q = sum(rates) / len(rates)
            print(f"  Quality run (temp=0.7): {fwd_t7} fwd, skip={avg_skip_q:.1%}")

            gen_tokens = out_t7[:, prompt_len:]
            qp, qt = check_quality(gen_tokens, label)

            results[label] = {
                'chaos_threshold': chaos_th,
                'time_s': time_t0, 'fwd_t0': fwd_t0,
                'ms_per_fwd': time_t0 * 1000 / fwd_t0,
                'avg_skip_rate_t0': avg_skip,
                'fwd_t7': fwd_t7, 'avg_skip_rate_t7': avg_skip_q,
                'quality_pass': qp, 'quality_total': qt,
            }

        # Cleanup
        lm_handle.remove()
        BlockDiffusionIteration.forward = orig_iter_fwd
        restore_all()

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Config':<20s} {'Time(s)':>8s} {'Fwd(t0)':>8s} {'ms/fwd':>8s} "
              f"{'Skip%':>7s} {'Fwd(t7)':>8s} {'Quality':>8s}")
        print(f"  {'-'*70}")
        base_fwd = None
        for label, r in results.items():
            if base_fwd is None:
                base_fwd = r['fwd_t0']
            dfwd = r['fwd_t0'] - base_fwd
            q_str = f"{r['quality_pass']}/{r['quality_total']}"
            print(f"  {label:<20s} {r['time_s']:>8.3f} {r['fwd_t0']:>8d} {r['ms_per_fwd']:>8.2f} "
                  f"{r['avg_skip_rate_t0']:>6.1%} {r['fwd_t7']:>8d} {q_str:>8s}")

        out_path = REPO_ROOT / "codex_coding" / "results" / "mask_chaos_skip_reliable.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
