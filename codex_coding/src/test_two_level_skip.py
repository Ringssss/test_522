#!/usr/bin/env python3
"""
Two-Level MoE Skip: Tier1 (confidence) + Tier2 (per-layer routing dispersion)

Tier 1: prev_conf < T1 → skip ALL layers (identity pass-through)
Tier 2: T1 ≤ prev_conf < T2 → per-layer: skip if max(sigmoid(logits)) < S
Tier 3: prev_conf ≥ T2 → no skip (full computation)

Configs:
  baseline: no skip
  flat_0.5: flat skip<0.5 (reference from previous experiments)
  A (conservative): T1=0.3, T2=0.5, S=0.4
  B (balanced):     T1=0.3, T2=0.6, S=0.4
  C (aggressive):   T1=0.3, T2=0.7, S=0.5

Based on test_heteval128.py architecture. Visual quality check at temp=0.7.
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

VERIFY_INDICES = [0, 8, 13, 19, 28]
VERIFY_LABELS = {0: "Math", 8: "QuadEq", 13: "Logic", 19: "Fib", 28: "Planets"}

# Experiment configs: (label, T1, T2, S)
# T1: full-skip threshold, T2: no-skip threshold, S: per-layer max_sigmoid threshold
# For "flat" mode: T1=threshold, T2=threshold, S=0 (no per-layer check)
CONFIGS = [
    ("baseline",   0.0, 0.0, 0.0),
    ("flat<0.5",   0.5, 0.5, 0.0),   # flat skip: all layers skip if conf < 0.5
    ("A_conserv",  0.3, 0.5, 0.4),    # T1=0.3, T2=0.5, S=0.4
    ("B_balanced", 0.3, 0.6, 0.4),    # T1=0.3, T2=0.6, S=0.4
    ("C_aggress",  0.3, 0.7, 0.5),    # T1=0.3, T2=0.7, S=0.5
]


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
    print("Two-Level MoE Skip: Tier1(conf) + Tier2(routing dispersion)")
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
            'prev_confidence': None,
            'T1': 0.0, 'T2': 0.0, 'S': 0.0,
            'saved_topk_w': {},
            # Stats per forward
            'tier1_skips': [],   # count of tier1 full-skip tokens
            'tier2_skips': [],   # count of tier2 layer-skips (sum across layers)
            'tier2_computed': [], # count of tier2 normal-compute (sum across layers)
            'total_tokens': [],
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

        def patch_two_level(ctrl):
            restore_all()
            # Per-forward accumulators (reset each forward by layer 0)
            fwd_stats = {'t1': 0, 't2_skip': 0, 't2_compute': 0, 'N': 0}

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

                def mk_moe_fwd(blk, layer_i, fwd_s):
                    def fwd(hidden_states):
                        shared_res = blk.shared_experts(hidden_states)
                        bsz, seq_len, h = hidden_states.shape
                        hs_flat = hidden_states.view(-1, h)
                        N = hs_flat.shape[0]

                        logits = blk.gate.get_logits(hs_flat)  # [N, 256]

                        # Normal routing + fused_experts
                        routed = blk.experts.forward_impl(
                            hidden_states=hs_flat, router_logits=logits)

                        T1, T2, S = state['T1'], state['T2'], state['S']
                        prev_conf = state['prev_confidence']

                        if T1 > 0 and prev_conf is not None and prev_conf.shape[0] == N:
                            topk_w = state['saved_topk_w'].get(layer_i)
                            if topk_w is not None:
                                # Compute identity pass-through
                                weight_sum = topk_w.sum(dim=1, keepdim=True).to(hs_flat.dtype)
                                identity = hs_flat * weight_sum  # [N, h]

                                # Tier 1: conf < T1 → full skip (all layers)
                                tier1_mask = prev_conf < T1  # [N] CPU bool

                                # Tier 2: T1 ≤ conf < T2 → check per-layer routing
                                tier2_candidates = (prev_conf >= T1) & (prev_conf < T2)

                                if S > 0 and tier2_candidates.any():
                                    # Compute max_sigmoid for tier2 candidates
                                    with torch.no_grad():
                                        max_sig = torch.sigmoid(logits.float()).max(dim=1).values  # [N]
                                        max_sig_cpu = max_sig.cpu()
                                    # Tier2 skip: routing is flat at this layer
                                    tier2_skip = tier2_candidates & (max_sig_cpu < S)
                                else:
                                    tier2_skip = torch.zeros(N, dtype=torch.bool)

                                # Combined skip mask
                                skip_mask = tier1_mask | tier2_skip

                                n_skip = int(skip_mask.sum())
                                if n_skip > 0:
                                    skip_gpu = skip_mask.to(device)
                                    routed = torch.where(skip_gpu.unsqueeze(1), identity, routed)

                                # Record stats (at first MoE layer per forward)
                                if layer_i == 0:
                                    fwd_s['N'] = N
                                    fwd_s['t1'] = int(tier1_mask.sum())
                                    fwd_s['t2_skip'] = 0
                                    fwd_s['t2_compute'] = 0

                                # Accumulate tier2 stats across layers
                                if S > 0:
                                    n_t2_skip = int(tier2_skip.sum())
                                    n_t2_compute = int(tier2_candidates.sum()) - n_t2_skip
                                    fwd_s['t2_skip'] += n_t2_skip
                                    fwd_s['t2_compute'] += n_t2_compute

                                # At last MoE layer, flush stats
                                if layer_i == 18:
                                    state['tier1_skips'].append(fwd_s['t1'])
                                    state['tier2_skips'].append(fwd_s['t2_skip'])
                                    state['tier2_computed'].append(fwd_s['t2_compute'])
                                    state['total_tokens'].append(fwd_s['N'])

                        routed = routed.view(bsz, seq_len, h)
                        return shared_res + routed
                    return fwd
                moe.forward = mk_moe_fwd(moe, eb_li, fwd_stats)

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
            state['tier1_skips'] = []; state['tier2_skips'] = []
            state['tier2_computed'] = []; state['total_tokens'] = []

        # Global warmup
        print("\nGlobal warmup...")
        ctrl_w = ColdOnlyEBController(num_layers=19, K=8, M=4, K_target=40,
                                       quality_floor=0.70, q_major=1.0, per_round_cap=8)
        state['T1'] = 0.0; state['T2'] = 0.0; state['S'] = 0.0
        patch_two_level(ctrl_w)
        reset_state(ctrl_w)
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Done: {dllm.diff_iteration.num_forwards} fwd")

        # Run all configs
        results = OrderedDict()

        for label, T1, T2, S in CONFIGS:
            print(f"\n{'='*60}")
            print(f"  {label} (T1={T1}, T2={T2}, S={S})")
            print(f"{'='*60}")

            state['T1'] = T1; state['T2'] = T2; state['S'] = S
            ctrl = ColdOnlyEBController(num_layers=19, K=8, M=4, K_target=40,
                                         quality_floor=0.70, q_major=1.0, per_round_cap=8)
            patch_two_level(ctrl)

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

            # Compute skip stats
            tier1_rate = 0; tier2_skip_rate = 0; tier2_compute_rate = 0; total_skip_rate = 0
            if state['total_tokens']:
                n_layers = 19
                tier1_total = sum(state['tier1_skips'])  # tokens (applied to all layers)
                tier2_skip_total = sum(state['tier2_skips'])  # token-layers
                tier2_compute_total = sum(state['tier2_computed'])  # token-layers
                token_total = sum(state['total_tokens'])

                tier1_rate = tier1_total / token_total if token_total > 0 else 0
                # Total token-layer operations: token_total * 19
                total_token_layers = token_total * n_layers
                # Tier1 skips all 19 layers
                tier1_layer_skips = tier1_total * n_layers
                # Total skipped token-layers
                total_skipped = tier1_layer_skips + tier2_skip_total
                total_skip_rate = total_skipped / total_token_layers if total_token_layers > 0 else 0
                tier2_skip_rate = tier2_skip_total / total_token_layers if total_token_layers > 0 else 0

            print(f"  temp=0: {time_t0:.3f}s, {fwd_t0} fwd, {time_t0*1000/fwd_t0:.2f} ms/fwd")
            print(f"  Tier1 (full skip): {tier1_rate:.1%} tokens")
            print(f"  Tier2 (layer skip): {tier2_skip_rate:.1%} token-layers")
            print(f"  Total skip: {total_skip_rate:.1%} token-layers")

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
                'T1': T1, 'T2': T2, 'S': S,
                'time_s': time_t0, 'fwd_t0': fwd_t0,
                'ms_per_fwd': time_t0 * 1000 / fwd_t0,
                'tier1_rate': tier1_rate,
                'tier2_skip_rate': tier2_skip_rate,
                'total_skip_rate': total_skip_rate,
                'fwd_t7': fwd_t7,
            }

        # Cleanup
        lm_handle.remove()
        BlockDiffusionIteration.forward = orig_iter_fwd
        restore_all()

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Config':<12s} {'T1':>4s} {'T2':>4s} {'S':>4s} {'Time':>7s} {'Fwd0':>5s} "
              f"{'ms/f':>6s} {'T1%':>6s} {'T2%':>6s} {'Tot%':>6s} {'Fwd7':>5s}")
        print(f"  {'-'*72}")
        for label, r in results.items():
            print(f"  {label:<12s} {r['T1']:>4.1f} {r['T2']:>4.1f} {r['S']:>4.1f} "
                  f"{r['time_s']:>7.3f} {r['fwd_t0']:>5d} {r['ms_per_fwd']:>6.2f} "
                  f"{r['tier1_rate']:>5.1%} {r['tier2_skip_rate']:>5.1%} "
                  f"{r['total_skip_rate']:>5.1%} {r['fwd_t7']:>5d}")

        out_path = REPO_ROOT / "codex_coding" / "results" / "two_level_skip_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
