#!/usr/bin/env python3
"""
Routing Dispersion Analysis: per-layer routing score distribution vs skip safety.

Collects per-token per-layer:
  - max(sigmoid(logits)): strongest expert affinity
  - std(logits): logit spread
  - prev_confidence: LM head confidence from previous forward

Then analyzes: how do these signals correlate? Can routing dispersion
enable per-layer skip decisions?

Also runs per-layer skip sweep: skip routed experts when max_sigmoid < threshold,
independently at each layer.
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
    print("Routing Dispersion Analysis")
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

        # Shared state
        state = {
            'prev_confidence': None,
            # Per-forward accumulation (reset each forward)
            'fwd_records': [],  # list of per-layer dicts
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

        # Aggregated results: per-layer statistics
        # For each layer, across all forwards, collect:
        #   - distribution of max_sigmoid for low-conf tokens vs high-conf tokens
        #   - distribution of std_logits for low-conf tokens vs high-conf tokens
        agg = {
            'by_layer': defaultdict(lambda: {
                'max_sig_all': [], 'std_log_all': [],
                'max_sig_lowconf': [], 'std_log_lowconf': [],
                'max_sig_highconf': [], 'std_log_highconf': [],
                'n_tokens': 0, 'n_lowconf': 0, 'n_highconf': 0,
            }),
            'by_fib': defaultdict(lambda: {
                'max_sig_mean': [], 'std_log_mean': [],
                'conf_mean': [], 'n': 0,
            }),
        }

        global_fwd = [0]

        def patch_collection(ctrl):
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
                        return w.to(go.dtype), ids
                    return fn
                gate.routing = mk_routing(b, r, tk, ng, tkg, eb_li, ctrl)

                def mk_moe_fwd(blk, layer_i):
                    def fwd(hidden_states):
                        shared_res = blk.shared_experts(hidden_states)
                        bsz, seq_len, h = hidden_states.shape
                        hs_flat = hidden_states.view(-1, h)
                        N = hs_flat.shape[0]

                        # Get logits (already computed as part of normal flow)
                        logits = blk.gate.get_logits(hs_flat)  # [N, 256]

                        # Compute dispersion metrics (lightweight)
                        with torch.no_grad():
                            sig = torch.sigmoid(logits.float())  # [N, 256]
                            max_sig = sig.max(dim=1).values      # [N]
                            std_log = logits.float().std(dim=1)  # [N]

                        # Record statistics (sampled to control memory)
                        prev_conf = state['prev_confidence']
                        if prev_conf is not None and prev_conf.shape[0] == N:
                            ms_cpu = max_sig.cpu().numpy()
                            sl_cpu = std_log.cpu().numpy()
                            pc_cpu = prev_conf.numpy()

                            la = agg['by_layer'][layer_i]
                            # Sample up to 2000 tokens per forward per layer
                            n_sample = min(N, 2000)
                            idx_sample = np.random.choice(N, n_sample, replace=False) if N > n_sample else np.arange(N)

                            ms_s = ms_cpu[idx_sample]
                            sl_s = sl_cpu[idx_sample]
                            pc_s = pc_cpu[idx_sample]

                            la['max_sig_all'].extend(ms_s.tolist())
                            la['std_log_all'].extend(sl_s.tolist())
                            la['n_tokens'] += n_sample

                            low_mask = pc_s < 0.3
                            high_mask = pc_s >= 0.7
                            if low_mask.any():
                                la['max_sig_lowconf'].extend(ms_s[low_mask].tolist())
                                la['std_log_lowconf'].extend(sl_s[low_mask].tolist())
                                la['n_lowconf'] += int(low_mask.sum())
                            if high_mask.any():
                                la['max_sig_highconf'].extend(ms_s[high_mask].tolist())
                                la['std_log_highconf'].extend(sl_s[high_mask].tolist())
                                la['n_highconf'] += int(high_mask.sum())

                            # By fwd_in_block (only layer 0, tracked via global counter)
                            if layer_i == 0:
                                fib = global_fwd[0] % 35  # approximate fwd_in_block
                                fb = agg['by_fib'][fib]
                                fb['max_sig_mean'].append(float(ms_cpu.mean()))
                                fb['std_log_mean'].append(float(sl_cpu.mean()))
                                fb['conf_mean'].append(float(pc_cpu.mean()))
                                fb['n'] += 1

                        # Normal forward
                        routed = blk.experts.forward_impl(
                            hidden_states=hs_flat, router_logits=logits)
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
            global_fwd[0] += 1
            return result
        BlockDiffusionIteration.forward = patched_iter_fwd

        decoder = ThresholdParallelDecoder(temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        ctrl = ColdOnlyEBController(num_layers=19, K=8, M=4, K_target=40,
                                     quality_floor=0.70, q_major=1.0, per_round_cap=8)
        patch_collection(ctrl)

        # Warmup
        print("\nWarmup...")
        state['prev_confidence'] = None
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Done: {dllm.diff_iteration.num_forwards} fwd")

        # Reset and collect
        ctrl.prev_N.clear(); ctrl.K_init.clear()
        ctrl.cold_count = 0; ctrl.hot_count = 0
        ctrl._bufs.clear(); ctrl.k_init_history.clear()
        ctrl.s_mask_cache.clear()
        state['prev_confidence'] = None
        global_fwd[0] = 0
        for k in agg['by_layer']:
            for v in agg['by_layer'][k].values():
                if isinstance(v, list): v.clear()
                elif isinstance(v, int): agg['by_layer'][k] = 0
        agg['by_layer'].clear()
        agg['by_fib'].clear()

        print("\nCollecting...")
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

        # Cleanup
        lm_handle.remove()
        BlockDiffusionIteration.forward = orig_iter_fwd
        restore_all()

        # ============================================================
        # Analysis
        # ============================================================
        print(f"\n{'='*80}")
        print("ROUTING DISPERSION ANALYSIS")
        print(f"{'='*80}")

        # 1. By layer: low-conf vs high-conf routing dispersion
        print(f"\n  Per-Layer: max_sigmoid distribution (low-conf<0.3 vs high-conf>0.7)")
        print(f"    {'Layer':>5s} {'All_mean':>10s} {'LowConf':>10s} {'HighConf':>10s} "
              f"{'Gap':>8s} {'N_low':>8s} {'N_high':>8s}")
        print(f"    {'-'*58}")
        layer_summary = {}
        for li in sorted(agg['by_layer'].keys()):
            la = agg['by_layer'][li]
            all_m = np.mean(la['max_sig_all']) if la['max_sig_all'] else 0
            low_m = np.mean(la['max_sig_lowconf']) if la['max_sig_lowconf'] else 0
            high_m = np.mean(la['max_sig_highconf']) if la['max_sig_highconf'] else 0
            gap = high_m - low_m
            print(f"    L{li:>3d} {all_m:>10.4f} {low_m:>10.4f} {high_m:>10.4f} "
                  f"{gap:>+8.4f} {la['n_lowconf']:>8d} {la['n_highconf']:>8d}")

            all_s = np.mean(la['std_log_all']) if la['std_log_all'] else 0
            low_s = np.mean(la['std_log_lowconf']) if la['std_log_lowconf'] else 0
            high_s = np.mean(la['std_log_highconf']) if la['std_log_highconf'] else 0
            layer_summary[li] = {
                'max_sig': {'all': all_m, 'low': low_m, 'high': high_m, 'gap': gap},
                'std_log': {'all': all_s, 'low': low_s, 'high': high_s, 'gap': high_s - low_s},
                'n_low': la['n_lowconf'], 'n_high': la['n_highconf'],
            }

        # 2. Same for std_logits
        print(f"\n  Per-Layer: std(logits) distribution")
        print(f"    {'Layer':>5s} {'All_mean':>10s} {'LowConf':>10s} {'HighConf':>10s} {'Gap':>8s}")
        print(f"    {'-'*45}")
        for li in sorted(agg['by_layer'].keys()):
            ls = layer_summary[li]['std_log']
            print(f"    L{li:>3d} {ls['all']:>10.4f} {ls['low']:>10.4f} "
                  f"{ls['high']:>10.4f} {ls['gap']:>+8.4f}")

        # 3. Distribution percentiles for max_sigmoid (across all layers)
        all_max_sig_low = []
        all_max_sig_high = []
        for li in agg['by_layer']:
            la = agg['by_layer'][li]
            all_max_sig_low.extend(la['max_sig_lowconf'])
            all_max_sig_high.extend(la['max_sig_highconf'])

        if all_max_sig_low and all_max_sig_high:
            print(f"\n  max_sigmoid percentiles (all layers combined):")
            print(f"    {'Percentile':>12s} {'LowConf<0.3':>12s} {'HighConf>0.7':>12s}")
            print(f"    {'-'*38}")
            for p in [10, 25, 50, 75, 90]:
                lv = np.percentile(all_max_sig_low, p)
                hv = np.percentile(all_max_sig_high, p)
                print(f"    P{p:<11d} {lv:>12.4f} {hv:>12.4f}")

        # 4. By fwd_in_block (temporal evolution)
        print(f"\n  Temporal evolution (L0, by fwd_in_block):")
        print(f"    {'fib':>4s} {'max_sig':>10s} {'std_log':>10s} {'conf':>10s} {'count':>6s}")
        print(f"    {'-'*44}")
        fib_summary = {}
        for fib in sorted(agg['by_fib'].keys()):
            fb = agg['by_fib'][fib]
            ms = np.mean(fb['max_sig_mean']) if fb['max_sig_mean'] else 0
            sl = np.mean(fb['std_log_mean']) if fb['std_log_mean'] else 0
            cf = np.mean(fb['conf_mean']) if fb['conf_mean'] else 0
            print(f"    {fib:>4d} {ms:>10.4f} {sl:>10.4f} {cf:>10.4f} {fb['n']:>6d}")
            fib_summary[fib] = {'max_sig': ms, 'std_log': sl, 'conf': cf}

        # 5. Correlation between confidence and max_sigmoid
        if all_max_sig_low and all_max_sig_high:
            # Simple analysis: what fraction of low-conf tokens have max_sigmoid < various thresholds?
            print(f"\n  Skip eligibility: fraction of tokens with max_sigmoid < threshold")
            print(f"    {'MaxSig_th':>10s} {'LowConf%':>10s} {'HighConf%':>10s} {'Selectivity':>12s}")
            print(f"    {'-'*44}")
            low_arr = np.array(all_max_sig_low)
            high_arr = np.array(all_max_sig_high)
            for mst in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                lf = (low_arr < mst).mean()
                hf = (high_arr < mst).mean()
                sel = lf / max(hf, 0.001)  # selectivity ratio
                print(f"    {mst:>10.1f} {lf:>9.1%} {hf:>9.1%} {sel:>11.1f}x")

        # Save
        save_data = {
            'config': {'batch': BATCH_SIZE, 'gen': GEN_LENGTH, 'total_fwd': total_fwd},
            'layer_summary': {str(k): v for k, v in layer_summary.items()},
            'fib_summary': {str(k): v for k, v in fib_summary.items()},
        }
        out_path = REPO_ROOT / "codex_coding" / "results" / "routing_dispersion_analysis.json"
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
