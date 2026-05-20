#!/usr/bin/env python3
"""
v0.1.15.8n — Tiling Config A/B Test for fused_moe_kernel

Tests 3 tiling configs on C10-M5 E2E (HetEval-128, gen_length=256):
  Baseline: (128, 256, 64, GROUP_SIZE_M=16, 8warps, 4stages) — original
  Alt-A:    (64,  256, 64, GROUP_SIZE_M=16, 8warps, 3stages) — smaller tile, same occ
  Alt-B:    (64,  256, 64, GROUP_SIZE_M=16, 8warps, 2stages) — smaller tile, target 25% occ

Uses monkey-patch on vllm's config lookup instead of modifying system files.
"""

from __future__ import annotations
import os, sys, time, socket, json, shutil, functools, importlib
from pathlib import Path

import torch

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
GEN_LENGTH = 256
BATCH_SIZE = 128

# Three configs to test
CONFIGS = {
    "Baseline": {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 4
    },
    "A2_128x128_w8s3": {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16, "num_warps": 8, "num_stages": 3
    },
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
    from transformers import AutoTokenizer, AutoConfig

    # Import the module for patching config lookup
    fused_moe_module = importlib.import_module('vllm.model_executor.layers.fused_moe.fused_moe')

    from test_fused_eb_triton import fused_routing, FusedEBController
    from test_m_skip_sweep import MSkipEBController
    from baseline_optimizations import apply_all_optimizations
    from test_heteval128 import PROMPTS

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print("Tiling Config A/B Test — C10-M5 E2E")
    print(f"  batch={BATCH_SIZE}, gen={GEN_LENGTH}, block={BLOCK_LENGTH}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0),
                      use_cache=False)
        apply_all_optimizations(model)

        # Build input
        all_ids = []
        for i in range(BATCH_SIZE):
            text = PROMPTS[i]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=False)
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                  if ids.shape[0] < mx else ids for ids in all_ids]
        input_ids = torch.stack(padded, dim=0).to(device)
        print(f"  Input shape: {input_ids.shape}")

        decoder = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True,
                inplace_cache_update=True)

        def patch_eb(ctrl):
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
            ctrl.eb_calls = 0; ctrl.eb_skips = 0
            ctrl._bufs.clear(); ctrl.k_init_history.clear()
            ctrl.s_mask_cache.clear(); ctrl.pop_cache.clear()
            ctrl._fwd_in_block.clear(); ctrl._block_idx.clear()

        # Save the original try_get_optimal_moe_config
        orig_try_get = fused_moe_module.try_get_optimal_moe_config

        # Active override config (set per test)
        active_override = [None]  # mutable container for closure

        def patched_try_get(w1_shape, w2_shape, top_k, dtype, M, **kwargs):
            """Intercept config lookup: for large M (>= 4096), use our override."""
            if active_override[0] is not None and M >= 4096:
                return active_override[0]
            return orig_try_get(w1_shape, w2_shape, top_k, dtype, M, **kwargs)

        results = {}

        for cfg_name, cfg_dict in CONFIGS.items():
            print(f"\n{'='*60}")
            print(f"  Config: {cfg_name}")
            print(f"  BLOCK_M={cfg_dict['BLOCK_SIZE_M']}, BLOCK_N={cfg_dict['BLOCK_SIZE_N']}, "
                  f"BLOCK_K={cfg_dict['BLOCK_SIZE_K']}, warps={cfg_dict['num_warps']}, "
                  f"stages={cfg_dict['num_stages']}")
            print(f"{'='*60}")

            # Set override
            active_override[0] = cfg_dict

            # Monkey-patch the config lookup
            fused_moe_module.try_get_optimal_moe_config = patched_try_get

            # Clear Triton cache to force recompile with new config
            triton_cache = Path.home() / ".triton" / "cache"
            if triton_cache.exists():
                shutil.rmtree(triton_cache)
                print(f"  Cleared Triton cache")

            # Setup controller
            ctrl = MSkipEBController(
                num_layers=19, K=8, M=4, K_target=40,
                quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)
            patch_eb(ctrl)

            # Warmup (JIT compile new kernel)
            print(f"  Warmup...")
            reset_ctrl(ctrl)
            dllm = make_dllm()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd")

            # Timed runs (2 runs)
            times, fwds = [], []
            for ri in range(2):
                reset_ctrl(ctrl)
                dllm = make_dllm()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)
                fwds.append(dllm.diff_iteration.num_forwards)
                print(f"    Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd, "
                      f"{(t1-t0)*1000/dllm.diff_iteration.num_forwards:.2f} ms/fwd")

            avg_time = sum(times) / len(times)
            avg_fwd = sum(fwds) / len(fwds)
            results[cfg_name] = {
                'config': cfg_dict,
                'avg_time': avg_time,
                'avg_fwd': avg_fwd,
                'ms_per_fwd': avg_time * 1000 / avg_fwd,
                'times': times,
                'fwds': fwds,
            }

        # Restore original config function
        fused_moe_module.try_get_optimal_moe_config = orig_try_get
        active_override[0] = None
        print(f"\n  Original config lookup restored.")

        # Summary
        print(f"\n{'='*80}")
        print(f"SUMMARY — Tiling Config A/B Test (C10-M5)")
        print(f"{'='*80}")
        baseline_time = results.get('Baseline', {}).get('avg_time', 1)
        print(f"  {'Config':<22s} {'Time(s)':>8s} {'Fwd':>5s} {'ms/fwd':>8s} {'vs Base':>8s}")
        print(f"  {'-'*54}")
        for name, r in results.items():
            delta = (r['avg_time'] - baseline_time) / baseline_time * 100
            print(f"  {name:<22s} {r['avg_time']:>8.3f} {r['avg_fwd']:>5.0f} "
                  f"{r['ms_per_fwd']:>8.2f} {delta:>+7.1f}%")

        # Save results
        out_path = REPO_ROOT / "codex_coding" / "results" / "tiling_config_ab_test.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
