#!/usr/bin/env python3
"""
v0.1.15.8n — ncu/nsys Profiling for fused_experts (C10-M5)

Real E2E profiling with HetEval-128, batch=128, gen_length=64 (2 blocks).
Uses cudaProfilerStart/Stop to bracket the profiled region.

Usage:
  # Step 0: nsys quick scan to identify kernel names
  CUDA_VISIBLE_DEVICES=4 nsys profile --stats=true \
    -o /tmp/ncu_kernelnames \
    /home/wuhang/miniconda3/envs/dllm/bin/python \
    /home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_ncu_fused_experts.py

  # Step 1: ncu profiling (after identifying kernel names)
  sudo /usr/local/cuda-13.0/bin/ncu \
    --set basic --profile-from-start off \
    --kernel-name "regex:<pattern>" \
    -o codex_coding/results/ncu_fused_experts_c10m5 \
    /home/wuhang/miniconda3/envs/dllm/bin/python \
    codex_coding/src/bench_ncu_fused_experts.py
"""

from __future__ import annotations
import os, sys, time, socket
from pathlib import Path

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
GEN_LENGTH = 64   # 2 blocks
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
    from transformers import AutoTokenizer, AutoConfig

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print("ncu/nsys Profiling: fused_experts C10-M5")
    print(f"  batch={BATCH_SIZE}, gen_length={GEN_LENGTH}, block_length={BLOCK_LENGTH}")
    print(f"  M=5, q_major=1.0, K_target=40")
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

        print("\nApplying baseline optimizations...")
        n_rms, n_fa = apply_all_optimizations(model)
        print(f"  Fused RMSNorm: {n_rms}, Flash-attn: {n_fa}")

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

        # Patch routing with C10-M5
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

        # Use FusedEBController with M=5 for real hot/cold dual-path behavior
        ctrl = FusedEBController(
            num_layers=19, K=8, M=5, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8)
        patch_eb(ctrl)

        # --- Warmup: full generation to JIT all Triton kernels ---
        print("\nWarmup (gen_length=64, JIT compilation)...")
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd, "
              f"cold={ctrl.cold_count}, hot={ctrl.hot_count}")

        # Reset controller state for profiling run
        ctrl.prev_N.clear(); ctrl.K_init.clear()
        ctrl.cold_count = 0; ctrl.hot_count = 0
        ctrl._bufs.clear()

        # --- Profiled run ---
        print("\n" + "=" * 80)
        print("PROFILED RUN — cudaProfilerStart/Stop bracketed")
        print("=" * 80)

        dllm = make_dllm()
        torch.cuda.synchronize()

        # Start profiler
        torch.cuda.cudart().cudaProfilerStart()

        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Stop profiler
        torch.cuda.cudart().cudaProfilerStop()

        num_fwd = dllm.diff_iteration.num_forwards
        print(f"\n  Time: {t1-t0:.3f}s")
        print(f"  Forwards: {num_fwd}")
        print(f"  ms/fwd: {(t1-t0)*1000/num_fwd:.2f}")
        print(f"  EB cold: {ctrl.cold_count}, hot: {ctrl.hot_count}")
        print(f"\n--- PROFILING COMPLETE ---")
        print(f"  Use nsys/ncu to analyze the captured region.")


if __name__ == "__main__":
    main()
