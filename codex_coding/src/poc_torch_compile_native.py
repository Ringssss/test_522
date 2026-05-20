#!/usr/bin/env python3
"""
POC: Test dInfer native torch.compile + cudagraph path.

Follows eval_dinfer.py's pattern exactly:
  1. Load model
  2. torch.compile(model.forward, mode='reduce-overhead')
  3. Warmup via dllm.generate() on bucket sizes
  4. Measure generate speed with and without compile

Tests progressively:
  G1: Eager baseline (no compile)
  G2: torch.compile(mode='reduce-overhead') + bucket warmup
  G3: Timing comparison

Usage (single GPU first):
  python poc_torch_compile_native.py --tp-size 1 --batch-size 1

  torchrun --nproc_per_node=8 poc_torch_compile_native.py --tp-size 4 --batch-size 64
"""

import argparse
import json
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
MASK_ID = 126336
EOS_ID = 156892
BLOCK_LENGTH = 32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gen-length", type=int, default=64)
    p.add_argument("--num-runs", type=int, default=3)
    p.add_argument("--results-suffix", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    tp_size = args.tp_size
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    # Standard init: tp=4, dp=2, ep=8 (same as bench script)
    dp_size = world_size // tp_size
    dp_rank = rank // tp_size

    pcfg_init = ParallelConfig(
        tensor_parallel_size=1, data_parallel_size=1,
        enable_expert_parallel=True,
    )
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(
            world_size, rank, "env://", local_rank, "nccl"
        )

    parallel_config = ParallelConfig(
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        enable_expert_parallel=True,
    )
    vllm_cfg = VllmConfig(parallel_config=parallel_config)
    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(tp_size, backend="nccl")

    from vllm.config import get_current_vllm_config
    from vllm.distributed import prepare_communication_buffer_for_model
    from dinfer import (
        BlockDiffusionLLM, BlockIteratorFactory,
        KVCacheFactory, ThresholdParallelDecoder,
    )
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoConfig

    if rank == 0:
        print("=" * 70)
        print(f"torch.compile Native Path POC")
        print(f"  world={world_size}, tp={tp_size}, batch={args.batch_size}")
        print(f"  gen={args.gen_length}, block={BLOCK_LENGTH}")
        print("=" * 70)

    # ---- Load model (dInfer native pattern) ----
    with set_current_vllm_config(vllm_cfg):
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        )
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        # Warmup eager forward
        x_warm = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
        from vllm.forward_context import set_forward_context
        with torch.inference_mode():
            with set_forward_context(
                attn_metadata=None, vllm_config=vllm_cfg,
                num_tokens=x_warm.numel(),
            ):
                _ = model(x_warm, use_cache=False)

        prepare_communication_buffer_for_model(model)

    if rank == 0:
        print(f"  Model loaded. GPU mem: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

    # ---- Build dllm + decoder ----
    decoder = ThresholdParallelDecoder(
        temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID,
    )

    def make_dllm():
        return BlockDiffusionLLM(
            model, decoder,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True,
            backend="vllm",
        )

    # ---- Build test input ----
    prompt_len = 64
    test_input = torch.randint(
        0, config.vocab_size, (args.batch_size, prompt_len),
        dtype=torch.long, device=device
    )
    # Replace last tokens with MASK to simulate generation
    test_input[:, -args.gen_length:] = MASK_ID

    results = {
        "rank": rank,
        "world_size": world_size,
        "tp_size": tp_size,
        "batch_size": args.batch_size,
        "gen_length": args.gen_length,
        "prompt_len": prompt_len,
    }

    # ==== G1: Eager baseline ====
    if rank == 0:
        print(f"\n{'='*70}")
        print("G1: Eager baseline (no compile)")
        print(f"{'='*70}")

    with set_current_vllm_config(vllm_cfg):
        eager_times = []
        for ri in range(args.num_runs):
            dllm = make_dllm()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = dllm.generate(
                    test_input.clone(),
                    gen_length=args.gen_length,
                    block_length=BLOCK_LENGTH,
                )
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            eager_times.append(dt)
            nfwd = dllm.num_forwards
            if rank == 0:
                print(f"  Run {ri+1}: {dt:.1f} ms, {nfwd} forwards, "
                      f"{dt/nfwd:.2f} ms/fwd")

    eager_median = sorted(eager_times)[len(eager_times) // 2]
    results["g1_eager_times_ms"] = eager_times
    results["g1_eager_median_ms"] = eager_median
    results["g1_num_forwards"] = nfwd

    if rank == 0:
        print(f"  Eager median: {eager_median:.1f} ms total, "
              f"{eager_median/nfwd:.2f} ms/fwd")

    # ==== G2: torch.compile + cudagraph ====
    if rank == 0:
        print(f"\n{'='*70}")
        print("G2: torch.compile(mode='reduce-overhead') + bucket warmup")
        print(f"{'='*70}")

    compile_ok = False
    with set_current_vllm_config(vllm_cfg):
        try:
            if rank == 0:
                print("  Applying torch.compile...")
            model.forward = torch.compile(
                model.forward,
                mode='reduce-overhead',
                fullgraph=False,
                dynamic=True,
            )
            if rank == 0:
                print("  torch.compile applied OK.")

            # Bucket warmup (same as dInfer's warmup_cudagraph)
            # Compute buckets based on our test input
            total_len = ((prompt_len + args.gen_length + BLOCK_LENGTH - 1)
                         // BLOCK_LENGTH * BLOCK_LENGTH)
            buckets = list(range(BLOCK_LENGTH, total_len + 1, BLOCK_LENGTH))
            if rank == 0:
                print(f"  Warming up {len(buckets)} buckets: {buckets}")

            for bi, bucket_len in enumerate(buckets):
                warmup_input = torch.randint(
                    0, config.vocab_size, (args.batch_size, max(1, bucket_len - args.gen_length)),
                    dtype=torch.long, device=device
                )
                dllm = make_dllm()
                with torch.inference_mode():
                    _ = dllm.generate(
                        warmup_input,
                        gen_length=args.gen_length,
                        block_length=BLOCK_LENGTH,
                    )
                torch.cuda.synchronize()
                if rank == 0:
                    print(f"    Bucket {bi+1}/{len(buckets)} "
                          f"(total_len={bucket_len}) OK")

            compile_ok = True
            if rank == 0:
                print("  Warmup complete!")

        except Exception as e:
            if rank == 0:
                print(f"  FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()

    results["g2_compile_ok"] = compile_ok

    if not compile_ok:
        results["conclusion"] = "torch.compile failed"
        _save_and_exit(results, args, rank)
        return

    # ==== G3: Compiled timing ====
    if rank == 0:
        print(f"\n{'='*70}")
        print("G3: Compiled generate timing")
        print(f"{'='*70}")

    with set_current_vllm_config(vllm_cfg):
        compiled_times = []
        for ri in range(args.num_runs):
            dllm = make_dllm()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out_compiled = dllm.generate(
                    test_input.clone(),
                    gen_length=args.gen_length,
                    block_length=BLOCK_LENGTH,
                )
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            compiled_times.append(dt)
            nfwd_c = dllm.num_forwards
            if rank == 0:
                print(f"  Run {ri+1}: {dt:.1f} ms, {nfwd_c} forwards, "
                      f"{dt/nfwd_c:.2f} ms/fwd")

    compiled_median = sorted(compiled_times)[len(compiled_times) // 2]
    saved = eager_median - compiled_median
    speedup = eager_median / compiled_median if compiled_median > 0 else 0

    results["g3_compiled_times_ms"] = compiled_times
    results["g3_compiled_median_ms"] = compiled_median
    results["g3_saved_ms"] = saved
    results["g3_speedup"] = speedup
    results["g3_num_forwards"] = nfwd_c

    if rank == 0:
        print(f"\n  Compiled median: {compiled_median:.1f} ms total, "
              f"{compiled_median/nfwd_c:.2f} ms/fwd")
        print(f"  Saved: {saved:.1f} ms ({saved/eager_median*100:.1f}%)")
        print(f"  Speedup: {speedup:.2f}x")

    # ==== Summary ====
    if rank == 0:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  Compile: {'OK' if compile_ok else 'FAILED'}")
        print(f"  Eager:    {eager_median:.1f} ms ({eager_median/nfwd:.2f} ms/fwd)")
        print(f"  Compiled: {compiled_median:.1f} ms ({compiled_median/nfwd_c:.2f} ms/fwd)")
        print(f"  Saved:    {saved:.1f} ms ({saved/eager_median*100:.1f}%)")

    _save_and_exit(results, args, rank)


def _save_and_exit(results, args, rank):
    if rank == 0:
        suffix = f"_{args.results_suffix}" if args.results_suffix else ""
        out_path = os.path.join(
            os.path.dirname(__file__), "..", "results",
            f"poc_torch_compile_native{suffix}.json"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {out_path}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
