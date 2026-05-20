#!/usr/bin/env python3
"""
Minimal torch.compile test for LLaDA2MoE on current env.
Tests: torch.compile(model.forward, mode='reduce-overhead') feasibility.

Usage:
  torchrun --nproc_per_node=8 test_torch_compile.py
"""

import os
import sys
import time
import torch
import torch.distributed as dist

MODEL_PATH = "/mnt/models/GSAI-ML/LLaDA-2-mini"


def main():
    from vllm.config import ParallelConfig, VllmConfig
    from vllm.model_executor.model_loader.utils import set_default_torch_dtype
    from vllm import distributed as vllm_dist
    from vllm.config import set_current_vllm_config

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group("nccl")

    tp_size = 4
    dp_size = world // tp_size
    dp_rank = rank // tp_size

    pcfg = ParallelConfig(
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        enable_expert_parallel=True,
        enable_eplb=False,
    )
    vllm_cfg = VllmConfig(parallel_config=pcfg)

    with set_current_vllm_config(vllm_cfg):
        vllm_dist.init_distributed_environment(
            world, rank, "env://", local_rank, "nccl"
        )
        vllm_dist.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, backend="nccl"
        )

    if rank == 0:
        print(f"torch={torch.__version__}, device={device}")
        print(f"tp={tp_size}, dp={dp_size}, ep={world}")

    from transformers import AutoTokenizer, AutoConfig
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeModelLM

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    with set_default_torch_dtype(torch.bfloat16):
        with set_current_vllm_config(vllm_cfg):
            model = LLaDA2MoeModelLM(config).to(device)

    if rank == 0:
        print("Model loaded. Testing eager forward...")

    # Test eager forward
    x = torch.randint(0, 1000, (2, 64), device=device)
    with torch.inference_mode():
        out = model(x, use_cache=False)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"  Eager forward OK. logits shape={out.logits.shape}")

    # Test torch.compile
    if rank == 0:
        print("\nTesting torch.compile(mode='reduce-overhead')...")

    try:
        compiled_forward = torch.compile(
            model.forward,
            mode='reduce-overhead',
            fullgraph=False,
            dynamic=True,
        )
        model.forward = compiled_forward
        if rank == 0:
            print("  torch.compile() returned successfully.")
    except Exception as e:
        if rank == 0:
            print(f"  torch.compile() FAILED: {type(e).__name__}: {e}")
        dist.barrier()
        dist.destroy_process_group()
        return

    # Run compiled forward (triggers actual compilation + graph capture)
    if rank == 0:
        print("  Running first compiled forward (JIT compile)...")

    try:
        with torch.inference_mode():
            out = model(x, use_cache=False)
        torch.cuda.synchronize()
        if rank == 0:
            print(f"  First compiled forward OK. logits shape={out.logits.shape}")
    except Exception as e:
        if rank == 0:
            print(f"  First compiled forward FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        dist.barrier()
        dist.destroy_process_group()
        return

    # Run second compiled forward (should use cached graph)
    if rank == 0:
        print("  Running second compiled forward (graph replay)...")

    try:
        with torch.inference_mode():
            t0 = time.perf_counter()
            out = model(x, use_cache=False)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
        if rank == 0:
            print(f"  Second compiled forward OK. {dt:.1f} ms")
    except Exception as e:
        if rank == 0:
            print(f"  Second compiled forward FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        dist.barrier()
        dist.destroy_process_group()
        return

    # Compare eager vs compiled timing
    if rank == 0:
        print("\n  Benchmarking: eager vs compiled (5 runs each)...")

    # Reload model for eager comparison
    # Actually just time the compiled version since model is already compiled
    times_compiled = []
    with torch.inference_mode():
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x, use_cache=False)
            torch.cuda.synchronize()
            times_compiled.append((time.perf_counter() - t0) * 1000)

    if rank == 0:
        median = sorted(times_compiled)[2]
        print(f"  Compiled: {median:.2f} ms (median of 5)")
        print("\n=== torch.compile test PASSED ===")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
