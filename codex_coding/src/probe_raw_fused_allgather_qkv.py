#!/usr/bin/env python3
"""Benchmark raw symm_mem fused AllGather+QKV GEMM on the target shape."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
DINFER_PYTHON = REPO_ROOT / "lib_cite/dInfer/python"
if str(DINFER_PYTHON) not in sys.path:
    sys.path.insert(0, str(DINFER_PYTHON))


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a - b).float().abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _all_gather_dict(local: dict[str, Any]) -> list[dict[str, Any]]:
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    return gathered


def _time_cuda(fn: Callable[[], torch.Tensor],
               warmup: int,
               iters: int) -> dict[str, float]:
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start

    checksum = float(out.float().mean().item())
    return {
        "total_s": elapsed_s,
        "mean_ms": elapsed_s * 1000.0 / iters,
        "checksum": checksum,
    }


def _rank_stats(items: list[dict[str, Any]], timing_key: str) -> dict[str, float]:
    values = [float(item["timing"][timing_key]["mean_ms"]) for item in items]
    return {
        "min_ms": min(values),
        "avg_ms": sum(values) / len(values),
        "max_ms": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--qkv-out-hidden", type=int, default=768)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--split-chunks", type=int, default=1)
    parser.add_argument("--return-a", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size % args.tp_size != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by tp_size={args.tp_size}"
        )
    if args.tokens % args.tp_size != 0:
        raise ValueError("--tokens must be divisible by --tp-size")
    if args.split_chunks < 1:
        raise ValueError("--split-chunks must be >= 1")
    if (args.tokens // args.tp_size) % args.split_chunks != 0:
        raise ValueError("--tokens / --tp-size must be divisible by --split-chunks")

    dp_size = world_size // args.tp_size
    dp_rank = rank // args.tp_size
    tp_rank = rank % args.tp_size
    local_tokens = args.tokens // args.tp_size

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    from torch.distributed._symmetric_memory import (
        enable_symm_mem_for_group,
        is_symm_mem_enabled_for_group,
    )
    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather

    pcfg_init = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=True,
    )
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(
            world_size, rank, "env://", local_rank, "nccl")

    pcfg = ParallelConfig(
        tensor_parallel_size=args.tp_size,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        enable_expert_parallel=True,
        rank=rank,
    )
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        vllm_dist.initialize_model_parallel(
            tensor_model_parallel_size=args.tp_size,
            backend="nccl",
        )

        tp_group = get_tp_group()
        group_name = tp_group.device_group.group_name
        enable_symm_mem_for_group(group_name)

        torch.manual_seed(1234 + dp_rank)
        x_full = torch.randn(
            args.tokens, args.hidden, device=device, dtype=torch.bfloat16)
        x_sp = x_full[
            tp_rank * local_tokens:(tp_rank + 1) * local_tokens
        ].contiguous()
        torch.manual_seed(4321 + rank)
        weight = (
            torch.randn(
                args.hidden,
                args.qkv_out_hidden,
                device=device,
                dtype=torch.bfloat16,
            ) * 0.02
        ).contiguous()

        x_sp_chunks = list(x_sp.chunk(args.split_chunks, dim=0))

        def eager_one(chunk: torch.Tensor) -> torch.Tensor:
            hidden = tensor_model_parallel_all_gather(chunk, dim=0)
            return torch.ops.aten.mm.default(hidden, weight)

        def fused_one(chunk: torch.Tensor) -> torch.Tensor:
            _, outs = torch.ops.symm_mem.fused_all_gather_matmul(
                chunk,
                [weight],
                gather_dim=0,
                group_name=group_name,
                return_A=args.return_a,
            )
            return outs[0]

        def eager_fn() -> torch.Tensor:
            if args.split_chunks == 1:
                return eager_one(x_sp)
            return torch.cat([eager_one(chunk) for chunk in x_sp_chunks], dim=0)

        def raw_fused_fn() -> torch.Tensor:
            if args.split_chunks == 1:
                return fused_one(x_sp)
            return torch.cat([fused_one(chunk) for chunk in x_sp_chunks], dim=0)

        def eager_full_fn() -> torch.Tensor:
            hidden = tensor_model_parallel_all_gather(x_sp, dim=0)
            return torch.ops.aten.mm.default(hidden, weight)

        local_result: dict[str, Any] = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "tp_size": args.tp_size,
            "tp_rank": tp_rank,
            "dp_size": dp_size,
            "dp_rank": dp_rank,
            "tokens": args.tokens,
            "local_tokens": local_tokens,
            "hidden": args.hidden,
            "qkv_out_hidden": args.qkv_out_hidden,
            "split_chunks": args.split_chunks,
            "chunk_local_tokens": local_tokens // args.split_chunks,
            "chunk_global_tokens": args.tokens // args.split_chunks,
            "native_async_env": os.environ.get(
                "TORCH_SYMM_MEM_ENABLE_NATIVE_ASYNC_TP"),
            "dtype": str(x_sp.dtype),
            "group_name": group_name,
            "return_a": args.return_a,
            "symm_mem_enabled": is_symm_mem_enabled_for_group(group_name),
            "has_fused_all_gather_matmul": hasattr(
                torch.ops.symm_mem, "fused_all_gather_matmul"),
            "shapes": {
                "x_sp": list(x_sp.shape),
                "x_sp_chunk": list(x_sp_chunks[0].shape),
                "weight": list(weight.shape),
                "output": [args.tokens, args.qkv_out_hidden],
            },
        }

        with torch.inference_mode():
            torch.cuda.synchronize()
            eager = eager_fn()
            fused = raw_fused_fn()
            torch.cuda.synchronize()
            diff = _max_diff(eager, fused)
            timing = {
                "eager_allgather_qkv_gemm": _time_cuda(
                    eager_fn, args.warmup, args.iters),
                "raw_fused_allgather_qkv_gemm": _time_cuda(
                    raw_fused_fn, args.warmup, args.iters),
                "eager_full_unsplit_allgather_qkv_gemm": _time_cuda(
                    eager_full_fn, args.warmup, args.iters),
            }
            eager_after = eager_fn()
            fused_after = raw_fused_fn()
            fused_after_2 = raw_fused_fn()
            torch.cuda.synchronize()
            diff_after_timing = _max_diff(eager_after, fused_after)
            diff_fused_repeat = _max_diff(fused_after, fused_after_2)

        eager_ms = timing["eager_allgather_qkv_gemm"]["mean_ms"]
        fused_ms = timing["raw_fused_allgather_qkv_gemm"]["mean_ms"]
        local_result.update({
            "semantic_ok": (
                diff["max_abs"] <= args.atol
                and diff_after_timing["max_abs"] <= args.atol
                and diff_fused_repeat["max_abs"] <= args.atol
            ),
            "diff": diff,
            "diff_after_timing": diff_after_timing,
            "diff_fused_repeat": diff_fused_repeat,
            "timing": timing,
            "speedup_fused_vs_eager": (
                eager_ms / fused_ms if fused_ms > 0 else None),
        })

        gathered = _all_gather_dict(local_result)
        if rank == 0:
            semantic_ok = all(item["semantic_ok"] for item in gathered)
            eager_stats = _rank_stats(gathered, "eager_allgather_qkv_gemm")
            fused_stats = _rank_stats(gathered, "raw_fused_allgather_qkv_gemm")
            result = {
                "ok": semantic_ok,
                "semantic_ok": semantic_ok,
                "perf_summary": {
                    "eager_allgather_qkv_gemm": eager_stats,
                    "raw_fused_allgather_qkv_gemm": fused_stats,
                    "rankmax_speedup_fused_vs_eager": (
                        eager_stats["max_ms"] / fused_stats["max_ms"]
                        if fused_stats["max_ms"] > 0 else None),
                    "avg_speedup_fused_vs_eager": (
                        eager_stats["avg_ms"] / fused_stats["avg_ms"]
                        if fused_stats["avg_ms"] > 0 else None),
                },
                "ranks": gathered,
                "interpretation": {
                    "target": "raw symm_mem.fused_all_gather_matmul on QKV shape",
                    "eager_reference": "vLLM tensor_model_parallel_all_gather + aten.mm",
                    "topology": "dp2,tp4,ep8 when launched with 8 ranks and --tp-size 4",
                    "does_not_use_torch_compile": True,
                    "split_chunks": args.split_chunks,
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
