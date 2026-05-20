#!/usr/bin/env python3
"""Validate raw symm_mem fused matmul-reduce-scatter SUM semantics."""

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

    # Use the output so Dynamo/Inductor cannot treat this as dead work if the
    # probe is later compiled or refactored.
    checksum = float(out.float().mean().item())
    return {
        "total_s": elapsed_s,
        "mean_ms": elapsed_s * 1000.0 / iters,
        "checksum": checksum,
    }


def _semantic_ok(diff: dict[str, float], atol: float) -> bool:
    return diff["max_abs"] <= atol


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument(
        "--in-hidden",
        type=int,
        default=None,
        help="Input feature dimension. Defaults to --hidden for square probes.",
    )
    parser.add_argument(
        "--out-hidden",
        type=int,
        default=None,
        help="Output feature dimension. Defaults to --hidden for square probes.",
    )
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--atol", type=float, default=5e-2)
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
    in_hidden = args.in_hidden if args.in_hidden is not None else args.hidden
    out_hidden = args.out_hidden if args.out_hidden is not None else args.hidden

    dp_size = world_size // args.tp_size
    dp_rank = rank // args.tp_size
    tp_rank = rank % args.tp_size

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Importing the module registers torch.ops.symm_mem.* and exposes the
    # enable_symm_mem_for_group helper needed by the fused kernels.
    from torch.distributed._symmetric_memory import (
        enable_symm_mem_for_group,
        is_symm_mem_enabled_for_group,
    )
    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        tensor_model_parallel_reduce_scatter,
    )

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

        # x is identical within each TP group, while weights differ by rank.
        # This mirrors row-parallel output projection semantics and prevents a
        # degenerate same-partial-only test.
        torch.manual_seed(1234 + dp_rank)
        x_full = torch.randn(
            args.tokens, in_hidden, device=device, dtype=torch.bfloat16)
        torch.manual_seed(4321 + rank)
        weight = torch.randn(
            in_hidden, out_hidden, device=device, dtype=torch.bfloat16) * 0.02
        x_full = x_full.contiguous()
        weight = weight.contiguous()

        def eager_sum_fn() -> torch.Tensor:
            y = torch.ops.aten.mm.default(x_full, weight)
            return tensor_model_parallel_reduce_scatter(y, dim=0)

        def fused_sum_fn() -> torch.Tensor:
            return torch.ops.symm_mem.fused_matmul_reduce_scatter(
                x_full,
                weight,
                "sum",
                scatter_dim=0,
                group_name=group_name,
            )

        def fused_avg_fn() -> torch.Tensor:
            return torch.ops.symm_mem.fused_matmul_reduce_scatter(
                x_full,
                weight,
                "avg",
                scatter_dim=0,
                group_name=group_name,
            )

        local_result: dict[str, Any] = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "tp_size": args.tp_size,
            "tp_rank": tp_rank,
            "dp_size": dp_size,
            "dp_rank": dp_rank,
            "tokens": args.tokens,
            "hidden": args.hidden,
            "in_hidden": in_hidden,
            "out_hidden": out_hidden,
            "shape_kind": (
                "square" if in_hidden == out_hidden else "rectangular"),
            "dtype": str(x_full.dtype),
            "group_name": group_name,
            "symm_mem_enabled": is_symm_mem_enabled_for_group(group_name),
            "has_fused_matmul_reduce_scatter": hasattr(
                torch.ops.symm_mem, "fused_matmul_reduce_scatter"),
            "atol": args.atol,
        }

        with torch.inference_mode():
            torch.cuda.synchronize()
            eager_sum = eager_sum_fn()
            fused_sum = fused_sum_fn()
            fused_avg = fused_avg_fn()
            torch.cuda.synchronize()

            diff_sum_vs_eager = _max_diff(fused_sum, eager_sum)
            diff_avg_vs_eager_div_tp = _max_diff(
                fused_avg, eager_sum / args.tp_size)
            diff_sum_vs_avg_times_tp = _max_diff(
                fused_sum, fused_avg * args.tp_size)

            timing = {
                "eager_sum": _time_cuda(eager_sum_fn, args.warmup, args.iters),
                "fused_sum": _time_cuda(fused_sum_fn, args.warmup, args.iters),
                "fused_avg": _time_cuda(fused_avg_fn, args.warmup, args.iters),
            }

        eager_ms = timing["eager_sum"]["mean_ms"]
        fused_sum_ms = timing["fused_sum"]["mean_ms"]
        local_result.update({
            "shapes": {
                "x_full": list(x_full.shape),
                "weight": list(weight.shape),
                "eager_sum": list(eager_sum.shape),
                "fused_sum": list(fused_sum.shape),
                "fused_avg": list(fused_avg.shape),
            },
            "diffs": {
                "fused_sum_vs_eager_sum": diff_sum_vs_eager,
                "fused_avg_vs_eager_sum_div_tp": diff_avg_vs_eager_div_tp,
                "fused_sum_vs_fused_avg_times_tp": diff_sum_vs_avg_times_tp,
            },
            "semantic_ok": {
                "sum_matches_eager": _semantic_ok(
                    diff_sum_vs_eager, args.atol),
                "avg_matches_eager_div_tp": _semantic_ok(
                    diff_avg_vs_eager_div_tp, args.atol),
            },
            "timing": timing,
            "speedup_fused_sum_vs_eager": (
                eager_ms / fused_sum_ms if fused_sum_ms > 0 else None),
            "fused_sum_not_slower_than_eager": fused_sum_ms <= eager_ms * 1.05,
        })

        gathered = _all_gather_dict(local_result)
        if rank == 0:
            sum_ok = all(
                item["semantic_ok"]["sum_matches_eager"] for item in gathered)
            avg_ok = all(
                item["semantic_ok"]["avg_matches_eager_div_tp"]
                for item in gathered)
            perf_ok = all(
                item["fused_sum_not_slower_than_eager"] for item in gathered)
            result = {
                "ok": sum_ok and avg_ok and perf_ok,
                "semantic_ok": sum_ok and avg_ok,
                "sum_semantic_ok": sum_ok,
                "avg_control_ok": avg_ok,
                "perf_ok": perf_ok,
                "pass_criteria": {
                    "sum_matches_eager": sum_ok,
                    "avg_matches_eager_div_tp": avg_ok,
                    "fused_sum_not_materially_slower": perf_ok,
                    "materially_slower_threshold": "fused_sum_ms <= eager_ms * 1.05",
                },
                "ranks": gathered,
                "interpretation": {
                    "raw_fused_sum_target": (
                        "symm_mem.fused_matmul_reduce_scatter(..., 'sum')"),
                    "eager_reference": (
                        "tensor_model_parallel_reduce_scatter(x @ weight, dim=0)"),
                    "topology": "dp2,tp4,ep8 when launched with 8 ranks and --tp-size 4",
                    "does_not_use_vllm_asynctp_pass": True,
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
