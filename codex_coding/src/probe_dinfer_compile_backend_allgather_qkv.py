#!/usr/bin/env python3
"""Benchmark AsyncTP fused AllGather+QKV GEMM through DInferCompileBackend."""

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
import torch.nn as nn

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
DINFER_PYTHON = REPO_ROOT / "lib_cite/dInfer/python"
if str(DINFER_PYTHON) not in sys.path:
    sys.path.insert(0, str(DINFER_PYTHON))


class AllGatherQKVGemmModule(nn.Module):
    def __init__(self, hidden: int, qkv_out_hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden, qkv_out_hidden))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x_sp: torch.Tensor) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_all_gather

        hidden = tensor_model_parallel_all_gather(x_sp, dim=0)
        return torch.ops.aten.mm.default(hidden, self.weight)


def _summarize_cache(cache_dir: Path) -> dict[str, Any]:
    files = []
    counters = {
        "symm_mem.fused_all_gather_matmul": 0,
        "symm_mem.fused_all_gather_scaled_matmul": 0,
        "vllm.all_gather": 0,
        "aten.mm": 0,
    }
    if not cache_dir.exists():
        return {"exists": False, "files": files, "counters": counters}
    for path in cache_dir.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".txt", ".log"}:
            continue
        rel = str(path.relative_to(cache_dir))
        files.append(rel)
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for key in counters:
            counters[key] += text.count(key)
    return {"exists": True, "files": sorted(files)[:100], "counters": counters}


def _all_gather_dict(local: dict[str, Any]) -> list[dict[str, Any]]:
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    return gathered


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = (a - b).float().abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT / "codex_coding/results/dinfer_compile_cache",
    )
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

    dp_size = world_size // args.tp_size
    dp_rank = rank // args.tp_size
    tp_rank = rank % args.tp_size
    local_tokens = args.tokens // args.tp_size

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    from dinfer.compile_backend import DInferCompileBackend
    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

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

        torch.manual_seed(1234 + dp_rank)
        x_full = torch.randn(
            args.tokens, args.hidden, device=device, dtype=torch.bfloat16)
        x_sp = x_full[
            tp_rank * local_tokens:(tp_rank + 1) * local_tokens
        ].contiguous()

        torch.manual_seed(4321 + rank)
        module = AllGatherQKVGemmModule(
            args.hidden, args.qkv_out_hidden).to(
                device=device, dtype=torch.bfloat16).eval()

        args.cache_root.mkdir(parents=True, exist_ok=True)
        cache_dir = args.cache_root / "allgather_qkv"
        torch._dynamo.mark_dynamic(x_sp, 0)
        adapter = DInferCompileBackend.from_distributed_env(
            world_size=world_size,
            rank=rank,
            tp_size=args.tp_size,
            cache_dir=cache_dir,
            compile_sizes=(local_tokens,),
            prefix="dinfer_compile_allgather_qkv",
        )
        vllm_cfg = adapter.make_vllm_config()
        compiled = torch.compile(
            module,
            backend=adapter.new_backend(),
            fullgraph=True,
            dynamic=True,
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
            "local_tokens": local_tokens,
            "hidden": args.hidden,
            "qkv_out_hidden": args.qkv_out_hidden,
            "dtype": str(x_sp.dtype),
            "cache_dir": str(cache_dir),
            "pass_config": {
                "enable_sequence_parallelism": (
                    vllm_cfg.compilation_config.pass_config
                    .enable_sequence_parallelism),
                "enable_async_tp": (
                    vllm_cfg.compilation_config.pass_config.enable_async_tp),
            },
            "shapes": {
                "x_sp": list(x_sp.shape),
                "gathered_hidden": [args.tokens, args.hidden],
                "weight": list(module.weight.shape),
                "output": [args.tokens, args.qkv_out_hidden],
            },
        }

        with torch.inference_mode():
            torch.cuda.synchronize()
            compile_start = time.perf_counter()
            eager_out = module(x_sp)
            compiled_out = compiled(x_sp)
            compiled_out_2 = compiled(x_sp)
            torch.cuda.synchronize()
            compile_and_two_run_s = time.perf_counter() - compile_start

            diff_first = _max_diff(eager_out, compiled_out)
            diff_second = _max_diff(eager_out, compiled_out_2)
            cache_summary = _summarize_cache(cache_dir)
            fused_hit = (
                cache_summary["counters"]
                .get("symm_mem.fused_all_gather_matmul", 0) > 0
            )

            timing = {
                "eager_allgather_qkv_gemm": _time_cuda(
                    lambda: module(x_sp), args.warmup, args.iters),
                "compiled_fused_allgather_qkv_gemm": _time_cuda(
                    lambda: compiled(x_sp), args.warmup, args.iters),
            }

        eager_ms = timing["eager_allgather_qkv_gemm"]["mean_ms"]
        fused_ms = timing["compiled_fused_allgather_qkv_gemm"]["mean_ms"]
        local_result.update({
            "compiled_ok": True,
            "semantic_ok": diff_second["max_abs"] <= args.atol,
            "fused_hit": fused_hit,
            "diff_first": diff_first,
            "diff_second": diff_second,
            "compile_and_two_run_s": compile_and_two_run_s,
            "timing": timing,
            "speedup_compiled_vs_eager": (
                eager_ms / fused_ms if fused_ms > 0 else None),
            "cache_summary": cache_summary,
        })

        gathered = _all_gather_dict(local_result)
        if rank == 0:
            semantic_ok = all(item["semantic_ok"] for item in gathered)
            fused_hit_all = all(item["fused_hit"] for item in gathered)
            compiled_ok = all(item["compiled_ok"] for item in gathered)
            eager_stats = _rank_stats(gathered, "eager_allgather_qkv_gemm")
            fused_stats = _rank_stats(
                gathered, "compiled_fused_allgather_qkv_gemm")
            rankmax_speedup = (
                eager_stats["max_ms"] / fused_stats["max_ms"]
                if fused_stats["max_ms"] > 0 else None
            )
            avg_speedup = (
                eager_stats["avg_ms"] / fused_stats["avg_ms"]
                if fused_stats["avg_ms"] > 0 else None
            )
            result = {
                "ok": compiled_ok and semantic_ok and fused_hit_all,
                "compiled_ok": compiled_ok,
                "semantic_ok": semantic_ok,
                "fused_hit": fused_hit_all,
                "perf_summary": {
                    "eager_allgather_qkv_gemm": eager_stats,
                    "compiled_fused_allgather_qkv_gemm": fused_stats,
                    "rankmax_speedup_compiled_vs_eager": rankmax_speedup,
                    "avg_speedup_compiled_vs_eager": avg_speedup,
                },
                "ranks": gathered,
                "interpretation": {
                    "target": "attention-input tensor_model_parallel_all_gather -> QKV GEMM",
                    "shape": "x_sp=[tokens/tp, hidden], weight=[hidden, qkv_out_hidden]",
                    "topology": "dp2,tp4,ep8 when launched with 8 ranks and --tp-size 4",
                    "compile_path": "dInfer DInferCompileBackend reusing vLLM VllmBackend/AsyncTPPass",
                    "eager_reference": "vLLM tensor_model_parallel_all_gather + aten.mm",
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
