#!/usr/bin/env python3
"""Distributed AsyncTP pattern-hit probe through DInferCompileBackend."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
DINFER_PYTHON = REPO_ROOT / "lib_cite/dInfer/python"
if str(DINFER_PYTHON) not in sys.path:
    sys.path.insert(0, str(DINFER_PYTHON))


class GemmReduceScatterModule(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_reduce_scatter

        y = torch.ops.aten.mm.default(x, self.weight)
        return tensor_model_parallel_reduce_scatter(y, dim=0)


class AllGatherGemmModule(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_all_gather

        y = tensor_model_parallel_all_gather(x, dim=0)
        return torch.ops.aten.mm.default(y, self.weight)


def _summarize_cache(cache_dir: Path) -> dict[str, Any]:
    files = []
    counters = {
        "symm_mem.fused_matmul_reduce_scatter": 0,
        "symm_mem.fused_all_gather_matmul": 0,
        "vllm.reduce_scatter": 0,
        "vllm.all_gather": 0,
        "aten.mm": 0,
    }
    if not cache_dir.exists():
        return {"exists": False, "files": files, "counters": counters}
    for path in cache_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(cache_dir))
        if path.suffix not in {".py", ".txt", ".log"}:
            continue
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


def _fused_hit(pattern_name: str, counters: dict[str, int]) -> bool:
    if pattern_name == "gemm_reduce_scatter":
        return counters.get("symm_mem.fused_matmul_reduce_scatter", 0) > 0
    if pattern_name == "all_gather_gemm":
        return counters.get("symm_mem.fused_all_gather_matmul", 0) > 0
    return False


def _semantic_ok(diff: dict[str, float], atol: float = 1e-3) -> bool:
    return diff["max_abs"] <= atol


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--tp-size", type=int, default=4)
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
    dp_size = world_size // args.tp_size
    dp_rank = rank // args.tp_size

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
            world_size, rank, "env://", local_rank, "nccl"
        )

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

        n_local = args.tokens // args.tp_size
        if args.tokens % args.tp_size != 0:
            raise ValueError("--tokens must be divisible by --tp-size")

        torch.manual_seed(1234)
        x_full = torch.randn(
            args.tokens, args.hidden, device=device, dtype=torch.bfloat16)
        tp_rank = rank % args.tp_size
        x_sp = x_full[tp_rank * n_local:(tp_rank + 1) * n_local].contiguous()

        modules = {
            "gemm_reduce_scatter": (GemmReduceScatterModule(args.hidden), x_full),
            "all_gather_gemm": (AllGatherGemmModule(args.hidden), x_sp),
        }

        local_results: dict[str, Any] = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "tp_size": args.tp_size,
            "dp_size": dp_size,
            "dp_rank": dp_rank,
            "tokens": args.tokens,
            "local_tokens": n_local,
            "hidden": args.hidden,
            "patterns": {},
        }

        args.cache_root.mkdir(parents=True, exist_ok=True)
        for name, (module, example) in modules.items():
            module = module.to(device=device, dtype=torch.bfloat16).eval()
            cache_dir = args.cache_root / name
            runtime_shape = int(example.shape[0])
            torch._dynamo.mark_dynamic(example, 0)
            adapter = DInferCompileBackend.from_distributed_env(
                world_size=world_size,
                rank=rank,
                tp_size=args.tp_size,
                cache_dir=cache_dir,
                compile_sizes=(runtime_shape,),
                prefix=f"dinfer_compile_{name}",
            )
            vllm_cfg = adapter.make_vllm_config()
            compiled = torch.compile(
                module,
                backend=adapter.new_backend(),
                fullgraph=True,
                dynamic=True,
            )
            torch.cuda.synchronize()
            compile_start = time.perf_counter()
            with torch.inference_mode():
                eager_out = module(example)
                torch.cuda.synchronize()
                compiled_out = compiled(example)
                torch.cuda.synchronize()
                compiled_out_2 = compiled(example)
                torch.cuda.synchronize()
            compile_elapsed = time.perf_counter() - compile_start
            cache_summary = _summarize_cache(cache_dir)
            diff_first = _max_diff(eager_out, compiled_out)
            diff_second = _max_diff(eager_out, compiled_out_2)
            diff_second_vs_eager_div_tp = _max_diff(
                eager_out / args.tp_size, compiled_out_2)
            local_results["patterns"][name] = {
                "compiled_ok": True,
                "semantic_ok": _semantic_ok(diff_second),
                "fused_hit": _fused_hit(name, cache_summary["counters"]),
                "compile_and_two_run_s": compile_elapsed,
                "eager_shape": list(eager_out.shape),
                "compiled_shape": list(compiled_out.shape),
                "runtime_shape": runtime_shape,
                "diff_first": diff_first,
                "diff_second": diff_second,
                "diff_second_vs_eager_div_tp": diff_second_vs_eager_div_tp,
                "cache_dir": str(cache_dir),
                "pass_config": {
                    "enable_sequence_parallelism": (
                        vllm_cfg.compilation_config.pass_config
                        .enable_sequence_parallelism),
                    "enable_async_tp": (
                        vllm_cfg.compilation_config.pass_config.enable_async_tp),
                },
                "cache_summary": cache_summary,
            }

        gathered = _all_gather_dict(local_results)
        if rank == 0:
            all_patterns = [
                pattern
                for item in gathered
                for pattern in item["patterns"].values()
            ]
            result = {
                "compiled_ok": all(
                    pattern.get("compiled_ok", False)
                    for pattern in all_patterns),
                "semantic_ok": all(
                    pattern.get("semantic_ok", False)
                    for pattern in all_patterns),
                "fused_hit": all(
                    pattern.get("fused_hit", False)
                    for pattern in all_patterns),
                "ok": all(
                    pattern.get("compiled_ok", False)
                    and pattern.get("semantic_ok", False)
                    and pattern.get("fused_hit", False)
                    for pattern in all_patterns),
                "ranks": gathered,
                "interpretation": {
                    "compiled_with_dinfer_compile_backend": True,
                    "fused_op_evidence_is_cache_text_search": True,
                    "requires_manual_review_if_fused_counts_zero": True,
                    "semantic_ok_compares_against_eager_vllm_collectives": True,
                    "gemm_reduce_scatter_avg_check": (
                        "diff_second_vs_eager_div_tp near zero means the "
                        "fused path used avg semantics while eager vllm "
                        "reduce_scatter used sum semantics."),
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
