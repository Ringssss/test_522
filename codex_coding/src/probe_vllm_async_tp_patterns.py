#!/usr/bin/env python3
"""Probe vLLM sequence-parallel / AsyncTP pattern prerequisites.

This is intentionally source- and environment-level only. It does not launch a
distributed benchmark and does not modify model code.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
BENCH = REPO_ROOT / "codex_coding/src/bench_bsp_moe_dp2.py"


def module_spec(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
        return {
            "found": spec is not None,
            "origin": getattr(spec, "origin", None) if spec else None,
            "error": None,
        }
    except Exception as exc:
        return {
            "found": False,
            "origin": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def import_status(name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(name)
        return {
            "ok": True,
            "version": getattr(mod, "__version__", None),
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "version": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def torch_op_exists(torch_ops: Any, dotted: str) -> dict[str, Any]:
    obj = torch_ops
    try:
        for part in dotted.split("."):
            obj = getattr(obj, part)
        return {"exists": True, "error": None}
    except Exception as exc:
        return {"exists": False, "error": f"{type(exc).__name__}: {exc}"}


def scan_source(path: Path) -> dict[str, Any]:
    text = path.read_text()
    lines = text.splitlines()

    def count(pattern: str) -> int:
        return len(re.findall(pattern, text))

    def has_ordered_window(first: str, second: str, window: int) -> bool:
        first_lines = [i for i, line in enumerate(lines) if first in line]
        second_lines = [i for i, line in enumerate(lines) if second in line]
        for a in first_lines:
            if any(a <= b <= a + window for b in second_lines):
                return True
        return False

    return {
        "path": str(path),
        "counts": {
            "sequence_parallel_chunk": count(r"\bsequence_parallel_chunk\b"),
            "tensor_model_parallel_reduce_scatter": count(
                r"\btensor_model_parallel_reduce_scatter\b"),
            "tensor_model_parallel_all_gather": count(
                r"\btensor_model_parallel_all_gather\b"),
            "torch_ops_vllm_reduce_scatter": count(
                r"torch\.ops\.vllm\.reduce_scatter"),
            "torch_ops_vllm_all_gather": count(r"torch\.ops\.vllm\.all_gather"),
            "dense_quant_apply": count(r"dense\.quant_method\.apply"),
            "attention_query_key_value": count(r"\.query_key_value\("),
            "sp_attention_input": count(r"\bSPAttentionInput\b"),
            "attn_sp_result": count(r"\bAttnSPResult\b"),
        },
        "ordered_patterns": {
            "dense_apply_then_reduce_scatter_within_20_lines":
                has_ordered_window(
                    "dense.quant_method.apply",
                    "tensor_model_parallel_reduce_scatter",
                    20,
                ),
            "all_gather_then_attention_call_within_20_lines":
                has_ordered_window(
                    "tensor_model_parallel_all_gather",
                    "attention(",
                    20,
                ),
            "sp_input_gather_then_qkv_within_40_lines":
                has_ordered_window(
                    "tensor_model_parallel_all_gather",
                    "attention.query_key_value",
                    40,
                ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, default=BENCH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    module_specs = {
        name: module_spec(name)
        for name in [
            "vllm",
            "flashinfer",
            "flashinfer.comm",
            "deep_ep",
            "pplx_kernels",
            "pplx_kernels_C",
            "torch.distributed._symmetric_memory",
        ]
    }

    # Match the benchmark guard. Current environment has deep_ep installed but
    # direct import can fail due NCCL symbol drift; BSP-G experiments disable it.
    sys.modules["deep_ep"] = None

    import torch

    imports = {}
    for name in [
        "vllm",
        "vllm.distributed.parallel_state",
        "vllm.model_executor.layers.layernorm",
        "vllm.compilation.sequence_parallelism",
        "vllm.compilation.collective_fusion",
    ]:
        imports[name] = import_status(name)

    op_checks = {
        name: torch_op_exists(torch.ops, name)
        for name in [
            "vllm.all_reduce.default",
            "vllm.reduce_scatter.default",
            "vllm.all_gather.default",
            "_C.rms_norm.default",
            "_C.fused_add_rms_norm.default",
            "symm_mem.fused_matmul_reduce_scatter",
            "symm_mem.fused_all_gather_matmul",
            "symm_mem.fused_scaled_matmul_reduce_scatter",
            "symm_mem.fused_all_gather_scaled_matmul",
        ]
    }

    flashinfer_comm = None
    if imports["vllm.compilation.collective_fusion"]["ok"]:
        try:
            mod = importlib.import_module("vllm.compilation.collective_fusion")
            flashinfer_comm = getattr(mod, "flashinfer_comm", None) is not None
        except Exception:
            flashinfer_comm = None

    symm_mem = import_status("torch.distributed._symmetric_memory")
    symm_mem_attrs = []
    if symm_mem["ok"]:
        try:
            symm_mem_attrs = [
                name for name in dir(torch.ops.symm_mem)
                if "matmul" in name or "reduce" in name or "gather" in name
            ]
        except Exception:
            symm_mem_attrs = []

    source = scan_source(args.bench)

    result = {
        "module_specs": module_specs,
        "imports_with_deep_ep_guard": imports,
        "torch_ops_after_vllm_import": op_checks,
        "flashinfer_comm_available_in_collective_fusion": flashinfer_comm,
        "symm_mem_import": symm_mem,
        "symm_mem_collective_attrs": symm_mem_attrs,
        "source_scan": source,
        "interpretation": {
            "sequence_parallelism_pass_prereqs": (
                op_checks["vllm.all_reduce.default"]["exists"]
                and op_checks["_C.rms_norm.default"]["exists"]
            ),
            "async_tp_pass_importable": imports[
                "vllm.compilation.collective_fusion"]["ok"],
            "async_tp_fused_symm_mem_ops_registered": (
                op_checks["symm_mem.fused_matmul_reduce_scatter"]["exists"]
                and op_checks["symm_mem.fused_all_gather_matmul"]["exists"]
            ),
            "bsp_g_has_conceptual_gemm_reduce_scatter_shape": source[
                "ordered_patterns"][
                    "dense_apply_then_reduce_scatter_within_20_lines"],
            "bsp_g2_has_conceptual_allgather_gemm_shape": source[
                "ordered_patterns"]["sp_input_gather_then_qkv_within_40_lines"],
            "current_benchmark_is_not_using_vllm_compile_pass_manager": True,
        },
    }

    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
