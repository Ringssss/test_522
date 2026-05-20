#!/usr/bin/env python3
"""Inventory vLLM MoE communication/backend availability.

This probe avoids distributed initialization and does not instantiate the model.
It reports package presence, import health, vLLM capability helpers, and the
environment variables that select MoE/all2all backends.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


def spec_status(name: str) -> dict[str, Any]:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    env = {
        name: os.environ.get(name)
        for name in [
            "VLLM_ALL2ALL_BACKEND",
            "VLLM_MOE_DP_CHUNK_SIZE",
            "VLLM_USE_FLASHINFER_MOE_FP16",
            "VLLM_DEEPEP_BUFFER_SIZE_MB",
            "VLLM_ALLREDUCE_USE_SYMM_MEM",
        ]
    }

    specs_no_guard = {
        name: spec_status(name)
        for name in [
            "deep_ep",
            "pplx_kernels",
            "flashinfer",
            "flashinfer.comm",
            "flashinfer.comm.mnnvl",
            "flashinfer.comm.trtllm_alltoall",
            "flashinfer.fused_moe",
        ]
    }

    imports_no_guard = {
        "deep_ep": import_status("deep_ep"),
    }

    # Match the benchmark: disable DeepEP to avoid importing an ABI-broken
    # extension while still allowing vLLM and FlashInfer modules to load.
    sys.modules["deep_ep"] = None

    import torch
    import vllm.envs as vllm_envs
    from vllm.utils import has_deep_ep, has_pplx
    from vllm.utils.flashinfer import (
        has_flashinfer_all2all,
        has_flashinfer_cutlass_fused_moe,
        has_flashinfer_moe,
    )

    guarded = {
        "torch": torch.__version__,
        "vllm_env_defaults": {
            "VLLM_ALL2ALL_BACKEND": vllm_envs.VLLM_ALL2ALL_BACKEND,
            "VLLM_MOE_DP_CHUNK_SIZE": vllm_envs.VLLM_MOE_DP_CHUNK_SIZE,
            "VLLM_USE_FLASHINFER_MOE_FP16":
                vllm_envs.VLLM_USE_FLASHINFER_MOE_FP16,
            "VLLM_DEEPEP_BUFFER_SIZE_MB":
                vllm_envs.VLLM_DEEPEP_BUFFER_SIZE_MB,
        },
        "vllm_helpers": {
            "has_deep_ep": has_deep_ep(),
            "has_pplx": has_pplx(),
            "has_flashinfer_moe": has_flashinfer_moe(),
            "has_flashinfer_all2all": has_flashinfer_all2all(),
            "has_flashinfer_cutlass_fused_moe":
                has_flashinfer_cutlass_fused_moe(),
        },
        "imports": {
            "vllm.distributed.device_communicators.all2all":
                import_status("vllm.distributed.device_communicators.all2all"),
            "vllm.model_executor.layers.fused_moe.layer":
                import_status("vllm.model_executor.layers.fused_moe.layer"),
        },
    }

    result = {
        "env": env,
        "module_specs_no_guard": specs_no_guard,
        "imports_no_guard": imports_no_guard,
        "with_deep_ep_guard": guarded,
        "backend_matrix": {
            "allgather_reducescatter": {
                "available": True,
                "current_default": guarded["vllm_env_defaults"][
                    "VLLM_ALL2ALL_BACKEND"] == "allgather_reducescatter",
                "notes": "Current AgRs path; supports SP by switching DP group to EP group.",
            },
            "naive": {
                "available": True,
                "recommended": False,
                "notes": "Debug fallback using broadcasts/all-reduce; not a performance target.",
            },
            "pplx": {
                "available": guarded["vllm_helpers"]["has_pplx"],
                "recommended": False,
                "notes": "Package not present in current environment.",
            },
            "deepep_high_throughput": {
                "available_under_guard": guarded["vllm_helpers"]["has_deep_ep"],
                "import_without_guard_ok": imports_no_guard["deep_ep"]["ok"],
                "recommended": False,
                "notes": "deep_ep package is found without guard but import fails in this environment; rebuild/ABI fix required before use.",
            },
            "deepep_low_latency": {
                "available_under_guard": guarded["vllm_helpers"]["has_deep_ep"],
                "import_without_guard_ok": imports_no_guard["deep_ep"]["ok"],
                "recommended": False,
                "notes": "Same DeepEP ABI constraint as HT path.",
            },
            "flashinfer_all2allv": {
                "available": guarded["vllm_helpers"]["has_flashinfer_all2all"],
                "recommended": "smoke_first",
                "notes": "Capability helper is true; needs distributed smoke because MNNVL workspace/group assumptions may differ from current C12 layout.",
            },
            "flashinfer_cutlass_fused_moe": {
                "available": guarded["vllm_helpers"][
                    "has_flashinfer_cutlass_fused_moe"],
                "env_enabled": guarded["vllm_env_defaults"][
                    "VLLM_USE_FLASHINFER_MOE_FP16"],
                "recommended": False,
                "notes": "vLLM enables unquantized FP16 Cutlass MoE only under specific EP/DP/device/env conditions; current env flag is off and current route is bf16/unquantized LLaDA2.",
            },
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
