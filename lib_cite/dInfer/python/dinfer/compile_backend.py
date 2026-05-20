"""dInfer wrapper for reusing vLLM compilation backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from vllm.config import (CUDAGraphMode, CompilationConfig, ParallelConfig,
                         VllmConfig)
from vllm.config.compilation import CompilationLevel, PassConfig
from vllm.compilation.backends import VllmBackend


@dataclass(frozen=True)
class DInferCompileBackend:
    """Factory for fresh vLLM torch.compile backend instances.

    vLLM's ``VllmBackend`` asserts that each backend object is called only once
    by Dynamo. This adapter therefore exposes ``new_backend`` instead of sharing
    one backend across multiple compiled modules.
    """

    tp_size: int
    dp_size: int = 1
    dp_rank: int = 0
    rank: int = 0
    cache_dir: Optional[str] = None
    compile_sizes: Optional[tuple[int, ...]] = None
    enable_async_tp: bool = True
    enable_sequence_parallelism: bool = True
    prefix: str = "dinfer_compile"

    def make_vllm_config(self) -> VllmConfig:
        pass_config = PassConfig(
            enable_sequence_parallelism=self.enable_sequence_parallelism,
            enable_async_tp=self.enable_async_tp,
        )
        compilation_config = CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            cudagraph_mode=CUDAGraphMode.NONE,
            use_inductor=True,
            cache_dir=self.cache_dir or "",
            compile_sizes=list(self.compile_sizes)
            if self.compile_sizes is not None else None,
            pass_config=pass_config,
        )
        parallel_config = ParallelConfig(
            tensor_parallel_size=self.tp_size,
            data_parallel_size=self.dp_size,
            data_parallel_rank=self.dp_rank,
            enable_expert_parallel=True,
            rank=self.rank,
        )
        return VllmConfig(
            parallel_config=parallel_config,
            compilation_config=compilation_config,
        )

    def new_backend(self) -> Callable:
        return VllmBackend(self.make_vllm_config(), prefix=self.prefix)

    @classmethod
    def from_distributed_env(
        cls,
        *,
        world_size: int,
        rank: int,
        tp_size: int,
        cache_dir: Optional[Path | str] = None,
        compile_sizes: Optional[tuple[int, ...]] = None,
        prefix: str = "dinfer_compile",
    ) -> "DInferCompileBackend":
        if world_size % tp_size != 0:
            raise ValueError(
                f"world_size={world_size} must be divisible by tp_size={tp_size}"
            )
        dp_size = world_size // tp_size
        dp_rank = rank // tp_size
        return cls(
            tp_size=tp_size,
            dp_size=dp_size,
            dp_rank=dp_rank,
            rank=rank,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            compile_sizes=compile_sizes,
            prefix=prefix,
        )
