#!/usr/bin/env python3
"""
Minimal diagnostic for dInfer EPLB runtime state wiring.

Validates:
1) build_and_set_eplb_runtime_state works under distributed vLLM context.
2) Per-layer experts.enable_eplb toggles correctly.
3) EPLB tensor shapes are coherent for sparse MoE layers.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

REPO_ROOT = "/home/wuhang/wuhang/dllm_wh"
sys.path.insert(0, REPO_ROOT + "/lib_cite/dInfer/python")

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
TP_SIZE = 4


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dp_size = world_size // TP_SIZE
    dp_rank = rank // TP_SIZE
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from transformers import AutoConfig
    from dinfer.model import LLaDA2MoeModelLM

    pcfg_init = ParallelConfig(
        tensor_parallel_size=1, data_parallel_size=1, enable_expert_parallel=True
    )
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(
            world_size, rank, "env://", local_rank, "nccl"
        )

    pcfg = ParallelConfig(
        tensor_parallel_size=TP_SIZE,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        enable_expert_parallel=True,
    )
    # Match runtime build path: EPLB config must be visible before model construction.
    pcfg.enable_eplb = True
    pcfg.eplb_config.num_redundant_experts = 13
    pcfg.eplb_config.window_size = 16
    pcfg.eplb_config.step_interval = 16
    vllm_cfg = VllmConfig(parallel_config=pcfg)

    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(
            tensor_model_parallel_size=TP_SIZE, backend="nccl"
        )
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        )
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        model.build_and_set_eplb_runtime_state(
            num_redundant_experts=13,
            expert_load_window_size=16,
            expert_rearrangement_step_interval=16,
            device=device,
        )
        state = model.get_eplb_runtime_state()
        if state is None:
            raise RuntimeError("EPLB state should not be None after build.")

        sparse_layers = state["sparse_layer_ids"]
        elv = state["expert_load_view"]
        l2p = state["logical_to_physical_map"]
        lrc = state["logical_replica_count"]

        # Basic shape checks.
        if elv.shape[0] != len(sparse_layers):
            raise RuntimeError(f"unexpected expert_load_view shape: {tuple(elv.shape)}")
        if l2p.shape[0] != len(sparse_layers):
            raise RuntimeError(f"unexpected logical_to_physical_map shape: {tuple(l2p.shape)}")
        if lrc.shape[0] != len(sparse_layers):
            raise RuntimeError(f"unexpected logical_replica_count shape: {tuple(lrc.shape)}")

        # Ensure per-layer experts are EPLB-enabled.
        enabled_count = 0
        physical_oob = 0
        physical_max = -1
        for layer_id in sparse_layers:
            mlp = model.model.layers[layer_id].mlp
            if getattr(mlp.experts, "enable_eplb", False):
                enabled_count += 1
            local_i = sparse_layers.index(layer_id)
            l2p_layer = l2p[local_i]
            valid = l2p_layer[l2p_layer >= 0]
            if valid.numel() > 0:
                physical_max = max(physical_max, int(valid.max().item()))
                physical_oob += int((valid >= mlp.experts.global_num_experts).sum().item())
        if enabled_count != len(sparse_layers):
            raise RuntimeError(
                f"EPLB enable mismatch: enabled={enabled_count}, sparse_layers={len(sparse_layers)}"
            )

        if rank == 0:
            print("EPLB runtime state OK")
            print(f"  sparse_layers={len(sparse_layers)}")
            print(f"  expert_load_view.shape={tuple(elv.shape)}")
            print(f"  logical_to_physical_map.shape={tuple(l2p.shape)}")
            print(f"  logical_replica_count.shape={tuple(lrc.shape)}")
            print(f"  logical_to_physical max_id={physical_max}")
            print(f"  logical_to_physical oob_vs_layer_global_num_experts={physical_oob}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
