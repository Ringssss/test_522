#!/usr/bin/env python3
"""Diagnose: can vllm 0.10.2 create DP+EP groups with dp>1?
Key insight: init_distributed_environment with dp=1 first (skip DP rank adjustment),
then set dp=N before initialize_model_parallel (which creates the actual groups).
"""
import os, sys, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("VLLM_ALL2ALL_BACKEND", "naive")

local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))

device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)

from vllm import distributed as vllm_dist
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

# Step 1: Init distributed with dp=1 (no DP rank/port adjustment)
pcfg_init = ParallelConfig(
    tensor_parallel_size=1,
    data_parallel_size=1,
    enable_expert_parallel=True,
)
print(f"[rank {rank}] Step 1: init_distributed_environment (dp=1, no adjustment)", flush=True)
with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
    vllm_dist.init_distributed_environment(world_size, rank, "env://", local_rank, "nccl")

# Step 2: Now set dp=world_size and create groups
pcfg = ParallelConfig(
    tensor_parallel_size=1,
    data_parallel_size=world_size,
    data_parallel_rank=rank,
    enable_expert_parallel=True,
)
vllm_cfg = VllmConfig(parallel_config=pcfg)
print(f"[rank {rank}] Step 2: initialize_model_parallel(tp=1) with dp={world_size}", flush=True)
with set_current_vllm_config(vllm_cfg):
    vllm_dist.initialize_model_parallel(tensor_model_parallel_size=1, backend="nccl")
    print(f"[rank {rank}] Step 3: check groups", flush=True)

    from vllm.distributed import (get_tp_group, get_tensor_model_parallel_world_size)
    tp_ws = get_tensor_model_parallel_world_size()
    tp_rank = get_tp_group().rank_in_group
    print(f"[rank {rank}]   TP: size={tp_ws}, rank_in_group={tp_rank}", flush=True)

    try:
        from vllm.distributed import get_dp_group
        dp = get_dp_group()
        print(f"[rank {rank}]   DP: size={dp.world_size}, rank_in_group={dp.rank_in_group}", flush=True)
    except Exception as e:
        print(f"[rank {rank}]   DP: FAILED — {e}", flush=True)

    try:
        from vllm.distributed import get_ep_group
        ep = get_ep_group()
        print(f"[rank {rank}]   EP: size={ep.world_size}, rank_in_group={ep.rank_in_group}", flush=True)
    except Exception as e:
        print(f"[rank {rank}]   EP: FAILED — {e}", flush=True)

    # Test EP communicator all2all status
    try:
        comm = ep.device_communicator
        has_a2a = hasattr(comm, 'all2all_manager') and comm.all2all_manager is not None
        a2a_cls = type(comm.all2all_manager).__name__ if has_a2a else "None"
        print(f"[rank {rank}]   AllToAll manager: {a2a_cls}", flush=True)
    except Exception as e:
        print(f"[rank {rank}]   AllToAll check: FAILED — {e}", flush=True)

print(f"[rank {rank}] DONE", flush=True)

import torch.distributed as dist
dist.barrier()
dist.destroy_process_group()
