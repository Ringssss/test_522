#!/usr/bin/env python3
"""Minimal distributed DeepEP high-throughput dispatch/combine smoke.

This intentionally avoids the dInfer/vLLM model stack.  It verifies that the
installed DeepEP package can initialize a distributed Buffer and complete a
simple identity expert round trip on the current 8-GPU node.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def _init_dist() -> tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _all_gather_dict(data: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [None for _ in range(dist.get_world_size())]  # type: ignore[list-item]
    dist.all_gather_object(out, data)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--buffer-mb", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rank, world_size, local_rank = _init_dist()
    device = torch.device("cuda", local_rank)

    started = time.perf_counter()
    status: dict[str, Any] = {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "device_name": torch.cuda.get_device_name(device),
        "tokens": args.tokens,
        "hidden": args.hidden,
        "top_k": args.top_k,
        "ok": False,
        "stage": "start",
        "error": None,
    }

    try:
        import deep_ep
        import deep_ep_cpp

        status["stage"] = "import"
        status["deep_ep_file"] = getattr(deep_ep, "__file__", None)
        status["deep_ep_cpp_file"] = getattr(deep_ep_cpp, "__file__", None)
        status["deep_ep_cpp_sm90"] = bool(deep_ep.Buffer.is_sm90_compiled())

        if args.top_k != 1:
            raise ValueError("This smoke currently expects --top-k 1.")
        if args.hidden * torch.bfloat16.itemsize % 512 != 0:
            raise ValueError(
                "DeepEP HT intranode path expects hidden bytes to be 512-byte aligned; "
                f"got hidden={args.hidden} bf16 bytes={args.hidden * torch.bfloat16.itemsize}"
            )

        num_experts = args.num_experts or world_size
        if num_experts % world_size != 0:
            raise ValueError(
                f"num_experts must be divisible by world_size: {num_experts} vs {world_size}"
            )
        status["num_experts"] = num_experts

        torch.manual_seed(1000 + rank)
        x = torch.randn(args.tokens, args.hidden, device=device, dtype=torch.bfloat16)
        token_ids = torch.arange(args.tokens, device=device, dtype=torch.long)
        topk_idx = ((token_ids + rank) % num_experts).view(args.tokens, 1).contiguous()
        topk_weights = torch.ones(args.tokens, 1, device=device, dtype=torch.float32)

        status["stage"] = "buffer_init"
        buffer = deep_ep.Buffer(
            group=dist.group.WORLD,
            num_nvl_bytes=args.buffer_mb * 1024 * 1024,
            num_rdma_bytes=0,
            low_latency_mode=False,
            num_qps_per_rank=1,
        )
        torch.cuda.synchronize()
        dist.barrier()

        status["stage"] = "layout"
        layout = buffer.get_dispatch_layout(
            topk_idx=topk_idx,
            num_experts=num_experts,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _layout_event,
        ) = layout

        status["tokens_per_rank"] = [int(v) for v in num_tokens_per_rank.cpu().tolist()]
        status["tokens_per_rdma_rank"] = (
            None
            if num_tokens_per_rdma_rank is None
            else [int(v) for v in num_tokens_per_rdma_rank.cpu().tolist()]
        )
        status["tokens_per_expert"] = [int(v) for v in num_tokens_per_expert.cpu().tolist()]

        status["stage"] = "dispatch"
        recv_x, recv_topk_idx, recv_topk_weights, recv_per_expert, handle, event = buffer.dispatch(
            x=x,
            handle=None,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            expert_alignment=1,
            config=deep_ep.Buffer.get_dispatch_config(world_size),
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        if event.event is not None:
            event.current_stream_wait()
        torch.cuda.synchronize()

        status["recv_shape"] = list(recv_x.shape)
        status["recv_topk_idx_shape"] = None if recv_topk_idx is None else list(recv_topk_idx.shape)
        status["recv_topk_weights_shape"] = (
            None if recv_topk_weights is None else list(recv_topk_weights.shape)
        )
        status["recv_per_expert"] = [int(v) for v in recv_per_expert]

        status["stage"] = "combine"
        combined, combined_topk_weights, combine_event = buffer.combine(
            x=recv_x,
            handle=handle,
            topk_weights=None,
            config=deep_ep.Buffer.get_combine_config(world_size),
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        if combine_event.event is not None:
            combine_event.current_stream_wait()
        torch.cuda.synchronize()
        dist.barrier()

        diff = (combined - x).float().abs()
        status["combined_shape"] = list(combined.shape)
        status["combined_topk_weights_shape"] = (
            None if combined_topk_weights is None else list(combined_topk_weights.shape)
        )
        status["max_abs_diff"] = float(diff.max().item()) if diff.numel() else 0.0
        status["mean_abs_diff"] = float(diff.mean().item()) if diff.numel() else 0.0
        status["shape_ok"] = list(combined.shape) == list(x.shape)
        status["numerical_ok"] = status["max_abs_diff"] <= 0.0
        status["ok"] = bool(status["shape_ok"] and status["numerical_ok"])
        status["stage"] = "done"
    except Exception as exc:  # noqa: BLE001 - smoke should report exact failure
        status["ok"] = False
        status["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        status["elapsed_s"] = time.perf_counter() - started

    gathered = _all_gather_dict(status)
    if rank == 0:
        result = {
            "ok": all(item.get("ok", False) for item in gathered),
            "world_size": world_size,
            "args": vars(args) | {"output": str(args.output) if args.output else None},
            "ranks": gathered,
        }
        encoded = json.dumps(result, indent=2, sort_keys=True)
        print(encoded)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n")

    dist.barrier()
    dist.destroy_process_group()
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
