#!/usr/bin/env python3
"""
NCCL Communication Micro-benchmark for EP dispatch/combine analysis.

Measures AllGather and ReduceScatter performance on ep_group (8 ranks)
at various message sizes, bf16 vs fp8, single vs 19-loop.

Usage:
  torchrun --nproc_per_node=8 nccl_comm_bench.py
"""

import time
import torch
import torch.distributed as dist


def bench_collective(fn, warmup=5, repeat=20):
    """Benchmark a collective operation, return median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"NCCL Comm Bench: {world} GPUs")
        print(f"{'size_MB':>8} {'AG_bf16':>8} {'AG_fp8':>8} {'RS_bf16':>8} "
              f"{'AG_19x':>8} {'RS_19x':>8} {'AG_fp8_19x':>8}")
        print("-" * 62)

    sizes_mb = [1, 2, 4, 8, 9, 16, 32, 64, 128, 256]

    for size_mb in sizes_mb:
        numel_bf16 = size_mb * 1024 * 1024 // 2  # bf16 = 2 bytes
        numel_fp8 = size_mb * 1024 * 1024  # fp8 = 1 byte, same MB

        # Buffers
        local_bf16 = torch.randn(numel_bf16, device=device, dtype=torch.bfloat16)
        gathered_bf16 = torch.empty(numel_bf16 * world, device=device, dtype=torch.bfloat16)

        # fp8: half the numel to get half the bytes
        numel_half = numel_bf16 // 2
        local_fp8 = torch.zeros(numel_half, device=device, dtype=torch.float8_e4m3fn)
        gathered_fp8 = torch.empty(numel_half * world, device=device, dtype=torch.float8_e4m3fn)

        rs_input_bf16 = torch.randn(numel_bf16 * world, device=device, dtype=torch.bfloat16)
        rs_output_bf16 = torch.empty(numel_bf16, device=device, dtype=torch.bfloat16)

        # AllGather bf16
        def ag_bf16():
            dist.all_gather_into_tensor(gathered_bf16, local_bf16)

        # AllGather fp8 (half the bytes)
        def ag_fp8():
            dist.all_gather_into_tensor(gathered_fp8, local_fp8)

        # ReduceScatter bf16
        def rs_bf16():
            dist.reduce_scatter_tensor(rs_output_bf16, rs_input_bf16)

        t_ag_bf16 = bench_collective(ag_bf16)
        t_ag_fp8 = bench_collective(ag_fp8)
        t_rs_bf16 = bench_collective(rs_bf16)

        # 19x loop (simulating per-layer calls)
        def ag_bf16_19x():
            for _ in range(19):
                dist.all_gather_into_tensor(gathered_bf16, local_bf16)

        def rs_bf16_19x():
            for _ in range(19):
                dist.reduce_scatter_tensor(rs_output_bf16, rs_input_bf16)

        def ag_fp8_19x():
            for _ in range(19):
                dist.all_gather_into_tensor(gathered_fp8, local_fp8)

        t_ag_19 = bench_collective(ag_bf16_19x, warmup=3, repeat=10)
        t_rs_19 = bench_collective(rs_bf16_19x, warmup=3, repeat=10)
        t_ag_fp8_19 = bench_collective(ag_fp8_19x, warmup=3, repeat=10)

        if rank == 0:
            print(f"{size_mb:>8} {t_ag_bf16:>8.3f} {t_ag_fp8:>8.3f} {t_rs_bf16:>8.3f} "
                  f"{t_ag_19:>8.3f} {t_rs_19:>8.3f} {t_ag_fp8_19:>8.3f}")

    # Special test: actual dispatch/combine payload sizes
    if rank == 0:
        print()
        print("=== Actual dispatch/combine payload simulation ===")
        print(f"{'desc':>30} {'time_ms':>8} {'BW_GBs':>8}")
        print("-" * 50)

    # GS dispatch: 2048 tokens × 2048 hidden × bf16 = 8 MB per rank
    dispatch_local = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    dispatch_gathered = torch.empty(2048 * world, 2048, device=device, dtype=torch.bfloat16)

    def gs_dispatch():
        dist.all_gather_into_tensor(
            dispatch_gathered.view(-1), dispatch_local.view(-1))

    t = bench_collective(gs_dispatch)
    payload = 2048 * 2048 * 2 * 7 / 8  # ring: each rank sends 7/8 of data
    bw = payload / (t / 1000) / 1e9
    if rank == 0:
        print(f"{'GS dispatch 1 layer (8MB)':>30} {t:>8.3f} {bw:>8.1f}")

    # GS dispatch × 19
    def gs_dispatch_19():
        for _ in range(19):
            dist.all_gather_into_tensor(
                dispatch_gathered.view(-1), dispatch_local.view(-1))

    t19 = bench_collective(gs_dispatch_19, warmup=3, repeat=10)
    bw19 = payload * 19 / (t19 / 1000) / 1e9
    if rank == 0:
        print(f"{'GS dispatch 19 layers':>30} {t19:>8.3f} {bw19:>8.1f}")

    # GS combine: 16384 tokens × 2048 hidden × bf16 = 64 MB total → RS
    combine_input = torch.randn(16384, 2048, device=device, dtype=torch.bfloat16)
    combine_output = torch.empty(2048, 2048, device=device, dtype=torch.bfloat16)

    def gs_combine():
        dist.reduce_scatter_tensor(
            combine_output.view(-1), combine_input.view(-1))

    t = bench_collective(gs_combine)
    payload_rs = 16384 * 2048 * 2 * 7 / 8
    bw_rs = payload_rs / (t / 1000) / 1e9
    if rank == 0:
        print(f"{'GS combine 1 layer (64MB)':>30} {t:>8.3f} {bw_rs:>8.1f}")

    # GS combine × 19
    def gs_combine_19():
        for _ in range(19):
            dist.reduce_scatter_tensor(
                combine_output.view(-1), combine_input.view(-1))

    t19 = bench_collective(gs_combine_19, warmup=3, repeat=10)
    bw19_rs = payload_rs * 19 / (t19 / 1000) / 1e9
    if rank == 0:
        print(f"{'GS combine 19 layers':>30} {t19:>8.3f} {bw19_rs:>8.1f}")

    # fp8 dispatch (half payload)
    dispatch_fp8 = dispatch_local.view(torch.float8_e4m3fn)[:2048*1024]
    dispatch_fp8_gathered = torch.empty(2048 * 1024 * world, device=device, dtype=torch.float8_e4m3fn)

    def gs_dispatch_fp8():
        dist.all_gather_into_tensor(dispatch_fp8_gathered, dispatch_fp8)

    t_fp8 = bench_collective(gs_dispatch_fp8)
    if rank == 0:
        print(f"{'GS dispatch fp8 1 layer (4MB)':>30} {t_fp8:>8.3f} {'':>8}")

    def gs_dispatch_fp8_19():
        for _ in range(19):
            dist.all_gather_into_tensor(dispatch_fp8_gathered, dispatch_fp8)

    t_fp8_19 = bench_collective(gs_dispatch_fp8_19, warmup=3, repeat=10)
    if rank == 0:
        print(f"{'GS dispatch fp8 19 layers':>30} {t_fp8_19:>8.3f} {'':>8}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
