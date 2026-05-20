#!/usr/bin/env python3
"""
Dispatch/Combine overhead isolation experiment.

Groups:
  1. Pure NCCL 19× pre-alloc (baseline)
  2. + torch.empty per call (alloc cost)
  3. + dummy kernel between calls (kernel gap)
  4. + kernel + explicit sync (stream barrier)
  5. Full cycle D→K→C pre-alloc (interleave cost)
  6. Full cycle + alloc (full simulation)
  7. Full cycle fp8 (comm compression)

Usage:
  torchrun --nproc_per_node=8 nccl_overhead_isolation.py
"""

import time
import torch
import torch.distributed as dist


def bench(fn, warmup=5, repeat=15):
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

    # Sizes matching GS path batch=512
    # dispatch: 2048 tokens × 2048 hidden = 8 MB per rank (bf16)
    # combine: 16384 tokens × 2048 hidden = 64 MB total input
    HS = 2048
    N_LOCAL = 2048  # tokens per rank (SP)
    N_TOTAL = N_LOCAL * world  # 16384 after AllGather

    # Pre-allocated buffers
    local_hs = torch.randn(N_LOCAL, HS, device=device, dtype=torch.bfloat16)
    out_hs = torch.empty(N_TOTAL, HS, device=device, dtype=torch.bfloat16)
    rs_input = torch.randn(N_TOTAL, HS, device=device, dtype=torch.bfloat16)
    rs_out = torch.empty(N_LOCAL, HS, device=device, dtype=torch.bfloat16)

    # Dummy kernel (~1ms matmul to simulate per-layer MoE kernel)
    dummy_a = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    dummy_b = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)

    # fp8 buffers (half the bytes)
    local_fp8 = torch.zeros(N_LOCAL, HS // 2, device=device, dtype=torch.float8_e4m3fn)
    out_fp8 = torch.empty(N_TOTAL, HS // 2, device=device, dtype=torch.float8_e4m3fn)
    rs_input_fp8 = torch.zeros(N_TOTAL, HS // 2, device=device, dtype=torch.float8_e4m3fn)
    rs_out_fp8 = torch.empty(N_LOCAL, HS // 2, device=device, dtype=torch.float8_e4m3fn)

    results = {}

    # === Group 1: Pure NCCL 19× pre-alloc ===
    def g1_ag():
        for _ in range(19):
            dist.all_gather_into_tensor(out_hs.view(-1), local_hs.view(-1))

    def g1_rs():
        for _ in range(19):
            dist.reduce_scatter_tensor(rs_out.view(-1), rs_input.view(-1))

    t_ag = bench(g1_ag)
    t_rs = bench(g1_rs)
    results["G1_ag"] = t_ag
    results["G1_rs"] = t_rs
    results["G1_total"] = t_ag + t_rs

    # === Group 2: + torch.empty per call ===
    def g2_ag():
        for _ in range(19):
            o = torch.empty(N_TOTAL, HS, device=device, dtype=torch.bfloat16)
            dist.all_gather_into_tensor(o.view(-1), local_hs.view(-1))

    def g2_rs():
        for _ in range(19):
            o = torch.empty(N_LOCAL, HS, device=device, dtype=torch.bfloat16)
            dist.reduce_scatter_tensor(o.view(-1), rs_input.view(-1))

    t_ag = bench(g2_ag)
    t_rs = bench(g2_rs)
    results["G2_ag"] = t_ag
    results["G2_rs"] = t_rs
    results["G2_total"] = t_ag + t_rs

    # === Group 3: + dummy kernel between calls ===
    def g3_ag():
        for _ in range(19):
            torch.mm(dummy_a, dummy_b)
            dist.all_gather_into_tensor(out_hs.view(-1), local_hs.view(-1))

    def g3_rs():
        for _ in range(19):
            torch.mm(dummy_a, dummy_b)
            dist.reduce_scatter_tensor(rs_out.view(-1), rs_input.view(-1))

    t_ag = bench(g3_ag)
    t_rs = bench(g3_rs)
    results["G3_ag"] = t_ag
    results["G3_rs"] = t_rs
    results["G3_total"] = t_ag + t_rs

    # === Group 4: + kernel + explicit sync ===
    def g4_ag():
        for _ in range(19):
            torch.mm(dummy_a, dummy_b)
            torch.cuda.synchronize()
            dist.all_gather_into_tensor(out_hs.view(-1), local_hs.view(-1))

    def g4_rs():
        for _ in range(19):
            torch.mm(dummy_a, dummy_b)
            torch.cuda.synchronize()
            dist.reduce_scatter_tensor(rs_out.view(-1), rs_input.view(-1))

    t_ag = bench(g4_ag)
    t_rs = bench(g4_rs)
    results["G4_ag"] = t_ag
    results["G4_rs"] = t_rs
    results["G4_total"] = t_ag + t_rs

    # === Group 5: Full cycle D→K→C pre-alloc ===
    def g5():
        for _ in range(19):
            dist.all_gather_into_tensor(out_hs.view(-1), local_hs.view(-1))
            torch.mm(dummy_a, dummy_b)
            dist.reduce_scatter_tensor(rs_out.view(-1), out_hs.view(-1))

    results["G5_total"] = bench(g5)

    # === Group 6: Full cycle + alloc ===
    def g6():
        for _ in range(19):
            o_ag = torch.empty(N_TOTAL, HS, device=device, dtype=torch.bfloat16)
            dist.all_gather_into_tensor(o_ag.view(-1), local_hs.view(-1))
            torch.mm(dummy_a, dummy_b)
            o_rs = torch.empty(N_LOCAL, HS, device=device, dtype=torch.bfloat16)
            dist.reduce_scatter_tensor(o_rs.view(-1), o_ag.view(-1))

    results["G6_total"] = bench(g6)

    # === Group 7: Full cycle fp8 ===
    def g7():
        for _ in range(19):
            dist.all_gather_into_tensor(out_fp8.view(-1), local_fp8.view(-1))
            torch.mm(dummy_a, dummy_b)
            dist.reduce_scatter_tensor(rs_out_fp8.view(-1), rs_input_fp8.view(-1))

    results["G7_total"] = bench(g7)

    # === Measure kernel-only time ===
    def kernel_only():
        for _ in range(19):
            torch.mm(dummy_a, dummy_b)

    results["kernel_19x"] = bench(kernel_only)

    # === Print results ===
    if rank == 0:
        k19 = results["kernel_19x"]
        print(f"NCCL Overhead Isolation (8 GPUs, 19 layers, batch=512 equivalent)")
        print(f"  dispatch payload/layer: {N_LOCAL*HS*2/1e6:.1f} MB (bf16)")
        print(f"  combine payload/layer:  {N_TOTAL*HS*2/1e6:.1f} MB (bf16)")
        print(f"  dummy kernel 19×:       {k19:.3f} ms")
        print()
        print(f"{'Group':<6} {'Description':<42} {'Time(ms)':>8} {'vs G1':>8} {'Note'}")
        print("-" * 85)
        g1 = results["G1_total"]
        print(f"{'G1':<6} {'Pure NCCL 19× pre-alloc (AG+RS)':<42} "
              f"{g1:>8.3f} {'—':>8} baseline")
        print(f"{'  AG':<6} {'  AllGather 19×':<42} {results['G1_ag']:>8.3f}")
        print(f"{'  RS':<6} {'  ReduceScatter 19×':<42} {results['G1_rs']:>8.3f}")
        print()

        g2 = results["G2_total"]
        print(f"{'G2':<6} {'+ torch.empty per call':<42} "
              f"{g2:>8.3f} {f'+{g2-g1:.1f}':>8} alloc cost")
        print(f"{'  AG':<6} {'  AllGather 19×':<42} {results['G2_ag']:>8.3f}")
        print(f"{'  RS':<6} {'  ReduceScatter 19×':<42} {results['G2_rs']:>8.3f}")
        print()

        g3 = results["G3_total"]
        print(f"{'G3':<6} {'+ dummy kernel between (no sync)':<42} "
              f"{g3:>8.3f} {f'+{g3-g1:.1f}':>8} kernel gap")
        print(f"{'  AG':<6} {'  AllGather 19×':<42} {results['G3_ag']:>8.3f}")
        print(f"{'  RS':<6} {'  ReduceScatter 19×':<42} {results['G3_rs']:>8.3f}")
        print()

        g4 = results["G4_total"]
        print(f"{'G4':<6} {'+ dummy kernel + cuda.sync':<42} "
              f"{g4:>8.3f} {f'+{g4-g1:.1f}':>8} sync overhead")
        print(f"{'  AG':<6} {'  AllGather 19×':<42} {results['G4_ag']:>8.3f}")
        print(f"{'  RS':<6} {'  ReduceScatter 19×':<42} {results['G4_rs']:>8.3f}")
        print()

        g5 = results["G5_total"]
        print(f"{'G5':<6} {'Full D→K→C cycle pre-alloc':<42} "
              f"{g5:>8.3f} {f'+{g5-g1:.1f}':>8} interleave")
        print()

        g6 = results["G6_total"]
        print(f"{'G6':<6} {'Full D→K→C + alloc per call':<42} "
              f"{g6:>8.3f} {f'+{g6-g1:.1f}':>8} full sim")
        print()

        g7 = results["G7_total"]
        print(f"{'G7':<6} {'Full D→K→C fp8 (half bytes)':<42} "
              f"{g7:>8.3f} {f'+{g7-g1:.1f}':>8} fp8 comm")
        print()

        print("=" * 85)
        print("ANALYSIS:")
        print(f"  Alloc overhead:     G2-G1 = {g2-g1:.3f} ms")
        print(f"  Kernel gap cost:    G3-G2 = {g3-g2:.3f} ms "
              f"(kernel alone = {k19:.3f} ms)")
        print(f"  Sync vs no-sync:    G4-G3 = {g4-g3:.3f} ms")
        print(f"  Interleave cost:    G5-(G1+kernel) = {g5-g1-k19:.3f} ms")
        print(f"  Alloc in full sim:  G6-G5 = {g6-g5:.3f} ms")
        print(f"  fp8 savings:        G5-G7 = {g5-g7:.3f} ms ({(g5-g7)/g5*100:.1f}%)")
        print()
        print(f"  Actual inference:   dispatch={8.0:.1f} + combine={13.7:.1f} = 21.7 ms")
        print(f"  Best micro-sim (G6): {g6:.1f} ms")
        print(f"  Remaining gap:      {21.7-g6:.1f} ms (Python/routing/gate/timer overhead)")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
