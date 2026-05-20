"""
Test A: Verify NCCL variable-size all_gatherv / reduce_scatterv
using vllm's pynccl wrapper with synthetic data.

Run: torchrun --nproc_per_node=8 codex_coding/src/test_compact_collective.py
"""
import os
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"=== Test A: NCCL variable-size collectives ===")
        print(f"world_size={world_size}")

    # --- Setup pynccl communicator ---
    # PyNcclCommunicator needs a non-NCCL (gloo) group
    cpu_group = dist.new_group(ranks=list(range(world_size)), backend='gloo')
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    comm = PyNcclCommunicator(
        group=cpu_group,
        device=device,
    )

    hidden_dim = 2048
    dtype = torch.bfloat16

    # ============================================================
    # Test 1: all_gatherv with variable sizes
    # ============================================================
    # Simulate: each rank has different N_compute_sp
    base_count = 480
    sizes = [base_count + rank * 3 for rank in range(world_size)]
    # sizes = [480, 483, 486, 489, 492, 495, 498, 501]
    my_n = sizes[rank]
    total_n = sum(sizes)

    # Create input: unique data per rank (small values for bf16 precision)
    input_tensor = (torch.arange(my_n, device=device, dtype=dtype).unsqueeze(1).expand(-1, hidden_dim).contiguous() + 1) * 0.01
    input_tensor = input_tensor + rank * 0.1  # small offset per rank

    # AllGatherV
    output_ag = torch.empty(total_n, hidden_dim, device=device, dtype=dtype)
    comm.all_gatherv(output_ag, input_tensor, sizes=sizes)
    torch.cuda.synchronize()

    # Verify: check each rank's chunk in the gathered output
    offset = 0
    all_correct = True
    for r in range(world_size):
        chunk = output_ag[offset:offset + sizes[r]]
        expected_first = (0 + 1) * 0.01 + r * 0.1
        actual_first = chunk[0, 0].item()
        if abs(actual_first - expected_first) > 0.01:  # bf16 tolerance
            if rank == 0:
                print(f"  FAIL: rank {r} chunk[0,0]={actual_first:.4f}, expected={expected_first:.4f}")
            all_correct = False
        offset += sizes[r]

    if rank == 0:
        print(f"Test 1 (all_gatherv): sizes={sizes}, total={total_n}")
        print(f"  Result: {'PASS' if all_correct else 'FAIL'}")

    # ============================================================
    # Test 2: reduce_scatterv with variable sizes
    # ============================================================
    # Input to reduce_scatterv: full gathered tensor (same on all ranks after all_gatherv)
    # But for reduce, each rank contributes its own version
    # After reduce_scatter: rank r gets sum of all ranks' contribution for its chunk

    # Create input: each rank has a full-size tensor
    input_rs = torch.ones(total_n, hidden_dim, device=device, dtype=dtype) * (rank + 1)

    # ReduceScatterV: rank r gets sizes[r] elements
    output_rs = torch.empty(my_n, hidden_dim, device=device, dtype=dtype)
    comm.reduce_scatterv(output_rs, input_rs, sizes=sizes)
    torch.cuda.synchronize()

    # Verify: each element should be sum of (rank+1) for all ranks = 1+2+...+8 = 36
    expected_sum = sum(range(1, world_size + 1))  # 36 for 8 ranks
    actual_val = output_rs[0, 0].item()
    rs_correct = abs(actual_val - expected_sum) < 1

    if rank == 0:
        print(f"Test 2 (reduce_scatterv): my_n={my_n}, expected_sum={expected_sum}, actual={actual_val}")
        print(f"  Result: {'PASS' if rs_correct else 'FAIL'}")

    # ============================================================
    # Test 3: roundtrip (all_gatherv → reduce_scatterv)
    # ============================================================
    # This simulates: dispatch(compact) → compute → combine(compact)
    # After roundtrip with world_size=8, result = input * world_size (because reduce sums)

    # Recreate clean input
    input_rt = torch.randn(my_n, hidden_dim, device=device, dtype=dtype)

    # Step 1: AllGatherV
    gathered = torch.empty(total_n, hidden_dim, device=device, dtype=dtype)
    comm.all_gatherv(gathered, input_rt, sizes=sizes)
    torch.cuda.synchronize()

    # Step 2: ReduceScatterV (same gathered tensor on all ranks → sum → scatter)
    output_rt = torch.empty(my_n, hidden_dim, device=device, dtype=dtype)
    comm.reduce_scatterv(output_rt, gathered, sizes=sizes)
    torch.cuda.synchronize()

    # After gather + reduce_scatter: output = input * world_size
    # Because all_gatherv puts input on all ranks, reduce_scatterv sums world_size copies
    expected_rt = input_rt * world_size
    diff_rt = (output_rt - expected_rt).abs().max().item()
    rt_correct = diff_rt < 1.0  # bf16 accumulation of 8 ranks

    if rank == 0:
        print(f"Test 3 (roundtrip): diff_max={diff_rt:.6f}")
        print(f"  Result: {'PASS' if rt_correct else 'FAIL'}")

    # ============================================================
    # Test 4: all_gatherv + reduce_scatterv matching dispatch/combine semantics
    # ============================================================
    # In MoE: dispatch gathers ALL ranks' tokens, combine reduces back
    # The combine should give each rank its LOCAL contribution to the total output
    # Simulate: rank r's MoE output is processed_gathered (same on all ranks after kernel)
    # combine should give rank r the reduce of its chunk

    # Each rank produces unique compact hidden states
    hs_sp = torch.randn(my_n, hidden_dim, device=device, dtype=dtype)

    # Dispatch: AllGatherV
    hs_g = torch.empty(total_n, hidden_dim, device=device, dtype=dtype)
    comm.all_gatherv(hs_g, hs_sp, sizes=sizes)
    torch.cuda.synchronize()

    # Simulate MoE: just pass through (identity)
    moe_output = hs_g.clone()

    # Combine: ReduceScatterV
    y_sp = torch.empty(my_n, hidden_dim, device=device, dtype=dtype)
    comm.reduce_scatterv(y_sp, moe_output, sizes=sizes)
    torch.cuda.synchronize()

    # After identity MoE: combine output should = hs_sp * world_size
    diff_dc = (y_sp - hs_sp * world_size).abs().max().item()
    dc_correct = diff_dc < 1.0

    if rank == 0:
        print(f"Test 4 (dispatch→identity→combine): diff_max={diff_dc:.6f}")
        print(f"  Result: {'PASS' if dc_correct else 'FAIL'}")

    # ============================================================
    # Test 5: extreme size variation
    # ============================================================
    extreme_sizes = [100, 2000, 50, 1500, 300, 1800, 75, 1200]
    my_ext_n = extreme_sizes[rank]
    total_ext = sum(extreme_sizes)

    input_ext = torch.randn(my_ext_n, hidden_dim, device=device, dtype=dtype)
    gathered_ext = torch.empty(total_ext, hidden_dim, device=device, dtype=dtype)
    comm.all_gatherv(gathered_ext, input_ext, sizes=extreme_sizes)

    output_ext = torch.empty(my_ext_n, hidden_dim, device=device, dtype=dtype)
    comm.reduce_scatterv(output_ext, gathered_ext, sizes=extreme_sizes)
    torch.cuda.synchronize()

    diff_ext = (output_ext - input_ext * world_size).abs().max().item()
    ext_correct = diff_ext < 1.0

    if rank == 0:
        print(f"Test 5 (extreme sizes {extreme_sizes}): diff_max={diff_ext:.6f}")
        print(f"  Result: {'PASS' if ext_correct else 'FAIL'}")

    # ============================================================
    # Test 6: dynamic sizes — change sizes between consecutive calls
    # Simulates: forward step 1 has different N_compute_sp than step 2
    # ============================================================
    sizes_step1 = [480 + rank * 3 for rank in range(world_size)]
    sizes_step2 = [200 + rank * 7 for rank in range(world_size)]  # very different!
    sizes_step3 = [600 - rank * 5 for rank in range(world_size)]  # another pattern

    dyn_correct = True
    for step_i, step_sizes in enumerate([sizes_step1, sizes_step2, sizes_step3]):
        my_step_n = step_sizes[rank]
        total_step = sum(step_sizes)
        inp = torch.randn(my_step_n, hidden_dim, device=device, dtype=dtype)
        gathered = torch.empty(total_step, hidden_dim, device=device, dtype=dtype)
        comm.all_gatherv(gathered, inp, sizes=step_sizes)
        out = torch.empty(my_step_n, hidden_dim, device=device, dtype=dtype)
        comm.reduce_scatterv(out, gathered, sizes=step_sizes)
        torch.cuda.synchronize()
        diff = (out - inp * world_size).abs().max().item()
        if diff >= 1.0:
            dyn_correct = False
            if rank == 0:
                print(f"  Step {step_i} FAIL: sizes={step_sizes}, diff={diff:.6f}")

    if rank == 0:
        print(f"Test 6 (dynamic sizes across 3 steps): {'PASS' if dyn_correct else 'FAIL'}")

    # ============================================================
    # Test 7: rapid alternation — simulate 19 layers × multiple steps
    # Each "step" uses different sizes, 19 consecutive gather+scatter pairs
    # ============================================================
    import random
    random.seed(42)
    n_steps = 5
    n_layers = 19
    rapid_correct = True

    for step in range(n_steps):
        # Each step has a different compact size per rank
        step_sizes = [random.randint(300, 600) for _ in range(world_size)]
        my_sn = step_sizes[rank]
        total_sn = sum(step_sizes)

        for layer in range(n_layers):
            inp = torch.randn(my_sn, hidden_dim, device=device, dtype=dtype)
            gathered = torch.empty(total_sn, hidden_dim, device=device, dtype=dtype)
            comm.all_gatherv(gathered, inp, sizes=step_sizes)
            out = torch.empty(my_sn, hidden_dim, device=device, dtype=dtype)
            comm.reduce_scatterv(out, gathered, sizes=step_sizes)
            torch.cuda.synchronize()
            diff = (out - inp * world_size).abs().max().item()
            if diff >= 1.0:
                rapid_correct = False
                if rank == 0:
                    print(f"  Step {step} Layer {layer} FAIL: diff={diff:.6f}")

    if rank == 0:
        print(f"Test 7 (5 steps × 19 layers, dynamic sizes): {'PASS' if rapid_correct else 'FAIL'}")

    # ============================================================
    # Test 8: sizes change AND equal-size mixed
    # Simulates: some forwards use compact (variable), some use full (equal)
    # ============================================================
    full_size = 2048  # all ranks same
    full_sizes = [full_size] * world_size
    compact_test_sizes = [486, 490, 483, 495, 488, 492, 480, 487]

    mixed_correct = True
    for iteration in range(10):
        if iteration % 3 == 0:
            # Full (equal sizes) — like G path / refresh step
            use_sizes = full_sizes
        else:
            # Compact (variable sizes) — like sparse step
            use_sizes = compact_test_sizes

        my_mn = use_sizes[rank]
        total_mn = sum(use_sizes)
        inp = torch.randn(my_mn, hidden_dim, device=device, dtype=dtype)
        gathered = torch.empty(total_mn, hidden_dim, device=device, dtype=dtype)
        comm.all_gatherv(gathered, inp, sizes=use_sizes)
        out = torch.empty(my_mn, hidden_dim, device=device, dtype=dtype)
        comm.reduce_scatterv(out, gathered, sizes=use_sizes)
        torch.cuda.synchronize()
        diff = (out - inp * world_size).abs().max().item()
        if diff >= 1.0:
            mixed_correct = False
            if rank == 0:
                print(f"  Iter {iteration} FAIL: sizes={use_sizes}, diff={diff:.6f}")

    if rank == 0:
        print(f"Test 8 (full/compact alternation, 10 iters): {'PASS' if mixed_correct else 'FAIL'}")
        print()
        all_pass = (all_correct and rs_correct and rt_correct and dc_correct
                    and ext_correct and dyn_correct and rapid_correct and mixed_correct)
        print(f"=== ALL {'PASS' if all_pass else 'FAIL'} ===")

    comm.del_comm()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
