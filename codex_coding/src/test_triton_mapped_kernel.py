"""
Standalone test for Triton mapped kernel scattered read behavior.
Goal: isolate whether large-offset / scattered access via input_map
produces wrong results in _fused_moe_kernel_mapped.

Run: python codex_coding/src/test_triton_mapped_kernel.py
     python codex_coding/src/test_triton_mapped_kernel.py --real  (use dumped real data)
(single GPU, no torchrun needed)
"""
import torch
import triton
import triton.language as tl
import sys

# ============================================================
# Kernel definition (from bench_bsp_moe_dp2.py)
# ============================================================

@triton.jit
def _fused_moe_kernel_mapped(
    a_ptr, b_ptr, c_ptr,
    input_map_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr, compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # Expert not on this rank — write zeros
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask)
        return

    # Always use input_map for indirection
    max_row_idx = tl.maximum(num_valid_tokens // top_k - 1, 0)
    compact_tok = tl.minimum(offs_token // top_k, max_row_idx)
    a_row_idx = tl.load(input_map_ptr + compact_tok.to(tl.int64),
                        mask=token_mask,
                        other=0).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (a_row_idx[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_mapped_kernel(A, B, C, input_map, topk_weights,
                         sorted_token_ids, expert_ids,
                         num_tokens_post_padded,
                         mul_routed_weight, top_k,
                         num_valid_tokens, config):
    EM = sorted_token_ids.size(0)
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.size(1), META['BLOCK_SIZE_N']), )
    C_flat = C.view(-1, C.size(-1)) if C.ndim == 3 else C
    _fused_moe_kernel_mapped[grid](
        A, B, C_flat,
        input_map,
        topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        B.size(1), A.size(1), EM, num_valid_tokens,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C_flat.stride(0), C_flat.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k,
        compute_type=tl.bfloat16,
        **config,
    )


# ============================================================
# Test with real dumped data
# ============================================================

def test_real_data():
    """Load real tensors from bench script and reproduce the discrepancy."""
    dump_dir = "/tmp/tv4m_debug"
    print(f"Loading real data from {dump_dir}...")
    d = torch.load(f"{dump_dir}/tv4m_tensors_rank0.pt", map_location='cpu')
    o = torch.load(f"{dump_dir}/tv4m_outputs_rank0.pt", map_location='cpu')

    device = 'cuda'
    hs_g = d['hs_g'].to(device)
    hs_extract = d['hs_extract'].to(device)
    w13 = d['w13_weight'].to(device)
    input_map = d['input_map'].to(device)
    c_wts = d['c_wts'].to(device)
    sorted_token_ids = d['sorted_token_ids'].to(device)
    expert_ids = d['expert_ids'].to(device)
    num_tokens_post_padded = d['num_tokens_post_padded'].to(device)
    n_valid = d['n_valid']
    top_k_num = d['top_k_num']
    N = d['N']
    K = d['K']
    N_compute = d['N_compute']
    config = d['config']
    mul_weight = d['apply_router_weight_on_input']
    expert_map = d['expert_map']

    ref_output = o['ref_output'].to(device)
    bench_identity_output = o['identity_output'].to(device)
    bench_mapped_output = o['mapped_output'].to(device)

    print(f"  N_total={hs_g.shape[0]}, N_compute={N_compute}, K={K}, N={N}")
    print(f"  top_k={top_k_num}, n_valid={n_valid}")
    print(f"  config={config}")
    print(f"  mul_weight={mul_weight}")
    print(f"  input_map: dtype={input_map.dtype}, min={input_map.min().item()}, max={input_map.max().item()}")
    print(f"  sorted_token_ids: dtype={sorted_token_ids.dtype}, shape={sorted_token_ids.shape}, max={sorted_token_ids.max().item()}")
    print(f"  expert_ids: dtype={expert_ids.dtype}, shape={expert_ids.shape}, min={expert_ids.min().item()}, max={expert_ids.max().item()}")
    print(f"  expert_map: {expert_map}")
    print(f"  w13: shape={list(w13.shape)}")
    print(f"  c_wts: shape={list(c_wts.shape)}, c_wts.view(-1).shape={c_wts.view(-1).shape}")
    print()

    # Verify data identity
    verify = (hs_g[input_map] - hs_extract).abs().max().item()
    print(f"  Data verify: hs_g[input_map] == hs_extract: diff={verify}")
    print()

    # --- Run our kernel: identity map + hs_extract ---
    C_identity = torch.zeros(n_valid, N, device=device, dtype=torch.bfloat16)
    identity_map = torch.arange(N_compute, device=device, dtype=torch.int64)
    invoke_mapped_kernel(
        hs_extract, w13, C_identity, identity_map, c_wts.view(-1),
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_weight, top_k_num, n_valid, config)

    # --- Run our kernel: real map + hs_g ---
    C_mapped = torch.zeros(n_valid, N, device=device, dtype=torch.bfloat16)
    invoke_mapped_kernel(
        hs_g, w13, C_mapped, input_map, c_wts.view(-1),
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_weight, top_k_num, n_valid, config)

    # Compare
    d_identity = (C_identity - ref_output).abs()
    d_mapped = (C_mapped - ref_output).abs()
    d_identity_vs_bench = (C_identity - bench_identity_output).abs()
    d_mapped_vs_bench = (C_mapped - bench_mapped_output).abs()

    print("--- Results ---")
    print(f"  identity_vs_ref:     max={d_identity.max().item():.6f}  mean={d_identity.mean().item():.8f}")
    print(f"  mapped_vs_ref:       max={d_mapped.max().item():.6f}  mean={d_mapped.mean().item():.8f}")
    print(f"  identity_vs_bench:   max={d_identity_vs_bench.max().item():.6f}  (should be ~0)")
    print(f"  mapped_vs_bench:     max={d_mapped_vs_bench.max().item():.6f}  (should be ~0)")
    print()

    if d_mapped.max().item() > 0.5:
        print("  *** REPRODUCED: mapped kernel has large diff ***")
        print()
        # Find where the diff is largest
        max_row = d_mapped.max(dim=1).values.argmax().item()
        print(f"  Worst row: {max_row} (token={max_row // top_k_num}, slot={max_row % top_k_num})")
        print(f"    ref[{max_row},:5]   = {ref_output[max_row,:5].tolist()}")
        print(f"    mapped[{max_row},:5] = {C_mapped[max_row,:5].tolist()}")
        print(f"    identity[{max_row},:5] = {C_identity[max_row,:5].tolist()}")
        print()

        # What input_map value does this row use?
        tok_idx = max_row // top_k_num
        print(f"  Token {tok_idx}: input_map[{tok_idx}] = {input_map[tok_idx].item()}")
        print(f"    hs_g[{input_map[tok_idx].item()},:5] = {hs_g[input_map[tok_idx].item(),:5].tolist()}")
        print(f"    hs_extract[{tok_idx},:5] = {hs_extract[tok_idx,:5].tolist()}")
    else:
        print("  ✓ No discrepancy found — kernel works correctly with real data!")
        print("  The issue must be in the bench script's calling context.")


# ============================================================
# Synthetic test (original experiment matrix)
# ============================================================

def test_synthetic():
    """Run synthetic experiment matrix."""
    torch.manual_seed(42)
    device = 'cuda'

    N_total = 16384
    N_compute = 3890
    hidden_dim = 2048
    inter_dim = 1024
    num_experts = 32
    top_k = 4

    print(f"=== Synthetic Experiment Matrix ===")
    print(f"N_total={N_total}, N_compute={N_compute}, hidden={hidden_dim}, "
          f"inter={inter_dim}, E={num_experts}, top_k={top_k}")
    print()

    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

    A_full = torch.randn(N_total, hidden_dim, device=device, dtype=torch.bfloat16)
    B = torch.randn(num_experts, inter_dim, hidden_dim, device=device, dtype=torch.bfloat16) * 0.01
    topk_ids = torch.randint(0, num_experts, (N_compute, top_k), device=device, dtype=torch.int32)
    topk_weights = torch.rand(N_compute, top_k, device=device, dtype=torch.bfloat16)
    config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config['BLOCK_SIZE_M'], num_experts, None)
    num_valid_tokens = N_compute * top_k

    compute_indices = torch.sort(
        torch.randperm(N_total, device=device)[:N_compute]).values.to(torch.int64)
    A_extract = A_full[compute_indices].contiguous()

    identity_map = torch.arange(N_compute, device=device, dtype=torch.int64)

    # Reference
    C_ref = torch.zeros(num_valid_tokens, inter_dim, device=device, dtype=torch.bfloat16)
    invoke_mapped_kernel(A_extract, B, C_ref, identity_map, topk_weights.view(-1),
                         sorted_token_ids, expert_ids, num_tokens_post_padded,
                         False, top_k, num_valid_tokens, config)

    # Mapped (scattered)
    C_mapped = torch.zeros(num_valid_tokens, inter_dim, device=device, dtype=torch.bfloat16)
    invoke_mapped_kernel(A_full, B, C_mapped, compute_indices, topk_weights.view(-1),
                         sorted_token_ids, expert_ids, num_tokens_post_padded,
                         False, top_k, num_valid_tokens, config)

    diff = (C_mapped - C_ref).abs()
    print(f"  identity_vs_mapped: max={diff.max().item():.6f}  mean={diff.mean().item():.8f}")
    if diff.max().item() < 0.5:
        print("  ✓ PASS — scattered access works correctly")
    else:
        print("  ✗ FAIL — discrepancy found")


if __name__ == "__main__":
    if "--real" in sys.argv:
        test_real_data()
    else:
        test_synthetic()
