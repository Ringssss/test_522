"""
Fused MoE Triton kernel for Epoch-Spark.

Based on vLLM's fused_moe_kernel but stripped of quantization complexity
and forward-context dependencies. Supports bf16/fp16 with SiLU-gated activation.

Flow:
  1. moe_align_block_size: sort tokens by expert assignment
  2. GEMM1: hidden @ w13.T → gate_up (gate concat up)
  3. SiLU activation: silu(gate) * up → intermediate
  4. GEMM2: intermediate @ w2.T → expert_out
  5. Weighted reduce by topk_weights
"""

import torch
import triton
import triton.language as tl


# ════════════════════════════════════════════════════════════════
# Token sorting kernel
# ════════════════════════════════════════════════════════════════

@triton.jit
def _moe_align_kernel(
    topk_ids_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    total_cnt_ptr,
    num_experts: tl.constexpr,
    tokens_per_expert_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offs = start + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    expert = tl.load(topk_ids_ptr + offs, mask=mask, other=0)
    # Atomic count per expert
    for i in range(BLOCK_SIZE):
        if start + i < numel:
            e = tl.load(topk_ids_ptr + start + i)
            tl.atomic_add(tokens_per_expert_ptr + e, 1)


def moe_align_block_size_simple(topk_ids, block_size, num_experts):
    """Sort token indices by expert assignment with block alignment.

    Returns: sorted_token_ids, expert_ids, num_tokens_post_padded
    """
    N_flat = topk_ids.numel()
    flat_ids = topk_ids.view(-1)

    tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    for e in range(num_experts):
        tokens_per_expert[e] = (flat_ids == e).sum()

    # Compute padded counts and cumsum for sorted positions
    padded_per_expert = ((tokens_per_expert + block_size - 1) // block_size) * block_size
    cumsum = torch.zeros(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
    cumsum[1:] = padded_per_expert.cumsum(0)
    total_padded = cumsum[-1].item()

    sorted_token_ids = torch.full((total_padded,), N_flat, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.full((total_padded // block_size,), -1, dtype=torch.int32, device=topk_ids.device)

    # Fill sorted positions
    write_pos = cumsum[:-1].clone()
    for idx in range(N_flat):
        e = flat_ids[idx].item()
        pos = write_pos[e].item()
        sorted_token_ids[pos] = idx
        write_pos[e] += 1

    # Fill expert_ids for each block
    for e in range(num_experts):
        start_block = cumsum[e].item() // block_size
        n_blocks = padded_per_expert[e].item() // block_size
        for b in range(n_blocks):
            expert_ids[start_block + b] = e

    num_tokens_post_padded = torch.tensor([total_padded], dtype=torch.int32, device=topk_ids.device)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


# ════════════════════════════════════════════════════════════════
# Fused MoE GEMM kernel (simplified from vLLM)
# ════════════════════════════════════════════════════════════════

@triton.jit
def fused_moe_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(tl.load(num_tokens_post_padded_ptr), BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    if group_size_m <= 0:
        return

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_token_id = pid_m * BLOCK_SIZE_M + offs_m
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id.to(tl.int64))
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a.to(compute_type), b.to(compute_type), acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ════════════════════════════════════════════════════════════════
# SiLU-gated activation kernel
# ════════════════════════════════════════════════════════════════

@triton.jit
def _silu_and_mul_kernel(
    input_ptr, output_ptr,
    d: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < d

    gate = tl.load(input_ptr + pid * 2 * d + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(input_ptr + pid * 2 * d + d + offs, mask=mask, other=0.0).to(tl.float32)

    gate_act = gate * tl.sigmoid(gate)  # silu
    result = gate_act * up

    tl.store(output_ptr + pid * d + offs, result.to(tl.bfloat16), mask=mask)


def silu_and_mul(x, d):
    """Apply SiLU-gated activation: silu(x[:, :d]) * x[:, d:]"""
    n = x.shape[0]
    out = torch.empty(n, d, dtype=x.dtype, device=x.device)
    BLOCK = triton.next_power_of_2(d)
    if BLOCK > 8192:
        BLOCK = 8192
    _silu_and_mul_kernel[(n,)](x, out, d, BLOCK=BLOCK)
    return out


# ════════════════════════════════════════════════════════════════
# Top-level fused_experts function
# ════════════════════════════════════════════════════════════════

def fused_experts_triton(
    hidden_states: torch.Tensor,  # [M, K]
    w1: torch.Tensor,             # [E, 2*N, K] — gate_up stacked
    w2: torch.Tensor,             # [E, K, N]
    topk_weights: torch.Tensor,   # [M, top_k]
    topk_ids: torch.Tensor,       # [M, top_k]
    activation: str = "silu",
) -> torch.Tensor:
    """Fused MoE expert computation with Triton.

    Matches vLLM's fused_experts_impl flow:
      GEMM1 → SiLU+Mul → GEMM2 → weighted reduce
    """
    M = hidden_states.shape[0]
    E, two_N, K = w1.shape
    N = two_N // 2
    top_k = topk_ids.shape[1]

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    else:
        compute_type = tl.float32

    # Config
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    # Step 1: Sort tokens by expert
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_simple(
        topk_ids, BLOCK_SIZE_M, E
    )
    num_valid = M * top_k

    # Step 2: GEMM1 — hidden @ w1.T → intermediate_cache1 [M*top_k, 2*N]
    intermediate1 = torch.empty(M * top_k, two_N, dtype=hidden_states.dtype, device=hidden_states.device)

    EM = num_tokens_post_padded.item()
    grid1 = (triton.cdiv(EM, BLOCK_SIZE_M) * triton.cdiv(two_N, BLOCK_SIZE_N),)

    fused_moe_gemm_kernel[grid1](
        hidden_states, w1, intermediate1,
        topk_weights.view(-1),
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        two_N, K, num_valid,
        hidden_states.stride(0), hidden_states.stride(1),
        w1.stride(0), w1.stride(1), w1.stride(2),
        intermediate1.stride(0), intermediate1.stride(1),
        MUL_ROUTED_WEIGHT=False,
        top_k=top_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        compute_type=compute_type,
    )

    # Step 3: SiLU-gated activation
    intermediate2 = silu_and_mul(intermediate1, N)

    # Step 4: GEMM2 — intermediate2 @ w2.T → out [M*top_k, K], with weight multiply
    out_cache = torch.zeros(M * top_k, K, dtype=hidden_states.dtype, device=hidden_states.device)

    grid2 = (triton.cdiv(EM, BLOCK_SIZE_M) * triton.cdiv(K, BLOCK_SIZE_N),)

    fused_moe_gemm_kernel[grid2](
        intermediate2, w2, out_cache,
        topk_weights.view(-1),
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        K, N, num_valid,
        intermediate2.stride(0), intermediate2.stride(1),
        w2.stride(0), w2.stride(1), w2.stride(2),
        out_cache.stride(0), out_cache.stride(1),
        MUL_ROUTED_WEIGHT=True,
        top_k=top_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        compute_type=compute_type,
    )

    # Step 5: Reduce — sum over top_k dimension
    out = out_cache.view(M, top_k, K).sum(dim=1)
    return out


# ════════════════════════════════════════════════════════════════
# Grouped top-k routing (LLaDA2-style)
# ════════════════════════════════════════════════════════════════

def grouped_topk(router_logits, num_experts=256, top_k=8, n_group=8, topk_group=4):
    """LLaDA2-style grouped routing: top groups → top experts within."""
    scores = torch.sigmoid(router_logits.float())
    group_size = num_experts // n_group
    grouped = scores.view(-1, n_group, group_size)
    group_max = grouped.max(dim=2).values
    _, top_groups = group_max.topk(topk_group, dim=1)

    # Build group mask efficiently
    N = scores.shape[0]
    device = scores.device
    group_mask = torch.zeros(N, num_experts, device=device, dtype=scores.dtype)
    for g in range(topk_group):
        starts = top_groups[:, g] * group_size  # [N]
        for offset in range(group_size):
            group_mask[torch.arange(N, device=device), starts + offset] = 1.0

    masked_scores = scores * group_mask
    topk_weights, topk_ids = masked_scores.topk(top_k, dim=1)
    topk_weights = topk_weights / (topk_weights.sum(dim=1, keepdim=True) + 1e-8)
    return topk_weights, topk_ids
