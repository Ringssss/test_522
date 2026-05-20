"""
Padding-Free MoE implementation based on X-MoE PFT (Padding-Free Token) approach.

Replaces vllm's fused_experts which pads each expert's token count to BLOCK_SIZE_M,
causing 50-90% compute waste. Instead uses:
  1. Compact dispatch (sort + histogram, no padding)
  2. Triton gather/scatter kernels for token reordering
  3. Grouped GEMM with persistent kernel for padding-free expert computation

Reference: X-MoE (SC 2025) - deepspeed/moe/v2opt/
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional
import functools


# ============================================================================
# Section 1: Dispatch metadata (from X-MoE v2opt/gating.py)
# ============================================================================

def indices_and_bins(
    num_experts: int,
    top_experts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort token-expert assignments by expert id and compute bin boundaries.

    Args:
        num_experts: total number of experts (256)
        top_experts: [num_tokens * top_k] flattened expert ids

    Returns:
        indices: [num_tokens * top_k] permutation indices (sorted order)
        bin_ids: [num_tokens * top_k] sorted expert ids
        bins: [num_experts] cumulative token count per expert
        tokens_per_expert: [num_experts] token count per expert
    """
    top_experts = top_experts.int()
    bin_ids, indices = torch.sort(top_experts)

    tokens_per_expert = torch.histc(
        top_experts.float(), num_experts, 0, num_experts - 1
    ).int()

    bins = torch.cumsum(tokens_per_expert, 0).int()

    return indices, bin_ids, bins, tokens_per_expert


# ============================================================================
# Section 2: Triton gather/scatter kernels (from X-MoE v2opt/kernels.py)
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 64}, num_warps=2),
        triton.Config({'BLOCK_X': 128}, num_warps=2),
        triton.Config({'BLOCK_X': 256}, num_warps=2),
        triton.Config({'BLOCK_X': 128}, num_warps=4),
        triton.Config({'BLOCK_X': 256}, num_warps=4),
    ],
    key=['NUM_COLUMNS'],
)
@triton.jit
def _padded_copy(
    a, b, indices, bin_ids, weights, bins, padded_bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    SCALE: tl.constexpr,
):
    # Index into source array
    index_a = tl.load(indices + tl.program_id(0))

    # Which expert bin this token belongs to
    bin_idx = tl.load(bin_ids + tl.program_id(0))

    # Offset within our bin
    offset_in_bin = tl.program_id(0)
    if bin_idx > 0:
        offset_in_bin -= tl.load(bins + bin_idx - 1)

    # Starting index in output array (with potential padding)
    index_b = offset_in_bin
    if bin_idx > 0:
        index_b += tl.load(padded_bins + bin_idx - 1)

    # Set up pointers: A_TO_B means gather (input→sorted), !A_TO_B means scatter
    offset = index_a // TOP_K if A_TO_B else index_a
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    scale = tl.load(weights + index_a) if SCALE else 1

    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a

    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        x = tl.load(iptr + offsets, mask=mask)
        x = x.to(tl.float32) * scale.to(tl.float32)
        tl.store(optr + offsets, x.to(optr.dtype.element_ty), mask=mask)
        offsets += BLOCK_X


def pft_gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Gather tokens into expert-sorted order (padding-free).

    Args:
        x: [num_tokens, hidden_size] input hidden states
        indices, bin_ids, bins: from indices_and_bins()
        top_k: number of experts per token

    Returns:
        sorted_tokens: [num_tokens * top_k, hidden_size] tightly packed
    """
    output_rows = x.shape[0] * top_k
    out = torch.empty((output_rows, x.shape[1]), dtype=x.dtype, device=x.device)
    _padded_copy[(indices.shape[0],)](
        x, out, indices, bin_ids, None, bins, bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=True,
        TOP_K=top_k,
        SCALE=False,
    )
    return out


def pft_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: torch.Tensor,
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Scatter expert outputs back to original token order with weighted sum.

    Args:
        x: [num_tokens * top_k, hidden_size] expert outputs (sorted by expert)
        indices, bin_ids, bins: from indices_and_bins()
        weights: [num_tokens * top_k] routing weights (flattened)
        top_k: number of experts per token

    Returns:
        output: [num_tokens, hidden_size] weighted sum across experts
    """
    tokens = indices.shape[0] // top_k
    out = torch.empty((tokens, top_k, x.shape[1]), dtype=x.dtype, device=x.device)
    _padded_copy[(indices.shape[0],)](
        out, x, indices, bin_ids, weights, bins, bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=False,
        TOP_K=top_k,
        SCALE=True,
    )
    return out.sum(dim=1) if top_k > 1 else out.view(tokens, x.shape[1])


# ============================================================================
# Section 3: Stacked grouped GEMM kernel (no Python loop, no pointer arrays)
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'NUM_SM': 132}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'NUM_SM': 132}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'NUM_SM': 132}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'NUM_SM': 132}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'NUM_SM': 132}),
    ],
    key=['N', 'K'],
)
@triton.jit
def stacked_grouped_gemm_kernel(
    # Input tokens: [total_tokens, K], sorted by expert
    a_ptr,
    # Expert weights: [E, K, N], stacked contiguously
    b_ptr,
    # Output: [total_tokens, N]
    c_ptr,
    # Per-expert token offsets: [num_experts] (exclusive prefix sum)
    expert_offsets_ptr,
    # Per-expert token counts: [num_experts]
    expert_counts_ptr,
    # Dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    num_experts: tl.constexpr,
    # Strides
    stride_ak,   # = K (row stride of A)
    stride_be,   # = K * N (expert stride of B)
    stride_bk,   # = N (row stride of B [K, N] row-major)
    stride_cn,   # = N (row stride of C)
    # Meta
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Persistent grouped GEMM on stacked weights.
    Iterates over all experts (constexpr loop for unrolling).
    Empty experts are skipped naturally (num_tiles=0 → while body never runs).
    """
    tile_idx = tl.program_id(0)
    last_problem_end = 0

    for g in range(num_experts):
        gm = tl.load(expert_counts_ptr + g)

        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            token_offset = tl.load(expert_offsets_ptr + g)

            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            a_tile_ptr = a_ptr + (token_offset + offs_am[:, None]) * stride_ak + offs_k[None, :]
            b_tile_ptr = b_ptr + g * stride_be + offs_k[:, None] * stride_bk + offs_bn[None, :]

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                a = tl.load(a_tile_ptr,
                            mask=(offs_am[:, None] < gm) &
                                 (offs_k[None, :] + kk * BLOCK_SIZE_K < K),
                            other=0.0)
                b = tl.load(b_tile_ptr,
                            mask=(offs_k[:, None] + kk * BLOCK_SIZE_K < K) &
                                 (offs_bn[None, :] < N),
                            other=0.0)
                accumulator += tl.dot(a, b)
                a_tile_ptr += BLOCK_SIZE_K
                b_tile_ptr += BLOCK_SIZE_K * stride_bk

            c_tile = accumulator.to(tl.bfloat16)
            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_tile_ptr = c_ptr + (token_offset + offs_cm[:, None]) * stride_cn + offs_cn[None, :]
            tl.store(c_tile_ptr, c_tile,
                     mask=(offs_cm[:, None] < gm) & (offs_cn[None, :] < N))

            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles


def grouped_gemm_bf16(
    x: torch.Tensor,
    weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Padding-free grouped GEMM for MoE expert computation.
    Uses stacked weights directly — no Python loop over experts.

    Args:
        x: [total_pairs, K] sorted tokens (from pft_gather)
        weights: [E, K, N] expert weights (already transposed to [K, N] per expert)
        tokens_per_expert: [num_experts] token count per expert
        num_experts: total expert count

    Returns:
        output: [total_pairs, N] expert outputs
    """
    total_tokens = x.shape[0]
    K = x.shape[1]
    _, K_w, N = weights.shape

    # Compute exclusive prefix sum for token offsets (GPU, no Python loop)
    expert_offsets = torch.zeros(num_experts, dtype=torch.int32, device=x.device)
    expert_offsets[1:] = torch.cumsum(tokens_per_expert[:-1], 0).int()
    expert_counts = tokens_per_expert.int()

    output = torch.empty((total_tokens, N), dtype=x.dtype, device=x.device)
    if total_tokens == 0:
        return output

    grid = lambda META: (META['NUM_SM'],)
    stacked_grouped_gemm_kernel[grid](
        x, weights, output,
        expert_offsets, expert_counts,
        N=N, K=K, num_experts=num_experts,
        stride_ak=K,
        stride_be=K_w * N,
        stride_bk=N,
        stride_cn=N,
    )

    return output


# ============================================================================
# Section 4: Weight transpose cache
# ============================================================================

_weight_transpose_cache = {}

def get_transposed_weight(w: torch.Tensor, key: str) -> torch.Tensor:
    """
    Transpose weight from [E, N, K] to [E, K, N] with caching.
    Only done once per weight tensor.
    """
    cache_id = (key, w.data_ptr())
    if cache_id not in _weight_transpose_cache:
        _weight_transpose_cache[cache_id] = w.transpose(1, 2).contiguous()
    return _weight_transpose_cache[cache_id]


# ============================================================================
# Section 5: Main entry point
# ============================================================================

@torch.no_grad()
def padding_free_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int = 256,
    top_k: int = 8,
) -> torch.Tensor:
    """
    Padding-free MoE forward pass.

    Args:
        hidden_states: [num_tokens, hidden_size] input
        w13_weight: [E, 2*intermediate, hidden_size] gate+up projection weights
        w2_weight: [E, hidden_size, intermediate] down projection weights
        topk_weights: [num_tokens, top_k] routing weights
        topk_ids: [num_tokens, top_k] selected expert ids
        num_experts: total expert count
        top_k: experts per token

    Returns:
        output: [num_tokens, hidden_size] MoE output
    """
    num_tokens, hidden_size = hidden_states.shape

    # Step 1: Flatten expert ids and compute dispatch metadata
    top_experts = topk_ids.flatten()  # [num_tokens * top_k]
    indices, bin_ids, bins, tokens_per_expert = indices_and_bins(num_experts, top_experts)

    # Step 2: Gather tokens into expert-sorted order
    sorted_tokens = pft_gather(hidden_states, indices, bin_ids, bins, top_k)
    # sorted_tokens: [num_tokens * top_k, hidden_size]

    # Step 3: Grouped GEMM for w1 (gate + up projection)
    # w13_weight: [E, 2*intermediate, hidden] -> need [E, hidden, 2*intermediate]
    w13_T = get_transposed_weight(w13_weight, 'w13')
    intermediate = grouped_gemm_bf16(sorted_tokens, w13_T, tokens_per_expert, num_experts)
    # intermediate: [num_tokens * top_k, 2*intermediate]

    # Step 4: SiLU activation with gate multiplication
    N = intermediate.shape[1]
    act_out = torch.empty(
        (intermediate.shape[0], N // 2), dtype=intermediate.dtype, device=intermediate.device
    )
    torch.ops._C.silu_and_mul(act_out, intermediate)
    # act_out: [num_tokens * top_k, intermediate]

    # Step 5: Grouped GEMM for w2 (down projection)
    # w2_weight: [E, hidden, intermediate] -> need [E, intermediate, hidden]
    w2_T = get_transposed_weight(w2_weight, 'w2')
    expert_out = grouped_gemm_bf16(act_out, w2_T, tokens_per_expert, num_experts)
    # expert_out: [num_tokens * top_k, hidden_size]

    # Step 6: Scatter back with routing weight multiplication
    flat_weights = topk_weights.flatten()  # [num_tokens * top_k]
    output = pft_scatter(expert_out, indices, bin_ids, flat_weights, bins, top_k)
    # output: [num_tokens, hidden_size]

    return output
