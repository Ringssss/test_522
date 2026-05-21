#!/usr/bin/env python3
"""Test fused MoE kernel correctness and performance."""
import sys, time
sys.path.insert(0, '.')
import torch
from triton_moe import fused_experts, grouped_topk

E, N, K = 256, 512, 2048
w1 = torch.randn(E, 2*N, K, dtype=torch.bfloat16, device='cuda:0') * 0.01
w2 = torch.randn(E, K, N, dtype=torch.bfloat16, device='cuda:0') * 0.01

print("=== Correctness ===")
for M in [32, 512, 2048]:
    hs = torch.randn(M, K, dtype=torch.bfloat16, device='cuda:0')
    rl = torch.randn(M, E, dtype=torch.bfloat16, device='cuda:0')
    topk_w, topk_ids = grouped_topk(rl, num_experts=E)

    out_fused = fused_experts(hs, w1, w2, topk_w, topk_ids)

    # Reference: naive expert loop
    out_ref = torch.zeros_like(hs)
    for eid in topk_ids.unique().tolist():
        mask = (topk_ids == eid)
        has = mask.any(dim=1)
        if not has.any():
            continue
        idx = has.nonzero(as_tuple=True)[0]
        h = hs[idx].float()
        gate_up = h @ w1[eid].T.float()
        gate = torch.nn.functional.silu(gate_up[:, :N])
        up = gate_up[:, N:]
        eo = (gate * up) @ w2[eid].T.float()
        ew = (topk_w[idx] * mask[idx].float()).sum(dim=1, keepdim=True)
        out_ref[idx] += (eo * ew).to(out_ref.dtype)

    diff = (out_fused.float() - out_ref.float()).abs()
    rel = diff.max() / (out_ref.float().abs().max() + 1e-8)
    print(f"  M={M:4d}: abs_err={diff.max():.6f} rel_err={rel:.6f}")

print("\n=== Benchmark ===")
for M in [32, 128, 512, 1024, 2048]:
    hs = torch.randn(M, K, dtype=torch.bfloat16, device='cuda:0')
    rl = torch.randn(M, E, dtype=torch.bfloat16, device='cuda:0')
    topk_w, topk_ids = grouped_topk(rl, num_experts=E)

    for _ in range(5):
        fused_experts(hs, w1, w2, topk_w, topk_ids)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(20):
        fused_experts(hs, w1, w2, topk_w, topk_ids)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"  M={M:4d}: {ms:.2f} ms")

print("DONE")
