#!/usr/bin/env python3
"""Quick test of Triton MoE kernel."""
import sys; sys.path.insert(0, '.')
import torch, time

from utils import load_model_and_tokenizer, gpu_mem_mb
from triton_moe import fused_experts_triton, grouped_topk

model, tokenizer, config = load_model_and_tokenizer(device='cuda:0')
print(f'Model loaded, GPU: {gpu_mem_mb(0):.0f} MB')

moe_block = None
for name, mod in model.named_modules():
    if mod.__class__.__name__ == 'LLaDA2MoeSparseMoeBlock':
        moe_block = mod
        break

w13 = moe_block.experts.w13_weight
w2 = moe_block.experts.w2_weight
print(f'w13: {w13.shape}, w2: {w2.shape}')

# Test correctness
N, H = 32, 2048
hs = torch.randn(N, H, dtype=torch.bfloat16, device='cuda:0')
rl = torch.randn(N, 256, dtype=torch.bfloat16, device='cuda:0')
topk_w, topk_ids = grouped_topk(rl)
print(f'topk_ids: {topk_ids.shape}, topk_w: {topk_w.shape}')

out = fused_experts_triton(hs, w13, w2, topk_w, topk_ids)
print(f'Output: {out.shape}, norm: {out.float().norm():.4f}')

# Benchmark
for N in [32, 512, 2048]:
    hs = torch.randn(N, H, dtype=torch.bfloat16, device='cuda:0')
    rl = torch.randn(N, 256, dtype=torch.bfloat16, device='cuda:0')
    topk_w, topk_ids = grouped_topk(rl)
    for _ in range(3):
        fused_experts_triton(hs, w13, w2, topk_w, topk_ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        fused_experts_triton(hs, w13, w2, topk_w, topk_ids)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 10 * 1000
    print(f'N={N:4d}: triton {ms:.2f} ms')

print('DONE')
