"""
Benchmark and correctness test for padding-free MoE vs vllm fused_experts.

Tests:
1. Correctness: compare padding_free_moe output vs fused_experts output
2. Kernel performance: per-call time at various batch sizes
3. End-to-end: full generate with both paths (if correctness passes)
"""

import torch
import time
import json
import statistics
import sys
import os

# Add dInfer to path
sys.path.insert(0, '/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python')

def test_correctness():
    """Compare padding_free_moe output against vllm fused_experts for numerical correctness."""
    print("=" * 60)
    print("TEST 1: Correctness verification")
    print("=" * 60)

    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from dinfer.decoding.padding_free_moe import padding_free_moe

    # Model config matching LLaDA2.0-mini
    E = 256
    hidden = 2048
    intermediate = 512
    top_k = 8

    # Create weights matching vllm layout
    w13 = torch.randn(E, 2 * intermediate, hidden, dtype=torch.bfloat16, device='cuda')
    w2 = torch.randn(E, hidden, intermediate, dtype=torch.bfloat16, device='cuda')

    results = {}

    for num_tokens in [32, 64, 128, 256]:
        hs = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device='cuda')

        # Random routing (simulate gate output)
        topk_weights = torch.randn(num_tokens, top_k, dtype=torch.bfloat16, device='cuda').softmax(dim=-1)
        topk_ids = torch.randint(0, E, (num_tokens, top_k), dtype=torch.int64, device='cuda')

        # Baseline: vllm fused_experts
        out_baseline = fused_experts(
            hs.clone(), w13, w2,
            topk_weights.clone().to(torch.float32),
            topk_ids.clone().to(torch.int32),
        )

        # New: padding_free_moe
        out_pf = padding_free_moe(
            hs.clone(), w13, w2,
            topk_weights.clone(),
            topk_ids.clone(),
            num_experts=E,
            top_k=top_k,
        )

        # Compare
        cos_sim = torch.nn.functional.cosine_similarity(
            out_baseline.flatten().float(),
            out_pf.flatten().float(),
            dim=0,
        ).item()
        max_diff = (out_baseline.float() - out_pf.float()).abs().max().item()
        mean_diff = (out_baseline.float() - out_pf.float()).abs().mean().item()

        status = "PASS" if cos_sim > 0.99 else "FAIL"
        print(f"  tokens={num_tokens:4d}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} [{status}]")

        results[f"tokens_{num_tokens}"] = {
            "cos_sim": cos_sim,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "status": status,
        }

    return results


def test_kernel_performance():
    """Benchmark padding_free_moe vs fused_experts per-call time."""
    print("\n" + "=" * 60)
    print("TEST 2: Kernel performance comparison")
    print("=" * 60)

    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from dinfer.decoding.padding_free_moe import padding_free_moe

    E = 256
    hidden = 2048
    intermediate = 512
    top_k = 8

    w13 = torch.randn(E, 2 * intermediate, hidden, dtype=torch.bfloat16, device='cuda')
    w2 = torch.randn(E, hidden, intermediate, dtype=torch.bfloat16, device='cuda')

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    results = {}

    for batch_size in [1, 4, 8, 16, 32]:
        num_tokens = batch_size * 32  # block_length = 32
        hs = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device='cuda')
        topk_weights = torch.randn(num_tokens, top_k, dtype=torch.bfloat16, device='cuda').softmax(dim=-1)
        topk_ids = torch.randint(0, E, (num_tokens, top_k), dtype=torch.int64, device='cuda')

        # Benchmark fused_experts (baseline)
        topk_w_f32 = topk_weights.float()
        topk_i_i32 = topk_ids.int()
        for _ in range(5):
            fused_experts(hs, w13, w2, topk_w_f32, topk_i_i32)
        torch.cuda.synchronize()

        times_baseline = []
        for _ in range(50):
            start_ev.record()
            fused_experts(hs, w13, w2, topk_w_f32, topk_i_i32)
            end_ev.record()
            torch.cuda.synchronize()
            times_baseline.append(start_ev.elapsed_time(end_ev))

        # Benchmark padding_free_moe
        for _ in range(5):
            padding_free_moe(hs, w13, w2, topk_weights, topk_ids, E, top_k)
        torch.cuda.synchronize()

        times_pf = []
        for _ in range(50):
            start_ev.record()
            padding_free_moe(hs, w13, w2, topk_weights, topk_ids, E, top_k)
            end_ev.record()
            torch.cuda.synchronize()
            times_pf.append(start_ev.elapsed_time(end_ev))

        t_base = statistics.mean(times_baseline)
        t_pf = statistics.mean(times_pf)
        speedup = t_base / t_pf if t_pf > 0 else 0

        print(f"  batch={batch_size:2d} (tokens={num_tokens:4d}): "
              f"baseline={t_base:.3f}ms, padding_free={t_pf:.3f}ms, "
              f"speedup={speedup:.2f}x")

        results[f"batch_{batch_size}"] = {
            "num_tokens": num_tokens,
            "baseline_ms": round(t_base, 3),
            "padding_free_ms": round(t_pf, 3),
            "speedup": round(speedup, 2),
        }

    return results


if __name__ == '__main__':
    print("Padding-Free MoE Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    all_results = {}

    # Test 1: Correctness
    try:
        all_results["correctness"] = test_correctness()
    except Exception as e:
        print(f"\nCorrectness test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        all_results["correctness"] = {"error": str(e)}

    # Test 2: Performance (only if correctness passes)
    correctness_ok = all(
        v.get("status") == "PASS"
        for v in all_results.get("correctness", {}).values()
        if isinstance(v, dict) and "status" in v
    )

    if correctness_ok:
        try:
            all_results["performance"] = test_kernel_performance()
        except Exception as e:
            print(f"\nPerformance test FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            all_results["performance"] = {"error": str(e)}
    else:
        print("\nSkipping performance test due to correctness failure.")

    # Save results
    results_path = '/home/wuhang/wuhang/dllm_wh/codex_coding/results/padding_free_moe_benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
