#!/usr/bin/env python3
"""
v0.1.15.2 Step 1 — fused_experts Kernel Micro-Benchmark

Gate-keeper test: does weight=0 save real kernel time?

Source code analysis already confirmed situation γ (weight multiplication
happens AFTER all GEMMs in the Triton kernel, line 468-472). This benchmark
provides quantitative confirmation and measures physical reduction speedup.

Benchmarks:
  A: baseline     — [N, 8] normal weights, normal routing
  B: weight-zero  — [N, 8] 50% weights=0, same topk_ids
  C: physical-4   — [N, 4] only top-4 experts (physically removed)
  D: physical-1   — [N, 1] only top-1 expert

Also measures:
  Q1: moe_align_block_size grid size under each condition
  Q2: active expert count with real top-p routing
"""

from __future__ import annotations
import os, sys, json, time
from pathlib import Path

import torch

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
DEVICE = "cuda:0"


def benchmark_kernel(hidden_states, w1, w2, topk_weights, topk_ids,
                     warmup=20, repeats=100, label=""):
    """Time fused_experts with torch.cuda.Event."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    # Warmup
    for _ in range(warmup):
        fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, inplace=False)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, inplace=False)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end) / repeats
    return elapsed_ms


def measure_grid_size(topk_ids, num_experts, block_size_m):
    """Measure moe_align_block_size output for given topk_ids."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, num_experts)
    ntp = num_tokens_post_padded.item()
    num_blocks = (ntp + block_size_m - 1) // block_size_m

    # Count active experts (experts with at least 1 token)
    flat_ids = topk_ids.view(-1)
    active_experts = flat_ids.unique().numel()

    return {
        "num_tokens_post_padded": ntp,
        "num_blocks": num_blocks,
        "active_experts": active_experts,
        "total_pairs": topk_ids.numel(),
    }


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)
    sys.path.insert(0, str(REPO_ROOT / "lib_cite" / "dInfer" / "python"))

    import socket
    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoTokenizer, AutoConfig

    print("=" * 80)
    print("v0.1.15.2 Step 1 — fused_experts Kernel Micro-Benchmark")
    print("=" * 80)

    # Load model to get real weights
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

    # Extract MoE weights from layer 10 (representative middle layer)
    moe_layer = model.model.layers[10].mlp
    w1 = moe_layer.experts.w13_weight  # [E, N, K]
    w2 = moe_layer.experts.w2_weight   # [E, K, N]
    gate = moe_layer.gate
    E = w1.shape[0]
    print(f"MoE weights: w1={list(w1.shape)}, w2={list(w2.shape)}, E={E}")
    print(f"  w1 dtype={w1.dtype}, w2 dtype={w2.dtype}")

    TOP_K = 8
    SHARED_RATE = 0.419
    ROUTING_RATE = 0.581
    BLOCK_SIZE_M = 16  # from autotuning config

    # Token counts to test
    TOKEN_COUNTS = [32, 256, 1024]

    # Generate realistic routing data
    print(f"\n{'='*80}")
    print(f"Generating realistic routing data from model forward...")
    print(f"{'='*80}")

    routing_data = {}
    with torch.inference_mode():
        for N in TOKEN_COUNTS:
            hs = torch.randn(N, config.hidden_size, dtype=torch.bfloat16, device=device)
            topk_idx, topk_weight, _ = gate(hs)
            routing_data[N] = {
                "hidden_states": hs,
                "topk_idx": topk_idx,       # [N, 8]
                "topk_weight": topk_weight,  # [N, 8]
            }
            print(f"  N={N}: topk_idx={list(topk_idx.shape)}, "
                  f"active_experts={topk_idx.view(-1).unique().numel()}/{E}")

    # ================================================================
    # Q1: moe_align_block_size grid comparison
    # ================================================================
    print(f"\n{'='*80}")
    print(f"Q1: moe_align_block_size grid size comparison")
    print(f"{'='*80}")

    for N in TOKEN_COUNTS:
        rd = routing_data[N]
        topk_idx = rd["topk_idx"]
        topk_weight = rd["topk_weight"]

        # Condition A: normal [N, 8]
        grid_a = measure_grid_size(topk_idx, E, BLOCK_SIZE_M)

        # Condition B: same ids (weight=0 doesn't change ids)
        grid_b = grid_a  # topk_ids identical → grid identical

        # Condition C: physical top-4
        sorted_w, sort_order = topk_weight.sort(dim=1, descending=True)
        topk_idx_4 = topk_idx.gather(1, sort_order[:, :4])
        grid_c = measure_grid_size(topk_idx_4, E, BLOCK_SIZE_M)

        # Condition D: physical top-1
        topk_idx_1 = topk_idx.gather(1, sort_order[:, :1])
        grid_d = measure_grid_size(topk_idx_1, E, BLOCK_SIZE_M)

        # Q2: top-p=0.75 applied — how many experts become fully inactive?
        total_routing = topk_weight.sum(dim=1, keepdim=True)
        needed_frac = (0.75 - SHARED_RATE) / ROUTING_RATE
        threshold = needed_frac * total_routing
        cumsum = sorted_w.cumsum(dim=1)
        enough = (cumsum >= threshold)
        enough[:, -1] = True
        cutoff = enough.float().argmax(dim=1) + 1  # [N]

        # Build mask of kept experts
        rank_pos = torch.arange(TOP_K, device=device).unsqueeze(0)
        keep_sorted = rank_pos < cutoff.unsqueeze(1)  # [N, 8] in sorted order
        # Map back to original order
        keep_mask = torch.zeros_like(topk_weight, dtype=torch.bool)
        keep_mask.scatter_(1, sort_order, keep_sorted)

        # Active experts under top-p (experts with at least 1 kept token)
        active_tp = set()
        for i in range(N):
            for j in range(TOP_K):
                if keep_mask[i, j]:
                    active_tp.add(topk_idx[i, j].item())

        # Physical top-p: build variable-k tensors padded to max_kept_k
        max_kept = cutoff.max().item()
        avg_kept = cutoff.float().mean().item()
        # Build [N, max_kept] ids and weights
        topk_idx_tp = torch.zeros(N, max_kept, dtype=topk_idx.dtype, device=device)
        topk_weight_tp = torch.zeros(N, max_kept, dtype=topk_weight.dtype, device=device)
        for i in range(N):
            k = int(cutoff[i].item())
            # Top-k indices in sorted weight order
            kept_idx = topk_idx[i].gather(0, sort_order[i, :k])
            kept_w = topk_weight[i].gather(0, sort_order[i, :k])
            # Renormalize
            orig_sum = topk_weight[i].sum()
            kept_sum = kept_w.sum()
            kept_w = kept_w * (orig_sum / (kept_sum + 1e-8))
            topk_idx_tp[i, :k] = kept_idx
            topk_weight_tp[i, :k] = kept_w

        grid_tp = measure_grid_size(topk_idx_tp, E, BLOCK_SIZE_M)

        print(f"\n  N={N}:")
        print(f"    A (baseline top-8): pairs={grid_a['total_pairs']:>6d}  "
              f"blocks={grid_a['num_blocks']:>5d}  active={grid_a['active_experts']:>3d}/256")
        print(f"    B (weight=0 top-8): pairs={grid_a['total_pairs']:>6d}  "
              f"blocks={grid_a['num_blocks']:>5d}  active={grid_a['active_experts']:>3d}/256  "
              f"(same as A — ids unchanged)")
        print(f"    C (physical top-4): pairs={grid_c['total_pairs']:>6d}  "
              f"blocks={grid_c['num_blocks']:>5d}  active={grid_c['active_experts']:>3d}/256  "
              f"({grid_c['num_blocks']/grid_a['num_blocks']:.2f}x)")
        print(f"    D (physical top-1): pairs={grid_d['total_pairs']:>6d}  "
              f"blocks={grid_d['num_blocks']:>5d}  active={grid_d['active_experts']:>3d}/256  "
              f"({grid_d['num_blocks']/grid_a['num_blocks']:.2f}x)")
        print(f"    top-p=0.75 stats:   avg_kept={avg_kept:.2f}  max_kept={max_kept}  "
              f"active_experts={len(active_tp)}/256")
        print(f"    E (physical top-p): pairs={grid_tp['total_pairs']:>6d}  "
              f"blocks={grid_tp['num_blocks']:>5d}  active={grid_tp['active_experts']:>3d}/256  "
              f"({grid_tp['num_blocks']/grid_a['num_blocks']:.2f}x)")

    # ================================================================
    # Q3: fused_experts kernel timing
    # ================================================================
    print(f"\n{'='*80}")
    print(f"Q3: fused_experts kernel timing (warmup=20, repeats=100)")
    print(f"{'='*80}")

    all_results = {}
    with torch.inference_mode():
        for N in TOKEN_COUNTS:
            rd = routing_data[N]
            hs = rd["hidden_states"]
            topk_idx = rd["topk_idx"]
            topk_weight = rd["topk_weight"]

            results = {}

            # A: baseline [N, 8]
            t_a = benchmark_kernel(hs, w1, w2, topk_weight, topk_idx, label="A")
            results["A_baseline"] = t_a

            # B: weight-zero [N, 8] — zero out bottom 4 weights
            sorted_w, sort_order = topk_weight.sort(dim=1, descending=True)
            mask_zero = torch.ones_like(topk_weight)
            mask_zero.scatter_(1, sort_order[:, 4:], 0.0)
            topk_weight_b = topk_weight * mask_zero
            # Renormalize
            kept_sum = topk_weight_b.sum(dim=1, keepdim=True)
            orig_sum = topk_weight.sum(dim=1, keepdim=True)
            topk_weight_b = topk_weight_b * (orig_sum / (kept_sum + 1e-8))
            t_b = benchmark_kernel(hs, w1, w2, topk_weight_b, topk_idx, label="B")
            results["B_weight_zero"] = t_b

            # C: physical top-4 [N, 4]
            topk_idx_4 = topk_idx.gather(1, sort_order[:, :4])
            topk_weight_4 = topk_weight.gather(1, sort_order[:, :4])
            w4_sum = topk_weight_4.sum(dim=1, keepdim=True)
            topk_weight_4 = topk_weight_4 * (orig_sum / (w4_sum + 1e-8))
            t_c = benchmark_kernel(hs, w1, w2, topk_weight_4, topk_idx_4, label="C")
            results["C_physical_4"] = t_c

            # D: physical top-1 [N, 1]
            topk_idx_1 = topk_idx.gather(1, sort_order[:, :1])
            topk_weight_1 = topk_weight.gather(1, sort_order[:, :1])
            w1_sum = topk_weight_1.sum(dim=1, keepdim=True)
            topk_weight_1 = topk_weight_1 * (orig_sum / (w1_sum + 1e-8))
            t_d = benchmark_kernel(hs, w1, w2, topk_weight_1, topk_idx_1, label="D")
            results["D_physical_1"] = t_d

            # E: physical top-p=0.75 [N, max_kept]
            total_routing = topk_weight.sum(dim=1, keepdim=True)
            needed_frac = (0.75 - SHARED_RATE) / ROUTING_RATE
            threshold_vals = needed_frac * total_routing
            cumsum = sorted_w.cumsum(dim=1)
            enough = (cumsum >= threshold_vals)
            enough[:, -1] = True
            cutoff = enough.float().argmax(dim=1) + 1
            max_kept = int(cutoff.max().item())
            avg_kept = cutoff.float().mean().item()

            topk_idx_e = torch.zeros(N, max_kept, dtype=topk_idx.dtype, device=device)
            topk_weight_e = torch.zeros(N, max_kept, dtype=topk_weight.dtype, device=device)
            for i in range(N):
                k = int(cutoff[i].item())
                kept_idx = topk_idx[i].gather(0, sort_order[i, :k])
                kept_w = topk_weight[i].gather(0, sort_order[i, :k])
                kept_w = kept_w * (topk_weight[i].sum() / (kept_w.sum() + 1e-8))
                topk_idx_e[i, :k] = kept_idx
                topk_weight_e[i, :k] = kept_w

            t_e = benchmark_kernel(hs, w1, w2, topk_weight_e, topk_idx_e, label="E")
            results["E_physical_topp"] = t_e

            all_results[N] = results

            print(f"\n  N={N} (batch={N//32}):")
            print(f"    A baseline (top-8): {t_a:.4f} ms")
            print(f"    B weight=0 (top-8): {t_b:.4f} ms  ({t_b/t_a:.3f}x vs A)")
            print(f"    C physical (top-4): {t_c:.4f} ms  ({t_c/t_a:.3f}x vs A)")
            print(f"    D physical (top-1): {t_d:.4f} ms  ({t_d/t_a:.3f}x vs A)")
            print(f"    E physical (top-p): {t_e:.4f} ms  ({t_e/t_a:.3f}x vs A)  "
                  f"avg_kept={avg_kept:.1f} max_kept={max_kept}")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")

    print(f"\n  {'N':>5s} | {'A(top8)':>10s} | {'B(wt=0)':>10s} | "
          f"{'C(top4)':>10s} | {'D(top1)':>10s} | {'E(topp)':>10s} | {'B/A':>6s} | {'C/A':>6s} | {'D/A':>6s} | {'E/A':>6s}")
    print(f"  {'-'*100}")
    for N in TOKEN_COUNTS:
        r = all_results[N]
        a = r["A_baseline"]
        b = r["B_weight_zero"]
        c = r["C_physical_4"]
        d = r["D_physical_1"]
        e = r["E_physical_topp"]
        print(f"  {N:>5d} | {a:>9.4f}ms | {b:>9.4f}ms | "
              f"{c:>9.4f}ms | {d:>9.4f}ms | {e:>9.4f}ms | "
              f"{b/a:>5.3f} | {c/a:>5.3f} | {d/a:>5.3f} | {e/a:>5.3f}")

    # Situation determination
    print(f"\n  Situation determination:")
    for N in TOKEN_COUNTS:
        r = all_results[N]
        ba = r["B_weight_zero"] / r["A_baseline"]
        ca = r["C_physical_4"] / r["A_baseline"]
        if ba > 0.95:
            if ca < 0.70:
                sit = "γ (weight=0 no effect, physical reduction works)"
            else:
                sit = "γ-worst (neither weight=0 nor physical reduction helps)"
        elif ba > 0.80:
            sit = "β (partial indirect effect)"
        else:
            sit = "α (weight=0 achieves physical skip)"
        print(f"    N={N}: B/A={ba:.3f}, C/A={ca:.3f} → {sit}")

    # ================================================================
    # Save results
    # ================================================================
    results_path = REPO_ROOT / "codex_coding" / "results" / "fused_experts_weight_zero_benchmark.json"
    save_data = {"token_counts": TOKEN_COUNTS}
    for N in TOKEN_COUNTS:
        save_data[f"N={N}"] = all_results[N]
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
