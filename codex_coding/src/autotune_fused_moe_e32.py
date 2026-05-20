#!/usr/bin/env python3
"""
Auto-tune FusedMoE tiling config for E=32, N=512 on H100.

Steps:
  (a) Compare existing E=32 vs E=62/128 configs at key M values
  (b) Grid search over all config combinations
  (c) Output optimal config JSON

Usage (single GPU):
  python autotune_fused_moe_e32.py --mode compare   # step (a)
  python autotune_fused_moe_e32.py --mode search     # steps (b)+(c)
  python autotune_fused_moe_e32.py --mode all         # all steps

  # Quick test with specific M values:
  python autotune_fused_moe_e32.py --m-values 256,2048 --mode all
"""

import argparse
import itertools
import json
import os
import sys
import time

import torch
import triton
import triton.language as tl

# ── Problem constants ──
E_LOCAL = 32
N_INTER = 512
K_HIDDEN = 2048
TOP_K = 4
DTYPE = torch.bfloat16

VLLM_CONFIGS_DIR = (
    "/home/wuhang/miniconda3/envs/dllm/lib/python3.10/site-packages"
    "/vllm/model_executor/layers/fused_moe/configs"
)

SEARCH_SPACE = {
    "BLOCK_SIZE_M": [16, 32, 64, 128],
    "BLOCK_SIZE_N": [32, 64, 128, 256],
    "BLOCK_SIZE_K": [64, 128],
    "GROUP_SIZE_M": [1, 16],
    "num_warps": [4, 8],
    "num_stages": [2, 3, 4],
}

# M values covering our workload range
# batch=512 GS path: M≈2048; A path: M≈8192; smaller/larger batches scale proportionally
DEFAULT_M_VALUES = [64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]


def load_config_json(e, n=512):
    fname = f"E={e},N={n},device_name=NVIDIA_H100_80GB_HBM3.json"
    fpath = os.path.join(VLLM_CONFIGS_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    return None


def bench_kernel(M, config, gemm="gemm1", warmup_ms=30, rep_ms=150):
    """Benchmark one config at a given M. Returns time in ms (median)."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe_kernel
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    device = "cuda"
    BSM = config["BLOCK_SIZE_M"]
    BSN = config["BLOCK_SIZE_N"]
    BSK = config["BLOCK_SIZE_K"]
    GSM = config["GROUP_SIZE_M"]
    warps = config["num_warps"]
    stages = config["num_stages"]

    if gemm == "gemm1":
        # A=[M, K=2048], B=w1=[E, 2N=1024, K=2048]
        K_in = K_HIDDEN      # 2048
        N_out = 2 * N_INTER   # 1024
    else:
        # A=[M*topk, K=512], B=w2=[E, N=2048, K=512]
        K_in = N_INTER        # 512
        N_out = K_HIDDEN      # 2048
        M = M * TOP_K

    if K_in % BSK != 0:
        return float("inf")

    A = torch.randn(M, K_in, dtype=DTYPE, device=device)
    B = torch.randn(E_LOCAL, N_out, K_in, dtype=DTYPE, device=device)
    C = torch.empty(M, TOP_K, N_out, dtype=DTYPE, device=device)
    topk_weights = torch.ones(M, TOP_K, dtype=torch.float32, device=device)

    topk_ids = torch.randint(0, E_LOCAL, (M, TOP_K), device=device, dtype=torch.int32)
    expert_map = torch.arange(E_LOCAL, device=device, dtype=torch.int32)

    try:
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, BSM, E_LOCAL, expert_map
        )
    except Exception:
        return float("inf")

    EM = sorted_token_ids.size(0)
    num_tokens = M * TOP_K

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N_out, META["BLOCK_SIZE_N"]),
    )

    def _run():
        fused_moe_kernel[grid](
            A, B, C,
            None, None, None,       # bias, A_scale, B_scale
            topk_weights,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            N_out, K_in, EM, num_tokens,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(2), B.stride(1),
            C.stride(1), C.stride(2),
            0, 0,                   # A_scale strides
            0, 0, 0,               # B_scale strides
            0, 0,                   # B_bias strides
            0, 0,                   # block_shape
            MUL_ROUTED_WEIGHT=False,
            top_k=TOP_K,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            per_channel_quant=False,
            HAS_BIAS=False,
            BLOCK_SIZE_K=BSK,
            BLOCK_SIZE_M=BSM,
            BLOCK_SIZE_N=BSN,
            GROUP_SIZE_M=GSM,
            num_warps=warps,
            num_stages=stages,
        )

    try:
        ms = triton.testing.do_bench(_run, warmup=warmup_ms, rep=rep_ms)
        return ms
    except Exception:
        return float("inf")


def gen_all_configs():
    keys = list(SEARCH_SPACE.keys())
    return [dict(zip(keys, v)) for v in itertools.product(*SEARCH_SPACE.values())]


def step_compare(M_values):
    """Step (a): Compare E=32 config vs E=62/128 at each M."""
    print("\n" + "=" * 72)
    print("Step (a): Comparing existing configs across E values")
    print("=" * 72)

    cfgs_by_e = {}
    for e in [32, 62, 128]:
        c = load_config_json(e)
        if c:
            cfgs_by_e[e] = c
            print(f"  Loaded E={e} config ({len(c)} M entries)")

    results = {}
    for M in M_values:
        print(f"\n  M={M}:")
        best_t, best_e = float("inf"), None
        row = {}
        for e, cfgs in cfgs_by_e.items():
            closest = min(cfgs.keys(), key=lambda x: abs(x - M))
            cfg = cfgs[closest]
            t = bench_kernel(M, cfg, "gemm1")
            tag = " ← CURRENT" if e == 32 else ""
            print(f"    E={e:>3} (cfg_M={closest:>4}): {t:7.4f} ms  "
                  f"BSM={cfg['BLOCK_SIZE_M']:>3} BSN={cfg['BLOCK_SIZE_N']:>3} "
                  f"BSK={cfg['BLOCK_SIZE_K']:>3} GSM={cfg['GROUP_SIZE_M']:>2} "
                  f"w={cfg['num_warps']} s={cfg['num_stages']}{tag}")
            row[e] = t
            if t < best_t:
                best_t, best_e = t, e

        e32_t = row.get(32, float("inf"))
        if best_e != 32 and best_t < float("inf"):
            pct = (e32_t - best_t) / e32_t * 100 if e32_t < float("inf") else 0
            print(f"    → Best: E={best_e} ({best_t:.4f} ms, {pct:+.1f}% vs E=32)")
        results[M] = row

    return results


def step_search(M_values):
    """Step (b): Grid search for optimal config at each M."""
    print("\n" + "=" * 72)
    print("Step (b): Grid search — all configs per M")
    print("=" * 72)

    all_cfgs = gen_all_configs()
    n_cfgs = len(all_cfgs)
    print(f"  Total configs: {n_cfgs}")

    e32_cfgs = load_config_json(32)
    optimal = {}

    for M in M_values:
        print(f"\n  M={M}: benchmarking {n_cfgs} configs ...")
        best_t = float("inf")
        best_cfg = None
        results_list = []
        t0 = time.time()

        for i, cfg in enumerate(all_cfgs):
            t = bench_kernel(M, cfg, "gemm1")
            results_list.append((t, cfg))
            if t < best_t:
                best_t = t
                best_cfg = cfg.copy()
            if (i + 1) % 64 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_cfgs - i - 1)
                print(f"    {i+1:>3}/{n_cfgs}  best={best_t:.4f} ms  "
                      f"ETA={eta:.0f}s")

        # Current E=32 baseline
        if e32_cfgs:
            closest = min(e32_cfgs.keys(), key=lambda x: abs(x - M))
            cur_cfg = e32_cfgs[closest]
            cur_t = bench_kernel(M, cur_cfg, "gemm1")
        else:
            cur_t = float("inf")

        speedup = cur_t / best_t if best_t > 0 and cur_t < float("inf") else 1.0
        print(f"    ── M={M} RESULT ──")
        print(f"    Current E=32: {cur_t:.4f} ms")
        print(f"    Best found:   {best_t:.4f} ms  ({speedup:.2f}x)")
        print(f"    Config: {best_cfg}")

        results_list.sort(key=lambda x: x[0])
        print(f"    Top-5:")
        for rank, (t, c) in enumerate(results_list[:5]):
            print(f"      #{rank+1}: {t:.4f} ms  "
                  f"BSM={c['BLOCK_SIZE_M']:>3} BSN={c['BLOCK_SIZE_N']:>3} "
                  f"BSK={c['BLOCK_SIZE_K']:>3} GSM={c['GROUP_SIZE_M']:>2} "
                  f"w={c['num_warps']} s={c['num_stages']}")

        optimal[str(M)] = best_cfg

    return optimal


def write_config(optimal, path):
    print(f"\n  Writing optimal config → {path}")
    with open(path, "w") as f:
        json.dump(optimal, f, indent=4)
    print(f"  Done ({len(optimal)} M entries)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compare", "search", "all"], default="all")
    parser.add_argument("--m-values", type=str, default=None,
                        help="Comma-separated M values (e.g. '256,2048')")
    parser.add_argument("--output", type=str,
                        default="/home/wuhang/wuhang/dllm_wh/codex_coding/results/"
                                "autotune_e32_n512_optimal.json")
    args = parser.parse_args()

    torch.cuda.set_device(0)

    M_values = (
        [int(x) for x in args.m_values.split(",")]
        if args.m_values
        else DEFAULT_M_VALUES
    )

    print(f"FusedMoE Auto-Tune: E={E_LOCAL}, N_inter={N_INTER}, K_hidden={K_HIDDEN}")
    print(f"GEMM1: [M, {K_HIDDEN}] × [{E_LOCAL}, {2*N_INTER}, {K_HIDDEN}]")
    print(f"M values: {M_values}")
    print(f"Search space: {len(gen_all_configs())} configs\n")

    if args.mode in ("compare", "all"):
        step_compare(M_values)

    if args.mode in ("search", "all"):
        optimal = step_search(M_values)
        write_config(optimal, args.output)

        print("\n" + "=" * 72)
        print("FINAL OPTIMAL CONFIG")
        print("=" * 72)
        for m, cfg in sorted(optimal.items(), key=lambda x: int(x[0])):
            print(f"  M={m:>5}: BSM={cfg['BLOCK_SIZE_M']:>3} BSN={cfg['BLOCK_SIZE_N']:>3} "
                  f"BSK={cfg['BLOCK_SIZE_K']:>3} GSM={cfg['GROUP_SIZE_M']:>2} "
                  f"w={cfg['num_warps']} s={cfg['num_stages']}")


if __name__ == "__main__":
    main()
