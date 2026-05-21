#!/usr/bin/env python3
"""
Phase 5: Benchmark and Validation.

Runs baseline and epoch-spark in separate invocations (to avoid FusedMoE
global registry conflicts) and compiles comparison results.

Usage:
    cd ~/epoch_spark
    /home/zhujianian/miniconda3/envs/crossstage/bin/python benchmark.py
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PYTHON = "/home/zhujianian/miniconda3/envs/crossstage/bin/python"
CWD = str(Path(__file__).parent)


def run_mode(mode, num_prompts=4, gen_length=128, steps=10, gpu_budget=100, gpu=0):
    """Run generate.py in a subprocess and parse results."""
    cmd = [
        PYTHON, "generate.py",
        f"--mode={mode}",
        f"--num-prompts={num_prompts}",
        f"--gen-length={gen_length}",
        f"--steps-per-block={steps}",
        f"--gpu={gpu}",
    ]
    if mode == "epoch-spark":
        cmd.append(f"--gpu-budget={gpu_budget}")

    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=CWD, timeout=1200)

    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"STDERR (last 1000): {result.stderr[-1000:]}")
        return None

    results_file = Path(CWD) / "e2e_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu-budgets", type=str, default="60,80,100,120",
                        help="Comma-separated GPU budgets to test")
    args = parser.parse_args()

    all_results = {}

    # Run baseline
    r = run_mode("baseline", args.num_prompts, args.gen_length, args.steps, gpu=args.gpu)
    if r and "baseline" in r:
        all_results["baseline"] = r["baseline"]

    # Run epoch-spark at different GPU budgets
    for budget in [int(b) for b in args.gpu_budgets.split(",")]:
        r = run_mode("epoch-spark", args.num_prompts, args.gen_length, args.steps,
                     gpu_budget=budget, gpu=args.gpu)
        if r and "epoch_spark" in r:
            all_results[f"epoch_spark_b{budget}"] = r["epoch_spark"]

    # Print comparison table
    print("\n" + "=" * 90)
    print("BENCHMARK COMPARISON")
    print("=" * 90)
    print(f"{'Config':<25} | {'Avg ms/fwd':>12} | {'Speedup':>8} | {'GPU MB':>10} | {'Hit Rate':>10}")
    print("-" * 90)

    baseline_ms = all_results.get("baseline", {}).get("avg_forward_ms", 0)
    for name, data in sorted(all_results.items()):
        ms = data.get("avg_forward_ms", 0)
        speedup = baseline_ms / ms if ms > 0 and baseline_ms > 0 else 0
        gpu_mb = data.get("gpu_mem_mb", data.get("avg_gpu_cache_mb", 0))
        hit = data.get("avg_gpu_hit_rate", -1)
        hit_str = f"{hit:.4f}" if hit >= 0 else "N/A"
        print(f"{name:<25} | {ms:12.2f} | {speedup:8.2f}x | {gpu_mb:10.0f} | {hit_str:>10}")

    print("=" * 90)

    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nFull results saved to benchmark_results.json")


if __name__ == "__main__":
    main()
