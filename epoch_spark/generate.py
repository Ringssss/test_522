#!/usr/bin/env python3
"""
Phase 4: E2E Diffusion Generation with Epoch-Spark.

Full pipeline: load model → split weights → block-boundary planning →
async staging → compact MoE execution → threshold decode.

Modes:
  A: baseline (direct MoE, all weights on GPU)
  B: epoch-spark (block-scoped residency + decoded-token cache)

Usage:
    cd ~/epoch_spark
    /home/zhujianian/miniconda3/envs/crossstage/bin/python generate.py --mode baseline
    /home/zhujianian/miniconda3/envs/crossstage/bin/python generate.py --mode epoch-spark --gpu-budget 80
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MASK_ID, BLOCK_LENGTH, PROMPTS, NUM_EXPERTS,
    DEFAULT_GPU_EXPERT_BUDGET, DEFAULT_DECODED_CACHE_REFRESH_M,
)
from utils import load_model_and_tokenizer, gpu_mem_mb, Timer


@torch.no_grad()
def generate_baseline(model, tokenizer, prompt, gen_length=128,
                      steps_per_block=10, device="cuda:0"):
    """Baseline: standard diffusion generation, all weights on GPU."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    n_blocks = (gen_length + BLOCK_LENGTH - 1) // BLOCK_LENGTH

    x = torch.cat([
        input_ids,
        torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)
    ], dim=1)
    seq_len = x.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    timings = []

    for block_id in range(n_blocks):
        block_start = prompt_len + block_id * BLOCK_LENGTH
        block_end = min(block_start + BLOCK_LENGTH, seq_len)

        remaining = (x[0, block_start:block_end] == MASK_ID).sum().item()
        if remaining == 0:
            continue

        for step in range(steps_per_block):
            if remaining <= 0:
                break
            n_transfer = max(1, remaining // (steps_per_block - step))

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            out = model(input_ids=x, position_ids=position_ids,
                        use_cache=False, return_dict=True)

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000)

            logits = out.logits[0, block_start:block_end]
            block_x = x[0, block_start:block_end]
            live = (block_x == MASK_ID)
            if not live.any():
                break

            probs = torch.softmax(logits.float(), dim=-1)
            pred = logits.argmax(dim=-1)
            conf = probs.max(dim=-1).values
            conf[~live] = -1.0

            n_dec = min(n_transfer, live.sum().item())
            if n_dec > 0:
                _, top_idx = conf.topk(n_dec)
                x[0, block_start + top_idx] = pred[top_idx]
                remaining -= n_dec

    return x, timings


@torch.no_grad()
def generate_epoch_spark(model, tokenizer, prompt, gen_length=128,
                         steps_per_block=10, device="cuda:0",
                         gpu_budget=DEFAULT_GPU_EXPERT_BUDGET,
                         refresh_m=DEFAULT_DECODED_CACHE_REFRESH_M,
                         rmgr=None, controller=None, already_patched=False):
    """Epoch-Spark: block-scoped residency + decoded-token cache."""
    from residency_manager import BlockResidencyManager
    from block_moe_forward import BlockMoEController, patch_model_with_block_moe, profile_routing

    if rmgr is None:
        rmgr = BlockResidencyManager(model, device=device, gpu_budget_per_layer=gpu_budget)
    if controller is None:
        controller = BlockMoEController(rmgr, refresh_m=refresh_m)
    if not already_patched:
        patch_model_with_block_moe(model, controller)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    n_blocks = (gen_length + BLOCK_LENGTH - 1) // BLOCK_LENGTH

    x = torch.cat([
        input_ids,
        torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)
    ], dim=1)
    seq_len = x.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    timings = []
    plan_timings = []

    for block_id in range(n_blocks):
        block_start = prompt_len + block_id * BLOCK_LENGTH
        block_end = min(block_start + BLOCK_LENGTH, seq_len)

        remaining = (x[0, block_start:block_end] == MASK_ID).sum().item()
        if remaining == 0:
            continue

        # Block boundary: profile routing from first forward
        torch.cuda.synchronize()
        tp0 = time.perf_counter()

        routing_profiles = profile_routing(model, x, position_ids)
        controller.on_block_start(block_id, x[0], routing_profiles)

        torch.cuda.synchronize()
        tp1 = time.perf_counter()
        plan_timings.append((tp1 - tp0) * 1000)

        for step in range(steps_per_block):
            if remaining <= 0:
                break
            n_transfer = max(1, remaining // (steps_per_block - step))

            controller.on_iter_start(step, x[0])

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            out = model(input_ids=x, position_ids=position_ids,
                        use_cache=False, return_dict=True)

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000)

            controller.age_caches()

            logits = out.logits[0, block_start:block_end]
            block_x = x[0, block_start:block_end]
            live = (block_x == MASK_ID)
            if not live.any():
                break

            probs = torch.softmax(logits.float(), dim=-1)
            pred = logits.argmax(dim=-1)
            conf = probs.max(dim=-1).values
            conf[~live] = -1.0

            n_dec = min(n_transfer, live.sum().item())
            if n_dec > 0:
                _, top_idx = conf.topk(n_dec)
                x[0, block_start + top_idx] = pred[top_idx]
                remaining -= n_dec

    extra_stats = {
        "rmgr_stats": rmgr.get_stats(),
        "gpu_hit_rate": rmgr.get_gpu_hit_rate(),
        "gpu_cache_mb": rmgr.gpu_cache_mb(),
        "cpu_pool_mb": rmgr.cpu_pool_mb(),
        "controller_stats": dict(controller.stats),
        "plan_timings_ms": plan_timings,
    }
    return x, timings, extra_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both", choices=["baseline", "epoch-spark", "both"])
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps-per-block", type=int, default=10)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu-budget", type=int, default=DEFAULT_GPU_EXPERT_BUDGET)
    parser.add_argument("--refresh-m", type=int, default=DEFAULT_DECODED_CACHE_REFRESH_M)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    prompts = PROMPTS[:args.num_prompts]

    print(f"[E2E] Mode: {args.mode}, gen_length={args.gen_length}, "
          f"prompts={len(prompts)}, gpu_budget={args.gpu_budget}")

    results = {}

    if args.mode in ("baseline", "both"):
        print("\n" + "=" * 70)
        print("Running BASELINE (all weights GPU, standard MoE)")
        print("=" * 70)

        model, tokenizer, config = load_model_and_tokenizer(device=device)
        mem_baseline = gpu_mem_mb(args.gpu)
        print(f"[baseline] GPU mem: {mem_baseline:.0f} MB")

        all_timings = []
        for i, prompt in enumerate(prompts):
            print(f"\n[baseline] Prompt {i+1}: {prompt[:60]}...")
            x, timings = generate_baseline(
                model, tokenizer, prompt,
                gen_length=args.gen_length,
                steps_per_block=args.steps_per_block,
                device=device,
            )
            all_timings.extend(timings)
            text = tokenizer.decode(x[0], skip_special_tokens=True)
            print(f"  Output: {text[:200]}")
            print(f"  Avg forward: {sum(timings)/len(timings):.2f} ms ({len(timings)} forwards)")

        results["baseline"] = {
            "avg_forward_ms": sum(all_timings) / len(all_timings),
            "total_forwards": len(all_timings),
            "gpu_mem_mb": mem_baseline,
        }
        del model
        torch.cuda.empty_cache()

    if args.mode in ("epoch-spark", "both"):
        print("\n" + "=" * 70)
        print(f"Running EPOCH-SPARK (gpu_budget={args.gpu_budget}, refresh_m={args.refresh_m})")
        print("=" * 70)

        model, tokenizer, config = load_model_and_tokenizer(device=device)

        from residency_manager import BlockResidencyManager
        from block_moe_forward import BlockMoEController, patch_model_with_block_moe
        shared_rmgr = BlockResidencyManager(model, device=device, gpu_budget_per_layer=args.gpu_budget)
        shared_controller = BlockMoEController(shared_rmgr, refresh_m=args.refresh_m)
        patch_model_with_block_moe(model, shared_controller)

        all_timings = []
        all_extra = []
        for i, prompt in enumerate(prompts):
            print(f"\n[epoch-spark] Prompt {i+1}: {prompt[:60]}...")

            x, timings, extra = generate_epoch_spark(
                model, tokenizer, prompt,
                gen_length=args.gen_length,
                steps_per_block=args.steps_per_block,
                device=device,
                gpu_budget=args.gpu_budget,
                refresh_m=args.refresh_m,
                rmgr=shared_rmgr,
                controller=shared_controller,
                already_patched=True,
            )
            all_timings.extend(timings)
            all_extra.append(extra)
            text = tokenizer.decode(x[0], skip_special_tokens=True)
            print(f"  Output: {text[:200]}")
            print(f"  Avg forward: {sum(timings)/len(timings):.2f} ms ({len(timings)} forwards)")
            print(f"  GPU hit rate: {extra['gpu_hit_rate']:.4f}")
            print(f"  GPU cache: {extra['gpu_cache_mb']:.1f} MB, CPU pool: {extra['cpu_pool_mb']:.1f} MB")
            print(f"  Controller: {extra['controller_stats']}")

        results["epoch_spark"] = {
            "avg_forward_ms": sum(all_timings) / len(all_timings),
            "total_forwards": len(all_timings),
            "gpu_budget": args.gpu_budget,
            "avg_gpu_hit_rate": sum(e["gpu_hit_rate"] for e in all_extra) / len(all_extra),
            "avg_gpu_cache_mb": sum(e["gpu_cache_mb"] for e in all_extra) / len(all_extra),
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for mode, data in results.items():
        print(f"\n{mode}:")
        for k, v in data.items():
            print(f"  {k}: {v}")

    with open("e2e_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to e2e_results.json")


if __name__ == "__main__":
    main()
