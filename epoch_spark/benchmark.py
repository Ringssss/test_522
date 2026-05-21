#!/usr/bin/env python3
"""
Comprehensive Epoch-Spark Benchmark.

Tests baseline vs epoch-spark with fused Triton MoE at bs=1, 16, 64.
Reports: latency/forward, throughput (tokens/sec), GPU memory, and output quality comparison.

Usage:
    cd ~/epoch_spark
    /home/zhujianian/miniconda3/envs/crossstage/bin/python benchmark.py
"""
import sys, time, json, argparse
sys.path.insert(0, '.')
import torch
import numpy as np

from config import MASK_ID, BLOCK_LENGTH, PROMPTS, DEFAULT_GPU_EXPERT_BUDGET
from utils import load_model_and_tokenizer, gpu_mem_mb


# ════════════════════════════════════════════════════════════════
# Generation loop (shared by baseline and epoch-spark)
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def diffusion_generate(model, x, prompt_len, gen_length, steps_per_block,
                       controller=None):
    """Run diffusion generation. Returns output tokens and per-forward timings."""
    device = x.device
    seq_len = x.shape[1]
    n_blocks = (gen_length + BLOCK_LENGTH - 1) // BLOCK_LENGTH
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(x.shape[0], -1)

    timings = []

    for block_id in range(n_blocks):
        block_start = prompt_len + block_id * BLOCK_LENGTH
        block_end = min(block_start + BLOCK_LENGTH, seq_len)

        remaining = (x[:, block_start:block_end] == MASK_ID).sum(dim=1).max().item()
        if remaining == 0:
            continue

        if controller is not None:
            controller.on_block_start(block_id, x[0])

        first_iter_of_block = True

        for step in range(steps_per_block):
            if remaining <= 0:
                break
            n_transfer = max(1, remaining // (steps_per_block - step))

            if controller is not None:
                controller.on_iter_start(step, x[0])

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(input_ids=x, position_ids=position_ids,
                        use_cache=False, return_dict=True)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000)

            if controller is not None:
                controller.age_caches()
                first_iter_of_block = False

            logits = out.logits[:, block_start:block_end]  # [B, block, V]
            block_x = x[:, block_start:block_end]
            live = (block_x == MASK_ID)

            if not live.any():
                break

            probs = torch.softmax(logits.float(), dim=-1)
            pred = logits.argmax(dim=-1)
            conf = probs.max(dim=-1).values
            conf[~live] = -1.0

            # Decode per-sample
            for b in range(x.shape[0]):
                b_live = live[b]
                if not b_live.any():
                    continue
                b_conf = conf[b].clone()
                b_conf[~b_live] = -1.0
                n_dec = min(n_transfer, b_live.sum().item())
                if n_dec > 0:
                    _, top_idx = b_conf.topk(n_dec)
                    x[b, block_start + top_idx] = pred[b, top_idx]

            remaining = (x[:, block_start:block_end] == MASK_ID).sum(dim=1).max().item()

    return x, timings


# ════════════════════════════════════════════════════════════════
# Main benchmark
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu-budget", type=int, default=100)
    parser.add_argument("--batch-sizes", type=str, default="1,16,64")
    parser.add_argument("--num-prompts", type=int, default=8)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    print("=" * 80)
    print(f"Epoch-Spark Comprehensive Benchmark")
    print(f"  Model: LLaDA2.0-mini (16B, 256 experts)")
    print(f"  GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"  gen_length={args.gen_length}, steps={args.steps}")
    print(f"  batch_sizes={batch_sizes}")
    print("=" * 80)

    model, tokenizer, config = load_model_and_tokenizer(device=device)
    mem_after_load = gpu_mem_mb(args.gpu)
    print(f"[model] GPU memory after load: {mem_after_load:.0f} MB")

    # Prepare batched inputs
    all_prompts = PROMPTS[:args.num_prompts]

    results = {}

    for bs in batch_sizes:
        print(f"\n{'='*70}")
        print(f"  Batch size = {bs}")
        print(f"{'='*70}")

        prompts_for_batch = [all_prompts[i % len(all_prompts)] for i in range(bs)]
        encoded = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in prompts_for_batch]
        max_prompt_len = max(e.shape[0] for e in encoded)

        # Pad prompts to same length + append gen region
        x_list = []
        for e in encoded:
            pad_len = max_prompt_len - e.shape[0]
            padded = torch.cat([
                torch.full((pad_len,), tokenizer.pad_token_id or 0, dtype=torch.long),
                e,
                torch.full((args.gen_length,), MASK_ID, dtype=torch.long),
            ])
            x_list.append(padded)
        x_batch = torch.stack(x_list).to(device)  # [bs, prompt+gen]
        prompt_len = max_prompt_len
        total_seq = x_batch.shape[1]

        print(f"  Input shape: {x_batch.shape}, prompt_len={prompt_len}")

        # ── Baseline ──
        print(f"\n  [BASELINE] Running...")
        torch.cuda.reset_peak_memory_stats(args.gpu)
        x_base = x_batch.clone()
        x_base_out, timings_base = diffusion_generate(
            model, x_base, prompt_len, args.gen_length, args.steps,
        )
        peak_base = torch.cuda.max_memory_allocated(args.gpu) / (1024**2)
        avg_base = np.mean(timings_base)
        total_tokens_base = bs * args.gen_length
        total_time_base = sum(timings_base) / 1000
        tps_base = total_tokens_base / total_time_base

        base_texts = [tokenizer.decode(x_base_out[i], skip_special_tokens=True) for i in range(bs)]
        print(f"  [BASELINE] avg={avg_base:.2f} ms/fwd, peak_mem={peak_base:.0f} MB, "
              f"throughput={tps_base:.1f} tok/s ({len(timings_base)} fwds)")
        print(f"  [BASELINE] Sample output: {base_texts[0][:150]}")

        # ── Epoch-Spark ──
        print(f"\n  [EPOCH-SPARK] Running (budget={args.gpu_budget})...")
        from block_moe_forward import BlockMoEController, patch_model_with_block_moe

        controller = BlockMoEController(residency_mgr=None, refresh_m=5)
        patch_model_with_block_moe(model, controller)

        torch.cuda.reset_peak_memory_stats(args.gpu)
        x_es = x_batch.clone()
        x_es_out, timings_es = diffusion_generate(
            model, x_es, prompt_len, args.gen_length, args.steps,
            controller=controller,
        )
        peak_es = torch.cuda.max_memory_allocated(args.gpu) / (1024**2)
        avg_es = np.mean(timings_es)
        total_time_es = sum(timings_es) / 1000
        tps_es = total_tokens_base / total_time_es

        es_texts = [tokenizer.decode(x_es_out[i], skip_special_tokens=True) for i in range(bs)]
        speedup = avg_base / avg_es if avg_es > 0 else 0
        cached_pct = controller.stats["cached_tokens"] / max(controller.stats["total_tokens"], 1) * 100

        print(f"  [EPOCH-SPARK] avg={avg_es:.2f} ms/fwd, peak_mem={peak_es:.0f} MB, "
              f"throughput={tps_es:.1f} tok/s ({len(timings_es)} fwds)")
        print(f"  [EPOCH-SPARK] speedup={speedup:.2f}x, cached={cached_pct:.1f}%")
        print(f"  [EPOCH-SPARK] Sample output: {es_texts[0][:150]}")

        # ── Quality comparison ──
        match_count = sum(1 for a, b in zip(base_texts, es_texts) if a == b)
        token_match = (x_base_out == x_es_out).float().mean().item() * 100
        print(f"\n  [QUALITY] Exact text match: {match_count}/{bs}")
        print(f"  [QUALITY] Token-level match: {token_match:.1f}%")

        # Reset patches for next batch size — restore fused forward (not block-scoped)
        from utils import _fused_moe_forward
        from vllm.model_executor.layers.fused_moe import FusedMoE
        for name, mod in model.named_modules():
            if isinstance(mod, FusedMoE):
                mod.forward = lambda *a, _mod=mod, **kw: _fused_moe_forward(
                    _mod,
                    kw.get("hidden_states", a[0] if a else None),
                    kw.get("router_logits", a[1] if len(a) > 1 else None),
                )
        controller.stats.clear()

        results[f"bs{bs}"] = {
            "baseline": {
                "avg_ms": round(avg_base, 2),
                "throughput_tps": round(tps_base, 1),
                "peak_mem_mb": round(peak_base, 0),
                "n_forwards": len(timings_base),
            },
            "epoch_spark": {
                "avg_ms": round(avg_es, 2),
                "throughput_tps": round(tps_es, 1),
                "peak_mem_mb": round(peak_es, 0),
                "speedup": round(speedup, 2),
                "cached_pct": round(cached_pct, 1),
                "n_forwards": len(timings_es),
            },
            "quality": {
                "exact_match": f"{match_count}/{bs}",
                "token_match_pct": round(token_match, 1),
            },
        }

    # ── Final summary table ──
    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)
    print(f"{'BS':>4} | {'Mode':<14} | {'ms/fwd':>8} | {'Speedup':>8} | {'tok/s':>10} | "
          f"{'PeakMB':>8} | {'Cached%':>8} | {'TokMatch':>9}")
    print("-" * 100)

    for bsk in sorted(results.keys(), key=lambda k: int(k[2:])):
        bs_val = bsk[2:]
        r = results[bsk]
        b = r["baseline"]
        e = r["epoch_spark"]
        q = r["quality"]
        print(f"{bs_val:>4} | {'Baseline':<14} | {b['avg_ms']:8.2f} | {'1.00x':>8} | "
              f"{b['throughput_tps']:10.1f} | {b['peak_mem_mb']:8.0f} | {'—':>8} | {'—':>9}")
        print(f"{bs_val:>4} | {'Epoch-Spark':<14} | {e['avg_ms']:8.2f} | {e['speedup']:.2f}x"
              f"   | {e['throughput_tps']:10.1f} | {e['peak_mem_mb']:8.0f} | "
              f"{e['cached_pct']:7.1f}% | {q['token_match_pct']:8.1f}%")

    print("=" * 100)

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
