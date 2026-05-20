#!/usr/bin/env python3
"""
Benchmark + correctness verification for Stable Cache MoE optimization.

Compares:
  A: BD-attnmask (no-cache baseline)
  B: BD + cache-opt (cache baseline, lazy+inplace)
  C: BD + cache-opt + stable_cache (new optimization)

Tests:
  1. Performance: tok/s, fwd/s at batch=1 and batch=32
  2. Correctness: C vs B exact match at temperature=0.0
  3. Quality: token-level diff rate when not exact match
"""

from __future__ import annotations

import json
import os
import socket
import time
from contextlib import closing
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
RESULTS_DIR = REPO_ROOT / "codex_coding" / "results"

MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
GEN_LENGTH = 128
BLOCK_LENGTH = 32
THRESHOLD = 0.90
MASK_ID = 156895
EOS_ID = 156892
DEVICE = "cuda:0"

LONG_PROMPT = """Please solve the following problems step by step.

Problem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h. The total distance between the two cities is 240 km. What is the average speed for the entire round trip?

Problem 2: A rectangular garden has a perimeter of 56 meters. If the length is 4 meters more than twice the width, find the dimensions of the garden.

Problem 3: In a class of 40 students, 25 study Mathematics, 20 study Physics, and 10 study both subjects. How many students study neither Mathematics nor Physics?

Problem 4: A cone has a radius of 7 cm and a slant height of 25 cm. Calculate the total surface area and the volume of the cone.

Problem 5: A bank offers compound interest at 8% per annum, compounded quarterly. If you deposit $5000, how much will you have after 3 years?

Problem 6: Two pipes can fill a tank. Pipe A fills the tank in 12 hours and Pipe B fills it in 18 hours. If both pipes are opened together, but Pipe B is closed after 4 hours, how long will it take Pipe A alone to fill the remaining tank?

Problem 7: A sequence is defined as follows: a(1) = 2, a(2) = 5, and for n >= 3, a(n) = 2*a(n-1) - a(n-2) + 3. Find the first 8 terms.

Problem 8: A factory produces widgets on two assembly lines. Line A produces 300 widgets per hour with a defect rate of 2%. Line B produces 200 widgets per hour with a defect rate of 1.5%. If the factory runs both lines for 8 hours, what is the overall defect rate?

Problem 9: A cylindrical water tank with radius 3 meters and height 10 meters is being filled at a rate of 2 cubic meters per minute while being drained at 0.5 cubic meters per minute. How long will it take to fill completely?"""

BATCH_SIZES = [1, 32]
WARMUP_RUNS = 1
MEASURED_RUNS = 2


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sync_cuda(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def trim_after_eos(t, eos_id):
    p = (t == eos_id).nonzero(as_tuple=True)[0]
    return t[: int(p[0].item())] if p.numel() > 0 else t


def bench(name, dllm, input_ids, bs, device):
    print(f"\n  --- {name} batch={bs} ---", flush=True)
    batched = input_ids.repeat(bs, 1)
    plen = input_ids.shape[1]
    try:
        with torch.inference_mode():
            for _ in range(WARMUP_RUNS):
                dllm.generate(batched, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        runs = []
        with torch.inference_mode():
            for r in range(MEASURED_RUNS):
                pf = dllm.num_forwards
                sync_cuda(device)
                t0 = time.perf_counter()
                out = dllm.generate(batched, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
                sync_cuda(device)
                wall = time.perf_counter() - t0
                fwd = dllm.num_forwards - pf
                ttok = sum(int(trim_after_eos(out[b][plen:], EOS_ID).numel()) for b in range(bs))
                tp = ttok / wall
                fps = fwd / wall
                runs.append({"bs": bs, "tok": ttok, "fwd": fwd, "t": wall, "tp": tp, "fps": fps})
                print(f"    R{r+1}: {ttok}tok {fwd}fwd {wall:.3f}s tp={tp:.0f} fps={fps:.1f}", flush=True)
        med = median(r_["tp"] for r_ in runs)
        return min(runs, key=lambda r_: abs(r_["tp"] - med))
    except torch.cuda.OutOfMemoryError:
        print(f"    OOM!", flush=True)
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"    FAIL: {e}", flush=True)
        import traceback; traceback.print_exc()
        torch.cuda.empty_cache()
        return None


def correctness_check(name_b, dllm_b, name_c, dllm_c, input_ids, device, tokenizer):
    """Compare B (cache baseline) vs C (stable_cache) outputs for correctness."""
    print(f"\n{'='*80}")
    print(f"  CORRECTNESS CHECK: {name_b} vs {name_c}")
    print(f"{'='*80}")

    plen = input_ids.shape[1]
    results = {}

    with torch.inference_mode():
        # Run B
        out_b = dllm_b.generate(input_ids, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        fwd_b = dllm_b.num_forwards
        gen_b = out_b[0][plen:]
        text_b = tokenizer.decode(trim_after_eos(gen_b, EOS_ID), skip_special_tokens=True)

        # Run C
        out_c = dllm_c.generate(input_ids, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        fwd_c = dllm_c.num_forwards
        gen_c = out_c[0][plen:]
        text_c = tokenizer.decode(trim_after_eos(gen_c, EOS_ID), skip_special_tokens=True)

    # Compare
    gen_b_trimmed = trim_after_eos(gen_b, EOS_ID)
    gen_c_trimmed = trim_after_eos(gen_c, EOS_ID)

    exact_match = torch.equal(gen_b_trimmed, gen_c_trimmed)
    fwd_match = (fwd_b == fwd_c)

    # Token-level diff
    min_len = min(len(gen_b_trimmed), len(gen_c_trimmed))
    max_len = max(len(gen_b_trimmed), len(gen_c_trimmed))
    if min_len > 0:
        matching_tokens = (gen_b_trimmed[:min_len] == gen_c_trimmed[:min_len]).sum().item()
        token_match_rate = matching_tokens / max_len * 100
    else:
        token_match_rate = 0.0

    print(f"  Forward count:  B={fwd_b}, C={fwd_c} {'MATCH' if fwd_match else 'DIFFER'}")
    print(f"  Token lengths:  B={len(gen_b_trimmed)}, C={len(gen_c_trimmed)}")
    print(f"  Exact match:    {'YES' if exact_match else 'NO'}")
    print(f"  Token match:    {token_match_rate:.1f}% ({matching_tokens}/{max_len})")
    print(f"\n  B text: {text_b[:200]}...")
    print(f"  C text: {text_c[:200]}...")

    if not exact_match and min_len > 0:
        # Find first divergence point
        diff_positions = (gen_b_trimmed[:min_len] != gen_c_trimmed[:min_len]).nonzero(as_tuple=True)[0]
        if len(diff_positions) > 0:
            first_diff = diff_positions[0].item()
            print(f"\n  First divergence at position {first_diff}:")
            print(f"    B: ...{tokenizer.decode(gen_b_trimmed[max(0,first_diff-3):first_diff+5])}...")
            print(f"    C: ...{tokenizer.decode(gen_c_trimmed[max(0,first_diff-3):first_diff+5])}...")

    results = {
        "exact_match": exact_match,
        "fwd_match": fwd_match,
        "fwd_b": fwd_b,
        "fwd_c": fwd_c,
        "len_b": len(gen_b_trimmed),
        "len_c": len(gen_c_trimmed),
        "token_match_rate": token_match_rate,
        "text_b_preview": text_b[:300],
        "text_c_preview": text_c[:300],
    }
    return results


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (
        BlockDiffusionLLM,
        BlockDiffusionLLMAttnmask,
        BlockIteratorFactory,
        KVCacheFactory,
        StableCacheBlockDiffusionLLM,
        ThresholdParallelDecoder,
    )
    from dinfer.model import LLaDA2MoeModelLM

    port = find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("Loading model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=cfg).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)

        def dec():
            return ThresholdParallelDecoder(temperature=0.0, threshold=THRESHOLD, mask_id=MASK_ID, eos_id=EOS_ID)

        def tok(text):
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                     add_generation_prompt=True, tokenize=False)
            return tokenizer(text, return_tensors="pt")["input_ids"].to(device)

        long_ids = tok(LONG_PROMPT)
        print(f"Long prompt: {long_ids.shape[1]} tokens\n")

        all_results = {"perf": {}, "correctness": {}}

        # === Phase 1: Correctness check (batch=1, temperature=0.0) ===
        print(f"\n{'#'*90}")
        print(f"  PHASE 1: CORRECTNESS VERIFICATION (batch=1, temperature=0.0)")
        print(f"{'#'*90}")

        dllm_b = BlockDiffusionLLM(
            model, dec(), BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, lazy_cache_update=True, inplace_cache_update=True)

        dllm_c = StableCacheBlockDiffusionLLM(
            model, dec(), BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, lazy_cache_update=True, inplace_cache_update=True)

        corr = correctness_check("B:cache-opt", dllm_b, "C:stable_cache", dllm_c,
                                 long_ids, device, tokenizer)
        all_results["correctness"]["long_prompt"] = corr

        # === Phase 2: Performance benchmark ===
        print(f"\n{'#'*90}")
        print(f"  PHASE 2: PERFORMANCE BENCHMARK")
        print(f"{'#'*90}")

        for bs in BATCH_SIZES:
            print(f"\n{'='*80}")
            print(f"  BATCH SIZE = {bs}")
            print(f"{'='*80}")

            # A: no-cache baseline
            dllm = BlockDiffusionLLMAttnmask(
                model, dec(), BlockIteratorFactory(use_block_diffusion=True), early_stop=True)
            r = bench("A:no-cache", dllm, long_ids, bs, device)
            all_results["perf"].setdefault(f"batch_{bs}", {})["A:no-cache"] = r

            # B: cache-opt baseline
            dllm = BlockDiffusionLLM(
                model, dec(), BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, lazy_cache_update=True, inplace_cache_update=True)
            r = bench("B:cache-opt", dllm, long_ids, bs, device)
            all_results["perf"][f"batch_{bs}"]["B:cache-opt"] = r

            # C: stable_cache
            dllm = StableCacheBlockDiffusionLLM(
                model, dec(), BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, lazy_cache_update=True, inplace_cache_update=True)
            r = bench("C:stable_cache", dllm, long_ids, bs, device)
            all_results["perf"][f"batch_{bs}"]["C:stable_cache"] = r

        # === Summary ===
        print(f"\n{'#'*90}")
        print(f"  SUMMARY")
        print(f"{'#'*90}")

        print(f"\n  Correctness: exact_match={corr['exact_match']}, "
              f"token_match={corr['token_match_rate']:.1f}%, "
              f"fwd B={corr['fwd_b']} C={corr['fwd_c']}")

        print(f"\n  {'Config':<20s}", end="")
        for bs in BATCH_SIZES:
            print(f"  {'b='+str(bs)+' tp':>12s} {'fwd/s':>8s}", end="")
        print()
        print(f"  {'-'*20}" + "".join(f"  {'-'*12} {'-'*8}" for _ in BATCH_SIZES))

        for path in ["A:no-cache", "B:cache-opt", "C:stable_cache"]:
            print(f"  {path:<20s}", end="")
            for bs in BATCH_SIZES:
                r = all_results["perf"].get(f"batch_{bs}", {}).get(path)
                if r:
                    print(f"  {r['tp']:>12.0f} {r['fps']:>8.1f}", end="")
                else:
                    print(f"  {'OOM':>12s} {'—':>8s}", end="")
            print()

        # Speedup
        for bs in BATCH_SIZES:
            b_r = all_results["perf"].get(f"batch_{bs}", {}).get("B:cache-opt")
            c_r = all_results["perf"].get(f"batch_{bs}", {}).get("C:stable_cache")
            if b_r and c_r:
                speedup = c_r["tp"] / b_r["tp"]
                print(f"  stable_cache speedup (b={bs}): {speedup:.3f}x")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "stable_cache_benchmark_results.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2) + "\n")
    print(f"\nSaved: {out_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
