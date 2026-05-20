#!/usr/bin/env python3
"""
Benchmark: BD + prefix cache with lazy cache update vs existing approaches.

Tests three paths x short/long prompt:
  A. BD attnmask (baseline, no cache, full recompute)
  B. BD + prefix cache (existing, per-step kv_cache.update)
  C. BD + prefix cache + lazy update (skip per-step update)

Includes output verification: compares generated text and forward counts.
"""

from __future__ import annotations

import json
import os
import socket
import sys
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

SHORT_PROMPT = (
    "Lily can run 12 kilometers per hour for 4 hours. "
    "After that, she can run 6 kilometers per hour. "
    "How many kilometers can she run in 8 hours?"
)

LONG_PROMPT = """Please solve the following problems step by step.

Problem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h. The total distance between the two cities is 240 km. What is the average speed for the entire round trip? Show your complete working.

Problem 2: A rectangular garden has a perimeter of 56 meters. If the length is 4 meters more than twice the width, find the dimensions of the garden. Then calculate the area and determine how many square tiles of side 0.5 meters would be needed to cover the entire garden.

Problem 3: In a class of 40 students, 25 study Mathematics, 20 study Physics, and 10 study both subjects. How many students study neither Mathematics nor Physics? Draw a Venn diagram description and explain your reasoning step by step.

Problem 4: A cone has a radius of 7 cm and a slant height of 25 cm. Calculate the total surface area and the volume of the cone. Use pi = 3.14159 and show all intermediate calculations.

Problem 5: A bank offers compound interest at 8% per annum, compounded quarterly. If you deposit $5000, how much will you have after 3 years? What would be the difference if the interest were compounded monthly instead? Show the formula and all steps.

Problem 6: Two pipes can fill a tank. Pipe A fills the tank in 12 hours and Pipe B fills it in 18 hours. If both pipes are opened together, but Pipe B is closed after 4 hours, how long will it take Pipe A alone to fill the remaining tank? Explain each step.

Problem 7: A sequence is defined as follows: a(1) = 2, a(2) = 5, and for n >= 3, a(n) = 2*a(n-1) - a(n-2) + 3. Find the first 8 terms of the sequence and determine whether there is a pattern in the differences between consecutive terms.

Problem 8: A factory produces widgets on two assembly lines. Line A produces 300 widgets per hour with a defect rate of 2%. Line B produces 200 widgets per hour with a defect rate of 1.5%. If the factory runs both lines for 8 hours, how many total widgets are produced, how many are expected to be defective, and what is the overall defect rate? Show all calculations step by step.

Problem 9: A cylindrical water tank with radius 3 meters and height 10 meters is being filled at a rate of 2 cubic meters per minute. At the same time, water is being drained from the bottom at a rate of 0.5 cubic meters per minute. How long will it take to fill the tank completely if it starts empty? What if the drain rate increases to 1 cubic meter per minute after the tank is half full?"""

WARMUP_RUNS = 1
MEASURED_RUNS = 3


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def trim_after_eos(token_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    eos_pos = (token_ids == eos_id).nonzero(as_tuple=True)[0]
    if eos_pos.numel() > 0:
        return token_ids[: int(eos_pos[0].item())]
    return token_ids


def bench_dllm(name, dllm, input_ids, device, tokenizer):
    """Benchmark a dLLM object and return result dict with generated text."""
    print(f"\n--- {name} ---", flush=True)
    try:
        # Warmup
        print(f"  Warmup ({WARMUP_RUNS})...", flush=True)
        with torch.inference_mode():
            for _ in range(WARMUP_RUNS):
                dllm.generate(input_ids, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)

        runs = []
        gen_text = None
        with torch.inference_mode():
            for r in range(MEASURED_RUNS):
                prev_fwd = dllm.num_forwards
                sync_cuda(device)
                t0 = time.perf_counter()
                out = dllm.generate(input_ids, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
                sync_cuda(device)
                wall = time.perf_counter() - t0
                fwd = dllm.num_forwards - prev_fwd
                gen_ids = trim_after_eos(out[0][input_ids.shape[1]:], EOS_ID)
                tok = int(gen_ids.numel())
                tps = tok / wall if wall > 0 else 0
                fps = fwd / wall if wall > 0 else 0
                runs.append({"tokens": tok, "forwards": fwd, "time": wall, "tok/s": tps, "fwd/s": fps})
                print(f"  Run {r+1}: {tok} tok, {fwd} fwd, {wall:.3f}s, {tps:.1f} tok/s, {fps:.1f} fwd/s", flush=True)
                # Save text from median-like run
                if r == MEASURED_RUNS - 1:
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        med_tps = median(r_["tok/s"] for r_ in runs)
        best = min(runs, key=lambda r_: abs(r_["tok/s"] - med_tps))
        return {"name": name, **best, "all_runs": runs, "generated_text": gen_text}
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()
        return None


def verify_results(results, label):
    """Verify output consistency between paths."""
    print(f"\n{'='*80}")
    print(f"  VERIFICATION: {label}")
    print(f"{'='*80}")

    valid = [r for r in results if r is not None]
    if len(valid) < 2:
        print("  Not enough valid results to compare.")
        return

    # Find specific paths
    baseline = next((r for r in valid if "attnmask" in r["name"]), None)
    existing_cache = next((r for r in valid if "existing" in r["name"]), None)
    lazy_cache = next((r for r in valid if r is not None and "lazy" in r["name"] and "inplace" not in r["name"]), None)
    inplace_cache = next((r for r in valid if r is not None and "inplace" in r["name"] and "lazy" not in r["name"]), None)
    combined_cache = next((r for r in valid if r is not None and "lazy+inplace" in r["name"]), None)

    all_pass = True

    # Check each optimized path vs existing cache
    for opt_name, opt_result in [("lazy", lazy_cache), ("inplace", inplace_cache), ("lazy+inplace", combined_cache)]:
        if existing_cache and opt_result:
            fwd_match = existing_cache["forwards"] == opt_result["forwards"]
            status = "PASS" if fwd_match else "WARN"
            if not fwd_match:
                all_pass = False
            print(f"  [{status}] Forward count: existing={existing_cache['forwards']}, {opt_name}={opt_result['forwards']}")

            text_match = existing_cache["generated_text"] == opt_result["generated_text"]
            status = "PASS" if text_match else "WARN"
            if not text_match:
                all_pass = False
            print(f"  [{status}] Text match ({opt_name} vs existing): {'identical' if text_match else 'DIFFERENT'}")
            if not text_match:
                print(f"    existing: {existing_cache['generated_text'][:100]}...")
                print(f"    {opt_name}:  {opt_result['generated_text'][:100]}...")

    # fwd/s comparison
    print(f"\n  fwd/s comparison:")
    if baseline:
        print(f"    baseline (no cache):  {baseline['fwd/s']:.1f}")
    if existing_cache:
        print(f"    existing cache:       {existing_cache['fwd/s']:.1f}")
    for opt_name, opt_result in [("lazy", lazy_cache), ("inplace", inplace_cache), ("lazy+inplace", combined_cache)]:
        if opt_result and existing_cache:
            delta = (opt_result["fwd/s"] - existing_cache["fwd/s"]) / existing_cache["fwd/s"] * 100
            gap = (opt_result["fwd/s"] - baseline["fwd/s"]) / baseline["fwd/s"] * 100 if baseline else 0
            print(f"    {opt_name:<20s} {opt_result['fwd/s']:.1f}  ({delta:+.1f}% vs existing, {gap:+.1f}% vs baseline)")

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED - see above'}")


def run_suite(label, input_ids, device, model, make_decoder, tokenizer):
    """Run all three paths for one prompt length."""
    from dinfer import (
        BlockDiffusionLLMAttnmask,
        BlockDiffusionLLM,
        BlockIteratorFactory,
        KVCacheFactory,
    )

    prompt_len = input_ids.shape[1]
    print(f"\n{'='*90}")
    print(f"  {label}  (prompt={prompt_len} tok, gen={GEN_LENGTH}, block={BLOCK_LENGTH}, threshold={THRESHOLD})")
    print(f"{'='*90}")

    results = []

    # A. BD attnmask (baseline, no cache)
    dllm = BlockDiffusionLLMAttnmask(
        model, make_decoder(),
        BlockIteratorFactory(use_block_diffusion=True),
        early_stop=True,
    )
    results.append(bench_dllm("A: BD attnmask (baseline)", dllm, input_ids, device, tokenizer))

    # B. BD + prefix cache (existing, per-step update)
    dllm = BlockDiffusionLLM(
        model, make_decoder(),
        BlockIteratorFactory(use_block_diffusion=True),
        cache_factory=KVCacheFactory("prefix", is_bd_model=True),
        early_stop=True,
        lazy_cache_update=False,
    )
    results.append(bench_dllm("B: BD+cache existing", dllm, input_ids, device, tokenizer))

    # C. BD + prefix cache + lazy update (skip per-step update)
    dllm = BlockDiffusionLLM(
        model, make_decoder(),
        BlockIteratorFactory(use_block_diffusion=True),
        cache_factory=KVCacheFactory("prefix", is_bd_model=True),
        early_stop=True,
        lazy_cache_update=True,
    )
    results.append(bench_dllm("C: BD+cache lazy", dllm, input_ids, device, tokenizer))

    # D. BD + prefix cache + inplace write (avoid slice_scatter allocation)
    dllm = BlockDiffusionLLM(
        model, make_decoder(),
        BlockIteratorFactory(use_block_diffusion=True),
        cache_factory=KVCacheFactory("prefix", is_bd_model=True),
        early_stop=True,
        inplace_cache_update=True,
    )
    results.append(bench_dllm("D: BD+cache inplace", dllm, input_ids, device, tokenizer))

    # E. BD + prefix cache + lazy + inplace (combined)
    dllm = BlockDiffusionLLM(
        model, make_decoder(),
        BlockIteratorFactory(use_block_diffusion=True),
        cache_factory=KVCacheFactory("prefix", is_bd_model=True),
        early_stop=True,
        lazy_cache_update=True,
        inplace_cache_update=True,
    )
    results.append(bench_dllm("E: BD+cache lazy+inplace", dllm, input_ids, device, tokenizer))

    return results


def print_summary(label, results, prompt_len):
    """Print sorted summary table."""
    valid = [r for r in results if r is not None]
    valid.sort(key=lambda r: r["tok/s"], reverse=True)

    print(f"\n{'='*95}")
    print(f"  {label}  (prompt={prompt_len}, gen={GEN_LENGTH}, block={BLOCK_LENGTH}, threshold={THRESHOLD})")
    print(f"{'='*95}")
    print(f"{'Path':<30} {'tok/s':<10} {'fwd/s':<10} {'tokens':<8} {'fwd':<8} {'time(s)':<10}")
    print("-" * 95)
    for i, r in enumerate(valid):
        marker = " <-- BEST" if i == 0 else ""
        print(
            f"{r['name']:<30} "
            f"{r['tok/s']:<10.1f} "
            f"{r['fwd/s']:<10.1f} "
            f"{r['tokens']:<8d} "
            f"{r['forwards']:<8d} "
            f"{r['time']:<10.3f}"
            f"{marker}"
        )
    return valid


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    from dinfer import ThresholdParallelDecoder
    from dinfer.model import LLaDA2MoeModelLM

    port = find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print(f"Loading model from {MODEL_PATH} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    parallel_config = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        model = LLaDA2MoeModelLM(config=model_config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        # Model kernel warmup
        warmup_x = torch.arange(50 + GEN_LENGTH, dtype=torch.long, device=device).unsqueeze(0)
        with torch.inference_mode():
            _ = model(warmup_x, use_cache=False)

        def make_decoder():
            return ThresholdParallelDecoder(temperature=0.0, threshold=THRESHOLD, mask_id=MASK_ID, eos_id=EOS_ID)

        # Prepare inputs
        def tokenize(prompt_text):
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    add_generation_prompt=True, tokenize=False,
                )
            return tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)

        short_ids = tokenize(SHORT_PROMPT)
        long_ids = tokenize(LONG_PROMPT)
        print(f"Short prompt: {short_ids.shape[1]} tokens", flush=True)
        print(f"Long prompt: {long_ids.shape[1]} tokens", flush=True)

        all_output = {}

        # Short prompt suite
        short_results = run_suite("SHORT PROMPT", short_ids, device, model, make_decoder, tokenizer)
        short_valid = print_summary("SHORT PROMPT SUMMARY", short_results, short_ids.shape[1])
        verify_results(short_results, "SHORT PROMPT")
        all_output["short"] = [r for r in short_valid if r is not None]

        # Long prompt suite
        long_results = run_suite("LONG PROMPT", long_ids, device, model, make_decoder, tokenizer)
        long_valid = print_summary("LONG PROMPT SUMMARY", long_results, long_ids.shape[1])
        verify_results(long_results, "LONG PROMPT")
        all_output["long"] = [r for r in long_valid if r is not None]

    # Save results (without generated_text for compact JSON)
    save_output = {}
    for key in all_output:
        save_output[key] = []
        for r in all_output[key]:
            r_copy = {k: v for k, v in r.items() if k != "generated_text"}
            save_output[key].append(r_copy)

    output_path = RESULTS_DIR / "lazy_cache_benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(save_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to: {output_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
