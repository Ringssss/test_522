#!/usr/bin/env python3
"""
Benchmark throughput scaling: BD-attnmask (no cache) vs BD+cache (inplace+lazy)
at batch=1,4,8,16,32, short/long prompt.

Goal: find where each path hits its bottleneck and whether cache path
eventually wins at large batch due to lower per-forward compute.
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

SHORT_PROMPT = (
    "Lily can run 12 kilometers per hour for 4 hours. "
    "After that, she can run 6 kilometers per hour. "
    "How many kilometers can she run in 8 hours?"
)

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

BATCH_SIZES = [1, 4, 8, 16, 32]
WARMUP_RUNS = 1
MEASURED_RUNS = 2


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


def bench_one(name, dllm, input_ids, batch_size, device):
    """Benchmark a single config. Returns result dict or None."""
    print(f"\n--- {name} batch={batch_size} ---", flush=True)
    batched_input = input_ids.repeat(batch_size, 1)
    prompt_len = input_ids.shape[1]

    try:
        print(f"  Warmup...", flush=True)
        with torch.inference_mode():
            for _ in range(WARMUP_RUNS):
                dllm.generate(batched_input, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)

        runs = []
        with torch.inference_mode():
            for r in range(MEASURED_RUNS):
                prev_fwd = dllm.num_forwards
                sync_cuda(device)
                t0 = time.perf_counter()
                out = dllm.generate(batched_input, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
                sync_cuda(device)
                wall = time.perf_counter() - t0
                fwd = dllm.num_forwards - prev_fwd

                total_tok = 0
                for b in range(batch_size):
                    gen_ids = trim_after_eos(out[b][prompt_len:], EOS_ID)
                    total_tok += int(gen_ids.numel())

                tp = total_tok / wall if wall > 0 else 0
                fps = fwd / wall if wall > 0 else 0
                runs.append({"batch": batch_size, "total_tok": total_tok, "fwd": fwd,
                             "time": wall, "throughput": tp, "per_seq": tp / batch_size, "fwd/s": fps})
                print(f"  Run {r+1}: {total_tok} tok, {fwd} fwd, {wall:.3f}s, "
                      f"tp={tp:.1f}, per_seq={tp/batch_size:.1f}, fwd/s={fps:.1f}", flush=True)

        med = median(r_["throughput"] for r_ in runs)
        best = min(runs, key=lambda r_: abs(r_["throughput"] - med))
        return {"name": name, **best}

    except torch.cuda.OutOfMemoryError:
        print(f"  OOM!", flush=True)
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()
        torch.cuda.empty_cache()
        return None


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    from dinfer import (ThresholdParallelDecoder, BlockDiffusionLLMAttnmask,
                        BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory)
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

        warmup_x = torch.arange(50 + GEN_LENGTH, dtype=torch.long, device=device).unsqueeze(0)
        with torch.inference_mode():
            _ = model(warmup_x, use_cache=False)

        def make_decoder():
            return ThresholdParallelDecoder(temperature=0.0, threshold=THRESHOLD, mask_id=MASK_ID, eos_id=EOS_ID)

        def tokenize(prompt_text):
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    add_generation_prompt=True, tokenize=False,
                )
            return tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)

        short_ids = tokenize(SHORT_PROMPT)
        long_ids = tokenize(LONG_PROMPT)
        print(f"Short prompt: {short_ids.shape[1]} tokens")
        print(f"Long prompt: {long_ids.shape[1]} tokens")

        all_output = {}

        for label, input_ids in [("SHORT", short_ids), ("LONG", long_ids)]:
            prompt_len = input_ids.shape[1]
            print(f"\n{'='*100}")
            print(f"  {label} PROMPT (prompt={prompt_len})")
            print(f"{'='*100}")

            nocache_results = []
            cache_results = []

            for bs in BATCH_SIZES:
                # Path A: BD-attnmask (no cache, SOTA per-step)
                dllm = BlockDiffusionLLMAttnmask(
                    model, make_decoder(),
                    BlockIteratorFactory(use_block_diffusion=True),
                    early_stop=True,
                )
                nocache_results.append(bench_one("no-cache", dllm, input_ids, bs, device))

                # Path B: BD+cache inplace+lazy (best cache config)
                dllm = BlockDiffusionLLM(
                    model, make_decoder(),
                    BlockIteratorFactory(use_block_diffusion=True),
                    cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                    early_stop=True,
                    lazy_cache_update=True,
                    inplace_cache_update=True,
                )
                cache_results.append(bench_one("cache-opt", dllm, input_ids, bs, device))

            # Summary
            print(f"\n{'='*110}")
            print(f"  {label} PROMPT — THROUGHPUT SCALING COMPARISON")
            print(f"{'='*110}")
            print(f"{'batch':<7} {'no-cache tp':<14} {'no-cache fwd/s':<16} {'cache-opt tp':<14} {'cache-opt fwd/s':<16} {'cache/nocache'}")
            print("-" * 110)

            for nc, ca in zip(nocache_results, cache_results):
                nc_tp = f"{nc['throughput']:.1f}" if nc else "OOM"
                nc_fps = f"{nc['fwd/s']:.1f}" if nc else "—"
                ca_tp = f"{ca['throughput']:.1f}" if ca else "OOM"
                ca_fps = f"{ca['fwd/s']:.1f}" if ca else "—"
                ratio = ""
                if nc and ca:
                    ratio = f"{ca['throughput']/nc['throughput']:.2f}x"
                bs = nc["batch"] if nc else (ca["batch"] if ca else "?")
                print(f"{bs:<7} {nc_tp:<14} {nc_fps:<16} {ca_tp:<14} {ca_fps:<16} {ratio}")

            all_output[label.lower()] = {
                "no_cache": [r for r in nocache_results if r],
                "cache_opt": [r for r in cache_results if r],
            }

    output_path = RESULTS_DIR / "batch_scaling_benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to: {output_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
