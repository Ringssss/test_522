#!/usr/bin/env python3
"""
Comprehensive benchmark: compare ALL dInfer inference paths on LLaDA2.0-mini.

Paths tested:
  A. BlockDiffusionLLM + prefix cache (current best, naive batching)
  B. BlockDiffusionLLM + prefix cache + backend='sglang'
  C. BlockWiseDiffusionLLM + prefix cache (full-sequence forward)
  D. BlockWiseDiffusionLLM + dual cache
  E. BlockWiseDiffusionLLM + no cache
  F. generate_fastdllm prefix cache (simple function path)
  G. generate_fastdllm dual cache (simple function path)
  H. BlockDiffusionLLMAttnmask (attention mask, no KV cache)
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
PROMPT = (
    "Lily can run 12 kilometers per hour for 4 hours. "
    "After that, she can run 6 kilometers per hour. "
    "How many kilometers can she run in 8 hours?"
)
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


def bench_dllm(name, dllm, input_ids, device):
    """Benchmark a dLLM object that has .generate() and .num_forwards."""
    print(f"\n--- {name} ---", flush=True)
    try:
        # Warmup
        print(f"  Warmup ({WARMUP_RUNS})...", flush=True)
        with torch.inference_mode():
            for _ in range(WARMUP_RUNS):
                dllm.generate(input_ids, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)

        runs = []
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
                print(f"  Run {r+1}: {tok} tok, {fwd} fwd, {wall:.3f}s, {tps:.1f} tok/s", flush=True)

        med_tps = median(r["tok/s"] for r in runs)
        best = min(runs, key=lambda r: abs(r["tok/s"] - med_tps))
        return {"name": name, **best, "all_runs": runs}
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        return None


def bench_fastdllm_fn(name, gen_fn, model, input_ids, device, **kwargs):
    """Benchmark a generate_fastdllm function that returns (output, nfe)."""
    print(f"\n--- {name} ---", flush=True)
    try:
        print(f"  Warmup ({WARMUP_RUNS})...", flush=True)
        with torch.inference_mode():
            for _ in range(WARMUP_RUNS):
                gen_fn(model, input_ids, steps=GEN_LENGTH, gen_length=GEN_LENGTH,
                       block_length=BLOCK_LENGTH, temperature=0., mask_id=MASK_ID,
                       eos_id=EOS_ID, threshold=THRESHOLD, early_stop=True, **kwargs)

        runs = []
        with torch.inference_mode():
            for r in range(MEASURED_RUNS):
                sync_cuda(device)
                t0 = time.perf_counter()
                out, nfe = gen_fn(model, input_ids, steps=GEN_LENGTH, gen_length=GEN_LENGTH,
                                  block_length=BLOCK_LENGTH, temperature=0., mask_id=MASK_ID,
                                  eos_id=EOS_ID, threshold=THRESHOLD, early_stop=True, **kwargs)
                sync_cuda(device)
                wall = time.perf_counter() - t0
                gen_ids = trim_after_eos(out[0][input_ids.shape[1]:], EOS_ID)
                tok = int(gen_ids.numel())
                tps = tok / wall if wall > 0 else 0
                fps = nfe / wall if wall > 0 else 0
                runs.append({"tokens": tok, "forwards": nfe, "time": wall, "tok/s": tps, "fwd/s": fps})
                print(f"  Run {r+1}: {tok} tok, {nfe} fwd, {wall:.3f}s, {tps:.1f} tok/s", flush=True)

        med_tps = median(r["tok/s"] for r in runs)
        best = min(runs, key=lambda r: abs(r["tok/s"] - med_tps))
        return {"name": name, **best, "all_runs": runs}
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        return None


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    from dinfer import (
        BlockDiffusionLLM,
        BlockDiffusionLLMAttnmask,
        BlockIteratorFactory,
        BlockWiseDiffusionLLM,
        KVCacheFactory,
        ThresholdParallelDecoder,
    )
    from dinfer.model import LLaDA2MoeModelLM
    from dinfer.decoding.generate_fastdllm import (
        generate_with_prefix_cache,
        generate_with_dual_cache,
    )

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

        # Prepare input
        prompt_text = PROMPT
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": PROMPT}],
                add_generation_prompt=True,
                tokenize=False,
            )
        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
        print(f"Prompt tokens: {input_ids.shape[1]}", flush=True)

        # Model kernel warmup
        warmup_x = torch.arange(50 + GEN_LENGTH, dtype=torch.long, device=device).unsqueeze(0)
        with torch.inference_mode():
            _ = model(warmup_x, use_cache=False)
            _ = model(warmup_x, use_cache=True)

        all_results = []

        # Helper to build decoder
        def make_decoder():
            return ThresholdParallelDecoder(temperature=0.0, threshold=THRESHOLD, mask_id=MASK_ID, eos_id=EOS_ID)

        # === A. BlockDiffusionLLM + prefix cache (current best) ===
        dllm = BlockDiffusionLLM(
            model, make_decoder(),
            BlockIteratorFactory(start_block_align=True, use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True,
        )
        all_results.append(bench_dllm("A: BD + prefix cache", dllm, input_ids, device))

        # === B. BlockDiffusionLLM + prefix cache + sglang backend ===
        dllm = BlockDiffusionLLM(
            model, make_decoder(),
            BlockIteratorFactory(start_block_align=True, use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True, backend="sglang"),
            early_stop=True, backend="sglang",
        )
        all_results.append(bench_dllm("B: BD + prefix + sglang", dllm, input_ids, device))

        # === C. BlockWiseDiffusionLLM + prefix cache ===
        dllm = BlockWiseDiffusionLLM(
            model, make_decoder(),
            BlockIteratorFactory(start_block_align=True),
            cache_factory=KVCacheFactory("prefix"),
            early_stop=True,
        )
        all_results.append(bench_dllm("C: BW + prefix cache", dllm, input_ids, device))

        # === D. BlockWiseDiffusionLLM + dual cache ===
        dllm = BlockWiseDiffusionLLM(
            model, make_decoder(),
            BlockIteratorFactory(start_block_align=True),
            cache_factory=KVCacheFactory("dual"),
            early_stop=True,
        )
        all_results.append(bench_dllm("D: BW + dual cache", dllm, input_ids, device))

        # === E. BlockWiseDiffusionLLM + no cache ===
        dllm = BlockWiseDiffusionLLM(
            model, make_decoder(),
            BlockIteratorFactory(start_block_align=True),
            cache_factory=None,
            early_stop=True,
        )
        all_results.append(bench_dllm("E: BW + no cache", dllm, input_ids, device))

        # === F. generate_fastdllm prefix cache ===
        all_results.append(bench_fastdllm_fn(
            "F: fastdllm prefix cache",
            generate_with_prefix_cache, model, input_ids, device,
        ))

        # === G. generate_fastdllm dual cache ===
        all_results.append(bench_fastdllm_fn(
            "G: fastdllm dual cache",
            generate_with_dual_cache, model, input_ids, device,
        ))

        # === H. BlockDiffusionLLMAttnmask (no KV cache, attention mask only) ===
        dllm = BlockDiffusionLLMAttnmask(
            model, make_decoder(),
            BlockIteratorFactory(use_block_diffusion=True),
            early_stop=True,
        )
        all_results.append(bench_dllm("H: BD attnmask (no cache)", dllm, input_ids, device))

    # -- Summary --
    all_results = [r for r in all_results if r is not None]
    all_results.sort(key=lambda r: r["tok/s"], reverse=True)

    print("\n" + "=" * 90)
    print("ALL PATHS COMPARISON (sorted by tok/s)")
    print("=" * 90)
    print(f"{'Path':<32} {'tok/s':<10} {'fwd/s':<10} {'tokens':<8} {'fwd':<8} {'time(s)':<10}")
    print("-" * 90)
    for i, r in enumerate(all_results):
        marker = " <-- BEST" if i == 0 else ""
        print(
            f"{r['name']:<32} "
            f"{r['tok/s']:<10.1f} "
            f"{r['fwd/s']:<10.1f} "
            f"{r['tokens']:<8d} "
            f"{r['forwards']:<8d} "
            f"{r['time']:<10.3f}"
            f"{marker}"
        )

    best = all_results[0]
    print(f"\nFastest path: {best['name']}")
    print(f"  -> {best['tok/s']:.1f} tok/s, {best['fwd/s']:.1f} fwd/s")

    output_path = RESULTS_DIR / "all_paths_benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to: {output_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
