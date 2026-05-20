#!/usr/bin/env python3
"""
Sweep benchmark for dInfer + LLaDA2.0-mini to find the fastest config.

Fixed: use_bd=True, cache=prefix, gen_length=128, block_length=32
Sweep: threshold x enable_compile
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

# Fixed sweep parameters
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
GEN_LENGTH = 128
BLOCK_LENGTH = 32
CACHE_TYPE = "prefix"
MASK_ID = 156895
EOS_ID = 156892
DEVICE = "cuda:0"
PROMPT = (
    "Lily can run 12 kilometers per hour for 4 hours. "
    "After that, she can run 6 kilometers per hour. "
    "How many kilometers can she run in 8 hours?"
)

# Sweep grid
THRESHOLDS = [0.90, 0.95, 0.99]
# torch.compile is disabled: InductorError on torch 2.8.0 + LLaDA2MoE
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


def run_single(dllm, input_ids, device, gen_length, block_length):
    """Run one generation and return (generated_tokens, num_forwards, wall_time)."""
    prev_forwards = dllm.num_forwards
    sync_cuda(device)
    t0 = time.perf_counter()
    output_ids = dllm.generate(input_ids, gen_length=gen_length, block_length=block_length)
    sync_cuda(device)
    wall = time.perf_counter() - t0
    used_forwards = dllm.num_forwards - prev_forwards
    gen_ids = output_ids[0][input_ids.shape[1]:]
    gen_ids = trim_after_eos(gen_ids, EOS_ID)
    return int(gen_ids.numel()), int(used_forwards), wall


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    # -- Distributed init --
    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, get_current_vllm_config, set_current_vllm_config

    from dinfer import (
        BlockDiffusionLLM,
        BlockIteratorFactory,
        KVCacheFactory,
        ThresholdParallelDecoder,
    )
    from dinfer.model import LLaDA2MoeModelLM

    port = find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    # -- Load model once --
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

        # Minimal model warmup (ensure kernels are loaded)
        warmup_x = torch.arange(50 + GEN_LENGTH, dtype=torch.long, device=device).unsqueeze(0)
        with torch.inference_mode():
            _ = model(warmup_x, use_cache=False)
            _ = model(warmup_x, use_cache=True)

        all_results = []

        # -- Sweep thresholds (compile disabled due to InductorError on torch 2.8.0 + LLaDA2MoE) --
        print("\n=== Sweeping thresholds (compile=False) ===", flush=True)

        for threshold in THRESHOLDS:
            config_name = f"threshold={threshold}"
            print(f"\n--- {config_name} ---", flush=True)

            decoder = ThresholdParallelDecoder(
                temperature=0.0, threshold=threshold, mask_id=MASK_ID, eos_id=EOS_ID
            )
            cache_factory = KVCacheFactory(CACHE_TYPE, is_bd_model=True)
            dllm = BlockDiffusionLLM(
                model, decoder,
                BlockIteratorFactory(start_block_align=True, use_block_diffusion=True),
                cache_factory=cache_factory, early_stop=True,
            )

            # Warmup
            print(f"  Warmup ({WARMUP_RUNS} runs)...", flush=True)
            with torch.inference_mode():
                for _ in range(WARMUP_RUNS):
                    dllm.generate(input_ids, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)

            # Measured runs
            runs = []
            with torch.inference_mode():
                for r in range(MEASURED_RUNS):
                    gen_tokens, forwards, wall = run_single(
                        dllm, input_ids, device, GEN_LENGTH, BLOCK_LENGTH
                    )
                    tps = gen_tokens / wall if wall > 0 else 0
                    fps = forwards / wall if wall > 0 else 0
                    runs.append({
                        "generated_tokens": gen_tokens,
                        "num_forwards": forwards,
                        "generation_time_sec": wall,
                        "tokens_per_sec": tps,
                        "forwards_per_sec": fps,
                    })
                    print(f"  Run {r+1}: {gen_tokens} tok, {forwards} fwd, {wall:.3f}s, {tps:.1f} tok/s", flush=True)

            # Take median by tokens_per_sec
            median_tps = median(r["tokens_per_sec"] for r in runs)
            best_run = min(runs, key=lambda r: abs(r["tokens_per_sec"] - median_tps))

            result = {
                "enable_compile": False,
                "threshold": threshold,
                "use_bd": True,
                "cache": CACHE_TYPE,
                "gen_length": GEN_LENGTH,
                "block_length": BLOCK_LENGTH,
                **best_run,
                "all_runs": runs,
            }
            all_results.append(result)

    # -- Summary --
    all_results.sort(key=lambda r: r["tokens_per_sec"], reverse=True)

    print("\n" + "=" * 80)
    print("SWEEP RESULTS (sorted by tokens/sec)")
    print("=" * 80)
    print(f"{'threshold':<12} {'tok/s':<10} {'fwd/s':<10} {'tokens':<8} {'forwards':<10} {'time(s)':<10}")
    print("-" * 70)
    for i, r in enumerate(all_results):
        marker = " <-- BEST" if i == 0 else ""
        print(
            f"{r['threshold']:<12.2f} "
            f"{r['tokens_per_sec']:<10.1f} "
            f"{r['forwards_per_sec']:<10.1f} "
            f"{r['generated_tokens']:<8d} "
            f"{r['num_forwards']:<10d} "
            f"{r['generation_time_sec']:<10.3f}"
            f"{marker}"
        )

    best = all_results[0]
    print(f"\nBest baseline: threshold={best['threshold']}")
    print(f"  -> {best['tokens_per_sec']:.1f} tok/s, {best['forwards_per_sec']:.1f} fwd/s")
    print(f"  Config: use_bd=True, cache=prefix, block_length=32, gen_length=128, compile=False")
    print(f"  Note: torch.compile disabled due to InductorError on torch 2.8.0 + LLaDA2MoE")

    # Save results
    output_path = RESULTS_DIR / "sweep_dinfer_llada2_mini_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved results to: {output_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
