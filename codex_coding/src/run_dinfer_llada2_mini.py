#!/usr/bin/env python3
"""
Run a formal dInfer experiment on LLaDA2.0-mini with tp=1.

This follows the official dInfer benchmark path based on:
- dinfer.model.LLaDA2MoeModelLM
- vllm distributed init
- ThresholdParallelDecoder
- BlockWiseDiffusionLLM or BlockDiffusionLLM
"""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
from contextlib import closing
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer


REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
SGLANG_PYTHON = REPO_ROOT / "lib_cite" / "sglang" / "python"
if str(SGLANG_PYTHON) not in sys.path:
    sys.path.insert(0, str(SGLANG_PYTHON))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal dInfer run on LLaDA2.0-mini")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/wuhang/models/LLaDA2.0-mini",
        help="Local model directory.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        help="Prompt text.",
    )
    parser.add_argument(
        "--gen-length",
        type=int,
        default=128,
        help="Generated length budget.",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=32,
        help="Diffusion block length.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Threshold for parallel decoding.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="prefix",
        choices=["prefix", "dual", "none"],
        help="KV cache mode.",
    )
    parser.add_argument(
        "--use-bd",
        action="store_true",
        default=True,
        help="Use block diffusion path.",
    )
    parser.add_argument(
        "--disable-bd",
        dest="use_bd",
        action="store_false",
        help="Disable block diffusion path.",
    )
    parser.add_argument(
        "--enable-compile",
        action="store_true",
        help="Enable torch.compile on model.forward.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup generation runs before timing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Execution device.",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=str(REPO_ROOT / "codex_coding" / "results" / "dinfer_llada2_mini_metrics.json"),
        help="Where to save JSON metrics.",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, get_current_vllm_config, set_current_vllm_config

    from dinfer import (
        BlockDiffusionLLM,
        BlockIteratorFactory,
        BlockWiseDiffusionLLM,
        KVCacheFactory,
        ThresholdParallelDecoder,
    )
    from dinfer.model import LLaDA2MoeModelLM

    port = find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print(f"Loading tokenizer from {args.model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    prompt_text = args.prompt
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)

    print(f"Loading config from {args.model_path} ...", flush=True)
    model_config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Building model ...", flush=True)
    parallel_config = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=parallel_config)):
        vllm_config = get_current_vllm_config()
        print(f"EP Enabled: {vllm_config.parallel_config.enable_expert_parallel}", flush=True)

        model = LLaDA2MoeModelLM(config=model_config).eval()

        print("Loading weights ...", flush=True)
        load_start = time.perf_counter()
        model.load_weights(args.model_path, torch_dtype=torch.bfloat16, device=device)
        load_time = time.perf_counter() - load_start

        model = model.to(device)

        warmup_x = torch.arange(50 + args.gen_length, dtype=torch.long, device=device).unsqueeze(0)
        with torch.inference_mode():
            _ = model(warmup_x, use_cache=False)
            _ = model(warmup_x, use_cache=True)

        if args.enable_compile:
            print("Enabling torch.compile ...", flush=True)
            model.forward = torch.compile(
                model.forward,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=True,
            )

        decoder = ThresholdParallelDecoder(
            temperature=0.0,
            threshold=args.threshold,
            mask_id=156895,
            eos_id=156892,
        )

        cache_factory = None
        if args.cache in ("prefix", "dual"):
            cache_factory = KVCacheFactory(args.cache, is_bd_model=args.use_bd)

        if args.use_bd:
            dllm = BlockDiffusionLLM(
                model,
                decoder,
                BlockIteratorFactory(start_block_align=True, use_block_diffusion=True),
                cache_factory=cache_factory,
                early_stop=True,
            )
        else:
            dllm = BlockWiseDiffusionLLM(
                model,
                decoder,
                BlockIteratorFactory(start_block_align=True),
                cache_factory=cache_factory,
                early_stop=True,
            )

        print("Warmup ...", flush=True)
        for _ in range(args.warmup_runs):
            _ = dllm.generate(input_ids, gen_length=args.gen_length, block_length=args.block_length)

        prev_forwards = dllm.num_forwards
        sync_cuda(device)
        gen_start = time.perf_counter()
        output_ids = dllm.generate(input_ids, gen_length=args.gen_length, block_length=args.block_length)
        sync_cuda(device)
        gen_time = time.perf_counter() - gen_start
        used_forwards = dllm.num_forwards - prev_forwards

    output_ids = output_ids[0]
    output_ids = trim_after_eos(output_ids, 156892)
    generated_ids = output_ids[input_ids.shape[1] :]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    metrics = {
        "model_path": args.model_path,
        "device": str(device),
        "tp_size": 1,
        "use_bd": args.use_bd,
        "cache": args.cache,
        "threshold": args.threshold,
        "enable_compile": args.enable_compile,
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": int(generated_ids.numel()),
        "num_forwards": int(used_forwards),
        "load_time_sec": load_time,
        "generation_time_sec": gen_time,
        "tokens_per_sec": float(generated_ids.numel() / gen_time) if gen_time > 0 else 0.0,
        "forwards_per_sec": float(used_forwards / gen_time) if gen_time > 0 else 0.0,
        "prompt": args.prompt,
        "generated_text": generated_text,
    }

    print("\n=== Generation ===")
    print(generated_text)
    print("\n=== Metrics ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved metrics to: {metrics_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
