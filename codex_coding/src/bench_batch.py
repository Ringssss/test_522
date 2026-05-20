#!/usr/bin/env python3
"""
Benchmark BD-attnmask throughput with varying batch sizes.

Tests batch=1,2,4,8 × short/long prompt on the fastest path (BD-attnmask, no cache).
Measures total throughput (tokens/s across all sequences) and per-sequence latency.
Verifies all sequences produce identical output (deterministic with temperature=0).
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

BATCH_SIZES = [1, 2, 4, 8]
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


def bench_batch(name, dllm, input_ids, batch_size, device, tokenizer):
    """Benchmark with a specific batch size. Returns result dict or None on failure."""
    print(f"\n--- {name} (batch={batch_size}) ---", flush=True)

    # Create batched input: same prompt repeated
    batched_input = input_ids.repeat(batch_size, 1)
    prompt_len = input_ids.shape[1]

    try:
        # Warmup
        print(f"  Warmup ({WARMUP_RUNS})...", flush=True)
        with torch.inference_mode():
            for _ in range(WARMUP_RUNS):
                dllm.generate(batched_input, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)

        runs = []
        gen_text_seq0 = None
        all_seqs_identical = True
        with torch.inference_mode():
            for r in range(MEASURED_RUNS):
                prev_fwd = dllm.num_forwards
                sync_cuda(device)
                t0 = time.perf_counter()
                out = dllm.generate(batched_input, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
                sync_cuda(device)
                wall = time.perf_counter() - t0
                fwd = dllm.num_forwards - prev_fwd

                # Count generated tokens per sequence
                total_gen_tokens = 0
                seq_texts = []
                for b in range(batch_size):
                    gen_ids = trim_after_eos(out[b][prompt_len:], EOS_ID)
                    total_gen_tokens += int(gen_ids.numel())
                    seq_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

                throughput = total_gen_tokens / wall if wall > 0 else 0
                fps = fwd / wall if wall > 0 else 0
                per_seq_tps = throughput / batch_size

                runs.append({
                    "batch_size": batch_size,
                    "total_tokens": total_gen_tokens,
                    "forwards": fwd,
                    "time": wall,
                    "throughput": throughput,
                    "per_seq_tok/s": per_seq_tps,
                    "fwd/s": fps,
                })
                print(
                    f"  Run {r+1}: {total_gen_tokens} total tok, {fwd} fwd, {wall:.3f}s, "
                    f"throughput={throughput:.1f} tok/s, per_seq={per_seq_tps:.1f} tok/s, "
                    f"fwd/s={fps:.1f}",
                    flush=True,
                )

                # Check all sequences produce identical text
                if r == MEASURED_RUNS - 1:
                    gen_text_seq0 = seq_texts[0]
                    for b in range(1, batch_size):
                        if seq_texts[b] != seq_texts[0]:
                            all_seqs_identical = False
                            print(f"  WARNING: seq[{b}] differs from seq[0]!", flush=True)

        med_tp = median(r_["throughput"] for r_ in runs)
        best = min(runs, key=lambda r_: abs(r_["throughput"] - med_tp))
        return {
            "name": name,
            **best,
            "all_runs": runs,
            "generated_text": gen_text_seq0,
            "all_seqs_identical": all_seqs_identical,
        }
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM at batch={batch_size}!", flush=True)
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        import traceback; traceback.print_exc()
        return None


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    from dinfer import ThresholdParallelDecoder, BlockDiffusionLLMAttnmask, BlockIteratorFactory
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

        for label, input_ids in [("SHORT", short_ids), ("LONG", long_ids)]:
            prompt_len = input_ids.shape[1]
            print(f"\n{'='*95}")
            print(f"  {label} PROMPT (prompt={prompt_len}, gen={GEN_LENGTH}, block={BLOCK_LENGTH}, threshold={THRESHOLD})")
            print(f"{'='*95}")

            results = []
            for bs in BATCH_SIZES:
                dllm = BlockDiffusionLLMAttnmask(
                    model, make_decoder(),
                    BlockIteratorFactory(use_block_diffusion=True),
                    early_stop=True,
                )
                r = bench_batch(f"BD-attnmask", dllm, input_ids, bs, device, tokenizer)
                results.append(r)

            # Summary table
            valid = [r for r in results if r is not None]
            print(f"\n{'='*100}")
            print(f"  {label} PROMPT SUMMARY")
            print(f"{'='*100}")
            print(f"{'batch':<8} {'throughput':<14} {'per_seq tok/s':<16} {'fwd/s':<10} {'fwd':<8} {'time(s)':<10} {'seqs_identical'}")
            print("-" * 100)

            baseline_throughput = None
            for r in valid:
                if baseline_throughput is None:
                    baseline_throughput = r["throughput"]
                    scaling = "—"
                else:
                    scaling = f"{r['throughput']/baseline_throughput:.2f}x"
                print(
                    f"{r['batch_size']:<8d} "
                    f"{r['throughput']:<14.1f} "
                    f"{r['per_seq_tok/s']:<16.1f} "
                    f"{r['fwd/s']:<10.1f} "
                    f"{r['forwards']:<8d} "
                    f"{r['time']:<10.3f} "
                    f"{r['all_seqs_identical']}  {scaling}"
                )

            # Verify batch>1 produces same text as batch=1
            if len(valid) >= 2:
                base_text = valid[0]["generated_text"]
                print(f"\n  Output verification:")
                for r in valid[1:]:
                    match = r["generated_text"] == base_text
                    print(f"    batch={r['batch_size']} vs batch=1: {'PASS (identical)' if match else 'DIFFERENT'}")

            all_output[label.lower()] = [{k: v for k, v in r.items() if k not in ("generated_text", "all_runs")} for r in valid]

    # Save results
    output_path = RESULTS_DIR / "batch_benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to: {output_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
