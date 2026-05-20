#!/usr/bin/env python3
"""
Benchmark 4 paths at batch=1,4,8,16,32 × short/long prompt:
  A. BD-attnmask (no cache, SOTA per-step)
  B. BD+IterSmooth (no cache, high_w=0.5)
  C. BD+cache (lazy+inplace optimized)
  D. BD+IterSmooth+cache
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


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sync_cuda(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def trim_after_eos(t, eos_id):
    p = (t == eos_id).nonzero(as_tuple=True)[0]
    return t[:int(p[0].item())] if p.numel() > 0 else t


def bench(name, dllm, input_ids, bs, device):
    print(f"\n--- {name} batch={bs} ---", flush=True)
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
                runs.append({"bs": bs, "tok": ttok, "fwd": fwd, "t": wall, "tp": tp, "ps": tp/bs, "fps": fps})
                print(f"  R{r+1}: {ttok}tok {fwd}fwd {wall:.3f}s tp={tp:.0f} ps={tp/bs:.1f} fps={fps:.1f}", flush=True)
        med = median(r_["tp"] for r_ in runs)
        return min(runs, key=lambda r_: abs(r_["tp"] - med))
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM!", flush=True)
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"  FAIL: {e}", flush=True)
        import traceback; traceback.print_exc()
        torch.cuda.empty_cache()
        return None


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (ThresholdParallelDecoder, BlockDiffusionLLMAttnmask,
                        BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        IterSmoothBlockDiffusionLLMAttnmask, IterSmoothBlockDiffusionLLMCache)
    from dinfer.model import LLaDA2MoeModelLM

    port = find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print(f"Loading model ...", flush=True)
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
                text = tokenizer.apply_chat_template([{"role":"user","content":text}], add_generation_prompt=True, tokenize=False)
            return tokenizer(text, return_tensors="pt")["input_ids"].to(device)

        short_ids = tok(SHORT_PROMPT)
        long_ids = tok(LONG_PROMPT)
        print(f"Short: {short_ids.shape[1]} tok, Long: {long_ids.shape[1]} tok")

        all_out = {}
        for label, ids in [("SHORT", short_ids), ("LONG", long_ids)]:
            print(f"\n{'='*110}\n  {label} PROMPT (prompt={ids.shape[1]})\n{'='*110}")

            paths = {}
            for bs in BATCH_SIZES:
                # A: BD-attnmask (no cache)
                dllm = BlockDiffusionLLMAttnmask(model, dec(), BlockIteratorFactory(use_block_diffusion=True), early_stop=True)
                paths.setdefault("A:no-cache", []).append(bench("A:no-cache", dllm, ids, bs, device))

                # B: BD+IS (no cache, high_w)
                dllm = IterSmoothBlockDiffusionLLMAttnmask(
                    model, dec(), BlockIteratorFactory(use_block_diffusion=True), early_stop=True,
                    cont_weight=0.5, cont_weight_init=0.25, cont_weight_growth=0.03, threshold_decay=0.02)
                paths.setdefault("B:IS-nocache", []).append(bench("B:IS-nocache", dllm, ids, bs, device))

                # C: BD+cache (lazy+inplace)
                dllm = BlockDiffusionLLM(
                    model, dec(), BlockIteratorFactory(use_block_diffusion=True),
                    cache_factory=KVCacheFactory("prefix", is_bd_model=True), early_stop=True,
                    lazy_cache_update=True, inplace_cache_update=True)
                paths.setdefault("C:cache-opt", []).append(bench("C:cache-opt", dllm, ids, bs, device))

                # D: BD+IS+cache
                dllm = IterSmoothBlockDiffusionLLMCache(
                    model, dec(), BlockIteratorFactory(use_block_diffusion=True),
                    cache_factory=KVCacheFactory("prefix", is_bd_model=True), early_stop=True,
                    cont_weight=0.5, cont_weight_init=0.25, cont_weight_growth=0.03, threshold_decay=0.02)
                paths.setdefault("D:IS+cache", []).append(bench("D:IS+cache", dllm, ids, bs, device))

            # Summary
            print(f"\n{'='*120}\n  {label} — THROUGHPUT (tok/s) COMPARISON\n{'='*120}")
            header = f"{'batch':<7}"
            for pname in paths:
                header += f" {pname+' tp':<16} {pname+' fps':<14}"
            print(header)
            print("-" * 120)

            for i, bs in enumerate(BATCH_SIZES):
                line = f"{bs:<7}"
                for pname in paths:
                    r = paths[pname][i]
                    if r:
                        line += f" {r['tp']:<16.0f} {r['fps']:<14.1f}"
                    else:
                        line += f" {'OOM':<16} {'—':<14}"
                print(line)

            all_out[label.lower()] = {pn: [r for r in rs if r] for pn, rs in paths.items()}

    out_path = RESULTS_DIR / "batch_4paths_benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_out, ensure_ascii=False, indent=2) + "\n")
    print(f"\nSaved: {out_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
