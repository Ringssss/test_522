#!/usr/bin/env python3
"""
Kernel-level profiling inside model.forward() for cache vs no-cache paths.

Profiles one full generate() call with torch.profiler, then categorizes
CUDA kernels into: Attention, MoE, Linear/GEMM, Norm, RoPE, Decoder, Memory, Other.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile
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


# ============================================================
# Kernel categorization
# ============================================================
KERNEL_CATEGORIES = [
    ("MoE_fused",    [r"fused_moe", r"fused_topk", r"moe_align"]),
    ("Attention",    [r"flash_attn", r"efficient_attention", r"sdpa", r"scaled_dot_product",
                      r"fmha", r"_attention"]),
    ("GEMM",         [r"gemm", r"gemv", r"cublas", r"cutlass", r"matmul", r"addmm", r"mm_"]),
    ("Softmax",      [r"softmax", r"log_softmax"]),
    ("Norm",         [r"layer_norm", r"rmsnorm", r"rms_norm", r"native_layer_norm"]),
    ("RoPE",         [r"rotary", r"rope"]),
    ("Embedding",    [r"embedding", r"index_select"]),
    ("Elementwise",  [r"add_", r"mul_", r"where", r"copy_", r"fill_", r"cat",
                      r"slice", r"scatter", r"contiguous", r"clone", r"to\("]),
    ("Reduce",       [r"sum", r"max", r"argmax", r"topk", r"sort", r"cumsum"]),
    ("Memory",       [r"memcpy", r"memset", r"pad", r"empty", r"zeros", r"ones",
                      r"expand", r"repeat", r"view", r"reshape"]),
]


def categorize_kernel(name):
    name_lower = name.lower()
    for cat, patterns in KERNEL_CATEGORIES:
        for pat in patterns:
            if re.search(pat, name_lower):
                return cat
    return "Other"


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def trim_after_eos(t, eos_id):
    p = (t == eos_id).nonzero(as_tuple=True)[0]
    return t[: int(p[0].item())] if p.numel() > 0 else t


def profile_scenario(label, dllm_factory, input_ids, batch_size, device, model):
    """Profile one generate() call and return kernel breakdown."""
    batched = input_ids.repeat(batch_size, 1)
    plen = input_ids.shape[1]

    print(f"\n{'='*90}")
    print(f"  {label}  (batch={batch_size}, prompt={plen})")
    print(f"{'='*90}")

    # Warmup
    print("  Warmup ...", flush=True)
    dllm = dllm_factory()
    with torch.inference_mode():
        dllm.generate(batched, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
    torch.cuda.empty_cache()

    # Profiled run
    print("  Profiling ...", flush=True)
    dllm = dllm_factory()

    with torch.inference_mode():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = dllm.generate(batched, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize(device)
            wall = time.perf_counter() - t0

    fwd_count = dllm.num_forwards
    ttok = sum(
        int(trim_after_eos(out[b][plen:], EOS_ID).numel())
        for b in range(batch_size)
    )
    print(f"  {ttok} tok, {fwd_count} fwd, {wall:.3f}s, {ttok/wall:.0f} tok/s, {fwd_count/wall:.1f} fwd/s")

    # === Raw kernel table (top 50) ===
    print(f"\n  --- Top 50 CUDA kernels by self_cuda_time ---")
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50)
    print(table)

    # === Categorized breakdown ===
    cat_totals = defaultdict(lambda: {"cuda_us": 0, "cpu_us": 0, "count": 0, "kernels": []})
    total_cuda_us = 0

    def get_self_cuda(evt):
        for attr in ("self_cuda_time_total", "self_device_time_total", "cuda_time_total"):
            if hasattr(evt, attr):
                return getattr(evt, attr)
        return 0

    for evt in prof.key_averages():
        cuda_us = get_self_cuda(evt)
        if cuda_us <= 0:
            continue
        cat = categorize_kernel(evt.key)
        cat_totals[cat]["cuda_us"] += cuda_us
        cat_totals[cat]["cpu_us"] += evt.self_cpu_time_total
        cat_totals[cat]["count"] += evt.count
        cat_totals[cat]["kernels"].append({
            "name": evt.key,
            "self_cuda_us": cuda_us,
            "count": evt.count,
        })
        total_cuda_us += cuda_us

    # Sort categories by cuda time
    sorted_cats = sorted(cat_totals.items(), key=lambda x: -x[1]["cuda_us"])

    print(f"\n  --- Categorized CUDA time breakdown ---")
    print(f"  Total CUDA time: {total_cuda_us/1000:.1f} ms")
    print(f"  Wall time: {wall*1000:.1f} ms")
    print(f"\n  {'Category':<16s} {'CUDA(ms)':>10s} {'%CUDA':>7s} {'%Wall':>7s} {'Calls':>8s} {'Top kernel':<50s}")
    print(f"  {'-'*16} {'-'*10} {'-'*7} {'-'*7} {'-'*8} {'-'*50}")

    result_cats = {}
    for cat, data in sorted_cats:
        cuda_ms = data["cuda_us"] / 1000
        pct_cuda = data["cuda_us"] / total_cuda_us * 100 if total_cuda_us > 0 else 0
        pct_wall = cuda_ms / (wall * 1000) * 100
        # Find top kernel in this category
        top_kernel = max(data["kernels"], key=lambda k: k["self_cuda_us"])
        top_name = top_kernel["name"][:50]
        print(f"  {cat:<16s} {cuda_ms:>10.1f} {pct_cuda:>6.1f}% {pct_wall:>6.1f}% {data['count']:>8d} {top_name}")
        result_cats[cat] = {
            "cuda_ms": cuda_ms,
            "pct_cuda": pct_cuda,
            "pct_wall": pct_wall,
            "count": data["count"],
            "top_kernels": sorted(data["kernels"], key=lambda k: -k["self_cuda_us"])[:5],
        }

    # === Top 5 kernels per category ===
    print(f"\n  --- Top kernels per category ---")
    for cat, data in sorted_cats:
        if data["cuda_us"] / total_cuda_us < 0.02:  # skip <2%
            continue
        print(f"\n  [{cat}] total={data['cuda_us']/1000:.1f}ms")
        kernels = sorted(data["kernels"], key=lambda k: -k["self_cuda_us"])
        for k in kernels[:5]:
            pct = k["self_cuda_us"] / total_cuda_us * 100
            print(f"    {k['self_cuda_us']/1000:>8.1f}ms ({pct:>5.1f}%)  x{k['count']:<5d}  {k['name'][:70]}")

    return {
        "label": label,
        "batch_size": batch_size,
        "wall_s": wall,
        "tok": ttok,
        "fwd_count": fwd_count,
        "total_cuda_ms": total_cuda_us / 1000,
        "categories": result_cats,
    }


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
        ThresholdParallelDecoder,
    )
    from dinfer.model import LLaDA2MoeModelLM

    port = find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("Loading model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    cfg = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=cfg).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        with torch.inference_mode():
            _ = model(
                torch.arange(180, dtype=torch.long, device=device).unsqueeze(0),
                use_cache=False,
            )

        def make_decoder():
            return ThresholdParallelDecoder(
                temperature=0.0, threshold=THRESHOLD, mask_id=MASK_ID, eos_id=EOS_ID
            )

        prompt_text = LONG_PROMPT
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True,
                tokenize=False,
            )
        long_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
        print(f"Long prompt: {long_ids.shape[1]} tokens\n")

        results = {}

        # S1: no-cache, batch=1
        results["S1_nocache_b1"] = profile_scenario(
            "S1: no-cache b=1",
            lambda: BlockDiffusionLLMAttnmask(
                model, make_decoder(),
                BlockIteratorFactory(use_block_diffusion=True),
                early_stop=True,
            ),
            long_ids, batch_size=1, device=device, model=model,
        )

        # S2: cache-opt, batch=1
        results["S2_cache_b1"] = profile_scenario(
            "S2: cache-opt b=1",
            lambda: BlockDiffusionLLM(
                model, make_decoder(),
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True,
                lazy_cache_update=True,
                inplace_cache_update=True,
            ),
            long_ids, batch_size=1, device=device, model=model,
        )

        # S4: cache-opt, batch=32
        results["S4_cache_b32"] = profile_scenario(
            "S4: cache-opt b=32",
            lambda: BlockDiffusionLLM(
                model, make_decoder(),
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True,
                lazy_cache_update=True,
                inplace_cache_update=True,
            ),
            long_ids, batch_size=32, device=device, model=model,
        )

    # === Cross-scenario category comparison ===
    print(f"\n{'#'*100}")
    print(f"  CROSS-SCENARIO CATEGORY COMPARISON")
    print(f"{'#'*100}")

    all_cats = set()
    for r in results.values():
        all_cats.update(r["categories"].keys())
    all_cats = sorted(all_cats)

    hdr = f"  {'Category':<16s}"
    for key in results:
        hdr += f" │ {results[key]['label'][:18]:>18s}"
    print(hdr)
    print(f"  {'-'*16}" + "".join(f" │ {'-'*18}" for _ in results))

    for cat in all_cats:
        line = f"  {cat:<16s}"
        for key in results:
            data = results[key]["categories"].get(cat, {})
            ms = data.get("cuda_ms", 0)
            pct = data.get("pct_cuda", 0)
            line += f" │ {ms:>8.1f}ms {pct:>5.1f}%"
        print(line)

    # Totals
    line = f"  {'TOTAL':<16s}"
    for key in results:
        line += f" │ {results[key]['total_cuda_ms']:>8.1f}ms {'100.0':>5s}%"
    print(line)

    line = f"  {'Wall':<16s}"
    for key in results:
        line += f" │ {results[key]['wall_s']*1000:>8.1f}ms {'':>6s}"
    print(line)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "profile_kernel_results.json"

    # Clean for JSON
    json_out = {}
    for key, val in results.items():
        r = {k: v for k, v in val.items() if k != "categories"}
        r["categories"] = {}
        for cat, data in val["categories"].items():
            cat_clean = {k: v for k, v in data.items() if k != "top_kernels"}
            cat_clean["top_kernels"] = [
                {k2: v2 for k2, v2 in tk.items()} for tk in data.get("top_kernels", [])
            ]
            r["categories"][cat] = cat_clean
        json_out[key] = r

    out_path.write_text(json.dumps(json_out, ensure_ascii=False, indent=2) + "\n")
    print(f"\nSaved: {out_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
