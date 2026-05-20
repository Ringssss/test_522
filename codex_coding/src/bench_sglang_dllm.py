#!/usr/bin/env python3
"""
SGLang dLLM Throughput Benchmark.

Uses sgl.Engine offline mode with LowConfidence algorithm.
Measures end-to-end throughput for comparison with dInfer benchmarks.

Launch:
    conda run -n dllm_base python codex_coding/src/bench_sglang_dllm.py \
        --batch-size 32 --gen-length 256 --num-runs 2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# --- Compat shim: stub deep_gemm.utils.layout if missing ---
import types
import sys
try:
    import deep_gemm
    if not hasattr(deep_gemm, 'utils') or not hasattr(getattr(deep_gemm, 'utils', None), 'layout'):
        _u = types.ModuleType('deep_gemm.utils')
        _l = types.ModuleType('deep_gemm.utils.layout')
        _l.get_mn_major_tma_aligned_tensor = None
        sys.modules['deep_gemm.utils'] = _u
        sys.modules['deep_gemm.utils.layout'] = _l
        deep_gemm.utils = _u
        deep_gemm.utils.layout = _l
except ImportError:
    pass

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"


def main():
    parser = argparse.ArgumentParser(description="SGLang dLLM throughput benchmark")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--num-runs", type=int, default=2)
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--prompt-source", choices=["gsm8k", "humaneval", "mgsm", "mt_bench"],
                        default="gsm8k")
    parser.add_argument("--pad-to", type=int, default=128)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--max-running-requests", type=int, default=0,
                        help="Max concurrent requests (0=batch-size)")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--attention-backend", type=str, default="flashinfer")
    args = parser.parse_args()

    max_rr = args.max_running_requests if args.max_running_requests > 0 else args.batch_size

    # --- Load prompts (same as dInfer benchmarks) ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)

    _ds_map = {
        "gsm8k": ("data/gsm8k.jsonl", "question"),
        "humaneval": ("data/humaneval.jsonl", "prompt"),
        "mgsm": ("data/mgsm.jsonl", "question"),
        "mt_bench": ("data/mt_bench.jsonl", None),
    }
    _path, _key = _ds_map[args.prompt_source]
    _lines = (REPO_ROOT / _path).read_text().strip().split("\n")
    raw_prompts = []
    for _l in _lines:
        _obj = json.loads(_l)
        if _key:
            raw_prompts.append(_obj[_key])
        else:
            raw_prompts.append(_obj["turns"][0])

    # Apply chat template + pad/truncate to fixed token length
    prompt_texts = []
    for i in range(args.batch_size):
        text = raw_prompts[i % len(raw_prompts)]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=False,
            )
        # Tokenize to check/enforce length
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        if args.pad_to > 0:
            if len(ids) > args.pad_to:
                ids = ids[-args.pad_to:]
            # Decode back to text (so SGLang receives fixed-length prompts)
            text = tokenizer.decode(ids, skip_special_tokens=False)
        prompt_texts.append(text)

    print("=" * 70)
    print("SGLang dLLM Throughput Benchmark")
    print(f"  batch={args.batch_size} gen={args.gen_length} threshold={args.threshold}")
    print(f"  tp={args.tp_size} max_running_requests={max_rr}")
    print(f"  prompt_source={args.prompt_source} pad_to={args.pad_to}")
    print(f"  attention_backend={args.attention_backend} cuda_graph={'OFF' if args.disable_cuda_graph else 'ON'}")
    print(f"  num_runs={args.num_runs}")
    print("=" * 70)

    # --- Write threshold config ---
    config_path = REPO_ROOT / "codex_coding" / "src" / "_sglang_bench_config.yaml"
    import yaml
    with open(config_path, "w") as f:
        yaml.dump({"threshold": args.threshold, "block_size": 32}, f)

    # --- Create Engine ---
    import sglang as sgl

    print("  Launching SGLang Engine...")
    # Set up shared file for fwd counter (subprocess reads this env var)
    fwd_stats_path = str(REPO_ROOT / "codex_coding" / "results" / "_sglang_fwd_stats.txt")
    os.environ["SGLANG_FWD_STATS_PATH"] = fwd_stats_path

    engine_kwargs = dict(
        model_path=MODEL_PATH,
        tp_size=args.tp_size,
        dllm_algorithm="LowConfidence",
        dllm_algorithm_config=str(config_path),
        max_running_requests=max_rr,
        trust_remote_code=True,
        log_level="warning",
        mem_fraction_static=0.85,
        attention_backend=args.attention_backend,
    )
    if args.disable_cuda_graph:
        engine_kwargs["disable_cuda_graph"] = True
    engine = sgl.Engine(**engine_kwargs)
    print("  Engine ready.")

    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": args.gen_length,
    }

    def read_fwd_count():
        try:
            with open(fwd_stats_path, "r") as f:
                return int(f.read().strip())
        except Exception:
            return -1

    def reset_fwd_count():
        with open(fwd_stats_path, "w") as f:
            f.write("0")

    # --- Warmup ---
    print("  Warmup...")
    reset_fwd_count()
    _ = engine.generate(prompt_texts[:min(4, len(prompt_texts))], sampling_params)
    warmup_fwd = read_fwd_count()
    print(f"  Warmup done. (fwd={warmup_fwd})")

    # --- Timed runs ---
    times = []
    token_counts = []
    fwd_counts = []
    prev_fwd = read_fwd_count()
    for r_idx in range(args.num_runs):
        t0 = time.time()
        outputs = engine.generate(prompt_texts, sampling_params)
        elapsed = time.time() - t0
        times.append(elapsed)
        cur_fwd = read_fwd_count()
        nf = cur_fwd - prev_fwd
        prev_fwd = cur_fwd
        fwd_counts.append(nf)

        # Count generated tokens
        total_gen_tokens = 0
        for out in outputs:
            meta = out.get("meta_info", {})
            n_tok = meta.get("completion_tokens", 0)
            if n_tok == 0:
                gen_text = out.get("text", "")
                n_tok = len(tokenizer.encode(gen_text))
            total_gen_tokens += n_tok
        token_counts.append(total_gen_tokens)

        ms_fwd = elapsed / nf * 1000 if nf > 0 else -1
        tok_per_s = total_gen_tokens / elapsed
        print(f"    [SGLang] run {r_idx}: {elapsed:.3f}s, "
              f"{nf} fwd, {ms_fwd:.2f} ms/fwd, "
              f"{total_gen_tokens} tokens, {tok_per_s:.1f} tok/s")

        if not args.no_quality and r_idx == 0:
            for j in range(min(3, len(outputs))):
                gen_text = outputs[j].get("text", "")
                print(f"      === sample {j} ===")
                print(f"      {gen_text[:500]}")
                print()

    # --- Summary ---
    avg_time = sum(times) / len(times)
    avg_tokens = sum(token_counts) / len(token_counts)
    avg_tok_s = avg_tokens / avg_time

    avg_fwd = sum(fwd_counts) / len(fwd_counts) if fwd_counts else 0
    avg_ms_fwd = avg_time / avg_fwd * 1000 if avg_fwd > 0 else -1

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  SGLang LowConfidence: {avg_ms_fwd:.2f} ms/fwd, {avg_tok_s:.1f} tok/s, "
          f"{avg_time:.3f}s avg ({args.batch_size} seqs, {avg_fwd:.0f} fwd)")
    for i, (t, tc, nf) in enumerate(zip(times, token_counts, fwd_counts)):
        ms = t / nf * 1000 if nf > 0 else -1
        print(f"    run {i}: {t:.3f}s, {nf} fwd, {ms:.2f} ms/fwd, {tc} tokens, {tc/t:.1f} tok/s")
    print("=" * 70)

    results = {
        "sglang": {
            "times": times,
            "token_counts": token_counts,
            "fwd_counts": fwd_counts,
            "avg_tok_per_s": avg_tok_s,
            "avg_ms_fwd": avg_ms_fwd,
            "avg_time": avg_time,
        },
    }
    payload = {
        "batch_size": args.batch_size,
        "gen_length": args.gen_length,
        "threshold": args.threshold,
        "tp_size": args.tp_size,
        "max_running_requests": max_rr,
        "prompt_source": args.prompt_source,
        "pad_to": args.pad_to,
        "num_runs": args.num_runs,
        "results": results,
    }
    out_path = REPO_ROOT / "codex_coding" / "results" / "bench_sglang_dllm.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved to {out_path}")

    paper_dir = REPO_ROOT / "codex_coding" / "results" / "bench_paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    paper_path = paper_dir / f"sglang_{args.prompt_source}_b{args.batch_size}.json"
    with open(paper_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved to {paper_path}")

    engine.shutdown()


if __name__ == "__main__":
    main()
