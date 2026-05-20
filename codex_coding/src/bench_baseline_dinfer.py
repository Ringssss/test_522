#!/usr/bin/env python3
"""
Baseline dInfer Throughput Benchmark — unmodified dInfer, TP=8 EP=8.

Structure aligned with bench_tv6_throughput.py for fair A/B comparison.
Uses baseline_dInfer (lib_cite/baseline_dInfer) without any modifications.

Launch:
    cd /home/wuhang/wuhang/dllm_wh && \
    conda run -n dllm_base --no-banner \
    torchrun --nproc_per_node=8 codex_coding/src/bench_baseline_dinfer.py \
        --batch-size 512 --gen-length 256 --num-runs 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

# --- Compatibility shim: baseline dInfer uses is_torch_fx_available which
#     was removed in transformers 5.x. Inject it before importing dinfer. ---
import transformers.utils.import_utils as _tui
if not hasattr(_tui, 'is_torch_fx_available'):
    _tui.is_torch_fx_available = lambda: hasattr(torch, "fx")

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
if 'default' not in ROPE_INIT_FUNCTIONS:
    def _default_rope(config, device=None):
        base = getattr(config, 'rope_theta', 10000.0)
        prf = getattr(config, 'partial_rotary_factor', 1.0)
        hd = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
        dim = int(hd * prf)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, 1.0
    ROPE_INIT_FUNCTIONS['default'] = _default_rope

# --- Point sys.path to BASELINE dInfer (must come before any dinfer import) ---
REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
BASELINE_DINFER = REPO_ROOT / "lib_cite" / "baseline_dInfer" / "python"
sys.path.insert(0, str(BASELINE_DINFER))
sys.path.insert(1, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID = 156895
EOS_ID = 156892
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
BLOCK_LENGTH = 32


def main():
    parser = argparse.ArgumentParser(description="Baseline dInfer throughput benchmark")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--num-runs", type=int, default=2)
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--prompt-source", choices=["heteval", "gsm8k", "humaneval", "mgsm", "mt_bench"],
                        default="heteval")
    parser.add_argument("--pad-to", type=int, default=0,
                        help="Pad/truncate all prompts to this fixed token length (0=auto)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Use BlockDiffusionLLMAttnmask (no KV cache, batch=1 loop)")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    tp_size = world_size  # LLaDA2-mini has kv_heads=4, max TP=4
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # --- vllm distributed init (same pattern as bench_tv6_throughput.py) ---
    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    pcfg_init = ParallelConfig(
        tensor_parallel_size=1, data_parallel_size=1,
        enable_expert_parallel=True,
    )
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(
            world_size, rank, "env://", local_rank, "nccl")

    pcfg = ParallelConfig(
        tensor_parallel_size=tp_size, data_parallel_size=1,
        enable_expert_parallel=True,
    )
    vllm_cfg = VllmConfig(parallel_config=pcfg)

    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, backend="nccl")

        from dinfer import BlockDiffusionLLM, BlockDiffusionLLMAttnmask, BlockIteratorFactory, KVCacheFactory
        from dinfer import ThresholdParallelDecoder
        from dinfer.model import LLaDA2MoeModelLM
        from transformers import AutoConfig, AutoTokenizer
        from test_heteval512 import PROMPTS

        if rank == 0:
            import dinfer
            print("=" * 70)
            print("Baseline dInfer Throughput Benchmark")
            print(f"  batch={args.batch_size} gen={args.gen_length} block={BLOCK_LENGTH}")
            print(f"  tp={tp_size} dp=1 ep={tp_size}")
            print(f"  num_runs={args.num_runs}")
            print(f"  dInfer source: {dinfer.__file__}")
            print("=" * 70)

        # --- Model loading ---
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)
        config._attn_implementation = "sdpa"

        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model.tensor_parallel(tp_size)
        model = model.to(device)

        if rank == 0:
            print(f"  Model loaded. attn={config._attn_implementation}")
            attn_cls = model.model.layers[0].attention.__class__.__name__
            print(f"  Attention class: {attn_cls}, tp_size={model.model.layers[0].attention.tp_size}")

        # --- Warmup model forward ---
        with torch.inference_mode():
            warmup_tok = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            _ = model(warmup_tok, use_cache=False)
        torch.cuda.synchronize()

        # --- Load prompts ---
        if args.prompt_source == "heteval":
            prompt_list = PROMPTS
        else:
            _ds_map = {
                "gsm8k": ("data/gsm8k.jsonl", "question"),
                "humaneval": ("data/humaneval.jsonl", "prompt"),
                "mgsm": ("data/mgsm.jsonl", "question"),
                "mt_bench": ("data/mt_bench.jsonl", None),
            }
            _path, _key = _ds_map[args.prompt_source]
            _lines = (REPO_ROOT / _path).read_text().strip().split("\n")
            prompt_list = []
            for _l in _lines:
                _obj = json.loads(_l)
                if _key:
                    prompt_list.append(_obj[_key])
                else:
                    prompt_list.append(_obj["turns"][0])
            if rank == 0:
                print(f"  Loaded {len(prompt_list)} prompts from {args.prompt_source}")

        # --- Tokenization (same as bench_tv6_throughput.py) ---
        all_ids = []
        for i in range(args.batch_size):
            text = prompt_list[i % len(prompt_list)]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=False,
                )
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        target_len = args.pad_to if args.pad_to > 0 else max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = []
        for ids in all_ids:
            if ids.shape[0] > target_len:
                ids = ids[-target_len:]
            elif ids.shape[0] < target_len:
                ids = torch.cat([torch.full((target_len - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
            padded.append(ids)
        input_ids_full = torch.stack(padded, dim=0).to(device)

        if rank == 0:
            print(f"  Input shape: {tuple(input_ids_full.shape)}")

        # --- Benchmark function ---
        def run_benchmark():
            decoder = ThresholdParallelDecoder(
                temperature=0.0, threshold=0.90,
                mask_id=MASK_ID, eos_id=EOS_ID,
            )

            if args.no_cache:
                return run_benchmark_no_cache(decoder)

            # Warmup generate
            dllm_warmup = BlockDiffusionLLM(
                model, decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True,
            )
            with torch.inference_mode():
                dllm_warmup.diff_iteration.iter_no = 0
                _ = dllm_warmup.naive_batching_generate(
                    input_ids_full.clone(),
                    gen_length=args.gen_length,
                    block_length=BLOCK_LENGTH,
                )
            torch.cuda.synchronize()
            dist.barrier()
            if rank == 0:
                print(f"  Warmup done: {dllm_warmup.diff_iteration.num_forwards} fwd")

            # Timed runs
            times = []
            fwd_counts = []
            for r_idx in range(args.num_runs):
                dllm = BlockDiffusionLLM(
                    model, decoder,
                    BlockIteratorFactory(use_block_diffusion=True),
                    cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                    early_stop=True,
                )
                torch.cuda.synchronize()
                dist.barrier()
                t0 = time.time()
                with torch.inference_mode():
                    dllm.diff_iteration.iter_no = 0
                    out = dllm.naive_batching_generate(
                        input_ids_full.clone(),
                        gen_length=args.gen_length,
                        block_length=BLOCK_LENGTH,
                    )
                torch.cuda.synchronize()
                dist.barrier()
                elapsed = time.time() - t0
                nf = dllm.diff_iteration.num_forwards
                times.append(elapsed)
                fwd_counts.append(nf)

                if rank == 0:
                    ms_fwd = elapsed / nf * 1000
                    print(f"    [Baseline] run {r_idx}: {elapsed:.3f}s, "
                          f"{nf} fwd, {ms_fwd:.2f} ms/fwd")

                    if not args.no_quality and r_idx == 0:
                        n_show = min(5, out.shape[0])
                        for j in range(n_show):
                            toks = out[j]
                            valid = toks[(toks != 0) & (toks != EOS_ID) & (toks != MASK_ID)]
                            text = tokenizer.decode(valid, skip_special_tokens=True)
                            gen_start = text.find("ASSISTANT") + len("ASSISTANT") if "ASSISTANT" in text else 0
                            gen_text = text[gen_start:].strip()
                            print(f"      === sample {j} ===")
                            print(f"      {gen_text[:600]}")
                            print()

            return times, fwd_counts

        def run_benchmark_no_cache(decoder):
            """No-cache mode: BlockDiffusionLLMAttnmask with batch>1 support."""
            from dinfer.decoding.generate_uniform import BlockDiffusionRunner, BlockDiffusionIteration
            from dinfer.decoding.utils import TokenArray

            def batched_generate(dllm_obj, prompt, gen_length=256, block_length=32):
                batch_size = prompt.shape[0]
                prompt_length = prompt.shape[1]
                num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
                total_length = num_blocks * block_length
                new_gen_length = total_length - prompt_length

                block_mask = torch.tril(torch.ones(
                    num_blocks, num_blocks, device=model.device))
                bd_attn_mask = block_mask.repeat_interleave(block_length, dim=0) \
                    .repeat_interleave(block_length, dim=1).unsqueeze(0).repeat(batch_size, 1, 1)
                pos_ids = torch.arange(
                    total_length, device=model.device).unsqueeze(0).repeat(batch_size, 1)

                x = TokenArray(prompt, new_gen_length,
                               dllm_obj.decoder.mask_id, dllm_obj.decoder.eos_id, model.device)
                it = dllm_obj.iterator_factory.create(x, block_length)

                dllm_obj.diff_iteration.iter_no = 0
                for block_id, (block_loc, block) in enumerate(it):
                    dllm_obj.decoder.block_init(block, block_id)
                    decode_compl = dllm_obj.block_runner.decode(
                        model, dllm_obj.decoder, x, None, block, block_loc,
                        block_id, pos_ids, bd_attn_mask)
                    if torch.all(decode_compl):
                        break
                return x.get_generated_tokens()

            def make_dllm():
                dllm = BlockDiffusionLLMAttnmask(
                    model, decoder,
                    BlockIteratorFactory(use_block_diffusion=True),
                    early_stop=True,
                )
                return dllm

            # Warmup
            dllm_w = make_dllm()
            with torch.inference_mode():
                _ = batched_generate(dllm_w, input_ids_full.clone(),
                                     args.gen_length, BLOCK_LENGTH)
            torch.cuda.synchronize()
            dist.barrier()
            if rank == 0:
                print(f"  Warmup done (no-cache): {dllm_w.diff_iteration.num_forwards} fwd")

            times = []
            fwd_counts = []
            for r_idx in range(args.num_runs):
                dllm = make_dllm()
                torch.cuda.synchronize()
                dist.barrier()
                t0 = time.time()
                with torch.inference_mode():
                    out = batched_generate(dllm, input_ids_full.clone(),
                                           args.gen_length, BLOCK_LENGTH)
                torch.cuda.synchronize()
                dist.barrier()
                elapsed = time.time() - t0
                nf = dllm.diff_iteration.num_forwards
                times.append(elapsed)
                fwd_counts.append(nf)

                if rank == 0:
                    ms_fwd = elapsed / nf * 1000
                    print(f"    [Baseline no-cache] run {r_idx}: {elapsed:.3f}s, "
                          f"{nf} fwd, {ms_fwd:.2f} ms/fwd")

            return times, fwd_counts

        # --- Run ---
        cache_label = "no-cache" if args.no_cache else "prefix-cache"
        if rank == 0:
            print(f"\n  --- Baseline (dInfer native, TP={tp_size}, K=8, {cache_label}) ---")
        times, fwd_counts = run_benchmark()

        # --- Summary ---
        if rank == 0:
            mean_ms_fwd = sum(t / f * 1000 for t, f in zip(times, fwd_counts)) / len(times)
            print("\n" + "=" * 70)
            print("Summary:")
            print(f"  Baseline: {mean_ms_fwd:.2f} ms/fwd "
                  f"(avg over {len(times)} runs)")
            for i, (t, f) in enumerate(zip(times, fwd_counts)):
                print(f"    run {i}: {t:.3f}s, {f} fwd, {t/f*1000:.2f} ms/fwd")
            print("=" * 70)

            results = {
                "baseline": {
                    "times": times,
                    "fwd_counts": fwd_counts,
                    "mean_ms_fwd": mean_ms_fwd,
                },
            }
            payload = {
                "batch_size": args.batch_size,
                "gen_length": args.gen_length,
                "block_length": BLOCK_LENGTH,
                "tp_size": tp_size,
                "dp_size": 1,
                "num_runs": args.num_runs,
                "prompt_source": args.prompt_source,
                "dinfer_source": str(BASELINE_DINFER),
                "attn_implementation": "sdpa",
                "cache_mode": "nocache" if args.no_cache else "cache",
                "results": results,
            }
            out_path = REPO_ROOT / "codex_coding" / "results" / "bench_baseline_dinfer.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"  Saved to {out_path}")

            tag = "baseline_nocache" if args.no_cache else "baseline_cache"
            paper_dir = REPO_ROOT / "codex_coding" / "results" / "bench_paper"
            paper_dir.mkdir(parents=True, exist_ok=True)
            paper_path = paper_dir / f"{tag}_{args.prompt_source}_b{args.batch_size}.json"
            with open(paper_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"  Saved to {paper_path}")

        dist.barrier()


if __name__ == "__main__":
    main()
