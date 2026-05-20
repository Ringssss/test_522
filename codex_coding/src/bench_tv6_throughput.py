#!/usr/bin/env python3
"""
TV6 Throughput Benchmark — A/B comparison of baseline G vs TV6.

Based on bench_bsp_moe_dp2.py's dp=2 tp=4 ep=8 infrastructure,
uses tv6_patch.py for clean TV6 injection.

Launch:
    cd /home/wuhang/wuhang/dllm_wh && \\
    DINF_SKIP_LOGITS_FLOAT=1 DINF_SP_LM_HEAD=1 \\
    torchrun --nproc_per_node=8 codex_coding/src/bench_tv6_throughput.py \\
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

if os.environ.get("BSP_DISABLE_DEEP_EP", "1") != "0":
    sys.modules["deep_ep"] = None

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID = 156895
EOS_ID = 156892
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
BLOCK_LENGTH = 32


def main():
    parser = argparse.ArgumentParser(description="TV6 throughput A/B benchmark")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--num-runs", type=int, default=2)
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--tv6-only", action="store_true", help="Skip baseline, only run TV6")
    parser.add_argument("--baseline-only", action="store_true", help="Skip TV6, only run baseline")
    parser.add_argument("--prompt-source", choices=["heteval", "gsm8k", "humaneval", "mgsm", "mt_bench"],
                        default="heteval", help="Prompt source for quality evaluation")
    parser.add_argument("--pad-to", type=int, default=0,
                        help="Pad/truncate all prompts to this fixed token length (0=auto)")
    parser.add_argument("--tp-size", type=int, default=0,
                        help="Override TP size (0=auto: 4 for 8GPU, world_size for others)")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    tp_size = args.tp_size if args.tp_size > 0 else (4 if world_size == 8 else world_size)
    dp_size = world_size // tp_size
    dp_rank = rank // tp_size
    local_bs = args.batch_size // dp_size
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

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
        tensor_parallel_size=tp_size, data_parallel_size=dp_size,
        data_parallel_rank=dp_rank, enable_expert_parallel=True,
    )
    vllm_cfg = VllmConfig(parallel_config=pcfg)

    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, backend="nccl")

        from vllm.distributed import prepare_communication_buffer_for_model
        from vllm.forward_context import set_forward_context
        from dinfer import (
            BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
            ThresholdParallelDecoder,
        )
        from dinfer.model import LLaDA2MoeModelLM
        from transformers import AutoConfig, AutoTokenizer
        from test_heteval512 import PROMPTS
        from baseline_optimizations import apply_all_optimizations

        if rank == 0:
            print("=" * 70)
            print("TV6 Throughput A/B Benchmark")
            print(f"  batch={args.batch_size} gen={args.gen_length} block={BLOCK_LENGTH}")
            print(f"  tp={tp_size} dp={dp_size} ep={world_size}")
            print(f"  num_runs={args.num_runs}")
            print("=" * 70)

        # --- Model loading ---
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)

        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        with torch.inference_mode():
            warmup_tok = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(
                attn_metadata=None, vllm_config=vllm_cfg,
                num_tokens=warmup_tok.numel(),
            ):
                _ = model(warmup_tok, use_cache=False)

        apply_all_optimizations(model)
        prepare_communication_buffer_for_model(model)

        # --- Load prompts ---
        if args.prompt_source == "heteval":
            from test_heteval512 import PROMPTS
            prompt_list = PROMPTS
        else:
            import json as _json
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
                _obj = _json.loads(_l)
                if _key:
                    prompt_list.append(_obj[_key])
                else:
                    prompt_list.append(_obj["turns"][0])
            if rank == 0:
                print(f"  Loaded {len(prompt_list)} prompts from {args.prompt_source}")

        # --- Tokenization ---
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
        input_ids_full = torch.stack(padded, dim=0)
        my_input = input_ids_full[dp_rank * local_bs: (dp_rank + 1) * local_bs].to(device)

        if rank == 0:
            print(f"  Input shape (local): {tuple(my_input.shape)}")

        # --- Run function ---
        def run_benchmark(label, setup_fn=None):
            decoder = ThresholdParallelDecoder(
                temperature=0.0, threshold=0.90,
                mask_id=MASK_ID, eos_id=EOS_ID,
            )

            tv6_info = None
            if setup_fn is not None:
                tv6_info = setup_fn(model, decoder, vllm_cfg)

            dllm = BlockDiffusionLLM(
                model, decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend="vllm", lazy_cache_update=True, inplace_cache_update=True,
            )

            # Warmup
            with torch.inference_mode():
                dllm.diff_iteration.iter_no = 0
                _ = dllm.generate(my_input.clone(), gen_length=args.gen_length,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            dist.barrier()

            # Timed runs
            times = []
            fwd_counts = []
            for r_idx in range(args.num_runs):
                dllm2 = BlockDiffusionLLM(
                    model, decoder,
                    BlockIteratorFactory(use_block_diffusion=True),
                    cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                    early_stop=True, maximum_unroll=4, expected_tpf=15,
                    backend="vllm", lazy_cache_update=True, inplace_cache_update=True,
                )
                torch.cuda.synchronize()
                dist.barrier()
                t0 = time.time()
                with torch.inference_mode():
                    dllm2.diff_iteration.iter_no = 0
                    out = dllm2.generate(
                        my_input.clone(), gen_length=args.gen_length,
                        block_length=BLOCK_LENGTH,
                    )
                torch.cuda.synchronize()
                dist.barrier()
                elapsed = time.time() - t0
                nf = dllm2.diff_iteration.num_forwards
                times.append(elapsed)
                fwd_counts.append(nf)

                if rank == 0:
                    ms_fwd = elapsed / nf * 1000
                    print(f"    [{label}] run {r_idx}: {elapsed:.3f}s, "
                          f"{nf} fwd, {ms_fwd:.2f} ms/fwd")

                    if not args.no_quality and r_idx == 0:
                        n_show = min(5, out.shape[0])
                        for j in range(n_show):
                            toks = out[j]
                            valid = toks[(toks != 0) & (toks != EOS_ID) & (toks != MASK_ID)]
                            text = tokenizer.decode(valid, skip_special_tokens=True)
                            # Show only generated part (after prompt)
                            prompt_text = prompt_list[j % len(prompt_list)]
                            gen_start = text.find("ASSISTANT") + len("ASSISTANT") if "ASSISTANT" in text else 0
                            gen_text = text[gen_start:].strip()
                            print(f"      === sample {j} ===")
                            print(f"      {gen_text[:600]}")
                            print()

            if rank == 0 and tv6_info and 'debug_sp_counts' in tv6_info:
                sp = tv6_info['debug_sp_counts']
                print(f"    SP-LM debug: sp_hit={sp[0]}, sp_miss={sp[1]}")

            return times, fwd_counts

        # --- A/B runs ---
        results = {}

        if not args.tv6_only:
            if rank == 0:
                print("\n  --- Baseline (G path) ---")
            # Baseline: no TV6 patch, just the model as-is with BSP-G from source
            # Note: apply_all_optimizations already applied flash_attn etc.
            # The "baseline" here uses the dInfer native path (no BSP-G monkey-patch)
            times_b, fwd_b = run_benchmark("Baseline")
            results["baseline"] = {
                "times": times_b,
                "fwd_counts": fwd_b,
                "mean_ms_fwd": sum(t / f * 1000 for t, f in zip(times_b, fwd_b)) / len(times_b),
            }

        if not args.baseline_only:
            if rank == 0:
                print("\n  --- TV6 (compact dispatch/combine) ---")
            from tv6_patch import apply_tv6
            times_tv6, fwd_tv6 = run_benchmark("TV6", setup_fn=apply_tv6)
            results["tv6"] = {
                "times": times_tv6,
                "fwd_counts": fwd_tv6,
                "mean_ms_fwd": sum(t / f * 1000 for t, f in zip(times_tv6, fwd_tv6)) / len(times_tv6),
            }

        # --- Summary ---
        if rank == 0:
            print("\n" + "=" * 70)
            print("Summary:")
            for name, data in results.items():
                print(f"  {name}: {data['mean_ms_fwd']:.2f} ms/fwd "
                      f"(avg over {len(data['times'])} runs)")
            if "baseline" in results and "tv6" in results:
                b = results["baseline"]["mean_ms_fwd"]
                t = results["tv6"]["mean_ms_fwd"]
                delta = (t - b) / b * 100
                print(f"  TV6 vs Baseline: {delta:+.2f}%")
            print("=" * 70)

            payload = {
                "batch_size": args.batch_size,
                "gen_length": args.gen_length,
                "block_length": BLOCK_LENGTH,
                "tp_size": tp_size,
                "dp_size": dp_size,
                "num_runs": args.num_runs,
                "prompt_source": args.prompt_source,
                "results": results,
            }
            out_path = REPO_ROOT / "codex_coding" / "results" / "bench_tv6_throughput.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"  Saved to {out_path}")

            paper_dir = REPO_ROOT / "codex_coding" / "results" / "bench_paper"
            paper_dir.mkdir(parents=True, exist_ok=True)
            ps = args.prompt_source
            bs = args.batch_size
            if "baseline" in results:
                p = paper_dir / f"vanilla_{ps}_b{bs}.json"
                with open(p, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"  Saved to {p}")
            if "tv6" in results:
                p = paper_dir / f"tv6_{ps}_b{bs}.json"
                with open(p, "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"  Saved to {p}")

        dist.barrier()


if __name__ == "__main__":
    main()
