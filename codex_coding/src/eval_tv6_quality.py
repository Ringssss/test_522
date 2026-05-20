#!/usr/bin/env python3
"""
TV6 Quality Evaluation — Uses lm-eval task API for prompt/scoring,
our torchrun dp=2 tp=4 pipeline for inference.

Supports baseline (dInfer native) and TV6 comparison on GSM8K.

Launch:
    cd /home/wuhang/wuhang/dllm_wh && \\
    DINF_SKIP_LOGITS_FLOAT=1 DINF_SP_LM_HEAD=1 \\
    torchrun --nproc_per_node=8 codex_coding/src/eval_tv6_quality.py \\
        --tasks gsm8k --limit 0 --mode both
"""
from __future__ import annotations

import argparse
import json
import os
import re
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


# =====================================================================
# lm-eval task utilities
# =====================================================================

def load_eval_prompts(task_name: str, tokenizer, num_fewshot: int = 0, limit: int = 0):
    """Use lm-eval task API to build prompts and references."""
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    task_dict = tm.load_task_or_group(task_name)
    task = task_dict[task_name]

    docs = list(task.test_docs())
    if limit > 0:
        docs = docs[:limit]

    prompts = []
    references = []
    stop_tokens = task.config.generation_kwargs.get("until", [])

    for doc in docs:
        ctx = task.fewshot_context(doc, num_fewshot)
        prompts.append(ctx)
        ref = task.doc_to_target(doc)
        references.append(ref)

    return prompts, references, stop_tokens, task


def score_gsm8k(generated: str, reference: str) -> dict:
    """Score a single GSM8K sample using lm-eval's extraction logic."""
    # Reference: extract number after ####
    ref_match = re.search(r'####\s*(.+)', reference)
    ref_answer = ref_match.group(1).strip() if ref_match else reference.strip()
    ref_answer = ref_answer.replace(",", "").replace("$", "").rstrip(".")

    # Strict match: extract #### NUMBER from generated
    strict_match = re.search(r'####\s*(\-?[0-9\.\,]+)', generated)
    strict_answer = strict_match.group(1).replace(",", "").rstrip(".") if strict_match else ""
    strict_correct = (strict_answer == ref_answer)

    # Flexible extract: last number in generated
    flex_matches = re.findall(r'(-?[$0-9.,]{2,})|(-?[0-9]+)', generated)
    if flex_matches:
        last = [m[0] or m[1] for m in flex_matches][-1]
        flex_answer = last.replace(",", "").replace("$", "").rstrip(".")
    else:
        flex_answer = ""
    flex_correct = (flex_answer == ref_answer)

    return {
        "ref": ref_answer,
        "strict": strict_correct,
        "strict_pred": strict_answer,
        "flex": flex_correct,
        "flex_pred": flex_answer,
    }


def truncate_at_stop(text: str, stop_tokens: list[str]) -> str:
    for st in stop_tokens:
        idx = text.find(st)
        if idx >= 0:
            text = text[:idx]
    return text


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="TV6 quality evaluation")
    parser.add_argument("--tasks", type=str, default="gsm8k")
    parser.add_argument("--limit", type=int, default=0, help="0 = full test set")
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mode", choices=["baseline", "tv6", "both"], default="both")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    tp_size = 4 if world_size == 8 else 2
    dp_size = world_size // tp_size
    dp_rank = rank // tp_size
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
        from baseline_optimizations import apply_all_optimizations

        if rank == 0:
            print("=" * 70)
            print("TV6 Quality Evaluation (torchrun + lm-eval tasks)")
            print(f"  tasks={args.tasks} limit={args.limit} fewshot={args.num_fewshot}")
            print(f"  gen={args.gen_length} batch={args.batch_size}")
            print(f"  tp={tp_size} dp={dp_size} ep={world_size}")
            print(f"  mode={args.mode}")
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

        # --- Load lm-eval prompts (rank 0 only, then broadcast) ---
        prompts_text = None
        references = None
        stop_tokens = None
        if rank == 0:
            prompts_text, references, stop_tokens, task = load_eval_prompts(
                args.tasks, tokenizer, args.num_fewshot, args.limit)
            print(f"  Loaded {len(prompts_text)} prompts from {args.tasks}")

        # Broadcast metadata
        obj_list = [{"prompts": prompts_text, "references": references,
                     "stop_tokens": stop_tokens}]
        dist.broadcast_object_list(obj_list, src=0)
        prompts_text = obj_list[0]["prompts"]
        references = obj_list[0]["references"]
        stop_tokens = obj_list[0]["stop_tokens"]

        n_total = len(prompts_text)

        # --- Tokenize all prompts ---
        pad_id = tokenizer.pad_token_id or 0

        # --- Generate function ---
        def run_eval(label: str, setup_fn=None):
            decoder = ThresholdParallelDecoder(
                temperature=0.0, threshold=0.90,
                mask_id=MASK_ID, eos_id=EOS_ID,
            )
            if setup_fn is not None:
                setup_fn(model, decoder, vllm_cfg)

            dllm = BlockDiffusionLLM(
                model, decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend="vllm", lazy_cache_update=True, inplace_cache_update=True,
            )

            # Warmup
            dummy_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]
            dummy_batch = dummy_ids.repeat(args.batch_size // dp_size, 1).to(device)
            with torch.inference_mode():
                dllm.diff_iteration.iter_no = 0
                _ = dllm.generate(dummy_batch, gen_length=64, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            dist.barrier()

            # Process in batches
            local_bs = args.batch_size // dp_size
            all_generated = []
            total_fwd = 0
            t0 = time.time()

            for batch_start in range(0, n_total, args.batch_size):
                batch_end = min(batch_start + args.batch_size, n_total)
                batch_prompts = prompts_text[batch_start:batch_end]

                # Tokenize batch
                batch_ids = []
                for text in batch_prompts:
                    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
                    batch_ids.append(ids)

                # Pad to same length
                mx = max(x.shape[0] for x in batch_ids)
                padded = [
                    torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                    if ids.shape[0] < mx else ids
                    for ids in batch_ids
                ]

                # Pad batch to full batch_size if needed
                actual_bs = len(padded)
                while len(padded) < args.batch_size:
                    padded.append(padded[-1].clone())

                input_full = torch.stack(padded, dim=0)
                my_input = input_full[dp_rank * local_bs: (dp_rank + 1) * local_bs].to(device)

                # Generate
                dllm2 = BlockDiffusionLLM(
                    model, decoder,
                    BlockIteratorFactory(use_block_diffusion=True),
                    cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                    early_stop=True, maximum_unroll=4, expected_tpf=15,
                    backend="vllm", lazy_cache_update=True, inplace_cache_update=True,
                )
                with torch.inference_mode():
                    dllm2.diff_iteration.iter_no = 0
                    out = dllm2.generate(
                        my_input.clone(), gen_length=args.gen_length,
                        block_length=BLOCK_LENGTH,
                    )
                total_fwd += dllm2.diff_iteration.num_forwards

                # Decode outputs (dp_rank 0 handles first half of batch)
                if dp_rank == 0:
                    for j in range(min(local_bs, actual_bs)):
                        prompt_len = batch_ids[j].shape[0]
                        gen_tokens = out[j, prompt_len:]
                        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                        text = truncate_at_stop(text, stop_tokens)
                        all_generated.append(text)

                # dp_rank 1 handles second half
                if dp_rank == 1:
                    for j in range(min(local_bs, actual_bs - local_bs)):
                        idx = local_bs + j
                        if idx < actual_bs:
                            prompt_len = batch_ids[idx].shape[0]
                            gen_tokens = out[j, prompt_len:]
                            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                            text = truncate_at_stop(text, stop_tokens)
                            all_generated.append(text)

                torch.cuda.synchronize()
                dist.barrier()

                if rank == 0 and batch_start % (args.batch_size * 5) == 0:
                    print(f"    [{label}] batch {batch_start}/{n_total}")

            elapsed = time.time() - t0

            # Gather generated texts from dp_rank 1 to dp_rank 0
            gathered_texts = [None] * world_size
            dist.all_gather_object(gathered_texts, all_generated)

            if rank == 0:
                # Merge: dp_rank 0 (ranks 0-3) has first half, dp_rank 1 (ranks 4-7) has second half
                texts_dp0 = gathered_texts[0]  # rank 0 = dp_rank 0
                texts_dp1 = gathered_texts[tp_size]  # rank tp_size = dp_rank 1
                # Interleave: for each batch, dp0 has first local_bs, dp1 has second local_bs
                merged = []
                i0, i1 = 0, 0
                for batch_start in range(0, n_total, args.batch_size):
                    batch_end = min(batch_start + args.batch_size, n_total)
                    actual_bs = batch_end - batch_start
                    n_dp0 = min(local_bs, actual_bs)
                    n_dp1 = actual_bs - n_dp0
                    merged.extend(texts_dp0[i0:i0 + n_dp0])
                    i0 += n_dp0
                    merged.extend(texts_dp1[i1:i1 + n_dp1])
                    i1 += n_dp1

                return merged, total_fwd, elapsed
            return None, total_fwd, elapsed

        # --- Run evaluations ---
        results_summary = {}

        for mode in (["baseline", "tv6"] if args.mode == "both" else [args.mode]):
            if rank == 0:
                print(f"\n  --- {mode.upper()} ---")

            setup_fn = None
            if mode == "tv6":
                from tv6_patch import apply_tv6
                setup_fn = apply_tv6

            generated, total_fwd, elapsed = run_eval(mode, setup_fn)

            if rank == 0:
                print(f"    {total_fwd} total fwd in {elapsed:.1f}s")

                # Score
                strict_correct = 0
                flex_correct = 0
                n_scored = min(len(generated), len(references))

                for i in range(n_scored):
                    result = score_gsm8k(generated[i], references[i])
                    if result["strict"]:
                        strict_correct += 1
                    if result["flex"]:
                        flex_correct += 1
                    if i < 3:
                        print(f"    Sample {i}: ref={result['ref']}, "
                              f"strict={result['strict_pred']}, "
                              f"flex={result['flex_pred']}")

                strict_acc = strict_correct / n_scored if n_scored > 0 else 0
                flex_acc = flex_correct / n_scored if n_scored > 0 else 0
                print(f"    [{mode}] strict_match: {strict_correct}/{n_scored} = {strict_acc:.4f}")
                print(f"    [{mode}] flex_extract: {flex_correct}/{n_scored} = {flex_acc:.4f}")

                results_summary[mode] = {
                    "strict_match": round(strict_acc, 4),
                    "flex_extract": round(flex_acc, 4),
                    "n_scored": n_scored,
                    "total_fwd": total_fwd,
                    "elapsed": round(elapsed, 1),
                }

            dist.barrier()

        # --- Save ---
        if rank == 0:
            print("\n" + "=" * 70)
            print("Summary:")
            for mode, data in results_summary.items():
                print(f"  {mode}: strict={data['strict_match']}, flex={data['flex_extract']}")
            print("=" * 70)

            out_path = REPO_ROOT / "codex_coding" / "results" / "eval_tv6_quality.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({
                    "task": args.tasks,
                    "limit": args.limit,
                    "num_fewshot": args.num_fewshot,
                    "gen_length": args.gen_length,
                    "batch_size": args.batch_size,
                    "results": results_summary,
                }, f, indent=2)
            print(f"  Saved to {out_path}")

        dist.barrier()


if __name__ == "__main__":
    main()
