#!/usr/bin/env python3
"""
Diagnose quality: compare prefill logits between AllReduce EP and DP AllToAll EP.

Usage:
  # AllReduce EP (tp=4, dp=1) — baseline
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 diag_prefill_logits.py --mode allreduce

  # DP AllToAll EP (tp=1, dp=4)
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 diag_prefill_logits.py --mode alltoall
"""
import os, sys, argparse, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("VLLM_ALL2ALL_BACKEND", "naive")

sys.path.insert(0, "/home/wuhang/wuhang/dllm_wh/codex_coding/src")

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["allreduce", "alltoall"], required=True)
args = parser.parse_args()

from vllm import distributed as vllm_dist
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from dinfer.model import LLaDA2MoeModelLM
from dinfer.model.modeling_llada2_moe import _maybe_patch_all_reduce
from transformers import AutoTokenizer, AutoConfig
from test_heteval128 import PROMPTS

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"

if args.mode == "allreduce":
    # Old mode: tp=world_size, dp=1
    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        vllm_dist.init_distributed_environment(world_size, rank, "env://", local_rank, "nccl")
        vllm_dist.initialize_model_parallel(world_size, backend="nccl")
        _maybe_patch_all_reduce()
else:
    # New mode: tp=1, dp=world_size
    pcfg_init = ParallelConfig(tensor_parallel_size=1, data_parallel_size=1, enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(world_size, rank, "env://", local_rank, "nccl")
    pcfg = ParallelConfig(tensor_parallel_size=1, data_parallel_size=world_size,
                          data_parallel_rank=rank, enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        vllm_dist.initialize_model_parallel(tensor_model_parallel_size=1, backend="nccl")
        _maybe_patch_all_reduce()

vllm_cfg = VllmConfig(parallel_config=pcfg)

with set_current_vllm_config(vllm_cfg):
    from vllm.distributed import get_ep_group
    ep = get_ep_group()
    if rank == 0:
        print(f"Mode: {args.mode}, EP size={ep.world_size}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    model = LLaDA2MoeModelLM(config=config).eval()
    model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
    model = model.to(device)

    # Warmup
    from vllm.forward_context import set_forward_context
    with torch.inference_mode():
        warmup = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
        if args.mode == "alltoall":
            with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg, num_tokens=warmup.numel()):
                _ = model(warmup, use_cache=False)
        else:
            _ = model(warmup, use_cache=False)

    from baseline_optimizations import apply_all_optimizations
    apply_all_optimizations(model)

    # Patch routing (C5 fused_routing, no EB)
    from test_fused_eb_triton import fused_routing
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "LLaDA2MoeGate":
            b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                 mod.top_k, mod.n_group, mod.topk_group)
            def mk(bb, rr, tt, nn, gg):
                def fn(hs, go, topk, renorm):
                    w, i = fused_routing(go, bb, rr, s_mask=None, K=tt, ng=nn, tkg=gg)
                    return w.to(go.dtype), i
                return fn
            mod.routing = mk(b, r, tk, ng, tkg)

    # Prepare input: prompt #0 only
    text = PROMPTS[0]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)  # [1, L]

    if args.mode == "allreduce":
        # All ranks get the same input
        my_input = input_ids
    else:
        # DP mode: only rank 0 gets prompt #0, others get padding
        # But to make prefill work without hanging, all ranks must have same-length input
        # Give all ranks prompt #0 (same input) — we only care about rank 0's output
        my_input = input_ids

    if rank == 0:
        print(f"Input shape: {my_input.shape}, prompt: '{PROMPTS[0][:50]}...'", flush=True)

    # Run a single forward (prefill) — no KV cache, no decode
    with torch.inference_mode():
        if args.mode == "alltoall":
            with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                     num_tokens=my_input.numel()):
                output = model(my_input, use_cache=False)
        else:
            output = model(my_input, use_cache=False)

    logits = output.logits  # [1, L, vocab]
    if rank == 0:
        # Print diagnostics for prompt #0
        last_logits = logits[0, -1]  # last position
        pred = last_logits.argmax().item()
        top5_vals, top5_ids = last_logits.topk(5)
        logits_sum = logits.sum().item()
        logits_norm = logits.float().norm().item()
        first10 = logits[0, -1, :10].tolist()

        print(f"\n=== Prefill Logits (rank 0, prompt #0) ===", flush=True)
        print(f"  pred={pred} ({tokenizer.decode([pred])})", flush=True)
        print(f"  logits_sum={logits_sum:.4f}", flush=True)
        print(f"  logits_norm={logits_norm:.4f}", flush=True)
        print(f"  top5: {[(i.item(), f'{v.item():.4f}') for i, v in zip(top5_ids, top5_vals)]}", flush=True)
        print(f"  first10: {[f'{x:.4f}' for x in first10]}", flush=True)

        # Save for comparison
        torch.save({
            'logits_last': logits[0, -1].cpu(),
            'logits_sum': logits_sum,
            'pred': pred,
            'mode': args.mode,
        }, f"/tmp/diag_prefill_{args.mode}.pt")
        print(f"  Saved to /tmp/diag_prefill_{args.mode}.pt", flush=True)

import torch.distributed as dist
dist.barrier()
dist.destroy_process_group()
