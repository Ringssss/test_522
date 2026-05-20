#!/usr/bin/env python3
"""Quick check: what tokens does DeepEP actually produce?"""
from __future__ import annotations
import os, sys, torch
from pathlib import Path

REPO = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO / "codex_coding" / "src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BACKEND = os.environ.get("VLLM_ALL2ALL_BACKEND", "naive")
MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32

local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"

from vllm import distributed as vllm_dist
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                    ThresholdParallelDecoder)
from dinfer.model import LLaDA2MoeModelLM
from dinfer.model.modeling_llada2_moe import _maybe_patch_all_reduce
from transformers import AutoTokenizer, AutoConfig
import torch.distributed as dist

pcfg_init = ParallelConfig(tensor_parallel_size=1, data_parallel_size=1,
                           enable_expert_parallel=True)
with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
    vllm_dist.init_distributed_environment(world_size, rank, "env://",
                                           local_rank, "nccl")

pcfg = ParallelConfig(tensor_parallel_size=1, data_parallel_size=world_size,
                      data_parallel_rank=rank, enable_expert_parallel=True)
vllm_cfg = VllmConfig(parallel_config=pcfg)

with set_current_vllm_config(vllm_cfg):
    vllm_dist.initialize_model_parallel(tensor_model_parallel_size=1,
                                        backend="nccl")
    _maybe_patch_all_reduce()

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                        local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                              local_files_only=True)
    model = LLaDA2MoeModelLM(config=config).eval()
    model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
    model = model.to(device)

    # Apply fused_routing
    from test_fused_eb_triton import fused_routing
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "LLaDA2MoeGate":
            b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                 mod.top_k, mod.n_group, mod.topk_group)
            def mk(bb, rr, tt, nn, gg):
                def fn(hs, go, topk, renorm):
                    w, i = fused_routing(go, bb, rr, s_mask=None, K=tt,
                                         ng=nn, tkg=gg)
                    return w.to(go.dtype), i
                return fn
            mod.routing = mk(b, r, tk, ng, tkg)

    from baseline_optimizations import apply_all_optimizations
    apply_all_optimizations(model)

    # Single prompt, 2 GPUs
    prompt = "What is 17 * 23?"
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False)
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]

    # Each rank gets same prompt
    batch = tokens.unsqueeze(0).to(device)
    prompt_len = batch.shape[1]

    decoder = ThresholdParallelDecoder(
        temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
    dllm = BlockDiffusionLLM(
        model, decoder,
        BlockIteratorFactory(use_block_diffusion=True),
        cache_factory=KVCacheFactory("prefix", is_bd_model=True),
        early_stop=True, maximum_unroll=4, expected_tpf=15,
        backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

    with torch.inference_mode():
        dllm.diff_iteration.num_forwards = 0
        out = dllm.generate(batch.clone(), gen_length=64,
                            block_length=BLOCK_LENGTH)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        gen = out[0, prompt_len:]
        n_mask = (gen == MASK_ID).sum().item()
        n_eos = (gen == EOS_ID).sum().item()
        n_pad = (gen == 0).sum().item()
        n_other = len(gen) - n_mask - n_eos - n_pad
        print(f"Backend: {BACKEND}")
        print(f"Forwards: {dllm.diff_iteration.num_forwards}")
        print(f"Gen length: {len(gen)}")
        print(f"  MASK tokens: {n_mask}")
        print(f"  EOS tokens:  {n_eos}")
        print(f"  PAD tokens:  {n_pad}")
        print(f"  Other:       {n_other}")
        print(f"Raw tokens (first 64): {gen[:64].tolist()}")
        valid = gen[(gen != 0) & (gen != EOS_ID) & (gen != MASK_ID)]
        text = tokenizer.decode(valid, skip_special_tokens=True)
        print(f"Decoded text: '{text[:200]}'")

dist.barrier()
dist.destroy_process_group()
