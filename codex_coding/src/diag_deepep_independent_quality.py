#!/usr/bin/env python3
"""
Independent quality evaluation for DeepEP HT.
5 math prompts with known answers, evaluated independently (not compared to naive).

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_ALL2ALL_BACKEND=deepep_high_throughput \
    torchrun --nproc_per_node=4 codex_coding/src/diag_deepep_independent_quality.py
"""
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

# Two-phase init
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

    # Apply baseline optimizations
    from baseline_optimizations import apply_all_optimizations
    apply_all_optimizations(model)

    # ★★★ FIX: Initialize DeepEP communication buffers
    from vllm.distributed import prepare_communication_buffer_for_model
    prepare_communication_buffer_for_model(model)

    # If USE_DEEPEP_V2=1, replace with V2 ElasticBuffer
    if int(os.environ.get("USE_DEEPEP_V2", "0")):
        from deepep_v2_pf import replace_with_deepep_v2
        from vllm.distributed import get_ep_group
        ep = get_ep_group()
        n_replaced, v2_sms = replace_with_deepep_v2(
            model, ep.cpu_group,
            num_local_experts=256 // world_size,
            num_experts=256, top_k=8, hidden=2048,
            max_tokens_per_rank=10000,
        )
        if rank == 0:
            print(f"  V2 ElasticBuffer: replaced {n_replaced} layers, num_sms={v2_sms}")

    # If deepep_v2 mode, replace PrepareAndFinalize with V2 ElasticBuffer
    USE_V2 = int(os.environ.get("USE_DEEPEP_V2", "0"))
    if USE_V2:
        from deepep_v2_pf import replace_with_deepep_v2
        from vllm.distributed import get_ep_group
        ep = get_ep_group()
        n_replaced, v2_sms = replace_with_deepep_v2(
            model, ep.cpu_group,
            num_local_experts=256 // world_size,
            num_experts=256, top_k=8, hidden=2048,
            max_tokens_per_rank=10000,
        )
        if rank == 0:
            print(f"  V2 ElasticBuffer: replaced {n_replaced} layers, "
                  f"num_sms={v2_sms}")

    # 5 prompts with known answers (evaluated independently)
    EVAL_PROMPTS = [
        ("What is 17 * 23?", "391"),
        ("What is the square root of 144?", "12"),
        ("If a car travels at 60 km/h for 2.5 hours, how far does it go?", "150"),
        ("What is the sum of the first 10 positive integers?", "55"),
        ("What is 2^10?", "1024"),
    ]

    total = len(EVAL_PROMPTS)
    # Pad to be divisible by world_size
    while len(EVAL_PROMPTS) % world_size != 0:
        EVAL_PROMPTS.append(EVAL_PROMPTS[-1])  # duplicate last
    padded_total = len(EVAL_PROMPTS)
    local_bs = padded_total // world_size

    # Tokenize all prompts
    all_ids = []
    for prompt, _ in EVAL_PROMPTS:
        text = prompt
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True, tokenize=False)
        all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])

    mx = max(x.shape[0] for x in all_ids)
    pad_id = tokenizer.pad_token_id or 0
    padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
              if ids.shape[0] < mx else ids for ids in all_ids]
    input_ids_full = torch.stack(padded, dim=0)

    # Slice for this rank
    my_start = rank * local_bs
    my_input = input_ids_full[my_start:my_start + local_bs].to(device)
    prompt_len = my_input.shape[1]

    if rank == 0:
        print(f"Backend: {BACKEND}, GPUs: {world_size}, "
              f"prompts: {total} (padded: {padded_total}), local_bs: {local_bs}")

    decoder = ThresholdParallelDecoder(
        temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
    dllm = BlockDiffusionLLM(
        model, decoder,
        BlockIteratorFactory(use_block_diffusion=True),
        cache_factory=KVCacheFactory("prefix", is_bd_model=True),
        early_stop=True, maximum_unroll=4, expected_tpf=15,
        backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

    # Generate
    with torch.inference_mode():
        dllm.diff_iteration.num_forwards = 0
        local_out = dllm.generate(my_input.clone(), gen_length=128,
                                  block_length=BLOCK_LENGTH)
    torch.cuda.synchronize()
    dist.barrier()

    # Gather outputs
    gen_local = local_out[:, prompt_len:].contiguous()
    all_gen = [torch.zeros_like(gen_local) for _ in range(world_size)]
    dist.all_gather(all_gen, gen_local)

    if rank == 0:
        full_gen = torch.cat(all_gen, dim=0)
        print(f"\nForwards: {dllm.diff_iteration.num_forwards}")
        print(f"\n{'='*70}")
        print(f"Independent Quality Evaluation — Backend: {BACKEND}")
        print(f"{'='*70}")

        passed = 0
        for i in range(total):
            prompt, expected = EVAL_PROMPTS[i]
            gt = full_gen[i]
            valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
            text = tokenizer.decode(valid, skip_special_tokens=True)
            # Check if expected answer appears in the output
            found = expected in text
            status = "PASS" if found else "FAIL"
            if found:
                passed += 1
            print(f"\n  [{status}] Q: {prompt}")
            print(f"         Expected: {expected}")
            print(f"         Output (first 300 chars):")
            print(f"         {text[:300]}")

        print(f"\n{'='*70}")
        print(f"Result: {passed}/{total} PASS")
        print(f"{'='*70}")

dist.barrier()
dist.destroy_process_group()
