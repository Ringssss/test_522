#!/usr/bin/env python3
"""
Diagnostic Step 2: Hook inside MoE layer to compare routing + expert output.
Captures MoE input/output at Layer 1 (first MoE layer) for both naive and DeepEP HT.

Usage:
  CUDA_VISIBLE_DEVICES=4,5 VLLM_ALL2ALL_BACKEND=naive \
    torchrun --nproc_per_node=2 codex_coding/src/diag_deepep_moe_detail.py

  CUDA_VISIBLE_DEVICES=4,5 VLLM_ALL2ALL_BACKEND=deepep_high_throughput \
    torchrun --nproc_per_node=2 codex_coding/src/diag_deepep_moe_detail.py
"""
from __future__ import annotations
import os, sys, torch
from pathlib import Path

REPO = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO / "codex_coding" / "src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BACKEND = os.environ.get("VLLM_ALL2ALL_BACKEND", "naive")

local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)

from vllm import distributed as vllm_dist
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from dinfer.model import LLaDA2MoeModelLM
from dinfer.model.modeling_llada2_moe import _maybe_patch_all_reduce
from transformers import AutoTokenizer, AutoConfig

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"

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

    # Apply fused_routing (same as bench)
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

    # ==========================================
    # Hook into Layer 1 MoE block to capture
    # (a) MoE input  (b) MoE output  (c) shared_expert output
    # ==========================================
    moe_captures = {}

    # Hook on LLaDA2MoeSparseMoeBlock (the MoE block, not the full decoder layer)
    target_layer_idx = 1  # First MoE layer
    moe_block = model.model.layers[target_layer_idx].mlp

    # Hook the experts (FusedMoE) forward to capture input/output
    def experts_hook(module, input, output):
        # input is (hidden_states, router_logits) via forward_impl
        # But we hook on the outer MoE block instead
        pass

    # Hook the whole MoE block
    orig_forward = moe_block.forward
    def patched_moe_forward(hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_flat = hidden_states.view(-1, h)

        # Capture input
        moe_captures['moe_input'] = hidden_flat.detach().clone()

        # Capture router_logits
        router_logits = moe_block.gate.get_logits(hidden_flat)
        moe_captures['router_logits'] = router_logits.detach().clone()

        # Run actual forward
        result = orig_forward(hidden_states)

        # Capture output
        moe_captures['moe_output'] = result.detach().clone()
        return result

    moe_block.forward = patched_moe_forward

    # Prepare input
    prompt_text = "What is the average speed if a train travels 240km at 80km/h and returns at 60km/h?"
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True, tokenize=False)
    tokens = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
    input_ids = tokens.unsqueeze(0).to(device)

    # Run forward
    from vllm.forward_context import set_forward_context
    with torch.inference_mode():
        with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                 num_tokens=input_ids.numel()):
            output = model(input_ids, use_cache=False)

    # Report
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"MoE Layer {target_layer_idx} Detail — Backend: {BACKEND}, Rank: {rank}")
        print(f"{'='*70}")

        for key in ['moe_input', 'router_logits', 'moe_output']:
            t = moe_captures[key].float()
            print(f"  {key:20s}: shape={list(t.shape)}, "
                  f"norm={t.norm().item():.4f}, "
                  f"mean={t.mean().item():.8f}, "
                  f"max={t.max().item():.4f}, "
                  f"min={t.min().item():.4f}")

        # Check router_logits top-k distribution
        rl = moe_captures['router_logits'].float()
        topk_vals, topk_idx = rl.topk(8, dim=-1)
        print(f"\n  Router logits topk (first 3 tokens):")
        for ti in range(min(3, topk_idx.shape[0])):
            print(f"    token {ti}: experts={topk_idx[ti].tolist()}, "
                  f"logits={topk_vals[ti].tolist()[:4]}...")

        # Diff between input and output to see MoE contribution
        moe_in = moe_captures['moe_input'].float()
        moe_out = moe_captures['moe_output'].float().view_as(moe_in)
        moe_contrib = moe_out - moe_in  # residual from shared_experts is added
        print(f"\n  MoE contribution (out - in):")
        print(f"    norm={moe_contrib.norm().item():.4f}, "
              f"mean={moe_contrib.mean().item():.8f}")

        # Final model output
        if hasattr(output, 'logits'):
            logits = output.logits.float()
        else:
            logits = output.float()
        print(f"\n  Final output: norm={logits.norm().item():.4f}, "
              f"mean={logits.mean().item():.6f}")

import torch.distributed as dist
dist.barrier()
dist.destroy_process_group()
