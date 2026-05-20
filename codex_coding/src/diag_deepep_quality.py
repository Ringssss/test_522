#!/usr/bin/env python3
"""
Per-layer hidden_states norm: single forward pass, all 20 layers.
naive AllToAll and DeepEP HT to find where quality diverges.

Usage (run twice, compare output):
  # Naive
  CUDA_VISIBLE_DEVICES=4,5 VLLM_ALL2ALL_BACKEND=naive \
    torchrun --nproc_per_node=2 codex_coding/src/diag_deepep_quality.py

  # DeepEP HT
  CUDA_VISIBLE_DEVICES=4,5 VLLM_ALL2ALL_BACKEND=deepep_high_throughput \
    torchrun --nproc_per_node=2 codex_coding/src/diag_deepep_quality.py
"""
from __future__ import annotations
import os, sys, torch
from pathlib import Path

REPO = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO / "codex_coding" / "src"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DEEPEP_DIAG"] = "1"  # Enable finalize diagnostics
BACKEND = os.environ.get("VLLM_ALL2ALL_BACKEND", "naive")
USE_FP32 = int(os.environ.get("USE_FP32", "0"))

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

# --- Two-phase init (same as bench_multi_gpu_dp.py) ---
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

    from vllm.distributed import get_ep_group
    ep = get_ep_group()
    if rank == 0:
        print(f"Backend: {BACKEND}, world_size={world_size}, "
              f"ep_size={ep.world_size}")

    # --- Load model ---
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                        local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                              local_files_only=True)
    model = LLaDA2MoeModelLM(config=config).eval()
    load_dtype = torch.float16 if USE_FP32 else torch.bfloat16
    model.load_weights(MODEL_PATH, torch_dtype=load_dtype, device=device)
    model = model.to(device)

    # Apply fused_routing (same as bench_multi_gpu_dp.py C5)
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

    # ★★★ Initialize DeepEP communication buffers for modular kernel
    from vllm.distributed import prepare_communication_buffer_for_model
    prepare_communication_buffer_for_model(model)

    # If deepep_v2 mode, replace PrepareAndFinalize with V2 ElasticBuffer
    USE_V2 = int(os.environ.get("USE_DEEPEP_V2", "0"))
    if USE_V2:
        from deepep_v2_pf import replace_with_deepep_v2
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

    # If V1_OPT mode, replace with async-optimized V1
    USE_V1_OPT = int(os.environ.get("USE_V1_OPT", "0"))
    if USE_V1_OPT:
        from deepep_v1_optimized_pf import replace_with_optimized_v1
        v1_sms = int(os.environ.get("V1_NUM_SMS", "0")) or None
        n_replaced = replace_with_optimized_v1(model, num_sms=v1_sms)
        if rank == 0:
            print(f"  V1 Optimized: replaced {n_replaced} layers"
                  + (f", num_sms={v1_sms}" if v1_sms else ""))

    # Check what kernel the FusedMoE layer uses
    if rank == 0:
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "FusedMoE":
                qm = mod.quant_method
                print(f"  quant_method type: {type(qm).__name__}")
                print(f"  quant_method attrs: {[a for a in dir(qm) if not a.startswith('_')]}")
                if hasattr(qm, 'fused_experts'):
                    fe = qm.fused_experts
                    print(f"  qm.fused_experts: {type(fe).__name__}")
                # Check the FusedMoE layer's own fused_experts
                if hasattr(mod, 'fused_experts'):
                    print(f"  mod.fused_experts: {type(mod.fused_experts).__name__}")
                # Check what apply() actually calls
                import inspect
                src = inspect.getsource(type(qm).apply)
                # Find 'fused_experts' in apply source
                for line in src.split('\n'):
                    if 'fused_experts' in line or 'FusedMoE' in line:
                        print(f"    apply src: {line.strip()}")
                print(f"  dp={mod.dp_size}, ep={mod.ep_size}, "
                      f"deepep_ht={mod.moe_parallel_config.use_deepep_ht_kernels}")
                break

    # --- Register hooks to capture hidden_states at every layer ---
    layer_norms = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            layer_norms[layer_idx] = hs.detach().float().norm().item()
        return hook_fn

    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(make_hook(i))

    # Same prompt on all ranks
    prompt_text = "What is 17 * 23?"
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True, tokenize=False)
    tokens = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    input_ids = tokens.to(device)

    from vllm.forward_context import set_forward_context
    with torch.inference_mode():
        with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                 num_tokens=input_ids.numel()):
            out = model(input_ids, use_cache=False)

    import torch.distributed as _dist
    _dist.barrier()

    if rank == 0:
        if hasattr(out, 'logits'):
            logits_norm = out.logits.detach().float().norm().item()
        else:
            logits_norm = out.detach().float().norm().item()

        dtype_str = "fp16" if USE_FP32 else "bf16"
        print(f"\nBACKEND={BACKEND}  GPUs={world_size}  dtype={dtype_str}")
        print(f"{'Layer':>6} | {'Norm':>12} | {'Type':>6}")
        print(f"{'-'*35}")
        for i in sorted(layer_norms.keys()):
            ltype = "dense" if i == 0 else "MoE"
            print(f"{'L'+str(i):>6} | {layer_norms[i]:>12.4f} | {ltype:>6}")
        print(f"{'logits':>6} | {logits_norm:>12.4f} |")

    import torch.distributed as __dist
    __dist.barrier()
    __dist.destroy_process_group()
