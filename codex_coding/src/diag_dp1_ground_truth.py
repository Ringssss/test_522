#!/usr/bin/env python3
"""dp=1 single GPU ground truth: per-layer norms, no EP, no communication."""
import os, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path("/home/wuhang/wuhang/dllm_wh/codex_coding/src")))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vllm import distributed as vllm_dist
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from dinfer.model import LLaDA2MoeModelLM
from transformers import AutoTokenizer, AutoConfig

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"

pcfg = ParallelConfig(tensor_parallel_size=1, data_parallel_size=1,
                      enable_expert_parallel=False)
vllm_cfg = VllmConfig(parallel_config=pcfg)

# Need distributed init even for single GPU
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29599")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

with set_current_vllm_config(vllm_cfg):
    vllm_dist.init_distributed_environment(1, 0, "env://", 0, "nccl")
    vllm_dist.initialize_model_parallel(tensor_model_parallel_size=1, backend="nccl")

with set_current_vllm_config(vllm_cfg):
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = LLaDA2MoeModelLM(config=config).eval()
    model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device="cuda")
    model = model.to("cuda")

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

    from baseline_optimizations import apply_all_optimizations
    apply_all_optimizations(model)

    layer_norms = {}
    def make_hook(idx):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            layer_norms[idx] = hs.detach().float().norm().item()
        return hook_fn
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(make_hook(i))

    prompt_text = "What is 17 * 23?"
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True, tokenize=False)
    tokens = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to("cuda")

    with torch.inference_mode():
        out = model(tokens, use_cache=False)
    torch.cuda.synchronize()

    logits_norm = (out.logits if hasattr(out, "logits") else out).detach().float().norm().item()

    print(f"\nBACKEND=none(dp=1)  GPUs=1  dtype=bf16")
    print(f"{'Layer':>6} | {'Norm':>12} | {'Type':>6}")
    print(f"{'-'*35}")
    for i in sorted(layer_norms.keys()):
        ltype = "dense" if i == 0 else "MoE"
        print(f"{'L'+str(i):>6} | {layer_norms[i]:>12.4f} | {ltype:>6}")
    print(f"{'logits':>6} | {logits_norm:>12.4f} |")
