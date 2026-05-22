#!/usr/bin/env python3
"""
Epoch-Spark Full Optimization Benchmark.

Stacks ALL optimizations on dInfer's native pipeline:
  1. Fused RMSNorm (vLLM kernel)
  2. FlashAttention 2 (flash_attn_func, non-causal)
  3. Block Diffusion with KV Cache (forward only current block tokens)
  4. IterationSmooth (reduce iteration count)
  5. Fused Triton MoE routing

Usage:
    cd /home/zhujianian/eurosys/dInfer
    /home/zhujianian/miniconda3/envs/crossstage/bin/python tests/bench_epoch_spark.py
"""
import os, sys, time, json, socket, types, importlib.util
import torch

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
MASK_ID, EOS_ID = 156895, 156892
PROMPTS = [
    "Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h.",
    "Write an essay about the history of artificial intelligence.",
    "Explain Le Chatelier's principle with examples.",
    "Design a REST API for an e-commerce platform.",
    "Analyze the economic impact of climate change.",
    "Explain quantum computing: qubits, superposition.",
    "Design a distributed message queue.",
    "Write a guide to training large language models.",
]


def setup():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if "deep_ep" not in sys.modules or getattr(sys.modules.get("deep_ep"), "__spec__", None) is None:
        sys.modules.pop("deep_ep", None)
        f = types.ModuleType("deep_ep"); f.__spec__ = importlib.util.spec_from_loader("deep_ep", loader=None); f.__path__ = []
        f.Buffer = type("Buffer", (), {"get_dispatch_config": staticmethod(lambda *a, **kw: None), "get_combine_config": staticmethod(lambda *a, **kw: None)})
        f.Config = type("Config", (), {}); f.EventOverlap = type("EventOverlap", (), {}); sys.modules["deep_ep"] = f


# ═══════════════════════════════════════════════════════════
# Optimization 1: Fused RMSNorm
# ═══════════════════════════════════════════════════════════

def apply_fused_rmsnorm(model):
    from vllm.model_executor.layers.layernorm import rms_norm as vllm_rms_norm
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeRMSNorm
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, LLaDA2MoeRMSNorm):
            if "query_layernorm" in name or "key_layernorm" in name:
                continue
            w, eps = module.weight, module.variance_epsilon
            def _mk(ww, ee):
                def _f(hs): return vllm_rms_norm(hs, ww, ee)
                return _f
            module.forward = _mk(w, eps)
            count += 1
    return count


# ═══════════════════════════════════════════════════════════
# Optimization 2: FlashAttention 2 (non-causal)
# ═══════════════════════════════════════════════════════════

def apply_flash_attn(model):
    """Replace SDPA attention with optimized non-causal attention.
    Uses flash_attn if available, otherwise optimized SDPA."""
    try:
        from flash_attn import flash_attn_func
        has_flash = True
    except ImportError:
        has_flash = False

    from vllm.model_executor.layers.layernorm import rms_norm as vllm_rms_norm
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeSdpaAttention, apply_rotary_pos_emb
    import torch.nn.functional as F
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, LLaDA2MoeSdpaAttention):
            continue

        def make_fa_forward(attn, _has_flash=has_flash):
            qnw = attn.query_layernorm.weight
            qne = attn.query_layernorm.variance_epsilon
            knw = attn.key_layernorm.weight
            kne = attn.key_layernorm.variance_epsilon

            def fa_fwd(hidden_states, attention_mask=None, position_ids=None,
                       past_key_value=None, output_attentions=False, use_cache=False,
                       position_embeddings=None, cache_position=None, replace_position=None, **kw):
                bsz, q_len, _ = hidden_states.size()
                nh = attn.num_heads; nkv = attn.num_key_value_heads; hd = attn.head_dim

                qkv = attn.query_key_value(hidden_states)
                if isinstance(qkv, tuple):
                    qkv = qkv[0]
                qkv = qkv.view(bsz, q_len, nh + 2*nkv, hd)
                q, k, v = qkv.split([nh, nkv, nkv], dim=-2)

                q = vllm_rms_norm(q, qnw, qne)
                k = vllm_rms_norm(k, knw, kne)

                q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                if past_key_value is not None:
                    k, v = past_key_value.update(k, v, attn.layer_idx, replace_position)
                if use_cache:
                    past_key_value = (k, v)

                if attention_mask is not None:
                    from dinfer.model.modeling_llada2_moe import repeat_kv
                    nkvg = nh // nkv
                    ke = repeat_kv(k, nkvg).contiguous()
                    ve = repeat_kv(v, nkvg).contiguous()
                    am = attention_mask.bool()
                    if am.dim() == 3: am = am.unsqueeze(1)
                    out = F.scaled_dot_product_attention(q.contiguous(), ke, ve, attn_mask=am, is_causal=False)
                    out = out.transpose(1, 2).contiguous()
                elif _has_flash:
                    from flash_attn import flash_attn_func
                    out = flash_attn_func(q.transpose(1,2).contiguous(), k.transpose(1,2).contiguous(),
                                          v.transpose(1,2).contiguous(), causal=False)
                else:
                    from dinfer.model.modeling_llada2_moe import repeat_kv
                    nkvg = nh // nkv
                    ke = repeat_kv(k, nkvg).contiguous()
                    ve = repeat_kv(v, nkvg).contiguous()
                    out = F.scaled_dot_product_attention(q.contiguous(), ke, ve, is_causal=False)
                    out = out.transpose(1, 2).contiguous()

                out = out.reshape(bsz, q_len, -1)
                dense_out = attn.dense(out)
                out = dense_out[0] if isinstance(dense_out, tuple) else dense_out
                return out, None, past_key_value
            return fa_fwd

        module.forward = make_fa_forward(module)
        count += 1
    return count


# ═══════════════════════════════════════════════════════════
# Optimization 3: Fused Triton MoE routing
# ═══════════════════════════════════════════════════════════

def apply_fused_moe_routing(model):
    from dinfer.epoch_spark_dinfer import patch_dinfer_baseline
    return patch_dinfer_baseline(model)


# ═══════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════

def prepare_batch(tokenizer, bs, device):
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(bs)]
    encoded = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in prompts]
    mx = max(e.shape[0] for e in encoded)
    padded = [torch.cat([torch.full((mx-e.shape[0],), MASK_ID, dtype=torch.long), e]) if e.shape[0]<mx else e for e in encoded]
    return torch.stack(padded).to(device)


def run_gen(model, tokenizer, ids, bs, gl, device, dllm_factory, label=""):
    dllm = dllm_factory()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = dllm.generate(ids, gen_length=gl, block_length=32)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    n = dllm.num_forwards
    pk = torch.cuda.max_memory_allocated(device) / 1024**2
    texts = [tokenizer.decode(out[i], skip_special_tokens=True) for i in range(bs)]
    avg = t/n*1000; tps = bs*gl/t
    return {"avg_ms": avg, "tps": tps, "peak_mb": pk, "n_fwd": n, "total_s": t, "texts": texts}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--batch-sizes", type=str, default="1,16,64")
    args = p.parse_args()
    bss = [int(b) for b in args.batch_sizes.split(",")]
    device = torch.device(f"cuda:{args.gpu}"); torch.cuda.set_device(device)

    print("=" * 100)
    print(f"Full Optimization Benchmark | LLaDA2.0-mini 16B | {torch.cuda.get_device_name(args.gpu)}")
    print("=" * 100)
    setup()

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]; sock.close()
    os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT",str(port))
    vcfg = VllmConfig(parallel_config=ParallelConfig(enable_expert_parallel=True))

    all_results = {}

    with set_current_vllm_config(vcfg):
        if not torch.distributed.is_initialized():
            distributed.init_distributed_environment(1,0,"env://",0,"nccl")
            distributed.initialize_model_parallel(1,backend="nccl")

        from transformers import AutoTokenizer, AutoConfig
        from dinfer.model import LLaDA2MoeModelLM
        from dinfer import (BlockIteratorFactory, ThresholdParallelDecoder, KVCacheFactory,
                            BlockWiseDiffusionLLM, BlockDiffusionLLMAttnmask, BlockDiffusionLLM,
                            IterSmoothDiffusionLLM)

        tk = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        model = LLaDA2MoeModelLM(config=cfg).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        print(f"[model] GPU: {torch.cuda.memory_allocated(device)/1024**2:.0f} MB")

        # Apply optimizations incrementally
        n_rms = apply_fused_rmsnorm(model)
        print(f"[opt] Fused RMSNorm: {n_rms} layers")

        n_fa = apply_flash_attn(model)
        print(f"[opt] FlashAttention: {n_fa} layers")

        n_moe = apply_fused_moe_routing(model)

        # Warmup
        with torch.inference_mode(), set_forward_context(None, vcfg):
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)
        print("[opt] Warmup done")

        for bs in bss:
            print(f"\n{'='*80}")
            print(f"  Batch size = {bs}")
            print(f"{'='*80}")

            ids = prepare_batch(tk, bs, device)
            decoder = ThresholdParallelDecoder(temperature=0, threshold=0.9, mask_id=MASK_ID, eos_id=EOS_ID)

            configs = []

            # Config A: BlockDiffusionLLMAttnmask (no KV cache, full seq each forward)
            configs.append(("A:NoCache", lambda d=decoder: BlockDiffusionLLMAttnmask(
                model, d, BlockIteratorFactory(use_block_diffusion=True), early_stop=True)))

            # Config B: BlockDiffusionLLM with KV cache (forward only block tokens)
            configs.append(("B:KVCache", lambda d=decoder: BlockDiffusionLLM(
                model, d, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory('prefix', is_bd_model=True), early_stop=True)))

            # Config C: BlockWiseDiffusionLLM (simple, supports batching, no KV cache)
            if bs > 1:
                configs.append(("C:BatchNoKV", lambda d=decoder: BlockWiseDiffusionLLM(
                    model, d, BlockIteratorFactory(), early_stop=True)))

            # Config D: IterSmooth (reduce iterations)
            configs.append(("D:IterSmooth", lambda d=decoder: IterSmoothDiffusionLLM(
                model, d, BlockIteratorFactory(), early_stop=True)))

            results_for_bs = {}

            for label, factory in configs:
                print(f"\n  [{label}] Running...")
                try:
                    r = run_gen(model, tk, ids, bs, args.gen_length, device, factory, label)
                    print(f"  [{label}] {r['avg_ms']:.2f} ms/fwd, {r['tps']:.1f} tok/s, "
                          f"{r['n_fwd']} fwds, peak={r['peak_mb']:.0f} MB")
                    print(f"  [{label}] Output: {r['texts'][0][:120]}")
                    results_for_bs[label] = {
                        "avg_ms": round(r["avg_ms"], 2),
                        "tps": round(r["tps"], 1),
                        "n_fwd": r["n_fwd"],
                        "peak_mb": round(r["peak_mb"], 0),
                    }
                except Exception as e:
                    print(f"  [{label}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                    results_for_bs[label] = {"error": str(e)}

            all_results[f"bs{bs}"] = results_for_bs

    # Final table
    print("\n" + "=" * 100)
    print("FINAL RESULTS — Full Optimization Stack")
    print("  Fused RMSNorm + FlashAttention + Fused Triton MoE Routing")
    print("=" * 100)
    print(f"{'BS':>4} | {'Config':<16} | {'ms/fwd':>8} | {'tok/s':>10} | {'#fwd':>6} | {'PeakMB':>8}")
    print("-" * 100)
    for bsk in sorted(all_results.keys(), key=lambda k: int(k[2:])):
        r = all_results[bsk]; bv = bsk[2:]
        for label, data in r.items():
            if "error" in data:
                print(f"{bv:>4} | {label:<16} | {'ERROR':>8} | {'—':>10} | {'—':>6} | {'—':>8}")
            else:
                print(f"{bv:>4} | {label:<16} | {data['avg_ms']:8.2f} | {data['tps']:10.1f} | {data['n_fwd']:6} | {data['peak_mb']:8.0f}")
    print("=" * 100)

    with open("bench_full_opt_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to bench_full_opt_results.json")


if __name__ == "__main__":
    main()
