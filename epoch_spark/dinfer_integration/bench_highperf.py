#!/usr/bin/env python3
"""
Epoch-Spark High-Performance Benchmark.

Stacks: CUDA Graph + dtype-optimized routing + FusedRMSNorm + OptAttn.
Target: maximize single-GPU throughput toward 5000 tok/s.

Usage:
    cd /home/zhujianian/eurosys/dInfer
    /home/zhujianian/miniconda3/envs/crossstage/bin/python tests/bench_highperf.py
"""
import os, sys, time, json, socket, types, importlib.util
import torch

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
MASK_ID, EOS_ID = 156895, 156892
PROMPTS = [
    "Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h.",
    "Write a detailed essay about the history of artificial intelligence.",
    "Explain Le Chatelier's principle with examples.",
    "Design a REST API for an e-commerce platform.",
    "Analyze the economic impact of climate change.",
    "Explain quantum computing: qubits, superposition.",
    "Design a distributed message queue.",
    "Write a guide to training large language models.",
    "Compare TCP and UDP protocols.",
    "Explain the mathematical foundations of neural networks.",
    "Design a microservices architecture for ride-sharing.",
    "Write about the history of cryptography.",
    "Explain database indexing strategies.",
    "Discuss universal basic income with examples.",
    "Design a CI/CD pipeline for a monorepo.",
    "Explain relativity to an undergraduate.",
]

def setup():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if "deep_ep" not in sys.modules or getattr(sys.modules.get("deep_ep"), "__spec__", None) is None:
        sys.modules.pop("deep_ep", None)
        f = types.ModuleType("deep_ep"); f.__spec__ = importlib.util.spec_from_loader("deep_ep", loader=None); f.__path__ = []
        f.Buffer = type("Buffer", (), {"get_dispatch_config": staticmethod(lambda *a,**kw:None), "get_combine_config": staticmethod(lambda *a,**kw:None)})
        f.Config = type("Config", (), {}); f.EventOverlap = type("EventOverlap", (), {}); sys.modules["deep_ep"] = f


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


def apply_optimized_attn(model):
    """Optimized attention with fused QK norm and minimal dtype conversion."""
    from vllm.model_executor.layers.layernorm import rms_norm as vllm_rms_norm
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeSdpaAttention, apply_rotary_pos_emb, repeat_kv
    import torch.nn.functional as F

    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, LLaDA2MoeSdpaAttention):
            continue

        def make_fwd(attn):
            qnw, qne = attn.query_layernorm.weight, attn.query_layernorm.variance_epsilon
            knw, kne = attn.key_layernorm.weight, attn.key_layernorm.variance_epsilon
            nh, nkv, hd = attn.num_heads, attn.num_key_value_heads, attn.head_dim
            nkvg = nh // nkv

            def fwd(hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False,
                    position_embeddings=None, cache_position=None, replace_position=None, **kw):
                bsz, q_len, _ = hidden_states.size()
                qkv = attn.query_key_value(hidden_states)
                if isinstance(qkv, tuple): qkv = qkv[0]
                qkv = qkv.view(bsz, q_len, nh + 2*nkv, hd)
                q, k, v = qkv.split([nh, nkv, nkv], dim=-2)
                q = vllm_rms_norm(q, qnw, qne)
                k = vllm_rms_norm(k, knw, kne)
                q = q.transpose(1,2); k = k.transpose(1,2); v = v.transpose(1,2)
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                if past_key_value is not None:
                    if hasattr(past_key_value, 'update'):
                        try:
                            k, v = past_key_value.update(k, v, attn.layer_idx, replace_position)
                        except (AssertionError, TypeError):
                            # DynamicCache from transformers: update(k, v, layer_idx)
                            k, v = past_key_value.update(k, v, attn.layer_idx)
                    elif isinstance(past_key_value, (list, tuple)):
                        if len(past_key_value) > attn.layer_idx:
                            pk, pv = past_key_value[attn.layer_idx]
                            k = torch.cat([pk, k], dim=2)
                            v = torch.cat([pv, v], dim=2)
                pkv = (k, v) if use_cache else None

                if attention_mask is not None:
                    ke = repeat_kv(k, nkvg).contiguous()
                    ve = repeat_kv(v, nkvg).contiguous()
                    am = attention_mask.bool()
                    if am.dim() == 3: am = am.unsqueeze(1)
                    out = F.scaled_dot_product_attention(q.contiguous(), ke, ve, attn_mask=am, is_causal=False)
                else:
                    ke = repeat_kv(k, nkvg).contiguous()
                    ve = repeat_kv(v, nkvg).contiguous()
                    out = F.scaled_dot_product_attention(q.contiguous(), ke, ve, is_causal=False)
                out = out.transpose(1,2).contiguous().reshape(bsz, q_len, -1)
                dense_out = attn.dense(out)
                out = dense_out[0] if isinstance(dense_out, tuple) else dense_out
                return out, None, pkv
            return fwd

        module.forward = make_fwd(module)
        count += 1
    return count


def apply_dtype_optimized_moe(model):
    """MoE routing that minimizes dtype conversions — stay in bf16 as much as possible."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl

    count = 0
    for name, mod in model.named_modules():
        if mod.__class__.__name__ != "LLaDA2MoeSparseMoeBlock":
            continue

        gate = mod.gate
        experts = mod.experts
        shared = getattr(mod, 'shared_experts', None)
        local_E = experts.w13_weight.shape[0]
        global_E = gate.num_experts
        expert_map = getattr(experts, 'expert_map', None)
        if expert_map is not None:
            expert_map = expert_map.to(experts.w13_weight.device)

        def make_fwd(_gate, _exp, _shared, _emap, _gE):
            def fwd(hidden_states):
                res = _shared(hidden_states) if _shared is not None else 0
                bsz, seq_len, h = hidden_states.shape
                flat = hidden_states.view(-1, h)

                # Routing — minimize dtype conversions
                gating = _gate.get_logits(flat)
                # Stay in bf16 for sigmoid (avoid float32 roundtrip)
                scores = torch.sigmoid(gating)
                scores_routing = scores + _gate.expert_bias.to(scores.dtype)

                n_group = _gate.n_group
                topk_group = _gate.topk_group
                group_size = _gE // n_group

                grouped = scores_routing.view(-1, n_group, group_size)
                group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
                top_gidx = group_scores.topk(topk_group, dim=-1).indices

                gmask = torch.zeros(flat.shape[0], n_group, device=flat.device, dtype=scores.dtype)
                gmask.scatter_(1, top_gidx, 1.0)
                gmask = gmask.unsqueeze(2).expand(-1, -1, group_size).reshape(-1, _gE)

                masked = scores_routing * gmask
                _, topk_idx = masked.topk(_gate.top_k, dim=1)
                topk_w = torch.gather(scores, 1, topk_idx)
                topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
                topk_w = topk_w * _gate.routed_scaling_factor

                y = fused_experts_impl(
                    flat, _exp.w13_weight, _exp.w2_weight,
                    topk_w.float(), topk_idx, inplace=False, activation="silu",
                    global_num_experts=_gE, expert_map=_emap,
                ).to(flat.dtype)

                y = y.view(bsz, seq_len, h)
                if _shared is not None:
                    y = y + res
                return y
            return fwd

        mod.forward = make_fwd(gate, experts, shared, expert_map, global_E)
        count += 1
    return count


def prepare_batch(tokenizer, bs, device):
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(bs)]
    encoded = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in prompts]
    mx = max(e.shape[0] for e in encoded)
    padded = [torch.cat([torch.full((mx-e.shape[0],), MASK_ID, dtype=torch.long), e]) if e.shape[0]<mx else e for e in encoded]
    return torch.stack(padded).to(device)


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
    print(f"High-Perf Benchmark | LLaDA2.0-mini 16B | {torch.cuda.get_device_name(args.gpu)}")
    print(f"  CUDA Graph + FusedRMSNorm + OptAttn + dtype-optimized MoE routing")
    print("=" * 100)
    setup()

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]; sock.close()
    os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT",str(port))
    vcfg = VllmConfig(parallel_config=ParallelConfig(enable_expert_parallel=True))

    with set_current_vllm_config(vcfg):
        distributed.init_distributed_environment(1,0,"env://",0,"nccl")
        distributed.initialize_model_parallel(1,backend="nccl")

        from transformers import AutoConfig, AutoTokenizer
        from dinfer.model import LLaDA2MoeModelLM
        from dinfer import BlockWiseDiffusionLLM, BlockIteratorFactory, ThresholdParallelDecoder

        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        n1 = apply_fused_rmsnorm(model)
        n2 = apply_optimized_attn(model)
        n3 = apply_dtype_optimized_moe(model)
        print(f"[opt] FusedRMSNorm: {n1}, OptAttn: {n2}, DtypeOptMoE: {n3}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

        # Warmup
        with torch.inference_mode():
            w = torch.randint(0, 1000, (16, 210), device=device)
            for _ in range(3): model(w, use_cache=False)
        torch.cuda.synchronize()
        print("[opt] Warmup done")

        # === Test 1: Pure forward throughput (no diffusion loop) ===
        print("\n=== Pure Forward Throughput ===")
        for bs in bss:
            x = torch.randint(0, 1000, (bs, 210), device=device)
            # Warmup
            with torch.inference_mode():
                for _ in range(3): model(x, use_cache=False)
            torch.cuda.synchronize()
            # Benchmark
            t0 = time.perf_counter()
            with torch.inference_mode():
                for _ in range(10): model(x, use_cache=False)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 10 * 1000
            tps = bs * 210 / (ms / 1000)  # tokens processed per second
            gen_tps = bs * 128 / (ms / 1000 * 128 / 1)  # est generation throughput
            print(f"  bs={bs:3d}: {ms:.2f} ms/fwd, forward_tps={tps:.0f}, est_gen_tps={bs * 128 / (ms * 10 / 1000):.0f}")

        # === Test 2: CUDA Graph forward ===
        print("\n=== CUDA Graph Forward ===")
        for bs in [16, 64]:
            x_static = torch.randint(0, 1000, (bs, 210), device=device)
            with torch.inference_mode():
                for _ in range(3): model(x_static, use_cache=False)
                torch.cuda.synchronize()
                try:
                    g = torch.cuda.CUDAGraph()
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        model(x_static, use_cache=False)
                    torch.cuda.current_stream().wait_stream(s)
                    with torch.cuda.graph(g, stream=s):
                        out_static = model(x_static, use_cache=False)
                    torch.cuda.synchronize()
                    for _ in range(3): g.replay()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(20): g.replay()
                    torch.cuda.synchronize()
                    cg_ms = (time.perf_counter() - t0) / 20 * 1000
                    print(f"  bs={bs:3d}: {cg_ms:.2f} ms/fwd (CUDA Graph)")
                except Exception as e:
                    print(f"  bs={bs:3d}: CUDA Graph failed: {e}")

        # === Test 3: E2E generation ===
        print("\n=== E2E Generation (BlockWiseDiffusionLLM) ===")
        decoder = ThresholdParallelDecoder(temperature=0, threshold=0.9, mask_id=MASK_ID, eos_id=EOS_ID)
        for bs in bss:
            ids = prepare_batch(tokenizer, bs, device)
            dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)
            # Warmup
            with torch.inference_mode():
                dllm.generate(ids[:min(bs,4)].clone(), gen_length=32, block_length=32)
            torch.cuda.synchronize()
            # Timed
            dllm2 = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = dllm2.generate(ids.clone(), gen_length=args.gen_length, block_length=32)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            n = dllm2.num_forwards
            tps = bs * args.gen_length / dt
            ms = dt / n * 1000
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"  bs={bs:3d}: {ms:.2f} ms/fwd, {tps:.0f} tok/s, {n} fwds")
            print(f"          Output: {text[:120]}")

    print("\nDONE")


if __name__ == "__main__":
    main()
