#!/usr/bin/env python3
"""
v0.1.15.8a — Component-level profiling for C5 and C10

Instruments MoE sub-components with CUDA events + NVTX markers:
  shared_experts | gate.get_logits | EB (cold/hot) | routing | fused_experts | residual
Also instruments: Attention, RMSNorm (pre/post), Embedding, LMHead

Config: gen_length=64 (2 blocks), batch=32, block=32, threshold=0.90, temp=0

Usage:
  # Python-only timing:
  CUDA_VISIBLE_DEVICES=4 conda run -n dllm python codex_coding/src/nsys_c5_c10_profiling.py --config c5
  CUDA_VISIBLE_DEVICES=4 conda run -n dllm python codex_coding/src/nsys_c5_c10_profiling.py --config c10

  # With nsys:
  CUDA_VISIBLE_DEVICES=4 nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx \
    -o codex_coding/results/c5_nsys_profile \
    conda run -n dllm python codex_coding/src/nsys_c5_c10_profiling.py --config c5
"""

from __future__ import annotations
import argparse, os, sys, time, socket, json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

from test_fused_eb_triton import (
    fused_routing, FusedEBController,
    _kernel_A, _kernel_A_cold, _kernel_B_v2,
    _kernel_C, _kernel_D_v2, _kernel_zero_init,
)
from baseline_optimizations import apply_all_optimizations

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
GEN_LENGTH = 64  # 2 blocks

PROMPTS = [
    "Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h. The total distance between the two cities is 240 km. What is the average speed for the entire round trip?\n\nProblem 2: A rectangular garden has a perimeter of 56 meters.",
    "Write a detailed essay about the history of artificial intelligence, covering the Dartmouth conference of 1956, the AI winters, the rise of machine learning in the 1990s, and deep learning breakthroughs.",
    "You are a chemistry professor. Explain Le Chatelier's principle with examples and how it applies to industrial ammonia production via the Haber process.",
    "Design a complete REST API for an e-commerce platform with endpoints for user authentication, product management, shopping cart operations, and order processing.",
    "Analyze the global economic impact of climate change across agriculture, energy, real estate, and healthcare sectors with specific examples.",
    "Explain quantum computing to a classical CS background: qubits, superposition, entanglement, Shor's algorithm, and current hardware approaches.",
    "You are a systems architect. Design a distributed message queue with partition-based storage, consumer groups, replication, and exactly-once semantics.",
    "Write a comprehensive guide to training large language models covering data collection, tokenizer training, architecture decisions, and distributed training strategies.",
    "Solve the quadratic equation x^2 - 5x + 6 = 0 step by step. Show the factoring method, then verify both solutions by substituting them back into the original equation.",
    "Explain the mathematical foundations of neural networks: backpropagation, gradient descent, loss functions, and the universal approximation theorem.",
    "Design a microservices architecture for a ride-sharing application with real-time matching, pricing, routing, payments, and driver management.",
    "Write about the history of cryptography from Caesar ciphers through RSA, elliptic curve cryptography, and post-quantum cryptographic algorithms.",
    "Explain database indexing strategies: B-trees, hash indexes, bitmap indexes, and their trade-offs for OLTP vs OLAP workloads.",
    "Solve this logic puzzle step by step: If A is true, then B is true. If B is true, then C is true. A is true. What can we conclude about B and C? Then, if D is true only when both B and C are true, what can we conclude about D?",
    "Design a CI/CD pipeline for a large monorepo with microservices, including build caching, parallel testing, canary deployments, and rollback strategies.",
    "Explain the theory of relativity to a physics undergraduate, covering special relativity, time dilation, length contraction, and general relativity basics.",
    "Write a comprehensive comparison of Python, Rust, and Go for systems programming, covering memory safety, concurrency models, and ecosystem maturity.",
    "Design a real-time recommendation engine for a video streaming platform that handles cold start, user preferences, and content diversity.",
    "Explain the CAP theorem and its practical implications for distributed database design, with examples from Cassandra, MongoDB, and CockroachDB.",
    "Write a Python function to compute the nth Fibonacci number. Show the function, then compute fib(1) through fib(10) step by step and list all 10 values.",
    "Design a fraud detection system for a payment processing company using machine learning, rule engines, and real-time streaming analytics.",
    "Explain compiler optimization techniques including SSA form, loop unrolling, vectorization, and register allocation strategies.",
    "Write about the history and future of space exploration, from Apollo missions through SpaceX reusability to planned Mars colonization.",
    "Design an observability platform with distributed tracing, log aggregation, metrics collection, and intelligent alerting for microservices.",
    "Explain the mathematics behind public key cryptography, including modular arithmetic, Euler's theorem, and the RSA algorithm step by step.",
    "Write a guide to modern CSS layout techniques including Flexbox, Grid, Container Queries, and responsive design best practices.",
    "Design a multi-tenant SaaS platform architecture with data isolation, custom domains, billing integration, and horizontal scaling.",
    "Explain how garbage collectors work in JVM, Go, and Python, comparing mark-sweep, generational, and reference counting approaches.",
    "List all 8 planets in our solar system in order from closest to farthest from the Sun. For each planet, state whether it is a terrestrial or gas/ice giant planet, and give its approximate orbital period in Earth years.",
    "Design a real-time collaborative document editor like Google Docs with conflict resolution, offline support, and version history.",
    "Explain operating system memory management: virtual memory, page tables, TLB, demand paging, and memory-mapped files.",
    "Write a comprehensive guide to Kubernetes architecture including pods, services, ingress, operators, and cluster autoscaling.",
]


# ================================================================
# CUDA Event Timer (zero-overhead recording, read after sync)
# ================================================================
class ComponentTimer:
    def __init__(self):
        self._stack = []
        self.data = defaultdict(list)  # tag -> [(start_ev, end_ev), ...]

    def start(self, tag):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._stack.append((tag, ev))

    def end(self):
        tag, start_ev = self._stack.pop()
        end_ev = torch.cuda.Event(enable_timing=True)
        end_ev.record()
        self.data[tag].append((start_ev, end_ev))

    def reset(self):
        self._stack.clear()
        self.data.clear()

    def summarize(self):
        torch.cuda.synchronize()
        result = {}
        for tag in sorted(self.data.keys()):
            pairs = self.data[tag]
            times = [s.elapsed_time(e) for s, e in pairs]
            result[tag] = {
                'count': len(times),
                'total_ms': sum(times),
                'avg_ms': sum(times) / len(times) if times else 0,
            }
        return result


# ================================================================
# Instrumented EB Controller (NVTX on sub-kernels)
# ================================================================
class InstrumentedEBController(FusedEBController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_was_cold = {}

    def get_s_mask(self, layer_idx, logits, bias):
        N = logits.shape[0]
        is_cold = self.is_new_block(layer_idx, N)
        self.last_was_cold[layer_idx] = is_cold
        if is_cold:
            return self.cold_path(layer_idx, logits, bias)
        else:
            return self.hot_path(layer_idx, logits, bias)

    def hot_path(self, layer_idx, logits, bias):
        nvtx = torch.cuda.nvtx
        N, E = logits.shape
        K_init = self.K_init.get(layer_idx, 142)
        b = self._get_bufs(N, E, logits.device)
        lf = logits.float()

        nvtx.range_push(f"EB_L{layer_idx}_KA")
        _kernel_A[(N,)](lf, bias.float(), b['pop'],
                        N, self.rsf, lf.stride(0), lf.stride(1),
                        E=E, KEXT=self.K_ext, KEXT_PAD=16)
        nvtx.range_pop()

        nvtx.range_push(f"EB_L{layer_idx}_KB")
        _kernel_B_v2[(1,)](b['pop'], b['s_mask'], K_init, E=E, MAX_K=256)
        nvtx.range_pop()

        self.hot_count += 1
        return b['s_mask']

    def cold_path(self, layer_idx, logits, bias):
        nvtx = torch.cuda.nvtx
        N, E = logits.shape
        b = self._get_bufs(N, E, logits.device)

        nvtx.range_push(f"EB_L{layer_idx}_zero_init")
        _kernel_zero_init[(1,)](b['pop'], b['sat_flag'], b['sat_count'],
                                b['G'], b['H'], E=E)
        nvtx.range_pop()

        lf = logits.float()
        bf = bias.float()

        nvtx.range_push(f"EB_L{layer_idx}_KA_cold")
        _kernel_A_cold[(N,)](lf, bf, b['pop'], b['topkm_idx'], b['topkm_w'], b['r'],
                             N, self.rsf, self.quality_floor,
                             lf.stride(0), lf.stride(1),
                             b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                             E=E, KEXT=self.K_ext, KEXT_PAD=16, K=self.K)
        nvtx.range_pop()

        nvtx.range_push(f"EB_L{layer_idx}_KB_cold")
        _kernel_B_v2[(1,)](b['pop'], b['s_mask'], self.K_target, E=E, MAX_K=256)
        nvtx.range_pop()

        q_major_x1000 = int(self.q_major * 1000)
        nvtx.range_push(f"EB_L{layer_idx}_KCDC_loop")
        for _ in range(self.MAX_ROUNDS):
            _kernel_C[(N,)](b['topkm_idx'], b['topkm_w'], b['r'],
                           b['s_mask'], b['sat_flag'],
                           b['sat_count'], b['G'], b['H'],
                           N, b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                           E=E, KEXT=self.K_ext, KEXT_PAD=16)
            _kernel_D_v2[(1,)](b['s_mask'], b['sat_flag'], b['sat_count'],
                              b['G'], b['H'], N, q_major_x1000,
                              E=E, CAP=self.cap)
        nvtx.range_pop()

        self.K_init[layer_idx] = self.K_target + self.MAX_ROUNDS * self.cap
        self.cold_count += 1
        return b['s_mask']


# ================================================================
# Peek cold/hot without updating state
# ================================================================
def peek_is_cold(ctrl, layer_idx, N):
    prev = ctrl.prev_N.get(layer_idx, -1)
    return prev == -1 or N > prev


# ================================================================
# Install all instrumentation
# ================================================================
def install_instrumentation(model, timer, config_name, eb_ctrl=None):
    nvtx = torch.cuda.nvtx
    hooks = []

    def hook_mod(mod, tag):
        def pre(m, inp):
            nvtx.range_push(tag)
            timer.start(tag)
        def post(m, inp, out):
            timer.end()
            nvtx.range_pop()
        hooks.append(mod.register_forward_pre_hook(pre))
        hooks.append(mod.register_forward_hook(post))

    # Embedding
    emb = getattr(model.model, 'word_embeddings',
                  getattr(model.model, 'embed_tokens', None))
    if emb:
        hook_mod(emb, "Embedding")

    for li, layer in enumerate(model.model.layers):
        is_moe = hasattr(layer.mlp, 'gate')

        hook_mod(layer.input_layernorm, f"RMSNorm_pre_L{li}")
        attn = layer.attention if hasattr(layer, 'attention') else layer.self_attn
        hook_mod(attn, f"Attention_L{li}")
        hook_mod(layer.post_attention_layernorm, f"RMSNorm_post_L{li}")

        if not is_moe:
            hook_mod(layer.mlp, f"DenseMLP_L{li}")
            continue

        # --- MoE layer: monkey-patch forward + routing ---
        moe = layer.mlp
        gate = moe.gate
        g_bias = gate.expert_bias
        g_rsf = gate.routed_scaling_factor
        g_tk, g_ng, g_tkg = gate.top_k, gate.n_group, gate.topk_group

        # Patch gate.routing
        def make_routing_fn(bias, rsf, tk, ng, tkg, layer_i, ctrl):
            def fn(hidden_states, gating_output, topk, renormalize):
                if ctrl is not None:
                    N = gating_output.shape[0]
                    is_cold = peek_is_cold(ctrl, layer_i, N)
                    tag = f"EB_cold_L{layer_i}" if is_cold else f"EB_hot_L{layer_i}"
                    nvtx.range_push(tag)
                    timer.start(tag)
                    s_mask = ctrl.get_s_mask(layer_i, gating_output, bias)
                    timer.end()
                    nvtx.range_pop()
                else:
                    s_mask = None

                nvtx.range_push(f"routing_L{layer_i}")
                timer.start(f"routing_L{layer_i}")
                w, idx = fused_routing(gating_output, bias, rsf,
                                       s_mask=s_mask, K=tk, ng=ng, tkg=tkg)
                timer.end()
                nvtx.range_pop()
                return w.to(gating_output.dtype), idx
            return fn

        gate.routing = make_routing_fn(
            g_bias, g_rsf, g_tk, g_ng, g_tkg, li,
            eb_ctrl if config_name == "C10" else None)

        # Patch MoE block forward
        def make_moe_fwd(blk, layer_i):
            def fwd(hidden_states):
                nvtx.range_push(f"shared_L{layer_i}")
                timer.start(f"shared_L{layer_i}")
                res = blk.shared_experts(hidden_states)
                timer.end()
                nvtx.range_pop()

                bsz, seq_len, h = hidden_states.shape
                hs_flat = hidden_states.view(-1, h)

                nvtx.range_push(f"getlogits_L{layer_i}")
                timer.start(f"getlogits_L{layer_i}")
                logits = blk.gate.get_logits(hs_flat)
                timer.end()
                nvtx.range_pop()

                nvtx.range_push(f"fwd_impl_L{layer_i}")
                timer.start(f"fwd_impl_L{layer_i}")
                y = blk.experts.forward_impl(
                    hidden_states=hs_flat, router_logits=logits)
                timer.end()
                nvtx.range_pop()

                y = y.view(bsz, seq_len, h)

                nvtx.range_push(f"residual_L{layer_i}")
                timer.start(f"residual_L{layer_i}")
                y = y + res
                timer.end()
                nvtx.range_pop()

                return y
            return fwd

        moe.forward = make_moe_fwd(moe, li)

    # Final RMSNorm + LMHead
    hook_mod(model.model.norm, "FinalRMSNorm")
    hook_mod(model.lm_head, "LMHead")

    return hooks


# ================================================================
# Analyze and print results
# ================================================================
def analyze(raw_stats, config_name, total_time_s, total_fwd, eb_ctrl=None):
    total_ms = total_time_s * 1000

    # Aggregate by component type
    agg = defaultdict(lambda: {'count': 0, 'total_ms': 0.0})
    for tag, s in raw_stats.items():
        if tag.startswith("shared_L"):       comp = "shared_experts"
        elif tag.startswith("getlogits_L"):  comp = "gate_getlogits"
        elif tag.startswith("EB_cold_L"):    comp = "EB_cold"
        elif tag.startswith("EB_hot_L"):     comp = "EB_hot"
        elif tag.startswith("routing_L"):    comp = "routing"
        elif tag.startswith("fwd_impl_L"):   comp = "fwd_impl"
        elif tag.startswith("residual_L"):   comp = "residual"
        elif tag.startswith("Attention_L"):  comp = "Attention"
        elif tag.startswith("RMSNorm_pre_L"):  comp = "RMSNorm_pre"
        elif tag.startswith("RMSNorm_post_L"): comp = "RMSNorm_post"
        elif tag.startswith("DenseMLP_L"):   comp = "DenseMLP"
        elif tag == "Embedding":    comp = "Embedding"
        elif tag == "FinalRMSNorm": comp = "FinalRMSNorm"
        elif tag == "LMHead":       comp = "LMHead"
        else:                        comp = "other"
        agg[comp]['count'] += s['count']
        agg[comp]['total_ms'] += s['total_ms']

    # Derived: fused_experts = fwd_impl - EB - routing
    eb_total = agg['EB_cold']['total_ms'] + agg['EB_hot']['total_ms']
    routing_ms = agg['routing']['total_ms']
    fwd_impl_ms = agg['fwd_impl']['total_ms']
    fused_experts_ms = max(0, fwd_impl_ms - eb_total - routing_ms)

    # MoE subtotal = shared + getlogits + fwd_impl + residual
    moe_sub = (agg['shared_experts']['total_ms'] + agg['gate_getlogits']['total_ms']
               + fwd_impl_ms + agg['residual']['total_ms'])

    # Non-MoE
    non_moe = (agg['Attention']['total_ms'] + agg['RMSNorm_pre']['total_ms']
               + agg['RMSNorm_post']['total_ms'] + agg['DenseMLP']['total_ms']
               + agg['Embedding']['total_ms'] + agg['FinalRMSNorm']['total_ms']
               + agg['LMHead']['total_ms'])

    instrumented = moe_sub + non_moe
    gap = total_ms - instrumented

    # Print
    print(f"\n{'='*80}")
    print(f"COMPONENT BREAKDOWN — {config_name}")
    print(f"  Wall-clock: {total_time_s:.3f}s ({total_ms:.1f}ms) | "
          f"Fwd: {total_fwd} | ms/fwd: {total_ms/total_fwd:.2f}")
    if eb_ctrl:
        print(f"  EB cold: {eb_ctrl.cold_count} | EB hot: {eb_ctrl.hot_count}")
    print(f"{'='*80}")

    rows = []
    rows.append(("=== MoE (L1-L19) ===", None))
    rows.append(("  shared_experts", agg['shared_experts']['total_ms']))
    rows.append(("  gate_getlogits", agg['gate_getlogits']['total_ms']))
    if config_name == "C10":
        rows.append(("  EB_cold", agg['EB_cold']['total_ms']))
        rows.append(("  EB_hot", agg['EB_hot']['total_ms']))
        rows.append(("  EB_total", eb_total))
    rows.append(("  routing (Triton)", routing_ms))
    rows.append(("  fused_experts *", fused_experts_ms))
    rows.append(("  fwd_impl (rout+exp)", fwd_impl_ms))
    rows.append(("  residual_add", agg['residual']['total_ms']))
    rows.append(("  --- MoE subtotal ---", moe_sub))
    rows.append(("", None))
    rows.append(("=== Non-MoE ===", None))
    rows.append(("  Attention", agg['Attention']['total_ms']))
    rows.append(("  RMSNorm_pre", agg['RMSNorm_pre']['total_ms']))
    rows.append(("  RMSNorm_post", agg['RMSNorm_post']['total_ms']))
    rows.append(("  DenseMLP (L0)", agg['DenseMLP']['total_ms']))
    rows.append(("  Embedding", agg['Embedding']['total_ms']))
    rows.append(("  FinalRMSNorm", agg['FinalRMSNorm']['total_ms']))
    rows.append(("  LMHead", agg['LMHead']['total_ms']))
    rows.append(("  --- Non-MoE subtotal ---", non_moe))
    rows.append(("", None))
    rows.append(("Instrumented total", instrumented))
    rows.append(("Gap (Python/decoder)", gap))

    print(f"\n  {'Component':<28s} {'Total(ms)':>10s} {'Avg/fwd':>10s} "
          f"{'%wall':>7s} {'%instr':>7s}")
    print(f"  {'-'*63}")
    for name, ms in rows:
        if ms is None:
            print(f"  {name}")
            continue
        if not name:
            print()
            continue
        pw = ms / total_ms * 100
        pi = ms / instrumented * 100 if instrumented > 0 else 0
        af = ms / total_fwd
        print(f"  {name:<28s} {ms:>10.1f} {af:>10.3f} {pw:>6.1f}% {pi:>6.1f}%")

    # Build per-layer detail for fused_experts
    per_layer = {}
    for li in range(20):
        fi_key = f"fwd_impl_L{li}"
        rt_key = f"routing_L{li}"
        eb_c_key = f"EB_cold_L{li}"
        eb_h_key = f"EB_hot_L{li}"
        if fi_key in raw_stats:
            fi = raw_stats[fi_key]['total_ms']
            rt = raw_stats.get(rt_key, {}).get('total_ms', 0)
            ec = raw_stats.get(eb_c_key, {}).get('total_ms', 0)
            eh = raw_stats.get(eb_h_key, {}).get('total_ms', 0)
            fe = fi - rt - ec - eh
            per_layer[li] = {
                'fwd_impl_ms': fi, 'routing_ms': rt,
                'eb_cold_ms': ec, 'eb_hot_ms': eh,
                'fused_experts_ms': fe,
            }

    result = {
        'config': config_name, 'gen_length': GEN_LENGTH,
        'total_time_s': total_time_s, 'total_fwd': total_fwd,
        'ms_per_fwd': total_ms / total_fwd,
        'components': {
            'shared_experts': agg['shared_experts']['total_ms'],
            'gate_getlogits': agg['gate_getlogits']['total_ms'],
            'EB_cold': agg['EB_cold']['total_ms'],
            'EB_hot': agg['EB_hot']['total_ms'],
            'EB_total': eb_total,
            'routing': routing_ms,
            'fused_experts_derived': fused_experts_ms,
            'fwd_impl': fwd_impl_ms,
            'residual': agg['residual']['total_ms'],
            'MoE_subtotal': moe_sub,
            'Attention': agg['Attention']['total_ms'],
            'RMSNorm_pre': agg['RMSNorm_pre']['total_ms'],
            'RMSNorm_post': agg['RMSNorm_post']['total_ms'],
            'DenseMLP': agg['DenseMLP']['total_ms'],
            'Embedding': agg['Embedding']['total_ms'],
            'FinalRMSNorm': agg['FinalRMSNorm']['total_ms'],
            'LMHead': agg['LMHead']['total_ms'],
            'non_MoE_subtotal': non_moe,
            'instrumented_total': instrumented,
            'gap': gap,
        },
        'per_layer': per_layer,
        'eb_cold_count': eb_ctrl.cold_count if eb_ctrl else 0,
        'eb_hot_count': eb_ctrl.hot_count if eb_ctrl else 0,
        'raw_stats': raw_stats,
    }
    return result


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, choices=['c5', 'c10'])
    args = parser.parse_args()
    config_name = args.config.upper()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoTokenizer, AutoConfig

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print(f"Component Profiling — {config_name}")
    print(f"  gen_length={GEN_LENGTH}, batch=32, block=32, threshold=0.90")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0),
                      use_cache=False)

        print("\nApplying baseline optimizations...")
        n_rms, n_fa = apply_all_optimizations(model)
        print(f"  Fused RMSNorm: {n_rms}, Flash-attn: {n_fa}")

        # Build input (HetEval-32)
        BATCH_SIZE = 32
        all_ids = []
        for i in range(BATCH_SIZE):
            text = PROMPTS[i]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=False)
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                  if ids.shape[0] < mx else ids for ids in all_ids]
        input_ids = torch.stack(padded, dim=0).to(device)
        print(f"  Input shape: {input_ids.shape}")

        decoder = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True,
                inplace_cache_update=True)

        # Create timer and EB controller
        timer = ComponentTimer()
        eb_ctrl = None
        if config_name == "C10":
            eb_ctrl = InstrumentedEBController(
                num_layers=19, K=8, M=4, K_target=40,
                quality_floor=0.70, q_major=0.95, per_round_cap=8)

        # Install instrumentation
        print(f"\nInstalling instrumentation ({config_name})...")
        hooks = install_instrumentation(model, timer, config_name, eb_ctrl)
        print(f"  Hooks: {len(hooks)}")

        # Warmup (with instrumentation, to compile Triton kernels)
        print("\nWarmup...")
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd")

        # Reset timer and EB state
        timer.reset()
        if eb_ctrl:
            eb_ctrl.prev_N.clear()
            eb_ctrl.K_init.clear()
            eb_ctrl.cold_count = 0
            eb_ctrl.hot_count = 0
            eb_ctrl.last_was_cold.clear()
            eb_ctrl._bufs.clear()

        # Profiled generation
        print("\nProfiled generation...")
        dllm = make_dllm()
        torch.cuda.synchronize()

        torch.cuda.cudart().cudaProfilerStart()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        torch.cuda.cudart().cudaProfilerStop()

        total_fwd = dllm.diff_iteration.num_forwards
        total_time = t1 - t0
        print(f"  Done: {total_fwd} fwd, {total_time:.3f}s, "
              f"{total_fwd/total_time:.1f} fwd/s")

        # Analyze
        raw_stats = timer.summarize()
        result = analyze(raw_stats, config_name, total_time, total_fwd, eb_ctrl)

        # Save
        out_path = (REPO_ROOT / "codex_coding" / "results" /
                    f"{config_name.lower()}_component_profiling.json")
        # Remove raw event data (not JSON serializable in full)
        save_result = {k: v for k, v in result.items() if k != 'raw_stats'}
        save_result['raw_stats_summary'] = raw_stats
        with open(out_path, "w") as f:
            json.dump(save_result, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")

        # Cleanup
        for h in hooks:
            h.remove()

        print("\nDone. For nsys analysis:")
        print(f"  nsys stats --report nvtx_sum <output>.nsys-rep")


if __name__ == "__main__":
    main()
