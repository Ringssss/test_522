#!/usr/bin/env python3
"""
v0.1.15.5 — Baseline vs EB: Combined MoE Internal Profiling

5 Rounds:
  R1: Baseline end-to-end timing
  R2: EB end-to-end timing
  R3: Sync-based per-step profiling (baseline + EB, few forwards)
  R4: NVTX-based profiling for nsys (baseline + EB, full generation)
  R5: Quality verification (temp=0.7)

EB config: cap=8, QF=0.70, q_major=0.95, M=4, K_target=40
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
NUM_EXPERTS = 256
TOP_K_ORIG = 8

EB_M = 4
EB_K_TARGET = 40
EB_QF = 0.70
EB_Q_MAJOR = 0.95
EB_CAP = 8
EB_ALPHA = 1.0
EB_BETA = 0.5

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

VERIFIABLE = {
    0: "average speed = 480/7 ~ 68.57 km/h",
    8: "x = 2 and x = 3",
    13: "B is true, C is true, D is true",
    19: "fib(10) = 55",
    28: "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune",
}


# ==============================================================
# Baseline sync profiler: hooks on MoE internals
# ==============================================================
class BaselineMoEProfiler:
    """Sync-based profiling of baseline MoE internals."""
    def __init__(self):
        self.timings = defaultdict(list)

    def reset(self):
        self.timings.clear()

    def make_profiled_forward(self, moe_mod, layer_idx):
        profiler = self

        def profiled_forward(hidden_states):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            res = moe_mod.shared_experts(hidden_states)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            bsz, seq_len, h = hidden_states.shape
            hs_flat = hidden_states.view(-1, h)

            router_logits = moe_mod.gate.get_logits(hs_flat)
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            y = moe_mod.experts.forward_impl(
                hidden_states=hs_flat, router_logits=router_logits)
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            y = y.view(bsz, seq_len, h)
            if moe_mod.config.num_shared_experts is not None:
                y = y + res

            profiler.timings["BL_shared"].append((t1 - t0) * 1e6)
            profiler.timings["BL_get_logits"].append((t2 - t1) * 1e6)
            profiler.timings["BL_forward_impl"].append((t3 - t2) * 1e6)
            profiler.timings["BL_total"].append((t3 - t0) * 1e6)
            return y

        return profiled_forward


# ==============================================================
# EB sync profiler (simplified: just key sections)
# ==============================================================
class EBMoEProfiler:
    """Sync-based profiling of EB MoE internals."""
    def __init__(self):
        self.timings = defaultdict(list)
        self.active_counts = []
        self.round_counts = []

    def reset(self):
        self.timings.clear()
        self.active_counts.clear()
        self.round_counts.clear()

    def make_profiled_forward(self, moe_mod, layer_idx):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        profiler = self
        K = TOP_K_ORIG
        K_ext = K + EB_M
        rsf = moe_mod.gate.routed_scaling_factor

        def profiled_forward(hidden_states):
            bsz, seq_len, h = hidden_states.shape
            hs_flat = hidden_states.view(-1, h)
            N = hs_flat.shape[0]
            E = moe_mod.gate.num_experts

            torch.cuda.synchronize(); t0 = time.perf_counter()
            shared_res = moe_mod.shared_experts(hidden_states)
            torch.cuda.synchronize(); t1 = time.perf_counter()

            gate_logits = moe_mod.gate.get_logits(hs_flat)
            torch.cuda.synchronize(); t2 = time.perf_counter()

            # Steps C-E: candidate + threshold + init S
            scores_full = torch.sigmoid(gate_logits.float())
            topkm_score, topkm_idx = torch.topk(scores_full, k=K_ext, dim=1)
            topkm_weight = topkm_score / (topkm_score.sum(dim=1, keepdim=True) + 1e-20) * rsf
            sorted_w, _ = topkm_weight.sort(dim=1, descending=True)
            r = EB_QF * sorted_w[:, :K].sum(dim=1)
            popularity = torch.zeros(E, device=hs_flat.device, dtype=topkm_weight.dtype)
            popularity.scatter_add_(0, topkm_idx.reshape(-1), topkm_weight.reshape(-1))
            _, pop_order = popularity.sort(descending=True)
            S_mask = torch.zeros(E, dtype=torch.bool, device=hs_flat.device)
            S_mask[pop_order[:EB_K_TARGET]] = True
            token_ids = torch.arange(N, device=hs_flat.device).unsqueeze(1).expand(N, K_ext)
            torch.cuda.synchronize(); t3 = time.perf_counter()

            # Step F: batch-add
            n_rounds = 0
            for _ in range(999):
                in_S = S_mask[topkm_idx]
                c = (topkm_weight * in_S.float()).sum(dim=1)
                d = (r - c).clamp_min(0.0)
                satisfied = d <= 0
                sat_ratio = satisfied.float().mean().item()
                if sat_ratio >= EB_Q_MAJOR:
                    break
                unsat = ~satisfied
                edge_mask = unsat.unsqueeze(1).expand_as(topkm_idx)
                edge_expert = topkm_idx[edge_mask]
                edge_weight = topkm_weight[edge_mask]
                edge_token = token_ids[edge_mask]
                keep = ~S_mask[edge_expert]
                edge_expert = edge_expert[keep]
                edge_weight = edge_weight[keep]
                edge_token = edge_token[keep]
                if edge_expert.numel() == 0:
                    break
                G = torch.zeros(E, device=hs_flat.device, dtype=topkm_weight.dtype)
                H = torch.zeros(E, device=hs_flat.device, dtype=topkm_weight.dtype)
                gap_gain = torch.minimum(edge_weight, d[edge_token])
                hit_gain = (c[edge_token] + edge_weight >= r[edge_token]).to(topkm_weight.dtype)
                G.scatter_add_(0, edge_expert, gap_gain)
                H.scatter_add_(0, edge_expert, hit_gain)
                Score = EB_ALPHA * H + EB_BETA * G
                Score[S_mask] = -1e30
                _, add_order = Score.sort(descending=True)
                new_experts = add_order[:EB_CAP]
                valid = Score[new_experts] > 0
                new_experts = new_experts[valid]
                if new_experts.numel() == 0:
                    break
                S_mask[new_experts] = True
                n_rounds += 1
            torch.cuda.synchronize(); t4 = time.perf_counter()

            profiler.active_counts.append(S_mask.sum().item())
            profiler.round_counts.append(n_rounds)

            # Step G+H: mask + routing
            masked_logits = gate_logits.masked_fill(~S_mask.unsqueeze(0), float('-inf'))
            topk_weight_new, topk_idx_new = moe_mod.gate.routing(
                hs_flat, masked_logits, moe_mod.gate.top_k, True)
            torch.cuda.synchronize(); t5 = time.perf_counter()

            # Step I: fused_experts
            routed_y = fused_experts(
                hidden_states=hs_flat,
                w1=moe_mod.experts.w13_weight,
                w2=moe_mod.experts.w2_weight,
                topk_weights=topk_weight_new,
                topk_ids=topk_idx_new,
                inplace=False)
            torch.cuda.synchronize(); t6 = time.perf_counter()

            routed_y = routed_y.view(bsz, seq_len, h)
            out = routed_y + shared_res if moe_mod.config.num_shared_experts is not None else routed_y

            profiler.timings["EB_shared"].append((t1 - t0) * 1e6)
            profiler.timings["EB_get_logits"].append((t2 - t1) * 1e6)
            profiler.timings["EB_cde_prep"].append((t3 - t2) * 1e6)
            profiler.timings["EB_batchadd"].append((t4 - t3) * 1e6)
            profiler.timings["EB_mask_routing"].append((t5 - t4) * 1e6)
            profiler.timings["EB_fused_exp"].append((t6 - t5) * 1e6)
            profiler.timings["EB_total"].append((t6 - t0) * 1e6)
            return out

        return profiled_forward


# ==============================================================
# NVTX hooks for nsys (lightweight, no sync)
# ==============================================================
def install_baseline_nvtx(model):
    """NVTX markers on baseline MoE internals."""
    hooks = []
    nvtx = torch.cuda.nvtx
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeSparseMoeBlock

    for name, module in model.named_modules():
        if not isinstance(module, LLaDA2MoeSparseMoeBlock):
            continue
        # Extract layer index from name
        parts = name.split('.')
        li = None
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                li = parts[i + 1]
                break
        tag = f"L{li}" if li else name

        # Hook shared_experts
        def make_pre(t):
            def f(mod, inp): nvtx.range_push(t)
            return f
        def make_post(t):
            def f(mod, inp, out): nvtx.range_pop()
            return f

        hooks.append(module.shared_experts.register_forward_pre_hook(
            make_pre(f"BL_Shared_{tag}")))
        hooks.append(module.shared_experts.register_forward_hook(
            make_post(f"BL_Shared_{tag}")))

        # Hook gate (get_logits is called via gate.get_logits, but we hook the
        # full experts module which contains routing+fused_experts)
        hooks.append(module.gate.register_forward_pre_hook(
            make_pre(f"BL_Gate_{tag}")))
        hooks.append(module.gate.register_forward_hook(
            make_post(f"BL_Gate_{tag}")))

        hooks.append(module.experts.register_forward_pre_hook(
            make_pre(f"BL_Experts_{tag}")))
        hooks.append(module.experts.register_forward_hook(
            make_post(f"BL_Experts_{tag}")))

    return hooks


def install_eb_nvtx_hooks(model, eb_controller):
    """Install EB hooks with NVTX markers (no sync)."""
    hooks = []
    nvtx = torch.cuda.nvtx
    mi = 0

    for layer in model.model.layers:
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue
        moe = layer.mlp
        orig = moe.forward
        idx = mi

        def make_eb_nvtx_forward(moe_mod, layer_idx, ctrl):
            from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
            K = TOP_K_ORIG
            K_ext = K + EB_M
            rsf = moe_mod.gate.routed_scaling_factor

            def eb_nvtx_forward(hidden_states):
                bsz, seq_len, h = hidden_states.shape
                hs_flat = hidden_states.view(-1, h)
                N = hs_flat.shape[0]
                E = moe_mod.gate.num_experts

                nvtx.range_push(f"EB_Shared_L{layer_idx}")
                shared_res = moe_mod.shared_experts(hidden_states)
                nvtx.range_pop()

                nvtx.range_push(f"EB_Gate_L{layer_idx}")
                gate_logits = moe_mod.gate.get_logits(hs_flat)
                nvtx.range_pop()

                nvtx.range_push(f"EB_Prep_L{layer_idx}")
                scores_full = torch.sigmoid(gate_logits.float())
                topkm_score, topkm_idx = torch.topk(scores_full, k=K_ext, dim=1)
                topkm_weight = topkm_score / (topkm_score.sum(dim=1, keepdim=True) + 1e-20) * rsf
                sorted_w, _ = topkm_weight.sort(dim=1, descending=True)
                r = EB_QF * sorted_w[:, :K].sum(dim=1)
                popularity = torch.zeros(E, device=hs_flat.device, dtype=topkm_weight.dtype)
                popularity.scatter_add_(0, topkm_idx.reshape(-1), topkm_weight.reshape(-1))
                _, pop_order = popularity.sort(descending=True)
                S_mask = torch.zeros(E, dtype=torch.bool, device=hs_flat.device)
                S_mask[pop_order[:EB_K_TARGET]] = True
                token_ids = torch.arange(N, device=hs_flat.device).unsqueeze(1).expand(N, K_ext)
                nvtx.range_pop()

                nvtx.range_push(f"EB_BatchAdd_L{layer_idx}")
                for _ in range(999):
                    in_S = S_mask[topkm_idx]
                    c = (topkm_weight * in_S.float()).sum(dim=1)
                    d = (r - c).clamp_min(0.0)
                    satisfied = d <= 0
                    sat_ratio = satisfied.float().mean().item()
                    if sat_ratio >= EB_Q_MAJOR:
                        break
                    unsat = ~satisfied
                    edge_mask = unsat.unsqueeze(1).expand_as(topkm_idx)
                    edge_expert = topkm_idx[edge_mask]
                    edge_weight = topkm_weight[edge_mask]
                    edge_token = token_ids[edge_mask]
                    keep = ~S_mask[edge_expert]
                    edge_expert = edge_expert[keep]
                    edge_weight = edge_weight[keep]
                    edge_token = edge_token[keep]
                    if edge_expert.numel() == 0:
                        break
                    G = torch.zeros(E, device=hs_flat.device, dtype=topkm_weight.dtype)
                    H = torch.zeros(E, device=hs_flat.device, dtype=topkm_weight.dtype)
                    gap_gain = torch.minimum(edge_weight, d[edge_token])
                    hit_gain = (c[edge_token] + edge_weight >= r[edge_token]).to(topkm_weight.dtype)
                    G.scatter_add_(0, edge_expert, gap_gain)
                    H.scatter_add_(0, edge_expert, hit_gain)
                    Score = EB_ALPHA * H + EB_BETA * G
                    Score[S_mask] = -1e30
                    _, add_order = Score.sort(descending=True)
                    new_experts = add_order[:EB_CAP]
                    valid = Score[new_experts] > 0
                    new_experts = new_experts[valid]
                    if new_experts.numel() == 0:
                        break
                    S_mask[new_experts] = True
                nvtx.range_pop()

                nvtx.range_push(f"EB_Routing_L{layer_idx}")
                masked_logits = gate_logits.masked_fill(~S_mask.unsqueeze(0), float('-inf'))
                topk_weight_new, topk_idx_new = moe_mod.gate.routing(
                    hs_flat, masked_logits, moe_mod.gate.top_k, True)
                nvtx.range_pop()

                nvtx.range_push(f"EB_Fused_L{layer_idx}")
                routed_y = fused_experts(
                    hidden_states=hs_flat,
                    w1=moe_mod.experts.w13_weight,
                    w2=moe_mod.experts.w2_weight,
                    topk_weights=topk_weight_new,
                    topk_ids=topk_idx_new,
                    inplace=False)
                nvtx.range_pop()

                routed_y = routed_y.view(bsz, seq_len, h)
                out = routed_y + shared_res if moe_mod.config.num_shared_experts is not None else routed_y
                return out

            return eb_nvtx_forward

        moe.forward = make_eb_nvtx_forward(moe, idx, eb_controller)
        hooks.append((moe, orig))
        mi += 1

    return hooks


def remove_hooks(hooks):
    for moe, orig in hooks:
        moe.forward = orig


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoTokenizer, AutoConfig
    from baseline_optimizations import apply_all_optimizations

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print("Baseline vs EB: Combined MoE Internal Profiling")
    print(f"  EB: M={EB_M}, K_target={EB_K_TARGET}, QF={EB_QF}, "
          f"q_major={EB_Q_MAJOR}, cap={EB_CAP}")
    print("  HetEval-32: batch=32, gen=256, block=32")
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
        padded = []
        for ids in all_ids:
            if ids.shape[0] < mx:
                ids = torch.cat([torch.full((mx - ids.shape[0],), pad_id,
                                            dtype=ids.dtype), ids])
            padded.append(ids)
        input_ids = torch.stack(padded, dim=0).to(device)
        prompt_len = input_ids.shape[1]
        print(f"  Input shape: {input_ids.shape}")

        GEN_LENGTH = 256
        decoder_t0 = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        decoder_t7 = ThresholdParallelDecoder(
            temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm(decoder):
            return BlockDiffusionLLM(
                model, decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True,
                inplace_cache_update=True)

        # Warmup
        print("\nWarmup...")
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print("  Done.")

        # ==============================================================
        # Round 1: Baseline end-to-end
        # ==============================================================
        print(f"\n{'='*60}")
        print("Round 1: Baseline end-to-end")
        print("=" * 60)
        dllm = make_dllm(decoder_t0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd_bl = dllm.diff_iteration.num_forwards
        time_bl = t1 - t0
        print(f"  Fwd={fwd_bl}, time={time_bl:.3f}s, fwd/s={fwd_bl/time_bl:.1f}")

        # ==============================================================
        # Round 2: EB end-to-end (no profiling overhead)
        # ==============================================================
        print(f"\n{'='*60}")
        print("Round 2: EB end-to-end (no profiling)")
        print("=" * 60)
        # Install simple EB hooks (no profiling)
        from expanded_budgeted_routing_e2e import (
            ExpandedBudgetedRoutingController, install_hooks as install_eb_hooks)
        ctrl_simple = ExpandedBudgetedRoutingController(
            K=TOP_K_ORIG, M=EB_M, K_target=EB_K_TARGET,
            quality_floor=EB_QF, q_major=EB_Q_MAJOR,
            per_round_cap=EB_CAP, max_add_rounds=999)
        eb_hooks = install_eb_hooks(model, ctrl_simple)

        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        ctrl_simple.stats = defaultdict(list)
        dllm = make_dllm(decoder_t0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd_eb = dllm.diff_iteration.num_forwards
        time_eb = t1 - t0
        overhead = (time_eb - time_bl) / time_bl * 100
        print(f"  Fwd={fwd_eb}, time={time_eb:.3f}s ({overhead:+.1f}%), ΔFwd={fwd_eb-fwd_bl:+d}")
        remove_hooks(eb_hooks)

        # ==============================================================
        # Round 3: Sync-based per-step (baseline + EB, gen=32)
        # ==============================================================
        print(f"\n{'='*60}")
        print("Round 3: Sync-based per-step profiling (gen=32)")
        print("=" * 60)

        # 3a: Baseline
        print("\n  --- 3a: Baseline MoE internals ---")
        bl_prof = BaselineMoEProfiler()
        bl_hooks = []
        mi = 0
        for layer in model.model.layers:
            if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
                continue
            moe = layer.mlp
            orig = moe.forward
            moe.forward = bl_prof.make_profiled_forward(moe, mi)
            bl_hooks.append((moe, orig))
            mi += 1

        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=32,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        bl_fwd_prof = dllm.diff_iteration.num_forwards

        for step in ["BL_shared", "BL_get_logits", "BL_forward_impl", "BL_total"]:
            vals = bl_prof.timings[step]
            if vals:
                avg = sum(vals) / len(vals)
                print(f"    {step:<20s}  avg={avg:>8.0f} μs")

        remove_hooks(bl_hooks)

        # 3b: EB
        print("\n  --- 3b: EB MoE internals ---")
        eb_prof = EBMoEProfiler()
        eb_hooks = []
        mi = 0
        for layer in model.model.layers:
            if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
                continue
            moe = layer.mlp
            orig = moe.forward
            moe.forward = eb_prof.make_profiled_forward(moe, mi)
            eb_hooks.append((moe, orig))
            mi += 1

        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=32,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        eb_fwd_prof = dllm.diff_iteration.num_forwards

        for step in ["EB_shared", "EB_get_logits", "EB_cde_prep", "EB_batchadd",
                      "EB_mask_routing", "EB_fused_exp", "EB_total"]:
            vals = eb_prof.timings[step]
            if vals:
                avg = sum(vals) / len(vals)
                marker = " ★" if step in ["EB_cde_prep", "EB_batchadd"] else ""
                print(f"    {step:<20s}  avg={avg:>8.0f} μs{marker}")

        avg_active = sum(eb_prof.active_counts) / len(eb_prof.active_counts) if eb_prof.active_counts else 0
        avg_rounds = sum(eb_prof.round_counts) / len(eb_prof.round_counts) if eb_prof.round_counts else 0
        print(f"    Avg |S|={avg_active:.1f}, rounds={avg_rounds:.1f}")

        # Side-by-side comparison
        print(f"\n  --- Comparison (sync, per layer avg) ---")
        bl_fi = sum(bl_prof.timings["BL_forward_impl"]) / len(bl_prof.timings["BL_forward_impl"])
        eb_mr = sum(eb_prof.timings["EB_mask_routing"]) / len(eb_prof.timings["EB_mask_routing"])
        eb_fe = sum(eb_prof.timings["EB_fused_exp"]) / len(eb_prof.timings["EB_fused_exp"])
        print(f"    BL forward_impl (routing+fused):  {bl_fi:>8.0f} μs")
        print(f"    EB routing + fused_experts:        {eb_mr + eb_fe:>8.0f} μs  (routing={eb_mr:.0f} + fused={eb_fe:.0f})")
        print(f"    Savings from EB on compute path:   {bl_fi - eb_mr - eb_fe:>8.0f} μs/layer")
        print(f"    EB batch-add overhead:             {sum(eb_prof.timings['EB_batchadd'])/len(eb_prof.timings['EB_batchadd']):>8.0f} μs/layer")

        remove_hooks(eb_hooks)

        # ==============================================================
        # Round 4: NVTX for nsys (baseline gen + EB gen)
        # ==============================================================
        print(f"\n{'='*60}")
        print("Round 4: NVTX profiling for nsys")
        print("=" * 60)

        # 4a: Baseline with NVTX
        print("  Installing baseline NVTX hooks...")
        nvtx_hooks_bl = install_baseline_nvtx(model)
        print(f"  {len(nvtx_hooks_bl)} hooks installed.")

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("BASELINE_GENERATION")

        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)

        torch.cuda.nvtx.range_pop()

        # Remove baseline NVTX, install EB NVTX
        for h in nvtx_hooks_bl:
            h.remove()

        print("  Installing EB NVTX hooks...")
        eb_nvtx_hooks = install_eb_nvtx_hooks(model, None)
        print(f"  {len(eb_nvtx_hooks)} hooks installed.")

        torch.cuda.nvtx.range_push("EB_GENERATION")
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        print("  nsys capture done.")

        remove_hooks(eb_nvtx_hooks)

        # ==============================================================
        # Round 5: Quality verification (temp=0.7)
        # ==============================================================
        print(f"\n{'='*60}")
        print("Round 5: Quality verification (temp=0.7)")
        print("=" * 60)

        for cname, use_eb in [("Baseline", False), ("EB_cap8_qf70", True)]:
            if use_eb:
                ctrl_q = ExpandedBudgetedRoutingController(
                    K=TOP_K_ORIG, M=EB_M, K_target=EB_K_TARGET,
                    quality_floor=EB_QF, q_major=EB_Q_MAJOR,
                    per_round_cap=EB_CAP, max_add_rounds=999)
                q_hooks = install_eb_hooks(model, ctrl_q)

            dllm = make_dllm(decoder_t7)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                out = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                    block_length=BLOCK_LENGTH)
            gen_tokens = out[:, prompt_len:]

            print(f"\n  [{cname}]")
            for bi in sorted(VERIFIABLE.keys()):
                gt = gen_tokens[bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                print(f"    #{bi}: {text[:200]}")

            if use_eb:
                remove_hooks(q_hooks)

        # ==============================================================
        # Summary
        # ==============================================================
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"  Baseline: Fwd={fwd_bl}, time={time_bl:.3f}s")
        print(f"  EB hook:  Fwd={fwd_eb}, time={time_eb:.3f}s ({overhead:+.1f}%)")
        print(f"\n  Sync per-layer (avg μs):")
        print(f"    {'Component':<25s} {'Baseline':>10s} {'EB':>10s} {'Δ':>10s}")
        print(f"    {'-'*55}")

        bl_s = sum(bl_prof.timings["BL_shared"]) / len(bl_prof.timings["BL_shared"])
        bl_g = sum(bl_prof.timings["BL_get_logits"]) / len(bl_prof.timings["BL_get_logits"])
        eb_s = sum(eb_prof.timings["EB_shared"]) / len(eb_prof.timings["EB_shared"])
        eb_g = sum(eb_prof.timings["EB_get_logits"]) / len(eb_prof.timings["EB_get_logits"])
        eb_p = sum(eb_prof.timings["EB_cde_prep"]) / len(eb_prof.timings["EB_cde_prep"])
        eb_b = sum(eb_prof.timings["EB_batchadd"]) / len(eb_prof.timings["EB_batchadd"])

        print(f"    {'shared_experts':<25s} {bl_s:>10.0f} {eb_s:>10.0f} {eb_s-bl_s:>+10.0f}")
        print(f"    {'gate.get_logits':<25s} {bl_g:>10.0f} {eb_g:>10.0f} {eb_g-bl_g:>+10.0f}")
        print(f"    {'EB prep (C+D+E)':<25s} {'—':>10s} {eb_p:>10.0f} {'+' + str(int(eb_p)):>10s}")
        print(f"    {'EB batch-add (F)':<25s} {'—':>10s} {eb_b:>10.0f} {'+' + str(int(eb_b)):>10s}")
        print(f"    {'routing+fused_experts':<25s} {bl_fi:>10.0f} {eb_mr+eb_fe:>10.0f} {eb_mr+eb_fe-bl_fi:>+10.0f}")
        bl_t = sum(bl_prof.timings["BL_total"]) / len(bl_prof.timings["BL_total"])
        eb_t = sum(eb_prof.timings["EB_total"]) / len(eb_prof.timings["EB_total"])
        print(f"    {'-'*55}")
        print(f"    {'TOTAL':<25s} {bl_t:>10.0f} {eb_t:>10.0f} {eb_t-bl_t:>+10.0f}")

        print(f"\n  Key: routing+fused savings = {bl_fi - eb_mr - eb_fe:.0f} μs/layer")
        print(f"       batch-add overhead     = {eb_b:.0f} μs/layer")
        print(f"       Net per layer           = {bl_fi - eb_mr - eb_fe - eb_b - eb_p:.0f} μs/layer")

        # Save
        save_data = {
            "baseline": {"fwd": fwd_bl, "time_s": time_bl},
            "eb": {"fwd": fwd_eb, "time_s": time_eb, "overhead_pct": overhead},
            "sync_baseline": {k: sum(v)/len(v) for k, v in bl_prof.timings.items()},
            "sync_eb": {k: sum(v)/len(v) for k, v in eb_prof.timings.items()},
            "eb_stats": {"avg_active": avg_active, "avg_rounds": avg_rounds},
        }
        out_path = (REPO_ROOT / "codex_coding" / "results" /
                    "baseline_vs_eb_moe_profiling.json")
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")
        print(f"\n  nsys analysis: nsys stats --report nvtx_sum <output>.nsys-rep")
        print("\nDone.")


if __name__ == "__main__":
    main()
