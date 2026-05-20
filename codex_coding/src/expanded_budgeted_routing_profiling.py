#!/usr/bin/env python3
"""
v0.1.15.5 — Expanded Budgeted Routing: Performance Profiling

Config: cap=8, QF=0.70, q_major=0.95, M=4, K_target=40, max_rounds=999, no top-p

Round 1: Baseline end-to-end timing (no hooks)
Round 2: EB hook end-to-end timing (no per-step sync) — real overhead
Round 3: EB per-step profiling (with sync, few forwards) — bottleneck
Round 4: Quality verification (temp=0.7, baseline vs EB)
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

# EB config (locked)
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


class ProfilingEBController:
    """Expanded budgeted routing with optional per-step profiling."""

    def __init__(self, do_profile=False):
        self.K = TOP_K_ORIG
        self.M = EB_M
        self.K_ext = self.K + self.M
        self.K_target = EB_K_TARGET
        self.quality_floor = EB_QF
        self.q_major = EB_Q_MAJOR
        self.per_round_cap = EB_CAP
        self.alpha = EB_ALPHA
        self.beta = EB_BETA
        self.do_profile = do_profile
        self.timings = defaultdict(list)
        self.round_counts = []
        self.active_counts = []

    def reset(self):
        self.timings.clear()
        self.round_counts.clear()
        self.active_counts.clear()

    def hook_forward(self, moe_mod, layer_idx, hidden_states):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        N = hs_flat.shape[0]
        E = moe_mod.gate.num_experts
        K = self.K
        K_ext = self.K_ext
        rsf = moe_mod.gate.routed_scaling_factor

        def tick():
            if self.do_profile:
                torch.cuda.synchronize()
                return time.perf_counter()
            return 0

        t0 = tick()

        # A: shared_experts
        shared_res = moe_mod.shared_experts(hidden_states)
        t_a = tick()

        # B: gate.get_logits
        gate_logits = moe_mod.gate.get_logits(hs_flat)
        t_b = tick()

        # C: sigmoid + topk(K_ext) + normalize
        scores_full = torch.sigmoid(gate_logits.float())
        topkm_score, topkm_idx = torch.topk(scores_full, k=K_ext, dim=1)
        topkm_weight = topkm_score / (topkm_score.sum(dim=1, keepdim=True) + 1e-20)
        topkm_weight = topkm_weight * rsf
        t_c = tick()

        # D: sort + quality threshold
        sorted_w, sort_order = topkm_weight.sort(dim=1, descending=True)
        r = self.quality_floor * sorted_w[:, :K].sum(dim=1)
        t_d = tick()

        # E: popularity + init S
        popularity = torch.zeros(E, device=hs_flat.device, dtype=topkm_weight.dtype)
        popularity.scatter_add_(0, topkm_idx.reshape(-1), topkm_weight.reshape(-1))
        _, pop_order = popularity.sort(descending=True)
        S_mask = torch.zeros(E, dtype=torch.bool, device=hs_flat.device)
        S_mask[pop_order[:self.K_target]] = True
        token_ids = torch.arange(N, device=hs_flat.device).unsqueeze(1).expand(N, K_ext)
        t_e = tick()

        # F: batch-add loop
        n_rounds = 0
        t_f1_total = 0; t_f2_total = 0; t_f3_total = 0; t_f4_total = 0

        for _ in range(999):
            tf0 = tick()

            # F1: coverage check
            in_S = S_mask[topkm_idx]
            c = (topkm_weight * in_S.float()).sum(dim=1)
            d = (r - c).clamp_min(0.0)
            satisfied = d <= 0
            sat_ratio = satisfied.float().mean().item()
            tf1 = tick()

            if sat_ratio >= self.q_major:
                if self.do_profile:
                    t_f1_total += (tf1 - tf0)
                break

            # F2: edge construction
            unsat = ~satisfied
            edge_mask = unsat.unsqueeze(1).expand_as(topkm_idx)
            edge_expert = topkm_idx[edge_mask]
            edge_weight = topkm_weight[edge_mask]
            edge_token = token_ids[edge_mask]
            keep = ~S_mask[edge_expert]
            edge_expert = edge_expert[keep]
            edge_weight = edge_weight[keep]
            edge_token = edge_token[keep]
            tf2 = tick()

            if edge_expert.numel() == 0:
                if self.do_profile:
                    t_f1_total += (tf1 - tf0)
                    t_f2_total += (tf2 - tf1)
                break

            # F3: scoring
            G = torch.zeros(E, device=hs_flat.device, dtype=topkm_weight.dtype)
            H = torch.zeros(E, device=hs_flat.device, dtype=topkm_weight.dtype)
            gap_gain = torch.minimum(edge_weight, d[edge_token])
            hit_gain = (c[edge_token] + edge_weight >= r[edge_token]).to(topkm_weight.dtype)
            G.scatter_add_(0, edge_expert, gap_gain)
            H.scatter_add_(0, edge_expert, hit_gain)
            Score = self.alpha * H + self.beta * G
            Score[S_mask] = -1e30
            tf3 = tick()

            # F4: expert selection
            _, add_order = Score.sort(descending=True)
            new_experts = add_order[:self.per_round_cap]
            valid = Score[new_experts] > 0
            new_experts = new_experts[valid]
            if new_experts.numel() == 0:
                if self.do_profile:
                    t_f1_total += (tf1 - tf0)
                    t_f2_total += (tf2 - tf1)
                    t_f3_total += (tf3 - tf2)
                break
            S_mask[new_experts] = True
            tf4 = tick()

            n_rounds += 1
            if self.do_profile:
                t_f1_total += (tf1 - tf0)
                t_f2_total += (tf2 - tf1)
                t_f3_total += (tf3 - tf2)
                t_f4_total += (tf4 - tf3)

        t_f = tick()
        self.round_counts.append(n_rounds)
        self.active_counts.append(S_mask.sum().item())

        # G: masked_fill
        masked_logits = gate_logits.masked_fill(~S_mask.unsqueeze(0), float('-inf'))
        t_g = tick()

        # H: gate.routing
        topk_weight_new, topk_idx_new = moe_mod.gate.routing(
            hs_flat, masked_logits, moe_mod.gate.top_k, True)
        t_h = tick()

        # I: fused_experts
        routed_y = fused_experts(
            hidden_states=hs_flat,
            w1=moe_mod.experts.w13_weight,
            w2=moe_mod.experts.w2_weight,
            topk_weights=topk_weight_new,
            topk_ids=topk_idx_new,
            inplace=False)
        t_i = tick()

        routed_y = routed_y.view(bsz, seq_len, h)
        out = routed_y + shared_res if moe_mod.config.num_shared_experts is not None else routed_y

        if self.do_profile:
            self.timings["A_shared"].append((t_a - t0) * 1e6)
            self.timings["B_get_logits"].append((t_b - t_a) * 1e6)
            self.timings["C_candidate"].append((t_c - t_b) * 1e6)
            self.timings["D_threshold"].append((t_d - t_c) * 1e6)
            self.timings["E_init_S"].append((t_e - t_d) * 1e6)
            self.timings["F_batchadd"].append((t_f - t_e) * 1e6)
            self.timings["F1_coverage"].append(t_f1_total * 1e6)
            self.timings["F2_edges"].append(t_f2_total * 1e6)
            self.timings["F3_scoring"].append(t_f3_total * 1e6)
            self.timings["F4_selection"].append(t_f4_total * 1e6)
            self.timings["G_mask"].append((t_g - t_f) * 1e6)
            self.timings["H_routing"].append((t_h - t_g) * 1e6)
            self.timings["I_fused_exp"].append((t_i - t_h) * 1e6)
            self.timings["total"].append((t_i - t0) * 1e6)

        return out


def install_hooks(model, ctrl):
    hooks = []
    mi = 0
    for layer in model.model.layers:
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue
        moe = layer.mlp
        orig = moe.forward
        idx = mi
        def make(m, i, c):
            def f(hs): return c.hook_forward(m, i, hs)
            return f
        moe.forward = make(moe, idx, ctrl)
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
    print("Expanded Budgeted Routing — Performance Profiling")
    print(f"  Config: M={EB_M}, K_target={EB_K_TARGET}, QF={EB_QF}, "
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
        print("Round 1: Baseline end-to-end (no hooks)")
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
        # Round 2: EB hook end-to-end (no sync)
        # ==============================================================
        print(f"\n{'='*60}")
        print("Round 2: EB hook end-to-end (no per-step sync)")
        print("=" * 60)
        ctrl_r2 = ProfilingEBController(do_profile=False)
        hooks = install_hooks(model, ctrl_r2)

        # Warmup with hooks
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        ctrl_r2.reset()
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

        avg_active = sum(ctrl_r2.active_counts) / len(ctrl_r2.active_counts) if ctrl_r2.active_counts else 0
        avg_rounds = sum(ctrl_r2.round_counts) / len(ctrl_r2.round_counts) if ctrl_r2.round_counts else 0

        print(f"  Fwd={fwd_eb}, time={time_eb:.3f}s, fwd/s={fwd_eb/time_eb:.1f}")
        print(f"  vs baseline: {overhead:+.1f}%, ΔFwd={fwd_eb-fwd_bl:+d}")
        print(f"  Avg |S|={avg_active:.1f}, avg rounds={avg_rounds:.1f}")

        remove_hooks(hooks)

        # ==============================================================
        # Round 3: Per-step profiling (with sync, gen=32 for few forwards)
        # ==============================================================
        print(f"\n{'='*60}")
        print("Round 3: Per-step profiling (with sync)")
        print("=" * 60)
        ctrl_r3 = ProfilingEBController(do_profile=True)
        hooks = install_hooks(model, ctrl_r3)

        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=32,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        n_profiled = dllm.diff_iteration.num_forwards

        print(f"  Profiled {n_profiled} forwards × 19 layers")
        n_samples = len(ctrl_r3.timings.get("total", []))
        print(f"  Total samples: {n_samples}")

        total_all = sum(ctrl_r3.timings.get("total", [0]))

        steps = [
            ("A_shared",     "shared_experts",    False),
            ("B_get_logits", "gate.get_logits",   False),
            ("C_candidate",  "candidate constr",  True),
            ("D_threshold",  "quality threshold", True),
            ("E_init_S",     "popularity+initS",  True),
            ("F_batchadd",   "batch-add TOTAL",   True),
            ("F1_coverage",  "  F1 coverage",     True),
            ("F2_edges",     "  F2 edges",        True),
            ("F3_scoring",   "  F3 scoring",      True),
            ("F4_selection", "  F4 selection",     True),
            ("G_mask",       "masked_fill",       True),
            ("H_routing",    "gate.routing",      False),
            ("I_fused_exp",  "fused_experts",     False),
        ]

        print(f"\n  {'Step':<22s} {'Avg(us)':>8s} {'Med(us)':>8s} "
              f"{'Total(ms)':>10s} {'%':>6s}")
        print(f"  {'-'*58}")

        for key, label, is_new in steps:
            vals = ctrl_r3.timings.get(key, [])
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            med = sorted(vals)[len(vals) // 2]
            tot = sum(vals) / 1000
            pct = sum(vals) / total_all * 100 if total_all > 0 else 0
            marker = " ★" if is_new else ""
            print(f"  {label:<22s} {avg:>8.0f} {med:>8.0f} "
                  f"{tot:>10.1f} {pct:>5.1f}%{marker}")

        print(f"  {'-'*58}")
        tot_vals = ctrl_r3.timings.get("total", [])
        if tot_vals:
            avg_t = sum(tot_vals) / len(tot_vals)
            print(f"  {'TOTAL per layer':<22s} {avg_t:>8.0f}")
            print(f"  {'per forward (×19)':<22s} {avg_t*19/1000:>8.1f} ms")
            print(f"\n  ★ = new overhead (not in baseline)")

        avg_r3_rounds = sum(ctrl_r3.round_counts) / len(ctrl_r3.round_counts) if ctrl_r3.round_counts else 0
        print(f"  Avg batch-add rounds: {avg_r3_rounds:.1f}")

        # Baseline MoE reference
        bl_moe_per_fwd = time_bl / fwd_bl * 0.60 * 1000  # 60% of baseline is MoE
        print(f"\n  For reference:")
        print(f"    Baseline MoE per fwd: ~{bl_moe_per_fwd:.1f} ms (60% of {time_bl/fwd_bl*1000:.1f} ms)")
        print(f"    EB overhead per fwd:  ~{time_eb/fwd_eb*1000 - time_bl/fwd_bl*1000:.1f} ms")

        remove_hooks(hooks)

        # ==============================================================
        # Round 4: Quality verification (temp=0.7)
        # ==============================================================
        print(f"\n{'='*60}")
        print("Round 4: Quality verification (temp=0.7)")
        print("=" * 60)

        for cname, use_eb in [("Baseline", False), ("EB_cap8_qf70_qm95", True)]:
            if use_eb:
                ctrl_q = ProfilingEBController(do_profile=False)
                hooks = install_hooks(model, ctrl_q)

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
            fwd_q = dllm.diff_iteration.num_forwards

            print(f"\n  [{cname}] Fwd={fwd_q}")
            for bi in sorted(VERIFIABLE.keys()):
                gt = gen_tokens[bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                print(f"    #{bi}: {text[:200]}")

            if use_eb:
                remove_hooks(hooks)

        # ==============================================================
        # Summary
        # ==============================================================
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"  Baseline:  Fwd={fwd_bl}, time={time_bl:.3f}s")
        print(f"  EB hook:   Fwd={fwd_eb}, time={time_eb:.3f}s "
              f"({overhead:+.1f}%), ΔFwd={fwd_eb-fwd_bl:+d}")
        print(f"  |S|={avg_active:.1f}, rounds={avg_rounds:.1f}")
        if tot_vals:
            avg_t = sum(tot_vals) / len(tot_vals)
            baseline_layer = time_bl / fwd_bl * 0.60 / 19 * 1000 * 1000  # us
            new_overhead = avg_t - baseline_layer if baseline_layer > 0 else avg_t
            print(f"\n  Per-layer profiled total: {avg_t:.0f} μs")
            print(f"  Estimated baseline MoE per layer: ~{baseline_layer:.0f} μs")

        # Save
        save_data = {
            "baseline": {"fwd": fwd_bl, "time_s": time_bl},
            "eb_hook": {"fwd": fwd_eb, "time_s": time_eb,
                        "overhead_pct": overhead,
                        "avg_active": avg_active, "avg_rounds": avg_rounds},
            "per_step": {k: {"avg_us": sum(v)/len(v), "total_ms": sum(v)/1000}
                         for k, v in ctrl_r3.timings.items() if v},
            "config": {"M": EB_M, "K_target": EB_K_TARGET, "QF": EB_QF,
                       "q_major": EB_Q_MAJOR, "cap": EB_CAP},
        }
        out_path = (REPO_ROOT / "codex_coding" / "results" /
                    "expanded_budgeted_routing_profiling.json")
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
