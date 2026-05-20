#!/usr/bin/env python3
"""
v0.1.15.5 — Expanded Top-(K+M) Budgeted Routing: E2E Quality & Active Set Verification

New algorithm: replace dense compute_active_set with expanded top-(K+M) budgeted routing.
All scores unified in sigmoid/expanded weight space. Batch-add experts instead of iterative loop.

Tests on HetEval-32 (optimized baseline):
  C0: Baseline (no EB)
  C1: M=4, K_target=40, QF=0.85, with top-p
  C2: M=4, K_target=40, QF=0.85, without top-p
  C3: M=2, K_target=40, QF=0.85, without top-p
  C4: M=8, K_target=40, QF=0.85, without top-p

Metrics: ΔFwd, |S|, sat_ratio, batch-add rounds, output quality (temp=0.7).
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
SHARED_RATE = 0.419
ROUTING_RATE = 0.581

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
# Expanded Top-(K+M) Budgeted Routing — Core Algorithm
# ==============================================================
class ExpandedBudgetedRoutingController:
    """Expanded top-(K+M) budgeted routing with batch-add experts."""

    def __init__(self, K=8, M=4, K_target=40, quality_floor=0.85,
                 q_major=0.95, per_round_cap=64, max_add_rounds=3,
                 alpha=1.0, beta=0.5, use_topp=False, top_p=0.75):
        self.K = K
        self.M = M
        self.K_ext = K + M
        self.K_target = K_target
        self.quality_floor = quality_floor
        self.q_major = q_major
        self.per_round_cap = per_round_cap
        self.max_add_rounds = max_add_rounds
        self.alpha = alpha
        self.beta = beta
        self.use_topp = use_topp
        self.top_p = top_p

        # Stats
        self.stats = defaultdict(list)

    def hook_forward(self, moe_mod, layer_idx, hidden_states):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        N = hs_flat.shape[0]
        E = moe_mod.gate.num_experts
        K = self.K
        K_ext = self.K_ext
        rsf = moe_mod.gate.routed_scaling_factor

        # Step 0: Shared experts (unchanged)
        shared_res = moe_mod.shared_experts(hidden_states)

        # Step 1: Gate logits + sigmoid (one call, no redundant F.linear)
        gate_logits = moe_mod.gate.get_logits(hs_flat)           # [N, E]
        scores_full = torch.sigmoid(gate_logits.float())          # [N, E]

        # Step 2: Construct top-(K+M) expanded candidates
        topkm_score, topkm_idx = torch.topk(
            scores_full, k=K_ext, dim=1)                          # [N, K_ext]
        topkm_weight = topkm_score / (
            topkm_score.sum(dim=1, keepdim=True) + 1e-20)        # [N, K_ext]
        topkm_weight = topkm_weight * rsf                         # [N, K_ext]

        # Step 3: Quality threshold (anchored to expanded top-K)
        sorted_w, sort_order = topkm_weight.sort(dim=1, descending=True)
        r = self.quality_floor * sorted_w[:, :K].sum(dim=1)      # [N]

        # Step 4: Initial active set (expanded popularity)
        popularity = torch.zeros(E, device=hs_flat.device, dtype=topkm_weight.dtype)
        popularity.scatter_add_(
            0, topkm_idx.reshape(-1), topkm_weight.reshape(-1))  # [E]
        _, pop_order = popularity.sort(descending=True)
        S_mask = torch.zeros(E, dtype=torch.bool, device=hs_flat.device)
        S_mask[pop_order[:self.K_target]] = True

        # Token indices for scatter_add in batch-add
        token_ids = torch.arange(
            N, device=hs_flat.device).unsqueeze(1).expand(N, K_ext)

        # Step 5-6: Coverage check + batch-add loop
        n_rounds = 0
        for round_i in range(self.max_add_rounds):
            # Coverage
            in_S = S_mask[topkm_idx]                               # [N, K_ext]
            c = (topkm_weight * in_S.float()).sum(dim=1)           # [N]
            d = (r - c).clamp_min(0.0)                             # [N]

            satisfied = d <= 0                                      # [N]
            sat_ratio = satisfied.float().mean().item()

            if sat_ratio >= self.q_major:
                break

            # Candidate edges: unsatisfied tokens' top-(K+M), not in S
            unsat = ~satisfied                                      # [N]
            edge_mask = unsat.unsqueeze(1).expand_as(topkm_idx)    # [N, K_ext]

            edge_expert = topkm_idx[edge_mask]                     # [M_edges]
            edge_weight = topkm_weight[edge_mask]                  # [M_edges]
            edge_token = token_ids[edge_mask]                      # [M_edges]

            # Filter out experts already in S
            keep = ~S_mask[edge_expert]
            edge_expert = edge_expert[keep]
            edge_weight = edge_weight[keep]
            edge_token = edge_token[keep]

            if edge_expert.numel() == 0:
                break

            # Score: alpha * H(e) + beta * G(e)
            G = torch.zeros(E, device=hs_flat.device, dtype=topkm_weight.dtype)
            H = torch.zeros(E, device=hs_flat.device, dtype=topkm_weight.dtype)

            gap_gain = torch.minimum(edge_weight, d[edge_token])
            hit_gain = (c[edge_token] + edge_weight >= r[edge_token]).to(
                topkm_weight.dtype)

            G.scatter_add_(0, edge_expert, gap_gain)
            H.scatter_add_(0, edge_expert, hit_gain)

            Score = self.alpha * H + self.beta * G
            Score[S_mask] = -1e30

            _, add_order = Score.sort(descending=True)
            new_experts = add_order[:self.per_round_cap]
            valid = Score[new_experts] > 0
            new_experts = new_experts[valid]

            if new_experts.numel() == 0:
                break

            S_mask[new_experts] = True
            n_rounds += 1

        # Final coverage for stats
        in_S_final = S_mask[topkm_idx]
        c_final = (topkm_weight * in_S_final.float()).sum(dim=1)
        sat_final = (c_final >= r).float().mean().item()
        active_count = S_mask.sum().item()

        self.stats["active_experts"].append(active_count)
        self.stats["sat_ratio"].append(sat_final)
        self.stats["n_rounds"].append(n_rounds)

        # Step 7: Reroute within S (unchanged gate.routing)
        masked_logits = gate_logits.masked_fill(
            ~S_mask.unsqueeze(0), float('-inf'))                   # [N, E]
        topk_weight_new, topk_idx_new = moe_mod.gate.routing(
            hs_flat, masked_logits, moe_mod.gate.top_k, True)     # [N, K], [N, K]

        # Step 8: Optional final top-p pruning
        if self.use_topp:
            sorted_tw, sort_ord = topk_weight_new.sort(dim=1, descending=True)
            total_routing = topk_weight_new.sum(dim=1, keepdim=True)
            needed_frac = (self.top_p - SHARED_RATE) / ROUTING_RATE
            threshold = needed_frac * total_routing
            cumsum = sorted_tw.cumsum(dim=1)
            enough = cumsum >= threshold
            enough[:, -1] = True
            cutoff = enough.float().argmax(dim=1) + 1

            rank_pos = torch.arange(
                TOP_K_ORIG, device=hs_flat.device).unsqueeze(0)
            keep_sorted = rank_pos < cutoff.unsqueeze(1)
            pruning_mask = torch.zeros_like(topk_weight_new, dtype=torch.bool)
            pruning_mask.scatter_(1, sort_ord, keep_sorted)

            kept_sum = (topk_weight_new * pruning_mask.float()).sum(
                dim=1, keepdim=True)
            orig_sum = topk_weight_new.sum(dim=1, keepdim=True)
            scale = orig_sum / (kept_sum + 1e-8)
            new_weights = topk_weight_new * pruning_mask.float() * scale
        else:
            new_weights = topk_weight_new

        # Step 9: fused_experts (unchanged)
        routed_y = fused_experts(
            hidden_states=hs_flat,
            w1=moe_mod.experts.w13_weight,
            w2=moe_mod.experts.w2_weight,
            topk_weights=new_weights,
            topk_ids=topk_idx_new,
            inplace=False)

        routed_y = routed_y.view(bsz, seq_len, h)
        out = routed_y + shared_res if moe_mod.config.num_shared_experts is not None else routed_y
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
    print("Expanded Top-(K+M) Budgeted Routing — E2E Verification")
    print("  Baseline: max_unroll=4 + Fused RMSNorm + Classic flash-attn 2.8.3")
    print("  HetEval-32: batch=32, gen=256, block=32, threshold=0.90")
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

        # Apply baseline optimizations
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
        # Configs
        # ==============================================================
        CONFIGS = [
            # (name, M, K_target, QF, use_topp)
            ("C0:baseline",    None, None, None, False),
            ("C1:M4_tp_on",    4,    40,   0.85, True),
            ("C2:M4_tp_off",   4,    40,   0.85, False),
            ("C3:M2_tp_off",   2,    40,   0.85, False),
            ("C4:M8_tp_off",   8,    40,   0.85, False),
        ]

        results = {}

        for cname, M, K_target, QF, use_topp in CONFIGS:
            print(f"\n{'='*60}")
            print(f"{cname}: M={M}, K_target={K_target}, QF={QF}, top-p={'ON' if use_topp else 'OFF'}")
            print(f"{'='*60}")

            if M is not None:
                ctrl = ExpandedBudgetedRoutingController(
                    K=TOP_K_ORIG, M=M, K_target=K_target,
                    quality_floor=QF, q_major=0.95,
                    per_round_cap=64, max_add_rounds=3,
                    alpha=1.0, beta=0.5,
                    use_topp=use_topp, top_p=0.75)
                hooks = install_hooks(model, ctrl)

            # Timing run (temp=0)
            dllm = make_dllm(decoder_t0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fwd = dllm.diff_iteration.num_forwards
            wall = t1 - t0

            if M is not None:
                avg_active = sum(ctrl.stats["active_experts"]) / max(len(ctrl.stats["active_experts"]), 1)
                avg_sat = sum(ctrl.stats["sat_ratio"]) / max(len(ctrl.stats["sat_ratio"]), 1)
                avg_rounds = sum(ctrl.stats["n_rounds"]) / max(len(ctrl.stats["n_rounds"]), 1)
                print(f"  Fwd={fwd}, time={wall:.3f}s, fwd/s={fwd/wall:.1f}")
                print(f"  Avg |S|={avg_active:.1f}, sat_ratio={avg_sat:.3f}, avg_rounds={avg_rounds:.1f}")
            else:
                avg_active = 220.8  # all experts active
                avg_sat = 1.0
                avg_rounds = 0
                print(f"  Fwd={fwd}, time={wall:.3f}s, fwd/s={fwd/wall:.1f}")

            results[cname] = {
                "fwd": fwd, "time_s": wall,
                "avg_active": avg_active, "avg_sat": avg_sat,
                "avg_rounds": avg_rounds,
                "M": M, "K_target": K_target, "use_topp": use_topp,
            }

            if M is not None:
                remove_hooks(hooks)

        # ==============================================================
        # Quality check (temp=0.7)
        # ==============================================================
        print(f"\n{'='*80}")
        print("QUALITY CHECK (temp=0.7)")
        print(f"{'='*80}")

        decoder_t7 = ThresholdParallelDecoder(
            temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        for cname, M, K_target, QF, use_topp in CONFIGS:
            if M is not None:
                ctrl = ExpandedBudgetedRoutingController(
                    K=TOP_K_ORIG, M=M, K_target=K_target,
                    quality_floor=QF, q_major=0.95,
                    per_round_cap=64, max_add_rounds=3,
                    alpha=1.0, beta=0.5,
                    use_topp=use_topp, top_p=0.75)
                hooks = install_hooks(model, ctrl)

            dllm = make_dllm(decoder_t7)
            # Warmup
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            # Generate
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

            if M is not None:
                remove_hooks(hooks)

        # ==============================================================
        # Summary
        # ==============================================================
        bl_fwd = results["C0:baseline"]["fwd"]
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Config':<20s} {'Fwd':>4s} {'dFwd':>5s} {'Time':>7s} "
              f"{'|S|':>6s} {'Sat%':>6s} {'Rnds':>5s} {'top-p':>5s}")
        print(f"  {'-'*60}")
        for cname in results:
            r = results[cname]
            delta = r["fwd"] - bl_fwd
            tp_str = "ON" if r.get("use_topp") else "OFF"
            if cname == "C0:baseline":
                tp_str = "—"
            print(f"  {cname:<20s} {r['fwd']:>4d} {delta:>+4d} "
                  f"{r['time_s']:>7.1f}s {r['avg_active']:>6.1f} "
                  f"{r['avg_sat']:>5.1%} {r['avg_rounds']:>5.1f} {tp_str:>5s}")

        # Save
        out_path = (REPO_ROOT / "codex_coding" / "results" /
                    "expanded_budgeted_routing_e2e.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
