#!/usr/bin/env python3
"""
v0.1.15.2 — Active Expert Count Analysis

For each Pareto configuration, measure per-layer unique active expert counts
to determine whether pair-level savings translate to expert-level HBM savings.

Configs:
  C0: baseline (top-8, no reuse)
  C1: D2:tp75 (top-p=0.75, no reuse)
  C2: D1+D2:tp75_m90 (top-p=0.75 + margin>0.90, L4-14)
  C3: D1+D2:tp70_m70_d02 (top-p=0.70 + margin>0.70, drift<0.02, L4-14)

batch=32, 32 heterogeneous prompts, gen_length=128, temp=0
"""

from __future__ import annotations
import os, sys, socket, json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
TOP_K_ORIG = 8
SHARED_RATE = 0.419
ROUTING_RATE = 0.581
NUM_EXPERTS = 256
EXPERT_W1_MB = 1024 * 2048 * 2 / 1e6  # ~4MB
EXPERT_W2_MB = 2048 * 512 * 2 / 1e6   # ~2MB
EXPERT_TOTAL_MB = EXPERT_W1_MB + EXPERT_W2_MB  # ~6MB

PROMPTS = [
    "Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h. The total distance between the two cities is 240 km. What is the average speed for the entire round trip?\n\nProblem 2: A rectangular garden has a perimeter of 56 meters.",
    "Write a detailed essay about the history of artificial intelligence, covering the Dartmouth conference of 1956, the AI winters, the rise of machine learning in the 1990s, and deep learning breakthroughs.",
    "You are a chemistry professor. Explain Le Chatelier's principle with examples and how it applies to industrial ammonia production via the Haber process.",
    "Design a complete REST API for an e-commerce platform with endpoints for user authentication, product management, shopping cart operations, and order processing.",
    "Analyze the global economic impact of climate change across agriculture, energy, real estate, and healthcare sectors with specific examples.",
    "Explain quantum computing to a classical CS background: qubits, superposition, entanglement, Shor's algorithm, and current hardware approaches.",
    "You are a systems architect. Design a distributed message queue with partition-based storage, consumer groups, replication, and exactly-once semantics.",
    "Write a comprehensive guide to training large language models covering data collection, tokenizer training, architecture decisions, and distributed training strategies.",
    "Compare and contrast TCP and UDP protocols, including their use cases in modern distributed systems, gaming, streaming, and IoT applications.",
    "Explain the mathematical foundations of neural networks: backpropagation, gradient descent, loss functions, and the universal approximation theorem.",
    "Design a microservices architecture for a ride-sharing application with real-time matching, pricing, routing, payments, and driver management.",
    "Write about the history of cryptography from Caesar ciphers through RSA, elliptic curve cryptography, and post-quantum cryptographic algorithms.",
    "Explain database indexing strategies: B-trees, hash indexes, bitmap indexes, and their trade-offs for OLTP vs OLAP workloads.",
    "Discuss the economic and social implications of universal basic income with examples from pilot programs in Finland, Kenya, and the United States.",
    "Design a CI/CD pipeline for a large monorepo with microservices, including build caching, parallel testing, canary deployments, and rollback strategies.",
    "Explain the theory of relativity to a physics undergraduate, covering special relativity, time dilation, length contraction, and general relativity basics.",
    "Write a comprehensive comparison of Python, Rust, and Go for systems programming, covering memory safety, concurrency models, and ecosystem maturity.",
    "Design a real-time recommendation engine for a video streaming platform that handles cold start, user preferences, and content diversity.",
    "Explain the CAP theorem and its practical implications for distributed database design, with examples from Cassandra, MongoDB, and CockroachDB.",
    "Write about the evolution of computer graphics from rasterization to ray tracing, including GPU architecture changes and real-time rendering techniques.",
    "Design a fraud detection system for a payment processing company using machine learning, rule engines, and real-time streaming analytics.",
    "Explain compiler optimization techniques including SSA form, loop unrolling, vectorization, and register allocation strategies.",
    "Write about the history and future of space exploration, from Apollo missions through SpaceX reusability to planned Mars colonization.",
    "Design an observability platform with distributed tracing, log aggregation, metrics collection, and intelligent alerting for microservices.",
    "Explain the mathematics behind public key cryptography, including modular arithmetic, Euler's theorem, and the RSA algorithm step by step.",
    "Write a guide to modern CSS layout techniques including Flexbox, Grid, Container Queries, and responsive design best practices.",
    "Design a multi-tenant SaaS platform architecture with data isolation, custom domains, billing integration, and horizontal scaling.",
    "Explain how garbage collectors work in JVM, Go, and Python, comparing mark-sweep, generational, and reference counting approaches.",
    "Write about the ethical implications of large language models including bias, misinformation, copyright, and environmental impact.",
    "Design a real-time collaborative document editor like Google Docs with conflict resolution, offline support, and version history.",
    "Explain operating system memory management: virtual memory, page tables, TLB, demand paging, and memory-mapped files.",
    "Write a comprehensive guide to Kubernetes architecture including pods, services, ingress, operators, and cluster autoscaling.",
]


class ExpertCountController:
    """Collects per-layer active expert statistics during generation."""

    def __init__(self, top_p=None, margin_threshold=None, drift_budget=None,
                 reuse_layers=None):
        # D2 config
        self.top_p = top_p
        # D1 config
        self.margin_threshold = margin_threshold
        self.drift_budget = drift_budget
        self.reuse_layers = reuse_layers or set()

        # State
        self.routed_cache = {}
        self.shared_cache = {}
        self.qualifying_mask = None
        self.step = 0

        # Statistics: per (step, layer)
        self.layer_stats = defaultdict(list)
        # Each entry: {active_baseline, active_after_opt, tokens_total, tokens_fresh,
        #              per_expert_hist_baseline, per_expert_hist_opt}

    def reset_block(self):
        self.routed_cache.clear()
        self.shared_cache.clear()
        self.qualifying_mask = None
        self.step = 0

    def update(self, logits):
        if self.margin_threshold is not None:
            with torch.no_grad():
                probs = F.softmax(logits.float(), dim=-1)
                top2 = probs.topk(2, dim=-1).values
                self.qualifying_mask = (top2[:, :, 0] - top2[:, :, 1]) > self.margin_threshold
        self.step += 1

    def hook_forward(self, moe_mod, layer_idx, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        n_tokens = bsz * seq_len

        # Always compute shared (fresh)
        shared_res = moe_mod.shared_experts(hidden_states)

        # Get full routing for ALL tokens
        topk_idx, topk_weight, _ = moe_mod.gate(hs_flat)
        # topk_idx: [N, 8], topk_weight: [N, 8]

        # === Baseline active experts (all tokens, all 8 experts) ===
        baseline_active = topk_idx.unique().numel()
        baseline_hist = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device="cpu")
        for eid in topk_idx.view(-1).cpu():
            baseline_hist[eid.item()] += 1

        # === Apply D2 (top-p pruning) to weights ===
        if self.top_p is not None:
            sorted_w, sort_order = topk_weight.sort(dim=1, descending=True)
            total_routing = topk_weight.sum(dim=1, keepdim=True)
            needed_frac = (self.top_p - SHARED_RATE) / ROUTING_RATE
            threshold = needed_frac * total_routing
            cumsum = sorted_w.cumsum(dim=1)
            enough = (cumsum >= threshold)
            enough[:, -1] = True
            cutoff = enough.float().argmax(dim=1) + 1
            rank_pos = torch.arange(TOP_K_ORIG, device=topk_weight.device).unsqueeze(0)
            keep_sorted = rank_pos < cutoff.unsqueeze(1)
            pruning_mask = torch.zeros_like(topk_weight, dtype=torch.bool)
            pruning_mask.scatter_(1, sort_order, keep_sorted)
        else:
            pruning_mask = torch.ones_like(topk_weight, dtype=torch.bool)

        # === Apply D1 (token reuse) — determine fresh tokens ===
        fresh_token_mask = torch.ones(n_tokens, dtype=torch.bool, device=hs_flat.device)
        if (self.margin_threshold is not None
                and layer_idx in self.reuse_layers
                and self.qualifying_mask is not None
                and layer_idx in self.routed_cache):
            cached = self.routed_cache[layer_idx]
            if cached.shape[0] >= bsz and cached.shape[1] == seq_len:
                reuse_mask = self.qualifying_mask[:bsz].clone()
                # Drift guard
                if self.drift_budget is not None and layer_idx in self.shared_cache:
                    sp = self.shared_cache[layer_idx][:bsz].float()
                    sn = shared_res[:bsz].float()
                    cos_sim = F.cosine_similarity(sn, sp, dim=-1)
                    drift_ok = ((1.0 - cos_sim) < self.drift_budget)
                    reuse_mask = reuse_mask & drift_ok
                fresh_token_mask = (~reuse_mask).view(-1)

        n_fresh = fresh_token_mask.sum().item()

        # === Optimized active experts: only fresh tokens with non-pruned experts ===
        if n_fresh == 0:
            opt_active = 0
            opt_hist = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device="cpu")
        else:
            # Get the expert IDs that would actually be computed
            fresh_idx = topk_idx[fresh_token_mask]        # [n_fresh, 8]
            fresh_prune = pruning_mask[fresh_token_mask]   # [n_fresh, 8]
            active_ids = fresh_idx[fresh_prune]            # [?] flat
            if active_ids.numel() == 0:
                opt_active = 0
                opt_hist = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device="cpu")
            else:
                opt_active = active_ids.unique().numel()
                opt_hist = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device="cpu")
                for eid in active_ids.cpu():
                    opt_hist[eid.item()] += 1

        # Record stats
        self.layer_stats[layer_idx].append({
            "baseline_active": baseline_active,
            "opt_active": opt_active,
            "n_tokens": n_tokens,
            "n_fresh": n_fresh,
            "baseline_1tok": int((baseline_hist == 1).sum().item()),
            "opt_1tok": int((opt_hist == 1).sum().item()),
        })

        # === Actual computation (monkey-patch style for forward correctness) ===
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        if self.top_p is not None:
            kept_sum = (topk_weight * pruning_mask.float()).sum(dim=1, keepdim=True)
            orig_sum = topk_weight.sum(dim=1, keepdim=True)
            scale = orig_sum / (kept_sum + 1e-8)
            new_weights = topk_weight * pruning_mask.float() * scale
            routed_y = fused_experts(
                hidden_states=hs_flat, w1=moe_mod.experts.w13_weight,
                w2=moe_mod.experts.w2_weight,
                topk_weights=new_weights, topk_ids=topk_idx, inplace=False)
        else:
            router_logits = moe_mod.gate.get_logits(hs_flat)
            routed_y = moe_mod.experts.forward_impl(
                hidden_states=hs_flat, router_logits=router_logits)
        routed_y = routed_y.view(bsz, seq_len, h)

        # D1 reuse: replace stable positions with cache
        if not fresh_token_mask.all() and layer_idx in self.routed_cache:
            reuse_positions = (~fresh_token_mask).view(bsz, seq_len)
            cached_r = self.routed_cache[layer_idx][:bsz]
            routed_y = routed_y.clone()
            routed_y[reuse_positions] = cached_r[reuse_positions].to(routed_y.device)

        # Update cache
        self.routed_cache[layer_idx] = routed_y.detach().clone()
        self.shared_cache[layer_idx] = shared_res.detach().clone()

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


def gen_with_ctrl(dllm, input_ids, ctrl, gl=128):
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner
    oif = BlockDiffusionIteration.forward
    ord_ = BlockDiffusionRunner.decode
    def pd(self_runner, model, decoder, x, kv_cache, block, block_loc,
           block_id, pos_ids, attn_mask, block_length=32, cross_block_attn_mask=None):
        ctrl.reset_block()
        return ord_(self_runner, model, decoder, x, kv_cache, block, block_loc,
                    block_id, pos_ids, attn_mask, block_length, cross_block_attn_mask)
    def pf(self_iter, model, decoder, x, kv_cache, block, block_loc,
           block_id, pos_ids, attn_mask, past_key_values,
           replace_position, backend, is_cross_block=False, block_length=32):
        out = oif(self_iter, model, decoder, x, kv_cache, block, block_loc,
                  block_id, pos_ids, attn_mask, past_key_values,
                  replace_position, backend, is_cross_block, block_length)
        if not is_cross_block:
            ctrl.update(out.logits)
        return out
    BlockDiffusionIteration.forward = pf
    BlockDiffusionRunner.decode = pd
    try:
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            dllm.diff_iteration.iter_no = 0
            out = dllm.generate(input_ids.clone(), gen_length=gl, block_length=BLOCK_LENGTH)
    finally:
        BlockDiffusionIteration.forward = oif
        BlockDiffusionRunner.decode = ord_
    return out


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

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print("v0.1.15.2 — Active Expert Count Analysis")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)

        BATCH_SIZE = 32
        all_ids = []
        for i in range(BATCH_SIZE):
            text = PROMPTS[i % len(PROMPTS)]
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
                ids = torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
            padded.append(ids)
        input_ids = torch.stack(padded, dim=0).to(device)
        print(f"Input: {input_ids.shape}")

        decoder = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm = BlockDiffusionLLM(
            model, decoder,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=1, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Warmup
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print("Warmup done.\n")

        CONFIGS = [
            ("C0:baseline",       None,  None, None, None),
            ("C1:D2_tp75",        0.75,  None, None, None),
            ("C2:D1D2_tp75_m90",  0.75,  0.90, None, set(range(4, 15))),
            ("C3:D1D2_tp70_m70",  0.70,  0.70, 0.02, set(range(4, 15))),
        ]

        all_results = {}

        for cname, top_p, margin, drift, reuse_layers in CONFIGS:
            print(f"{'='*60}")
            print(f"Running: {cname}")
            print(f"  top_p={top_p}, margin={margin}, drift={drift}")
            print(f"{'='*60}")

            ctrl = ExpertCountController(
                top_p=top_p, margin_threshold=margin,
                drift_budget=drift, reuse_layers=reuse_layers or set())
            hooks = install_hooks(model, ctrl)
            try:
                out = gen_with_ctrl(dllm, input_ids, ctrl)
                fwd_count = dllm.diff_iteration.num_forwards
            finally:
                remove_hooks(hooks)

            print(f"  Forwards: {fwd_count}")

            # Aggregate per-layer stats
            layer_summary = {}
            for li in range(19):
                entries = ctrl.layer_stats[li]
                if not entries:
                    continue
                avg_bl = sum(e["baseline_active"] for e in entries) / len(entries)
                avg_opt = sum(e["opt_active"] for e in entries) / len(entries)
                avg_fresh = sum(e["n_fresh"] for e in entries) / len(entries)
                avg_total = sum(e["n_tokens"] for e in entries) / len(entries)
                avg_bl_1tok = sum(e["baseline_1tok"] for e in entries) / len(entries)
                avg_opt_1tok = sum(e["opt_1tok"] for e in entries) / len(entries)
                layer_summary[li] = {
                    "baseline_active": avg_bl,
                    "opt_active": avg_opt,
                    "eliminated": avg_bl - avg_opt,
                    "fresh_ratio": avg_fresh / max(avg_total, 1),
                    "baseline_1tok": avg_bl_1tok,
                    "opt_1tok": avg_opt_1tok,
                }

            all_results[cname] = {
                "fwd_count": fwd_count,
                "layer_summary": layer_summary,
            }

        # ============================================================
        # TABLE 1: Per-layer active expert counts
        # ============================================================
        print(f"\n{'='*80}")
        print(f"TABLE 1: Average Active Experts per Layer")
        print(f"{'='*80}")
        print(f"  {'Layer':>5s}  {'C0:base':>8s}  {'C1:tp75':>8s}  {'C2:+m90':>8s}  {'C3:+m70':>8s}  "
              f"{'C1-elim':>7s}  {'C2-elim':>7s}  {'C3-elim':>7s}")
        print(f"  {'-'*75}")

        for li in range(19):
            vals = []
            for cname, _, _, _, _ in CONFIGS:
                ls = all_results[cname]["layer_summary"].get(li, {})
                vals.append(ls)
            if not vals[0]:
                continue
            bl = vals[0]["baseline_active"]
            print(f"  {li:>5d}  {bl:>8.1f}", end="")
            for v in vals[1:]:
                print(f"  {v.get('opt_active', 0):>8.1f}", end="")
            for v in vals[1:]:
                elim = v.get("eliminated", 0)
                print(f"  {elim:>+7.1f}", end="")
            print()

        # Averages
        print(f"  {'AVG':>5s}", end="")
        for ci, (cname, _, _, _, _) in enumerate(CONFIGS):
            ls = all_results[cname]["layer_summary"]
            if ci == 0:
                avg = sum(v["baseline_active"] for v in ls.values()) / max(len(ls), 1)
            else:
                avg = sum(v["opt_active"] for v in ls.values()) / max(len(ls), 1)
            print(f"  {avg:>8.1f}", end="")
        for ci, (cname, _, _, _, _) in enumerate(CONFIGS):
            if ci == 0:
                print(f"  {'—':>7s}", end="")
            else:
                ls = all_results[cname]["layer_summary"]
                avg_elim = sum(v["eliminated"] for v in ls.values()) / max(len(ls), 1)
                print(f"  {avg_elim:>+7.1f}", end="")
        print()

        # ============================================================
        # TABLE 2: Theoretical HBM savings
        # ============================================================
        print(f"\n{'='*80}")
        print(f"TABLE 2: Theoretical HBM Savings per Forward")
        print(f"{'='*80}")
        print(f"  {'Config':<25s}  {'Fwds':>4s}  {'Avg Active':>10s}  {'vs Base':>7s}  "
              f"{'HBM/fwd':>8s}  {'Total HBM':>10s}  {'Saved':>6s}")
        print(f"  {'-'*75}")

        baseline_avg_active = None
        baseline_total_hbm = None

        for cname, _, _, _, _ in CONFIGS:
            r = all_results[cname]
            ls = r["layer_summary"]
            fwd = r["fwd_count"]

            if cname.startswith("C0"):
                avg_active = sum(v["baseline_active"] for v in ls.values()) / max(len(ls), 1)
                baseline_avg_active = avg_active
            else:
                avg_active = sum(v["opt_active"] for v in ls.values()) / max(len(ls), 1)

            hbm_per_fwd = avg_active * EXPERT_TOTAL_MB * 19  # 19 MoE layers
            total_hbm = hbm_per_fwd * fwd

            if baseline_total_hbm is None:
                baseline_total_hbm = total_hbm

            vs_base = (avg_active / baseline_avg_active - 1) * 100 if baseline_avg_active else 0
            saved = (1 - total_hbm / baseline_total_hbm) * 100 if baseline_total_hbm else 0

            print(f"  {cname:<25s}  {fwd:>4d}  {avg_active:>10.1f}  {vs_base:>+6.1f}%  "
                  f"{hbm_per_fwd:>7.0f}MB  {total_hbm:>9.0f}MB  {saved:>5.1f}%")

        # ============================================================
        # TABLE 3: Per-expert token distribution (sample layers)
        # ============================================================
        print(f"\n{'='*80}")
        print(f"TABLE 3: 1-Token Expert Count (most easily eliminated)")
        print(f"{'='*80}")
        print(f"  {'Layer':>5s}  {'C0 1-tok':>10s}  {'C1 1-tok':>10s}  {'C2 1-tok':>10s}  {'C3 1-tok':>10s}")
        print(f"  {'-'*50}")

        for li in [0, 4, 9, 14, 18]:
            print(f"  {li:>5d}", end="")
            for ci, (cname, _, _, _, _) in enumerate(CONFIGS):
                ls = all_results[cname]["layer_summary"].get(li, {})
                if ci == 0:
                    v = ls.get("baseline_1tok", 0)
                else:
                    v = ls.get("opt_1tok", 0)
                print(f"  {v:>10.1f}", end="")
            print()

        # ============================================================
        # TABLE 4: Fresh token ratio for D1 configs
        # ============================================================
        print(f"\n{'='*80}")
        print(f"TABLE 4: Fresh Token Ratio (D1 configs only)")
        print(f"{'='*80}")
        for cname in ["C2:D1D2_tp75_m90", "C3:D1D2_tp70_m70"]:
            ls = all_results[cname]["layer_summary"]
            print(f"\n  {cname}:")
            for li in [0, 4, 9, 14, 18]:
                v = ls.get(li, {})
                fr = v.get("fresh_ratio", 1.0) * 100
                print(f"    Layer {li}: {fr:.1f}% fresh")

        # ============================================================
        # CONCLUSION
        # ============================================================
        print(f"\n{'='*80}")
        print(f"CONCLUSION")
        print(f"{'='*80}")

        c1_avg = sum(v["opt_active"] for v in all_results["C1:D2_tp75"]["layer_summary"].values()) / 19
        bl_avg = baseline_avg_active
        reduction_pct = (1 - c1_avg / bl_avg) * 100 if bl_avg else 0

        print(f"  Baseline avg active experts: {bl_avg:.1f}")
        print(f"  D2:tp75 avg active experts:  {c1_avg:.1f}")
        print(f"  Unique expert reduction:     {reduction_pct:.1f}%")
        print(f"  (vs pair-level savings of 51.5%)")

        if reduction_pct < 10:
            print(f"\n  >>> Pair savings do NOT translate to expert-level savings.")
            print(f"  >>> Most experts remain active. HBM traffic barely reduced.")
            print(f"  >>> Physical top-k reduction alone cannot yield wall-clock speedup.")
        elif reduction_pct < 30:
            print(f"\n  >>> PARTIAL translation: {reduction_pct:.0f}% expert reduction.")
            print(f"  >>> Some HBM savings possible but less than pair savings suggest.")
        else:
            print(f"\n  >>> GOOD translation: {reduction_pct:.0f}% expert reduction.")
            print(f"  >>> Significant HBM savings achievable.")

        # Save results
        save_data = {}
        for cname, r in all_results.items():
            save_data[cname] = {
                "fwd_count": r["fwd_count"],
                "layer_summary": {str(k): v for k, v in r["layer_summary"].items()},
            }
        out_path = REPO_ROOT / "codex_coding" / "results" / "active_expert_count_analysis.json"
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
