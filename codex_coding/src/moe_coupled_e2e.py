#!/usr/bin/env python3
"""
v0.1.15.1 — Coupled Optimization E2E: Direction 1 x Direction 2

Tests the coupled optimization of:
  Direction 1 (Token Temporal Reuse): skip MoE for stable tokens
  Direction 2 (Expert Adaptive Pruning): top-p expert selection

Architecture: D2 as always-on base layer, D1 as additive layer on top.
Decision logic per (step, layer, token):
  stable token + safe layer -> full skip (D1, save 100% expert compute)
  otherwise                 -> top-p pruning (D2, save ~50% expert compute)

Measures:
  - delta_fwd: forward count change vs baseline
  - total_expert_savings: combined D1 + D2 savings percentage
  - reuse_rate: D1 coverage (fraction of positions reused)
  - avg_expert: average experts computed per non-reused position
  - output quality: temp=0.7 text comparison

batch=32, gen_length=128, block_length=32, threshold=0.90
"""

from __future__ import annotations
import os, sys, time, socket, json
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


class CoupledController:
    """Controls both Direction 1 (temporal reuse) and Direction 2 (top-p pruning).

    For each (step, layer, token) position:
      - D1 eligible (stable + safe layer): reuse cached output, 0 expert slots
      - Otherwise: compute with top-p pruning, ~4 expert slots (or 8 if no pruning)
    """

    def __init__(self, top_p=None, margin_threshold=None, drift_budget=None):
        # Direction 2 config
        self.top_p = top_p  # None = no pruning (full top-8)

        # Direction 1 config
        self.margin_threshold = margin_threshold  # None = no reuse
        self.drift_budget = drift_budget  # None = no drift guard
        self.reuse_layers = set(range(4, 15))  # MoE layers 4-14

        # State (reset per block)
        self.routed_cache = {}
        self.shared_cache = {}
        self.qualifying_mask = None  # [bsz, seq_len] bool
        self.step = 0

        # Stats (accumulated across entire generation)
        self.total_positions = 0
        self.d1_reused = 0
        self.total_computed_expert_slots = 0  # what real impl would compute

    def reset_block(self):
        """Called at each new block boundary."""
        self.routed_cache.clear()
        self.shared_cache.clear()
        self.qualifying_mask = None
        self.step = 0

    def update(self, logits):
        """Update qualifying mask from decoder logits (called after each forward)."""
        if self.margin_threshold is None:
            return
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            top2 = probs.topk(2, dim=-1).values
            margin = top2[:, :, 0] - top2[:, :, 1]
            self.qualifying_mask = margin > self.margin_threshold
        self.step += 1

    def hook_forward(self, moe_mod, layer_idx, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        n_tokens = hs_flat.shape[0]
        device = hs_flat.device

        # Shared expert (always fresh)
        shared_res = moe_mod.shared_experts(hidden_states)

        # --- Direction 2: compute routed output (with or without top-p) ---
        if self.top_p is not None:
            topk_idx, topk_weight, _ = moe_mod.gate(hs_flat)
            sorted_w, sort_order = topk_weight.sort(dim=1, descending=True)

            # Global top-p threshold
            total_routing = topk_weight.sum(dim=1, keepdim=True)
            needed_frac = (self.top_p - SHARED_RATE) / ROUTING_RATE
            threshold = needed_frac * total_routing
            cumsum = sorted_w.cumsum(dim=1)
            enough = (cumsum >= threshold)
            enough[:, -1] = True
            cutoff = enough.float().argmax(dim=1) + 1  # [n_tokens], 1-indexed

            # Build pruning mask
            rank_pos = torch.arange(TOP_K_ORIG, device=device).unsqueeze(0)
            keep_sorted = rank_pos < cutoff.unsqueeze(1)
            mask_pruning = torch.zeros_like(topk_weight)
            mask_pruning.scatter_(1, sort_order, keep_sorted.float())

            # Renormalize
            kept_sum = (topk_weight * mask_pruning).sum(dim=1, keepdim=True)
            orig_sum = topk_weight.sum(dim=1, keepdim=True)
            scale = orig_sum / (kept_sum + 1e-8)
            new_weights = topk_weight * mask_pruning * scale

            from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
            routed_y = fused_experts(
                hidden_states=hs_flat,
                w1=moe_mod.experts.w13_weight,
                w2=moe_mod.experts.w2_weight,
                topk_weights=new_weights,
                topk_ids=topk_idx,
                inplace=False,
            )
        else:
            # No pruning: standard path
            router_logits = moe_mod.gate.get_logits(hs_flat)
            routed_y = moe_mod.experts.forward_impl(
                hidden_states=hs_flat, router_logits=router_logits)
            cutoff = None

        routed_y = routed_y.view(bsz, seq_len, h)

        # --- Direction 1: temporal reuse for qualifying positions ---
        # per_token_experts tracks what a real implementation would compute
        per_token_experts = torch.zeros(n_tokens, device=device)

        d1_active = (self.margin_threshold is not None
                     and layer_idx in self.reuse_layers
                     and self.qualifying_mask is not None
                     and layer_idx in self.routed_cache)

        reuse_mask_flat = torch.zeros(n_tokens, dtype=torch.bool, device=device)

        if d1_active:
            cached = self.routed_cache[layer_idx]
            if cached.shape[0] >= bsz and cached.shape[1] == seq_len:
                mask_reuse = self.qualifying_mask[:bsz].clone()

                # Drift guard
                if self.drift_budget is not None and layer_idx in self.shared_cache:
                    sp = self.shared_cache[layer_idx][:bsz].float()
                    sn = shared_res[:bsz].float()
                    cos_sim = F.cosine_similarity(sn, sp, dim=-1)
                    drift_ok = ((1.0 - cos_sim) < self.drift_budget)
                    mask_reuse = mask_reuse & drift_ok

                n_reused = mask_reuse.sum().item()
                if n_reused > 0:
                    routed_y = routed_y.clone()
                    routed_y[mask_reuse] = cached[:bsz][mask_reuse].to(device)
                    reuse_mask_flat = mask_reuse.view(-1)
                    self.d1_reused += n_reused

        # --- Stats: per-token expert count ---
        if cutoff is not None:
            per_token_experts = cutoff.float().clone()
        else:
            per_token_experts = torch.full((n_tokens,), TOP_K_ORIG, dtype=torch.float, device=device)

        # D1 reused positions: 0 expert slots
        if reuse_mask_flat.any():
            per_token_experts[reuse_mask_flat] = 0.0

        self.total_positions += n_tokens
        self.total_computed_expert_slots += per_token_experts.sum().item()

        # --- Update cache ---
        self.routed_cache[layer_idx] = routed_y.detach().clone()
        self.shared_cache[layer_idx] = shared_res.detach().clone()

        # --- Combine ---
        out = routed_y + shared_res if moe_mod.config.num_shared_experts is not None else routed_y
        return out

    def get_summary(self):
        baseline_slots = self.total_positions * TOP_K_ORIG
        savings_pct = (1 - self.total_computed_expert_slots / max(baseline_slots, 1)) * 100
        reuse_pct = self.d1_reused / max(self.total_positions, 1) * 100
        non_reused = self.total_positions - self.d1_reused
        avg_expert = self.total_computed_expert_slots / max(non_reused, 1)
        return {
            "total_positions": self.total_positions,
            "d1_reused": self.d1_reused,
            "reuse_pct": reuse_pct,
            "total_computed_expert_slots": int(self.total_computed_expert_slots),
            "baseline_slots": baseline_slots,
            "savings_pct": savings_pct,
            "avg_expert_non_reused": avg_expert,
        }


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


def gen_with_ctrl(dllm, input_ids, ctrl, gl=128, enabled=True):
    """Generate with coupled controller, patching block/iteration boundaries."""
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner

    orig_iter_fwd = BlockDiffusionIteration.forward
    orig_runner_decode = BlockDiffusionRunner.decode

    def patched_decode(self_runner, model, decoder, x, kv_cache, block, block_loc,
                       block_id, pos_ids, attn_mask, block_length=32,
                       cross_block_attn_mask=None):
        ctrl.reset_block()
        return orig_runner_decode(self_runner, model, decoder, x, kv_cache, block,
                                  block_loc, block_id, pos_ids, attn_mask,
                                  block_length, cross_block_attn_mask)

    def patched_iter_fwd(self_iter, model, decoder, x, kv_cache, block, block_loc,
                         block_id, pos_ids, attn_mask, past_key_values,
                         replace_position, backend, is_cross_block=False,
                         block_length=32):
        out = orig_iter_fwd(self_iter, model, decoder, x, kv_cache, block, block_loc,
                            block_id, pos_ids, attn_mask, past_key_values,
                            replace_position, backend, is_cross_block, block_length)
        if not is_cross_block and enabled:
            ctrl.update(out.logits)
        return out

    BlockDiffusionIteration.forward = patched_iter_fwd
    BlockDiffusionRunner.decode = patched_decode
    try:
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            dllm.diff_iteration.iter_no = 0
            out = dllm.generate(input_ids.clone(), gen_length=gl, block_length=BLOCK_LENGTH)
    finally:
        BlockDiffusionIteration.forward = orig_iter_fwd
        BlockDiffusionRunner.decode = orig_runner_decode
    return out


# ---- Configuration table ----
# (name, top_p, margin_threshold, drift_budget)
CONFIGS = [
    # Baselines
    ("baseline",           None, None, None),
    ("D2:tp80",            0.80, None, None),
    ("D2:tp75",            0.75, None, None),
    ("D2:tp70",            0.70, None, None),
    # D1-only references
    ("D1:m90",             None, 0.90, None),
    ("D1:m70_d02",         None, 0.70, 0.02),
    # Coupled: conservative D2 (0.80) x D1
    ("D1+D2:tp80_m90",    0.80, 0.90, None),
    ("D1+D2:tp80_m70_d02", 0.80, 0.70, 0.02),
    # Coupled: optimal D2 (0.75) x D1
    ("D1+D2:tp75_m90",    0.75, 0.90, None),
    ("D1+D2:tp75_m70_d02", 0.75, 0.70, 0.02),
    ("D1+D2:tp75_m50_d01", 0.75, 0.50, 0.01),
    # Coupled: aggressive D2 (0.70) x D1
    ("D1+D2:tp70_m70_d02", 0.70, 0.70, 0.02),
    ("D1+D2:tp70_m50_d01", 0.70, 0.50, 0.01),
]


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)
    sys.path.insert(0, str(REPO_ROOT / "lib_cite" / "dInfer" / "python"))

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
    print("v0.1.15.1 — Coupled Optimization E2E: Direction 1 x Direction 2")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        # Warmup
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)

        # Prepare batch
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
        prompt_len = input_ids.shape[1]
        print(f"Input: {input_ids.shape} (batch={BATCH_SIZE}, prompt_len={prompt_len})")

        # Build dllm engine
        decoder_t0 = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm = BlockDiffusionLLM(
            model, decoder_t0,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=1, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Warmup generate
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        # ================================================================
        # PART 1: Quantitative (temp=0, N_RUNS per config)
        # ================================================================
        N_RUNS = 5
        print(f"\n{'='*80}")
        print(f"PART 1: Quantitative sweep (temp=0, {N_RUNS} runs, batch={BATCH_SIZE})")
        print(f"{'='*80}")

        all_results = {}
        for cname, top_p, margin, drift in CONFIGS:
            fwds_list = []
            summaries = []

            for r in range(N_RUNS):
                ctrl = CoupledController(top_p=top_p, margin_threshold=margin,
                                         drift_budget=drift)
                hooks = install_hooks(model, ctrl)
                try:
                    out = gen_with_ctrl(dllm, input_ids, ctrl, gl=128,
                                        enabled=(margin is not None))
                    fwds_list.append(dllm.diff_iteration.num_forwards)
                    summaries.append(ctrl.get_summary())
                    if r == 0:
                        gen_tokens = out[:, prompt_len:].cpu()
                finally:
                    remove_hooks(hooks)

            avg_fwd = sum(fwds_list) / len(fwds_list)
            # Average summary stats across runs
            avg_savings = sum(s["savings_pct"] for s in summaries) / len(summaries)
            avg_reuse = sum(s["reuse_pct"] for s in summaries) / len(summaries)
            avg_expert = sum(s["avg_expert_non_reused"] for s in summaries) / len(summaries)

            all_results[cname] = {
                "top_p": top_p, "margin": margin, "drift": drift,
                "fwds": fwds_list, "avg_fwd": avg_fwd,
                "savings_pct": avg_savings, "reuse_pct": avg_reuse,
                "avg_expert": avg_expert,
                "summaries": summaries,
            }

            print(f"\n  {cname:>24s}: fwds=[{','.join(str(f) for f in fwds_list)}] "
                  f"avg_fwd={avg_fwd:.0f}")
            print(f"  {'':>24s}  savings={avg_savings:.1f}%  reuse={avg_reuse:.1f}%  "
                  f"avg_expert={avg_expert:.2f}")

        # ================================================================
        # SUMMARY TABLE
        # ================================================================
        bl_fwd = all_results["baseline"]["avg_fwd"]
        print(f"\n{'='*80}")
        print(f"SUMMARY TABLE (baseline avg_fwd={bl_fwd:.0f})")
        print(f"{'='*80}")
        header = (f"  {'Config':>24s} {'AvgFwd':>7s} {'dFwd':>5s} "
                  f"{'Save%':>6s} {'Reuse%':>7s} {'AvgExp':>7s} {'Verdict':>10s}")
        print(header)
        print(f"  {'-'*72}")

        for cname in [c[0] for c in CONFIGS]:
            r = all_results[cname]
            delta = r["avg_fwd"] - bl_fwd
            # Verdict based on delta_fwd
            if cname == "baseline":
                verdict = "—"
            elif abs(delta) <= 2:
                verdict = "SAFE"
            elif delta <= 5:
                verdict = "MARGINAL"
            elif delta > 5:
                verdict = "BAD"
            else:
                verdict = "GOOD"  # negative delta = fewer forwards
            print(f"  {cname:>24s} {r['avg_fwd']:>7.0f} {delta:>+5.0f} "
                  f"{r['savings_pct']:>5.1f}% {r['reuse_pct']:>6.1f}% "
                  f"{r['avg_expert']:>7.2f} {verdict:>10s}")

        # ================================================================
        # STEP 1 ANALYSIS: D1 reuse rate comparison (pruned vs unpruned)
        # ================================================================
        print(f"\n{'='*80}")
        print(f"STEP 1 ANALYSIS: Does pruning affect D1 reuse rate?")
        print(f"{'='*80}")

        comparisons = [
            ("D1:m90",         "D1+D2:tp75_m90",     "margin>0.90"),
            ("D1:m70_d02",     "D1+D2:tp75_m70_d02", "margin>0.70+drift<0.02"),
        ]
        for d1_only, coupled, label in comparisons:
            r1 = all_results.get(d1_only, {})
            r2 = all_results.get(coupled, {})
            reuse1 = r1.get("reuse_pct", 0)
            reuse2 = r2.get("reuse_pct", 0)
            diff = reuse2 - reuse1
            effect = "POSITIVE" if diff > 1 else ("NEGATIVE" if diff < -1 else "NEUTRAL")
            print(f"  {label:>28s}: D1-only={reuse1:.1f}%  coupled={reuse2:.1f}%  "
                  f"diff={diff:+.1f}%  -> {effect}")

        # ================================================================
        # HYPOTHESIS CHECK
        # ================================================================
        print(f"\n{'='*80}")
        print(f"HYPOTHESIS CHECK")
        print(f"{'='*80}")

        # Check the expected sweet spot: tp75_m70_d02
        sweet = all_results.get("D1+D2:tp75_m70_d02", {})
        sweet_delta = sweet.get("avg_fwd", 0) - bl_fwd
        sweet_reuse = sweet.get("reuse_pct", 0)

        d1_ref = all_results.get("D1:m70_d02", {})
        d1_delta = d1_ref.get("avg_fwd", 0) - bl_fwd

        d2_ref = all_results.get("D2:tp75", {})
        d2_delta = d2_ref.get("avg_fwd", 0) - bl_fwd

        print(f"  Sweet spot (tp75+m70+d02):")
        print(f"    dFwd={sweet_delta:+.0f}  savings={sweet.get('savings_pct',0):.1f}%  "
              f"reuse={sweet_reuse:.1f}%")
        print(f"  D1 alone (m70+d02): dFwd={d1_delta:+.0f}")
        print(f"  D2 alone (tp75):    dFwd={d2_delta:+.0f}")
        print(f"  Sum of independent deltas: {d1_delta + d2_delta:+.0f}")

        if sweet_delta <= d1_delta + d2_delta + 3:
            print(f"  -> Hypothesis C (independent errors) appears to HOLD")
        else:
            print(f"  -> WARNING: Hypothesis C may be BROKEN (error amplification)")

        if sweet_reuse > 0 and d1_ref.get("reuse_pct", 0) > 0:
            ratio = sweet_reuse / d1_ref["reuse_pct"]
            if ratio > 0.8:
                print(f"  -> Hypothesis B (proxy validity) appears to HOLD "
                      f"(reuse ratio={ratio:.2f})")
            else:
                print(f"  -> WARNING: Hypothesis B may be BROKEN "
                      f"(reuse dropped to {ratio:.2f}x)")

        # ================================================================
        # PART 2: Output quality (temp=0.7)
        # ================================================================
        print(f"\n{'='*80}")
        print(f"PART 2: Output quality (temp=0.7)")
        print(f"{'='*80}")

        decoder_t7 = ThresholdParallelDecoder(
            temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm_t7 = BlockDiffusionLLM(
            model, decoder_t7,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=1, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Warmup
        with torch.inference_mode():
            dllm_t7.diff_iteration.num_forwards = 0
            _ = dllm_t7.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        # Configs to check quality
        QUALITY_CHECKS = [
            ("baseline",           None, None, None),
            ("D2:tp75",            0.75, None, None),
            ("D1+D2:tp75_m90",    0.75, 0.90, None),
            ("D1+D2:tp75_m70_d02", 0.75, 0.70, 0.02),
            ("D1+D2:tp70_m70_d02", 0.70, 0.70, 0.02),
        ]

        quality_outputs = {}
        for cname, top_p, margin, drift in QUALITY_CHECKS:
            ctrl = CoupledController(top_p=top_p, margin_threshold=margin,
                                     drift_budget=drift)
            hooks = install_hooks(model, ctrl)
            try:
                out = gen_with_ctrl(dllm_t7, input_ids, ctrl, gl=128,
                                    enabled=(margin is not None))
                quality_outputs[cname] = out[:, prompt_len:].cpu()
            finally:
                remove_hooks(hooks)

        for bi in range(4):
            print(f"\n{'─'*80}")
            print(f"  BATCH {bi}: {PROMPTS[bi][:65]}...")
            print(f"{'─'*80}")
            for cname, _, _, _ in QUALITY_CHECKS:
                gt = quality_outputs[cname][bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                print(f"\n  [{cname}]:")
                print(f"  {text[:300]}")

        # ================================================================
        # Save results JSON
        # ================================================================
        results_path = REPO_ROOT / "codex_coding" / "results" / "coupled_optimization_results.json"
        save_data = {}
        for cname, r in all_results.items():
            save_data[cname] = {
                "top_p": r["top_p"], "margin": r["margin"], "drift": r["drift"],
                "fwds": r["fwds"], "avg_fwd": r["avg_fwd"],
                "savings_pct": r["savings_pct"], "reuse_pct": r["reuse_pct"],
                "avg_expert": r["avg_expert"],
            }
        save_data["_meta"] = {
            "baseline_avg_fwd": bl_fwd,
            "batch_size": BATCH_SIZE,
            "gen_length": 128,
            "block_length": BLOCK_LENGTH,
            "n_runs": N_RUNS,
        }
        with open(results_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {results_path}")
        print(f"\nDone.")


if __name__ == "__main__":
    main()
