#!/usr/bin/env python3
"""
v0.1.15.3 Phase 2b — Boundary Exploration with Text Quality Criterion

Push Expert Budgeting + Top-p + Token Reuse to find the quality boundary.
Quality judged by output text (especially math correctness), NOT by ΔFwd.

gen_length=256, batch=32, temp=0 (1 run for stats) + temp=0.7 (full text output)
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


# ============================================================
# Reuse compute_active_set from main script
# ============================================================
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))
from expert_budgeting_e2e import compute_active_set


# ============================================================
# Combined Controller: Expert Budgeting + Top-p + Token Reuse
# ============================================================
class FullStackController:
    """Three-layer optimization: Expert Budgeting + Top-p + Token Reuse."""

    def __init__(self, K_target=None, quality_floor=0.85, top_p=None,
                 margin_threshold=None, reuse_layers=None):
        # Layer 0: Expert Budgeting
        self.K_target = K_target
        self.quality_floor = quality_floor
        # Layer 1: Top-p
        self.top_p = top_p  # None = no top-p (use all 8)
        # Layer 2: Token Reuse
        self.margin_threshold = margin_threshold  # None = no reuse
        self.reuse_layers = reuse_layers or set()

        # State
        self.routed_cache = {}
        self.shared_cache = {}
        self.qualifying_mask = None

        # Stats
        self.total_active_experts = []
        self.total_experts_per_token = []
        self.total_tokens = 0
        self.reused_tokens = 0

    def reset_block(self):
        self.routed_cache.clear()
        self.shared_cache.clear()
        self.qualifying_mask = None

    def update(self, logits):
        if self.margin_threshold is not None:
            with torch.no_grad():
                probs = F.softmax(logits.float(), dim=-1)
                top2 = probs.topk(2, dim=-1).values
                self.qualifying_mask = (top2[:, :, 0] - top2[:, :, 1]) > self.margin_threshold

    def hook_forward(self, moe_mod, layer_idx, hidden_states):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        N = bsz * seq_len

        shared_res = moe_mod.shared_experts(hidden_states)
        self.total_tokens += N

        # ---- Layer 2: Token Reuse decision ----
        reuse_mask = None
        if (self.margin_threshold is not None
                and layer_idx in self.reuse_layers
                and self.qualifying_mask is not None
                and layer_idx in self.routed_cache):
            cached = self.routed_cache[layer_idx]
            if cached.shape[0] >= bsz and cached.shape[1] == seq_len:
                reuse_mask = self.qualifying_mask[:bsz].clone()
                # Drift guard (shared_cos)
                if layer_idx in self.shared_cache:
                    sp = self.shared_cache[layer_idx][:bsz].float()
                    sn = shared_res[:bsz].float()
                    cos_sim = F.cosine_similarity(sn, sp, dim=-1)
                    reuse_mask = reuse_mask & ((1.0 - cos_sim) < 0.02)

        n_reuse = reuse_mask.sum().item() if reuse_mask is not None else 0
        self.reused_tokens += n_reuse

        # ---- Get gate routing for ALL tokens (needed for budgeting) ----
        gate_logits = moe_mod.gate.get_logits(hs_flat)
        topk_idx, topk_weight, _ = moe_mod.gate(hs_flat)

        # ---- Layer 0: Expert Budgeting ----
        if self.K_target is not None:
            effective_top_p = self.top_p if self.top_p is not None else 0.75
            S_mask = compute_active_set(
                gate_logits, topk_idx, topk_weight,
                self.K_target, self.quality_floor, effective_top_p)
            self.total_active_experts.append(S_mask.sum().item())

            # Mask logits
            masked_logits = gate_logits.clone()
            masked_logits[:, ~S_mask] = float('-inf')

            # Re-route with masked logits
            topk_weight_new, topk_idx_new = moe_mod.gate.routing(
                hs_flat, masked_logits, moe_mod.gate.top_k, True)
        else:
            topk_idx_new = topk_idx
            topk_weight_new = topk_weight

        # ---- Layer 1: Top-p pruning ----
        if self.top_p is not None:
            sorted_w, sort_order = topk_weight_new.sort(dim=1, descending=True)
            total_routing = topk_weight_new.sum(dim=1, keepdim=True)
            needed_frac = (self.top_p - SHARED_RATE) / ROUTING_RATE
            threshold = needed_frac * total_routing
            cumsum = sorted_w.cumsum(dim=1)
            enough = cumsum >= threshold
            enough[:, -1] = True
            cutoff = enough.float().argmax(dim=1) + 1

            rank_pos = torch.arange(TOP_K_ORIG, device=topk_weight_new.device).unsqueeze(0)
            keep_sorted = rank_pos < cutoff.unsqueeze(1)
            pruning_mask = torch.zeros_like(topk_weight_new, dtype=torch.bool)
            pruning_mask.scatter_(1, sort_order, keep_sorted)

            kept_sum = (topk_weight_new * pruning_mask.float()).sum(dim=1, keepdim=True)
            orig_sum = topk_weight_new.sum(dim=1, keepdim=True)
            scale = orig_sum / (kept_sum + 1e-8)
            final_weights = topk_weight_new * pruning_mask.float() * scale
            self.total_experts_per_token.append(cutoff.float().mean().item())

            routed_y = fused_experts(
                hidden_states=hs_flat,
                w1=moe_mod.experts.w13_weight,
                w2=moe_mod.experts.w2_weight,
                topk_weights=final_weights,
                topk_ids=topk_idx_new,
                inplace=False)
        else:
            if self.K_target is not None:
                # Budgeting but no top-p: use masked routing through forward_impl
                masked_logits = gate_logits.clone()
                S_mask = compute_active_set(
                    gate_logits, topk_idx, topk_weight,
                    self.K_target, self.quality_floor, 0.75)
                masked_logits[:, ~S_mask] = float('-inf')
                routed_y = moe_mod.experts.forward_impl(
                    hidden_states=hs_flat, router_logits=masked_logits)
            else:
                router_logits = moe_mod.gate.get_logits(hs_flat)
                routed_y = moe_mod.experts.forward_impl(
                    hidden_states=hs_flat, router_logits=router_logits)

        routed_y = routed_y.view(bsz, seq_len, h)

        # ---- Layer 2: Apply token reuse ----
        if reuse_mask is not None and n_reuse > 0:
            cached_r = self.routed_cache[layer_idx][:bsz]
            routed_y = routed_y.clone()
            routed_y[reuse_mask] = cached_r[reuse_mask].to(routed_y.device)

        # Cache
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


def gen_with_ctrl(dllm, input_ids, ctrl, gl=256):
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner
    oif = BlockDiffusionIteration.forward
    ord_ = BlockDiffusionRunner.decode
    def pd(sr, model, decoder, x, kv_cache, block, block_loc,
           block_id, pos_ids, attn_mask, block_length=32, cross_block_attn_mask=None):
        ctrl.reset_block()
        return ord_(sr, model, decoder, x, kv_cache, block, block_loc,
                    block_id, pos_ids, attn_mask, block_length, cross_block_attn_mask)
    def pf(si, model, decoder, x, kv_cache, block, block_loc,
           block_id, pos_ids, attn_mask, past_key_values,
           replace_position, backend, is_cross_block=False, block_length=32):
        out = oif(si, model, decoder, x, kv_cache, block, block_loc,
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
    print("v0.1.15.3 Phase 2b — Boundary Exploration")
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
        prompt_len = input_ids.shape[1]
        print(f"Input: {input_ids.shape}")

        GEN_LENGTH = 256
        REUSE_LAYERS = set(range(4, 15))

        # ---- Configs: (name, K, QF, top_p, margin, reuse_layers) ----
        CONFIGS = [
            ("A0:baseline",      None, 0.85, None, None, None),
            ("A1:K150_tp75",     150,  0.85, 0.75, None, None),
            ("A2:K150_tp75_D1",  150,  0.85, 0.75, 0.90, REUSE_LAYERS),
            ("A3:K150_tp60",     150,  0.85, 0.60, None, None),
            ("A4:K120_tp60",     120,  0.75, 0.60, None, None),
            ("A5:K100_tp50",     100,  0.65, 0.50, None, None),
            ("A6:K80_tp50",       80,  0.50, 0.50, None, None),
            ("A7:K100_tp60_D1",  100,  0.75, 0.60, 0.90, REUSE_LAYERS),
            ("A8:K80_tp50_D1",    80,  0.65, 0.50, 0.90, REUSE_LAYERS),
        ]

        # ---- Part 1: temp=0 stats ----
        print(f"\n{'='*80}")
        print(f"PART 1: Quantitative Stats (temp=0, gen_length={GEN_LENGTH})")
        print(f"{'='*80}")

        decoder_t0 = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm = BlockDiffusionLLM(
            model, decoder_t0,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=1, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Warmup
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print("Warmup done.\n")

        results = {}
        for cname, K, qf, tp, margin, rl in CONFIGS:
            ctrl = FullStackController(
                K_target=K, quality_floor=qf, top_p=tp,
                margin_threshold=margin, reuse_layers=rl or set())
            hooks = install_hooks(model, ctrl)
            try:
                out = gen_with_ctrl(dllm, input_ids, ctrl, gl=GEN_LENGTH)
                fwd = dllm.diff_iteration.num_forwards
                gen_tokens = out[:, prompt_len:].cpu()
            finally:
                remove_hooks(hooks)

            avg_active = (sum(ctrl.total_active_experts) /
                          max(len(ctrl.total_active_experts), 1)) if ctrl.total_active_experts else 220.8
            avg_ept = (sum(ctrl.total_experts_per_token) /
                       max(len(ctrl.total_experts_per_token), 1)) if ctrl.total_experts_per_token else 8.0
            reuse_pct = ctrl.reused_tokens / max(ctrl.total_tokens, 1) * 100

            print(f"  {cname:<22s} Fwd={fwd:>4d}  Active={avg_active:>6.1f}  "
                  f"Exp/tok={avg_ept:>4.1f}  Reuse={reuse_pct:>5.1f}%")

            results[cname] = {
                "fwd": fwd, "avg_active": avg_active, "avg_ept": avg_ept,
                "reuse_pct": reuse_pct, "gen_tokens": gen_tokens,
            }

        # Summary table
        bl_fwd = results["A0:baseline"]["fwd"]
        print(f"\n  {'Config':<22s} {'Fwd':>4s} {'dFwd':>5s} {'Active':>7s} {'E/tok':>5s} "
              f"{'Reuse':>6s} {'HBM%':>6s}")
        print(f"  {'-'*60}")
        for cname, r in results.items():
            d = r["fwd"] - bl_fwd
            h = (1 - r["avg_active"] / 220.8) * 100
            print(f"  {cname:<22s} {r['fwd']:>4d} {d:>+4d} {r['avg_active']:>7.1f} "
                  f"{r['avg_ept']:>5.1f} {r['reuse_pct']:>5.1f}% {h:>5.1f}%")

        # ---- Part 2: temp=0.7 full text output ----
        print(f"\n{'='*80}")
        print(f"PART 2: Output Quality (temp=0.7, gen_length={GEN_LENGTH})")
        print(f"{'='*80}")

        decoder_t7 = ThresholdParallelDecoder(
            temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm_t7 = BlockDiffusionLLM(
            model, decoder_t7,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=1, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)
        with torch.inference_mode():
            dllm_t7.diff_iteration.num_forwards = 0
            _ = dllm_t7.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        quality_outputs = {}
        for cname, K, qf, tp, margin, rl in CONFIGS:
            ctrl = FullStackController(
                K_target=K, quality_floor=qf, top_p=tp,
                margin_threshold=margin, reuse_layers=rl or set())
            hooks = install_hooks(model, ctrl)
            try:
                out = gen_with_ctrl(dllm_t7, input_ids, ctrl, gl=GEN_LENGTH)
                quality_outputs[cname] = out[:, prompt_len:].cpu()
            finally:
                remove_hooks(hooks)

        # Show outputs for batches 0-3
        for bi in range(4):
            print(f"\n{'='*80}")
            print(f"  BATCH {bi}: {PROMPTS[bi][:70]}...")
            print(f"{'='*80}")
            for cname in [c[0] for c in CONFIGS]:
                gt = quality_outputs[cname][bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                print(f"\n  [{cname}]:")
                print(f"  {text[:500]}")
                if len(text) > 500:
                    print(f"  ... (total {len(text)} chars)")

        # Save
        save_data = {}
        for cname, r in results.items():
            save_data[cname] = {k: v for k, v in r.items() if k != "gen_tokens"}
        out_path = REPO_ROOT / "codex_coding" / "results" / "expert_budgeting_boundary_results.json"
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n\nResults saved to {out_path}")
        print("Done.")


if __name__ == "__main__":
    main()
