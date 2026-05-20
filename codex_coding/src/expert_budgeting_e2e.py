#!/usr/bin/env python3
"""
v0.1.15.3 Phase 2 — Expert Budgeting End-to-End Quality Verification

Integrates expert budgeting into actual generation:
  1. Before each MoE layer, compute gate logits for all tokens
  2. Select active set S via popularity + safety constraints
  3. Mask out non-S experts (set logits to -inf)
  4. Proceed with normal routing on restricted expert set

Tests:
  C0: baseline (no budgeting)
  C1: K_target=200, QF=0.85
  C2: K_target=180, QF=0.85
  C3: K_target=150, QF=0.85
  C4: K_target=120, QF=0.85
  C5: K_target=150, QF=0.75 (more aggressive)
  C6: K_target=150, QF=0.85, + top-p=0.75 (budgeting + D2 combined)

batch=32, temp=0, 5 runs each + temp=0.7 output quality check
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


def compute_active_set(gate_logits, topk_idx, topk_w, K_target, quality_floor,
                       top_p=0.75):
    """
    Compute the active expert set S with safety constraints.
    Returns: S_mask [256] bool tensor on same device as gate_logits.
    """
    N = gate_logits.shape[0]
    device = gate_logits.device

    # Full gate weights (softmax)
    gate_w = torch.softmax(gate_logits.float(), dim=-1)  # [N, 256]

    # Popularity-based initial S
    popularity = gate_w.sum(dim=0)  # [256]
    _, pop_order = popularity.sort(descending=True)
    S_mask = torch.zeros(NUM_EXPERTS, dtype=torch.bool, device=device)
    S_mask[pop_order[:K_target]] = True

    # Per-token k_budget from top-p
    sorted_rw, sort_order = topk_w.sort(dim=1, descending=True)
    total_routing = topk_w.sum(dim=1, keepdim=True)
    needed_frac = (top_p - SHARED_RATE) / ROUTING_RATE
    threshold = needed_frac * total_routing
    cumsum = sorted_rw.cumsum(dim=1)
    enough = cumsum >= threshold
    enough[:, -1] = True
    k_budgets = (enough.float().argmax(dim=1) + 1).int()  # [N]

    # Original top-p expert weights (in gate_w scale)
    sorted_idx = topk_idx.gather(1, sort_order)
    original_gate_vals = gate_w.gather(1, sorted_idx)  # [N, 8]
    positions = torch.arange(TOP_K_ORIG, device=device).unsqueeze(0)
    topp_mask = positions < k_budgets.unsqueeze(1)
    original_weight = (original_gate_vals * topp_mask.float()).sum(dim=1)  # [N]

    # Iterative safety check
    for _ in range(30):
        s_indices = S_mask.nonzero(as_tuple=True)[0]
        s_gate_w = gate_w[:, s_indices]  # [N, |S|]
        s_sorted_w, _ = s_gate_w.sort(dim=1, descending=True)

        s_positions = torch.arange(s_sorted_w.shape[1], device=device).unsqueeze(0)
        s_topp_mask = s_positions < k_budgets.unsqueeze(1)
        covered = (s_sorted_w * s_topp_mask.float()).sum(dim=1)

        safe = original_weight > 1e-8
        cov_ratio = torch.where(safe, covered / original_weight.clamp(min=1e-8),
                                torch.ones(N, device=device))

        violated = (cov_ratio < quality_floor) & safe
        if not violated.any():
            break

        violated_idx = violated.nonzero(as_tuple=True)[0]
        v_gate = gate_w[violated_idx]
        v_gate_masked = v_gate.clone()
        v_gate_masked[:, S_mask] = -1
        best_missing = v_gate_masked.argmax(dim=1)
        for e in best_missing.unique():
            S_mask[e.item()] = True

    return S_mask


class ExpertBudgetingController:
    """Applies expert budgeting by masking gate logits before routing."""

    def __init__(self, K_target=None, quality_floor=0.85, top_p_routing=None):
        self.K_target = K_target  # None = no budgeting
        self.quality_floor = quality_floor
        self.top_p_routing = top_p_routing  # None = no top-p post-pruning
        self.stats = {"total_active": [], "total_pairs": 0, "kept_pairs": 0}

    def hook_forward(self, moe_mod, layer_idx, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        N = hs_flat.shape[0]

        shared_res = moe_mod.shared_experts(hidden_states)

        if self.K_target is not None:
            # Get full gate logits and routing
            gate_logits = moe_mod.gate.get_logits(hs_flat)  # [N, 256]
            topk_idx, topk_weight, _ = moe_mod.gate(hs_flat)

            # Compute active set S
            S_mask = compute_active_set(
                gate_logits, topk_idx, topk_weight,
                self.K_target, self.quality_floor)

            self.stats["total_active"].append(S_mask.sum().item())

            # Mask gate logits: set non-S experts to -inf
            masked_logits = gate_logits.clone()
            masked_logits[:, ~S_mask] = float('-inf')

            # Re-route with masked logits through the standard path
            routed_y = moe_mod.experts.forward_impl(
                hidden_states=hs_flat, router_logits=masked_logits)

            # Optional: top-p pruning on top of budgeting
            if self.top_p_routing is not None:
                # Top-p is applied within forward_impl via custom_routing_function
                # We need a different approach: call gate with masked logits,
                # then manually prune and call fused_experts
                # For now, the masked logits already restrict expert selection
                # Top-p would further reduce within the active set
                pass  # handled below separately
        else:
            # Baseline: standard path
            router_logits = moe_mod.gate.get_logits(hs_flat)
            routed_y = moe_mod.experts.forward_impl(
                hidden_states=hs_flat, router_logits=router_logits)

        routed_y = routed_y.view(bsz, seq_len, h)
        out = routed_y + shared_res if moe_mod.config.num_shared_experts is not None else routed_y
        return out


class BudgetingPlusTopPController:
    """Expert Budgeting + Top-p pruning combined."""

    def __init__(self, K_target, quality_floor=0.85, top_p=0.75):
        self.K_target = K_target
        self.quality_floor = quality_floor
        self.top_p = top_p
        self.stats = {"total_active": [], "avg_experts_per_token": []}

    def hook_forward(self, moe_mod, layer_idx, hidden_states):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        N = hs_flat.shape[0]

        shared_res = moe_mod.shared_experts(hidden_states)

        # Get full routing
        gate_logits = moe_mod.gate.get_logits(hs_flat)
        topk_idx, topk_weight, _ = moe_mod.gate(hs_flat)

        # Expert budgeting: compute S
        S_mask = compute_active_set(
            gate_logits, topk_idx, topk_weight,
            self.K_target, self.quality_floor, self.top_p)
        self.stats["total_active"].append(S_mask.sum().item())

        # Mask logits and re-route via gate.routing()
        masked_logits = gate_logits.clone()
        masked_logits[:, ~S_mask] = float('-inf')
        topk_weight_new, topk_idx_new = moe_mod.gate.routing(
            hs_flat, masked_logits, moe_mod.gate.top_k, True)

        # Apply top-p pruning
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
        new_weights = topk_weight_new * pruning_mask.float() * scale

        self.stats["avg_experts_per_token"].append(cutoff.float().mean().item())

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

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print("v0.1.15.3 Phase 2 — Expert Budgeting E2E Quality Verification")
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

        # ---- Setup DLLM ----
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
            _ = dllm.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print("Warmup done.\n")

        # ---- Configs ----
        CONFIGS = [
            ("C0:baseline",   None,  0.85, False),
            ("C1:K=200",      200,   0.85, False),
            ("C2:K=180",      180,   0.85, False),
            ("C3:K=150",      150,   0.85, False),
            ("C4:K=120",      120,   0.85, False),
            ("C5:K=150,QF75", 150,   0.75, False),
            ("C6:K=150+tp75", 150,   0.85, True),   # budgeting + top-p
        ]
        N_RUNS = 5

        results = {}

        for cname, K, qf, use_topp in CONFIGS:
            print(f"{'='*60}")
            print(f"{cname}: K={K}, QF={qf}, top-p={'0.75' if use_topp else 'no'}")
            print(f"{'='*60}")

            fwds_list = []
            for run in range(N_RUNS):
                if K is None:
                    ctrl = ExpertBudgetingController(K_target=None)
                elif use_topp:
                    ctrl = BudgetingPlusTopPController(K_target=K, quality_floor=qf, top_p=0.75)
                else:
                    ctrl = ExpertBudgetingController(K_target=K, quality_floor=qf)

                hooks = install_hooks(model, ctrl)
                try:
                    with torch.inference_mode():
                        dllm.diff_iteration.num_forwards = 0
                        dllm.diff_iteration.iter_no = 0
                        out = dllm.generate(input_ids.clone(), gen_length=128,
                                            block_length=BLOCK_LENGTH)
                    fwds_list.append(dllm.diff_iteration.num_forwards)
                    if run == 0:
                        gen_tokens = out[:, prompt_len:].cpu()
                        avg_active = (sum(ctrl.stats["total_active"]) /
                                      max(len(ctrl.stats["total_active"]), 1))
                finally:
                    remove_hooks(hooks)

            avg_fwd = sum(fwds_list) / len(fwds_list)
            print(f"  Fwds: {fwds_list}  avg={avg_fwd:.0f}")
            if K is not None:
                print(f"  Avg active experts: {avg_active:.1f}")

            results[cname] = {
                "K": K, "qf": qf, "use_topp": use_topp,
                "avg_fwd": avg_fwd, "fwds": fwds_list,
                "avg_active": avg_active if K is not None else 220.8,
                "gen_tokens": gen_tokens,
            }

        # ---- Summary ----
        bl_fwd = results["C0:baseline"]["avg_fwd"]
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Config':<18s} {'AvgFwd':>7s} {'dFwd':>5s} {'AvgActive':>10s} "
              f"{'HBM Save':>8s} {'Verdict':>10s}")
        print(f"  {'-'*65}")

        for cname, r in results.items():
            delta = r["avg_fwd"] - bl_fwd
            hbm_save = (1 - r["avg_active"] / 220.8) * 100
            if abs(delta) <= 2:
                verdict = "SAFE"
            elif abs(delta) <= 5:
                verdict = "MARGINAL"
            else:
                verdict = "BAD"
            if cname == "C0:baseline":
                verdict = "—"
            print(f"  {cname:<18s} {r['avg_fwd']:>7.0f} {delta:>+4.0f} "
                  f"{r['avg_active']:>10.1f} {hbm_save:>7.1f}% {verdict:>10s}")

        # ---- Output quality (temp=0.7) for select configs ----
        print(f"\n{'='*80}")
        print(f"OUTPUT QUALITY (temp=0.7)")
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
            _ = dllm_t7.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        quality_configs = ["C0:baseline", "C3:K=150", "C5:K=150,QF75", "C6:K=150+tp75"]
        quality_outputs = {}

        for cname in quality_configs:
            r = results[cname]
            if r["K"] is None:
                ctrl = ExpertBudgetingController(K_target=None)
            elif r["use_topp"]:
                ctrl = BudgetingPlusTopPController(K_target=r["K"], quality_floor=r["qf"], top_p=0.75)
            else:
                ctrl = ExpertBudgetingController(K_target=r["K"], quality_floor=r["qf"])

            hooks = install_hooks(model, ctrl)
            try:
                with torch.inference_mode():
                    dllm_t7.diff_iteration.num_forwards = 0
                    out = dllm_t7.generate(input_ids.clone(), gen_length=128,
                                           block_length=BLOCK_LENGTH)
                quality_outputs[cname] = out[:, prompt_len:].cpu()
            finally:
                remove_hooks(hooks)

        for bi in range(min(3, BATCH_SIZE)):
            print(f"\n{'─'*80}")
            print(f"  BATCH {bi}: {PROMPTS[bi][:65]}...")
            print(f"{'─'*80}")
            for cname in quality_configs:
                gt = quality_outputs[cname][bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                print(f"\n  [{cname}]:")
                print(f"  {text[:300]}")

        # Save
        save_data = {}
        for cname, r in results.items():
            save_data[cname] = {k: v for k, v in r.items() if k != "gen_tokens"}
        out_path = REPO_ROOT / "codex_coding" / "results" / "expert_budgeting_e2e_results.json"
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n\nResults saved to {out_path}")
        print("Done.")


if __name__ == "__main__":
    main()
