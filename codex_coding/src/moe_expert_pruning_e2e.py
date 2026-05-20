#!/usr/bin/env python3
"""
v0.1.14.9b — Expert Pruning End-to-End Impact

Test the effect of reducing expert count on actual generation quality.
Method A: zero out low-weight experts + renormalize remaining weights.

Configs:
  baseline: top-8 (no pruning)
  top-3:    keep only top-3 experts per token, zero rest, renormalize
  top-2:    keep only top-2 experts per token
  top-p90:  keep experts until cumulative weight >= 0.90 of total

batch=32, L0-L18 (all layers — this is a universal optimization, not layer-selective)
temp=0 (5 runs, forward count) + temp=0.7 (output text)
"""

from __future__ import annotations
import os, sys, time, socket, json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
TOP_K = 8

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


class ExpertPruningController:
    """Prune low-weight experts: zero out + renormalize."""

    def __init__(self, mode="baseline", top_p=0.90, fixed_k=None):
        """
        mode: "baseline" | "fixed_k" | "top_p"
        fixed_k: for mode="fixed_k", keep only top-K experts (e.g., 3)
        top_p: for mode="top_p", keep experts until cumsum >= top_p * total_weight
        """
        self.mode = mode
        self.top_p = top_p
        self.fixed_k = fixed_k
        self.total_experts_computed = 0  # total expert slots (tokens × 8)
        self.experts_kept = 0  # how many were actually kept

    def hook_forward(self, moe_mod, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        n_tokens = hs_flat.shape[0]

        shared_res = moe_mod.shared_experts(hidden_states)

        if self.mode == "baseline":
            router_logits = moe_mod.gate.get_logits(hs_flat)
            routed_y = moe_mod.experts.forward_impl(
                hidden_states=hs_flat, router_logits=router_logits)
            self.total_experts_computed += n_tokens * TOP_K
            self.experts_kept += n_tokens * TOP_K
        else:
            # Get routing decisions
            topk_idx, topk_weight, full_logits = moe_mod.gate(hs_flat)
            # topk_idx: [n_tokens, 8], topk_weight: [n_tokens, 8]

            # Sort by weight descending per token
            sorted_weights, sort_order = topk_weight.sort(dim=1, descending=True)
            sorted_idx = topk_idx.gather(1, sort_order)

            if self.mode == "fixed_k":
                k = self.fixed_k
                # Zero out experts beyond top-k
                mask = torch.zeros_like(topk_weight)
                mask.scatter_(1, sort_order[:, :k], 1.0)
                # Renormalize: scale kept weights so their sum equals original total
                kept_sum = (topk_weight * mask).sum(dim=1, keepdim=True)
                orig_sum = topk_weight.sum(dim=1, keepdim=True)
                scale = orig_sum / (kept_sum + 1e-8)
                new_weights = topk_weight * mask * scale
                self.experts_kept += n_tokens * k

            elif self.mode == "top_p":
                # Cumulative sum of sorted weights
                cumsum = sorted_weights.cumsum(dim=1)
                total_per_token = topk_weight.sum(dim=1, keepdim=True)
                threshold = total_per_token * self.top_p

                # Find cutoff: first position where cumsum >= threshold
                enough = (cumsum >= threshold)
                enough[:, -1] = True  # guarantee at least one
                cutoff = enough.float().argmax(dim=1) + 1  # [n_tokens], 1-indexed

                # Build mask
                rank_positions = torch.arange(TOP_K, device=topk_weight.device).unsqueeze(0)
                keep_in_sorted = rank_positions < cutoff.unsqueeze(1)  # [n_tokens, 8]
                # Map back to original order
                mask = torch.zeros_like(topk_weight)
                mask.scatter_(1, sort_order, keep_in_sorted.float())

                # Renormalize
                kept_sum = (topk_weight * mask).sum(dim=1, keepdim=True)
                orig_sum = topk_weight.sum(dim=1, keepdim=True)
                scale = orig_sum / (kept_sum + 1e-8)
                new_weights = topk_weight * mask * scale
                self.experts_kept += cutoff.sum().item()

            self.total_experts_computed += n_tokens * TOP_K

            # Rebuild router_logits that will produce the modified weights
            # We need to go through forward_impl which takes router_logits
            # Approach: use gate.get_logits but modify topk_weight before dispatch
            # Actually, forward_impl internally calls gate again, so we need
            # to hook at a different level.
            #
            # Simpler approach: modify the gate output directly.
            # forward_impl calls self.gate(hs_flat) internally via get_logits.
            # Let's use a different path: call experts with explicit routing.
            #
            # Actually, the cleanest approach is to compute full routed output
            # then apply the weight modification as a correction.
            # full_routed = sum(w_i * expert_i(x)) for all 8
            # We want: sum(w_i' * expert_i(x)) for kept experts only
            # = sum( (w_i * scale * mask_i) * expert_i(x) )
            # = scale * sum( mask_i * w_i * expert_i(x) )
            # But we can't decompose the fused output...
            #
            # Alternative: just call forward_impl normally, it gives us
            # sum(w_i * expert_i(x)). Then we need to "subtract" the
            # contribution of pruned experts. But we don't have per-expert outputs.
            #
            # The ONLY way to truly skip expert computation is to modify
            # the routing weights BEFORE dispatch. We need to intercept
            # between gate and expert dispatch.
            #
            # Let's temporarily replace the gate's forward to return modified weights.

            orig_gate_forward = moe_mod.gate.forward

            def modified_gate_forward(hidden_states_flat):
                # Return the pre-computed modified routing
                return topk_idx, new_weights, full_logits

            moe_mod.gate.forward = modified_gate_forward
            try:
                router_logits = moe_mod.gate.get_logits(hs_flat)
                routed_y = moe_mod.experts.forward_impl(
                    hidden_states=hs_flat, router_logits=router_logits)
            finally:
                moe_mod.gate.forward = orig_gate_forward

        routed_y = routed_y.view(bsz, seq_len, h)
        out = routed_y + shared_res if moe_mod.config.num_shared_experts is not None else routed_y
        return out


def install_hooks(model, ctrl):
    hooks = []
    for layer in model.model.layers:
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue
        moe = layer.mlp
        orig = moe.forward
        def make(m, c):
            def f(hs): return c.hook_forward(m, hs)
            return f
        moe.forward = make(moe, ctrl)
        hooks.append((moe, orig))
    return hooks

def remove_hooks(hooks):
    for moe, orig in hooks:
        moe.forward = orig


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
    print("v0.1.14.9b — Expert Pruning Impact Test")
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

        CONFIGS = [
            ("baseline", {"mode": "baseline"}),
            ("top-3", {"mode": "fixed_k", "fixed_k": 3}),
            ("top-2", {"mode": "fixed_k", "fixed_k": 2}),
            ("top-p=0.90", {"mode": "top_p", "top_p": 0.90}),
            ("top-p=0.85", {"mode": "top_p", "top_p": 0.85}),
        ]
        N_RUNS = 5

        # ---- PART 1: temp=0 ----
        print(f"\n{'='*80}")
        print(f"PART 1: Quantitative (temp=0, {N_RUNS} runs, batch=32)")
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
            _ = dllm.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        results = {}
        for cname, cargs in CONFIGS:
            fwds_list = []
            expert_stats = {}

            for r in range(N_RUNS):
                ctrl = ExpertPruningController(**cargs)
                hooks = install_hooks(model, ctrl)
                try:
                    with torch.inference_mode():
                        dllm.diff_iteration.num_forwards = 0
                        dllm.diff_iteration.iter_no = 0
                        out = dllm.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
                    fwds_list.append(dllm.diff_iteration.num_forwards)
                    if r == 0:
                        expert_stats = {
                            "total_slots": ctrl.total_experts_computed,
                            "kept": ctrl.experts_kept,
                        }
                finally:
                    remove_hooks(hooks)

            avg_fwd = sum(fwds_list) / len(fwds_list)
            kept_pct = expert_stats.get("kept", 0) / max(expert_stats.get("total_slots", 1), 1) * 100
            saved_pct = 100 - kept_pct

            print(f"\n  {cname}: fwds=[{','.join(str(f) for f in fwds_list)}] avg={avg_fwd:.0f}")
            print(f"    Experts kept: {kept_pct:.1f}%, saved: {saved_pct:.1f}%")

            results[cname] = {
                "avg_fwd": avg_fwd, "fwds": fwds_list,
                "kept_pct": kept_pct, "saved_pct": saved_pct,
            }

        # Summary
        bl_fwd = results["baseline"]["avg_fwd"]
        print(f"\n{'='*80}")
        print(f"SUMMARY (temp=0, batch=32)")
        print(f"{'='*80}")
        print(f"  {'Config':>12s} {'AvgFwd':>7s} {'ΔFwd':>5s} {'ExpertSaved':>12s} {'Verdict':>10s}")
        print(f"  {'-'*50}")
        for cname, r in results.items():
            delta = r["avg_fwd"] - bl_fwd
            if abs(delta) <= 1:
                verdict = "OK"
            elif delta <= -2:
                verdict = "BETTER"
            elif delta <= 3:
                verdict = "MARGINAL"
            else:
                verdict = "BAD"
            if cname == "baseline":
                verdict = "—"
            print(f"  {cname:>12s} {r['avg_fwd']:>7.0f} {delta:>+4.0f} "
                  f"{r['saved_pct']:>10.1f}% {verdict:>10s}")

        # ---- PART 2: temp=0.7, output text ----
        print(f"\n{'='*80}")
        print(f"PART 2: Output quality (temp=0.7, batch=32)")
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

        # Baseline
        ctrl_bl = ExpertPruningController(mode="baseline")
        hooks = install_hooks(model, ctrl_bl)
        try:
            with torch.inference_mode():
                dllm_t7.diff_iteration.num_forwards = 0
                bl_out = dllm_t7.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
        finally:
            remove_hooks(hooks)
        bl_tokens = bl_out[:, prompt_len:].cpu()

        CHECK = ["top-3", "top-2", "top-p=0.90"]
        check_outputs = {}
        for cname in CHECK:
            cargs = dict(CONFIGS)[[c[0] for c in CONFIGS].index(cname)][1] if False else None
            for cn, ca in CONFIGS:
                if cn == cname:
                    cargs = ca
                    break
            ctrl = ExpertPruningController(**cargs)
            hooks = install_hooks(model, ctrl)
            try:
                with torch.inference_mode():
                    dllm_t7.diff_iteration.num_forwards = 0
                    out = dllm_t7.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
                check_outputs[cname] = out[:, prompt_len:].cpu()
            finally:
                remove_hooks(hooks)

        for bi in range(4):
            print(f"\n{'─'*80}")
            print(f"  BATCH {bi}: {PROMPTS[bi][:65]}...")
            print(f"{'─'*80}")

            gt = bl_tokens[bi]
            valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
            print(f"\n  [baseline]:")
            print(f"  {tokenizer.decode(valid, skip_special_tokens=True)[:300]}")

            for cname in CHECK:
                gt2 = check_outputs[cname][bi]
                valid2 = gt2[(gt2 != 0) & (gt2 != EOS_ID) & (gt2 != MASK_ID)]
                print(f"\n  [{cname}]:")
                print(f"  {tokenizer.decode(valid2, skip_special_tokens=True)[:300]}")

        print(f"\nDone.")


if __name__ == "__main__":
    main()
