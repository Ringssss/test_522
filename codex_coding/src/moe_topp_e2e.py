#!/usr/bin/env python3
"""
v0.1.14.9e — Adaptive Top-p Expert Pruning E2E

Complete the Q3 verification: test adaptive top-p alongside fixed top-k.
Uses the same correct implementation: gate() → modify topk_weights → fused_experts().

Configs:
  baseline:    top-8 (no pruning)
  fixed top-4: keep top-4 experts (Q3 reference: ΔFwd=+1)
  top-p=0.80:  keep until 80% global info (predicted avg 4.52 experts)
  top-p=0.75:  keep until 75% global info (predicted avg 3.80 experts)
  top-p=0.70:  keep until 70% global info (predicted avg 3.19 experts)

batch=32, all layers, temp=0 (5 runs) + temp=0.7 (output text)
"""

from __future__ import annotations
import os, sys, time, socket
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


def install_pruning_hooks(model, mode, keep_k=None, top_p=None):
    """
    mode="baseline": no pruning
    mode="fixed_k":  keep top keep_k experts
    mode="top_p":    keep experts until global info >= top_p
    """
    hooks = []
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    stats = {"total_expert_slots": 0, "kept_expert_slots": 0}

    for layer in model.model.layers:
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue
        moe = layer.mlp
        orig = moe.forward

        def make_hook(moe_mod, m, kk, tp, st):
            def hooked_forward(hidden_states):
                bsz, seq_len, h = hidden_states.shape
                hs_flat = hidden_states.view(-1, h)
                n_tokens = hs_flat.shape[0]

                shared_res = moe_mod.shared_experts(hidden_states)

                if m == "baseline":
                    router_logits = moe_mod.gate.get_logits(hs_flat)
                    routed_y = moe_mod.experts.forward_impl(
                        hidden_states=hs_flat, router_logits=router_logits)
                    st["total_expert_slots"] += n_tokens * TOP_K_ORIG
                    st["kept_expert_slots"] += n_tokens * TOP_K_ORIG
                else:
                    topk_idx, topk_weight, _ = moe_mod.gate(hs_flat)
                    # Sort descending by weight
                    sorted_w, sort_order = topk_weight.sort(dim=1, descending=True)

                    if m == "fixed_k":
                        # Keep top-kk, zero rest
                        mask = torch.zeros_like(topk_weight)
                        mask.scatter_(1, sort_order[:, :kk], 1.0)
                        st["kept_expert_slots"] += n_tokens * kk
                    elif m == "top_p":
                        # Global top-p: find min k such that
                        # shared_rate + routing_rate * (cumsum_k / total_routing) >= tp
                        total_routing = topk_weight.sum(dim=1, keepdim=True)
                        needed_frac = (tp - SHARED_RATE) / ROUTING_RATE
                        threshold = needed_frac * total_routing
                        cumsum = sorted_w.cumsum(dim=1)
                        enough = (cumsum >= threshold)
                        enough[:, -1] = True
                        cutoff = enough.float().argmax(dim=1) + 1  # [N], 1-indexed

                        # Build mask in sorted space, then map back
                        rank_pos = torch.arange(TOP_K_ORIG, device=topk_weight.device).unsqueeze(0)
                        keep_sorted = rank_pos < cutoff.unsqueeze(1)
                        mask = torch.zeros_like(topk_weight)
                        mask.scatter_(1, sort_order, keep_sorted.float())
                        st["kept_expert_slots"] += cutoff.sum().item()

                    st["total_expert_slots"] += n_tokens * TOP_K_ORIG

                    # Renormalize
                    kept_sum = (topk_weight * mask).sum(dim=1, keepdim=True)
                    orig_sum = topk_weight.sum(dim=1, keepdim=True)
                    scale = orig_sum / (kept_sum + 1e-8)
                    new_weights = topk_weight * mask * scale

                    routed_y = fused_experts(
                        hidden_states=hs_flat,
                        w1=moe_mod.experts.w13_weight,
                        w2=moe_mod.experts.w2_weight,
                        topk_weights=new_weights,
                        topk_ids=topk_idx,
                        inplace=False,
                    )

                routed_y = routed_y.view(bsz, seq_len, h)
                out = routed_y + shared_res if moe_mod.config.num_shared_experts is not None else routed_y
                return out
            return hooked_forward

        moe.forward = make_hook(moe, mode, keep_k, top_p, stats)
        hooks.append((moe, orig))
    return hooks, stats


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
    print("v0.1.14.9e — Adaptive Top-p Expert Pruning E2E")
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
            ("baseline",   "baseline", None, None),
            ("fixed-4",    "fixed_k",  4,    None),
            ("top-p=0.80", "top_p",    None, 0.80),
            ("top-p=0.75", "top_p",    None, 0.75),
            ("top-p=0.70", "top_p",    None, 0.70),
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
        for cname, mode, kk, tp in CONFIGS:
            fwds_list = []
            avg_expert = 0

            for r in range(N_RUNS):
                hooks, stats = install_pruning_hooks(model, mode, keep_k=kk, top_p=tp)
                try:
                    with torch.inference_mode():
                        dllm.diff_iteration.num_forwards = 0
                        dllm.diff_iteration.iter_no = 0
                        out = dllm.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
                    fwds_list.append(dllm.diff_iteration.num_forwards)
                    if r == 0:
                        total_slots = stats["total_expert_slots"]
                        kept_slots = stats["kept_expert_slots"]
                        if total_slots > 0:
                            avg_expert = kept_slots / (total_slots / TOP_K_ORIG)
                        gen_tokens = out[:, prompt_len:].cpu()
                finally:
                    remove_hooks(hooks)

            avg_fwd = sum(fwds_list) / len(fwds_list)
            saved_pct = (1 - kept_slots / max(total_slots, 1)) * 100 if mode != "baseline" else 0

            print(f"  {cname:>12s}: fwds=[{','.join(str(f) for f in fwds_list)}] "
                  f"avg_fwd={avg_fwd:.0f}  avg_expert={avg_expert:.2f}  saved={saved_pct:.1f}%")

            results[cname] = {
                "avg_fwd": avg_fwd, "fwds": fwds_list,
                "avg_expert": avg_expert, "saved_pct": saved_pct,
                "gen_tokens": gen_tokens,
            }

        # Summary
        bl_fwd = results["baseline"]["avg_fwd"]
        print(f"\n{'='*80}")
        print(f"SUMMARY: Fixed top-k vs Adaptive top-p")
        print(f"{'='*80}")
        print(f"  {'Config':>12s} {'AvgFwd':>7s} {'ΔFwd':>5s} {'AvgExpert':>10s} "
              f"{'Saved%':>7s} {'Verdict':>10s}")
        print(f"  {'-'*58}")
        for cname, r in results.items():
            delta = r["avg_fwd"] - bl_fwd
            if cname == "baseline":
                verdict = "—"
            elif abs(delta) <= 1:
                verdict = "SAFE"
            elif delta <= 3:
                verdict = "MARGINAL"
            else:
                verdict = "BAD"
            print(f"  {cname:>12s} {r['avg_fwd']:>7.0f} {delta:>+4.0f} "
                  f"{r['avg_expert']:>10.2f} {r['saved_pct']:>6.1f}% {verdict:>10s}")

        # Key comparison
        print(f"\n  Key comparison at similar avg expert count:")
        if "fixed-4" in results and "top-p=0.75" in results:
            f4 = results["fixed-4"]
            tp75 = results["top-p=0.75"]
            print(f"    fixed-4:    {f4['avg_expert']:.2f} experts, ΔFwd={f4['avg_fwd']-bl_fwd:+.0f}")
            print(f"    top-p=0.75: {tp75['avg_expert']:.2f} experts, ΔFwd={tp75['avg_fwd']-bl_fwd:+.0f}")

        # ---- PART 2: temp=0.7 output ----
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

        with torch.inference_mode():
            dllm_t7.diff_iteration.num_forwards = 0
            _ = dllm_t7.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        CHECK = [
            ("baseline", "baseline", None, None),
            ("fixed-4", "fixed_k", 4, None),
            ("top-p=0.80", "top_p", None, 0.80),
            ("top-p=0.75", "top_p", None, 0.75),
        ]
        check_outputs = {}
        for cname, mode, kk, tp in CHECK:
            hooks, _ = install_pruning_hooks(model, mode, keep_k=kk, top_p=tp)
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
            for cname in ["baseline", "fixed-4", "top-p=0.80", "top-p=0.75"]:
                gt = check_outputs[cname][bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                print(f"\n  [{cname}]:")
                print(f"  {text[:300]}")

        print(f"\nDone.")


if __name__ == "__main__":
    main()
