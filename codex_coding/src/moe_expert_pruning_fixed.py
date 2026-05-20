#!/usr/bin/env python3
"""
v0.1.14.9c — Fixed Expert Pruning + Shared Rate + Correct Top-p

Fixes:
  1. Directly modify topk_weights/topk_ids BEFORE calling fused_experts
     (not through router_logits which bypasses our modification)
  2. Measure shared_rate from fresh data for correct top-p framing
  3. Sanity check: top-k=0 (zero all routing) must degrade quality

Configs:
  baseline:  top-8 (no pruning)
  top-k=0:   zero ALL routing experts (sanity — must fail)
  top-k=1:   keep only top-1 expert
  top-k=2:   keep only top-2
  top-k=3:   keep only top-3
  top-k=4:   keep only top-4

batch=32, all layers, temp=0 (5 runs) + temp=0.7 (output text)
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
TOP_K_ORIG = 8

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


def install_pruning_hooks(model, keep_k):
    """Hook MoE blocks to prune routing weights BEFORE expert dispatch.

    Strategy: call gate to get topk_idx/topk_weight, then zero out experts
    beyond rank keep_k, renormalize remaining weights, and call fused_experts
    directly with the modified routing.

    keep_k=0: zero ALL routing (sanity check)
    keep_k=8: no pruning (baseline equivalent)
    """
    hooks = []
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    for layer in model.model.layers:
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue
        moe = layer.mlp
        orig = moe.forward
        kk = keep_k

        def make_hook(moe_mod, k):
            def hooked_forward(hidden_states):
                bsz, seq_len, h = hidden_states.shape
                hs_flat = hidden_states.view(-1, h)

                # Shared expert (always fresh)
                shared_res = moe_mod.shared_experts(hidden_states)

                if k == 0:
                    # Zero all routing — output is only shared
                    routed_y = torch.zeros_like(hs_flat)
                elif k >= TOP_K_ORIG:
                    # No pruning — standard path
                    router_logits = moe_mod.gate.get_logits(hs_flat)
                    routed_y = moe_mod.experts.forward_impl(
                        hidden_states=hs_flat, router_logits=router_logits)
                else:
                    # === Fixed: directly modify topk_weights before fused_experts ===
                    # Step 1: get routing from gate
                    topk_idx, topk_weight, _ = moe_mod.gate(hs_flat)
                    # topk_idx: [N, 8] int, topk_weight: [N, 8] float

                    # Step 2: sort by weight descending, zero out beyond rank k
                    sorted_weights, sort_order = topk_weight.sort(dim=1, descending=True)
                    # Create mask: keep top-k, zero rest
                    mask = torch.zeros_like(topk_weight)
                    # sort_order[:, :k] are the indices of top-k in original order
                    mask.scatter_(1, sort_order[:, :k], 1.0)

                    # Step 3: renormalize so kept weights sum to original total
                    kept_sum = (topk_weight * mask).sum(dim=1, keepdim=True)
                    orig_sum = topk_weight.sum(dim=1, keepdim=True)
                    scale = orig_sum / (kept_sum + 1e-8)
                    new_weights = topk_weight * mask * scale

                    # Step 4: call fused_experts directly with modified weights
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

        moe.forward = make_hook(moe, kk)
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
    print("v0.1.14.9c — Fixed Expert Pruning + Sanity Check")
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

        # ---- Shared rate measurement from fresh data ----
        print(f"\n{'='*80}")
        print(f"SHARED vs ROUTED RATE (from fresh run data)")
        print(f"{'='*80}")
        data_path = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction" / "full_fresh_run_data.pt"
        fresh_data = torch.load(data_path, map_location="cpu")
        shared_norms, routed_norms = [], []
        for step in range(0, len(fresh_data["shared_output"]), 3):
            for layer in range(0, len(fresh_data["shared_output"][step]), 4):
                s = fresh_data["shared_output"][step][layer].float()
                r = fresh_data["routed_output"][step][layer].float()
                shared_norms.append(s.norm(dim=-1).mean().item())
                routed_norms.append(r.norm(dim=-1).mean().item())
        avg_shared = sum(shared_norms) / len(shared_norms)
        avg_routed = sum(routed_norms) / len(routed_norms)
        shared_rate = avg_shared / (avg_shared + avg_routed)
        print(f"  Avg shared norm: {avg_shared:.4f}")
        print(f"  Avg routed norm: {avg_routed:.4f}")
        print(f"  Shared rate: {shared_rate:.4f} ({shared_rate*100:.1f}%)")
        print(f"  Routed rate: {1-shared_rate:.4f} ({(1-shared_rate)*100:.1f}%)")
        print(f"\n  For top-p=0.90 globally:")
        needed_routing = (0.90 - shared_rate) / (1 - shared_rate)
        print(f"    Need {needed_routing*100:.1f}% of routing weight")
        print(f"    (shared already covers {shared_rate*100:.1f}%, need {(0.90-shared_rate)*100:.1f}% more from routing)")
        del fresh_data

        # ---- E2E experiments ----
        CONFIGS = [0, 1, 2, 3, 4, 8]  # keep_k values
        N_RUNS = 5

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
        for keep_k in CONFIGS:
            fwds_list = []
            label = f"top-{keep_k}" if keep_k > 0 else "NO_ROUTING"
            if keep_k == 8:
                label = "baseline(top-8)"

            for r in range(N_RUNS):
                hooks = install_pruning_hooks(model, keep_k)
                try:
                    with torch.inference_mode():
                        dllm.diff_iteration.num_forwards = 0
                        dllm.diff_iteration.iter_no = 0
                        out = dllm.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
                    fwds_list.append(dllm.diff_iteration.num_forwards)
                    if r == 0:
                        gen_tokens = out[:, prompt_len:].cpu()
                finally:
                    remove_hooks(hooks)

            avg_fwd = sum(fwds_list) / len(fwds_list)
            print(f"  {label:>16s}: fwds=[{','.join(str(f) for f in fwds_list)}] avg={avg_fwd:.0f}")
            results[keep_k] = {"label": label, "avg_fwd": avg_fwd, "fwds": fwds_list,
                               "gen_tokens": gen_tokens}

        # Summary
        bl_fwd = results[8]["avg_fwd"]
        print(f"\n{'='*80}")
        print(f"SUMMARY (temp=0, batch=32)")
        print(f"{'='*80}")
        print(f"  {'Config':>16s} {'AvgFwd':>7s} {'ΔFwd':>5s} {'Expert%':>8s} {'Verdict':>10s}")
        print(f"  {'-'*52}")
        for keep_k in CONFIGS:
            r = results[keep_k]
            delta = r["avg_fwd"] - bl_fwd
            expert_pct = keep_k / TOP_K_ORIG * 100
            if keep_k == 0:
                verdict = "SANITY"
            elif abs(delta) <= 1:
                verdict = "OK"
            elif delta > 5:
                verdict = "BAD"
            elif delta > 2:
                verdict = "MARGINAL"
            else:
                verdict = "GOOD"
            if keep_k == 8:
                verdict = "—"
            print(f"  {r['label']:>16s} {r['avg_fwd']:>7.0f} {delta:>+4.0f} "
                  f"{expert_pct:>7.0f}% {verdict:>10s}")

        # ---- PART 2: temp=0.7 output text ----
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

        CHECK_K = [0, 2, 4, 8]
        check_outputs = {}
        for keep_k in CHECK_K:
            hooks = install_pruning_hooks(model, keep_k)
            try:
                with torch.inference_mode():
                    dllm_t7.diff_iteration.num_forwards = 0
                    out = dllm_t7.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
                check_outputs[keep_k] = out[:, prompt_len:].cpu()
            finally:
                remove_hooks(hooks)

        for bi in range(4):
            print(f"\n{'─'*80}")
            print(f"  BATCH {bi}: {PROMPTS[bi][:65]}...")
            print(f"{'─'*80}")

            for keep_k in CHECK_K:
                label = f"top-{keep_k}" if keep_k > 0 else "NO_ROUTING"
                if keep_k == 8: label = "baseline(top-8)"
                gt = check_outputs[keep_k][bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                print(f"\n  [{label}]:")
                print(f"  {text[:300]}")

        print(f"\nDone.")


if __name__ == "__main__":
    main()
