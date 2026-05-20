#!/usr/bin/env python3
"""
v0.1.14.8 — Token Margin Threshold Sweep

Find the tradeoff boundary: how far can we relax token_margin threshold
while maintaining output quality?

Sweep: margin ∈ {0.999, 0.99, 0.97, 0.95, 0.90, 0.80, 0.50}
Config: batch=32, L4-14, heterogeneous prompts
Metrics: reuse%, forward count, per-layer/per-step reuse distribution
Quality: temp=0 (quantitative), temp=0.7 (output text for human review)
"""

from __future__ import annotations
import os, sys, time, socket, json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
NUM_MOE_LAYERS = 19

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


class ThresholdSweepController:
    def __init__(self, margin_threshold=0.99, reuse_layers=None):
        self.margin_threshold = margin_threshold
        self.reuse_layers = reuse_layers or set(range(4, 15))
        self.routed_cache = {}
        self.qualifying_mask = None
        # Stats
        self.total = 0
        self.reused = 0
        self.per_layer_total = defaultdict(int)
        self.per_layer_reused = defaultdict(int)
        self.per_step_total = defaultdict(int)
        self.per_step_reused = defaultdict(int)
        self.current_step = 0

    def reset_block(self):
        self.routed_cache.clear()
        self.qualifying_mask = None
        self.current_step = 0

    def update(self, logits):
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            top2 = probs.topk(2, dim=-1).values
            self.qualifying_mask = (top2[:, :, 0] - top2[:, :, 1]) > self.margin_threshold
        self.current_step += 1

    def hook_forward(self, moe_mod, mi, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        shared_res = moe_mod.shared_experts(hidden_states)
        router_logits = moe_mod.gate.get_logits(hs_flat)
        routed_y = moe_mod.experts.forward_impl(
            hidden_states=hs_flat, router_logits=router_logits)
        routed_y = routed_y.view(bsz, seq_len, h)

        n_pos = bsz * seq_len
        self.total += n_pos
        self.per_layer_total[mi] += n_pos
        self.per_step_total[self.current_step] += n_pos

        if (mi in self.reuse_layers and self.qualifying_mask is not None
                and mi in self.routed_cache):
            cached = self.routed_cache[mi]
            if cached.shape[0] >= bsz and cached.shape[1] == seq_len:
                mask = self.qualifying_mask[:bsz]
                n = mask.sum().item()
                if n > 0:
                    routed_y = routed_y.clone()
                    routed_y[mask] = cached[:bsz][mask].to(routed_y.device)
                    self.reused += n
                    self.per_layer_reused[mi] += n
                    self.per_step_reused[self.current_step] += n

        self.routed_cache[mi] = routed_y.detach().clone()
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

def gen_with_ctrl(dllm, input_ids, ctrl, gl=128, enabled=True):
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner
    oif, ord_ = BlockDiffusionIteration.forward, BlockDiffusionRunner.decode

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
        if not is_cross_block and enabled:
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
    print("v0.1.14.8 — Token Margin Threshold Sweep (batch=32)")
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

        # Tokenize batch=32
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

        THRESHOLDS = [0.999, 0.99, 0.97, 0.95, 0.90, 0.80, 0.50]
        N_RUNS = 5

        # ============================================================
        # PART 1: temp=0, quantitative (5 runs each)
        # ============================================================
        print(f"\n{'='*80}")
        print(f"PART 1: Quantitative sweep (temp=0, {N_RUNS} runs)")
        print(f"{'='*80}")

        decoder_t0 = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm_t0 = BlockDiffusionLLM(
            model, decoder_t0,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=1, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Warmup
        with torch.inference_mode():
            dllm_t0.diff_iteration.num_forwards = 0
            dllm_t0.diff_iteration.iter_no = 0
            _ = dllm_t0.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        # Baseline
        bl_fwds = []
        for r in range(N_RUNS):
            ctrl = ThresholdSweepController()
            torch.cuda.synchronize()
            gen_with_ctrl(dllm_t0, input_ids, ctrl, enabled=False)
            bl_fwds.append(dllm_t0.diff_iteration.num_forwards)
        bl_avg_fwd = sum(bl_fwds) / len(bl_fwds)
        print(f"\n  Baseline: fwds=[{','.join(str(f) for f in bl_fwds)}] avg={bl_avg_fwd:.0f}")

        sweep_results = {}
        for thresh in THRESHOLDS:
            fwds_list = []
            reuse_pcts = []
            last_ctrl = None
            for r in range(N_RUNS):
                ctrl = ThresholdSweepController(margin_threshold=thresh)
                hooks = install_hooks(model, ctrl)
                try:
                    gen_with_ctrl(dllm_t0, input_ids, ctrl, enabled=True)
                    fwds_list.append(dllm_t0.diff_iteration.num_forwards)
                    if ctrl.total > 0:
                        reuse_pcts.append(ctrl.reused / ctrl.total * 100)
                    if r == 0:
                        last_ctrl = ctrl
                finally:
                    remove_hooks(hooks)

            avg_fwd = sum(fwds_list) / len(fwds_list)
            avg_reuse = sum(reuse_pcts) / len(reuse_pcts) if reuse_pcts else 0
            fwd_delta = avg_fwd - bl_avg_fwd

            print(f"\n  margin>{thresh}: fwds=[{','.join(str(f) for f in fwds_list)}] "
                  f"avg={avg_fwd:.0f} (Δ={fwd_delta:+.0f}) reuse={avg_reuse:.1f}%")

            # Per-layer reuse distribution (from first run)
            if last_ctrl:
                print(f"    Per-layer reuse%: ", end="")
                for li in range(NUM_MOE_LAYERS):
                    lt = last_ctrl.per_layer_total.get(li, 0)
                    lr = last_ctrl.per_layer_reused.get(li, 0)
                    pct = lr / lt * 100 if lt > 0 else 0
                    if li in last_ctrl.reuse_layers:
                        print(f"L{li}={pct:.0f}%", end=" ")
                print()

                # Per-step reuse (binned)
                steps = sorted(last_ctrl.per_step_total.keys())
                if steps:
                    n_steps = max(steps) + 1
                    bins = {"early(1-3)": [], "mid(4-8)": [], "late(9-15)": [], "final(16+)": []}
                    for s in steps:
                        st = last_ctrl.per_step_total[s]
                        sr = last_ctrl.per_step_reused.get(s, 0)
                        pct = sr / st * 100 if st > 0 else 0
                        if s <= 3: bins["early(1-3)"].append(pct)
                        elif s <= 8: bins["mid(4-8)"].append(pct)
                        elif s <= 15: bins["late(9-15)"].append(pct)
                        else: bins["final(16+)"].append(pct)
                    print(f"    Per-step reuse: ", end="")
                    for bname, vals in bins.items():
                        avg = sum(vals) / len(vals) if vals else 0
                        print(f"{bname}={avg:.1f}%", end="  ")
                    print()

            sweep_results[thresh] = {
                "avg_fwd": avg_fwd, "fwd_delta": fwd_delta,
                "avg_reuse_pct": avg_reuse, "fwds": fwds_list,
            }

        # Summary table
        print(f"\n{'='*80}")
        print(f"SUMMARY TABLE (temp=0, batch=32)")
        print(f"{'='*80}")
        print(f"  {'Threshold':>10s} {'Reuse%':>7s} {'AvgFwd':>7s} {'ΔFwd':>6s} {'Theoretical':>12s}")
        print(f"  {'':>10s} {'':>7s} {'':>7s} {'':>6s} {'Savings':>12s}")
        print(f"  {'-'*48}")
        print(f"  {'baseline':>10s} {'0.0%':>7s} {bl_avg_fwd:>7.0f} {'+0':>6s} {'0%':>12s}")
        for thresh in THRESHOLDS:
            r = sweep_results[thresh]
            # Theoretical savings: reuse% × routed_fraction(65%) × moe_fraction(55%)
            theory = r["avg_reuse_pct"] * 0.65 * 0.55 / 100
            print(f"  {'>' + str(thresh):>10s} {r['avg_reuse_pct']:>6.1f}% {r['avg_fwd']:>7.0f} "
                  f"{r['fwd_delta']:>+5.0f} {theory*100:>11.1f}%")

        # ============================================================
        # PART 2: temp=0.7, output quality (1 run, print text)
        # ============================================================
        print(f"\n{'='*80}")
        print(f"PART 2: Output quality check (temp=0.7, batch=32)")
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

        # Baseline output
        ctrl_bl = ThresholdSweepController()
        bl_output = gen_with_ctrl(dllm_t7, input_ids, ctrl_bl, enabled=False)
        bl_tokens = bl_output[:, prompt_len:].cpu()

        # Key thresholds to compare: 0.99, 0.95, 0.80
        CHECK_THRESHOLDS = [0.99, 0.95, 0.80]
        thresh_outputs = {}
        for thresh in CHECK_THRESHOLDS:
            ctrl = ThresholdSweepController(margin_threshold=thresh)
            hooks = install_hooks(model, ctrl)
            try:
                out = gen_with_ctrl(dllm_t7, input_ids, ctrl, enabled=True)
                thresh_outputs[thresh] = out[:, prompt_len:].cpu()
            finally:
                remove_hooks(hooks)

        # Print comparison for first 4 batches
        for bi in range(4):
            print(f"\n{'─'*80}")
            print(f"  BATCH {bi}: {PROMPTS[bi][:65]}...")
            print(f"{'─'*80}")

            # Baseline
            gt = bl_tokens[bi]
            valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
            text = tokenizer.decode(valid, skip_special_tokens=True)
            print(f"\n  [baseline]:")
            print(f"  {text[:350]}")

            for thresh in CHECK_THRESHOLDS:
                gt2 = thresh_outputs[thresh][bi]
                valid2 = gt2[(gt2 != 0) & (gt2 != EOS_ID) & (gt2 != MASK_ID)]
                text2 = tokenizer.decode(valid2, skip_special_tokens=True)
                print(f"\n  [margin>{thresh}]:")
                print(f"  {text2[:350]}")

        # Save
        save_path = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction" / "threshold_sweep_results.json"
        with open(save_path, "w") as f:
            json.dump({str(k): v for k, v in sweep_results.items()}, f, indent=2)
        print(f"\nSaved to {save_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
