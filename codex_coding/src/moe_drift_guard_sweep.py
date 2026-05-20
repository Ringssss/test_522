#!/usr/bin/env python3
"""
v0.1.14.8c — Drift Guard Effectiveness Sweep

Test whether drift guard can rescue aggressive margin thresholds.
margin ∈ {0.90, 0.80, 0.70, 0.50, 0.30} × drift_budget ∈ {0.02, 0.01, 0.005}
batch=32, L4-14, temp=0 (5 runs) + temp=0.7 (output text)
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


class DriftGuardController:
    def __init__(self, margin_threshold=0.99, drift_budget=0.01):
        self.margin_threshold = margin_threshold
        self.drift_budget = drift_budget
        self.reuse_layers = set(range(4, 15))
        self.routed_cache = {}
        self.shared_cache = {}
        self.qualifying_mask = None
        self.total = 0
        self.reused = 0
        self.drift_blocked = 0
        self.step = 0

    def reset_block(self):
        self.routed_cache.clear()
        self.shared_cache.clear()
        self.qualifying_mask = None
        self.step = 0

    def update(self, logits):
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            top2 = probs.topk(2, dim=-1).values
            self.qualifying_mask = (top2[:, :, 0] - top2[:, :, 1]) > self.margin_threshold
        self.step += 1

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

        if (mi in self.reuse_layers and self.qualifying_mask is not None
                and mi in self.routed_cache):
            cached = self.routed_cache[mi]
            if cached.shape[0] >= bsz and cached.shape[1] == seq_len:
                # Start with margin mask
                mask = self.qualifying_mask[:bsz].clone()

                # Drift guard: check shared_cos
                if mi in self.shared_cache:
                    sp = self.shared_cache[mi][:bsz].to(shared_res.device).float()
                    sn = shared_res[:bsz].float()
                    cos_sim = F.cosine_similarity(sn, sp, dim=-1)  # [bsz, seq_len]
                    drift_ok = ((1.0 - cos_sim) < self.drift_budget)
                    before = mask.sum().item()
                    mask = mask & drift_ok
                    self.drift_blocked += int(before - mask.sum().item())

                n = mask.sum().item()
                if n > 0:
                    routed_y = routed_y.clone()
                    routed_y[mask] = cached[:bsz][mask].to(routed_y.device)
                    self.reused += n

        self.routed_cache[mi] = routed_y.detach().clone()
        self.shared_cache[mi] = shared_res.detach().clone()
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
    print("v0.1.14.8c — Drift Guard Effectiveness Sweep")
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

        MARGINS = [0.90, 0.80, 0.70, 0.50, 0.30]
        DRIFTS = [0.02, 0.01, 0.005]
        N_RUNS = 5

        # No-guard baselines (from previous sweep)
        NO_GUARD = {
            0.90: (23.2, 148, -1),
            0.80: (25.4, 150, +1),
            0.70: (27.8, 152, +3),
            0.50: (30.2, 153, +4),
            0.30: (33.2, 154, +5),
        }

        # ---- PART 1: temp=0 ----
        print(f"\n{'='*80}")
        print(f"PART 1: Quantitative (temp=0, {N_RUNS} runs each)")
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

        # Baseline fwd count
        bl_fwds = []
        for r in range(N_RUNS):
            ctrl = DriftGuardController()
            gen_with_ctrl(dllm, input_ids, ctrl, enabled=False)
            bl_fwds.append(dllm.diff_iteration.num_forwards)
        bl_avg = sum(bl_fwds) / len(bl_fwds)
        print(f"\n  Baseline: fwds={bl_fwds} avg={bl_avg:.0f}")

        all_results = {}
        for margin in MARGINS:
            ng_reuse, ng_fwd, ng_delta = NO_GUARD[margin]
            print(f"\n  --- margin > {margin} (no guard: reuse={ng_reuse}%, fwd={ng_fwd}, Δ={ng_delta:+d}) ---")

            for drift in DRIFTS:
                fwds_list, reuse_pcts, blocked_list = [], [], []
                for r in range(N_RUNS):
                    ctrl = DriftGuardController(margin_threshold=margin, drift_budget=drift)
                    hooks = install_hooks(model, ctrl)
                    try:
                        gen_with_ctrl(dllm, input_ids, ctrl, enabled=True)
                        fwds_list.append(dllm.diff_iteration.num_forwards)
                        if ctrl.total > 0:
                            reuse_pcts.append(ctrl.reused / ctrl.total * 100)
                            blocked_list.append(ctrl.drift_blocked)
                    finally:
                        remove_hooks(hooks)

                avg_fwd = sum(fwds_list) / len(fwds_list)
                avg_reuse = sum(reuse_pcts) / len(reuse_pcts) if reuse_pcts else 0
                avg_blocked = sum(blocked_list) / len(blocked_list) if blocked_list else 0
                delta = avg_fwd - bl_avg
                reuse_drop = ng_reuse - avg_reuse

                improved = "IMPROVED" if delta < ng_delta else ("SAME" if delta == ng_delta else "WORSE")

                print(f"    drift<{drift}: fwd={avg_fwd:.0f} (Δ={delta:+.0f}) "
                      f"reuse={avg_reuse:.1f}% (↓{reuse_drop:.1f}%) "
                      f"blocked={avg_blocked:.0f} → {improved}")

                all_results[(margin, drift)] = {
                    "avg_fwd": avg_fwd, "delta": delta,
                    "avg_reuse": avg_reuse, "reuse_drop": reuse_drop,
                    "avg_blocked": avg_blocked,
                }

        # ---- Summary ----
        print(f"\n{'='*80}")
        print(f"CROSS-COMPARISON: No Guard vs Drift Guard")
        print(f"{'='*80}")
        print(f"  {'Margin':>7s} {'Config':>12s} {'Reuse%':>7s} {'ΔFwd':>5s} {'Blocked':>8s} {'Effect':>10s}")
        print(f"  {'-'*55}")

        for margin in MARGINS:
            ng_reuse, ng_fwd, ng_delta = NO_GUARD[margin]
            print(f"  {'>'+str(margin):>7s} {'no_guard':>12s} {ng_reuse:>6.1f}% {ng_delta:>+4d} {'—':>8s} {'baseline':>10s}")

            for drift in DRIFTS:
                r = all_results[(margin, drift)]
                eff = "BETTER" if r["delta"] < ng_delta else ("SAME" if abs(r["delta"] - ng_delta) < 0.5 else "WORSE")
                print(f"  {'':>7s} {'drift<'+str(drift):>12s} {r['avg_reuse']:>6.1f}% "
                      f"{r['delta']:>+4.0f} {r['avg_blocked']:>7.0f} {eff:>10s}")

        # ---- PART 2: temp=0.7 output for interesting configs ----
        print(f"\n{'='*80}")
        print(f"PART 2: Output quality (temp=0.7)")
        print(f"  Testing: margin>0.70+drift<0.01, margin>0.50+drift<0.005")
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
        ctrl_bl = DriftGuardController()
        bl_out = gen_with_ctrl(dllm_t7, input_ids, ctrl_bl, enabled=False)
        bl_tokens = bl_out[:, prompt_len:].cpu()

        # Interesting configs
        QUALITY_CHECKS = [
            (0.70, 0.01, "margin>0.70 + drift<0.01"),
            (0.50, 0.005, "margin>0.50 + drift<0.005"),
            (0.30, 0.005, "margin>0.30 + drift<0.005"),
        ]

        check_outputs = {}
        for margin, drift, label in QUALITY_CHECKS:
            ctrl = DriftGuardController(margin_threshold=margin, drift_budget=drift)
            hooks = install_hooks(model, ctrl)
            try:
                out = gen_with_ctrl(dllm_t7, input_ids, ctrl, enabled=True)
                check_outputs[label] = out[:, prompt_len:].cpu()
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

            for margin, drift, label in QUALITY_CHECKS:
                gt2 = check_outputs[label][bi]
                valid2 = gt2[(gt2 != 0) & (gt2 != EOS_ID) & (gt2 != MASK_ID)]
                print(f"\n  [{label}]:")
                print(f"  {tokenizer.decode(valid2, skip_special_tokens=True)[:300]}")

        print(f"\nDone.")


if __name__ == "__main__":
    main()
