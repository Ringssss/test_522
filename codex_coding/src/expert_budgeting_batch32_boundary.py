#!/usr/bin/env python3
"""
v0.1.15.3 Phase 2d — Batch=32 Heterogeneous K Boundary Exploration

32 unique heterogeneous prompts including 5 precisely verifiable tasks.
Fixed top-p=0.75, K sweep: 120/100/80/60/40/20, with and without D1.
gen_length=256, temp=0 (stats) + temp=0.7 (full text for quality check)
"""

from __future__ import annotations
import os, sys, socket, json
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

# 32 unique prompts — 5 precisely verifiable (marked with [V])
PROMPTS = [
    # [V] #0: Math — average speed (answer: 480/7 ≈ 68.57 km/h)
    "Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h. The total distance between the two cities is 240 km. What is the average speed for the entire round trip?\n\nProblem 2: A rectangular garden has a perimeter of 56 meters.",
    # #1: AI history essay
    "Write a detailed essay about the history of artificial intelligence, covering the Dartmouth conference of 1956, the AI winters, the rise of machine learning in the 1990s, and deep learning breakthroughs.",
    # #2: Chemistry
    "You are a chemistry professor. Explain Le Chatelier's principle with examples and how it applies to industrial ammonia production via the Haber process.",
    # #3: REST API design
    "Design a complete REST API for an e-commerce platform with endpoints for user authentication, product management, shopping cart operations, and order processing.",
    # #4: Climate change
    "Analyze the global economic impact of climate change across agriculture, energy, real estate, and healthcare sectors with specific examples.",
    # #5: Quantum computing
    "Explain quantum computing to a classical CS background: qubits, superposition, entanglement, Shor's algorithm, and current hardware approaches.",
    # #6: Message queue
    "You are a systems architect. Design a distributed message queue with partition-based storage, consumer groups, replication, and exactly-once semantics.",
    # #7: LLM training
    "Write a comprehensive guide to training large language models covering data collection, tokenizer training, architecture decisions, and distributed training strategies.",
    # [V] #8: Math — quadratic equation (answer: x=2 and x=3)
    "Solve the quadratic equation x^2 - 5x + 6 = 0 step by step. Show the factoring method, then verify both solutions by substituting them back into the original equation.",
    # #9: Neural network math
    "Explain the mathematical foundations of neural networks: backpropagation, gradient descent, loss functions, and the universal approximation theorem.",
    # #10: Ride-sharing architecture
    "Design a microservices architecture for a ride-sharing application with real-time matching, pricing, routing, payments, and driver management.",
    # #11: Cryptography history
    "Write about the history of cryptography from Caesar ciphers through RSA, elliptic curve cryptography, and post-quantum cryptographic algorithms.",
    # #12: Database indexing
    "Explain database indexing strategies: B-trees, hash indexes, bitmap indexes, and their trade-offs for OLTP vs OLAP workloads.",
    # [V] #13: Logic reasoning (answer: C must be true)
    "Solve this logic puzzle step by step: If A is true, then B is true. If B is true, then C is true. A is true. What can we conclude about B and C? Then, if D is true only when both B and C are true, what can we conclude about D?",
    # #14: CI/CD pipeline
    "Design a CI/CD pipeline for a large monorepo with microservices, including build caching, parallel testing, canary deployments, and rollback strategies.",
    # #15: Relativity
    "Explain the theory of relativity to a physics undergraduate, covering special relativity, time dilation, length contraction, and general relativity basics.",
    # #16: Programming languages
    "Write a comprehensive comparison of Python, Rust, and Go for systems programming, covering memory safety, concurrency models, and ecosystem maturity.",
    # #17: Recommendation engine
    "Design a real-time recommendation engine for a video streaming platform that handles cold start, user preferences, and content diversity.",
    # #18: CAP theorem
    "Explain the CAP theorem and its practical implications for distributed database design, with examples from Cassandra, MongoDB, and CockroachDB.",
    # [V] #19: Programming — Fibonacci (verifiable: fib(10)=55)
    "Write a Python function to compute the nth Fibonacci number. Show the function, then compute fib(1) through fib(10) step by step and list all 10 values.",
    # #20: Fraud detection
    "Design a fraud detection system for a payment processing company using machine learning, rule engines, and real-time streaming analytics.",
    # #21: Compiler optimization
    "Explain compiler optimization techniques including SSA form, loop unrolling, vectorization, and register allocation strategies.",
    # #22: Space exploration
    "Write about the history and future of space exploration, from Apollo missions through SpaceX reusability to planned Mars colonization.",
    # #23: Observability platform
    "Design an observability platform with distributed tracing, log aggregation, metrics collection, and intelligent alerting for microservices.",
    # #24: Public key crypto math
    "Explain the mathematics behind public key cryptography, including modular arithmetic, Euler's theorem, and the RSA algorithm step by step.",
    # #25: CSS layout
    "Write a guide to modern CSS layout techniques including Flexbox, Grid, Container Queries, and responsive design best practices.",
    # #26: Multi-tenant SaaS
    "Design a multi-tenant SaaS platform architecture with data isolation, custom domains, billing integration, and horizontal scaling.",
    # #27: Garbage collectors
    "Explain how garbage collectors work in JVM, Go, and Python, comparing mark-sweep, generational, and reference counting approaches.",
    # [V] #28: Factual — solar system planets (answer: Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune)
    "List all 8 planets in our solar system in order from closest to farthest from the Sun. For each planet, state whether it is a terrestrial or gas/ice giant planet, and give its approximate orbital period in Earth years.",
    # #29: Collaborative editor
    "Design a real-time collaborative document editor like Google Docs with conflict resolution, offline support, and version history.",
    # #30: OS memory management
    "Explain operating system memory management: virtual memory, page tables, TLB, demand paging, and memory-mapped files.",
    # #31: Kubernetes
    "Write a comprehensive guide to Kubernetes architecture including pods, services, ingress, operators, and cluster autoscaling.",
]

# Verifiable prompt indices and expected answers
VERIFIABLE = {
    0: "average speed ≈ 68.57 km/h (480/7)",
    8: "x = 2 and x = 3",
    13: "B is true, C is true, D is true",
    19: "fib(10) = 55",
    28: "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune",
}

sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))
from expert_budgeting_e2e import compute_active_set
from expert_budgeting_boundary import FullStackController, install_hooks, remove_hooks, gen_with_ctrl


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
    print("v0.1.15.3 Phase 2d — Batch=32 K Boundary (Heterogeneous)")
    print(f"32 unique prompts, 5 verifiable tasks")
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
                ids = torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
            padded.append(ids)
        input_ids = torch.stack(padded, dim=0).to(device)
        prompt_len = input_ids.shape[1]
        print(f"Input: {input_ids.shape}")

        GEN_LENGTH = 256
        REUSE_LAYERS = set(range(4, 15))
        TOP_P = 0.75
        QF = 0.85

        CONFIGS = [
            ("baseline",       None, False),
            ("K120_tp75",      120,  False),
            ("K100_tp75",      100,  False),
            ("K80_tp75",        80,  False),
            ("K60_tp75",        60,  False),
            ("K40_tp75",        40,  False),
            ("K20_tp75",        20,  False),
            ("K80_tp75_D1",     80,  True),
            ("K60_tp75_D1",     60,  True),
            ("K40_tp75_D1",     40,  True),
            ("K20_tp75_D1",     20,  True),
        ]

        # ---- Part 1: temp=0 stats ----
        print(f"\n{'='*80}")
        print(f"PART 1: Stats (temp=0, gen_length={GEN_LENGTH}, batch={BATCH_SIZE})")
        print(f"{'='*80}")

        decoder_t0 = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm = BlockDiffusionLLM(
            model, decoder_t0,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=1, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print("Warmup done.\n")

        results = {}
        for cname, K, use_d1 in CONFIGS:
            margin = 0.90 if use_d1 else None
            rl = REUSE_LAYERS if use_d1 else set()
            tp = TOP_P if K is not None else None

            ctrl = FullStackController(
                K_target=K, quality_floor=QF, top_p=tp,
                margin_threshold=margin, reuse_layers=rl)
            hooks = install_hooks(model, ctrl)
            try:
                out = gen_with_ctrl(dllm, input_ids, ctrl, gl=GEN_LENGTH)
                fwd = dllm.diff_iteration.num_forwards
            finally:
                remove_hooks(hooks)

            avg_active = (sum(ctrl.total_active_experts) /
                          max(len(ctrl.total_active_experts), 1)) if ctrl.total_active_experts else 220.8
            avg_ept = (sum(ctrl.total_experts_per_token) /
                       max(len(ctrl.total_experts_per_token), 1)) if ctrl.total_experts_per_token else 8.0
            reuse_pct = ctrl.reused_tokens / max(ctrl.total_tokens, 1) * 100

            results[cname] = {
                "fwd": fwd, "avg_active": avg_active, "avg_ept": avg_ept,
                "reuse_pct": reuse_pct, "K": K, "D1": use_d1,
            }
            print(f"  {cname:<16s} Fwd={fwd:>4d}  Active={avg_active:>6.1f}  "
                  f"E/tok={avg_ept:>4.1f}  Reuse={reuse_pct:>5.1f}%")

        bl_fwd = results["baseline"]["fwd"]
        bl_active = results["baseline"]["avg_active"]
        print(f"\n  {'Config':<16s} {'Fwd':>4s} {'dFwd':>5s} {'Active':>7s} "
              f"{'E/tok':>5s} {'Reuse':>6s} {'HBM%':>6s}")
        print(f"  {'-'*52}")
        for cname, r in results.items():
            d = r["fwd"] - bl_fwd
            h = (1 - r["avg_active"] / bl_active) * 100 if r["K"] else 0
            print(f"  {cname:<16s} {r['fwd']:>4d} {d:>+4d} {r['avg_active']:>7.1f} "
                  f"{r['avg_ept']:>5.1f} {r['reuse_pct']:>5.1f}% {h:>5.1f}%")

        # ---- Part 2: temp=0.7 text output ----
        print(f"\n{'='*80}")
        print(f"PART 2: Output Quality (temp=0.7, gen_length={GEN_LENGTH})")
        print(f"Focus: 5 verifiable prompts (#0, #8, #13, #19, #28)")
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
        for cname, K, use_d1 in CONFIGS:
            margin = 0.90 if use_d1 else None
            rl = REUSE_LAYERS if use_d1 else set()
            tp = TOP_P if K is not None else None

            ctrl = FullStackController(
                K_target=K, quality_floor=QF, top_p=tp,
                margin_threshold=margin, reuse_layers=rl)
            hooks = install_hooks(model, ctrl)
            try:
                out = gen_with_ctrl(dllm_t7, input_ids, ctrl, gl=GEN_LENGTH)
                quality_outputs[cname] = out[:, prompt_len:].cpu()
            finally:
                remove_hooks(hooks)

        # Show verifiable prompts
        for bi in sorted(VERIFIABLE.keys()):
            print(f"\n{'='*80}")
            print(f"  [VERIFIABLE] BATCH {bi}: {PROMPTS[bi][:70]}...")
            print(f"  Expected: {VERIFIABLE[bi]}")
            print(f"{'='*80}")
            for cname, _, _ in CONFIGS:
                gt = quality_outputs[cname][bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                print(f"\n  [{cname}]:")
                print(f"  {text[:600]}")
                if len(text) > 600:
                    print(f"  ... ({len(text)} chars)")

        # Also show 2 non-verifiable for general quality sense
        for bi in [1, 3]:
            print(f"\n{'='*80}")
            print(f"  BATCH {bi}: {PROMPTS[bi][:70]}...")
            print(f"{'='*80}")
            for cname in ["baseline", "K80_tp75", "K40_tp75", "K20_tp75",
                          "K80_tp75_D1", "K20_tp75_D1"]:
                gt = quality_outputs[cname][bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                print(f"\n  [{cname}]:")
                print(f"  {text[:400]}")

        # Save
        save_data = {cname: {k: v for k, v in r.items()} for cname, r in results.items()}
        out_path = REPO_ROOT / "codex_coding" / "results" / "expert_budgeting_batch32_boundary.json"
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n\nResults saved to {out_path}")
        print("Done.")


if __name__ == "__main__":
    main()
