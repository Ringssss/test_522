#!/usr/bin/env python3
"""Quick robustness check: does P2 compute-then-replace speedup hold across batch sizes?"""

from __future__ import annotations
import os, sys, time, socket
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from typing import Dict, Optional, Set

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
DEVICE = "cuda:0"

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


class P2Controller:
    """P2: margin>0.99, L4-14, compute-then-replace."""
    def __init__(self):
        self.routed_cache = {}
        self.qualifying_mask = None
        self.reuse_layers = set(range(4, 15))
        self.total = 0
        self.reused = 0

    def reset_block(self):
        self.routed_cache.clear()
        self.qualifying_mask = None

    def update(self, logits):
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            top2 = probs.topk(2, dim=-1).values
            self.qualifying_mask = (top2[:,:,0] - top2[:,:,1]) > 0.99

    def hook_forward(self, moe_mod, mi, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        shared_res = moe_mod.shared_experts(hidden_states)
        router_logits = moe_mod.gate.get_logits(hs_flat)
        routed_y = moe_mod.experts.forward_impl(
            hidden_states=hs_flat, router_logits=router_logits)
        routed_y = routed_y.view(bsz, seq_len, h)
        self.total += bsz * seq_len

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

    def pd(sr, model, decoder, x, kv, block, bloc, bid, pos, attn,
           block_length=32, cross_block_attn_mask=None):
        ctrl.reset_block()
        return ord_(sr, model, decoder, x, kv, block, bloc, bid, pos, attn,
                    block_length, cross_block_attn_mask)

    def pf(si, model, decoder, x, kv, block, bloc, bid, pos, attn,
           pkv, rp, be, is_cross_block=False, block_length=32):
        out = oif(si, model, decoder, x, kv, block, bloc, bid, pos, attn,
                  pkv, rp, be, is_cross_block, block_length)
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

    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)

        N_RUNS = 5
        GEN_LENGTH = 128

        for batch_size in [8, 32]:
            print(f"\n{'='*70}")
            print(f"BATCH SIZE = {batch_size}")
            print(f"{'='*70}")

            # Tokenize
            all_ids = []
            for i in range(batch_size):
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
                    ids = torch.cat([torch.full((mx-ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                padded.append(ids)
            input_ids = torch.stack(padded, dim=0).to(device)

            for temp in [0.0, 0.7]:
                decoder = ThresholdParallelDecoder(
                    temperature=temp, threshold=0.90,
                    mask_id=MASK_ID, eos_id=EOS_ID)
                dllm = BlockDiffusionLLM(
                    model, decoder,
                    BlockIteratorFactory(use_block_diffusion=True),
                    cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                    early_stop=True, maximum_unroll=1, expected_tpf=15,
                    backend='vllm', lazy_cache_update=True, inplace_cache_update=True,
                )

                # Warmup
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    dllm.diff_iteration.iter_no = 0
                    _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
                torch.cuda.synchronize()

                print(f"\n  temp={temp}:")
                for policy in ["baseline", "P2"]:
                    times, fwds, reuse_pcts = [], [], []
                    for r in range(N_RUNS):
                        ctrl = P2Controller()
                        enabled = (policy == "P2")
                        hooks = install_hooks(model, ctrl) if enabled else []
                        try:
                            torch.cuda.synchronize()
                            t0 = time.time()
                            out = gen_with_ctrl(dllm, input_ids, ctrl,
                                                gl=GEN_LENGTH, enabled=enabled)
                            torch.cuda.synchronize()
                            times.append(time.time() - t0)
                            fwds.append(dllm.diff_iteration.num_forwards)
                            if enabled and ctrl.total > 0:
                                reuse_pcts.append(ctrl.reused / ctrl.total * 100)
                        finally:
                            if hooks:
                                remove_hooks(hooks)

                    avg_t = sum(times) / len(times)
                    std_t = (sum((t - avg_t)**2 for t in times) / len(times)) ** 0.5
                    fwd_str = ",".join(str(f) for f in fwds)
                    reuse_str = f"reuse={sum(reuse_pcts)/len(reuse_pcts):.1f}%" if reuse_pcts else ""

                    print(f"    {policy:>10s}: {avg_t:.3f}s ±{std_t:.3f}s  "
                          f"fwds=[{fwd_str}]  {reuse_str}")

                # Speedup
                bl_times = []
                p2_times = []
                # Re-run for clean speedup calc
                for r in range(N_RUNS):
                    # baseline
                    ctrl = P2Controller()
                    torch.cuda.synchronize(); t0 = time.time()
                    gen_with_ctrl(dllm, input_ids, ctrl, gl=GEN_LENGTH, enabled=False)
                    torch.cuda.synchronize(); bl_times.append(time.time() - t0)

                    # P2
                    ctrl2 = P2Controller()
                    hooks = install_hooks(model, ctrl2)
                    try:
                        torch.cuda.synchronize(); t0 = time.time()
                        gen_with_ctrl(dllm, input_ids, ctrl2, gl=GEN_LENGTH, enabled=True)
                        torch.cuda.synchronize(); p2_times.append(time.time() - t0)
                    finally:
                        remove_hooks(hooks)

                bl_avg = sum(bl_times) / len(bl_times)
                p2_avg = sum(p2_times) / len(p2_times)
                print(f"    → Speedup: {bl_avg/p2_avg:.3f}x  "
                      f"(bl={bl_avg:.3f}s, p2={p2_avg:.3f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
