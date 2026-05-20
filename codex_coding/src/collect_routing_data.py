#!/usr/bin/env python3
"""
v0.1.15.3 Phase 1a — Collect gate routing data for offline Expert Budgeting analysis.

Runs baseline generation (batch=32, temp=0) and logs per-(step, layer) full gate
weights w[N, 256] and the original top-p routing decisions. Saves to a .pt file
for offline analysis without re-running the model.
"""

from __future__ import annotations
import os, sys, socket
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


class RoutingCollector:
    """Collects full gate weights and top-p decisions at each (step, layer)."""

    def __init__(self):
        # Store per (step, layer): gate_weights [N, 256] and topk results
        self.data = []  # list of dicts per step
        self.current_step = {}  # layer_idx -> {gate_w, topk_idx, topk_w}
        self.step_count = 0

    def reset_block(self):
        pass  # Don't clear data across blocks

    def start_step(self):
        self.current_step = {}

    def end_step(self):
        if self.current_step:
            self.data.append(self.current_step)
            self.step_count += 1
        self.current_step = {}

    def record_layer(self, layer_idx, gate_weights, topk_idx, topk_weight, n_tokens):
        # Save to CPU to avoid GPU OOM across many steps
        self.current_step[layer_idx] = {
            "gate_w": gate_weights.cpu(),    # [N, 256]
            "topk_idx": topk_idx.cpu(),      # [N, 8]
            "topk_w": topk_weight.cpu(),     # [N, 8]
            "n_tokens": n_tokens,
        }


def install_collection_hooks(model, collector):
    hooks = []
    mi = 0
    for layer in model.model.layers:
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue
        moe = layer.mlp
        orig = moe.forward
        idx = mi

        def make_hook(moe_mod, layer_idx, coll):
            def hooked_forward(hidden_states):
                bsz, seq_len, h = hidden_states.shape
                hs_flat = hidden_states.view(-1, h)
                n = bsz * seq_len

                # Get full gate output
                topk_idx, topk_weight, _ = moe_mod.gate(hs_flat)

                # Also get full logits for analysis
                gate_logits = moe_mod.gate.get_logits(hs_flat)  # [N, 256]
                # Convert to weights (softmax over experts)
                # Actually, the gate already applies its routing function.
                # For our analysis we need the raw routing weights per expert.
                # topk_weight already has the top-8 weights. For experts NOT in
                # top-8, we need to know their weights too.
                # Use gate_logits with softmax to get full distribution.
                gate_weights = torch.softmax(gate_logits.float(), dim=-1)  # [N, 256]

                coll.record_layer(layer_idx, gate_weights, topk_idx, topk_weight, n)

                # Normal forward (no modification)
                shared_res = moe_mod.shared_experts(hidden_states)
                router_logits = moe_mod.gate.get_logits(hs_flat)
                routed_y = moe_mod.experts.forward_impl(
                    hidden_states=hs_flat, router_logits=router_logits)
                routed_y = routed_y.view(bsz, seq_len, h)
                out = routed_y + shared_res if moe_mod.config.num_shared_experts is not None else routed_y
                return out
            return hooked_forward

        moe.forward = make_hook(moe, idx, collector)
        hooks.append((moe, orig))
        mi += 1
    return hooks


def remove_hooks(hooks):
    for moe, orig in hooks:
        moe.forward = orig


def gen_with_collector(dllm, input_ids, collector, gl=128):
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner
    oif = BlockDiffusionIteration.forward
    ord_ = BlockDiffusionRunner.decode

    def pd(self_runner, model, decoder, x, kv_cache, block, block_loc,
           block_id, pos_ids, attn_mask, block_length=32, cross_block_attn_mask=None):
        collector.reset_block()
        return ord_(self_runner, model, decoder, x, kv_cache, block, block_loc,
                    block_id, pos_ids, attn_mask, block_length, cross_block_attn_mask)

    def pf(self_iter, model, decoder, x, kv_cache, block, block_loc,
           block_id, pos_ids, attn_mask, past_key_values,
           replace_position, backend, is_cross_block=False, block_length=32):
        collector.start_step()
        out = oif(self_iter, model, decoder, x, kv_cache, block, block_loc,
                  block_id, pos_ids, attn_mask, past_key_values,
                  replace_position, backend, is_cross_block, block_length)
        collector.end_step()
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
    print("v0.1.15.3 Phase 1a — Routing Data Collection")
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
        print(f"Input: {input_ids.shape}")

        decoder = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm = BlockDiffusionLLM(
            model, decoder,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=1, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Warmup
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print("Warmup done.")

        # Collect
        print("Collecting routing data...")
        collector = RoutingCollector()
        hooks = install_collection_hooks(model, collector)
        try:
            out = gen_with_collector(dllm, input_ids, collector)
            fwd_count = dllm.diff_iteration.num_forwards
        finally:
            remove_hooks(hooks)

        print(f"Forwards: {fwd_count}")
        print(f"Steps collected: {collector.step_count}")
        print(f"Layers per step: {len(collector.data[0]) if collector.data else 0}")

        # Sample check
        if collector.data:
            s0 = collector.data[0]
            l0 = s0.get(0)
            if l0:
                print(f"Sample: step=0, layer=0, gate_w={list(l0['gate_w'].shape)}, "
                      f"topk_idx={list(l0['topk_idx'].shape)}")

        # Save - only save every 3rd step to manage file size
        # (149 steps × 19 layers × [1024, 256] float32 ≈ 3.6GB is too much)
        # Sample: every 3rd step, all layers
        sampled_data = []
        for si in range(0, len(collector.data), 3):
            sampled_data.append(collector.data[si])

        save_path = REPO_ROOT / "codex_coding" / "results" / "expert_budgeting_routing_data.pt"
        torch.save({
            "data": sampled_data,
            "total_steps": collector.step_count,
            "sampled_steps": len(sampled_data),
            "fwd_count": fwd_count,
            "n_layers": 19,
            "n_experts": 256,
            "batch_size": BATCH_SIZE,
        }, save_path)
        print(f"Saved {len(sampled_data)} sampled steps to {save_path}")
        print(f"File size: {save_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
