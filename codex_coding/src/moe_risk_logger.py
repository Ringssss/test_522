#!/usr/bin/env python3
"""
MoE Risk Logger — Full Fresh Run data collection for selective recompute research.

Captures per-step × per-layer × per-token intermediate states during dLLM inference
on the cache path (BlockDiffusionLLM). Designed as the foundation for:
  - Single-site counterfactual intervention (v0.1.14.2)
  - Proxy signal analysis (v0.1.14.4)

Collected per (step, layer, token):
  - pre-MoE hidden state (for hidden drift signals)
  - gate logits (full [E] vector)
  - top-k expert indices and routing weights
  - shared expert output
  - routed expert output
  - full MoE output (shared + routed)

Collected per (step):
  - token states: which positions are MASK / decoded
  - per-token confidence and margin from decoder logits
  - per-token top-1 prediction

Uses the cache path (BlockDiffusionLLM) with lazy+inplace KV cache optimization.
"""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ============================================================
# Constants
# ============================================================
MASK_ID = 156895
EOS_ID = 156892
NUM_EXPERTS = 256
TOP_K = 8
NUM_MOE_LAYERS = 19  # layers 1-19


# ============================================================
# MoERiskCollector: core data structure
# ============================================================
class MoERiskCollector:
    """Collects comprehensive per-(step, layer, token) data during MoE forwards.

    Data is stored in CPU memory to avoid GPU OOM.
    Only the target block's data is collected (controlled by active flag).
    """

    def __init__(self, block_length: int = 32, store_hidden: bool = True,
                 store_gate_logits: bool = True, store_outputs: bool = True,
                 max_store_batch: int = 8):
        self.block_length = block_length
        self.store_hidden = store_hidden
        self.store_gate_logits = store_gate_logits
        self.store_outputs = store_outputs
        self.max_store_batch = max_store_batch

        self.active = False
        self.recording = False
        self.current_iter = -1
        self.layer_counter = 0
        self.block_start = 0
        self.block_end = 0
        self.batch_size = 1

        # Per-iteration token state: mask_state[iter] = [batch, block_len] bool
        self.mask_state: Dict[int, torch.Tensor] = {}
        # Per-iteration token IDs: token_ids[iter] = [batch, block_len] long
        self.token_ids: Dict[int, torch.Tensor] = {}

        # Per-iteration logits info (from decoder):
        # logits_info[iter] = {confidence: [batch, block_len], margin: [batch, block_len],
        #                      top1_pred: [batch, block_len]}
        self.logits_info: Dict[int, dict] = {}

        # Per (iter, layer) data:
        # pre_moe_hidden[iter][layer] = [store_batch, block_len, hidden]
        self.pre_moe_hidden: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        # gate_logits_data[iter][layer] = [store_batch, block_len, num_experts]
        self.gate_logits_data: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        # topk_indices[iter][layer] = [store_batch, block_len, top_k]
        self.topk_indices: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        # topk_weights[iter][layer] = [store_batch, block_len, top_k]
        self.topk_weights: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        # shared_output[iter][layer] = [store_batch, block_len, hidden]
        self.shared_output: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        # routed_output[iter][layer] = [store_batch, block_len, hidden]
        self.routed_output: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        # full_moe_output[iter][layer] = [store_batch, block_len, hidden]
        self.full_moe_output: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)

    def reset(self):
        """Clear all collected data."""
        self.active = False
        self.recording = False
        self.current_iter = -1
        self.layer_counter = 0
        self.mask_state.clear()
        self.token_ids.clear()
        self.logits_info.clear()
        self.pre_moe_hidden = defaultdict(dict)
        self.gate_logits_data = defaultdict(dict)
        self.topk_indices = defaultdict(dict)
        self.topk_weights = defaultdict(dict)
        self.shared_output = defaultdict(dict)
        self.routed_output = defaultdict(dict)
        self.full_moe_output = defaultdict(dict)

    def start_iteration(self, iter_idx: int, block_tokens: torch.Tensor,
                        block_start: int, block_end: int, batch_size: int):
        """Called at the start of each diffusion iteration within the target block."""
        self.current_iter = iter_idx
        self.layer_counter = 0
        self.block_start = block_start
        self.block_end = block_end
        self.batch_size = batch_size
        self.recording = True

        # Store token state
        mask_positions = (block_tokens == MASK_ID)
        self.mask_state[iter_idx] = mask_positions.cpu()
        self.token_ids[iter_idx] = block_tokens.cpu()

    def end_iteration(self):
        """Called at the end of each diffusion iteration."""
        self.layer_counter = 0
        self.recording = False

    def record_logits_info(self, logits: torch.Tensor, block_tokens: torch.Tensor):
        """Extract confidence, margin, and top-1 prediction from decoder logits.

        Args:
            logits: [batch, block_len, vocab_size] — raw logits from model
            block_tokens: [batch, block_len] — current token IDs
        """
        if not self.active or not self.recording:
            return
        it = self.current_iter
        store_bsz = min(logits.shape[0], self.max_store_batch)

        with torch.no_grad():
            probs = F.softmax(logits[:store_bsz].float(), dim=-1)
            top2_probs, top2_idx = probs.topk(2, dim=-1)  # [bsz, block_len, 2]
            confidence = top2_probs[:, :, 0]  # [bsz, block_len]
            margin = top2_probs[:, :, 0] - top2_probs[:, :, 1]  # [bsz, block_len]
            top1_pred = top2_idx[:, :, 0]  # [bsz, block_len]

        self.logits_info[it] = {
            "confidence": confidence.cpu(),
            "margin": margin.cpu(),
            "top1_pred": top1_pred.cpu(),
        }

    def record_pre_moe_hidden(self, hidden_states: torch.Tensor):
        """Record hidden states entering MoE block (after post_attention_layernorm).

        Args:
            hidden_states: [batch, seq_len, hidden] — may be full seq or block only
        """
        if not self.active or not self.recording or not self.store_hidden:
            return
        layer = self.layer_counter
        it = self.current_iter
        store_bsz = min(hidden_states.shape[0], self.max_store_batch)
        seq_len = hidden_states.shape[1]
        block_len = self.block_end - self.block_start

        # Extract block portion
        if seq_len <= block_len:
            # Cache path: model only sees block tokens
            hs = hidden_states[:store_bsz]
        else:
            # No-cache path: model sees full sequence up to block_end
            hs = hidden_states[:store_bsz, self.block_start:self.block_end]

        self.pre_moe_hidden[it][layer] = hs.detach().cpu()

    def record_gate_output(self, gate_logits: torch.Tensor,
                           topk_idx: torch.Tensor, topk_weight: torch.Tensor):
        """Record gate logits and routing decisions.

        Args:
            gate_logits: [batch*seq, num_experts] — raw gate output
            topk_idx: [batch*seq, top_k] — selected expert indices
            topk_weight: [batch*seq, top_k] — normalized routing weights
        """
        if not self.active or not self.recording:
            return
        layer = self.layer_counter
        it = self.current_iter
        total_tokens = gate_logits.shape[0]
        seq_per_batch = total_tokens // self.batch_size
        block_len = self.block_end - self.block_start
        store_bsz = min(self.batch_size, self.max_store_batch)

        # Reshape to [batch, seq, ...]
        gl_3d = gate_logits.view(self.batch_size, seq_per_batch, -1)
        ti_3d = topk_idx.view(self.batch_size, seq_per_batch, -1)
        tw_3d = topk_weight.view(self.batch_size, seq_per_batch, -1)

        # Extract block portion
        if seq_per_batch <= block_len:
            gl_block = gl_3d[:store_bsz]
            ti_block = ti_3d[:store_bsz]
            tw_block = tw_3d[:store_bsz]
        else:
            gl_block = gl_3d[:store_bsz, self.block_start:self.block_end]
            ti_block = ti_3d[:store_bsz, self.block_start:self.block_end]
            tw_block = tw_3d[:store_bsz, self.block_start:self.block_end]

        if self.store_gate_logits:
            self.gate_logits_data[it][layer] = gl_block.detach().cpu()
        self.topk_indices[it][layer] = ti_block.detach().cpu()
        self.topk_weights[it][layer] = tw_block.detach().cpu()

    def record_moe_outputs(self, shared_out: torch.Tensor,
                           routed_out: torch.Tensor,
                           full_out: torch.Tensor):
        """Record shared expert output, routed expert output, and combined MoE output.

        Args:
            shared_out: [batch, seq_len, hidden]
            routed_out: [batch, seq_len, hidden]
            full_out: [batch, seq_len, hidden]
        """
        if not self.active or not self.recording or not self.store_outputs:
            return
        layer = self.layer_counter
        it = self.current_iter
        store_bsz = min(shared_out.shape[0], self.max_store_batch)
        seq_len = shared_out.shape[1]
        block_len = self.block_end - self.block_start

        if seq_len <= block_len:
            s = shared_out[:store_bsz]
            r = routed_out[:store_bsz]
            f = full_out[:store_bsz]
        else:
            s = shared_out[:store_bsz, self.block_start:self.block_end]
            r = routed_out[:store_bsz, self.block_start:self.block_end]
            f = full_out[:store_bsz, self.block_start:self.block_end]

        self.shared_output[it][layer] = s.detach().cpu()
        self.routed_output[it][layer] = r.detach().cpu()
        self.full_moe_output[it][layer] = f.detach().cpu()
        # Increment layer counter AFTER recording outputs (one MoE block = one layer)
        self.layer_counter += 1

    @property
    def num_iterations(self) -> int:
        return len(self.mask_state)

    def summary(self) -> str:
        n = self.num_iterations
        layers = sorted(self.topk_indices.get(0, {}).keys()) if n > 0 else []
        return (f"MoERiskCollector: {n} iterations, "
                f"{len(layers)} MoE layers, "
                f"store_hidden={self.store_hidden}, "
                f"store_gate_logits={self.store_gate_logits}, "
                f"store_outputs={self.store_outputs}")


# ============================================================
# Hook installation — instrument MoE layers
# ============================================================
def install_risk_hooks(model, collector: MoERiskCollector):
    """Install hooks on all MoE layers to capture intermediate states.

    Replaces each MoE block's forward to decompose and record:
      1. pre-MoE hidden state (input to MoE block)
      2. gate logits + routing decisions
      3. shared output, routed output, full MoE output

    Returns a list of hook handles for cleanup.
    """
    hooks = []
    layers = model.model.layers

    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue  # skip dense FFN layer (layer 0)

        moe_block = layer.mlp

        # Patch MoE block forward to capture all data
        orig_moe_forward = moe_block.forward

        def make_moe_hook(orig_fn, col, moe_mod):
            def hooked_moe_forward(hidden_states):
                # Record pre-MoE hidden (input to MoE after post_attn_layernorm)
                col.record_pre_moe_hidden(hidden_states)

                # Execute decomposed forward to capture shared + routed separately
                bsz, seq_len, h = hidden_states.shape
                hidden_states_flat = hidden_states.view(-1, h)

                # Shared expert (always fresh)
                shared_res = moe_mod.shared_experts(hidden_states)

                # Gate: get logits + routing
                gate_logits = moe_mod.gate.get_logits(hidden_states_flat)
                topk_idx, topk_weight, full_logits = moe_mod.gate(hidden_states_flat)

                # Record gate output
                col.record_gate_output(full_logits, topk_idx, topk_weight)

                # Routed experts via original fused_moe path
                # Use gate.get_logits path which goes through forward_impl → custom_routing
                routed_y = moe_mod.experts.forward_impl(
                    hidden_states=hidden_states_flat,
                    router_logits=gate_logits)
                routed_y = routed_y.view(bsz, seq_len, h)

                # Combine
                if moe_mod.config.num_shared_experts is not None:
                    full_out = routed_y + shared_res
                else:
                    full_out = routed_y

                # Record outputs
                col.record_moe_outputs(shared_res, routed_y, full_out)

                return full_out

            return hooked_moe_forward

        moe_block.forward = make_moe_hook(orig_moe_forward, collector, moe_block)
        hooks.append(('moe_forward', moe_block, orig_moe_forward))

    return hooks


def remove_risk_hooks(hooks):
    """Restore original forwards."""
    for kind, obj, orig in hooks:
        if kind == 'moe_forward':
            obj.forward = orig


# ============================================================
# Instrumented generate — patches iteration/runner to track steps
# ============================================================
def run_logged_generate(dllm, input_ids: torch.Tensor, collector: MoERiskCollector,
                        target_block_idx: int = 1,
                        gen_length: int = 128, block_length: int = 32):
    """Run generation with full logging on the target block.

    Uses cache path (BlockDiffusionLLM). Patches BlockDiffusionIteration.forward
    and BlockDiffusionRunner.decode to intercept iteration boundaries and capture
    logits info.

    Args:
        dllm: BlockDiffusionLLM instance
        input_ids: [batch, prompt_len] input token IDs
        collector: MoERiskCollector to store data
        target_block_idx: which generation block to instrument (0-indexed)
        gen_length: total generation length
        block_length: block size (default 32)

    Returns:
        output token IDs from dllm.generate
    """
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner

    orig_iter_forward = BlockDiffusionIteration.forward
    orig_runner_decode = BlockDiffusionRunner.decode
    current_block_idx = [0]
    iteration_in_block = [0]

    def patched_decode(self_runner, model, decoder, x, kv_cache, block, block_loc,
                       block_id, pos_ids, attn_mask, block_length=32,
                       cross_block_attn_mask=None):
        gen_block_idx = current_block_idx[0]
        is_target = (gen_block_idx == target_block_idx)
        if is_target:
            collector.active = True
            iteration_in_block[0] = 0
            print(f"  [Logger] Activated for gen block {gen_block_idx} "
                  f"(block_loc={block_loc.start}:{block_loc.end})", flush=True)
        result = orig_runner_decode(self_runner, model, decoder, x, kv_cache, block,
                                    block_loc, block_id, pos_ids, attn_mask,
                                    block_length, cross_block_attn_mask)
        if is_target:
            collector.active = False
            print(f"  [Logger] Deactivated. Collected {collector.num_iterations} iterations.",
                  flush=True)
        current_block_idx[0] += 1
        return result

    def patched_forward(self_iter, model, decoder, x, kv_cache, block, block_loc,
                        block_id, pos_ids, attn_mask, past_key_values,
                        replace_position, backend, is_cross_block=False,
                        block_length=32):
        gen_block_idx = current_block_idx[0]
        is_target = (gen_block_idx == target_block_idx)

        if is_target and collector.active and not is_cross_block:
            block_tokens = x.data[:, block_loc.start:block_loc.end]
            collector.start_iteration(
                iteration_in_block[0], block_tokens,
                block_loc.start, block_loc.end, x.data.shape[0])

        # Run original forward
        output = orig_iter_forward(
            self_iter, model, decoder, x, kv_cache, block, block_loc,
            block_id, pos_ids, attn_mask, past_key_values, replace_position,
            backend, is_cross_block, block_length)

        if is_target and collector.active and not is_cross_block:
            # Capture logits info (confidence, margin, top-1)
            logits = output.logits
            block_tokens = x.data[:, block_loc.start:block_loc.end]
            collector.record_logits_info(logits, block_tokens)
            collector.end_iteration()
            iteration_in_block[0] += 1

        return output

    # Monkey-patch
    BlockDiffusionIteration.forward = patched_forward
    BlockDiffusionRunner.decode = patched_decode

    try:
        with torch.inference_mode():
            out = dllm.generate(input_ids, gen_length=gen_length,
                                block_length=block_length)
    finally:
        # Restore originals
        BlockDiffusionIteration.forward = orig_iter_forward
        BlockDiffusionRunner.decode = orig_runner_decode

    return out


# ============================================================
# Verification utility
# ============================================================
def verify_logger_correctness(dllm, input_ids: torch.Tensor,
                              gen_length: int = 128, block_length: int = 32):
    """Verify that the logger does not change model output.

    Temporarily sets decoder temperature to 0 for deterministic comparison.
    Runs generation twice: once without hooks, once with hooks.
    Compares output tokens — they must be identical.
    """
    print("=" * 60)
    print("Verification: logger correctness check (temperature=0)")
    print("=" * 60)

    # Temporarily set temperature=0 for deterministic comparison
    orig_temp = dllm.decoder.temperature
    dllm.decoder.temperature = 0.0

    # Run 1: baseline (no hooks)
    dllm.diff_iteration.num_forwards = 0
    dllm.diff_iteration.iter_no = 0
    with torch.inference_mode():
        out_baseline = dllm.generate(input_ids.clone(), gen_length=gen_length,
                                     block_length=block_length)
    fwd_baseline = dllm.diff_iteration.num_forwards
    print(f"  Baseline: {fwd_baseline} forwards, output shape {out_baseline.shape}")

    # Run 2: with hooks
    collector = MoERiskCollector(block_length=block_length,
                                store_hidden=True,
                                store_gate_logits=True,
                                store_outputs=True,
                                max_store_batch=input_ids.shape[0])
    hooks = install_risk_hooks(dllm.model, collector)
    dllm.diff_iteration.num_forwards = 0
    dllm.diff_iteration.iter_no = 0
    try:
        out_logged = run_logged_generate(
            dllm, input_ids.clone(), collector,
            target_block_idx=1, gen_length=gen_length, block_length=block_length)
    finally:
        remove_risk_hooks(hooks)
    fwd_logged = dllm.diff_iteration.num_forwards
    print(f"  Logged:   {fwd_logged} forwards, output shape {out_logged.shape}")

    # Compare
    min_len = min(out_baseline.shape[1], out_logged.shape[1])
    match = (out_baseline[:, :min_len] == out_logged[:, :min_len]).all()
    fwd_match = (fwd_baseline == fwd_logged)
    print(f"  Tokens match: {match.item()}")
    print(f"  Forwards match: {fwd_match} ({fwd_baseline} vs {fwd_logged})")
    print(f"  Collector: {collector.summary()}")

    if match.item() and fwd_match:
        print("  PASS: Logger does not change model output.")
    else:
        print("  FAIL: Logger changed model output!")

    # Restore temperature
    dllm.decoder.temperature = orig_temp

    return match.item() and fwd_match, collector


# ============================================================
# Main: standalone test
# ============================================================
def main():
    import socket
    import sys
    from contextlib import closing
    from functools import partial

    from transformers import AutoConfig, AutoTokenizer

    REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
    MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
    GEN_LENGTH = 128
    BLOCK_LENGTH = 32
    THRESHOLD = 0.90
    TEMPERATURE = 0.7
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
    ]

    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    # --- Init ---
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    sys.path.insert(0, str(REPO_ROOT / "lib_cite" / "dInfer" / "python"))
    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (
        BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
        ThresholdParallelDecoder,
    )
    from dinfer.model import LLaDA2MoeModelLM

    port = find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    # --- Load model ---
    print("Loading model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        # Warmup
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)

        decoder = ThresholdParallelDecoder(
            temperature=TEMPERATURE,
            threshold=THRESHOLD,
            mask_id=MASK_ID,
            eos_id=EOS_ID,
        )

        # Build cache-path dLLM (BlockDiffusionLLM)
        dllm = BlockDiffusionLLM(
            model, decoder,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True,
            maximum_unroll=1,
            expected_tpf=15,
            backend='vllm',
            lazy_cache_update=True,
            inplace_cache_update=True,
        )

        # --- Tokenize ---
        BATCH_SIZE = 8
        print(f"\nTokenizing {BATCH_SIZE} prompts...", flush=True)
        all_ids = []
        for i in range(BATCH_SIZE):
            text = PROMPTS[i % len(PROMPTS)]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=False)
            ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
            all_ids.append(ids)
        max_len = max(ids.shape[0] for ids in all_ids)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        padded = []
        for ids in all_ids:
            if ids.shape[0] < max_len:
                pad = torch.full((max_len - ids.shape[0],), pad_id, dtype=ids.dtype)
                ids = torch.cat([pad, ids])
            padded.append(ids)
        input_ids = torch.stack(padded, dim=0).to(device)
        print(f"  Input shape: {input_ids.shape}")

        # --- Step 1: Verify correctness ---
        print("\n" + "=" * 60)
        print("STEP 1: Correctness verification")
        print("=" * 60)
        passed, collector = verify_logger_correctness(
            dllm, input_ids, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)

        if not passed:
            print("\nFATAL: Logger changes model output. Aborting.")
            return

        # --- Step 2: Full logged run ---
        print("\n" + "=" * 60)
        print("STEP 2: Full logged run with data dump")
        print("=" * 60)
        collector2 = MoERiskCollector(
            block_length=BLOCK_LENGTH,
            store_hidden=True,
            store_gate_logits=True,
            store_outputs=True,
            max_store_batch=BATCH_SIZE,
        )
        hooks = install_risk_hooks(model, collector2)
        dllm.diff_iteration.num_forwards = 0
        dllm.diff_iteration.iter_no = 0
        t0 = time.time()
        try:
            out = run_logged_generate(
                dllm, input_ids.clone(), collector2,
                target_block_idx=1, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        finally:
            remove_risk_hooks(hooks)
        elapsed = time.time() - t0
        print(f"\n  Generation time: {elapsed:.2f}s")
        print(f"  Forwards: {dllm.diff_iteration.num_forwards}")
        print(f"  {collector2.summary()}")

        # --- Print data summary ---
        n_iters = collector2.num_iterations
        print(f"\n  Iterations collected: {n_iters}")
        if n_iters > 0:
            layers = sorted(collector2.topk_indices[0].keys())
            print(f"  MoE layers: {layers}")
            print(f"  Data shapes (iter 0, layer {layers[0]}):")
            l0 = layers[0]
            if l0 in collector2.pre_moe_hidden.get(0, {}):
                print(f"    pre_moe_hidden: {collector2.pre_moe_hidden[0][l0].shape}")
            if l0 in collector2.gate_logits_data.get(0, {}):
                print(f"    gate_logits:    {collector2.gate_logits_data[0][l0].shape}")
            if l0 in collector2.topk_indices.get(0, {}):
                print(f"    topk_indices:   {collector2.topk_indices[0][l0].shape}")
            if l0 in collector2.topk_weights.get(0, {}):
                print(f"    topk_weights:   {collector2.topk_weights[0][l0].shape}")
            if l0 in collector2.shared_output.get(0, {}):
                print(f"    shared_output:  {collector2.shared_output[0][l0].shape}")
            if l0 in collector2.routed_output.get(0, {}):
                print(f"    routed_output:  {collector2.routed_output[0][l0].shape}")
            if l0 in collector2.full_moe_output.get(0, {}):
                print(f"    full_moe_output:{collector2.full_moe_output[0][l0].shape}")
            if 0 in collector2.logits_info:
                li = collector2.logits_info[0]
                print(f"    confidence:     {li['confidence'].shape}")
                print(f"    margin:         {li['margin'].shape}")
                print(f"    top1_pred:      {li['top1_pred'].shape}")

            # Quick sanity: show mask counts per iteration
            print(f"\n  Mask counts per iteration (batch avg):")
            for it in range(n_iters):
                ms = collector2.mask_state[it].float().sum(dim=1).mean().item()
                print(f"    iter {it:>2d}: avg {ms:.1f}/32 MASK tokens")

        # --- Save collector data ---
        results_dir = Path("/home/wuhang/wuhang/dllm_wh/codex_coding/results/proxy_risk_prediction")
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / "full_fresh_run_data.pt"
        save_dict = {
            "mask_state": dict(collector2.mask_state),
            "token_ids": dict(collector2.token_ids),
            "logits_info": dict(collector2.logits_info),
            "topk_indices": {it: dict(ld) for it, ld in collector2.topk_indices.items()},
            "topk_weights": {it: dict(ld) for it, ld in collector2.topk_weights.items()},
        }
        if collector2.store_hidden:
            save_dict["pre_moe_hidden"] = {it: dict(ld) for it, ld in collector2.pre_moe_hidden.items()}
        if collector2.store_outputs:
            save_dict["shared_output"] = {it: dict(ld) for it, ld in collector2.shared_output.items()}
            save_dict["routed_output"] = {it: dict(ld) for it, ld in collector2.routed_output.items()}
            save_dict["full_moe_output"] = {it: dict(ld) for it, ld in collector2.full_moe_output.items()}
        if collector2.store_gate_logits:
            save_dict["gate_logits_data"] = {it: dict(ld) for it, ld in collector2.gate_logits_data.items()}

        torch.save(save_dict, save_path)
        file_size_mb = save_path.stat().st_size / (1024 * 1024)
        print(f"\n  Saved to {save_path} ({file_size_mb:.1f} MB)")
        print("\nDone.")


if __name__ == "__main__":
    main()
