#!/usr/bin/env python3
"""
Pre-experiment 2+4: Per-layer ablation + periodic refresh simulation.

Experiment 2: For each MoE layer L, simulate "only layer L uses v1 cache for stable
positions, all other layers compute fully". Measure final logits cosine sim vs baseline.
This identifies which layers are safe to cache and which are not.

Experiment 4: Simulate periodic full refresh with different intervals N.
Every N-th iteration computes fully, other iterations use v1 cache for stable positions
(across ALL layers). Measure token match rate vs baseline.

Both use cache path (BlockDiffusionLLM), batch=1, long prompt, temperature=0.0.
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
RESULTS_DIR = REPO_ROOT / "codex_coding" / "results"

MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
GEN_LENGTH = 128
BLOCK_LENGTH = 32
THRESHOLD = 0.90
MASK_ID = 156895
EOS_ID = 156892
DEVICE = "cuda:0"
TARGET_BLOCK_IDX = 1

LONG_PROMPT = """Please solve the following problems step by step.

Problem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h. The total distance between the two cities is 240 km. What is the average speed for the entire round trip?

Problem 2: A rectangular garden has a perimeter of 56 meters. If the length is 4 meters more than twice the width, find the dimensions of the garden.

Problem 3: In a class of 40 students, 25 study Mathematics, 20 study Physics, and 10 study both subjects. How many students study neither Mathematics nor Physics?

Problem 4: A cone has a radius of 7 cm and a slant height of 25 cm. Calculate the total surface area and the volume of the cone.

Problem 5: A bank offers compound interest at 8% per annum, compounded quarterly. If you deposit $5000, how much will you have after 3 years?

Problem 6: Two pipes can fill a tank. Pipe A fills the tank in 12 hours and Pipe B fills it in 18 hours. If both pipes are opened together, but Pipe B is closed after 4 hours, how long will it take Pipe A alone to fill the remaining tank?

Problem 7: A sequence is defined as follows: a(1) = 2, a(2) = 5, and for n >= 3, a(n) = 2*a(n-1) - a(n-2) + 3. Find the first 8 terms.

Problem 8: A factory produces widgets on two assembly lines. Line A produces 300 widgets per hour with a defect rate of 2%. Line B produces 200 widgets per hour with a defect rate of 1.5%. If the factory runs both lines for 8 hours, what is the overall defect rate?

Problem 9: A cylindrical water tank with radius 3 meters and height 10 meters is being filled at a rate of 2 cubic meters per minute while being drained at 0.5 cubic meters per minute. How long will it take to fill completely?"""


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def trim_after_eos(t, eos_id):
    p = (t == eos_id).nonzero(as_tuple=True)[0]
    return t[: int(p[0].item())] if p.numel() > 0 else t


# ============================================================
# CacheInjector: hooks MoE blocks to selectively replace output
# with cached values for stable positions
# ============================================================
class CacheInjector:
    """Monkey-patches MoE blocks to inject cached output for stable positions.

    Modes:
    - 'single_layer': only cache at one specific layer (for exp 2)
    - 'all_layers': cache at all MoE layers (for exp 4)
    - 'disabled': normal computation, but still record outputs for cache building
    """

    def __init__(self, model, mask_id):
        self.mask_id = mask_id
        self.mode = 'disabled'
        self.target_layer = -1  # for single_layer mode
        self.prev_block_tokens = None
        self.stable_mask = None
        self.moe_cache = {}  # layer_idx -> [batch, seq, hidden]
        self._layer_counter = 0
        self._hooks_installed = False

        # Collect MoE layers
        self._moe_layers = []
        self._original_forwards = {}
        layers = model.model.layers
        for idx, layer in enumerate(layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                self._moe_layers.append((idx, layer.mlp))

    def reset_block(self):
        self.prev_block_tokens = None
        self.stable_mask = None
        self.moe_cache = {}

    def update_stable_mask(self, block_tokens):
        if self.prev_block_tokens is not None:
            self.stable_mask = (block_tokens == self.prev_block_tokens)
        else:
            self.stable_mask = None
        self.prev_block_tokens = block_tokens.clone()

    def should_cache_layer(self, layer_idx):
        if self.mode == 'disabled':
            return False
        if self.mode == 'single_layer':
            return layer_idx == self.target_layer
        if self.mode == 'all_layers':
            return True
        return False

    def install(self):
        if self._hooks_installed:
            return
        self._layer_counter = 0
        for idx, moe_mod in self._moe_layers:
            self._original_forwards[idx] = moe_mod.forward
            moe_mod.forward = self._make_forward(idx, moe_mod)
        self._hooks_installed = True

    def uninstall(self):
        for idx, moe_mod in self._moe_layers:
            if idx in self._original_forwards:
                moe_mod.forward = self._original_forwards[idx]
        self._original_forwards.clear()
        self._hooks_installed = False

    def reset_layer_counter(self):
        self._layer_counter = 0

    def _make_forward(self, layer_idx, moe_mod):
        injector = self

        def forward(hidden_states):
            bsz, seq_len, h = hidden_states.shape

            # Always compute full output
            res = moe_mod.shared_experts(hidden_states)
            hs_flat = hidden_states.view(-1, h)
            router_logits = moe_mod.gate.get_logits(hs_flat)
            y = moe_mod.experts.forward_impl(hidden_states=hs_flat, router_logits=router_logits)
            y = y.view(bsz, seq_len, h)
            full_output = y + res  # this is what the model would normally return

            sm = injector.stable_mask
            should_cache = injector.should_cache_layer(layer_idx)
            has_cache = layer_idx in injector.moe_cache

            if should_cache and sm is not None and sm.any() and has_cache:
                # Replace stable positions with cached output
                cached = injector.moe_cache[layer_idx]
                result = full_output.clone()
                for b in range(bsz):
                    if sm[b].any():
                        result[b, sm[b]] = cached[b, sm[b]]
                # Update cache with current full for active positions
                injector.moe_cache[layer_idx] = full_output.detach().clone()
                return result
            else:
                # No caching, just record for future use
                injector.moe_cache[layer_idx] = full_output.detach().clone()
                return full_output

        return forward


# ============================================================
# Instrumented generate with CacheInjector control
# ============================================================
def run_generate_with_injector(dllm, input_ids, injector, target_block_idx,
                               refresh_interval=0, mode='disabled', target_layer=-1):
    """Run generate with CacheInjector.

    refresh_interval: 0 = cache every iteration; N>0 = full compute every N-th iter
    mode: 'disabled' | 'single_layer' | 'all_layers'
    target_layer: which layer to cache (for single_layer mode)
    """
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner

    orig_forward = BlockDiffusionIteration.forward
    orig_decode = BlockDiffusionRunner.decode
    current_block_idx = [0]
    iteration_in_block = [0]

    def patched_decode(self_runner, model, decoder, x, kv_cache, block, block_loc,
                       block_id, pos_ids, attn_mask, block_length=32, cross_block_attn_mask=None):
        is_target = (current_block_idx[0] == target_block_idx)
        if is_target:
            injector.reset_block()
            injector.mode = mode
            injector.target_layer = target_layer
            iteration_in_block[0] = 0

        result = orig_decode(self_runner, model, decoder, x, kv_cache, block,
                             block_loc, block_id, pos_ids, attn_mask, block_length,
                             cross_block_attn_mask)

        if is_target:
            injector.mode = 'disabled'

        current_block_idx[0] += 1
        return result

    def patched_forward(self_iter, model, decoder, x, kv_cache, block, block_loc,
                        block_id, pos_ids, attn_mask, past_key_values, replace_position,
                        backend, is_cross_block=False, block_length=32):
        is_target = (current_block_idx[0] == target_block_idx)

        if is_target and not is_cross_block:
            block_tokens = x.data[:, block_loc.start:block_loc.end]
            injector.update_stable_mask(block_tokens)

            # Periodic refresh: disable caching on refresh iterations
            it = iteration_in_block[0]
            if refresh_interval > 0 and it % refresh_interval == 0:
                saved_mode = injector.mode
                injector.mode = 'disabled'
                result = orig_forward(self_iter, model, decoder, x, kv_cache, block,
                                      block_loc, block_id, pos_ids, attn_mask,
                                      past_key_values, replace_position, backend,
                                      is_cross_block, block_length)
                injector.mode = saved_mode
                # Still update cache even on refresh iterations
                # (the full output was computed, hooks recorded it)
            else:
                result = orig_forward(self_iter, model, decoder, x, kv_cache, block,
                                      block_loc, block_id, pos_ids, attn_mask,
                                      past_key_values, replace_position, backend,
                                      is_cross_block, block_length)
            iteration_in_block[0] += 1
        else:
            result = orig_forward(self_iter, model, decoder, x, kv_cache, block,
                                  block_loc, block_id, pos_ids, attn_mask,
                                  past_key_values, replace_position, backend,
                                  is_cross_block, block_length)
        return result

    BlockDiffusionIteration.forward = patched_forward
    BlockDiffusionRunner.decode = patched_decode
    try:
        with torch.inference_mode():
            out = dllm.generate(input_ids, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
    finally:
        BlockDiffusionIteration.forward = orig_forward
        BlockDiffusionRunner.decode = orig_decode
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

    port = find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("Loading model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=cfg).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)

        prompt_text = LONG_PROMPT
        if hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True, tokenize=False)
        long_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(device)
        plen = long_ids.shape[1]
        print(f"Prompt: {plen} tokens\n")

        def make_dllm():
            return BlockDiffusionLLM(
                model, ThresholdParallelDecoder(temperature=0.0, threshold=THRESHOLD,
                                                mask_id=MASK_ID, eos_id=EOS_ID),
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, lazy_cache_update=True, inplace_cache_update=True)

        injector = CacheInjector(model, MASK_ID)
        injector.install()

        # === Baseline: no caching ===
        print("Running baseline (no caching) ...", flush=True)
        dllm = make_dllm()
        out_baseline = run_generate_with_injector(
            dllm, long_ids, injector, TARGET_BLOCK_IDX, mode='disabled')
        gen_baseline = trim_after_eos(out_baseline[0][plen:], EOS_ID)
        fwd_baseline = dllm.num_forwards
        text_baseline = tokenizer.decode(gen_baseline, skip_special_tokens=True)
        print(f"  Baseline: {len(gen_baseline)} tokens, {fwd_baseline} forwards")
        print(f"  Text: {text_baseline[:150]}...\n")

        results = {"baseline": {"tokens": len(gen_baseline), "forwards": fwd_baseline}}

        # === Experiment 2: Single-layer ablation ===
        print(f"{'='*90}")
        print(f"  EXPERIMENT 2: Per-layer cache ablation")
        print(f"{'='*90}")
        print(f"  {'Layer':<8s} {'Fwd':>6s} {'TokenLen':>9s} {'Match%':>8s} {'ExactMatch':>11s}")
        print(f"  {'-'*8} {'-'*6} {'-'*9} {'-'*8} {'-'*11}")

        exp2_data = {}
        moe_layer_indices = [idx for idx, _ in injector._moe_layers]

        for layer_idx in moe_layer_indices:
            dllm = make_dllm()
            out = run_generate_with_injector(
                dllm, long_ids, injector, TARGET_BLOCK_IDX,
                mode='single_layer', target_layer=layer_idx)
            gen = trim_after_eos(out[0][plen:], EOS_ID)
            fwd = dllm.num_forwards

            min_len = min(len(gen), len(gen_baseline))
            max_len = max(len(gen), len(gen_baseline))
            if min_len > 0:
                match = (gen[:min_len] == gen_baseline[:min_len]).sum().item()
                match_pct = match / max_len * 100
            else:
                match_pct = 0
            exact = torch.equal(gen, gen_baseline)

            print(f"  {layer_idx:<8d} {fwd:>6d} {len(gen):>9d} {match_pct:>7.1f}% {'YES' if exact else 'NO':>11s}")
            exp2_data[layer_idx] = {
                "forwards": fwd, "token_len": len(gen),
                "match_pct": match_pct, "exact_match": exact
            }

        results["exp2_single_layer"] = exp2_data

        # === Experiment 4: Periodic refresh ===
        print(f"\n{'='*90}")
        print(f"  EXPERIMENT 4: Periodic refresh (all layers cached)")
        print(f"{'='*90}")
        print(f"  {'Interval':<10s} {'Fwd':>6s} {'TokenLen':>9s} {'Match%':>8s} {'ExactMatch':>11s} {'CacheRate':>10s}")
        print(f"  {'-'*10} {'-'*6} {'-'*9} {'-'*8} {'-'*11} {'-'*10}")

        exp4_data = {}
        refresh_intervals = [0, 2, 3, 4, 5, 10]
        # interval=0 means cache every step (equivalent to v1 stable_cache)
        # interval=N means full compute every N-th step, cache on others

        for interval in refresh_intervals:
            dllm = make_dllm()
            if interval == 0:
                # All cached (v1 behavior)
                out = run_generate_with_injector(
                    dllm, long_ids, injector, TARGET_BLOCK_IDX,
                    refresh_interval=0, mode='all_layers')
                cache_rate = "100%"
            else:
                out = run_generate_with_injector(
                    dllm, long_ids, injector, TARGET_BLOCK_IDX,
                    refresh_interval=interval, mode='all_layers')
                cache_rate = f"{(1 - 1/interval)*100:.0f}%"

            gen = trim_after_eos(out[0][plen:], EOS_ID)
            fwd = dllm.num_forwards

            min_len = min(len(gen), len(gen_baseline))
            max_len = max(len(gen), len(gen_baseline))
            if min_len > 0:
                match = (gen[:min_len] == gen_baseline[:min_len]).sum().item()
                match_pct = match / max_len * 100
            else:
                match_pct = 0
            exact = torch.equal(gen, gen_baseline)

            label = f"N={interval}" if interval > 0 else "all_cache"
            print(f"  {label:<10s} {fwd:>6d} {len(gen):>9d} {match_pct:>7.1f}% {'YES' if exact else 'NO':>11s} {cache_rate:>10s}")
            exp4_data[interval] = {
                "forwards": fwd, "token_len": len(gen),
                "match_pct": match_pct, "exact_match": exact,
                "cache_rate": cache_rate,
            }

        results["exp4_periodic_refresh"] = exp4_data

        injector.uninstall()

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "moe_layer_ablation_refresh_results.json"

    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean(v) for v in obj]
        elif isinstance(obj, float):
            return obj if not (math.isnan(obj) or math.isinf(obj)) else str(obj)
        return obj

    out_path.write_text(json.dumps(clean(results), ensure_ascii=False, indent=2) + "\n")
    print(f"\nSaved: {out_path}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
