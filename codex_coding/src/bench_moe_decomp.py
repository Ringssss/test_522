#!/usr/bin/env python3
"""
Pre-experiment 1+3: Shared-only approximation quality + routed expert magnitude.

For stable positions (token unchanged since last iteration), measure:
  - ||shared(h)|| vs ||routed(h)|| magnitude ratio per layer
  - cosine sim between full_output and shared_only per layer
  - Compare with v1's "cached_full vs fresh_full" baseline

Uses cache path (BlockDiffusionLLM), batch=1, long prompt, temperature=0.0.
Collects data for one target generation block (~12 iterations).
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


# ============================================================
# Data collector: hooks on MoE blocks to capture shared/routed decomposition
# ============================================================
class MoEDecompCollector:
    """Captures shared_expert output, routed_expert output, and full output per layer per iteration."""

    def __init__(self):
        self.active = False
        self.recording = False
        self.current_iter = -1
        self.layer_counter = 0
        self.prev_block_tokens = None
        self.stable_mask = None  # [batch, block_len]

        # Per-iteration, per-layer data (stored for block positions only)
        # Each entry: dict with shared_norm, routed_norm, full_output, shared_only_output
        self.data = defaultdict(dict)  # data[iter][layer] = {...}
        self.mask_state = {}  # mask_state[iter] = [batch, block_len]
        self.stable_state = {}  # stable_state[iter] = [batch, block_len] or None

        # For v1 comparison: cached full MoE output from previous iteration
        self.prev_full_output = {}  # layer -> [batch, block_len, hidden]

    def reset(self):
        self.active = False
        self.recording = False
        self.current_iter = -1
        self.layer_counter = 0
        self.prev_block_tokens = None
        self.stable_mask = None
        self.data = defaultdict(dict)
        self.mask_state = {}
        self.stable_state = {}
        self.prev_full_output = {}

    def start_iteration(self, iter_idx, block_tokens, mask_positions):
        self.current_iter = iter_idx
        self.layer_counter = 0
        self.recording = True
        self.mask_state[iter_idx] = mask_positions.clone().cpu()

        if self.prev_block_tokens is not None:
            self.stable_mask = (block_tokens == self.prev_block_tokens).cpu()
        else:
            self.stable_mask = None
        self.stable_state[iter_idx] = self.stable_mask.clone() if self.stable_mask is not None else None
        self.prev_block_tokens = block_tokens.clone()

    def end_iteration(self):
        # Move current full outputs to prev for next iteration's v1 comparison
        for layer_idx in list(self.data[self.current_iter].keys()):
            d = self.data[self.current_iter][layer_idx]
            if "full_output" in d:
                self.prev_full_output[layer_idx] = d["full_output"]
        self.recording = False
        self.layer_counter = 0

    def record(self, shared_out, routed_out, full_out):
        """Called from MoE hook. All tensors are [batch, seq_len, hidden]."""
        if not self.active or not self.recording:
            return
        layer = self.layer_counter
        self.layer_counter += 1

        # Store block portion only (for cache path, seq_len IS block_len)
        s = shared_out.detach().cpu()
        r = routed_out.detach().cpu()
        f = full_out.detach().cpu()

        entry = {
            "shared_output": s,   # [batch, block_len, hidden]
            "routed_output": r,
            "full_output": f,
        }

        # v1 comparison: cached_full from previous iteration
        if layer in self.prev_full_output:
            entry["prev_full_output"] = self.prev_full_output[layer]

        self.data[self.current_iter][layer] = entry

    def get_num_iterations(self):
        return len(self.mask_state)


# ============================================================
# Hook installation
# ============================================================
def install_decomp_hooks(model, collector):
    """Replace MoE block forward to capture shared/routed decomposition."""
    hooks = []
    layers = model.model.layers

    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue

        moe_block = layer.mlp
        original_forward = moe_block.forward

        def make_hook(orig_fwd, moe_mod):
            def hooked_forward(hidden_states):
                bsz, seq_len, h = hidden_states.shape

                # Compute shared expert
                shared_out = moe_mod.shared_experts(hidden_states)

                # Compute routed experts
                hs_flat = hidden_states.view(-1, h)
                router_logits = moe_mod.gate.get_logits(hs_flat)
                routed_out = moe_mod.experts.forward_impl(
                    hidden_states=hs_flat, router_logits=router_logits)
                routed_out = routed_out.view(bsz, seq_len, h)

                # Full output
                full_out = routed_out + shared_out

                # Record
                collector.record(shared_out, routed_out, full_out)

                return full_out
            return hooked_forward

        moe_block.forward = make_hook(original_forward, moe_block)
        hooks.append((moe_block, original_forward))

    return hooks


def remove_hooks(hooks):
    for moe_block, original_forward in hooks:
        moe_block.forward = original_forward


# ============================================================
# Instrumented generate (reuse pattern from routing analysis)
# ============================================================
def run_instrumented_generate(dllm, input_ids, collector, target_block_idx):
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner

    orig_forward = BlockDiffusionIteration.forward
    orig_decode = BlockDiffusionRunner.decode
    current_block_idx = [0]
    iteration_in_block = [0]

    def patched_decode(self_runner, model, decoder, x, kv_cache, block, block_loc,
                       block_id, pos_ids, attn_mask, block_length=32, cross_block_attn_mask=None):
        gen_block_idx = current_block_idx[0]
        is_target = (gen_block_idx == target_block_idx)
        if is_target:
            collector.active = True
            iteration_in_block[0] = 0
            print(f"  [Collector] Activated for gen block {gen_block_idx}", flush=True)
        result = orig_decode(self_runner, model, decoder, x, kv_cache, block,
                             block_loc, block_id, pos_ids, attn_mask, block_length,
                             cross_block_attn_mask)
        if is_target:
            collector.active = False
            print(f"  [Collector] Done. {collector.get_num_iterations()} iterations.", flush=True)
        current_block_idx[0] += 1
        return result

    def patched_forward(self_iter, model, decoder, x, kv_cache, block, block_loc,
                        block_id, pos_ids, attn_mask, past_key_values, replace_position,
                        backend, is_cross_block=False, block_length=32):
        is_target = (current_block_idx[0] == target_block_idx)
        if is_target and collector.active and not is_cross_block:
            block_tokens = x.data[:, block_loc.start:block_loc.end]
            mask_positions = (block_tokens == decoder.mask_id)
            collector.start_iteration(iteration_in_block[0], block_tokens, mask_positions)

        result = orig_forward(self_iter, model, decoder, x, kv_cache, block,
                              block_loc, block_id, pos_ids, attn_mask,
                              past_key_values, replace_position, backend,
                              is_cross_block, block_length)

        if is_target and collector.active and not is_cross_block:
            collector.end_iteration()
            iteration_in_block[0] += 1
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


# ============================================================
# Analysis
# ============================================================
def analyze(collector):
    num_iters = collector.get_num_iterations()
    if num_iters == 0:
        print("No data collected!")
        return {}

    results = {}

    print(f"\n{'='*100}")
    print(f"  SHARED vs ROUTED DECOMPOSITION — {num_iters} iterations")
    print(f"{'='*100}")

    # --- Experiment 3: ||routed|| / ||shared|| ratio per layer ---
    print(f"\n--- Exp 3: Routed/Shared magnitude ratio (stable positions only) ---")
    print(f"  {'Layer':<8s} {'||shared||':>12s} {'||routed||':>12s} {'ratio':>10s} "
          f"{'#stable':>8s} {'#total':>8s}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*8} {'-'*8}")

    exp3_data = {}
    all_layers = sorted(collector.data[0].keys())
    for layer in all_layers:
        shared_norms = []
        routed_norms = []
        n_stable = 0
        n_total = 0

        for it in range(num_iters):
            if layer not in collector.data[it]:
                continue
            d = collector.data[it][layer]
            sm = collector.stable_state.get(it)
            if sm is None:
                continue  # first iteration, no stable positions

            # batch=1
            stable = sm[0]
            if not stable.any():
                continue

            s_out = d["shared_output"][0][stable].float()  # [n_stable, hidden]
            r_out = d["routed_output"][0][stable].float()

            shared_norms.append(s_out.norm(dim=-1).mean().item())
            routed_norms.append(r_out.norm(dim=-1).mean().item())
            n_stable += stable.sum().item()
            n_total += stable.numel()

        if shared_norms:
            avg_s = sum(shared_norms) / len(shared_norms)
            avg_r = sum(routed_norms) / len(routed_norms)
            ratio = avg_r / avg_s if avg_s > 0 else 0
            print(f"  {layer:<8d} {avg_s:>12.2f} {avg_r:>12.2f} {ratio:>10.4f} "
                  f"{n_stable:>8d} {n_total:>8d}")
            exp3_data[layer] = {"shared_norm": avg_s, "routed_norm": avg_r,
                                "ratio": ratio, "n_stable": n_stable}
    results["exp3_magnitude_ratio"] = exp3_data

    # --- Experiment 1: Approximation quality ---
    print(f"\n--- Exp 1: Approximation quality for stable positions ---")
    print(f"  {'Layer':<8s} {'shared_only':>14s} {'v1_cached':>12s} {'fresh_full':>12s} "
          f"{'improvement':>13s}")
    print(f"  {'':>8s} {'cos_sim':>14s} {'cos_sim':>12s} {'(baseline)':>12s} "
          f"{'shared/v1':>13s}")
    print(f"  {'-'*8} {'-'*14} {'-'*12} {'-'*12} {'-'*13}")

    exp1_data = {}
    for layer in all_layers:
        shared_only_sims = []
        v1_cached_sims = []

        for it in range(1, num_iters):  # skip iter 0 (no prev data)
            if layer not in collector.data[it]:
                continue
            d = collector.data[it][layer]
            sm = collector.stable_state.get(it)
            if sm is None or not sm[0].any():
                continue

            stable = sm[0]
            full = d["full_output"][0][stable].float()       # ground truth
            shared = d["shared_output"][0][stable].float()    # shared-only approx

            # cosine sim: shared_only vs full
            cos_shared = F.cosine_similarity(shared, full, dim=-1)
            shared_only_sims.extend(cos_shared.tolist())

            # v1: cached full from previous iteration vs current full
            if "prev_full_output" in d:
                cached = d["prev_full_output"][0][stable].float()
                cos_cached = F.cosine_similarity(cached, full, dim=-1)
                v1_cached_sims.extend(cos_cached.tolist())

        avg_shared = sum(shared_only_sims) / len(shared_only_sims) if shared_only_sims else 0
        avg_v1 = sum(v1_cached_sims) / len(v1_cached_sims) if v1_cached_sims else 0

        # "improvement" = how much closer shared-only is to 1.0 compared to v1
        improvement = "N/A"
        if v1_cached_sims and shared_only_sims:
            # error reduction: (1-avg_v1) vs (1-avg_shared)
            err_v1 = 1.0 - avg_v1
            err_shared = 1.0 - avg_shared
            if err_v1 > 0:
                improvement = f"{err_shared/err_v1:.2f}x err"

        print(f"  {layer:<8d} {avg_shared:>14.6f} {avg_v1:>12.6f} {'1.000000':>12s} {improvement:>13s}")
        exp1_data[layer] = {"shared_only_cos": avg_shared, "v1_cached_cos": avg_v1,
                            "n_pairs": len(shared_only_sims)}
    results["exp1_approx_quality"] = exp1_data

    # --- Per-iteration breakdown ---
    print(f"\n--- Per-iteration: stable count + avg shared-only cosine sim (avg over layers) ---")
    print(f"  {'Iter':<6s} {'#MASK':>6s} {'#Stable':>8s} {'AvgSharedCos':>14s} {'AvgV1Cos':>12s}")
    print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*14} {'-'*12}")

    per_iter_data = []
    for it in range(num_iters):
        ms = collector.mask_state[it][0]
        n_mask = ms.sum().item()
        sm = collector.stable_state.get(it)
        n_stable = sm[0].sum().item() if sm is not None else 0

        shared_cos_list = []
        v1_cos_list = []
        for layer in all_layers:
            if layer not in collector.data[it]:
                continue
            d = collector.data[it][layer]
            if sm is None or not sm[0].any():
                continue
            stable = sm[0]
            full = d["full_output"][0][stable].float()
            shared = d["shared_output"][0][stable].float()
            cos_s = F.cosine_similarity(shared, full, dim=-1).mean().item()
            shared_cos_list.append(cos_s)
            if "prev_full_output" in d:
                cached = d["prev_full_output"][0][stable].float()
                cos_v = F.cosine_similarity(cached, full, dim=-1).mean().item()
                v1_cos_list.append(cos_v)

        avg_sc = sum(shared_cos_list) / len(shared_cos_list) if shared_cos_list else 0
        avg_vc = sum(v1_cos_list) / len(v1_cos_list) if v1_cos_list else 0
        print(f"  {it:<6d} {int(n_mask):>6d} {int(n_stable):>8d} "
              f"{avg_sc:>14.6f} {avg_vc:>12.6f}" if n_stable > 0 else
              f"  {it:<6d} {int(n_mask):>6d} {int(n_stable):>8d} {'(no stable)':>14s} {'':>12s}")
        per_iter_data.append({"iter": it, "n_mask": int(n_mask), "n_stable": int(n_stable),
                              "shared_cos": avg_sc, "v1_cos": avg_vc})
    results["per_iteration"] = per_iter_data

    return results


# ============================================================
# Main
# ============================================================
def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
        print(f"Prompt: {long_ids.shape[1]} tokens")

        collector = MoEDecompCollector()
        hooks = install_decomp_hooks(model, collector)
        print(f"Installed {len(hooks)} MoE decomposition hooks")

        # Warmup
        print("Warmup ...", flush=True)
        dllm = BlockDiffusionLLM(
            model, ThresholdParallelDecoder(temperature=0.0, threshold=THRESHOLD, mask_id=MASK_ID, eos_id=EOS_ID),
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, lazy_cache_update=True, inplace_cache_update=True)
        with torch.inference_mode():
            dllm.generate(long_ids, gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)

        # Profiled run
        print("\nProfiled run ...", flush=True)
        collector.reset()
        dllm = BlockDiffusionLLM(
            model, ThresholdParallelDecoder(temperature=0.0, threshold=THRESHOLD, mask_id=MASK_ID, eos_id=EOS_ID),
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, lazy_cache_update=True, inplace_cache_update=True)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        out = run_instrumented_generate(dllm, long_ids, collector, TARGET_BLOCK_IDX)
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - t0
        print(f"Generate: {wall:.3f}s, {dllm.num_forwards} forwards")

        remove_hooks(hooks)

        results = analyze(collector)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "moe_decomp_analysis_results.json"

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
