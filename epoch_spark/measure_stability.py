#!/usr/bin/env python3
"""
Phase 1: Neuron Activation Stability Measurement for Epoch-Spark.

Hooks each MoE layer during real diffusion generation to measure within-block
stability of expert routing and neuron-level activations.

Metrics per layer per block:
  - Expert set Jaccard across iterations
  - Neuron tile activation Jaccard within experts across iterations
  - Activation pattern cosine similarity
  - Freeze-at-iter-0 hit rate

Usage:
    cd ~/epoch_spark && /home/zhujianian/miniconda3/envs/crossstage/bin/python measure_stability.py
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    MASK_ID, BLOCK_LENGTH, NUM_EXPERTS, TOP_K,
    MOE_INTERMEDIATE_SIZE, NEURON_TILE_SIZE, TILES_PER_EXPERT,
    PROMPTS,
)
from utils import load_model_and_tokenizer, gpu_mem_mb, Timer


class ActivationCollector:
    """Hooks MoE layers post-forward to read routing + compute neuron activations."""

    def __init__(self, tile_size=NEURON_TILE_SIZE, activation_threshold=0.1):
        self.tile_size = tile_size
        self.activation_threshold = activation_threshold
        self.current_block_id = -1
        self.current_iter = -1
        # blocks[layer_id][block_id][iter_id] = record dict
        self.blocks = defaultdict(lambda: defaultdict(dict))
        self._hooks = []

    def on_block_start(self, block_id):
        self.current_block_id = block_id

    def on_iter_start(self, iter_id):
        self.current_iter = iter_id

    def install_hooks(self, model):
        layer_id = 0
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeSparseMoeBlock":
                layer_id += 1
                handle = mod.register_forward_hook(self._make_hook(mod, layer_id))
                self._hooks.append(handle)
        print(f"[collector] Installed hooks on {layer_id} MoE layers")

    def _make_hook(self, moe_block, layer_id):
        collector = self

        def hook_fn(module, input, output):
            if collector.current_block_id < 0 or collector.current_iter < 0:
                return

            hidden_states = input[0]  # [B, S, H]
            bsz, seq_len, h = hidden_states.shape
            hs_flat = hidden_states.view(-1, h)

            experts = module.experts
            topk_ids = getattr(experts, "_last_topk_ids", None)
            topk_weights = getattr(experts, "_last_topk_weights", None)
            if topk_ids is None:
                return

            topk_ids_cpu = topk_ids.detach().cpu()
            active_experts = set(topk_ids_cpu.flatten().tolist())

            # w13_weight: [E, 2*I, H] — gate_proj is [:I], up_proj is [I:]
            w13 = experts.w13_weight  # [256, 1024, 2048]
            I = MOE_INTERMEDIATE_SIZE

            neuron_activations = {}
            active_tiles = {}

            # Sample up to 20 experts for activation measurement
            sample_experts = sorted(active_experts)[:20]

            with torch.no_grad():
                for eid in sample_experts:
                    token_mask = (topk_ids_cpu == eid).any(dim=-1)
                    if token_mask.sum() == 0:
                        continue
                    idx = token_mask.nonzero(as_tuple=True)[0]
                    if len(idx) > 32:
                        idx = idx[:32]

                    hs_e = hs_flat[idx].float()  # [n, H]
                    w = w13[eid].float()  # [2*I, H]
                    gate_proj = w[:I]  # [I, H]
                    up_proj = w[I:]    # [I, H]

                    gate_out = torch.nn.functional.silu(hs_e @ gate_proj.T)
                    up_out = hs_e @ up_proj.T
                    act = gate_out * up_out  # [n, I]

                    neuron_mag = act.abs().mean(dim=0).cpu()  # [I]
                    neuron_activations[eid] = neuron_mag

                    n_tiles = I // self.tile_size
                    tile_mags = neuron_mag.reshape(n_tiles, self.tile_size).mean(dim=1)
                    active_t = set(
                        (tile_mags > self.activation_threshold).nonzero(as_tuple=True)[0].tolist()
                    )
                    active_tiles[eid] = active_t

            collector.blocks[layer_id][collector.current_block_id][collector.current_iter] = {
                "active_experts": active_experts,
                "topk_ids": topk_ids_cpu,
                "neuron_activations": neuron_activations,
                "active_tiles": active_tiles,
            }

        return hook_fn

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════

def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 1.0


def compute_metrics(collector):
    results = {}
    for layer_id in sorted(collector.blocks.keys()):
        layer_data = collector.blocks[layer_id]
        lr = {
            "expert_jaccard_matrix": [],
            "tile_jaccard": [],
            "activation_cosine": [],
            "freeze_hit_rate": [],
            "n_unique_experts": [],
        }

        for block_id in sorted(layer_data.keys()):
            block = layer_data[block_id]
            iters = sorted(block.keys())
            n = len(iters)
            if n < 2:
                continue

            # Expert Jaccard
            esets = [block[i]["active_experts"] for i in iters]
            jm = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    jm[i, j] = jaccard(esets[i], esets[j])
            lr["expert_jaccard_matrix"].append(jm.tolist())
            lr["n_unique_experts"].append([len(s) for s in esets])

            # Tile Jaccard
            tile_jacs = []
            for i in range(n):
                for j in range(i + 1, n):
                    common = set(block[iters[i]]["active_tiles"].keys()) & set(block[iters[j]]["active_tiles"].keys())
                    for eid in common:
                        tj = jaccard(block[iters[i]]["active_tiles"][eid],
                                     block[iters[j]]["active_tiles"][eid])
                        tile_jacs.append(tj)
            lr["tile_jaccard"].append(float(np.mean(tile_jacs)) if tile_jacs else -1.0)

            # Cosine similarity (consecutive iterations)
            cosines = []
            for idx in range(len(iters) - 1):
                na = block[iters[idx]]["neuron_activations"]
                nb = block[iters[idx + 1]]["neuron_activations"]
                for eid in set(na.keys()) & set(nb.keys()):
                    a, b = na[eid].float(), nb[eid].float()
                    if a.norm() > 0 and b.norm() > 0:
                        cosines.append(
                            torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
                        )
            lr["activation_cosine"].append(float(np.mean(cosines)) if cosines else -1.0)

            # Freeze-at-iter-0 hit rate
            if len(iters) > 1:
                iter0 = block[iters[0]]
                hr_e, hr_t = [], []
                for it in iters[1:]:
                    d = block[it]
                    if d["active_experts"]:
                        hr_e.append(len(iter0["active_experts"] & d["active_experts"]) / len(d["active_experts"]))
                    for eid in d["active_tiles"]:
                        if eid in iter0["active_tiles"] and d["active_tiles"][eid]:
                            hr_t.append(jaccard(iter0["active_tiles"][eid], d["active_tiles"][eid]))
                lr["freeze_hit_rate"].append({
                    "expert": float(np.mean(hr_e)) if hr_e else -1.0,
                    "tile": float(np.mean(hr_t)) if hr_t else -1.0,
                })

        results[f"layer_{layer_id}"] = lr
    return results


def print_summary(results):
    print("\n" + "=" * 95)
    print(f"{'Layer':>8} | {'Expert Jac':>11} | {'Tile Jac':>10} | {'Act Cosine':>11} | "
          f"{'FreezeHR(E)':>12} | {'FreezeHR(T)':>12} | {'#Experts':>8}")
    print("-" * 95)

    for key in sorted(results.keys(), key=lambda k: int(k.split("_")[1])):
        d = results[key]
        lid = int(key.split("_")[1])

        ej = [m[i][j] for mat in d["expert_jaccard_matrix"]
              for m in [np.array(mat)] for i in range(m.shape[0]) for j in range(i+1, m.shape[1])]
        tj = [v for v in d["tile_jaccard"] if v >= 0]
        ac = [v for v in d["activation_cosine"] if v >= 0]
        fhe = [v["expert"] for v in d["freeze_hit_rate"] if v["expert"] >= 0]
        fht = [v["tile"] for v in d["freeze_hit_rate"] if v["tile"] >= 0]
        ne = [np.mean(v) for v in d["n_unique_experts"]] if d["n_unique_experts"] else []

        print(f"  L{lid:02d}   | {np.mean(ej) if ej else -1:11.4f} | {np.mean(tj) if tj else -1:10.4f} | "
              f"{np.mean(ac) if ac else -1:11.4f} | {np.mean(fhe) if fhe else -1:12.4f} | "
              f"{np.mean(fht) if fht else -1:12.4f} | {np.mean(ne) if ne else -1:8.1f}")
    print("=" * 95)


# ════════════════════════════════════════════════════════════════
# Diffusion generation loop
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_generation(model, tokenizer, collector, prompt, gen_length=128,
                   steps_per_block=10, device="cuda:0"):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    total_gen = gen_length
    n_blocks = (total_gen + BLOCK_LENGTH - 1) // BLOCK_LENGTH

    mask_tokens = torch.full((1, total_gen), MASK_ID, dtype=torch.long, device=device)
    x = torch.cat([input_ids, mask_tokens], dim=1)
    seq_len = x.shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    for block_id in range(n_blocks):
        block_start = prompt_len + block_id * BLOCK_LENGTH
        block_end = min(block_start + BLOCK_LENGTH, seq_len)

        collector.on_block_start(block_id)

        block_tokens = x[0, block_start:block_end]
        n_mask = (block_tokens == MASK_ID).sum().item()
        if n_mask == 0:
            continue

        remaining = n_mask
        for step_idx in range(steps_per_block):
            if remaining <= 0:
                break
            n_transfer = max(1, remaining // (steps_per_block - step_idx))

            collector.on_iter_start(step_idx)

            outputs = model(input_ids=x, position_ids=position_ids,
                            use_cache=False, return_dict=True)
            logits = outputs.logits  # [1, S, V]

            block_logits = logits[0, block_start:block_end]
            block_x = x[0, block_start:block_end]
            live = (block_x == MASK_ID)
            if not live.any():
                break

            probs = torch.softmax(block_logits.float(), dim=-1)
            pred = block_logits.argmax(dim=-1)
            conf = probs.max(dim=-1).values
            conf[~live] = -1.0

            n_dec = min(n_transfer, live.sum().item())
            if n_dec > 0:
                _, top_idx = conf.topk(n_dec)
                x[0, block_start + top_idx] = pred[top_idx]
                remaining -= n_dec

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--steps-per-block", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=str, default="stability_results.json")
    parser.add_argument("--tile-size", type=int, default=NEURON_TILE_SIZE)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    print(f"[Phase1] Loading model on {device}...")
    model, tokenizer, config = load_model_and_tokenizer(device=device)
    print(f"[Phase1] GPU mem: {gpu_mem_mb(args.gpu):.0f} MB")

    collector = ActivationCollector(tile_size=args.tile_size)
    collector.install_hooks(model)

    prompts = PROMPTS[:args.num_prompts]
    print(f"[Phase1] Running {len(prompts)} prompts, gen_length={args.gen_length}")

    for i, prompt in enumerate(prompts):
        print(f"\n[Prompt {i+1}/{len(prompts)}] {prompt[:80]}...")
        with Timer(f"Prompt {i+1}"):
            x = run_generation(model, tokenizer, collector, prompt,
                               gen_length=args.gen_length,
                               steps_per_block=args.steps_per_block,
                               device=device)
        decoded_text = tokenizer.decode(x[0], skip_special_tokens=True)
        print(f"  Output: {decoded_text[:200]}")

    collector.remove_hooks()

    print("\n[Phase1] Computing metrics...")
    results = compute_metrics(collector)
    print_summary(results)

    # JSON serialize
    def serialize(obj):
        if isinstance(obj, (set, frozenset)):
            return sorted(list(obj))
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, dict):
            return {str(k): serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serialize(v) for v in obj]
        return obj

    with open(args.output, "w") as f:
        json.dump(serialize(results), f, indent=2)
    print(f"[Phase1] Results saved to {args.output}")


if __name__ == "__main__":
    main()
