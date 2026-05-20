#!/usr/bin/env python3
"""
v0.1.14.6c — Drift-Guarded Selective Recompute (End-to-End)

Adds shared_cos drift guard on top of token_margin + layer-range policy.
For each (layer, token), before reusing cached routed output, checks:
  drift = 1 - cos(shared_now, shared_prev_step)
  if drift > budget → force fresh (even if token_margin qualifies)

This catches sequences/tokens where contamination is accumulating,
while letting clean sequences enjoy full reuse speedup.

Compares:
  P0: Baseline (all fresh)
  P2: Token+Layer (margin>0.99, L4-14, no drift guard)
  P4: Token+Layer+Drift (margin>0.99, L4-14, shared_cos guard)

Prints full generation output for quality inspection.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")

MASK_ID = 156895
EOS_ID = 156892
BLOCK_LENGTH = 32
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
NUM_MOE_LAYERS = 19


# ============================================================
# Drift-Guarded Selective Recompute Controller
# ============================================================
class DriftGuardedController:
    """Selective recompute with shared_cos drift guard."""

    def __init__(self, policy: str, margin_threshold: float = 0.99,
                 reuse_layers: Optional[Set[int]] = None,
                 drift_budget: float = 0.015):
        self.policy = policy
        self.margin_threshold = margin_threshold
        self.reuse_layers = reuse_layers
        self.drift_budget = drift_budget
        self.enabled = (policy != "baseline")

        # Caches
        self.routed_cache: Dict[int, torch.Tensor] = {}
        self.shared_cache: Dict[int, torch.Tensor] = {}  # shared output from prev step
        self.qualifying_mask: Optional[torch.Tensor] = None

        # Stats
        self.total_positions = 0
        self.reused_positions = 0
        self.drift_blocked = 0  # reuse blocked by drift guard

    def reset_block(self):
        self.routed_cache.clear()
        self.shared_cache.clear()
        self.qualifying_mask = None

    def update_after_forward(self, logits: torch.Tensor):
        if not self.enabled:
            return
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            top2 = probs.topk(2, dim=-1).values
            margin = top2[:, :, 0] - top2[:, :, 1]
            self.qualifying_mask = (margin > self.margin_threshold)

    def cache_outputs(self, layer_idx: int, routed: torch.Tensor, shared: torch.Tensor):
        if self.enabled:
            self.routed_cache[layer_idx] = routed.detach().clone()
            self.shared_cache[layer_idx] = shared.detach().clone()

    def get_reuse_decision(self, layer_idx: int, shared_now: torch.Tensor, bsz: int):
        """Decide per-token reuse with drift guard.

        Returns:
            reuse_mask: [bsz, seq_len] bool — True where reuse is allowed
            cached_routed: tensor or None
        """
        if not self.enabled or self.qualifying_mask is None:
            return None, None
        if layer_idx not in self.routed_cache:
            return None, None
        if self.reuse_layers is not None and layer_idx not in self.reuse_layers:
            return None, None

        cached_routed = self.routed_cache[layer_idx]
        if cached_routed.shape[0] < bsz:
            return None, None

        # Start with token_margin qualifying mask
        mask = self.qualifying_mask[:bsz].clone()  # [bsz, seq_len]

        # Apply drift guard if we have shared cache from previous step
        if self.drift_budget is not None and layer_idx in self.shared_cache:
            shared_prev = self.shared_cache[layer_idx][:bsz]
            sn = shared_now[:bsz].float()
            sp = shared_prev.to(sn.device).float()

            # Per-token cosine similarity
            # sn, sp: [bsz, seq_len, hidden]
            cos_sim = F.cosine_similarity(sn, sp, dim=-1)  # [bsz, seq_len]
            drift = 1.0 - cos_sim
            drift_ok = (drift < self.drift_budget)

            # Count drift blocks
            was_qualifying = mask.sum().item()
            mask = mask & drift_ok
            now_qualifying = mask.sum().item()
            self.drift_blocked += int(was_qualifying - now_qualifying)

        return mask, cached_routed[:bsz]


# ============================================================
# Hook installer
# ============================================================
def install_hooks(model, controller: DriftGuardedController):
    hooks = []
    layers = model.model.layers
    moe_idx = 0

    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue

        moe_block = layer.mlp
        orig_forward = moe_block.forward
        current_moe_idx = moe_idx

        def make_hook(moe_mod, mi, ctrl):
            def hooked_forward(hidden_states):
                bsz, seq_len, h = hidden_states.shape
                hs_flat = hidden_states.view(-1, h)

                # Shared expert (always fresh)
                shared_res = moe_mod.shared_experts(hidden_states)

                # Gate + routed experts (always computed — needed for cache)
                router_logits = moe_mod.gate.get_logits(hs_flat)
                routed_y = moe_mod.experts.forward_impl(
                    hidden_states=hs_flat, router_logits=router_logits)
                routed_y = routed_y.view(bsz, seq_len, h)

                ctrl.total_positions += bsz * seq_len

                # Drift-guarded reuse decision
                reuse_mask, cached = ctrl.get_reuse_decision(mi, shared_res, bsz)
                if reuse_mask is not None and cached is not None:
                    n_reused = reuse_mask.sum().item()
                    if n_reused > 0:
                        routed_y = routed_y.clone()
                        c = cached.to(routed_y.device)
                        routed_y[reuse_mask] = c[reuse_mask]
                        ctrl.reused_positions += n_reused

                # Cache for next iteration
                ctrl.cache_outputs(mi, routed_y, shared_res)

                out = (routed_y + shared_res
                       if moe_mod.config.num_shared_experts is not None
                       else routed_y)
                return out
            return hooked_forward

        moe_block.forward = make_hook(moe_block, current_moe_idx, controller)
        hooks.append(('moe_forward', moe_block, orig_forward))
        moe_idx += 1

    return hooks


def remove_hooks(hooks):
    for kind, obj, orig in hooks:
        obj.forward = orig


# ============================================================
# Patched generation
# ============================================================
def generate_with_controller(dllm, input_ids, controller, gen_length=128):
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner

    orig_iter_forward = BlockDiffusionIteration.forward
    orig_runner_decode = BlockDiffusionRunner.decode

    def patched_decode(self_runner, model, decoder, x, kv_cache, block, block_loc,
                       block_id, pos_ids, attn_mask, block_length=32,
                       cross_block_attn_mask=None):
        controller.reset_block()
        return orig_runner_decode(self_runner, model, decoder, x, kv_cache, block,
                                  block_loc, block_id, pos_ids, attn_mask,
                                  block_length, cross_block_attn_mask)

    def patched_forward(self_iter, model, decoder, x, kv_cache, block, block_loc,
                        block_id, pos_ids, attn_mask, past_key_values,
                        replace_position, backend, is_cross_block=False,
                        block_length=32):
        output = orig_iter_forward(
            self_iter, model, decoder, x, kv_cache, block, block_loc,
            block_id, pos_ids, attn_mask, past_key_values, replace_position,
            backend, is_cross_block, block_length)
        if not is_cross_block:
            controller.update_after_forward(output.logits)
        return output

    BlockDiffusionIteration.forward = patched_forward
    BlockDiffusionRunner.decode = patched_decode

    try:
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            dllm.diff_iteration.iter_no = 0
            output = dllm.generate(input_ids.clone(), gen_length=gen_length,
                                    block_length=BLOCK_LENGTH)
    finally:
        BlockDiffusionIteration.forward = orig_iter_forward
        BlockDiffusionRunner.decode = orig_runner_decode

    return output


# ============================================================
# Main
# ============================================================
def main():
    import socket
    from contextlib import closing
    from transformers import AutoTokenizer, AutoConfig

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

    GEN_LENGTH = 128
    TEMPERATURE = 0.0  # deterministic for comparison
    THRESHOLD = 0.90

    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    print("=" * 80)
    print("v0.1.14.6c — Drift-Guarded Selective Recompute")
    print("=" * 80)

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

        decoder = ThresholdParallelDecoder(
            temperature=TEMPERATURE, threshold=THRESHOLD,
            mask_id=MASK_ID, eos_id=EOS_ID)

        dllm = BlockDiffusionLLM(
            model, decoder,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=1, expected_tpf=15,
            backend='vllm',
            lazy_cache_update=True, inplace_cache_update=True,
        )

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
        prompt_len = input_ids.shape[1]
        print(f"  Input shape: {input_ids.shape}")

        # ============================================================
        # Define policies
        # ============================================================
        POLICIES = [
            ("P0_baseline", {
                "policy": "baseline", "margin_threshold": 0.99,
                "reuse_layers": None, "drift_budget": None,
                "desc": "All fresh (baseline)",
            }),
            ("P2_no_guard", {
                "policy": "token_layer", "margin_threshold": 0.99,
                "reuse_layers": set(range(4, 15)), "drift_budget": None,
                "desc": "margin>0.99, L4-14, NO drift guard",
            }),
            ("P4_drift_loose", {
                "policy": "token_layer_drift", "margin_threshold": 0.99,
                "reuse_layers": set(range(4, 15)), "drift_budget": 0.03,
                "desc": "margin>0.99, L4-14, drift<0.03",
            }),
            ("P4_drift_mid", {
                "policy": "token_layer_drift", "margin_threshold": 0.99,
                "reuse_layers": set(range(4, 15)), "drift_budget": 0.015,
                "desc": "margin>0.99, L4-14, drift<0.015",
            }),
            ("P4_drift_tight", {
                "policy": "token_layer_drift", "margin_threshold": 0.99,
                "reuse_layers": set(range(4, 15)), "drift_budget": 0.005,
                "desc": "margin>0.99, L4-14, drift<0.005",
            }),
        ]

        results = {}
        all_outputs = {}

        for policy_name, pcfg_dict in POLICIES:
            print(f"\n{'='*60}")
            print(f"{policy_name}: {pcfg_dict['desc']}")
            print(f"{'='*60}")

            ctrl = DriftGuardedController(
                policy=pcfg_dict["policy"],
                margin_threshold=pcfg_dict["margin_threshold"],
                reuse_layers=pcfg_dict["reuse_layers"],
                drift_budget=pcfg_dict["drift_budget"],
            )

            hooks = install_hooks(model, ctrl)
            try:
                torch.cuda.synchronize()
                t0 = time.time()
                output = generate_with_controller(dllm, input_ids, ctrl, gen_length=GEN_LENGTH)
                torch.cuda.synchronize()
                elapsed = time.time() - t0

                fwd = dllm.diff_iteration.num_forwards
                gen_tokens = output[:, prompt_len:].cpu()
                total = ctrl.total_positions
                reused = ctrl.reused_positions
                drift_blk = ctrl.drift_blocked

                print(f"  Time: {elapsed:.3f}s  Forwards: {fwd}")
                print(f"  Reused: {reused}/{total} ({reused/max(total,1)*100:.1f}%)")
                if drift_blk > 0:
                    print(f"  Drift-blocked: {drift_blk} positions")

                results[policy_name] = {
                    "desc": pcfg_dict["desc"],
                    "elapsed": elapsed, "forwards": fwd,
                    "reused": reused, "total": total,
                    "drift_blocked": drift_blk,
                    "gen_tokens": gen_tokens,
                }
                all_outputs[policy_name] = gen_tokens
            finally:
                remove_hooks(hooks)

        # ============================================================
        # COMPARISON TABLE
        # ============================================================
        baseline_tokens = all_outputs["P0_baseline"]
        baseline_fwd = results["P0_baseline"]["forwards"]
        baseline_time = results["P0_baseline"]["elapsed"]

        print(f"\n{'='*80}")
        print(f"COMPARISON TABLE (temperature=0, deterministic)")
        print(f"{'='*80}")
        print(f"  {'Policy':<20s} {'Fwd':>5s} {'Time':>7s} {'Speed':>6s} "
              f"{'Reuse%':>7s} {'DriftBlk':>8s} {'TokMatch':>8s}")
        print(f"  {'-'*72}")

        for pn, r in results.items():
            gt = r["gen_tokens"]
            ml = min(baseline_tokens.shape[1], gt.shape[1])
            bt = baseline_tokens[:, :ml]
            gtt = gt[:, :ml]
            non_pad = (bt != 0) & (bt != EOS_ID)
            tok_match = ((bt == gtt) & non_pad).sum().item() / max(non_pad.sum().item(), 1)
            spd = baseline_time / max(r["elapsed"], 1e-6)

            print(f"  {pn:<20s} {r['forwards']:>5d} {r['elapsed']:>6.3f}s {spd:>5.2f}x "
                  f"{r['reused']/max(r['total'],1)*100:>6.1f}% "
                  f"{r['drift_blocked']:>8d} "
                  f"{tok_match*100:>7.1f}%")

        # ============================================================
        # FULL OUTPUT COMPARISON — side by side
        # ============================================================
        print(f"\n{'='*80}")
        print(f"FULL GENERATION OUTPUT COMPARISON")
        print(f"{'='*80}")

        policy_names_to_show = ["P0_baseline", "P2_no_guard", "P4_drift_mid"]

        for batch_idx in range(BATCH_SIZE):
            print(f"\n{'─'*80}")
            print(f"  BATCH {batch_idx} — Prompt: {PROMPTS[batch_idx][:80]}...")
            print(f"{'─'*80}")

            for pn in policy_names_to_show:
                gt = all_outputs[pn][batch_idx]
                # Remove padding and EOS
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                # Truncate for display
                text_display = text[:500]

                # Token match vs baseline
                bl = baseline_tokens[batch_idx]
                ml = min(len(bl), len(gt))
                match_rate = (bl[:ml] == gt[:ml]).float().mean().item() * 100

                label = f"{pn} ({match_rate:.0f}% match)"
                print(f"\n  [{label}]:")
                print(f"  {text_display}")
                if len(text) > 500:
                    print(f"  ... ({len(text)} chars total)")

        # ============================================================
        # Also test temperature=0.7 for the best drift-guarded policy
        # ============================================================
        print(f"\n{'='*80}")
        print(f"STOCHASTIC TEST (temperature=0.7) — 3 runs each")
        print(f"{'='*80}")

        decoder.temperature = 0.7
        stoch_policies = ["P0_baseline", "P2_no_guard", "P4_drift_mid"]
        stoch_results = defaultdict(list)

        for run_idx in range(3):
            for pn in stoch_policies:
                pcfg_dict = dict(POLICIES)
                for name, cfg in POLICIES:
                    if name == pn:
                        pcfg_dict = cfg
                        break

                ctrl = DriftGuardedController(
                    policy=pcfg_dict["policy"],
                    margin_threshold=pcfg_dict["margin_threshold"],
                    reuse_layers=pcfg_dict["reuse_layers"],
                    drift_budget=pcfg_dict["drift_budget"],
                )
                hooks = install_hooks(model, ctrl)
                try:
                    torch.cuda.synchronize()
                    t0 = time.time()
                    output = generate_with_controller(dllm, input_ids, ctrl, gen_length=GEN_LENGTH)
                    torch.cuda.synchronize()
                    elapsed = time.time() - t0

                    stoch_results[pn].append({
                        "elapsed": elapsed,
                        "forwards": dllm.diff_iteration.num_forwards,
                        "reuse_ratio": ctrl.reused_positions / max(ctrl.total_positions, 1),
                        "gen_tokens": output[:, prompt_len:].cpu(),
                    })
                finally:
                    remove_hooks(hooks)
            print(f"  Run {run_idx+1}/3 done", flush=True)

        print(f"\n  {'Policy':<20s} {'AvgTime':>7s} {'AvgFwd':>6s} {'Reuse%':>7s} "
              f"{'Self-cons':>9s} {'vs-BL':>7s}")

        bl_runs = stoch_results["P0_baseline"]
        for pn in stoch_policies:
            runs = stoch_results[pn]
            avg_t = sum(r["elapsed"] for r in runs) / len(runs)
            avg_f = sum(r["forwards"] for r in runs) / len(runs)
            avg_r = sum(r["reuse_ratio"] for r in runs) / len(runs)

            # Self-consistency
            self_cons = []
            for i in range(len(runs)):
                for j in range(i+1, len(runs)):
                    ml = min(runs[i]["gen_tokens"].shape[1], runs[j]["gen_tokens"].shape[1])
                    m = (runs[i]["gen_tokens"][:,:ml] == runs[j]["gen_tokens"][:,:ml]).float().mean().item()
                    self_cons.append(m)
            avg_sc = sum(self_cons)/len(self_cons) if self_cons else 1.0

            # Cross vs baseline
            cross = []
            for pr in runs:
                for br in bl_runs:
                    ml = min(pr["gen_tokens"].shape[1], br["gen_tokens"].shape[1])
                    m = (pr["gen_tokens"][:,:ml] == br["gen_tokens"][:,:ml]).float().mean().item()
                    cross.append(m)
            avg_cr = sum(cross)/len(cross) if cross else 1.0

            spd_str = f"{sum(r['elapsed'] for r in bl_runs)/len(bl_runs)/max(avg_t,1e-6):.2f}x"
            print(f"  {pn:<20s} {avg_t:>6.3f}s {avg_f:>6.0f} {avg_r*100:>6.1f}% "
                  f"{avg_sc*100:>8.1f}% {avg_cr*100:>6.1f}%")

        # Show one stochastic sample for quality check
        print(f"\n  --- Sample stochastic output (batch 0, P4_drift_mid, run 0) ---")
        sample_tok = stoch_results["P4_drift_mid"][0]["gen_tokens"][0]
        sample_valid = sample_tok[(sample_tok != 0) & (sample_tok != EOS_ID) & (sample_tok != MASK_ID)]
        print(f"  {tokenizer.decode(sample_valid, skip_special_tokens=True)[:400]}")

        print(f"\nDone.")

        # Save
        save_data = {}
        for pn, r in results.items():
            save_data[pn] = {k: v for k, v in r.items() if k != "gen_tokens"}
        save_path = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction" / "e2e_drift_guarded_results.json"
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
