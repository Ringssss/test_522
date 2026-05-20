#!/usr/bin/env python3
"""
v0.1.15.2 — Monkey-Patch vs True-Skip Equivalence Test

Runs both implementation approaches under IDENTICAL conditions and compares:
  - Forward counts
  - Token-level outputs (exact match)
  - Per-layer MoE output cosine similarity (micro-level diagnosis)

If results differ, the per-layer logging identifies the first divergence point.

Conditions:
  batch=8, gen_length=128, block_length=32, temp=0, threshold=0.90
  Policy: P2 margin>0.99, L4-14, no drift guard
"""

from __future__ import annotations
import os, sys, socket, json
from collections import defaultdict
from pathlib import Path

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
]

MARGIN_THRESHOLD = 0.99
REUSE_LAYERS = set(range(4, 15))


# ============================================================
# Shared: qualifying mask computation
# ============================================================
class ReuseMaskTracker:
    """Tracks which tokens qualify for reuse (shared by both approaches)."""
    def __init__(self):
        self.qualifying_mask = None  # [bsz, seq_len] bool

    def update(self, logits):
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            top2 = probs.topk(2, dim=-1).values
            self.qualifying_mask = (top2[:, :, 0] - top2[:, :, 1]) > MARGIN_THRESHOLD

    def reset(self):
        self.qualifying_mask = None


# ============================================================
# Approach A: Monkey-Patch (compute all, then replace)
# ============================================================
def install_monkey_patch_hooks(model, tracker):
    hooks = []
    caches = {}
    stats = {"total": 0, "reused": 0}
    # Per-layer output log for comparison
    layer_outputs = defaultdict(list)  # mi -> list of [bsz, seq_len, h] per step

    mi = 0
    for layer in model.model.layers:
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue
        moe = layer.mlp
        orig = moe.forward
        idx = mi

        def make_hook(moe_mod, layer_idx, cch, st, tr, lo):
            def hooked_forward(hidden_states):
                bsz, seq_len, h = hidden_states.shape
                hs_flat = hidden_states.view(-1, h)
                n = bsz * seq_len

                # Always compute fresh
                shared_res = moe_mod.shared_experts(hidden_states)
                router_logits = moe_mod.gate.get_logits(hs_flat)
                routed_y = moe_mod.experts.forward_impl(
                    hidden_states=hs_flat, router_logits=router_logits)
                routed_y = routed_y.view(bsz, seq_len, h)

                st["total"] += n

                # Replace stable positions with cache
                if (tr.qualifying_mask is not None
                        and layer_idx in REUSE_LAYERS
                        and layer_idx in cch):
                    cached = cch[layer_idx]
                    if cached.shape[0] >= bsz and cached.shape[1] == seq_len:
                        mask = tr.qualifying_mask[:bsz]
                        n_reuse = mask.sum().item()
                        if n_reuse > 0:
                            routed_y = routed_y.clone()
                            routed_y[mask] = cached[:bsz][mask].to(routed_y.device)
                            st["reused"] += n_reuse

                # Cache
                cch[layer_idx] = routed_y.detach().clone()

                # Log output for comparison
                out = routed_y + shared_res if moe_mod.config.num_shared_experts is not None else routed_y
                lo[layer_idx].append(out.detach().clone())
                return out
            return hooked_forward

        moe.forward = make_hook(moe, idx, caches, stats, tracker, layer_outputs)
        hooks.append((moe, orig))
        mi += 1

    return hooks, stats, caches, layer_outputs


# ============================================================
# Approach B: True-Skip (only compute fresh tokens)
# ============================================================
def install_true_skip_hooks(model, tracker):
    hooks = []
    caches = {}
    stats = {"total": 0, "reused": 0}
    layer_outputs = defaultdict(list)

    mi = 0
    for layer in model.model.layers:
        if not hasattr(layer, 'mlp') or not hasattr(layer.mlp, 'gate'):
            continue
        moe = layer.mlp
        orig = moe.forward
        idx = mi

        def make_hook(moe_mod, layer_idx, cch, st, tr, lo):
            def hooked_forward(hidden_states):
                bsz, seq_len, h = hidden_states.shape
                hs_flat = hidden_states.view(-1, h)
                n = bsz * seq_len

                # Shared: always fresh
                shared_res = moe_mod.shared_experts(hidden_states)

                st["total"] += n

                # Determine reuse mask
                reuse_mask = None
                if (tr.qualifying_mask is not None
                        and layer_idx in REUSE_LAYERS
                        and layer_idx in cch):
                    cached = cch[layer_idx]
                    if cached.shape[0] >= bsz and cached.shape[1] == seq_len:
                        reuse_mask = tr.qualifying_mask[:bsz]

                if reuse_mask is not None and reuse_mask.any():
                    n_reuse = reuse_mask.sum().item()
                    n_fresh = n - n_reuse
                    fresh_mask_flat = (~reuse_mask).view(-1)

                    if n_fresh == 0:
                        # All reuse
                        routed_y = cch[layer_idx][:bsz].clone()
                        st["reused"] += n
                    else:
                        # True skip: only compute fresh tokens
                        fresh_indices = fresh_mask_flat.nonzero(as_tuple=True)[0]
                        fresh_hs = hs_flat[fresh_indices]

                        fresh_logits = moe_mod.gate.get_logits(fresh_hs)
                        fresh_routed = moe_mod.experts.forward_impl(
                            hidden_states=fresh_hs,
                            router_logits=fresh_logits)

                        # Assemble
                        cached_flat = cch[layer_idx][:bsz].view(-1, h)
                        routed_y = cached_flat.clone()
                        routed_y[fresh_indices] = fresh_routed
                        routed_y = routed_y.view(bsz, seq_len, h)

                        st["reused"] += n_reuse
                else:
                    # No reuse — standard path
                    router_logits = moe_mod.gate.get_logits(hs_flat)
                    routed_y = moe_mod.experts.forward_impl(
                        hidden_states=hs_flat,
                        router_logits=router_logits)
                    routed_y = routed_y.view(bsz, seq_len, h)

                # Cache
                cch[layer_idx] = routed_y.detach().clone()

                out = routed_y + shared_res if moe_mod.config.num_shared_experts is not None else routed_y
                lo[layer_idx].append(out.detach().clone())
                return out
            return hooked_forward

        moe.forward = make_hook(moe, idx, caches, stats, tracker, layer_outputs)
        hooks.append((moe, orig))
        mi += 1

    return hooks, stats, caches, layer_outputs


def remove_hooks(hooks):
    for moe, orig in hooks:
        moe.forward = orig


def gen_with_tracker(dllm, input_ids, tracker, approach_hooks, gl=128):
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionRunner
    oif = BlockDiffusionIteration.forward
    ord_ = BlockDiffusionRunner.decode

    def pd(self_runner, model, decoder, x, kv_cache, block, block_loc,
           block_id, pos_ids, attn_mask, block_length=32, cross_block_attn_mask=None):
        tracker.reset()
        # Clear caches in the approach hooks
        approach_hooks[2].clear()  # caches dict
        return ord_(self_runner, model, decoder, x, kv_cache, block, block_loc,
                    block_id, pos_ids, attn_mask, block_length, cross_block_attn_mask)

    def pf(self_iter, model, decoder, x, kv_cache, block, block_loc,
           block_id, pos_ids, attn_mask, past_key_values,
           replace_position, backend, is_cross_block=False, block_length=32):
        out = oif(self_iter, model, decoder, x, kv_cache, block, block_loc,
                  block_id, pos_ids, attn_mask, past_key_values,
                  replace_position, backend, is_cross_block, block_length)
        if not is_cross_block:
            tracker.update(out.logits)
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
    print("v0.1.15.2 — Monkey-Patch vs True-Skip Equivalence Test")
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

        BATCH_SIZE = 8
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
        print("Warmup done.\n")

        # ---- Run Approach A: Monkey-Patch ----
        print("=" * 60)
        print("APPROACH A: Monkey-Patch (compute all, replace stable)")
        print("=" * 60)
        tracker_a = ReuseMaskTracker()
        hooks_a, stats_a, caches_a, logs_a = install_monkey_patch_hooks(model, tracker_a)
        try:
            out_a = gen_with_tracker(dllm, input_ids, tracker_a,
                                     (hooks_a, stats_a, caches_a, logs_a))
            fwd_a = dllm.diff_iteration.num_forwards
            tokens_a = out_a[:, prompt_len:].cpu()
        finally:
            remove_hooks(hooks_a)

        reuse_pct_a = stats_a["reused"] / max(stats_a["total"], 1) * 100
        print(f"  Forwards: {fwd_a}")
        print(f"  Reused: {stats_a['reused']}/{stats_a['total']} ({reuse_pct_a:.1f}%)")
        print(f"  Layer output log entries: {sum(len(v) for v in logs_a.values())}")

        # ---- Run Approach B: True-Skip ----
        print(f"\n{'=' * 60}")
        print("APPROACH B: True-Skip (only compute fresh tokens)")
        print("=" * 60)
        tracker_b = ReuseMaskTracker()
        hooks_b, stats_b, caches_b, logs_b = install_true_skip_hooks(model, tracker_b)
        try:
            out_b = gen_with_tracker(dllm, input_ids, tracker_b,
                                     (hooks_b, stats_b, caches_b, logs_b))
            fwd_b = dllm.diff_iteration.num_forwards
            tokens_b = out_b[:, prompt_len:].cpu()
        finally:
            remove_hooks(hooks_b)

        reuse_pct_b = stats_b["reused"] / max(stats_b["total"], 1) * 100
        print(f"  Forwards: {fwd_b}")
        print(f"  Reused: {stats_b['reused']}/{stats_b['total']} ({reuse_pct_b:.1f}%)")
        print(f"  Layer output log entries: {sum(len(v) for v in logs_b.values())}")

        # ---- Macro Comparison ----
        print(f"\n{'=' * 80}")
        print("MACRO COMPARISON")
        print("=" * 80)
        print(f"  Forward count:  A={fwd_a}  B={fwd_b}  {'MATCH' if fwd_a == fwd_b else 'DIFFER !!!'}")
        print(f"  Reuse rate:     A={reuse_pct_a:.1f}%  B={reuse_pct_b:.1f}%")

        # Token comparison
        min_len = min(tokens_a.shape[1], tokens_b.shape[1])
        ta = tokens_a[:, :min_len]
        tb = tokens_b[:, :min_len]
        non_pad = (ta != 0) & (ta != EOS_ID)
        exact_match = ((ta == tb) & non_pad).sum().item()
        total_valid = non_pad.sum().item()
        match_pct = exact_match / max(total_valid, 1) * 100
        print(f"  Token match:    {exact_match}/{total_valid} ({match_pct:.1f}%)")

        if fwd_a == fwd_b and match_pct == 100.0:
            print(f"\n  >>> VERDICT: CASE A — Fully equivalent. <<<")
        elif fwd_a == fwd_b and match_pct > 95.0:
            print(f"\n  >>> VERDICT: CASE E — Same forward count, minor token diffs. <<<")
        elif fwd_a == fwd_b:
            print(f"\n  >>> VERDICT: CASE E — Same forward count, significant token diffs. <<<")
        elif fwd_b > fwd_a:
            print(f"\n  >>> VERDICT: CASE C — True-skip has MORE forwards ({fwd_b} > {fwd_a}). <<<")
            print(f"      Monkey-patch may over-estimate reuse quality.")
        else:
            print(f"\n  >>> VERDICT: CASE D — True-skip has FEWER forwards ({fwd_b} < {fwd_a}). <<<")

        # ---- Micro Comparison: Per-layer output cosine similarity ----
        print(f"\n{'=' * 80}")
        print("MICRO COMPARISON: Per-Layer MoE Output Cosine Similarity")
        print("=" * 80)

        # Compare the per-layer logs step by step
        n_layers = 19
        n_steps_a = len(logs_a.get(0, []))
        n_steps_b = len(logs_b.get(0, []))
        n_compare = min(n_steps_a, n_steps_b)
        print(f"  Steps logged: A={n_steps_a}, B={n_steps_b}, comparing first {n_compare}")

        first_diverge = None
        min_cos_overall = 1.0

        if n_compare > 0:
            # Sample a few layers and steps to keep output manageable
            sample_layers = [0, 4, 9, 14, 18]
            sample_steps = list(range(0, n_compare, max(1, n_compare // 10)))[:10]

            print(f"\n  {'Step':>4s}  ", end="")
            for li in sample_layers:
                print(f"{'L'+str(li):>8s}", end="  ")
            print()
            print(f"  {'-'*4}  " + "  ".join(['-'*8]*len(sample_layers)))

            for si in sample_steps:
                print(f"  {si:>4d}  ", end="")
                for li in sample_layers:
                    if li < len(logs_a) and si < len(logs_a[li]) and si < len(logs_b[li]):
                        oa = logs_a[li][si].float()
                        ob = logs_b[li][si].float()
                        # Flatten to [N, H] and compute cosine
                        oa_flat = oa.view(-1, oa.shape[-1])
                        ob_flat = ob.view(-1, ob.shape[-1])
                        cos = F.cosine_similarity(oa_flat, ob_flat, dim=-1).mean().item()
                        min_cos_overall = min(min_cos_overall, cos)
                        if cos < 0.999 and first_diverge is None:
                            first_diverge = (si, li, cos)
                        marker = " " if cos > 0.9999 else ("*" if cos > 0.999 else "!!")
                        print(f"{cos:.6f}{marker}", end="  ")
                    else:
                        print(f"{'N/A':>8s}", end="  ")
                print()

            print(f"\n  Min cosine overall: {min_cos_overall:.6f}")
            if first_diverge:
                print(f"  First significant divergence (<0.999): step={first_diverge[0]}, "
                      f"layer={first_diverge[1]}, cos={first_diverge[2]:.6f}")
            else:
                print(f"  No significant divergence found (all cos > 0.999)")

        # ---- Output text comparison ----
        print(f"\n{'=' * 80}")
        print("OUTPUT TEXT COMPARISON (first 2 batches)")
        print("=" * 80)
        for bi in range(min(BATCH_SIZE, 2)):
            print(f"\n  BATCH {bi}:")
            ga = tokens_a[bi]
            gb = tokens_b[bi]
            va = ga[(ga != 0) & (ga != EOS_ID) & (ga != MASK_ID)]
            vb = gb[(gb != 0) & (gb != EOS_ID) & (gb != MASK_ID)]
            text_a = tokenizer.decode(va, skip_special_tokens=True)[:300]
            text_b = tokenizer.decode(vb, skip_special_tokens=True)[:300]
            print(f"    [A monkey-patch]: {text_a}")
            print(f"    [B true-skip]:    {text_b}")
            if text_a == text_b:
                print(f"    → IDENTICAL")
            else:
                print(f"    → DIFFERENT")

        # ---- Final Summary ----
        print(f"\n{'=' * 80}")
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"  Forward count:       A={fwd_a}, B={fwd_b} ({'MATCH' if fwd_a==fwd_b else 'DIFFER'})")
        print(f"  Token exact match:   {match_pct:.1f}%")
        print(f"  Min cosine sim:      {min_cos_overall:.6f}")
        print(f"  Reuse rate:          A={reuse_pct_a:.1f}%, B={reuse_pct_b:.1f}%")

        if fwd_a == fwd_b and match_pct == 100.0 and min_cos_overall > 0.9999:
            print(f"\n  CONCLUSION: Approaches are EQUIVALENT. Pareto data is valid.")
        elif fwd_a == fwd_b and match_pct > 95.0:
            print(f"\n  CONCLUSION: Approaches are FUNCTIONALLY equivalent (bf16 noise).")
        else:
            print(f"\n  CONCLUSION: Approaches DIFFER. Need deeper investigation.")
            print(f"  Forward diff: {fwd_b - fwd_a:+d}")
            if first_diverge:
                print(f"  First divergence: step={first_diverge[0]}, layer={first_diverge[1]}")

        # Save results
        save_data = {
            "fwd_a": fwd_a, "fwd_b": fwd_b,
            "reuse_pct_a": reuse_pct_a, "reuse_pct_b": reuse_pct_b,
            "token_match_pct": match_pct,
            "min_cosine": min_cos_overall,
            "first_diverge": first_diverge,
            "stats_a": stats_a, "stats_b": stats_b,
        }
        out_path = REPO_ROOT / "codex_coding" / "results" / "monkey_patch_vs_true_skip_comparison.json"
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
