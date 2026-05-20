#!/usr/bin/env python3
"""
Weight Compaction Micro-bench: fused_experts with 256-expert sparse IDs vs
compact N-expert dense IDs.

Tests:
  A) Original: w[256, ...], topk_ids in [0,255]
  B) Compact:  w[161, ...], topk_ids remapped to [0,160]
  C) Sweep:    compact sizes 120, 140, 161, 200, 256

Also measures compaction overhead (weight copy + remap table).
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path

import torch
import numpy as np

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

from test_fused_eb_triton import (
    fused_routing, FusedEBController,
    _kernel_A, _kernel_A_cold, _kernel_B_v3,
    _kernel_C, _kernel_D_v2, _kernel_zero_init,
)
from baseline_optimizations import apply_all_optimizations
from test_heteval128 import PROMPTS

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"


def bench_fused_experts(hidden_states, w13, w2, topk_weights, topk_ids,
                        warmup=20, repeat=100, label=""):
    """Benchmark fused_experts with CUDA event timing."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    # Warmup
    for _ in range(warmup):
        _ = fused_experts(hidden_states, w13, w2, topk_weights, topk_ids)
    torch.cuda.synchronize()

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        _ = fused_experts(hidden_states, w13, w2, topk_weights, topk_ids)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg = sum(times) / len(times)
    std = (sum((t - avg)**2 for t in times) / len(times)) ** 0.5
    p50 = sorted(times)[len(times)//2]
    print(f"  {label:<35s} avg={avg:.3f}ms  std={std:.3f}ms  p50={p50:.3f}ms  "
          f"w_shape={list(w13.shape)}")
    return {'avg_ms': avg, 'std_ms': std, 'p50_ms': p50, 'w13_shape': list(w13.shape)}


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoTokenizer, AutoConfig

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print("Weight Compaction Micro-bench")
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

        print("\nApplying baseline optimizations...")
        n_rms, n_fa = apply_all_optimizations(model)
        print(f"  Fused RMSNorm: {n_rms}, Flash-attn: {n_fa}")

        # ============================================================
        # Step 1: Get real routing data from one forward
        # ============================================================
        print("\n--- Step 1: Collect real routing data ---")

        # Build batch=128 input
        all_ids = []
        for i in range(128):
            text = PROMPTS[i]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                  if ids.shape[0] < mx else ids for ids in all_ids]
        input_ids = torch.stack(padded, dim=0).to(device)

        # Run one forward to get hidden_states at a MoE layer
        target_layer = 10  # model layer 11 (MoE)
        captured = {}

        def capture_hook(mod, inp):
            # inp is a tuple, inp[0] is hidden_states [batch, seq, hidden]
            captured['hidden_states'] = inp[0].detach()

        moe_block = model.model.layers[target_layer].mlp
        hook = moe_block.register_forward_pre_hook(capture_hook)

        with torch.inference_mode():
            _ = model(input_ids[:, :120].clone(), use_cache=False)  # use first 120 tokens for shape
        hook.remove()

        hs = captured['hidden_states']
        bsz, seq_len, hidden = hs.shape
        hs_flat = hs.view(-1, hidden)
        N = hs_flat.shape[0]  # batch * seq_len
        print(f"  Captured hidden_states: N={N}, hidden={hidden}")

        # Get gate output and routing
        gate = moe_block.gate
        with torch.inference_mode():
            logits = gate.get_logits(hs_flat)

        # Get S_mask from EB cold path
        eb_ctrl = FusedEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8)
        s_mask = eb_ctrl.cold_path(target_layer, logits, gate.expert_bias)
        s_mask_size = int(s_mask.sum().item())
        active_ids = s_mask.nonzero(as_tuple=True)[0]
        print(f"  S_mask: {s_mask_size} active experts out of 256")
        print(f"  Active expert IDs (first 20): {active_ids[:20].tolist()}")

        # Run fused_routing to get topk_ids and topk_weights
        with torch.inference_mode():
            topk_weights, topk_ids = fused_routing(
                logits, gate.expert_bias, gate.routed_scaling_factor,
                s_mask=s_mask, K=8, ng=8, tkg=4)
        print(f"  topk_ids: {topk_ids.shape}, unique experts: {topk_ids.unique().numel()}")

        # Get weight tensors
        w13_full = moe_block.experts.w13_weight  # [256, 1024, 2048]
        w2_full = moe_block.experts.w2_weight    # [256, 2048, 512]
        print(f"  w13: {w13_full.shape}, w2: {w2_full.shape}")

        # ============================================================
        # Step 2: Build compact weights
        # ============================================================
        print("\n--- Step 2: Build compact weights ---")

        # Compaction: active_ids → contiguous 0..K
        remap_table = torch.full((256,), -1, dtype=torch.int32, device=device)
        for new_id, old_id in enumerate(active_ids):
            remap_table[old_id] = new_id

        # Compact weights
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        w13_compact = w13_full[active_ids].contiguous()  # [161, 1024, 2048]
        w2_compact = w2_full[active_ids].contiguous()    # [161, 2048, 512]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        compact_time_ms = (t1 - t0) * 1000
        print(f"  Compaction time: {compact_time_ms:.2f}ms")
        print(f"  w13_compact: {w13_compact.shape}, w2_compact: {w2_compact.shape}")

        # Remap IDs
        topk_ids_remapped = remap_table[topk_ids.long()].to(torch.int32)
        assert (topk_ids_remapped >= 0).all(), "Remap failed: some IDs not in active set"
        print(f"  topk_ids_remapped range: [{topk_ids_remapped.min()}, {topk_ids_remapped.max()}]")

        # Correctness check
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        with torch.inference_mode():
            out_orig = fused_experts(hs_flat, w13_full, w2_full,
                                     topk_weights, topk_ids)
            out_compact = fused_experts(hs_flat, w13_compact, w2_compact,
                                        topk_weights, topk_ids_remapped)
        diff = (out_orig - out_compact).abs().max().item()
        print(f"  Correctness check: max_diff={diff:.6e} → {'PASS' if diff < 1e-3 else 'FAIL'}")

        # ============================================================
        # Step 3: Micro-bench comparison
        # ============================================================
        print(f"\n--- Step 3: Micro-bench (N={N}, K=8) ---")
        print(f"  (warmup=20, repeat=100)")

        results = {}

        with torch.inference_mode():
            # A) Original: 256 experts, sparse IDs
            r = bench_fused_experts(hs_flat, w13_full, w2_full,
                                    topk_weights, topk_ids,
                                    label=f"A) Original [256]")
            results['original_256'] = r

            # B) Compact: 161 experts, dense IDs
            r = bench_fused_experts(hs_flat, w13_compact, w2_compact,
                                    topk_weights, topk_ids_remapped,
                                    label=f"B) Compact [{s_mask_size}]")
            results[f'compact_{s_mask_size}'] = r

            # C) Sweep different compact sizes
            print(f"\n  --- Compact size sweep ---")
            for target_size in [120, 140, 180, 200, 230, 256]:
                if target_size > 256:
                    continue
                if target_size == 256:
                    # Full size = original
                    r = bench_fused_experts(hs_flat, w13_full, w2_full,
                                            topk_weights, topk_ids,
                                            label=f"Sweep [{target_size}] (=original)")
                    results[f'sweep_{target_size}'] = r
                elif target_size <= s_mask_size:
                    # Use subset of active experts
                    sub_ids = active_ids[:target_size]
                    sub_remap = torch.full((256,), -1, dtype=torch.int32, device=device)
                    for new_id, old_id in enumerate(sub_ids):
                        sub_remap[old_id] = new_id
                    # Re-route: tokens going to experts outside subset → clamp to nearest in subset
                    sub_ids_remapped = sub_remap[topk_ids.long()]
                    # For IDs not in subset, map to 0 (just for benchmarking shapes, not correctness)
                    sub_ids_remapped = sub_ids_remapped.clamp(min=0).to(torch.int32)
                    w13_sub = w13_full[sub_ids].contiguous()
                    w2_sub = w2_full[sub_ids].contiguous()
                    r = bench_fused_experts(hs_flat, w13_sub, w2_sub,
                                            topk_weights, sub_ids_remapped,
                                            label=f"Sweep [{target_size}]")
                    results[f'sweep_{target_size}'] = r
                else:
                    # Expand: use active + some padding experts
                    n_extra = target_size - s_mask_size
                    inactive = (s_mask == 0).nonzero(as_tuple=True)[0]
                    extra_ids = inactive[:n_extra]
                    expanded_ids = torch.cat([active_ids, extra_ids])
                    exp_remap = torch.full((256,), -1, dtype=torch.int32, device=device)
                    for new_id, old_id in enumerate(expanded_ids):
                        exp_remap[old_id] = new_id
                    exp_ids_remapped = exp_remap[topk_ids.long()].clamp(min=0).to(torch.int32)
                    w13_exp = w13_full[expanded_ids].contiguous()
                    w2_exp = w2_full[expanded_ids].contiguous()
                    r = bench_fused_experts(hs_flat, w13_exp, w2_exp,
                                            topk_weights, exp_ids_remapped,
                                            label=f"Sweep [{target_size}]")
                    results[f'sweep_{target_size}'] = r

        # ============================================================
        # Step 4: Measure remap overhead
        # ============================================================
        print(f"\n--- Step 4: Remap overhead ---")
        # Measure remap_table lookup time
        torch.cuda.synchronize()
        for _ in range(100):
            _ = remap_table[topk_ids.long()].to(torch.int32)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(1000):
            _ = remap_table[topk_ids.long()].to(torch.int32)
        end.record()
        torch.cuda.synchronize()
        remap_time = start.elapsed_time(end) / 1000
        print(f"  Remap per call: {remap_time:.4f}ms ({remap_time*1000:.1f}us)")
        results['remap_per_call_ms'] = remap_time
        results['compact_once_ms'] = compact_time_ms

        # ============================================================
        # Summary
        # ============================================================
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        orig = results['original_256']['avg_ms']
        comp = results[f'compact_{s_mask_size}']['avg_ms']
        delta = comp - orig
        delta_pct = delta / orig * 100
        print(f"  Original [256]:  {orig:.3f}ms")
        print(f"  Compact [{s_mask_size}]:  {comp:.3f}ms  ({delta:+.3f}ms, {delta_pct:+.1f}%)")
        print(f"  Compaction cost: {compact_time_ms:.2f}ms (one-time per M=5 window)")
        print(f"  Remap cost:      {remap_time:.4f}ms (per forward)")
        print(f"  Break-even:      {compact_time_ms / max(-delta, 0.001):.0f} forwards" if delta < 0
              else f"  No benefit: compact is slower")

        print(f"\n  Sweep:")
        for key in sorted(results.keys()):
            if key.startswith('sweep_'):
                size = key.split('_')[1]
                r = results[key]
                d = r['avg_ms'] - orig
                print(f"    [{size:>3s}]: {r['avg_ms']:.3f}ms  ({d:+.3f}ms, {d/orig*100:+.1f}%)")

        # Save
        out_path = REPO_ROOT / "codex_coding" / "results" / "weight_compaction_microbench.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
