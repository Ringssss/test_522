#!/usr/bin/env python3
"""
M=5 Skip Behavior Verification

Verifies that MSkipEBController's hot_skip truly reuses cached s_mask
without silent modification. Records s_mask fingerprint at every
get_s_mask call and checks:
  1. Skip steps return identical s_mask as the last update/cold
  2. Update steps may change s_mask
  3. Layer caches are independent

Usage:
  CUDA_VISIBLE_DEVICES=4 python diag_m5_skip_verify.py
"""

from __future__ import annotations
import os, sys, socket, hashlib
from pathlib import Path
from collections import defaultdict

import torch

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

from test_fused_eb_triton import (
    fused_routing, FusedEBController,
    _kernel_A, _kernel_A_cold, _kernel_B_v3,
    _kernel_C, _kernel_D_v2, _kernel_zero_init,
)
from test_m_skip_sweep import MSkipEBController
from baseline_optimizations import apply_all_optimizations
from test_heteval128 import PROMPTS

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
GEN_LENGTH = 64   # 2 blocks, ~91 forwards
BATCH_SIZE = 128


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
    print("M=5 Skip Behavior Verification")
    print(f"  batch={BATCH_SIZE}, gen={GEN_LENGTH}, skip_m=5")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0),
                      use_cache=False)

        apply_all_optimizations(model)

        # Build input
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
        padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                  if ids.shape[0] < mx else ids for ids in all_ids]
        input_ids = torch.stack(padded, dim=0).to(device)

        decoder = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True,
                inplace_cache_update=True)

        # Create MSkipEBController with skip_m=5
        ctrl = MSkipEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8,
            skip_m=5)

        # Diagnostic state
        records = []  # (global_call_idx, layer_idx, event_type, s_sum, s_hash)
        global_call_idx = [0]

        # Wrap get_s_mask to record fingerprints
        original_get_s_mask = ctrl.get_s_mask

        def diag_get_s_mask(layer_idx, logits, bias):
            old_cold = ctrl.cold_count
            old_hot = ctrl.hot_count
            old_skips = ctrl.eb_skips

            s_mask = original_get_s_mask(layer_idx, logits, bias)

            # Determine event type
            if ctrl.cold_count > old_cold:
                event = "cold"
            elif ctrl.eb_skips > old_skips:
                event = "skip"
            else:
                event = "update"

            # Compute fingerprint: sum + md5 of full tensor
            torch.cuda.synchronize()
            s_cpu = s_mask.cpu()
            s_sum = int(s_cpu.sum().item())
            s_hash = hashlib.md5(s_cpu.numpy().tobytes()).hexdigest()[:12]

            records.append({
                'call_idx': global_call_idx[0],
                'layer': layer_idx,
                'event': event,
                's_sum': s_sum,
                's_hash': s_hash,
            })
            global_call_idx[0] += 1
            return s_mask

        ctrl.get_s_mask = diag_get_s_mask

        # Patch routing
        idx = 0
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                     mod.top_k, mod.n_group, mod.topk_group)
                li = idx
                def mk(bb, rr, tt, nn, gg, layer_i, cc):
                    def fn(hs, go, topk, renorm):
                        sm = cc.get_s_mask(layer_i, go, bb)
                        w, i = fused_routing(go, bb, rr, s_mask=sm, K=tt, ng=nn, tkg=gg)
                        return w.to(go.dtype), i
                    return fn
                mod.routing = mk(b, r, tk, ng, tkg, li, ctrl)
                idx += 1

        # Warmup
        print("\nWarmup...")
        dllm = make_dllm()
        with torch.inference_mode():
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Done (warmup, not recorded)")

        # Reset for diagnostic run
        ctrl.prev_N.clear(); ctrl.K_init.clear()
        ctrl.cold_count = 0; ctrl.hot_count = 0
        ctrl.eb_calls = 0; ctrl.eb_skips = 0
        ctrl._bufs.clear(); ctrl.k_init_history.clear()
        ctrl.s_mask_cache.clear(); ctrl.pop_cache.clear()
        ctrl._fwd_in_block.clear(); ctrl._block_idx.clear()
        records.clear()
        global_call_idx[0] = 0

        # Diagnostic run
        print("\nDiagnostic run (gen_length=64)...")
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        num_fwd = dllm.diff_iteration.num_forwards
        print(f"  {num_fwd} forwards, {len(records)} get_s_mask calls")
        print(f"  cold={ctrl.cold_count}, hot={ctrl.hot_count}, "
              f"eb_calls={ctrl.eb_calls}, eb_skips={ctrl.eb_skips}")

        # ============================================
        # Analyze records
        # ============================================
        print(f"\n{'='*80}")
        print("ANALYSIS")
        print(f"{'='*80}")

        # Group by layer
        by_layer = defaultdict(list)
        for r in records:
            by_layer[r['layer']].append(r)

        total_checks = 0
        fail_checks = 0

        # Check 1: skip fingerprint must match last cold/update
        print("\n--- Check 1: Skip steps must have same hash as last cold/update ---")
        for li in sorted(by_layer.keys()):
            seq = by_layer[li]
            last_active_hash = None
            layer_fails = 0
            layer_skips = 0

            for r in seq:
                if r['event'] in ('cold', 'update'):
                    last_active_hash = r['s_hash']
                elif r['event'] == 'skip':
                    layer_skips += 1
                    total_checks += 1
                    if last_active_hash is not None and r['s_hash'] != last_active_hash:
                        fail_checks += 1
                        layer_fails += 1

            status = "PASS" if layer_fails == 0 else f"FAIL ({layer_fails}/{layer_skips})"
            # Print first few layers in detail
            if li < 3 or layer_fails > 0:
                events = [f"{r['event'][0].upper()}:{r['s_sum']}" for r in seq[:25]]
                print(f"  L{li:>2d}: {status} ({layer_skips} skips) | {' '.join(events)}...")

        print(f"\n  Total skip checks: {total_checks}, Failures: {fail_checks}")

        # Check 2: updates can change hash
        print("\n--- Check 2: Update steps may change hash ---")
        update_changes = 0
        update_same = 0
        for li in sorted(by_layer.keys()):
            seq = by_layer[li]
            prev_hash = None
            for r in seq:
                if r['event'] == 'update':
                    if prev_hash is not None:
                        if r['s_hash'] != prev_hash:
                            update_changes += 1
                        else:
                            update_same += 1
                    prev_hash = r['s_hash']
                elif r['event'] == 'cold':
                    prev_hash = r['s_hash']
        print(f"  Updates that changed hash: {update_changes}")
        print(f"  Updates with same hash:    {update_same}")
        print(f"  (Same hash is OK if expert distribution is stable)")

        # Check 3: layer independence
        print("\n--- Check 3: Layer independence ---")
        # For each forward, check that different layers have different hashes
        # (not necessarily, but if ALL layers have identical hash, something is wrong)
        fwd_hashes = defaultdict(set)
        for r in records:
            fwd_idx = r['call_idx'] // 19  # approximate forward index
            fwd_hashes[fwd_idx].add(r['s_hash'])

        all_uniform = sum(1 for fwd, hashes in fwd_hashes.items() if len(hashes) == 1)
        print(f"  Forwards where all 19 layers had identical hash: {all_uniform}/{len(fwd_hashes)}")
        if all_uniform == 0:
            print(f"  PASS — layers have independent s_mask values")
        else:
            print(f"  WARNING — {all_uniform} forwards with uniform hash across layers")

        # Print detailed sequence for layer 0 and layer 10
        print(f"\n--- Detailed sequence: Layer 0 (first 40 calls) ---")
        seq0 = by_layer[0][:40]
        print(f"  {'#':>3} {'Event':>6} {'|S|':>5} {'Hash':>14}")
        for r in seq0:
            print(f"  {r['call_idx']//19:>3} {r['event']:>6} {r['s_sum']:>5} {r['s_hash']:>14}")

        print(f"\n--- Detailed sequence: Layer 10 (first 40 calls) ---")
        seq10 = by_layer[10][:40]
        print(f"  {'#':>3} {'Event':>6} {'|S|':>5} {'Hash':>14}")
        for r in seq10:
            print(f"  {r['call_idx']//19:>3} {r['event']:>6} {r['s_sum']:>5} {r['s_hash']:>14}")

        # Final verdict
        print(f"\n{'='*80}")
        if fail_checks == 0:
            print("VERDICT: M=5 skip is VERIFIED — cached s_mask is truly reused unchanged")
        else:
            print(f"VERDICT: M=5 skip FAILED — {fail_checks} skip steps returned modified s_mask!")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
