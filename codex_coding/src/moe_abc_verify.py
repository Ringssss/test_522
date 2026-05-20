#!/usr/bin/env python3
"""
v0.1.14.7c — Single-step A/B/C verification

For ONE model forward, compare three approaches:
  A: Compute all 256 tokens → replace reuse positions with cache
  B: Zero-pad reuse positions → compute all 256 → take fresh results + cache for reuse
  C: index_select only fresh tokens (~210) → compute → scatter back + cache for reuse

If A fresh == B fresh (bit-identical) → fused_moe is token-independent
If B fresh != C fresh → numerical diff from kernel tiling on different token count
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MASK_ID = 156895
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
TOP_K = 8


def main():
    import socket
    from contextlib import closing
    from transformers import AutoTokenizer, AutoConfig

    def find_free_port():
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    print("=" * 80)
    print("v0.1.14.7c — Single-step A/B/C Verification")
    print("=" * 80)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    sys.path.insert(0, str(REPO_ROOT / "lib_cite" / "dInfer" / "python"))
    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

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

        # ============================================================
        # Setup: get a real hidden_states tensor from a forward pass
        # ============================================================
        print("\nPreparing test data...", flush=True)

        # Use fresh run data for realistic hidden states
        data_path = REPO_ROOT / "codex_coding" / "results" / "proxy_risk_prediction" / "full_fresh_run_data.pt"
        fresh_data = torch.load(data_path, map_location="cpu")

        # Pick step=10, layer=8 (middle layer, good test case)
        TEST_STEP = 10
        TEST_LAYER = 8  # MoE layer index

        # Get the pre-MoE hidden states for this step/layer
        hidden_states_3d = fresh_data["pre_moe_hidden"][TEST_STEP][TEST_LAYER].to(device)
        bsz, seq_len, h = hidden_states_3d.shape
        print(f"  hidden_states: [{bsz}, {seq_len}, {h}]")

        # Get cached routed output from previous step
        cached_routed_3d = fresh_data["routed_output"][TEST_STEP - 1][TEST_LAYER].to(device)

        # Create a reuse mask: simulate margin>0.99
        # Use actual logits info to get realistic mask
        li = fresh_data["logits_info"][TEST_STEP]
        margin = li["margin"]  # [bsz, seq_len]
        reuse_mask_2d = (margin > 0.99)[:bsz]  # [bsz, seq_len]
        n_reuse = reuse_mask_2d.sum().item()
        n_total = bsz * seq_len
        n_fresh = n_total - n_reuse
        print(f"  Reuse mask: {n_reuse}/{n_total} tokens reuse ({n_reuse/n_total*100:.1f}%)")

        # Get the MoE block
        layers = model.model.layers
        moe_layers = [(idx, l) for idx, l in enumerate(layers)
                      if hasattr(l, 'mlp') and hasattr(l.mlp, 'gate')]
        moe_block = moe_layers[TEST_LAYER][1].mlp

        flat_reuse_mask = reuse_mask_2d.view(-1).to(device)  # [N]
        fresh_mask = ~flat_reuse_mask
        fresh_indices = fresh_mask.nonzero(as_tuple=True)[0]

        hs_flat = hidden_states_3d.view(-1, h)  # [N, h]
        cached_flat = cached_routed_3d[:bsz].view(-1, h).to(device)  # [N, h]

        print(f"  fresh_indices: {len(fresh_indices)} tokens")
        print(f"  Testing on MoE layer {TEST_LAYER}")

        # ============================================================
        # Approach A: Compute ALL tokens, replace reuse with cache
        # ============================================================
        print(f"\n{'='*60}")
        print("Approach A: Compute all → replace reuse positions")
        print(f"{'='*60}")

        with torch.inference_mode():
            router_logits_A = moe_block.gate.get_logits(hs_flat)
            routed_A_full = moe_block.experts.forward_impl(
                hidden_states=hs_flat, router_logits=router_logits_A)
            # Save fresh positions BEFORE replacement
            routed_A_fresh_only = routed_A_full[fresh_indices].clone()
            # Apply replacement
            routed_A = routed_A_full.clone()
            routed_A[flat_reuse_mask] = cached_flat[flat_reuse_mask]

        print(f"  routed_A shape: {routed_A.shape}")
        print(f"  routed_A_fresh_only shape: {routed_A_fresh_only.shape}")

        # ============================================================
        # Approach B: Zero-pad reuse positions, compute all 256
        # ============================================================
        print(f"\n{'='*60}")
        print("Approach B: Zero-pad reuse → compute all 256 → take fresh + cache")
        print(f"{'='*60}")

        with torch.inference_mode():
            hs_padded = hs_flat.clone()
            hs_padded[flat_reuse_mask] = 0.0  # zero out reuse positions

            router_logits_B = moe_block.gate.get_logits(hs_padded)
            routed_B_full = moe_block.experts.forward_impl(
                hidden_states=hs_padded, router_logits=router_logits_B)
            # Take only fresh positions
            routed_B_fresh_only = routed_B_full[fresh_indices].clone()
            # Assemble final: fresh from computation, reuse from cache
            routed_B = cached_flat.clone()
            routed_B[fresh_indices] = routed_B_fresh_only

        print(f"  routed_B shape: {routed_B.shape}")

        # ============================================================
        # Approach C: index_select only fresh tokens
        # ============================================================
        print(f"\n{'='*60}")
        print(f"Approach C: index_select {n_fresh} fresh tokens → compute → scatter")
        print(f"{'='*60}")

        with torch.inference_mode():
            fresh_hs = hs_flat[fresh_indices]  # [n_fresh, h]
            router_logits_C = moe_block.gate.get_logits(fresh_hs)
            routed_C_fresh_only = moe_block.experts.forward_impl(
                hidden_states=fresh_hs, router_logits=router_logits_C)
            # Assemble final
            routed_C = cached_flat.clone()
            routed_C[fresh_indices] = routed_C_fresh_only

        print(f"  routed_C_fresh_only shape: {routed_C_fresh_only.shape}")

        # ============================================================
        # COMPARISON
        # ============================================================
        print(f"\n{'='*80}")
        print("COMPARISON RESULTS")
        print(f"{'='*80}")

        def compare(name1, t1, name2, t2):
            """Compare two tensors, report stats."""
            bit_identical = torch.equal(t1, t2)
            diff = (t1.float() - t2.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            cos = F.cosine_similarity(t1.float().view(1, -1), t2.float().view(1, -1)).item()
            rel_diff = diff.sum().item() / (t1.float().abs().sum().item() + 1e-10)

            status = "BIT-IDENTICAL" if bit_identical else "DIFFERS"
            print(f"\n  {name1} vs {name2}: {status}")
            print(f"    max_abs_diff:  {max_diff:.2e}")
            print(f"    mean_abs_diff: {mean_diff:.2e}")
            print(f"    cosine_sim:    {cos:.10f}")
            print(f"    rel_diff:      {rel_diff:.2e}")
            return bit_identical

        print("\n--- Fresh token outputs (the tokens we care about) ---")

        ab = compare("A_fresh", routed_A_fresh_only, "B_fresh", routed_B_fresh_only)
        ac = compare("A_fresh", routed_A_fresh_only, "C_fresh", routed_C_fresh_only)
        bc = compare("B_fresh", routed_B_fresh_only, "C_fresh", routed_C_fresh_only)

        print("\n--- Final assembled output (fresh + cache) ---")

        ab_final = compare("A_final", routed_A, "B_final", routed_B)
        ac_final = compare("A_final", routed_A, "C_final", routed_C)
        bc_final = compare("B_final", routed_B, "C_final", routed_C)

        # ============================================================
        # Also check: does gate give same routing for fresh tokens?
        # ============================================================
        print(f"\n{'='*60}")
        print("Gate routing consistency check")
        print(f"{'='*60}")

        with torch.inference_mode():
            gate_full = moe_block.gate.get_logits(hs_flat)
            gate_fresh_from_full = gate_full[fresh_indices]

            gate_fresh_direct = moe_block.gate.get_logits(hs_flat[fresh_indices])

        gate_identical = torch.equal(gate_fresh_from_full, gate_fresh_direct)
        gate_diff = (gate_fresh_from_full.float() - gate_fresh_direct.float()).abs().max().item()
        print(f"  Gate logits (full batch vs fresh-only): "
              f"{'BIT-IDENTICAL' if gate_identical else 'DIFFERS'} "
              f"(max_diff={gate_diff:.2e})")

        # ============================================================
        # CONCLUSION
        # ============================================================
        print(f"\n{'='*80}")
        print("CONCLUSION")
        print(f"{'='*80}")

        if ab:
            print("  A==B (fresh): Zeroing reuse positions doesn't affect fresh token computation")
            print("         → fused_moe is token-independent (no cross-token contamination)")
        else:
            print("  A!=B (fresh): Reuse token values affect fresh token computation!")
            print("         → fused_moe has cross-token dependency")

        if ac:
            print("  A==C (fresh): index_select gives identical results to full-batch")
            print("         → No kernel tiling effect, true-skip is safe")
        else:
            if ab and not ac:
                print("  A==B but A!=C: Difference comes from kernel processing different #tokens")
                print("         → Tiling/accumulation order effect confirmed")
                print("         → Pad-placeholder (B) approach avoids this and gives bit-identical results!")
            elif not ab and not ac:
                print("  Both A!=B and A!=C: Complex interactions in fused_moe")

        if bc:
            print("  B==C (fresh): Zero-pad and index_select give same results")
        else:
            print("  B!=C (fresh): Zero-pad and index_select differ")

        print("\nDone.")


if __name__ == "__main__":
    main()
