#!/usr/bin/env python3
"""
v0.1.15.2 — fused_experts Kernel Micro-Benchmark: Physical Top-K Reduction

Goal: Determine whether reducing top_k (physically fewer token-expert pairs)
translates to wall-clock kernel speedup. This answers whether the MoE kernel
is compute-bound or memory-bound at our specific configuration.

Method:
  - Extract real w1, w2 weights from one MoE layer of LLaDA2.0-mini
  - Generate real routing data from a forward pass
  - Call fused_experts() directly with [N, k] for k in {8, 4, 2, 1}
  - Time with CUDA Events, 100 repeats, 20 warmup
  - Test N = 32, 256, 1024

Interpretation:
  top-4 / top-8 ≈ 0.5 → compute-bound (GEMM savings realizable)
  top-4 / top-8 ≈ 1.0 → memory-bound (GEMM not the bottleneck)
  0.5 < ratio < 1.0  → mixed
"""

from __future__ import annotations
import os, sys, socket, json
from pathlib import Path

import torch

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
RESULTS_DIR = REPO_ROOT / "codex_coding" / "results"


def time_fused_experts(hidden_states, w1, w2, topk_weights, topk_ids,
                       warmup=20, repeats=100):
    """Time fused_experts with CUDA Events."""
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    # Warmup
    for _ in range(warmup):
        fused_experts(hidden_states, w1, w2, topk_weights, topk_ids,
                      inplace=False)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fused_experts(hidden_states, w1, w2, topk_weights, topk_ids,
                      inplace=False)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def get_routing_data(model, tokenizer, device, n_tokens):
    """Run a forward pass to get real routing data for a specific layer."""
    # Build input of appropriate size
    text = "Explain the theory of relativity in detail, covering special and general relativity."
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True, tokenize=False)
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]

    # Repeat to fill n_tokens
    batch_size = max(1, n_tokens // 32)
    if batch_size > 1:
        input_ids = ids.unsqueeze(0).expand(batch_size, -1).contiguous().to(device)
    else:
        input_ids = ids.unsqueeze(0).to(device)

    # Hook a specific MoE layer (layer index 10, middle layer) to capture routing
    captured = {}
    target_layer = model.model.layers[10]  # MoE layer 9 (0-indexed, layer 0 is dense)
    moe = target_layer.mlp

    def capture_hook(hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)
        # Get routing
        topk_idx, topk_weight, _ = moe.gate(hs_flat)
        captured["hidden_states"] = hs_flat.detach()
        captured["topk_idx"] = topk_idx.detach()
        captured["topk_weight"] = topk_weight.detach()
        # Still need to return valid output
        shared_res = moe.shared_experts(hidden_states)
        router_logits = moe.gate.get_logits(hs_flat)
        routed_y = moe.experts.forward_impl(
            hidden_states=hs_flat, router_logits=router_logits)
        routed_y = routed_y.view(bsz, seq_len, h)
        return routed_y + shared_res if moe.config.num_shared_experts is not None else routed_y

    orig_forward = moe.forward
    moe.forward = capture_hook
    try:
        with torch.inference_mode():
            model(input_ids, use_cache=False)
    finally:
        moe.forward = orig_forward

    return captured


def build_topk_subset(topk_idx, topk_weight, keep_k):
    """From full top-8 routing, extract the top-keep_k by weight."""
    if keep_k >= topk_idx.size(1):
        return topk_idx, topk_weight
    # Sort by weight descending, take top keep_k
    sorted_w, sort_order = topk_weight.sort(dim=1, descending=True)
    kept_order = sort_order[:, :keep_k]
    new_idx = topk_idx.gather(1, kept_order)
    new_weight = topk_weight.gather(1, kept_order)
    # Renormalize
    w_sum = new_weight.sum(dim=1, keepdim=True)
    new_weight = new_weight / (w_sum + 1e-8) * topk_weight.sum(dim=1, keepdim=True)
    return new_idx.contiguous(), new_weight.contiguous()


def count_active_experts(topk_ids, num_experts=256):
    """Count unique active experts."""
    return topk_ids.unique().numel()


def count_padded_blocks(topk_ids, block_size, num_experts=256):
    """Estimate padded block count from moe_align_block_size."""
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    _, _, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size, num_experts)
    return num_tokens_post_padded.item()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)
    sys.path.insert(0, str(REPO_ROOT / "lib_cite" / "dInfer" / "python"))

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
    print("v0.1.15.2 — fused_experts Kernel Micro-Benchmark")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        # Warmup model
        with torch.inference_mode():
            _ = model(torch.arange(64, dtype=torch.long, device=device).unsqueeze(0),
                      use_cache=False)

        # Extract w1, w2 from MoE layer 10 (layer index 10)
        moe_layer = model.model.layers[10].mlp
        w1 = moe_layer.experts.w13_weight  # [E, N, K]
        w2 = moe_layer.experts.w2_weight   # [E, K, N]
        print(f"\nMoE weights: w1={list(w1.shape)}, w2={list(w2.shape)}, dtype={w1.dtype}")
        print(f"  E={w1.shape[0]}, N(w1)={w1.shape[1]}, K={w1.shape[2]}")

        # Get BLOCK_SIZE_M from autotuning config
        # Read the config to know what block sizes are used
        from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
        test_config = try_get_optimal_moe_config(
            w1.shape, w2.shape, 8, "torch.bfloat16", M=32)
        BLOCK_SIZE_M = test_config["BLOCK_SIZE_M"]
        print(f"  BLOCK_SIZE_M={BLOCK_SIZE_M}")

        TOKEN_COUNTS = [32, 256, 1024]
        TOP_KS = [8, 4, 2, 1]
        WARMUP = 20
        REPEATS = 100

        all_results = {}

        for n_tokens in TOKEN_COUNTS:
            print(f"\n{'='*80}")
            print(f"N = {n_tokens} tokens")
            print(f"{'='*80}")

            # Get real routing data
            routing = get_routing_data(model, tokenizer, device, n_tokens)
            hs_full = routing["hidden_states"]
            idx_full = routing["topk_idx"]
            w_full = routing["topk_weight"]

            # Trim or repeat to exact n_tokens
            if hs_full.shape[0] > n_tokens:
                hs = hs_full[:n_tokens].contiguous()
                idx8 = idx_full[:n_tokens].contiguous()
                w8 = w_full[:n_tokens].contiguous()
            elif hs_full.shape[0] < n_tokens:
                repeats_needed = (n_tokens + hs_full.shape[0] - 1) // hs_full.shape[0]
                hs = hs_full.repeat(repeats_needed, 1)[:n_tokens].contiguous()
                idx8 = idx_full.repeat(repeats_needed, 1)[:n_tokens].contiguous()
                w8 = w_full.repeat(repeats_needed, 1)[:n_tokens].contiguous()
            else:
                hs = hs_full.contiguous()
                idx8 = idx_full.contiguous()
                w8 = w_full.contiguous()

            print(f"  hidden_states: {list(hs.shape)}")
            print(f"  topk_ids (full): {list(idx8.shape)}")

            # Get autotuning config for this N
            cfg = try_get_optimal_moe_config(
                w1.shape, w2.shape, 8, "torch.bfloat16", M=n_tokens)
            print(f"  Autotuning config (top-8): BLOCK_SIZE_M={cfg['BLOCK_SIZE_M']}, "
                  f"BLOCK_SIZE_N={cfg['BLOCK_SIZE_N']}, BLOCK_SIZE_K={cfg['BLOCK_SIZE_K']}")

            results_n = {}
            baseline_time = None

            for top_k in TOP_KS:
                idx_k, w_k = build_topk_subset(idx8, w8, top_k)

                active = count_active_experts(idx_k)
                padded = count_padded_blocks(idx_k, cfg["BLOCK_SIZE_M"])
                total_pairs = idx_k.numel()

                # Get config for this top_k
                cfg_k = try_get_optimal_moe_config(
                    w1.shape, w2.shape, top_k, "torch.bfloat16", M=n_tokens)

                with torch.inference_mode():
                    t_ms = time_fused_experts(hs, w1, w2, w_k, idx_k,
                                              warmup=WARMUP, repeats=REPEATS)

                if top_k == 8:
                    baseline_time = t_ms

                ratio = t_ms / baseline_time if baseline_time else 1.0
                speedup = baseline_time / t_ms if t_ms > 0 else 0

                print(f"\n  top-{top_k}:")
                print(f"    kernel time:    {t_ms:.4f} ms")
                print(f"    ratio vs top-8: {ratio:.3f}  (speedup: {speedup:.2f}x)")
                print(f"    total pairs:    {total_pairs}")
                print(f"    padded blocks:  {padded // cfg['BLOCK_SIZE_M']}")
                print(f"    active experts: {active} / 256")
                print(f"    config: BSM={cfg_k['BLOCK_SIZE_M']}, BSN={cfg_k['BLOCK_SIZE_N']}, "
                      f"BSK={cfg_k['BLOCK_SIZE_K']}")

                results_n[f"top-{top_k}"] = {
                    "time_ms": t_ms,
                    "ratio_vs_top8": ratio,
                    "speedup": speedup,
                    "total_pairs": total_pairs,
                    "padded_blocks": padded // cfg["BLOCK_SIZE_M"],
                    "active_experts": active,
                    "config": cfg_k,
                }

            all_results[f"N={n_tokens}"] = results_n

        # ---- Summary ----
        print(f"\n{'='*80}")
        print(f"SUMMARY: Kernel Time Ratios (top-k / top-8)")
        print(f"{'='*80}")
        print(f"  {'N':>6s}  {'top-8':>8s}  {'top-4':>8s}  {'top-2':>8s}  {'top-1':>8s}  "
              f"{'top4/top8':>10s}  {'Verdict':>12s}")
        print(f"  {'-'*70}")

        for n_tokens in TOKEN_COUNTS:
            rn = all_results[f"N={n_tokens}"]
            t8 = rn["top-8"]["time_ms"]
            t4 = rn["top-4"]["time_ms"]
            t2 = rn["top-2"]["time_ms"]
            t1 = rn["top-1"]["time_ms"]
            r4 = rn["top-4"]["ratio_vs_top8"]

            if r4 < 0.6:
                verdict = "COMPUTE-BOUND"
            elif r4 < 0.85:
                verdict = "MIXED"
            else:
                verdict = "MEMORY-BOUND"

            print(f"  {n_tokens:>6d}  {t8:>7.3f}ms  {t4:>7.3f}ms  {t2:>7.3f}ms  {t1:>7.3f}ms  "
                  f"{r4:>10.3f}  {verdict:>12s}")

        # Active expert comparison
        print(f"\n  Active Experts (top-8 → top-4 → top-2 → top-1):")
        for n_tokens in TOKEN_COUNTS:
            rn = all_results[f"N={n_tokens}"]
            ae = [rn[f"top-{k}"]["active_experts"] for k in TOP_KS]
            print(f"    N={n_tokens:>4d}: {ae[0]:>3d} → {ae[1]:>3d} → {ae[2]:>3d} → {ae[3]:>3d}")

        # Padded blocks comparison
        print(f"\n  Padded Blocks (top-8 → top-4 → top-2 → top-1):")
        for n_tokens in TOKEN_COUNTS:
            rn = all_results[f"N={n_tokens}"]
            pb = [rn[f"top-{k}"]["padded_blocks"] for k in TOP_KS]
            print(f"    N={n_tokens:>4d}: {pb[0]:>4d} → {pb[1]:>4d} → {pb[2]:>4d} → {pb[3]:>4d}")

        # Conclusion
        print(f"\n{'='*80}")
        print(f"INTERPRETATION")
        print(f"{'='*80}")
        r4_1024 = all_results["N=1024"]["top-4"]["ratio_vs_top8"]
        if r4_1024 < 0.6:
            print(f"  At N=1024 (batch=32): top-4/top-8 = {r4_1024:.3f}")
            print(f"  → Kernel is COMPUTE-BOUND at production batch size.")
            print(f"  → Physical top-k reduction WILL yield wall-clock savings.")
            print(f"  → Next step: inline top-p pruning into modeling code.")
        elif r4_1024 < 0.85:
            print(f"  At N=1024 (batch=32): top-4/top-8 = {r4_1024:.3f}")
            print(f"  → Kernel is MIXED bound.")
            print(f"  → Physical top-k reduction yields PARTIAL wall-clock savings.")
            print(f"  → Savings come from both reduced GEMM and reduced token loads.")
        else:
            print(f"  At N=1024 (batch=32): top-4/top-8 = {r4_1024:.3f}")
            print(f"  → Kernel is MEMORY-BOUND at production batch size.")
            print(f"  → Physical top-k reduction alone is NOT sufficient.")
            print(f"  → Need to reduce active expert count or HBM traffic.")

        # Save results
        save_results = {}
        for k, v in all_results.items():
            save_results[k] = {}
            for kk, vv in v.items():
                save_results[k][kk] = {kk2: vv2 for kk2, vv2 in vv.items()
                                        if kk2 != "config"}
                save_results[k][kk]["config"] = {ck: cv for ck, cv in vv["config"].items()}
        out_path = RESULTS_DIR / "fused_experts_topk_benchmark.json"
        with open(out_path, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
