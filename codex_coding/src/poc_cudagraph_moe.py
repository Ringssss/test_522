#!/usr/bin/env python3
"""
POC: CUDA Graph capture of MoE forward_impl.

Tests whether the dispatch->kernel->combine pipeline can be captured as a
CUDA Graph to eliminate framework overhead (~8.7 ms/fwd measured in v0.1.15.17).

Groups:
  G1  Eager MoE forward_impl timing (baseline)
  G2  CUDA Graph capture attempt (full forward_impl)
  G3  CUDA Graph replay timing + correctness
  G4  Per-layer scaling estimate (19 layers sequential)

Usage:
  torchrun --nproc_per_node=8 poc_cudagraph_moe.py [--batch-size 512]
"""

import argparse
import json
import os
import sys
import time
import traceback

import torch
import torch.distributed as dist

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
BLOCK_LENGTH = 32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--tp-size", type=int, default=4)
    p.add_argument("--num-warmup", type=int, default=5)
    p.add_argument("--num-iters", type=int, default=20)
    p.add_argument("--results-suffix", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    tp_size = args.tp_size
    dp_size = world_size // tp_size
    dp_rank = rank // tp_size
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # ---- init distributed (same as bench script) ----
    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    pcfg_init = ParallelConfig(
        tensor_parallel_size=1, data_parallel_size=1,
        enable_expert_parallel=True,
    )
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(
            world_size, rank, "env://", local_rank, "nccl"
        )

    pcfg = ParallelConfig(
        tensor_parallel_size=tp_size,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        enable_expert_parallel=True,
    )
    vllm_cfg = VllmConfig(parallel_config=pcfg)

    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, backend="nccl"
        )

    from vllm.distributed import prepare_communication_buffer_for_model
    from vllm.forward_context import (
        DPMetadata, ForwardContext, create_forward_context,
        override_forward_context, set_forward_context,
    )
    from dinfer.model import LLaDA2MoeModelLM
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeSparseMoeBlock
    from transformers import AutoConfig
    from baseline_optimizations import apply_all_optimizations

    if rank == 0:
        print("=" * 70)
        print(f"CUDA Graph MoE POC — dp={dp_size} tp={tp_size} ep={world_size}")
        print(f"  batch={args.batch_size}, block={BLOCK_LENGTH}")
        print(f"  warmup={args.num_warmup}, iters={args.num_iters}")
        print("=" * 70)

    # ---- load model (inside vllm config context, same as bench script) ----
    with set_current_vllm_config(vllm_cfg):
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        )
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        with torch.inference_mode():
            warmup_tok = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(
                attn_metadata=None, vllm_config=vllm_cfg,
                num_tokens=warmup_tok.numel(),
            ):
                _ = model(warmup_tok, use_cache=False)

        apply_all_optimizations(model)
        prepare_communication_buffer_for_model(model)

    if rank == 0:
        print("  Model loaded and optimized.")

    # ---- identify MoE blocks ----
    moe_blocks = [
        m for _n, m in model.named_modules()
        if isinstance(m, LLaDA2MoeSparseMoeBlock)
    ]
    if rank == 0:
        print(f"  Found {len(moe_blocks)} MoE blocks.")

    # ---- construct static ForwardContext for decode phase ----
    # In decode, each DP rank has N = local_bs * block_length tokens.
    local_bs = args.batch_size // dp_size
    N_per_dp = local_bs * BLOCK_LENGTH
    num_tokens_across_dp_cpu = torch.zeros(dp_size, dtype=torch.int64)
    for r in range(dp_size):
        num_tokens_across_dp_cpu[r] = N_per_dp
    max_tokens_across_dp_cpu = torch.tensor(N_per_dp, dtype=torch.int64)
    static_dp_metadata = DPMetadata(max_tokens_across_dp_cpu,
                                     num_tokens_across_dp_cpu)
    static_fwd_ctx = create_forward_context(
        attn_metadata=None,
        vllm_config=vllm_cfg,
        virtual_engine=0,
        dp_metadata=static_dp_metadata,
    )

    if rank == 0:
        print(f"  Static ForwardContext: N_per_dp={N_per_dp}, dp_size={dp_size}")

    # ---- create synthetic MoE inputs ----
    # Use realistic shapes: hidden_states=[N_per_dp, hidden], router_logits=[N_per_dp, E_global]
    hidden_size = config.hidden_size  # 2048
    num_experts = config.num_experts  # 256
    hs_input = torch.randn(N_per_dp, hidden_size, dtype=torch.bfloat16, device=device)
    logits_input = torch.randn(N_per_dp, num_experts, dtype=torch.bfloat16, device=device)

    if rank == 0:
        print(f"  Synthetic inputs: hs={list(hs_input.shape)}, logits={list(logits_input.shape)}")

    # Pick a representative MoE layer (middle of network)
    test_layer_idx = len(moe_blocks) // 2
    experts_module = moe_blocks[test_layer_idx].experts

    results = {
        "rank": rank,
        "dp_size": dp_size,
        "tp_size": tp_size,
        "batch_size": args.batch_size,
        "N_per_dp": N_per_dp,
        "hidden_size": hidden_size,
        "num_experts": num_experts,
        "test_layer_idx": test_layer_idx,
        "num_moe_layers": len(moe_blocks),
    }

    with torch.inference_mode():

        # ==== G1: Eager MoE forward_impl timing ====
        if rank == 0:
            print(f"\n{'='*70}")
            print("G1: Eager MoE forward_impl timing (single layer)")
            print(f"{'='*70}")

        with override_forward_context(static_fwd_ctx):
            for _ in range(args.num_warmup):
                _ = experts_module.forward_impl(
                    hidden_states=hs_input.clone(), router_logits=logits_input.clone())
            torch.cuda.synchronize()

            eager_times = []
            for i in range(args.num_iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                eager_out = experts_module.forward_impl(
                    hidden_states=hs_input.clone(), router_logits=logits_input.clone())
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000
                eager_times.append(dt)

        eager_median = sorted(eager_times)[len(eager_times) // 2]
        eager_mean = sum(eager_times) / len(eager_times)
        if rank == 0:
            print(f"  Eager single-layer: median={eager_median:.3f} ms, "
                  f"mean={eager_mean:.3f} ms")
            print(f"  All times: {[f'{t:.2f}' for t in eager_times]}")

        results["g1_eager_median_ms"] = eager_median
        results["g1_eager_mean_ms"] = eager_mean
        results["g1_eager_times_ms"] = eager_times

        # Save eager output for correctness check
        with override_forward_context(static_fwd_ctx):
            eager_ref = experts_module.forward_impl(
                hidden_states=hs_input.clone(), router_logits=logits_input.clone())
        if isinstance(eager_ref, tuple):
            eager_ref = eager_ref[0]
        eager_ref = eager_ref.clone()

        # ==== G2: CUDA Graph capture attempt ====
        if rank == 0:
            print(f"\n{'='*70}")
            print("G2: CUDA Graph capture attempt (full forward_impl)")
            print(f"{'='*70}")

        graph_ok = False
        graph = None
        static_hs = None
        static_logits = None
        static_out = None

        try:
            static_hs = hs_input.clone()
            static_logits = logits_input.clone()

            # Warmup on the capture stream (required by CUDA Graph API)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with override_forward_context(static_fwd_ctx):
                    for _ in range(3):
                        _out = experts_module.forward_impl(
                            hidden_states=static_hs,
                            router_logits=static_logits)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            if rank == 0:
                print("  Stream warmup OK. Starting capture...")

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                with override_forward_context(static_fwd_ctx):
                    static_out = experts_module.forward_impl(
                        hidden_states=static_hs,
                        router_logits=static_logits)

            torch.cuda.synchronize()
            graph_ok = True
            if rank == 0:
                print("  CUDA Graph capture SUCCEEDED!")
                if isinstance(static_out, tuple):
                    print(f"  Output shape: {list(static_out[0].shape)}")
                else:
                    print(f"  Output shape: {list(static_out.shape)}")

        except Exception as e:
            if rank == 0:
                print(f"  CUDA Graph capture FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()

        results["g2_capture_ok"] = graph_ok

        if not graph_ok:
            # G2 failed — report and exit
            results["conclusion"] = "CUDA Graph capture of MoE forward_impl failed"
            _save_and_exit(results, args, rank)
            return

        # ==== G3: Correctness + graph replay timing ====
        if rank == 0:
            print(f"\n{'='*70}")
            print("G3: CUDA Graph correctness + replay timing")
            print(f"{'='*70}")

        # Correctness check: replay with same input, compare to eager
        static_hs.copy_(hs_input)
        static_logits.copy_(logits_input)
        graph.replay()
        torch.cuda.synchronize()

        graph_out = static_out[0].clone() if isinstance(static_out, tuple) else static_out.clone()
        cos_sim = torch.nn.functional.cosine_similarity(
            eager_ref.float().flatten(), graph_out.float().flatten(), dim=0
        ).item()
        abs_diff = (eager_ref.float() - graph_out.float()).abs().mean().item()

        if rank == 0:
            print(f"  Correctness: cosine_sim={cos_sim:.6f}, abs_mean_diff={abs_diff:.6e}")

        results["g3_cosine_sim"] = cos_sim
        results["g3_abs_mean_diff"] = abs_diff

        # Replay timing
        for _ in range(args.num_warmup):
            static_hs.copy_(hs_input)
            static_logits.copy_(logits_input)
            graph.replay()
        torch.cuda.synchronize()

        graph_times = []
        for i in range(args.num_iters):
            static_hs.copy_(hs_input)
            static_logits.copy_(logits_input)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            graph_times.append(dt)

        graph_median = sorted(graph_times)[len(graph_times) // 2]
        graph_mean = sum(graph_times) / len(graph_times)
        speedup = eager_median / graph_median if graph_median > 0 else float('inf')

        if rank == 0:
            print(f"  Graph single-layer: median={graph_median:.3f} ms, "
                  f"mean={graph_mean:.3f} ms")
            print(f"  Speedup: {speedup:.2f}x (eager/graph)")
            print(f"  Saved per layer: {eager_median - graph_median:.3f} ms")
            print(f"  All times: {[f'{t:.2f}' for t in graph_times]}")

        results["g3_graph_median_ms"] = graph_median
        results["g3_graph_mean_ms"] = graph_mean
        results["g3_speedup"] = speedup
        results["g3_graph_times_ms"] = graph_times

        # ==== G4: 19-layer sequential estimate ====
        if rank == 0:
            print(f"\n{'='*70}")
            print("G4: 19-layer sequential estimate")
            print(f"{'='*70}")

        num_layers = len(moe_blocks)

        # G4a: Eager 19 layers (each with same input for simplicity)
        with override_forward_context(static_fwd_ctx):
            for _ in range(2):
                for blk in moe_blocks:
                    _ = blk.experts.forward_impl(
                        hidden_states=hs_input.clone(),
                        router_logits=logits_input.clone())
            torch.cuda.synchronize()

        eager_19_times = []
        for i in range(args.num_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with override_forward_context(static_fwd_ctx):
                for blk in moe_blocks:
                    _ = blk.experts.forward_impl(
                        hidden_states=hs_input.clone(),
                        router_logits=logits_input.clone())
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            eager_19_times.append(dt)

        eager_19_median = sorted(eager_19_times)[len(eager_19_times) // 2]

        # G4b: Capture graph for each MoE layer, then replay all 19
        if rank == 0:
            print("  Capturing 19 CUDA Graphs (one per MoE layer)...")

        layer_graphs = []
        layer_static_hs = []
        layer_static_logits = []
        layer_static_out = []
        all_captured = True

        for li, blk in enumerate(moe_blocks):
            try:
                s_hs = hs_input.clone()
                s_lg = logits_input.clone()

                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    with override_forward_context(static_fwd_ctx):
                        for _ in range(3):
                            _ = blk.experts.forward_impl(
                                hidden_states=s_hs, router_logits=s_lg)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()

                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    with override_forward_context(static_fwd_ctx):
                        s_out = blk.experts.forward_impl(
                            hidden_states=s_hs, router_logits=s_lg)
                torch.cuda.synchronize()

                layer_graphs.append(g)
                layer_static_hs.append(s_hs)
                layer_static_logits.append(s_lg)
                layer_static_out.append(s_out)
            except Exception as e:
                if rank == 0:
                    print(f"  Layer {li} capture FAILED: {type(e).__name__}: {e}")
                all_captured = False
                break

        results["g4_all_19_captured"] = all_captured

        if all_captured:
            if rank == 0:
                print(f"  All {num_layers} graphs captured OK.")

            # Warmup replay
            for _ in range(2):
                for li in range(num_layers):
                    layer_static_hs[li].copy_(hs_input)
                    layer_static_logits[li].copy_(logits_input)
                    layer_graphs[li].replay()
            torch.cuda.synchronize()

            graph_19_times = []
            for i in range(args.num_iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for li in range(num_layers):
                    layer_static_hs[li].copy_(hs_input)
                    layer_static_logits[li].copy_(logits_input)
                    layer_graphs[li].replay()
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000
                graph_19_times.append(dt)

            graph_19_median = sorted(graph_19_times)[len(graph_19_times) // 2]
            saved_19 = eager_19_median - graph_19_median
            speedup_19 = eager_19_median / graph_19_median if graph_19_median > 0 else 0

            if rank == 0:
                print(f"\n  Eager 19-layer: median={eager_19_median:.3f} ms")
                print(f"  Graph 19-layer: median={graph_19_median:.3f} ms")
                print(f"  Saved: {saved_19:.3f} ms ({saved_19/eager_19_median*100:.1f}%)")
                print(f"  Speedup: {speedup_19:.2f}x")
                print(f"  Eager times: {[f'{t:.1f}' for t in eager_19_times]}")
                print(f"  Graph times: {[f'{t:.1f}' for t in graph_19_times]}")

            results["g4_eager_19_median_ms"] = eager_19_median
            results["g4_graph_19_median_ms"] = graph_19_median
            results["g4_saved_ms"] = saved_19
            results["g4_speedup"] = speedup_19
            results["g4_eager_19_times_ms"] = eager_19_times
            results["g4_graph_19_times_ms"] = graph_19_times

            # Extrapolate to e2e
            gs_baseline = 76.4  # ms/fwd from our measurements
            e2e_saved_pct = saved_19 / gs_baseline * 100

            if rank == 0:
                print(f"\n  E2E estimate: {saved_19:.1f} ms / {gs_baseline:.1f} ms = "
                      f"{e2e_saved_pct:.1f}% e2e improvement")
        else:
            if rank == 0:
                print(f"  Skipping 19-layer graph replay (not all captured).")
            eager_19_median_only = eager_19_median
            results["g4_eager_19_median_ms"] = eager_19_median_only

    # ==== Summary ====
    if rank == 0:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  Capture success: {graph_ok}")
        if graph_ok:
            print(f"  Correctness: cosine_sim={cos_sim:.6f}")
            print(f"  Single layer: eager={eager_median:.3f} ms, graph={graph_median:.3f} ms, "
                  f"speedup={speedup:.2f}x")
            if all_captured:
                print(f"  19 layers: eager={eager_19_median:.3f} ms, graph={graph_19_median:.3f} ms, "
                      f"saved={saved_19:.3f} ms ({e2e_saved_pct:.1f}% e2e)")

    _save_and_exit(results, args, rank)


def _save_and_exit(results, args, rank):
    if rank == 0:
        suffix = f"_{args.results_suffix}" if args.results_suffix else ""
        out_path = os.path.join(
            os.path.dirname(__file__), "..", "results",
            f"poc_cudagraph_moe{suffix}.json"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {out_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
