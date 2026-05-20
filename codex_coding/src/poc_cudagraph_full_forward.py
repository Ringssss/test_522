#!/usr/bin/env python3
"""
POC: CUDA Graph capture of FULL model forward.

v0.1.15.17 showed MoE-only graph saves only ~1ms (2%).  The ~8.7ms framework
overhead is spread across the entire forward path. This POC tests whether the
full model(input_ids, attn_mask, pos_ids) call can be captured as a single
CUDA Graph.

Key challenge: set_forward_context() inside each MoE layer's forward calls
DPMetadata.make() which does NCCL all_reduce + .cpu() -- not graph-safe.
Solution: pre-create a static ForwardContext and patch set_forward_context
to be a no-op during capture/replay.

Groups:
  G1  Eager model forward timing (baseline)
  G2  CUDA Graph capture attempt
  G3  Graph replay timing + correctness

Usage:
  torchrun --nproc_per_node=8 poc_cudagraph_full_forward.py [--batch-size 512]
"""

import argparse
import json
import os
import time
import traceback
from contextlib import contextmanager

import torch
import torch.distributed as dist

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
BLOCK_LENGTH = 32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--tp-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=64,
                    help="Sequence length for capture (simulates 2-block decode)")
    p.add_argument("--num-warmup", type=int, default=3)
    p.add_argument("--num-iters", type=int, default=10)
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

    # ---- init distributed ----
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
        DPMetadata, create_forward_context, override_forward_context,
        set_forward_context,
    )
    import vllm.forward_context as fwd_ctx_module
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoConfig
    from baseline_optimizations import apply_all_optimizations

    if rank == 0:
        print("=" * 70)
        print(f"CUDA Graph Full Forward POC — dp={dp_size} tp={tp_size} ep={world_size}")
        print(f"  batch={args.batch_size}, seq_len={args.seq_len}")
        print(f"  warmup={args.num_warmup}, iters={args.num_iters}")
        print("=" * 70)

    # ---- load model ----
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

    # ---- construct static ForwardContext ----
    batch_size = args.batch_size
    seq_len = args.seq_len
    N_tokens = batch_size * seq_len
    num_tokens_across_dp_cpu = torch.full((dp_size,), N_tokens, dtype=torch.int64)
    max_tokens_across_dp_cpu = torch.tensor(N_tokens, dtype=torch.int64)
    static_dp_metadata = DPMetadata(max_tokens_across_dp_cpu,
                                     num_tokens_across_dp_cpu)
    static_fwd_ctx = create_forward_context(
        attn_metadata=None,
        vllm_config=vllm_cfg,
        virtual_engine=0,
        dp_metadata=static_dp_metadata,
    )

    if rank == 0:
        print(f"  Static ForwardContext: N_tokens={N_tokens}, dp_size={dp_size}")

    # ---- build static inputs ----
    # Simulate a 2-block decode: input_ids, block-causal attn_mask, position_ids
    num_blocks = seq_len // BLOCK_LENGTH
    static_input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len),
        dtype=torch.long, device=device
    )
    block_mask = torch.tril(torch.ones(
        num_blocks, num_blocks, device=device
    ))
    static_attn_mask = (
        block_mask
        .repeat_interleave(BLOCK_LENGTH, dim=0)
        .repeat_interleave(BLOCK_LENGTH, dim=1)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
    )
    static_pos_ids = (
        torch.arange(seq_len, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    if rank == 0:
        print(f"  Static inputs: ids={list(static_input_ids.shape)}, "
              f"mask={list(static_attn_mask.shape)}, pos={list(static_pos_ids.shape)}")
        print(f"  GPU memory: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

    # ---- monkey-patch set_forward_context to no-op during graph mode ----
    _orig_set_forward_context = fwd_ctx_module.set_forward_context
    _graph_mode = False

    @contextmanager
    def _noop_set_forward_context(**kwargs):
        yield

    def _patched_set_forward_context(*args, **kwargs):
        if _graph_mode:
            return _noop_set_forward_context(**kwargs)
        return _orig_set_forward_context(*args, **kwargs)

    results = {
        "rank": rank,
        "dp_size": dp_size,
        "tp_size": tp_size,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "N_tokens": N_tokens,
    }

    with torch.inference_mode():

        # ==== G1: Eager model forward timing ====
        if rank == 0:
            print(f"\n{'='*70}")
            print("G1: Eager model forward timing")
            print(f"{'='*70}")

        with set_current_vllm_config(vllm_cfg):
            # warmup
            for _ in range(args.num_warmup):
                with set_forward_context(
                    attn_metadata=None, vllm_config=vllm_cfg,
                    num_tokens=N_tokens,
                ):
                    _ = model(static_input_ids, use_cache=False,
                              attention_mask=static_attn_mask,
                              position_ids=static_pos_ids)
            torch.cuda.synchronize()

            eager_times = []
            for i in range(args.num_iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with set_forward_context(
                    attn_metadata=None, vllm_config=vllm_cfg,
                    num_tokens=N_tokens,
                ):
                    eager_out = model(static_input_ids, use_cache=False,
                                      attention_mask=static_attn_mask,
                                      position_ids=static_pos_ids)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000
                eager_times.append(dt)

        eager_ref_logits = eager_out.logits.clone()
        eager_median = sorted(eager_times)[len(eager_times) // 2]
        eager_mean = sum(eager_times) / len(eager_times)

        if rank == 0:
            print(f"  Eager: median={eager_median:.3f} ms, mean={eager_mean:.3f} ms")
            print(f"  Logits shape: {list(eager_ref_logits.shape)}")
            print(f"  Times: {[f'{t:.1f}' for t in eager_times]}")

        results["g1_eager_median_ms"] = eager_median
        results["g1_eager_mean_ms"] = eager_mean
        results["g1_eager_times_ms"] = eager_times

        # ==== G2: CUDA Graph capture attempt ====
        if rank == 0:
            print(f"\n{'='*70}")
            print("G2: CUDA Graph capture attempt (full model forward)")
            print(f"{'='*70}")

        graph_ok = False
        graph = None
        capture_logits = None

        try:
            # Enable graph mode: patch set_forward_context to no-op
            fwd_ctx_module.set_forward_context = _patched_set_forward_context
            _graph_mode = True

            # Pre-set the static forward context
            # Warmup on side stream
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with set_current_vllm_config(vllm_cfg):
                    with override_forward_context(static_fwd_ctx):
                        for _ in range(3):
                            _ = model(static_input_ids, use_cache=False,
                                      attention_mask=static_attn_mask,
                                      position_ids=static_pos_ids)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            if rank == 0:
                print("  Stream warmup OK. Starting capture...")
                print(f"  GPU memory before capture: "
                      f"{torch.cuda.memory_allocated(device)/1e9:.1f} GB")

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                with set_current_vllm_config(vllm_cfg):
                    with override_forward_context(static_fwd_ctx):
                        capture_out = model(
                            static_input_ids, use_cache=False,
                            attention_mask=static_attn_mask,
                            position_ids=static_pos_ids)

            torch.cuda.synchronize()
            capture_logits = capture_out.logits
            graph_ok = True

            if rank == 0:
                print("  CUDA Graph capture SUCCEEDED!")
                print(f"  Capture logits shape: {list(capture_logits.shape)}")
                print(f"  GPU memory after capture: "
                      f"{torch.cuda.memory_allocated(device)/1e9:.1f} GB")

        except Exception as e:
            if rank == 0:
                print(f"  CUDA Graph capture FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
        finally:
            fwd_ctx_module.set_forward_context = _orig_set_forward_context
            _graph_mode = False

        results["g2_capture_ok"] = graph_ok

        if not graph_ok:
            results["conclusion"] = "Full model forward CUDA Graph capture failed"
            _save_and_exit(results, args, rank)
            return

        # ==== G3: Correctness + replay timing ====
        if rank == 0:
            print(f"\n{'='*70}")
            print("G3: CUDA Graph correctness + replay timing")
            print(f"{'='*70}")

        # Correctness: replay with same input, compare to eager
        fwd_ctx_module.set_forward_context = _patched_set_forward_context
        _graph_mode = True

        try:
            with override_forward_context(static_fwd_ctx):
                graph.replay()
            torch.cuda.synchronize()

            graph_logits = capture_logits.clone()
            cos_sim = torch.nn.functional.cosine_similarity(
                eager_ref_logits.float().flatten(),
                graph_logits.float().flatten(),
                dim=0,
            ).item()
            abs_diff = (eager_ref_logits.float() - graph_logits.float()).abs().mean().item()

            if rank == 0:
                print(f"  Correctness: cosine_sim={cos_sim:.6f}, "
                      f"abs_mean_diff={abs_diff:.6e}")

            results["g3_cosine_sim"] = cos_sim
            results["g3_abs_mean_diff"] = abs_diff

            # Replay timing
            with override_forward_context(static_fwd_ctx):
                for _ in range(args.num_warmup):
                    graph.replay()
                torch.cuda.synchronize()

                graph_times = []
                for i in range(args.num_iters):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    graph.replay()
                    torch.cuda.synchronize()
                    dt = (time.perf_counter() - t0) * 1000
                    graph_times.append(dt)

            graph_median = sorted(graph_times)[len(graph_times) // 2]
            graph_mean = sum(graph_times) / len(graph_times)
            saved = eager_median - graph_median
            speedup = eager_median / graph_median if graph_median > 0 else 0

            if rank == 0:
                print(f"  Graph: median={graph_median:.3f} ms, mean={graph_mean:.3f} ms")
                print(f"  Saved: {saved:.3f} ms ({saved/eager_median*100:.1f}%)")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Times: {[f'{t:.1f}' for t in graph_times]}")

            results["g3_graph_median_ms"] = graph_median
            results["g3_graph_mean_ms"] = graph_mean
            results["g3_saved_ms"] = saved
            results["g3_speedup"] = speedup
            results["g3_graph_times_ms"] = graph_times

            # Replay with DIFFERENT input to verify data flow
            if rank == 0:
                print("\n  Testing with different input data...")
            new_input = torch.randint(
                0, config.vocab_size, (batch_size, seq_len),
                dtype=torch.long, device=device
            )
            static_input_ids.copy_(new_input)
            with override_forward_context(static_fwd_ctx):
                graph.replay()
            torch.cuda.synchronize()
            new_graph_logits = capture_logits.clone()

            # Also run eager with same new input
            with set_current_vllm_config(vllm_cfg):
                fwd_ctx_module.set_forward_context = _orig_set_forward_context
                _graph_mode = False
                with set_forward_context(
                    attn_metadata=None, vllm_config=vllm_cfg,
                    num_tokens=N_tokens,
                ):
                    new_eager_out = model(new_input, use_cache=False,
                                          attention_mask=static_attn_mask,
                                          position_ids=static_pos_ids)
                fwd_ctx_module.set_forward_context = _patched_set_forward_context
                _graph_mode = True

            new_cos_sim = torch.nn.functional.cosine_similarity(
                new_eager_out.logits.float().flatten(),
                new_graph_logits.float().flatten(),
                dim=0,
            ).item()
            if rank == 0:
                print(f"  New-input correctness: cosine_sim={new_cos_sim:.6f}")
            results["g3_new_input_cosine_sim"] = new_cos_sim

        finally:
            fwd_ctx_module.set_forward_context = _orig_set_forward_context
            _graph_mode = False

    # ==== Summary ====
    if rank == 0:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  Capture: {'OK' if graph_ok else 'FAILED'}")
        if graph_ok:
            print(f"  Same-input cosine_sim: {cos_sim:.6f}")
            print(f"  New-input cosine_sim:  {new_cos_sim:.6f}")
            print(f"  Eager: {eager_median:.3f} ms")
            print(f"  Graph: {graph_median:.3f} ms")
            print(f"  Saved: {saved:.3f} ms ({saved/eager_median*100:.1f}%)")

    _save_and_exit(results, args, rank)


def _save_and_exit(results, args, rank):
    if rank == 0:
        suffix = f"_{args.results_suffix}" if args.results_suffix else ""
        out_path = os.path.join(
            os.path.dirname(__file__), "..", "results",
            f"poc_cudagraph_full_forward{suffix}.json"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {out_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
