#!/usr/bin/env python3
"""
POC: Single CUDA Graph for all blocks — pre-allocated KV cache + index_copy + SDPA mask.

Strategy:
  - Pre-allocate KV cache to max_seq_len (fixed tensor address)
  - Replace slice-based KV write with index_copy (data-driven position)
  - Pass attention_mask to trigger SDPA path (mask out invalid KV positions)
  - Capture ONE graph for input shape [batch, 32], replay for all blocks

Groups:
  G1  Eager forward timing (with patches, no graph)
  G2  CUDA Graph capture
  G3  Graph replay timing + correctness
  G4  Multi-block simulation (update write_idx/pos/mask between replays)

Usage:
  torchrun --nproc_per_node=8 poc_cudagraph_single_graph.py [--batch-size 512]
"""

import argparse
import json
import os
import time
import traceback

import torch
import torch.distributed as dist

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
MASK_ID = 126336
BLOCK_LENGTH = 32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--tp-size", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=320,
                    help="Max KV cache length (prompt+gen aligned to block)")
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
    batch = args.batch_size
    max_kv = args.max_seq_len

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
        tensor_parallel_size=tp_size, data_parallel_size=dp_size,
        data_parallel_rank=dp_rank, enable_expert_parallel=True,
    )
    vllm_cfg = VllmConfig(parallel_config=pcfg)
    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(tp_size, backend="nccl")

    from vllm.distributed import prepare_communication_buffer_for_model
    from vllm.forward_context import (
        DPMetadata, create_forward_context, override_forward_context,
        set_forward_context,
    )
    import vllm.forward_context as fwd_ctx_module
    from dinfer.model import LLaDA2MoeModelLM
    from dinfer.decoding.utils import KVCache
    from transformers import AutoConfig
    from baseline_optimizations import apply_all_optimizations
    from contextlib import contextmanager

    if rank == 0:
        print("=" * 70)
        print(f"Single CUDA Graph POC — dp={dp_size} tp={tp_size} ep={world_size}")
        print(f"  batch={batch}, max_kv={max_kv}, block={BLOCK_LENGTH}")
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
            w = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                      num_tokens=w.numel()):
                _ = model(w, use_cache=False)
        apply_all_optimizations(model)
        prepare_communication_buffer_for_model(model)
        model.set_bsp_sequence_parallel_moe(True)

    if rank == 0:
        print(f"  Model loaded. GPU mem: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

    # ---- model config ----
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads // tp_size
    head_dim = config.hidden_size // config.num_attention_heads

    # ---- pre-allocate KV cache ----
    kv_data = torch.zeros(
        num_layers, 2, batch, num_kv_heads, max_kv, head_dim,
        dtype=torch.bfloat16, device=device,
    )
    kv_cache = KVCache.__new__(KVCache)
    kv_cache._data = kv_data
    kv_cache.use_inplace_update = True
    kv_cache.backend = 'vllm'

    if rank == 0:
        kv_gb = kv_data.nelement() * 2 / 1e9
        print(f"  KV cache pre-allocated: {list(kv_data.shape)}, {kv_gb:.2f} GB")

    # ---- static buffers ----
    static_input_ids = torch.randint(0, config.vocab_size, (batch, BLOCK_LENGTH),
                                      dtype=torch.long, device=device)
    static_pos_ids = torch.arange(BLOCK_LENGTH, dtype=torch.long, device=device
                                   ).unsqueeze(0).expand(batch, -1).contiguous()
    write_idx = torch.arange(BLOCK_LENGTH, dtype=torch.long, device=device)
    static_attn_mask = torch.zeros(batch, 1, BLOCK_LENGTH, max_kv,
                                    dtype=torch.bool, device=device)
    static_attn_mask[:, :, :, :BLOCK_LENGTH] = True

    # ---- monkey-patch KVCache.update for index_copy ----
    _orig_kv_update = KVCache.update

    def _patched_kv_update(self, key_states, val_states, layer_idx,
                            replace_position=None, backend='vllm'):
        self._data[layer_idx, 0, :, :, write_idx, :] = key_states
        self._data[layer_idx, 1, :, :, write_idx, :] = val_states
        return self._data[layer_idx, 0], self._data[layer_idx, 1]

    KVCache.update = _patched_kv_update

    # ---- monkey-patch set_forward_context to no-op ----
    N_tokens = batch * BLOCK_LENGTH
    num_tokens_across_dp = torch.full((dp_size,), N_tokens, dtype=torch.int64)
    static_dp_metadata = DPMetadata(torch.tensor(N_tokens, dtype=torch.int64),
                                     num_tokens_across_dp)
    static_fwd_ctx = create_forward_context(
        attn_metadata=None, vllm_config=vllm_cfg,
        virtual_engine=0, dp_metadata=static_dp_metadata,
    )
    _orig_set_fwd_ctx = fwd_ctx_module.set_forward_context
    _graph_mode = [False]

    @contextmanager
    def _noop_ctx(**kw):
        yield

    def _patched_set_fwd_ctx(*a, **kw):
        if _graph_mode[0]:
            return _noop_ctx()
        return _orig_set_fwd_ctx(*a, **kw)

    fwd_ctx_module.set_forward_context = _patched_set_fwd_ctx

    results = {
        "rank": rank, "batch": batch, "tp_size": tp_size, "dp_size": dp_size,
        "max_kv": max_kv, "num_layers": num_layers,
    }

    def do_forward():
        return model(
            static_input_ids, use_cache=True,
            past_key_values=kv_cache,
            position_ids=static_pos_ids,
            replace_position=(0, BLOCK_LENGTH),
            attention_mask=static_attn_mask,
        )

    with torch.inference_mode():

        # ==== G1: Eager forward timing (with patches) ====
        if rank == 0:
            print(f"\n{'='*70}")
            print("G1: Eager forward timing (patched, no graph)")
            print(f"{'='*70}")

        _graph_mode[0] = True
        with override_forward_context(static_fwd_ctx):
            for _ in range(args.num_warmup):
                _ = do_forward()
            torch.cuda.synchronize()

            eager_times = []
            for i in range(args.num_iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = do_forward()
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000
                eager_times.append(dt)

        eager_ref = out.logits.clone()
        eager_median = sorted(eager_times)[len(eager_times) // 2]
        if rank == 0:
            print(f"  Eager: median={eager_median:.3f} ms, logits={list(eager_ref.shape)}")
            print(f"  Times: {[f'{t:.1f}' for t in eager_times]}")
        results["g1_eager_median_ms"] = eager_median
        results["g1_eager_times_ms"] = eager_times

        # ==== G2: CUDA Graph capture ====
        if rank == 0:
            print(f"\n{'='*70}")
            print("G2: CUDA Graph capture")
            print(f"{'='*70}")

        graph_ok = False
        graph = None
        capture_logits = None

        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                with override_forward_context(static_fwd_ctx):
                    for _ in range(3):
                        _ = do_forward()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            if rank == 0:
                print(f"  Warmup OK. GPU mem: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                with override_forward_context(static_fwd_ctx):
                    capture_out = do_forward()
            torch.cuda.synchronize()
            capture_logits = capture_out.logits
            graph_ok = True
            if rank == 0:
                print(f"  Capture SUCCEEDED! logits={list(capture_logits.shape)}")
                print(f"  GPU mem: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

        except Exception as e:
            if rank == 0:
                print(f"  Capture FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()

        results["g2_capture_ok"] = graph_ok
        if not graph_ok:
            _save_and_exit(results, args, rank)
            return

        # ==== G3: Graph replay timing + correctness ====
        if rank == 0:
            print(f"\n{'='*70}")
            print("G3: Graph replay timing + correctness")
            print(f"{'='*70}")

        with override_forward_context(static_fwd_ctx):
            graph.replay()
        torch.cuda.synchronize()
        # Compare a small slice to avoid OOM on full logits
        ref_slice = eager_ref[:4, :, :1000].float().flatten()
        cap_slice = capture_logits[:4, :, :1000].clone().float().flatten()
        cos_same = torch.nn.functional.cosine_similarity(
            ref_slice, cap_slice, dim=0,
        ).item()
        del ref_slice, cap_slice
        if rank == 0:
            print(f"  Same-input cosine_sim: {cos_same:.6f}")
        results["g3_cosine_same"] = cos_same

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
        saved = eager_median - graph_median
        if rank == 0:
            print(f"  Graph: median={graph_median:.3f} ms")
            print(f"  Saved: {saved:.3f} ms ({saved/eager_median*100:.1f}%)")
            print(f"  Times: {[f'{t:.1f}' for t in graph_times]}")
        results["g3_graph_median_ms"] = graph_median
        results["g3_saved_ms"] = saved
        results["g3_graph_times_ms"] = graph_times

        # ==== G4: Multi-block simulation ====
        if rank == 0:
            print(f"\n{'='*70}")
            print("G4: Multi-block simulation (change write_idx/pos/mask between replays)")
            print(f"{'='*70}")

        num_blocks = max_kv // BLOCK_LENGTH
        kv_data.zero_()

        # Eager multi-block
        with override_forward_context(static_fwd_ctx):
            for _ in range(2):
                kv_data.zero_()
                for blk in range(num_blocks):
                    bs = blk * BLOCK_LENGTH
                    be = bs + BLOCK_LENGTH
                    write_idx.copy_(torch.arange(BLOCK_LENGTH, device=device) + bs)
                    static_pos_ids.copy_(
                        torch.arange(bs, be, device=device).unsqueeze(0).expand(batch, -1))
                    static_attn_mask.zero_()
                    static_attn_mask[:, :, :, :be] = True
                    static_input_ids.random_(0, config.vocab_size)
                    _ = do_forward()
            torch.cuda.synchronize()

            eager_mb_times = []
            for _ in range(3):
                kv_data.zero_()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for blk in range(num_blocks):
                    bs = blk * BLOCK_LENGTH
                    be = bs + BLOCK_LENGTH
                    write_idx.copy_(torch.arange(BLOCK_LENGTH, device=device) + bs)
                    static_pos_ids.copy_(
                        torch.arange(bs, be, device=device).unsqueeze(0).expand(batch, -1))
                    static_attn_mask.zero_()
                    static_attn_mask[:, :, :, :be] = True
                    static_input_ids.random_(0, config.vocab_size)
                    _ = do_forward()
                torch.cuda.synchronize()
                eager_mb_times.append((time.perf_counter() - t0) * 1000)

        # Graph multi-block
        with override_forward_context(static_fwd_ctx):
            graph_mb_times = []
            for _ in range(3):
                kv_data.zero_()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for blk in range(num_blocks):
                    bs = blk * BLOCK_LENGTH
                    be = bs + BLOCK_LENGTH
                    write_idx.copy_(torch.arange(BLOCK_LENGTH, device=device) + bs)
                    static_pos_ids.copy_(
                        torch.arange(bs, be, device=device).unsqueeze(0).expand(batch, -1))
                    static_attn_mask.zero_()
                    static_attn_mask[:, :, :, :be] = True
                    static_input_ids.random_(0, config.vocab_size)
                    graph.replay()
                torch.cuda.synchronize()
                graph_mb_times.append((time.perf_counter() - t0) * 1000)

        eager_mb_med = sorted(eager_mb_times)[1]
        graph_mb_med = sorted(graph_mb_times)[1]
        mb_saved = eager_mb_med - graph_mb_med

        if rank == 0:
            print(f"  {num_blocks} blocks × 1 fwd each:")
            print(f"  Eager: {eager_mb_med:.1f} ms ({eager_mb_med/num_blocks:.2f} ms/fwd)")
            print(f"  Graph: {graph_mb_med:.1f} ms ({graph_mb_med/num_blocks:.2f} ms/fwd)")
            print(f"  Saved: {mb_saved:.1f} ms ({mb_saved/eager_mb_med*100:.1f}%)")

        results["g4_num_blocks"] = num_blocks
        results["g4_eager_mb_ms"] = eager_mb_times
        results["g4_graph_mb_ms"] = graph_mb_times
        results["g4_saved_ms"] = mb_saved

    # ==== Summary ====
    if rank == 0:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  Capture: {'OK' if graph_ok else 'FAILED'}")
        print(f"  Correctness: cosine_sim={cos_same:.6f}")
        print(f"  Single fwd: eager={eager_median:.3f} ms, graph={graph_median:.3f} ms, "
              f"saved={saved:.3f} ms ({saved/eager_median*100:.1f}%)")
        if graph_ok:
            print(f"  Multi-block: eager={eager_mb_med:.1f} ms, graph={graph_mb_med:.1f} ms, "
                  f"saved={mb_saved:.1f} ms ({mb_saved/eager_mb_med*100:.1f}%)")

    # Restore
    KVCache.update = _orig_kv_update
    fwd_ctx_module.set_forward_context = _orig_set_fwd_ctx

    _save_and_exit(results, args, rank)


def _save_and_exit(results, args, rank):
    if rank == 0:
        suffix = f"_{args.results_suffix}" if args.results_suffix else ""
        out_path = os.path.join(
            os.path.dirname(__file__), "..", "results",
            f"poc_cudagraph_single_graph{suffix}.json"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {out_path}")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
