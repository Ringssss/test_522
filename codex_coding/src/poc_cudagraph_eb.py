#!/usr/bin/env python3
"""
POC: Single CUDA Graph + EB routing (K=4).

Extends poc_cudagraph_single_graph.py with EB controller + fused_routing K=4,
matching the GS production path. Captures graph during hot_skip steady state.

Groups:
  G1  Eager forward timing (EB K=4, no graph)
  G2  CUDA Graph capture (in hot_skip state)
  G3  Graph replay timing + correctness

Usage:
  torchrun --nproc_per_node=8 poc_cudagraph_eb.py [--batch-size 512]
"""

import argparse
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
MASK_ID = 156895
BLOCK_LENGTH = 32

# Pre-import EB kernels (same as bench script)
from test_m_skip_sweep import MSkipEBController
from test_fused_eb_triton import _kernel_A, _kernel_B_v3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--tp-size", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=320)
    p.add_argument("--num-warmup", type=int, default=3)
    p.add_argument("--num-iters", type=int, default=10)
    p.add_argument("--results-suffix", type=str, default="")
    return p.parse_args()


class SimpleEBController(MSkipEBController):
    """Minimal EB controller with block-id clock and path tracking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_block_id = -1
        self._last_block_id = {}
        self.path_counts = {"cold": 0, "hot_skip": 0, "hot_update": 0}

    def note_block_start(self, block_id: int):
        self.current_block_id = int(block_id)

    def reset(self):
        self.current_block_id = -1
        self._last_block_id.clear()
        self.path_counts = {"cold": 0, "hot_skip": 0, "hot_update": 0}
        self.eb_calls = 0
        self.eb_skips = 0
        self._bufs.clear()
        self.k_init_history.clear()
        self.s_mask_cache.clear()
        self.pop_cache.clear()
        self._fwd_in_block.clear()
        self._block_idx.clear()

    def get_s_mask(self, layer_idx, logits, bias):
        prev = self._last_block_id.get(layer_idx, -1)
        if prev != self.current_block_id:
            self._last_block_id[layer_idx] = self.current_block_id
            self.path_counts["cold"] += 1
            return self.cold_path(layer_idx, logits, bias)

        prev_calls = self.eb_calls
        prev_skips = self.eb_skips
        out = self.hot_path(layer_idx, logits, bias)
        if self.eb_calls > prev_calls:
            self.path_counts["hot_update"] += 1
        elif self.eb_skips > prev_skips:
            self.path_counts["hot_skip"] += 1
        return out


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
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeSparseMoeBlock
    from dinfer.decoding.utils import KVCache
    from transformers import AutoConfig
    from baseline_optimizations import apply_all_optimizations
    from test_fused_eb_triton import fused_routing

    if rank == 0:
        print("=" * 70)
        print(f"CUDA Graph + EB POC — dp={dp_size} tp={tp_size} ep={world_size}")
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

    # ---- EB controller ----
    ctrl = SimpleEBController(
        num_layers=19, K=8, M=4, K_target=40,
        quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5,
    )

    # ---- patch gate routing with EB + K=4 ----
    moe_blocks = [
        m for _, m in model.named_modules()
        if isinstance(m, LLaDA2MoeSparseMoeBlock)
    ]
    gi = 0
    for _n, m in model.named_modules():
        if m.__class__.__name__ == "LLaDA2MoeGate":
            b = m.expert_bias
            r = m.routed_scaling_factor
            ng = m.n_group
            tkg = m.topk_group
            li = gi

            def mk(bb, rr, nn, gg, layer_i, cc):
                def fn(hs, go, topk, renorm):
                    sm = cc.get_s_mask(layer_i, go, bb)
                    w, idx = fused_routing(go, bb, rr, s_mask=sm, K=4, ng=nn, tkg=gg)
                    return w.to(go.dtype), idx
                return fn

            m.routing = mk(b, r, ng, tkg, li, ctrl)
            gi += 1

    if rank == 0:
        print(f"  EB routing patched: {gi} gates, K=4")

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
        print(f"  KV cache: {list(kv_data.shape)}, {kv_gb:.2f} GB")

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
        "max_kv": max_kv, "num_layers": num_layers, "eb_K": 4,
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
        _graph_mode[0] = True

        # ---- Initialize EB: run cold forwards to set up s_mask ----
        if rank == 0:
            print(f"\n  Initializing EB (cold path)...")
        ctrl.note_block_start(0)
        with override_forward_context(static_fwd_ctx):
            for i in range(3):
                _ = do_forward()
        torch.cuda.synchronize()
        if rank == 0:
            print(f"  EB paths after init: {ctrl.path_counts}")
            all_cached = all(li in ctrl.s_mask_cache for li in range(19))
            print(f"  All 19 s_mask cached: {all_cached}")

        # ==== G1: Eager forward timing (EB K=4, hot_skip) ====
        if rank == 0:
            print(f"\n{'='*70}")
            print("G1: Eager forward timing (EB K=4, hot_skip)")
            print(f"{'='*70}")

        with override_forward_context(static_fwd_ctx):
            for _ in range(args.num_warmup):
                static_input_ids.random_(0, config.vocab_size)
                _ = do_forward()
            torch.cuda.synchronize()

            eager_times = []
            for i in range(args.num_iters):
                static_input_ids.random_(0, config.vocab_size)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = do_forward()
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000
                eager_times.append(dt)

        eager_ref_slice = out.logits[:4, :, :1000].clone()
        eager_median = sorted(eager_times)[len(eager_times) // 2]
        if rank == 0:
            print(f"  Eager: median={eager_median:.3f} ms")
            print(f"  Times: {[f'{t:.1f}' for t in eager_times]}")
            print(f"  EB paths total: {ctrl.path_counts}")
        results["g1_eager_median_ms"] = eager_median
        results["g1_eager_times_ms"] = eager_times
        results["g1_eb_paths"] = dict(ctrl.path_counts)

        # ==== G2: CUDA Graph capture (in hot_skip state) ====
        if rank == 0:
            print(f"\n{'='*70}")
            print("G2: CUDA Graph capture (hot_skip state)")
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
                print(f"  Capture SUCCEEDED!")
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

        # Correctness with same input as last eager
        with override_forward_context(static_fwd_ctx):
            graph.replay()
        torch.cuda.synchronize()
        cap_slice = capture_logits[:4, :, :1000].clone()
        cos_same = torch.nn.functional.cosine_similarity(
            eager_ref_slice.float().flatten(),
            cap_slice.float().flatten(), dim=0,
        ).item()
        del cap_slice
        if rank == 0:
            print(f"  Same-input cosine_sim: {cos_same:.6f}")
        results["g3_cosine_same"] = cos_same

        # Correctness with new input
        new_ids = torch.randint(0, config.vocab_size, (batch, BLOCK_LENGTH),
                                 dtype=torch.long, device=device)
        static_input_ids.copy_(new_ids)
        with override_forward_context(static_fwd_ctx):
            graph.replay()
        torch.cuda.synchronize()
        graph_new_slice = capture_logits[:4, :, :1000].clone()

        # Run eager with same new input
        with override_forward_context(static_fwd_ctx):
            eager_new = do_forward()
        eager_new_slice = eager_new.logits[:4, :, :1000].clone()
        cos_new = torch.nn.functional.cosine_similarity(
            eager_new_slice.float().flatten(),
            graph_new_slice.float().flatten(), dim=0,
        ).item()
        del graph_new_slice, eager_new_slice
        if rank == 0:
            print(f"  New-input cosine_sim: {cos_new:.6f}")
        results["g3_cosine_new"] = cos_new

        # Replay timing
        with override_forward_context(static_fwd_ctx):
            for _ in range(args.num_warmup):
                static_input_ids.random_(0, config.vocab_size)
                graph.replay()
            torch.cuda.synchronize()
            graph_times = []
            for i in range(args.num_iters):
                static_input_ids.random_(0, config.vocab_size)
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

    # ==== Summary ====
    if rank == 0:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        gs_baseline = 76.4
        print(f"  EB K=4: {'YES'}")
        print(f"  Capture: {'OK' if graph_ok else 'FAILED'}")
        print(f"  Correctness: same={cos_same:.6f}, new={cos_new:.6f}")
        print(f"  Eager:  {eager_median:.3f} ms/fwd")
        print(f"  Graph:  {graph_median:.3f} ms/fwd")
        print(f"  Saved:  {saved:.3f} ms ({saved/eager_median*100:.1f}%)")
        print(f"  vs GS baseline ({gs_baseline} ms): "
              f"{(gs_baseline-graph_median)/gs_baseline*100:+.1f}% if applied to e2e")

    KVCache.update = _orig_kv_update
    fwd_ctx_module.set_forward_context = _orig_set_fwd_ctx
    _save_and_exit(results, args, rank)


def _save_and_exit(results, args, rank):
    if rank == 0:
        suffix = f"_{args.results_suffix}" if args.results_suffix else ""
        out_path = os.path.join(
            os.path.dirname(__file__), "..", "results",
            f"poc_cudagraph_eb{suffix}.json"
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
