#!/usr/bin/env python3
"""
C12 Component-Level Profiling: dispatch / routing / fused_experts / combine.

Instruments each MoE layer to measure communication vs compute breakdown,
with cold/hot path distinction (MSkipEB).

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter \
    torchrun --nproc_per_node=8 codex_coding/src/nsys_c12_profiling.py
"""

from __future__ import annotations
import os, sys, time, json, argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.distributed as dist

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"


# ---------- Timing infrastructure ----------
class CudaTimer:
    """Collects per-component CUDA event timings, organized by layer and path."""
    def __init__(self):
        self.records = defaultdict(lambda: defaultdict(list))
        # records[layer_id][(component, path)] = [ms, ms, ...]
        self._active = True

    def disable(self):
        self._active = False

    def enable(self):
        self._active = True

    def record(self, layer_id, component, path, start_event, end_event):
        if self._active:
            self.records[layer_id][(component, path)].append((start_event, end_event))

    def summarize(self):
        """Synchronize all events and compute ms per component."""
        torch.cuda.synchronize()
        summary = {}
        for layer_id, comps in sorted(self.records.items()):
            layer_data = {}
            for (comp, path), event_pairs in comps.items():
                times = [s.elapsed_time(e) for s, e in event_pairs]
                key = f"{comp}_{path}" if path else comp
                layer_data[key] = {
                    "count": len(times),
                    "total_ms": sum(times),
                    "avg_ms": sum(times) / len(times) if times else 0,
                }
            summary[layer_id] = layer_data
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gen-length", type=int, default=64)
    args = parser.parse_args()

    TP_SIZE = 4
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    assert world_size == 8, f"Requires 8 GPUs, got {world_size}"

    dp_size = world_size // TP_SIZE
    dp_rank = rank // TP_SIZE
    tp_rank_local = rank % TP_SIZE

    local_bs = args.batch_size // dp_size
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    alltoall_backend = os.environ.get("VLLM_ALL2ALL_BACKEND", "allgather_reducescatter")

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from dinfer.model.modeling_llada2_moe import (
        LLaDA2MoeSparseMoeBlock, _moe_forward_with_context)
    from transformers import AutoTokenizer, AutoConfig
    from test_heteval512 import PROMPTS
    from test_fused_eb_triton import fused_routing
    from test_m_skip_sweep import MSkipEBController
    from baseline_optimizations import apply_all_optimizations

    # --- Distributed init (same as bench_dp2_tp4_ep8.py) ---
    pcfg_init = ParallelConfig(
        tensor_parallel_size=1, data_parallel_size=1,
        enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(world_size, rank, "env://", local_rank, "nccl")

    pcfg = ParallelConfig(
        tensor_parallel_size=TP_SIZE, data_parallel_size=dp_size,
        data_parallel_rank=dp_rank, enable_expert_parallel=True)
    vllm_cfg = VllmConfig(parallel_config=pcfg)

    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(
            tensor_model_parallel_size=TP_SIZE, backend="nccl")

        from vllm.distributed import (get_tp_group, get_dp_group, get_ep_group)
        from vllm.forward_context import set_forward_context

        if rank == 0:
            print(f"C12 Component Profiling — 8 GPUs, dp=2 tp=4 ep=8")
            print(f"  batch={args.batch_size}, gen={args.gen_length}")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)

        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        with torch.inference_mode():
            warmup_tok = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                     num_tokens=warmup_tok.numel()):
                _ = model(warmup_tok, use_cache=False)

        apply_all_optimizations(model)

        from vllm.distributed import prepare_communication_buffer_for_model
        prepare_communication_buffer_for_model(model)

        # --- Timer ---
        timer = CudaTimer()

        # --- Patch MoE forward with instrumented version ---
        ctrl = MSkipEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)

        moe_blocks = []
        gate_params = []
        for name, mod in model.named_modules():
            if isinstance(mod, LLaDA2MoeSparseMoeBlock):
                moe_blocks.append(mod)
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                gate_params.append({
                    "bias": mod.expert_bias,
                    "rsf": mod.routed_scaling_factor,
                    "ng": mod.n_group,
                    "tkg": mod.topk_group,
                })

        def make_instrumented_forward(block, layer_id, gate_p, ctrl_ref, timer_ref):
            original_shared = block.shared_experts if block.config.num_shared_experts else None

            def instrumented_forward(hidden_states):
                bsz, seq_len, h = hidden_states.shape
                hs_flat = hidden_states.view(-1, h)

                is_hot = ctrl_ref.is_hot_check(layer_id)
                path = "hot" if is_hot else "cold"

                # --- Shared experts ---
                s0 = torch.cuda.Event(enable_timing=True)
                s1 = torch.cuda.Event(enable_timing=True)
                s0.record()
                shared_res = original_shared(hs_flat) if original_shared is not None else None
                s1.record()
                timer_ref.record(layer_id, "shared", path, s0, s1)

                # --- Gate logits ---
                s2 = torch.cuda.Event(enable_timing=True)
                s3 = torch.cuda.Event(enable_timing=True)
                s2.record()
                router_logits = block.gate.get_logits(hs_flat)
                s3.record()
                timer_ref.record(layer_id, "gate_logits", path, s2, s3)

                # --- Full MoE (dispatch + routing + fused_experts + combine) ---
                s4 = torch.cuda.Event(enable_timing=True)
                s5 = torch.cuda.Event(enable_timing=True)
                s4.record()
                y = _moe_forward_with_context(block.experts, hs_flat, router_logits)
                s5.record()
                timer_ref.record(layer_id, "moe_total", path, s4, s5)

                if shared_res is not None:
                    y = y + shared_res

                return y.view(bsz, seq_len, h)

            return instrumented_forward

        # Apply EB routing patch (same as bench_dp2_tp4_ep8.py)
        gate_idx = 0
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                b, r, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                 mod.n_group, mod.topk_group)
                li = gate_idx
                def mk_routing(bb, rr, nn, gg, layer_i, cc):
                    def fn(hs, go, topk, renorm):
                        sm = cc.get_s_mask(layer_i, go, bb)
                        w, idx = fused_routing(go, bb, rr, s_mask=sm, K=4, ng=nn, tkg=gg)
                        return w.to(go.dtype), idx
                    return fn
                mod.routing = mk_routing(b, r, ng, tkg, li, ctrl)
                gate_idx += 1

        # Apply instrumented forward
        for i, block in enumerate(moe_blocks):
            gp = gate_params[i] if i < len(gate_params) else gate_params[-1]
            block.forward = make_instrumented_forward(block, i, gp, ctrl, timer)

        # Add is_hot_check to controller (check without modifying state)
        def _is_hot_check(self, layer_id):
            return layer_id in self.s_mask_cache
        ctrl.is_hot_check = lambda lid: _is_hot_check(ctrl, lid)

        def reset():
            ctrl.prev_N.clear(); ctrl.K_init.clear()
            ctrl.cold_count = 0; ctrl.hot_count = 0
            ctrl.eb_calls = 0; ctrl.eb_skips = 0
            ctrl._bufs.clear(); ctrl.k_init_history.clear()
            ctrl.s_mask_cache.clear(); ctrl.pop_cache.clear()
            ctrl._fwd_in_block.clear(); ctrl._block_idx.clear()

        # --- Build input ---
        all_ids = []
        for i in range(args.batch_size):
            text = PROMPTS[i % len(PROMPTS)]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=False)
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                  if ids.shape[0] < mx else ids for ids in all_ids]
        input_ids_full = torch.stack(padded, dim=0)
        my_start = dp_rank * local_bs
        my_input = input_ids_full[my_start : my_start + local_bs].to(device)

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

        # --- Warmup (no timing) ---
        if rank == 0:
            print("\nWarmup...")
        reset()
        timer.disable()
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(my_input.clone(), gen_length=args.gen_length,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd")

        # --- Timed profiling run ---
        if rank == 0:
            print("\nProfiling run...")
        reset()
        timer.enable()
        dllm = make_dllm()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(my_input.clone(), gen_length=args.gen_length,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.perf_counter()

        num_fwd = dllm.diff_iteration.num_forwards
        wall_ms = (t1 - t0) * 1000
        ms_per_fwd = wall_ms / num_fwd if num_fwd > 0 else 0

        if rank == 0:
            print(f"\n  Wall: {wall_ms:.1f} ms, {num_fwd} fwd, {ms_per_fwd:.2f} ms/fwd")
            print(f"  cold={ctrl.cold_count}, hot={ctrl.hot_count}")

        # --- Aggregate results ---
        summary = timer.summarize()

        if rank == 0:
            # Aggregate across layers
            totals = defaultdict(lambda: {"count": 0, "total_ms": 0.0})
            for layer_id, comps in summary.items():
                for comp_key, data in comps.items():
                    totals[comp_key]["count"] += data["count"]
                    totals[comp_key]["total_ms"] += data["total_ms"]

            print(f"\n{'='*70}")
            print(f"COMPONENT BREAKDOWN (all layers aggregated, rank 0)")
            print(f"{'='*70}")
            print(f"{'Component':<30} {'Calls':>6} {'Total ms':>10} {'Avg ms':>10} {'%wall':>8}")
            print(f"{'-'*70}")

            for key in sorted(totals.keys()):
                d = totals[key]
                avg = d["total_ms"] / d["count"] if d["count"] > 0 else 0
                pct = d["total_ms"] / wall_ms * 100
                print(f"{key:<30} {d['count']:>6} {d['total_ms']:>10.1f} {avg:>10.3f} {pct:>7.1f}%")

            # Estimate dispatch+combine = moe_total - (gate_logits + routing time embedded in moe_total)
            # Note: gate_logits is separate, but routing is INSIDE moe_total
            moe_hot = totals.get("moe_total_hot", {"total_ms": 0})["total_ms"]
            moe_cold = totals.get("moe_total_cold", {"total_ms": 0})["total_ms"]
            shared_hot = totals.get("shared_hot", {"total_ms": 0})["total_ms"]
            shared_cold = totals.get("shared_cold", {"total_ms": 0})["total_ms"]
            gate_hot = totals.get("gate_logits_hot", {"total_ms": 0})["total_ms"]
            gate_cold = totals.get("gate_logits_cold", {"total_ms": 0})["total_ms"]

            print(f"\n--- High-level summary ---")
            print(f"  MoE total (hot):   {moe_hot:.1f} ms ({moe_hot/wall_ms*100:.1f}%)")
            print(f"  MoE total (cold):  {moe_cold:.1f} ms ({moe_cold/wall_ms*100:.1f}%)")
            print(f"  Shared (hot+cold): {shared_hot+shared_cold:.1f} ms ({(shared_hot+shared_cold)/wall_ms*100:.1f}%)")
            print(f"  Gate (hot+cold):   {gate_hot+gate_cold:.1f} ms ({(gate_hot+gate_cold)/wall_ms*100:.1f}%)")
            print(f"  Non-MoE (attention, RMSNorm, etc.): {wall_ms - moe_hot - moe_cold - shared_hot - shared_cold - gate_hot - gate_cold:.1f} ms")

            # Per-layer breakdown for first 3 layers
            print(f"\n--- Per-layer detail (layers 0-2) ---")
            for lid in range(min(3, len(summary))):
                if lid in summary:
                    print(f"  Layer {lid}:")
                    for comp_key, data in sorted(summary[lid].items()):
                        avg = data["total_ms"] / data["count"] if data["count"] > 0 else 0
                        print(f"    {comp_key:<25} n={data['count']:>3}  total={data['total_ms']:>7.2f}ms  avg={avg:.3f}ms")

            # Save results
            results = {
                "config": "C12_M5_K4_profiling",
                "batch_size": args.batch_size,
                "gen_length": args.gen_length,
                "num_fwd": num_fwd,
                "wall_ms": wall_ms,
                "ms_per_fwd": ms_per_fwd,
                "cold_count": ctrl.cold_count,
                "hot_count": ctrl.hot_count,
                "totals": {k: v for k, v in totals.items()},
                "per_layer": {str(k): {ck: cv for ck, cv in v.items()}
                              for k, v in summary.items()},
            }
            out_path = REPO_ROOT / "codex_coding" / "results" / "c12_component_profiling.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
