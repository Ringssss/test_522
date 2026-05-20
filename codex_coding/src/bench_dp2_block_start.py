#!/usr/bin/env python3
"""
Block-start vs steady-state MoE timing under dp=2 tp=4 ep=8 AllToAll.

Measures per-forward MoE time per layer, split by block position.
Also collects per-GPU expert load (topk_ids) to correlate skew with timing.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=naive \
    torchrun --nproc_per_node=8 codex_coding/src/bench_dp2_block_start.py
"""

from __future__ import annotations
import os, sys, time, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
TP_SIZE = 4
NUM_EXPERTS = 256


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    BATCH_SIZE = 512
    GEN_LENGTH = 64  # 2 blocks, fast profiling

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    dp_size = world_size // TP_SIZE
    dp_rank = rank // TP_SIZE
    local_bs = BATCH_SIZE // dp_size
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    alltoall_backend = os.environ.get("VLLM_ALL2ALL_BACKEND", "naive")

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeSparseMoeBlock, _moe_forward_with_context
    from transformers import AutoTokenizer, AutoConfig
    from test_fused_eb_triton import fused_routing
    from test_m_skip_sweep import MSkipEBController
    from baseline_optimizations import apply_all_optimizations
    from test_heteval512 import PROMPTS

    # Two-phase init
    pcfg_init = ParallelConfig(tensor_parallel_size=1, data_parallel_size=1,
                                enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(world_size, rank, "env://", local_rank, "nccl")

    pcfg = ParallelConfig(tensor_parallel_size=TP_SIZE, data_parallel_size=dp_size,
                           data_parallel_rank=dp_rank, enable_expert_parallel=True)
    vllm_cfg = VllmConfig(parallel_config=pcfg)

    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(tensor_model_parallel_size=TP_SIZE, backend="nccl")

        from vllm.distributed import (prepare_communication_buffer_for_model,
                                       get_ep_group)
        ep_size = get_ep_group().world_size

        if rank == 0:
            print(f"dp={dp_size} tp={TP_SIZE} ep={ep_size} | batch={BATCH_SIZE} gen={GEN_LENGTH}")
            print(f"Backend: {alltoall_backend}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        from vllm.forward_context import set_forward_context
        with torch.inference_mode():
            w = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                     num_tokens=w.numel()):
                _ = model(w, use_cache=False)
        apply_all_optimizations(model)
        prepare_communication_buffer_for_model(model)

        # Build input
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
        padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                  if ids.shape[0] < mx else ids for ids in all_ids]
        input_ids_full = torch.stack(padded, dim=0)
        my_input = input_ids_full[dp_rank * local_bs : (dp_rank + 1) * local_bs].to(device)

        # C11-M5-K4 routing patch with per-forward topk_ids collection (rank 0 only)
        ctrl = MSkipEBController(num_layers=19, K=8, M=4, K_target=40,
                                  quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)

        # Per-forward data collection
        fwd_counter = [0]
        # per_fwd_routing[fwd_idx] = {layer_idx: topk_ids_numpy}  (rank 0 only)
        per_fwd_routing = defaultdict(dict) if rank == 0 else None
        layer_seen_in_fwd = [set()]

        gate_idx = 0
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                     mod.top_k, mod.n_group, mod.topk_group)
                li = gate_idx
                def mk(bb, rr, nn, gg, layer_i, cc, rk, pfr, lsf, fc):
                    def fn(hs, go, topk, renorm):
                        sm = cc.get_s_mask(layer_i, go, bb)
                        w, idx = fused_routing(go, bb, rr, s_mask=sm, K=4, ng=nn, tkg=gg)
                        # Collect routing on rank 0
                        if rk == 0:
                            if layer_i == 0 and 0 in lsf[0]:
                                fc[0] += 1
                                lsf[0] = set()
                            lsf[0].add(layer_i)
                            pfr[fc[0]][layer_i] = idx.cpu().numpy().copy()
                        return w.to(go.dtype), idx
                    return fn
                mod.routing = mk(b, r, ng, tkg, li, ctrl, rank, per_fwd_routing,
                                 layer_seen_in_fwd, fwd_counter)
                gate_idx += 1

        # Instrument MoE forward with CUDA Event timing
        moe_fwd_times = defaultdict(list) if rank == 0 else None  # fwd_idx → [(layer, ms)]

        for name, mod in model.named_modules():
            if isinstance(mod, LLaDA2MoeSparseMoeBlock):
                orig_fwd = mod.forward
                lname = name
                def make_timed(orig, ln, rk, fc, mft):
                    def fwd(hidden_states):
                        if rk == 0:
                            s = torch.cuda.Event(enable_timing=True)
                            e = torch.cuda.Event(enable_timing=True)
                            s.record()
                        result = orig(hidden_states)
                        if rk == 0:
                            e.record()
                            mft[fc[0]].append((ln, s, e))
                        return result
                    return fwd
                mod.forward = make_timed(orig_fwd, lname, rank, fwd_counter, moe_fwd_times)

        def reset():
            ctrl.prev_N.clear(); ctrl.K_init.clear()
            ctrl.cold_count = 0; ctrl.hot_count = 0
            ctrl.eb_calls = 0; ctrl.eb_skips = 0
            ctrl._bufs.clear(); ctrl.k_init_history.clear()
            ctrl.s_mask_cache.clear(); ctrl.pop_cache.clear()
            ctrl._fwd_in_block.clear(); ctrl._block_idx.clear()
            fwd_counter[0] = 0
            layer_seen_in_fwd[0] = set()
            if per_fwd_routing is not None:
                per_fwd_routing.clear()
            if moe_fwd_times is not None:
                moe_fwd_times.clear()

        decoder = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Warmup
        if rank == 0:
            print("Warmup...")
        reset()
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(my_input.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        dist.barrier()

        # Profiled run
        if rank == 0:
            print("Profiled run...")
        reset()
        dllm = make_dllm()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(my_input.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.perf_counter()

        total_fwd = dllm.diff_iteration.num_forwards

        if rank == 0:
            total_time = t1 - t0
            torch.cuda.synchronize()
            print(f"\nTotal: {total_time:.3f}s, {total_fwd} fwd, "
                  f"{total_time*1000/total_fwd:.2f} ms/fwd")

            # Process CUDA events
            epg = NUM_EXPERTS // ep_size  # experts per GPU

            # Per-forward: total MoE time + routing skew
            fwd_analysis = []
            for fi in sorted(moe_fwd_times.keys()):
                entries = moe_fwd_times[fi]
                total_moe_ms = sum(s.elapsed_time(e) for _, s, e in entries)

                # Routing skew from topk_ids (Layer 0 as representative)
                skew = 0
                if fi in per_fwd_routing and 0 in per_fwd_routing[fi]:
                    ids = per_fwd_routing[fi][0].flatten()
                    gpu_counts = [np.sum((ids >= g*epg) & (ids < (g+1)*epg)) for g in range(ep_size)]
                    skew = max(gpu_counts) / max(min(gpu_counts), 1)

                fwd_analysis.append({
                    'fwd': fi, 'moe_ms': total_moe_ms, 'skew': skew,
                })

            # Detect block boundaries
            boundaries = [0]
            for i in range(1, len(fwd_analysis)):
                # Check N jump from routing data
                if i in per_fwd_routing and 0 in per_fwd_routing[i]:
                    n_curr = per_fwd_routing[i][0].shape[0]
                    if i-1 in per_fwd_routing and 0 in per_fwd_routing[i-1]:
                        n_prev = per_fwd_routing[i-1][0].shape[0]
                        if n_curr > n_prev:
                            boundaries.append(i)

            # Split block-start (first 5) vs steady (6+)
            bs_data = []
            ss_data = []
            for d in fwd_analysis:
                fi = d['fwd']
                in_bs = any(b <= fi <= b + 4 for b in boundaries)
                if in_bs:
                    bs_data.append(d)
                else:
                    ss_data.append(d)

            print(f"\n{'='*70}")
            print(f"BLOCK-START vs STEADY-STATE (dp=2 tp=4 ep=8 AllToAll)")
            print(f"{'='*70}")

            if bs_data:
                bs_times = [d['moe_ms'] for d in bs_data]
                bs_skews = [d['skew'] for d in bs_data if d['skew'] > 0]
                print(f"\n  Block-start (first 5 fwd/block): {len(bs_data)} forwards")
                print(f"    Avg MoE: {np.mean(bs_times):.2f} ms/fwd")
                print(f"    Max MoE: {max(bs_times):.2f} ms/fwd")
                if bs_skews:
                    print(f"    Avg routing skew (L1): {np.mean(bs_skews):.2f}x")
                    print(f"    Max routing skew (L1): {max(bs_skews):.2f}x")

            if ss_data:
                ss_times = [d['moe_ms'] for d in ss_data]
                ss_skews = [d['skew'] for d in ss_data if d['skew'] > 0]
                print(f"\n  Steady-state (fwd 6+): {len(ss_data)} forwards")
                print(f"    Avg MoE: {np.mean(ss_times):.2f} ms/fwd")
                print(f"    Max MoE: {max(ss_times):.2f} ms/fwd")
                if ss_skews:
                    print(f"    Avg routing skew (L1): {np.mean(ss_skews):.2f}x")
                    print(f"    Max routing skew (L1): {max(ss_skews):.2f}x")

            if bs_data and ss_data:
                ratio = np.mean(bs_times) / np.mean(ss_times)
                print(f"\n  Block-start / Steady-state MoE ratio: {ratio:.3f}x")
                if ratio > 1.05:
                    savings = sum(bs_times) - len(bs_data) * np.mean(ss_times)
                    print(f"  Potential savings if balanced: {savings:.1f} ms "
                          f"({savings/total_time/10:.2f}%)")
                else:
                    print(f"  No significant timing difference")

            # Per-block detail
            print(f"\n  Per-block detail (first 3 blocks):")
            for bi, bstart in enumerate(boundaries[:3]):
                print(f"\n  Block {bi} (start fwd {bstart}):")
                print(f"    {'Fwd':>5s} {'MoE_ms':>8s} {'Skew':>6s} {'Phase':>6s}")
                for offset in range(min(15, total_fwd - bstart)):
                    fi = bstart + offset
                    for d in fwd_analysis:
                        if d['fwd'] == fi:
                            phase = "BS" if fi - bstart < 5 else "SS"
                            print(f"    {fi:>5d} {d['moe_ms']:>8.2f} "
                                  f"{d['skew']:>6.2f} {phase:>6s}")
                            break

            # Save
            results = {
                'config': 'C11_M5_K4_dp2_tp4_ep8',
                'total_time_s': total_time, 'total_fwd': total_fwd,
                'block_start_avg_ms': np.mean(bs_times) if bs_data else 0,
                'steady_state_avg_ms': np.mean(ss_times) if ss_data else 0,
                'block_start_avg_skew': np.mean(bs_skews) if bs_skews else 0,
                'steady_state_avg_skew': np.mean(ss_skews) if ss_skews else 0,
                'per_fwd': fwd_analysis,
            }
            out_path = REPO_ROOT / "codex_coding" / "results" / "dp2_block_start_timing.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n  Saved to {out_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
