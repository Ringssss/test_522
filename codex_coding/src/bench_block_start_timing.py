#!/usr/bin/env python3
"""
Block-start vs steady-state MoE timing under AllToAll (AGR).

Measures per-forward MoE time split by position-in-block:
  - block_start (first 5 fwd): potentially high skew
  - steady_state (fwd 6+): lower skew

Runs C11-M5-K4 on 4 GPUs with AGR backend.

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter \
    torchrun --nproc_per_node=4 codex_coding/src/bench_block_start_timing.py
"""

from __future__ import annotations
import os, sys, time, json
from pathlib import Path
from collections import defaultdict

import torch
import torch.distributed as dist

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    BATCH_SIZE = 128
    GEN_LENGTH = 256

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

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
    from test_heteval128 import PROMPTS

    alltoall_backend = os.environ.get("VLLM_ALL2ALL_BACKEND", "")

    vllm_dist.init_distributed_environment(world_size, rank, "env://", local_rank, "nccl")
    vllm_dist.initialize_model_parallel(world_size, backend="nccl")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0),
                      use_cache=False)
        apply_all_optimizations(model)

        # Initialize AllToAll if backend set
        if alltoall_backend:
            from vllm.distributed import prepare_communication_buffer_for_model
            prepare_communication_buffer_for_model(model)
            if rank == 0:
                print(f"AllToAll buffers initialized (backend={alltoall_backend})")

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
        input_ids = torch.stack(padded, dim=0).to(device)

        # EB controller
        ctrl = MSkipEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)

        # --- Instrument MoE layers with per-forward timing ---
        # Track: fwd_idx, is_new_block, per-MoE-layer time
        fwd_counter = [0]
        prev_N = [{}]
        fwd_times = defaultdict(list)  # fwd_idx → list of (layer_idx, time_ms, N, is_block_start)

        gate_idx = 0
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                     mod.top_k, mod.n_group, mod.topk_group)
                li = gate_idx
                def mk(bb, rr, nn, gg, layer_i, cc):
                    def fn(hs, go, topk, renorm):
                        sm = cc.get_s_mask(layer_i, go, bb)
                        w, idx = fused_routing(go, bb, rr, s_mask=sm, K=4, ng=nn, tkg=gg)
                        return w.to(go.dtype), idx
                    return fn
                mod.routing = mk(b, r, ng, tkg, li, ctrl)
                gate_idx += 1

        # Patch MoE block forward to time each call
        for name, mod in model.named_modules():
            if isinstance(mod, LLaDA2MoeSparseMoeBlock):
                orig_fwd = mod.forward
                layer_name = name

                def make_timed_fwd(orig, lname, fc, pN, ft):
                    def timed_fwd(hidden_states):
                        N = hidden_states.shape[0] * hidden_states.shape[1]

                        # Detect block start (N jumps up)
                        is_bs = False
                        if lname not in pN[0]:
                            is_bs = True
                        elif N > pN[0][lname]:
                            is_bs = True
                            fc[0] += 1
                        pN[0][lname] = N

                        if rank == 0:
                            start_ev = torch.cuda.Event(enable_timing=True)
                            end_ev = torch.cuda.Event(enable_timing=True)
                            start_ev.record()

                        result = orig(hidden_states)

                        if rank == 0:
                            end_ev.record()
                            ft[fc[0]].append((lname, start_ev, end_ev, N, is_bs))

                        return result
                    return timed_fwd

                mod.forward = make_timed_fwd(orig_fwd, layer_name, fwd_counter, prev_N, fwd_times)

        # Warmup
        def reset():
            ctrl.prev_N.clear(); ctrl.K_init.clear()
            ctrl.cold_count = 0; ctrl.hot_count = 0
            ctrl.eb_calls = 0; ctrl.eb_skips = 0
            ctrl._bufs.clear(); ctrl.k_init_history.clear()
            ctrl.s_mask_cache.clear(); ctrl.pop_cache.clear()
            ctrl._fwd_in_block.clear(); ctrl._block_idx.clear()
            fwd_counter[0] = 0
            prev_N[0].clear()
            fwd_times.clear()

        decoder = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        if rank == 0:
            print("Warmup...")
        reset()
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        dist.barrier()

        # Timed run
        if rank == 0:
            print("Profiled run...")
        reset()
        dllm = make_dllm()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        dist.barrier()
        t1 = time.perf_counter()

        total_fwd = dllm.diff_iteration.num_forwards

        if rank == 0:
            total_time = t1 - t0
            print(f"\nTotal: {total_time:.3f}s, {total_fwd} fwd, "
                  f"{total_time*1000/total_fwd:.2f} ms/fwd")

            # Sync CUDA events and compute per-forward MoE times
            torch.cuda.synchronize()

            # Analyze: per model-forward, sum all MoE layer times
            # Detect block boundaries from the data
            block_starts = set()
            fwd_moe_times = {}  # fwd_idx → (total_moe_ms, N, is_block_start)

            for fi in sorted(fwd_times.keys()):
                entries = fwd_times[fi]
                total_ms = 0
                N_val = 0
                is_bs = False
                for (lname, sev, eev, N, bs_flag) in entries:
                    ms = sev.elapsed_time(eev)
                    total_ms += ms
                    N_val = N
                    if bs_flag:
                        is_bs = True
                if is_bs:
                    block_starts.add(fi)
                fwd_moe_times[fi] = (total_ms, N_val, is_bs)

            # Split into block-start and steady-state
            bs_times = []
            ss_times = []
            for fi in sorted(fwd_moe_times.keys()):
                total_ms, N, is_bs = fwd_moe_times[fi]
                # block-start = first 5 forwards of each block
                in_block_start_zone = False
                for bfi in sorted(block_starts):
                    if bfi <= fi <= bfi + 4:
                        in_block_start_zone = True
                        break
                if in_block_start_zone:
                    bs_times.append(total_ms)
                else:
                    ss_times.append(total_ms)

            print(f"\n{'='*70}")
            print(f"BLOCK-START vs STEADY-STATE MoE TIMING")
            print(f"  Backend: {alltoall_backend or 'AllReduce'}")
            print(f"{'='*70}")
            print(f"  Block-start (first 5 fwd/block): {len(bs_times)} forwards")
            if bs_times:
                print(f"    Avg MoE time: {sum(bs_times)/len(bs_times):.2f} ms/fwd")
                print(f"    Max MoE time: {max(bs_times):.2f} ms/fwd")
                print(f"    Min MoE time: {min(bs_times):.2f} ms/fwd")

            print(f"\n  Steady-state (fwd 6+): {len(ss_times)} forwards")
            if ss_times:
                print(f"    Avg MoE time: {sum(ss_times)/len(ss_times):.2f} ms/fwd")
                print(f"    Max MoE time: {max(ss_times):.2f} ms/fwd")
                print(f"    Min MoE time: {min(ss_times):.2f} ms/fwd")

            if bs_times and ss_times:
                ratio = (sum(bs_times)/len(bs_times)) / (sum(ss_times)/len(ss_times))
                print(f"\n  Block-start / Steady-state ratio: {ratio:.2f}x")
                # Potential savings if block-start used AllReduce instead
                bs_total = sum(bs_times)
                if ratio > 1.1:
                    savings = bs_total * (1 - 1/ratio)
                    print(f"  Block-start total MoE: {bs_total:.1f} ms")
                    print(f"  If balanced to steady-state level: save {savings:.1f} ms "
                          f"({savings/total_time/10:.1f}%)")

            # Per-block detail (first 4 blocks)
            print(f"\n  Per-block first 10 forwards:")
            blocks = sorted(block_starts)
            for bi, bfi in enumerate(blocks[:4]):
                print(f"\n  Block {bi} (start fwd {bfi}):")
                for offset in range(min(10, total_fwd - bfi)):
                    fi = bfi + offset
                    if fi in fwd_moe_times:
                        ms, N, is_bs = fwd_moe_times[fi]
                        tag = "BS" if fi - bfi < 5 else "SS"
                        print(f"    fwd {fi}: {ms:.2f} ms (N={N}, {tag})")

            # Save
            results = {
                'backend': alltoall_backend or 'AllReduce',
                'total_time_s': total_time,
                'total_fwd': total_fwd,
                'block_start_count': len(bs_times),
                'steady_state_count': len(ss_times),
                'block_start_avg_ms': sum(bs_times)/len(bs_times) if bs_times else 0,
                'steady_state_avg_ms': sum(ss_times)/len(ss_times) if ss_times else 0,
            }
            out_path = REPO_ROOT / "codex_coding" / "results" / "block_start_timing.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Saved to {out_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
