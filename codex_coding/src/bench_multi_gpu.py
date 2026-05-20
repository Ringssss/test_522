#!/usr/bin/env python3
"""
v0.1.15.8n — Multi-GPU EP Benchmark for LLaDA2.0-mini

8-GPU Expert Parallelism with AllReduce, testing C5 and C10-M5.
Uses torchrun for distributed launch.

Usage:
  torchrun --nproc_per_node=8 codex_coding/src/bench_multi_gpu.py
"""

from __future__ import annotations
import os, sys, time, json
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

import argparse

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
GEN_LENGTH = 256
DEFAULT_BATCH_SIZE = 128


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size

    # Distributed setup via torchrun
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
    from transformers import AutoTokenizer, AutoConfig

    from test_fused_eb_triton import fused_routing, FusedEBController
    from test_m_skip_sweep import MSkipEBController
    from baseline_optimizations import apply_all_optimizations

    # Select prompt set based on batch size
    if BATCH_SIZE <= 128:
        from test_heteval128 import PROMPTS, VERIFIABLE
    else:
        from test_heteval512 import PROMPTS, VERIFIABLE

    # Initialize distributed
    vllm_dist.init_distributed_environment(world_size, rank, "env://", local_rank, "nccl")
    vllm_dist.initialize_model_parallel(world_size, backend="nccl")

    if rank == 0:
        print("=" * 80)
        print(f"Multi-GPU EP Benchmark — {world_size} GPUs")
        print(f"  batch={BATCH_SIZE}, gen={GEN_LENGTH}, block={BLOCK_LENGTH}")
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

        if rank == 0:
            print("\nApplying baseline optimizations...")
        n_rms, n_fa = apply_all_optimizations(model)
        if rank == 0:
            print(f"  Fused RMSNorm: {n_rms}, Flash-attn: {n_fa}")

        # Check EP status
        for name, mod in model.named_modules():
            if hasattr(mod, 'expert_map') and mod.expert_map is not None:
                if rank == 0:
                    local_experts = (mod.expert_map >= 0).sum().item()
                    print(f"  EP active: {local_experts} local experts per GPU "
                          f"(total {mod.expert_map.shape[0]})")
                break

        # Build input (same on all ranks for deterministic generation)
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

        if rank == 0:
            print(f"  Input shape: {input_ids.shape}")
            mem_alloc = torch.cuda.memory_allocated(device) / 1e9
            print(f"  GPU {local_rank} memory: {mem_alloc:.1f} GB")

        decoder_t0 = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        decoder_t7 = ThresholdParallelDecoder(
            temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        # Skip quality check for large batch (gumbel noise OOMs at batch>=256)
        do_quality = BATCH_SIZE <= 128

        prompt_len = input_ids.shape[1]

        def make_dllm(dec):
            return BlockDiffusionLLM(
                model, dec,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True,
                inplace_cache_update=True)

        # --- Routing patch functions ---
        orig_routings = {}
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                orig_routings[name] = mod.routing

        def restore():
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate" and name in orig_routings:
                    mod.routing = orig_routings[name]

        def patch_c5():
            restore()
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate":
                    b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                         mod.top_k, mod.n_group, mod.topk_group)
                    def mk(bb, rr, tt, nn, gg):
                        def fn(hs, go, topk, renorm):
                            w, i = fused_routing(go, bb, rr, s_mask=None, K=tt, ng=nn, tkg=gg)
                            return w.to(go.dtype), i
                        return fn
                    mod.routing = mk(b, r, tk, ng, tkg)

        def patch_eb(ctrl):
            restore()
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

        def patch_eb_k4(ctrl, topk_override=4):
            """EB routing with topk compressed to K=4."""
            restore()
            idx = 0
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate":
                    b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                         mod.top_k, mod.n_group, mod.topk_group)
                    li = idx
                    def mk(bb, rr, tt, nn, gg, layer_i, cc, k_ovr):
                        def fn(hs, go, topk, renorm):
                            sm = cc.get_s_mask(layer_i, go, bb)
                            w, i = fused_routing(go, bb, rr, s_mask=sm, K=k_ovr, ng=nn, tkg=gg)
                            return w.to(go.dtype), i
                        return fn
                    mod.routing = mk(b, r, tk, ng, tkg, li, ctrl, topk_override)
                    idx += 1

        def reset_ctrl(ctrl):
            ctrl.prev_N.clear(); ctrl.K_init.clear()
            ctrl.cold_count = 0; ctrl.hot_count = 0
            ctrl.eb_calls = 0; ctrl.eb_skips = 0
            ctrl._bufs.clear(); ctrl.k_init_history.clear()
            ctrl.s_mask_cache.clear(); ctrl.pop_cache.clear()
            ctrl._fwd_in_block.clear(); ctrl._block_idx.clear()

        results = {}

        # ========== C5 ==========
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"  C5: Fused routing only ({world_size}-GPU EP)")
            print(f"{'='*60}")
        patch_c5()

        # Warmup
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        if rank == 0:
            print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd")

        # Timed runs
        c5_times, c5_fwds = [], []
        for ri in range(2):
            dllm = make_dllm(decoder_t0)
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            dist.barrier()
            t1 = time.perf_counter()
            c5_times.append(t1 - t0)
            c5_fwds.append(dllm.diff_iteration.num_forwards)
            if rank == 0:
                print(f"    Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd, "
                      f"{(t1-t0)*1000/dllm.diff_iteration.num_forwards:.2f} ms/fwd")

        # Quality check C5 (temp=0.7, rank 0 only prints)
        if do_quality:
            dllm = make_dllm(decoder_t7)
            with torch.inference_mode():
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            dllm = make_dllm(decoder_t7)
            with torch.inference_mode():
                out_c5 = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            if rank == 0:
                gen_c5 = out_c5[:, prompt_len:]
                print(f"  Quality (temp=0.7):")
                for bi in sorted(VERIFIABLE.keys()):
                    gt = gen_c5[bi]
                    valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                    text = tokenizer.decode(valid, skip_special_tokens=True)
                    print(f"    #{bi} [{VERIFIABLE[bi][:30]}]: {text[:150]}")
        elif rank == 0:
            print(f"  Quality check skipped (batch={BATCH_SIZE} > 128, OOM risk)")

        results['C5'] = {
            'avg_time': sum(c5_times) / 2,
            'avg_fwd': sum(c5_fwds) / 2,
            'ms_per_fwd': sum(c5_times) / 2 / (sum(c5_fwds) / 2) * 1000,
        }

        # ========== C10-M5 ==========
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"  C10-M5: EB skip_m=5 ({world_size}-GPU EP)")
            print(f"{'='*60}")

        ctrl = MSkipEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)
        patch_eb(ctrl)

        # Warmup
        reset_ctrl(ctrl)
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        if rank == 0:
            print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd, "
                  f"cold={ctrl.cold_count}, hot={ctrl.hot_count}")

        # Timed runs
        m5_times, m5_fwds = [], []
        for ri in range(2):
            reset_ctrl(ctrl)
            dllm = make_dllm(decoder_t0)
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            dist.barrier()
            t1 = time.perf_counter()
            m5_times.append(t1 - t0)
            m5_fwds.append(dllm.diff_iteration.num_forwards)
            if rank == 0:
                print(f"    Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd, "
                      f"{(t1-t0)*1000/dllm.diff_iteration.num_forwards:.2f} ms/fwd, "
                      f"cold={ctrl.cold_count} hot={ctrl.hot_count}")

        # Quality check C10-M5
        if do_quality:
            reset_ctrl(ctrl)
            dllm = make_dllm(decoder_t7)
            with torch.inference_mode():
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            reset_ctrl(ctrl)
            dllm = make_dllm(decoder_t7)
            with torch.inference_mode():
                out_m5 = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            if rank == 0:
                gen_m5 = out_m5[:, prompt_len:]
                print(f"  Quality (temp=0.7):")
                for bi in sorted(VERIFIABLE.keys()):
                    gt = gen_m5[bi]
                    valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                    text = tokenizer.decode(valid, skip_special_tokens=True)
                    print(f"    #{bi} [{VERIFIABLE[bi][:30]}]: {text[:150]}")
        elif rank == 0:
            print(f"  Quality check skipped (batch={BATCH_SIZE} > 128)")

        results['C10_M5'] = {
            'avg_time': sum(m5_times) / 2,
            'avg_fwd': sum(m5_fwds) / 2,
            'ms_per_fwd': sum(m5_times) / 2 / (sum(m5_fwds) / 2) * 1000,
        }

        # ========== C11-M5-K4 ==========
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"  C11-M5-K4: EB skip_m=5 + topk=4 ({world_size}-GPU EP)")
            print(f"{'='*60}")

        ctrl_k4 = MSkipEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)
        patch_eb_k4(ctrl_k4, topk_override=4)

        # Warmup
        reset_ctrl(ctrl_k4)
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        if rank == 0:
            print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd, "
                  f"cold={ctrl_k4.cold_count}, hot={ctrl_k4.hot_count}")

        # Timed runs
        k4_times, k4_fwds = [], []
        for ri in range(2):
            reset_ctrl(ctrl_k4)
            dllm = make_dllm(decoder_t0)
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            dist.barrier()
            t1 = time.perf_counter()
            k4_times.append(t1 - t0)
            k4_fwds.append(dllm.diff_iteration.num_forwards)
            if rank == 0:
                print(f"    Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd, "
                      f"{(t1-t0)*1000/dllm.diff_iteration.num_forwards:.2f} ms/fwd, "
                      f"cold={ctrl_k4.cold_count} hot={ctrl_k4.hot_count}")

        # Quality check C11-M5-K4
        if do_quality:
            reset_ctrl(ctrl_k4)
            dllm = make_dllm(decoder_t7)
            with torch.inference_mode():
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            reset_ctrl(ctrl_k4)
            dllm = make_dllm(decoder_t7)
            with torch.inference_mode():
                out_k4 = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            if rank == 0:
                gen_k4 = out_k4[:, prompt_len:]
                print(f"  Quality (temp=0.7):")
                for bi in sorted(VERIFIABLE.keys()):
                    gt = gen_k4[bi]
                    valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                    text = tokenizer.decode(valid, skip_special_tokens=True)
                    print(f"    #{bi} [{VERIFIABLE[bi][:30]}]: {text[:150]}")
        elif rank == 0:
            print(f"  Quality check skipped (batch={BATCH_SIZE} > 128)")

        results['C11_M5_K4'] = {
            'avg_time': sum(k4_times) / 2,
            'avg_fwd': sum(k4_fwds) / 2,
            'ms_per_fwd': sum(k4_times) / 2 / (sum(k4_fwds) / 2) * 1000,
        }

        # ========== C11 + AllToAll EP (if requested via --alltoall) ==========
        alltoall_backend = os.environ.get("VLLM_ALL2ALL_BACKEND", "")
        if alltoall_backend:
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"  C11-A2A: C5 + AllToAll EP ({alltoall_backend})")
                print(f"{'='*60}")

            # Initialize AllToAll communication buffers (creates MK)
            from vllm.distributed import prepare_communication_buffer_for_model
            prepare_communication_buffer_for_model(model)

            if rank == 0:
                print(f"  AllToAll buffers initialized (backend={alltoall_backend})")

            # Use C5 routing (already patched)
            patch_c5()

            # Warmup
            dllm = make_dllm(decoder_t0)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            if rank == 0:
                print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd")

            # Timed runs
            a2a_times, a2a_fwds = [], []
            for ri in range(2):
                dllm = make_dllm(decoder_t0)
                torch.cuda.synchronize()
                dist.barrier()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                      block_length=BLOCK_LENGTH)
                torch.cuda.synchronize()
                dist.barrier()
                t1 = time.perf_counter()
                a2a_times.append(t1 - t0)
                a2a_fwds.append(dllm.diff_iteration.num_forwards)
                if rank == 0:
                    print(f"    Run {ri+1}: {t1-t0:.3f}s, "
                          f"{dllm.diff_iteration.num_forwards} fwd, "
                          f"{(t1-t0)*1000/dllm.diff_iteration.num_forwards:.2f} ms/fwd")

            # Quality check
            if do_quality:
                dllm = make_dllm(decoder_t7)
                with torch.inference_mode():
                    _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                      block_length=BLOCK_LENGTH)
                dllm = make_dllm(decoder_t7)
                with torch.inference_mode():
                    out_a2a = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                            block_length=BLOCK_LENGTH)
                torch.cuda.synchronize()
                if rank == 0:
                    gen_a2a = out_a2a[:, prompt_len:]
                    print(f"  Quality (temp=0.7):")
                    for bi in sorted(VERIFIABLE.keys()):
                        gt = gen_a2a[bi]
                        valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                        text = tokenizer.decode(valid, skip_special_tokens=True)
                        print(f"    #{bi} [{VERIFIABLE[bi][:30]}]: {text[:150]}")
            elif rank == 0:
                print(f"  Quality check skipped (batch={BATCH_SIZE} > 128)")

            results[f'C11_A2A_{alltoall_backend}'] = {
                'avg_time': sum(a2a_times) / 2,
                'avg_fwd': sum(a2a_fwds) / 2,
                'ms_per_fwd': sum(a2a_times) / 2 / (sum(a2a_fwds) / 2) * 1000,
            }

            # ========== C11-M5-K4 + AllToAll ==========
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"  C11-M5-K4-A2A: EB+K4 + AllToAll ({alltoall_backend})")
                print(f"{'='*60}")

            ctrl_k4a = MSkipEBController(
                num_layers=19, K=8, M=4, K_target=40,
                quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)
            patch_eb_k4(ctrl_k4a, topk_override=4)

            # Warmup
            reset_ctrl(ctrl_k4a)
            dllm = make_dllm(decoder_t0)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            if rank == 0:
                print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd, "
                      f"cold={ctrl_k4a.cold_count}, hot={ctrl_k4a.hot_count}")

            # Timed runs
            k4a_times, k4a_fwds = [], []
            for ri in range(2):
                reset_ctrl(ctrl_k4a)
                dllm = make_dllm(decoder_t0)
                torch.cuda.synchronize()
                dist.barrier()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                      block_length=BLOCK_LENGTH)
                torch.cuda.synchronize()
                dist.barrier()
                t1 = time.perf_counter()
                k4a_times.append(t1 - t0)
                k4a_fwds.append(dllm.diff_iteration.num_forwards)
                if rank == 0:
                    print(f"    Run {ri+1}: {t1-t0:.3f}s, "
                          f"{dllm.diff_iteration.num_forwards} fwd, "
                          f"{(t1-t0)*1000/dllm.diff_iteration.num_forwards:.2f} ms/fwd, "
                          f"cold={ctrl_k4a.cold_count} hot={ctrl_k4a.hot_count}")

            # Quality check
            if do_quality:
                reset_ctrl(ctrl_k4a)
                dllm = make_dllm(decoder_t7)
                with torch.inference_mode():
                    _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                      block_length=BLOCK_LENGTH)
                reset_ctrl(ctrl_k4a)
                dllm = make_dllm(decoder_t7)
                with torch.inference_mode():
                    out_k4a = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                            block_length=BLOCK_LENGTH)
                torch.cuda.synchronize()
                if rank == 0:
                    gen_k4a = out_k4a[:, prompt_len:]
                    print(f"  Quality (temp=0.7):")
                    for bi in sorted(VERIFIABLE.keys()):
                        gt = gen_k4a[bi]
                        valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                        text = tokenizer.decode(valid, skip_special_tokens=True)
                        print(f"    #{bi} [{VERIFIABLE[bi][:30]}]: {text[:150]}")
            elif rank == 0:
                print(f"  Quality check skipped (batch={BATCH_SIZE} > 128)")

            results[f'C11_M5_K4_A2A_{alltoall_backend}'] = {
                'avg_time': sum(k4a_times) / 2,
                'avg_fwd': sum(k4a_fwds) / 2,
                'ms_per_fwd': sum(k4a_times) / 2 / (sum(k4a_fwds) / 2) * 1000,
            }

        # ========== Summary ==========
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"SUMMARY — {world_size}-GPU EP Benchmark")
            print(f"{'='*80}")
            print(f"  {'Config':<30s} {'Time(s)':>8s} {'Fwd':>5s} {'ms/fwd':>8s}")
            print(f"  {'-'*55}")
            for cn, r in results.items():
                print(f"  {cn:<30s} {r['avg_time']:>8.3f} {r['avg_fwd']:>5.0f} {r['ms_per_fwd']:>8.2f}")

            # Compare with single-GPU reference
            print(f"\n  Single-GPU reference (from previous runs):")
            print(f"    C5  1-GPU: 12.42s, 278 fwd, 44.69 ms/fwd")
            print(f"    C10 1-GPU: 12.02s, 280 fwd, 42.91 ms/fwd")

            out_path = REPO_ROOT / "codex_coding" / "results" / f"multi_gpu_{world_size}ep_b{BATCH_SIZE}_benchmark.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n  Saved to {out_path}")

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
