#!/usr/bin/env python3
"""
dp=2, tp=4, ep=8 Benchmark for LLaDA2.0-mini.

True AllToAll EP with DP: each TP group processes different data,
AllToAll dispatches tokens across all 8 GPUs to correct expert.

Config: C11-M5-K4 (TP attention + fused routing + EB M=5 + topk=4)

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter \
    torchrun --nproc_per_node=8 codex_coding/src/bench_dp2_tp4_ep8.py
"""

from __future__ import annotations
import sys; sys.modules['deep_ep'] = None  # Block broken deep_ep NCCL symbol (vllm 0.11.0)
import os, time, json, argparse
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gen-length", type=int, default=256)
    args = parser.parse_args()

    TP_SIZE = 4
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    assert world_size == 8, f"This script requires 8 GPUs, got {world_size}"

    dp_size = world_size // TP_SIZE  # 2
    dp_rank = rank // TP_SIZE        # 0 for rank 0-3, 1 for rank 4-7
    tp_rank_local = rank % TP_SIZE   # 0-3 within each TP group

    assert args.batch_size % dp_size == 0
    local_bs = args.batch_size // dp_size  # 256

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    alltoall_backend = os.environ.get("VLLM_ALL2ALL_BACKEND", "allgather_reducescatter")

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoTokenizer, AutoConfig
    from test_heteval512 import PROMPTS
    from test_heteval128 import VERIFIABLE
    from test_fused_eb_triton import fused_routing
    from test_m_skip_sweep import MSkipEBController
    from baseline_optimizations import apply_all_optimizations

    # --- Two-phase distributed init ---
    # Phase 1: init torch.distributed without DP port adjustment
    pcfg_init = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=True,
    )
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(world_size, rank, "env://", local_rank, "nccl")

    # Phase 2: create TP+DP+EP groups
    pcfg = ParallelConfig(
        tensor_parallel_size=TP_SIZE,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        enable_expert_parallel=True,
    )
    vllm_cfg = VllmConfig(parallel_config=pcfg)

    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(
            tensor_model_parallel_size=TP_SIZE, backend="nccl")

        # ★ Do NOT call _maybe_patch_all_reduce() — it replaces TP AllReduce
        # with world AllReduce, which is WRONG when dp>1 (world ≠ TP group)

        from vllm.distributed import (get_tp_group, get_dp_group, get_ep_group,
                                       get_tensor_model_parallel_world_size,
                                       get_tensor_model_parallel_rank)
        tp_ws = get_tensor_model_parallel_world_size()
        tp_rk = get_tensor_model_parallel_rank()
        dp_group = get_dp_group()
        ep_group = get_ep_group()

        if rank == 0:
            print("=" * 80)
            print(f"dp=2, tp=4, ep=8 Benchmark — {world_size} GPUs")
            print(f"  tp={tp_ws}, dp={dp_size}, ep={ep_group.world_size}")
            print(f"  batch={args.batch_size} (local={local_bs} per DP rank)")
            print(f"  gen={args.gen_length}, block={BLOCK_LENGTH}")
            print(f"  AllToAll backend: {alltoall_backend}")
            print("=" * 80)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)

        # Build model with TP=4 + EP=8
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        # Warmup (needs ForwardContext for dp>1)
        from vllm.forward_context import set_forward_context
        with torch.inference_mode():
            warmup_tokens = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                     num_tokens=warmup_tokens.numel()):
                _ = model(warmup_tokens, use_cache=False)

        apply_all_optimizations(model)

        # Initialize AllToAll communication buffers (MK)
        from vllm.distributed import prepare_communication_buffer_for_model
        prepare_communication_buffer_for_model(model)

        # Verify EP status
        for name, mod in model.named_modules():
            if hasattr(mod, 'expert_map') and mod.expert_map is not None:
                local_experts = (mod.expert_map >= 0).sum().item()
                if rank == 0:
                    print(f"  EP active: {local_experts} local experts per GPU "
                          f"(total {mod.expert_map.shape[0]})")
                break

        if rank == 0:
            mem_gb = torch.cuda.memory_allocated(device) / 1e9
            print(f"  GPU memory: {mem_gb:.1f} GB")

        # --- Build input: each DP rank gets different prompts ---
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

        # Slice for this DP rank
        my_start = dp_rank * local_bs
        my_input = input_ids_full[my_start : my_start + local_bs].to(device)
        prompt_len = my_input.shape[1]

        if rank == 0:
            print(f"  Full input: {input_ids_full.shape}, local: {my_input.shape}")

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

        # --- C11-M5-K4 routing patch ---
        ctrl = MSkipEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)

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

        def reset():
            ctrl.prev_N.clear(); ctrl.K_init.clear()
            ctrl.cold_count = 0; ctrl.hot_count = 0
            ctrl.eb_calls = 0; ctrl.eb_skips = 0
            ctrl._bufs.clear(); ctrl.k_init_history.clear()
            ctrl.s_mask_cache.clear(); ctrl.pop_cache.clear()
            ctrl._fwd_in_block.clear(); ctrl._block_idx.clear()

        # --- Warmup ---
        if rank == 0:
            print("\nWarmup...")
        reset()
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(my_input.clone(), gen_length=args.gen_length,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(f"  Warmup: {dllm.diff_iteration.num_forwards} fwd, "
                  f"cold={ctrl.cold_count}, hot={ctrl.hot_count}")

        # --- Timed runs ---
        times, fwds = [], []
        for ri in range(2):
            reset()
            dllm = make_dllm()
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                local_out = dllm.generate(my_input.clone(),
                                           gen_length=args.gen_length,
                                           block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            dist.barrier()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            fwds.append(dllm.diff_iteration.num_forwards)
            if rank == 0:
                print(f"  Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd, "
                      f"{(t1-t0)*1000/dllm.diff_iteration.num_forwards:.2f} ms/fwd, "
                      f"cold={ctrl.cold_count}, hot={ctrl.hot_count}")

        # --- Quality check (gather outputs on rank 0) ---
        gen_local = local_out[:, prompt_len:].contiguous()
        # Gather from all DP ranks (only need one representative per TP group)
        if tp_rank_local == 0:
            # Gather across DP group
            all_gen = [torch.zeros_like(gen_local) for _ in range(dp_size)]
            dist.all_gather(all_gen, gen_local, group=dp_group.device_group)
        else:
            all_gen = None

        if rank == 0:
            full_gen = torch.cat(all_gen, dim=0)
            print(f"\n  Quality (temp=0.0, first {min(5, len(VERIFIABLE))} verifiable):")
            for bi in sorted(VERIFIABLE.keys()):
                if bi >= full_gen.shape[0]:
                    continue
                gt = full_gen[bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                expected = VERIFIABLE[bi]
                passed = expected.lower().replace(" ", "") in text.lower().replace(" ", "")
                print(f"    #{bi} [{'PASS' if passed else 'FAIL'}]: {text[:150]}")

        # --- Summary ---
        avg_time = sum(times) / len(times)
        avg_fwd = sum(fwds) / len(fwds)
        ms_fwd = avg_time / avg_fwd * 1000

        if rank == 0:
            print(f"\n{'='*80}")
            print(f"SUMMARY — dp=2, tp=4, ep=8")
            print(f"  Config: C11-M5-K4 + AllToAll ({alltoall_backend})")
            print(f"{'='*80}")
            print(f"  Time:    {avg_time:.3f}s")
            print(f"  Fwd:     {avg_fwd:.0f}")
            print(f"  ms/fwd:  {ms_fwd:.2f}")
            print(f"  Throughput: {args.batch_size / avg_time:.1f} prompts/s")
            print(f"\n  Compare:")
            print(f"    C11-M5-K4 tp=4 ep=4 dp=1 b=512: 28.07s, 269 fwd, 104.35 ms/fwd")
            print(f"    C11-M5-K4 tp=4 ep=4 dp=1 b=128: ~ms/fwd ref")

            results = {
                'config': 'C11_M5_K4_dp2_tp4_ep8',
                'tp_size': TP_SIZE, 'dp_size': dp_size,
                'ep_size': ep_group.world_size,
                'batch_size': args.batch_size, 'local_bs': local_bs,
                'gen_length': args.gen_length,
                'backend': alltoall_backend,
                'avg_time': avg_time, 'avg_fwd': avg_fwd, 'ms_per_fwd': ms_fwd,
                'times': times, 'fwds': fwds,
            }
            out_path = REPO_ROOT / "codex_coding" / "results" / "dp2_tp4_ep8_benchmark.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n  Saved to {out_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
