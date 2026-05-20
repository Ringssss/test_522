#!/usr/bin/env python3
"""
v0.1.15.9 — Multi-GPU DP + AllToAll EP Benchmark (Phase 1: Naive)

dp=N, tp=1, ep=N. Each GPU processes batch/N prompts independently.
MoE layers use AllToAll dispatch/combine (naive AllGather+AllReduce backend).

Usage:
  # 2-GPU minimal test
  torchrun --nproc_per_node=2 codex_coding/src/bench_multi_gpu_dp.py \
      --batch-size 4 --gen-length 32

  # 8-GPU full HetEval-128
  torchrun --nproc_per_node=8 codex_coding/src/bench_multi_gpu_dp.py
"""

from __future__ import annotations
import os, sys, time, json, argparse
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEFAULT_BATCH_SIZE = 128
DEFAULT_GEN_LENGTH = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gen-length", type=int, default=DEFAULT_GEN_LENGTH)
    parser.add_argument("--no-optimizations", action="store_true",
                        help="Skip baseline optimizations (fused RMSNorm + flash-attn)")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Select AllToAll backend: naive for Phase 1
    os.environ.setdefault("VLLM_ALL2ALL_BACKEND", "naive")

    # Distributed setup via torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    assert args.batch_size % world_size == 0, \
        f"batch_size {args.batch_size} must be divisible by world_size {world_size}"
    local_bs = args.batch_size // world_size

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from dinfer.model.modeling_llada2_moe import _maybe_patch_all_reduce
    from transformers import AutoTokenizer, AutoConfig
    from test_heteval128 import PROMPTS, VERIFIABLE

    # --- DP + EP initialization (two-phase) ---
    # Phase 1: init distributed with dp=1 to avoid vllm's DP rank/port adjustment
    pcfg_init = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=True,
    )
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(world_size, rank, "env://", local_rank, "nccl")

    # Phase 2: set dp=world_size and create DP/EP groups
    pcfg = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=world_size,
        data_parallel_rank=rank,
        enable_expert_parallel=True,
    )
    vllm_cfg = VllmConfig(parallel_config=pcfg)

    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(
            tensor_model_parallel_size=1, backend="nccl")

        # Apply conditional monkey-patch (should be no-op with tp=1)
        _maybe_patch_all_reduce()

        # Verify group setup
        from vllm.distributed import (get_tp_group, get_dp_group, get_ep_group,
                                       get_tensor_model_parallel_world_size)
        tp_size = get_tensor_model_parallel_world_size()
        dp_group = get_dp_group()
        ep_group = get_ep_group()

        if rank == 0:
            print("=" * 80)
            print(f"DP + AllToAll EP Benchmark — {world_size} GPUs")
            print(f"  tp_size={tp_size}, dp_size={world_size}, ep_size={ep_group.world_size}")
            print(f"  batch={args.batch_size} (local={local_bs}), "
                  f"gen={args.gen_length}, block={BLOCK_LENGTH}")
            print(f"  AllToAll backend: {os.environ.get('VLLM_ALL2ALL_BACKEND', 'naive')}")
            print("=" * 80)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)

        # Build model — weight loading now uses EP rank via _get_ep_rank_and_size()
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        # Warmup forward to trigger JIT (needs ForwardContext with dp>1)
        from vllm.forward_context import set_forward_context
        with torch.inference_mode():
            warmup_tokens = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                     num_tokens=warmup_tokens.numel()):
                _ = model(warmup_tokens, use_cache=False)

        # Apply baseline optimizations
        if not args.no_optimizations:
            from baseline_optimizations import apply_all_optimizations
            n_rms, n_fa = apply_all_optimizations(model)
            if rank == 0:
                print(f"  Baseline optimizations: Fused RMSNorm={n_rms}, Flash-attn={n_fa}")
        else:
            if rank == 0:
                print("  [No optimizations applied]")

        # ★★★ Initialize DeepEP/pplx communication buffers for modular kernel
        from vllm.distributed import prepare_communication_buffer_for_model
        prepare_communication_buffer_for_model(model)

        # If USE_DEEPEP_V2=1, replace PrepareAndFinalize with V2 ElasticBuffer
        if int(os.environ.get("USE_DEEPEP_V2", "0")):
            from deepep_v2_pf import replace_with_deepep_v2
            n_replaced, v2_sms = replace_with_deepep_v2(
                model, ep_group.cpu_group,
                num_local_experts=256 // world_size,
                num_experts=256, top_k=8, hidden=2048,
                max_tokens_per_rank=local_bs * 300,
            )
            if rank == 0:
                print(f"  V2 ElasticBuffer: replaced {n_replaced} layers, "
                      f"num_sms={v2_sms}")

        # If USE_V1_OPT=1, replace with async-optimized V1
        if int(os.environ.get("USE_V1_OPT", "0")):
            from deepep_v1_optimized_pf import replace_with_optimized_v1
            v1_sms = int(os.environ.get("V1_NUM_SMS", "0")) or None
            n_replaced = replace_with_optimized_v1(model, num_sms=v1_sms)
            if rank == 0:
                print(f"  V1 Optimized: replaced {n_replaced} layers"
                      + (f", num_sms={v1_sms}" if v1_sms else ""))

        # Verify EP status
        for name, mod in model.named_modules():
            if hasattr(mod, 'expert_map') and mod.expert_map is not None:
                local_experts = (mod.expert_map >= 0).sum().item()
                if rank == 0:
                    print(f"  EP active: {local_experts} local experts per GPU "
                          f"(total {mod.expert_map.shape[0]})")
                assert local_experts > 0, \
                    f"Expected >0 local experts, got {local_experts}"
                break

        # --- Build input: each rank gets its slice ---
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

        # Slice for this rank
        my_start = rank * local_bs
        my_input = input_ids_full[my_start : my_start + local_bs].to(device)
        prompt_len = my_input.shape[1]

        if rank == 0:
            print(f"  Full input: {input_ids_full.shape}, local: {my_input.shape}")
            mem_alloc = torch.cuda.memory_allocated(device) / 1e9
            print(f"  GPU {local_rank} memory: {mem_alloc:.1f} GB")

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

        # ========== C5 (fused routing, no EB) ==========
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"  C5: Fused routing only (DP={world_size}, AllToAll EP)")
            print(f"{'='*60}")

        # Patch routing with fused_routing (no EB)
        from test_fused_eb_triton import fused_routing
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

        # Warmup
        if rank == 0:
            print("  Warmup...")
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(my_input.clone(), gen_length=args.gen_length,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd")

        # Timed runs
        times, fwds = [], []
        for ri in range(2):
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
                print(f"    Run {ri+1}: {t1-t0:.3f}s, "
                      f"{dllm.diff_iteration.num_forwards} fwd, "
                      f"{(t1-t0)*1000/dllm.diff_iteration.num_forwards:.2f} ms/fwd")

        # Gather outputs on rank 0 for quality check
        gen_tokens_local = local_out[:, prompt_len:].contiguous()
        all_gen = [torch.zeros_like(gen_tokens_local) for _ in range(world_size)]
        dist.all_gather(all_gen, gen_tokens_local)

        if rank == 0:
            full_gen = torch.cat(all_gen, dim=0)  # [batch_size, gen_len]
            print(f"\n  Quality check (verifiable prompts, temp=0.0):")
            for bi in sorted(VERIFIABLE.keys()):
                if bi < full_gen.shape[0]:
                    gt = full_gen[bi]
                    valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                    text = tokenizer.decode(valid, skip_special_tokens=True)
                    print(f"    #{bi} [{VERIFIABLE[bi][:30]}]: {text[:150]}")

        # ========== Summary ==========
        avg_time = sum(times) / len(times)
        avg_fwd = sum(fwds) / len(fwds)
        ms_fwd = avg_time / avg_fwd * 1000

        if rank == 0:
            print(f"\n{'='*80}")
            print(f"SUMMARY — DP={world_size} AllToAll EP")
            print(f"{'='*80}")
            print(f"  Time:    {avg_time:.3f}s")
            print(f"  Fwd:     {avg_fwd:.0f}")
            print(f"  ms/fwd:  {ms_fwd:.2f}")
            print(f"\n  Compare with AllReduce EP ({world_size}-GPU, same batch):")
            print(f"    C5 AllReduce: 11.14s, 282 fwd, 39.49 ms/fwd (8-GPU)")
            print(f"    C5 1-GPU:     12.42s, 278 fwd, 44.69 ms/fwd")

            results = {
                'config': f'C5_DP{world_size}_AllToAll_naive',
                'dp_size': world_size, 'tp_size': 1, 'ep_size': world_size,
                'batch_size': args.batch_size, 'local_bs': local_bs,
                'gen_length': args.gen_length,
                'avg_time': avg_time, 'avg_fwd': avg_fwd, 'ms_per_fwd': ms_fwd,
                'times': times, 'fwds': fwds,
                'backend': os.environ.get('VLLM_ALL2ALL_BACKEND', 'naive'),
            }
            out_path = REPO_ROOT / "codex_coding" / "results" / \
                f"multi_gpu_dp{world_size}_alltoall_benchmark.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n  Saved to {out_path}")

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
