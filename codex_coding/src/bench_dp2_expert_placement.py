#!/usr/bin/env python3
"""
Popularity-based expert placement for dp=2 tp=4 ep=8.

Approach: load model twice.
1st load (linear placement) → warmup → collect popularity
2nd load (popularity round-robin placement) → benchmark

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=naive \
    torchrun --nproc_per_node=8 codex_coding/src/bench_dp2_expert_placement.py
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


def compute_round_robin_placement(popularity, ep_size):
    """Sort experts by popularity, assign round-robin to GPUs.
    Returns expert_to_gpu[expert_id] = gpu_idx.
    """
    sorted_experts = np.argsort(-popularity)
    expert_to_gpu = np.zeros(NUM_EXPERTS, dtype=np.int64)
    for rank_i, eid in enumerate(sorted_experts):
        expert_to_gpu[eid] = rank_i % ep_size
    return expert_to_gpu


def load_model_with_placement(config, device, ep_rank, ep_size, expert_to_gpu=None):
    """Load model, optionally with custom expert placement.
    expert_to_gpu: [256] array mapping expert_id → gpu_idx. None = linear.
    """
    from dinfer.model import LLaDA2MoeModelLM
    import dinfer.model.modeling_llada2_moe as mmoe

    if expert_to_gpu is not None:
        # Find which experts belong to this ep_rank
        my_experts = sorted([eid for eid in range(NUM_EXPERTS)
                             if expert_to_gpu[eid] == ep_rank])
        my_expert_set = set(my_experts)
        # local index mapping: expert_id → local_idx
        eid_to_local = {eid: i for i, eid in enumerate(my_experts)}
    else:
        epg = NUM_EXPERTS // ep_size
        my_experts = list(range(ep_rank * epg, (ep_rank + 1) * epg))
        my_expert_set = set(my_experts)
        eid_to_local = {eid: eid - ep_rank * epg for eid in my_experts}

    # Monkey-patch _get_ep_rank_and_size to return correct ep info
    original_get_ep = mmoe._get_ep_rank_and_size

    model = LLaDA2MoeModelLM(config=config).eval()

    if expert_to_gpu is not None:
        # Custom load: override load_state_dict to use custom placement
        _original_load = model.load_state_dict

        def custom_load_state_dict(model_dir, strict=True, dtype=torch.bfloat16, device=None):
            """Modified load that uses custom expert-to-GPU mapping."""
            import tqdm
            from safetensors.torch import load_file
            from pathlib import Path
            from vllm.distributed import divide

            num_experts = config.num_experts
            moe_intermediate_size = config.moe_intermediate_size
            num_layers = config.num_hidden_layers

            index_path = Path(model_dir) / "model.safetensors.index.json"
            with open(index_path, "r") as f:
                index = json.load(f)

            weight_map = index["weight_map"]
            shard_files = {v for v in weight_map.values()}

            state_dict = {}
            for shard in tqdm.tqdm(sorted(shard_files), disable=(dist.get_rank() != 0)):
                shard_path = Path(model_dir) / shard
                with torch.inference_mode():
                    file_state_dict = load_file(str(shard_path))
                    filtered = {}
                    for key, value in file_state_dict.items():
                        if ".mlp.experts." in key:
                            expert_id = int(key.split(".mlp.experts.")[1].split(".")[0])
                            if expert_id in my_expert_set:
                                filtered[key] = value
                        else:
                            filtered[key] = value
                    state_dict.update(filtered)

            new_state_dict = {}
            gate_projs = [{} for _ in range(num_layers)]
            up_projs = [{} for _ in range(num_layers)]
            down_projs = [{} for _ in range(num_layers)]

            for key, value in state_dict.items():
                if ".mlp.experts." in key:
                    layer_id = int(key.split(".mlp.experts.")[0].split(".")[-1])
                    expert_id = int(key.split(".mlp.experts.")[1].split(".")[0])
                    if layer_id < num_layers and expert_id in my_expert_set:
                        local_idx = eid_to_local[expert_id]
                        if "gate_proj" in key:
                            gate_projs[layer_id][local_idx] = value
                        elif "up_proj" in key:
                            up_projs[layer_id][local_idx] = value
                        elif "down_proj" in key:
                            down_projs[layer_id][local_idx] = value
                else:
                    new_state_dict[key] = value

            del state_dict
            num_local = len(my_experts)
            for layer_id in range(num_layers):
                if 0 in gate_projs[layer_id]:
                    w13_weight = []
                    w2_weight = []
                    for local_idx in range(num_local):
                        gate_proj = gate_projs[layer_id][local_idx].to(device)
                        up_proj = up_projs[layer_id][local_idx].to(device)
                        down_proj = down_projs[layer_id][local_idx].to(device)
                        w13_weight.append(torch.cat([gate_proj, up_proj], dim=0))
                        w2_weight.append(down_proj)
                    w13_weight = torch.stack(w13_weight, dim=0)
                    w2_weight = torch.stack(w2_weight, dim=0)
                    new_state_dict[f"model.layers.{layer_id}.mlp.experts.w13_weight"] = w13_weight.contiguous().to(device)
                    new_state_dict[f"model.layers.{layer_id}.mlp.experts.w2_weight"] = w2_weight.contiguous().to(device)

            # Build custom expert_map
            for layer_id in range(num_layers):
                key = f"model.layers.{layer_id}.mlp.experts.expert_map"
                if key in dict(model.named_buffers()):
                    emap = torch.full((NUM_EXPERTS,), -1, dtype=torch.int32)
                    for eid in my_experts:
                        emap[eid] = eid_to_local[eid]
                    new_state_dict[key] = emap.to(device)

            # Load into model — use weight_loader for TP-sharded params
            for key, value in new_state_dict.items():
                new_state_dict[key] = value.to(device)
            params_dict = dict(model.named_parameters())
            buffer_dict = dict(model.named_buffers())
            for name, loaded_weight in new_state_dict.items():
                if name in params_dict:
                    param = params_dict[name]
                    wl = getattr(param, 'weight_loader', None)
                    if wl is not None and 'query_key_value' in name:
                        # TP attention: split combined QKV and use weight_loader
                        hd = config.head_dim or (config.hidden_size // config.num_attention_heads)
                        tq = config.num_attention_heads * hd
                        tkv = config.num_key_value_heads * hd
                        wl(param, loaded_weight[:tq], "q")
                        wl(param, loaded_weight[tq:tq+tkv], "k")
                        wl(param, loaded_weight[tq+tkv:], "v")
                    elif wl is not None:
                        # RowParallelLinear (dense), ColumnParallelLinear (lm_head), etc.
                        try:
                            wl(param, loaded_weight)
                        except Exception:
                            if param.shape == loaded_weight.shape:
                                param.data = loaded_weight
                    else:
                        if param.shape == loaded_weight.shape:
                            param.data = loaded_weight
                elif name in buffer_dict:
                    buffer_dict[name].data = loaded_weight
            for name, param in model.named_parameters():
                if '.mlp.gate.expert_bias' in name:
                    param.data = param.data.to(torch.float32)
                else:
                    param.data = param.data.to(dtype)

            model.init_h2e_module()

        custom_load_state_dict(MODEL_PATH, strict=False, dtype=torch.bfloat16, device=device)
    else:
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)

    model = model.to(device)
    return model


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    BATCH_SIZE = 512
    GEN_LENGTH = 256

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    dp_size = world_size // TP_SIZE
    dp_rank = rank // TP_SIZE
    local_bs = BATCH_SIZE // dp_size
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoTokenizer, AutoConfig
    from test_fused_eb_triton import fused_routing
    from test_m_skip_sweep import MSkipEBController
    from baseline_optimizations import apply_all_optimizations
    from test_heteval512 import PROMPTS

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
        ep_rank = get_ep_group().rank_in_group

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

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

        if rank == 0:
            print(f"dp={dp_size} tp={TP_SIZE} ep={ep_size} ep_rank={ep_rank}")

        # ========== Phase 1: Linear placement → collect popularity ==========
        if rank == 0:
            print("\n=== Phase 1: Linear placement + warmup ===")
        model = load_model_with_placement(config, device, ep_rank, ep_size, expert_to_gpu=None)
        from vllm.forward_context import set_forward_context
        with torch.inference_mode():
            w = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                     num_tokens=w.numel()):
                _ = model(w, use_cache=False)
        apply_all_optimizations(model)
        prepare_communication_buffer_for_model(model)

        # Collect popularity
        ctrl = MSkipEBController(num_layers=19, K=8, M=4, K_target=40,
                                  quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)
        pop_counts = torch.zeros(NUM_EXPERTS, device=device, dtype=torch.float32)

        gate_idx = 0
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                     mod.top_k, mod.n_group, mod.topk_group)
                li = gate_idx
                def mk_pop(bb, rr, nn, gg, layer_i, cc, pc):
                    def fn(hs, go, topk, renorm):
                        sm = cc.get_s_mask(layer_i, go, bb)
                        w, idx = fused_routing(go, bb, rr, s_mask=sm, K=4, ng=nn, tkg=gg)
                        flat = idx.flatten()
                        pc.scatter_add_(0, flat.long(), torch.ones_like(flat, dtype=torch.float32))
                        return w.to(go.dtype), idx
                    return fn
                mod.routing = mk_pop(b, r, ng, tkg, li, ctrl, pop_counts)
                gate_idx += 1

        decoder = ThresholdParallelDecoder(temperature=0.0, threshold=0.90,
                                           mask_id=MASK_ID, eos_id=EOS_ID)
        dllm = BlockDiffusionLLM(model, decoder, BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=4, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)
        with torch.inference_mode():
            _ = dllm.generate(my_input.clone(), gen_length=64, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        # AllReduce popularity across all ranks
        dist.all_reduce(pop_counts, op=dist.ReduceOp.SUM)
        popularity = pop_counts.cpu().numpy()

        epg = NUM_EXPERTS // ep_size
        if rank == 0:
            print(f"\n  Linear placement per-GPU popularity:")
            for g in range(ep_size):
                gpu_pop = popularity[g*epg:(g+1)*epg].sum()
                print(f"    GPU{g}: {gpu_pop:.0f} ({gpu_pop/popularity.sum()*100:.1f}%)")

        # Compute new placement
        expert_to_gpu = compute_round_robin_placement(popularity, ep_size)

        if rank == 0:
            print(f"\n  Round-robin placement per-GPU popularity:")
            for g in range(ep_size):
                gpu_experts = [eid for eid in range(NUM_EXPERTS) if expert_to_gpu[eid] == g]
                gpu_pop = sum(popularity[eid] for eid in gpu_experts)
                print(f"    GPU{g}: {gpu_pop:.0f} ({gpu_pop/popularity.sum()*100:.1f}%)")

        # Free phase 1 model
        del model, dllm
        torch.cuda.empty_cache()

        # Clear vllm's static_forward_context to allow re-creating FusedMoE layers
        from vllm.config import get_current_vllm_config
        cc = get_current_vllm_config().compilation_config
        cc.static_forward_context.clear()

        # ========== Phase 2: Popularity placement → benchmark ==========
        if rank == 0:
            print("\n=== Phase 2: Popularity placement + benchmark ===")

        model = load_model_with_placement(config, device, ep_rank, ep_size,
                                          expert_to_gpu=expert_to_gpu)
        with torch.inference_mode():
            w = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                     num_tokens=w.numel()):
                _ = model(w, use_cache=False)
        apply_all_optimizations(model)
        prepare_communication_buffer_for_model(model)

        # Verify expert_map
        for name, mod in model.named_modules():
            if hasattr(mod, 'expert_map') and mod.expert_map is not None:
                local_count = (mod.expert_map >= 0).sum().item()
                if rank == 0:
                    print(f"  EP: {local_count} local experts per GPU")
                break

        # C11-M5-K4 patch
        ctrl = MSkipEBController(num_layers=19, K=8, M=4, K_target=40,
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

        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # Warmup
        reset()
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(my_input.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(f"  Warmup: {dllm.diff_iteration.num_forwards} fwd")

        # Timed runs
        times, fwds = [], []
        for ri in range(2):
            reset()
            dllm = make_dllm()
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                out = dllm.generate(my_input.clone(), gen_length=GEN_LENGTH,
                                     block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            dist.barrier()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            fwds.append(dllm.diff_iteration.num_forwards)
            if rank == 0:
                print(f"  Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd, "
                      f"{(t1-t0)*1000/dllm.diff_iteration.num_forwards:.2f} ms/fwd")

        # Quality
        prompt_len = my_input.shape[1]
        if rank == 0:
            gen = out[:, prompt_len:]
            print(f"\n  Output samples:")
            for bi in [0, 28]:
                if bi < gen.shape[0]:
                    gt = gen[bi]
                    valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                    text = tokenizer.decode(valid, skip_special_tokens=True)
                    print(f"    #{bi}: {text[:200]}")

        avg_time = sum(times) / len(times)
        avg_fwd = sum(fwds) / len(fwds)
        ms_fwd = avg_time / avg_fwd * 1000

        if rank == 0:
            print(f"\n{'='*70}")
            print(f"SUMMARY — Popularity Placement dp=2 tp=4 ep=8")
            print(f"{'='*70}")
            print(f"  Time:    {avg_time:.3f}s | Fwd: {avg_fwd:.0f} | ms/fwd: {ms_fwd:.2f}")
            print(f"  Compare: linear placement: 19.89s, 266 fwd, 74.76 ms/fwd")
            delta = (ms_fwd - 74.76) / 74.76 * 100
            print(f"  Delta:   {delta:+.1f}%")

            results = {
                'config': 'C11_M5_K4_dp2_tp4_ep8_popularity',
                'avg_time': avg_time, 'avg_fwd': avg_fwd, 'ms_per_fwd': ms_fwd,
                'linear_ref_ms': 74.76,
            }
            out_path = REPO_ROOT / "codex_coding" / "results" / "dp2_expert_placement.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  Saved to {out_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
