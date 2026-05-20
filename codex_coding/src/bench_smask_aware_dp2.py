#!/usr/bin/env python3
"""
SMaskAware dp=2 tp=4 ep=8 — efficient comm + route-before-dispatch.

Three configs compared:
  A) C12 baseline (original naive backend: broadcast loop + all_reduce)
  B) C12 + efficient_comm (all_gather replaces broadcast loop, same routing flow)
  C) C12 + smask_aware (B + hot path: route locally → gather topk results)

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=naive \
    torchrun --nproc_per_node=8 codex_coding/src/bench_smask_aware_dp2.py
"""

from __future__ import annotations
import os, sys, time, json, argparse
from pathlib import Path
from functools import partial

import torch
import torch.distributed as dist

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"


# ========================================================================
# Efficient all_gather / combine (replaces naive broadcast loop)
# ========================================================================

def all_gather_2d(local_tensor, dp_pg, dp_size):
    """1x all_gather of [N, D] → [N*dp_size, D].  Zero-copy: pre-alloc contiguous buffer."""
    N, D = local_tensor.shape
    buf = torch.empty(N * dp_size, D, dtype=local_tensor.dtype, device=local_tensor.device)
    # chunk() returns views into buf — no extra allocation
    dist.all_gather(list(buf.chunk(dp_size)), local_tensor.contiguous(), group=dp_pg)
    return buf


def combine_allreduce_slice(full_out, dp_rank, local_N, dp_pg):
    """all_reduce then slice — same semantic as NaiveAll2AllManager.combine."""
    dist.all_reduce(full_out, group=dp_pg)
    start = dp_rank * local_N
    return full_out[start : start + local_N]


# ========================================================================
# Config B: efficient comm, routing AFTER dispatch (same semantics as A)
# ========================================================================

def make_forward_B(block, layer_id, gp, ctrl, dp_pg, dp_rank, dp_size):
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.distributed import tensor_model_parallel_all_reduce
    from test_fused_eb_triton import fused_routing
    shared = block.shared_experts if block.config.num_shared_experts else None
    exp = block.experts

    def fwd(hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hs = hidden_states.view(-1, h)
        local_N = hs.shape[0]

        sh = shared(hs) if shared is not None else None
        rl = block.gate.get_logits(hs)

        # Efficient dispatch: 2x all_gather (instead of 4x broadcast)
        g_hs = all_gather_2d(hs, dp_pg, dp_size)
        g_rl = all_gather_2d(rl, dp_pg, dp_size)

        # Route on gathered 512 tokens (same as baseline)
        sm = ctrl.get_s_mask(layer_id, g_rl, gp["bias"])
        tw, ti = fused_routing(g_rl, gp["bias"], gp["rsf"],
                               s_mask=sm, K=4, ng=gp["ng"], tkg=gp["tkg"])

        y = fused_experts(g_hs, exp.w13_weight, exp.w2_weight,
                          tw.to(g_rl.dtype), ti, inplace=True,
                          global_num_experts=256, expert_map=exp.expert_map)

        # Combine: DP all_reduce + slice, then TP all_reduce
        y = combine_allreduce_slice(y, dp_rank, local_N, dp_pg)
        y = tensor_model_parallel_all_reduce(y)

        if sh is not None:
            y = y + sh
        return y.view(bsz, seq_len, h)
    return fwd


# ========================================================================
# Config C: smask_aware — hot path routes BEFORE dispatch
# ========================================================================

def make_forward_C(block, layer_id, gp, ctrl, dp_pg, dp_rank, dp_size):
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.distributed import tensor_model_parallel_all_reduce
    from test_fused_eb_triton import fused_routing
    shared = block.shared_experts if block.config.num_shared_experts else None
    exp = block.experts

    def fwd(hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hs = hidden_states.view(-1, h)
        local_N = hs.shape[0]

        sh = shared(hs) if shared is not None else None
        rl = block.gate.get_logits(hs)

        # Peek: will the next get_s_mask call be hot?
        fib = ctrl._fwd_in_block.get(layer_id, -1)
        pN = ctrl.prev_N.get(layer_id, -1)
        new_blk = (pN < 0 or local_N > pN)
        nxt = 0 if new_blk else fib + 1
        hot = (nxt % ctrl.skip_m != 0) and (layer_id in ctrl.s_mask_cache)

        if hot:
            # --- HOT: route locally (256 tokens), gather results ---
            sm = ctrl.get_s_mask(layer_id, rl, gp["bias"])
            tw, ti = fused_routing(rl, gp["bias"], gp["rsf"],
                                   s_mask=sm, K=4, ng=gp["ng"], tkg=gp["tkg"])
            tw = tw.to(rl.dtype)

            g_hs = all_gather_2d(hs, dp_pg, dp_size)
            g_ti = all_gather_2d(ti.int(), dp_pg, dp_size).to(ti.dtype)
            g_tw = all_gather_2d(tw, dp_pg, dp_size)

            y = fused_experts(g_hs, exp.w13_weight, exp.w2_weight,
                              g_tw, g_ti, inplace=True,
                              global_num_experts=256, expert_map=exp.expert_map)
        else:
            # --- COLD: gather first, route on 512 tokens ---
            g_hs = all_gather_2d(hs, dp_pg, dp_size)
            g_rl = all_gather_2d(rl, dp_pg, dp_size)

            sm = ctrl.get_s_mask(layer_id, g_rl, gp["bias"])
            tw, ti = fused_routing(g_rl, gp["bias"], gp["rsf"],
                                   s_mask=sm, K=4, ng=gp["ng"], tkg=gp["tkg"])

            y = fused_experts(g_hs, exp.w13_weight, exp.w2_weight,
                              tw.to(g_rl.dtype), ti, inplace=True,
                              global_num_experts=256, expert_map=exp.expert_map)

        # Combine: DP all_reduce + slice, then TP all_reduce
        y = combine_allreduce_slice(y, dp_rank, local_N, dp_pg)
        y = tensor_model_parallel_all_reduce(y)

        if sh is not None:
            y = y + sh
        return y.view(bsz, seq_len, h)
    return fwd


# ========================================================================
# Main
# ========================================================================

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
    assert world_size == 8, f"Requires 8 GPUs, got {world_size}"

    dp_size = world_size // TP_SIZE
    dp_rank = rank // TP_SIZE
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeSparseMoeBlock
    from transformers import AutoTokenizer, AutoConfig
    from test_heteval512 import PROMPTS
    from test_fused_eb_triton import fused_routing
    from test_m_skip_sweep import MSkipEBController
    from baseline_optimizations import apply_all_optimizations

    # --- Distributed init ---
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

        from vllm.distributed import get_dp_group, get_ep_group
        dp_group = get_dp_group()

        if rank == 0:
            print("=" * 80)
            print(f"SMaskAware Benchmark — dp=2 tp=4 ep=8, {world_size} GPUs")
            print(f"  batch={args.batch_size}, gen={args.gen_length}")
            print("=" * 80)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)

        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        from vllm.forward_context import set_forward_context
        with torch.inference_mode():
            warmup_tok = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                     num_tokens=warmup_tok.numel()):
                _ = model(warmup_tok, use_cache=False)

        apply_all_optimizations(model)
        from vllm.distributed import prepare_communication_buffer_for_model
        prepare_communication_buffer_for_model(model)

        if rank == 0:
            print(f"  GPU memory: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

        # --- Collect MoE blocks and gate params ---
        moe_blocks = []
        gate_params = []
        for name, mod in model.named_modules():
            if isinstance(mod, LLaDA2MoeSparseMoeBlock):
                moe_blocks.append(mod)
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                gate_params.append({"bias": mod.expert_bias, "rsf": mod.routed_scaling_factor,
                                    "ng": mod.n_group, "tkg": mod.topk_group})

        orig_forwards = [b.forward for b in moe_blocks]
        dp_pg = dp_group.device_group

        # --- Input ---
        local_bs = args.batch_size // dp_size
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
        my_input = input_ids_full[dp_rank * local_bs : (dp_rank + 1) * local_bs].to(device)
        prompt_len = my_input.shape[1]

        decoder = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # --- Config setup helpers ---
        def setup_routing(ctrl_ref):
            gi = 0
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate":
                    b, r, ng, tkg = mod.expert_bias, mod.routed_scaling_factor, mod.n_group, mod.topk_group
                    li = gi
                    def mk(bb, rr, nn, gg, layer_i, cc):
                        def fn(hs, go, topk, renorm):
                            sm = cc.get_s_mask(layer_i, go, bb)
                            w, idx = fused_routing(go, bb, rr, s_mask=sm, K=4, ng=nn, tkg=gg)
                            return w.to(go.dtype), idx
                        return fn
                    mod.routing = mk(b, r, ng, tkg, li, ctrl_ref)
                    gi += 1

        def setup_A(ctrl_ref):
            """Config A: original naive backend flow."""
            setup_routing(ctrl_ref)
            for blk, of in zip(moe_blocks, orig_forwards):
                blk.forward = of

        def setup_B(ctrl_ref):
            """Config B: efficient all_gather, routing after dispatch."""
            for i, blk in enumerate(moe_blocks):
                blk.forward = make_forward_B(blk, i, gate_params[i], ctrl_ref,
                                             dp_pg, dp_rank, dp_size)

        def setup_C(ctrl_ref):
            """Config C: efficient all_gather + hot path route-before-dispatch."""
            setup_routing(ctrl_ref)  # cold path still uses gate.routing
            for i, blk in enumerate(moe_blocks):
                blk.forward = make_forward_C(blk, i, gate_params[i], ctrl_ref,
                                             dp_pg, dp_rank, dp_size)

        def reset(ctrl_ref):
            ctrl_ref.prev_N.clear(); ctrl_ref.K_init.clear()
            ctrl_ref.cold_count = 0; ctrl_ref.hot_count = 0
            ctrl_ref.eb_calls = 0; ctrl_ref.eb_skips = 0
            ctrl_ref._bufs.clear(); ctrl_ref.k_init_history.clear()
            ctrl_ref.s_mask_cache.clear(); ctrl_ref.pop_cache.clear()
            ctrl_ref._fwd_in_block.clear(); ctrl_ref._block_idx.clear()

        def run_config(label, ctrl_ref, setup_fn, num_runs=2):
            if rank == 0:
                print(f"\n--- {label} ---")
            setup_fn(ctrl_ref)

            # Warmup
            reset(ctrl_ref)
            dllm = make_dllm()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(my_input.clone(), gen_length=args.gen_length,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize(); dist.barrier()
            if rank == 0:
                print(f"  Warmup: {dllm.diff_iteration.num_forwards} fwd")

            times, fwds = [], []
            for ri in range(num_runs):
                reset(ctrl_ref)
                dllm = make_dllm()
                torch.cuda.synchronize(); dist.barrier()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    out = dllm.generate(my_input.clone(), gen_length=args.gen_length,
                                        block_length=BLOCK_LENGTH)
                torch.cuda.synchronize(); dist.barrier()
                dt = time.perf_counter() - t0
                nf = dllm.diff_iteration.num_forwards
                times.append(dt); fwds.append(nf)
                if rank == 0:
                    print(f"  Run {ri+1}: {dt:.3f}s, {nf} fwd, {dt*1000/nf:.2f} ms/fwd, "
                          f"cold={ctrl_ref.cold_count} hot={ctrl_ref.hot_count}")

            # Quality: print first 3 outputs
            if rank == 0:
                gen = out[:, prompt_len:]
                print(f"  Quality ({label}):")
                for bi in range(min(3, gen.shape[0])):
                    toks = gen[bi]
                    valid = toks[(toks != 0) & (toks != EOS_ID) & (toks != MASK_ID)]
                    text = tokenizer.decode(valid, skip_special_tokens=True)
                    print(f"    #{bi}: {text[:200]}")

            t = sum(times)/len(times)
            f = sum(fwds)/len(fwds)
            return {"config": label, "time_s": t, "fwd": f, "ms_fwd": t/f*1000}

        # --- Run all three configs ---
        ctrl_a = MSkipEBController(num_layers=19, K=8, M=4, K_target=40,
                                   quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)
        ctrl_b = MSkipEBController(num_layers=19, K=8, M=4, K_target=40,
                                   quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)
        ctrl_c = MSkipEBController(num_layers=19, K=8, M=4, K_target=40,
                                   quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)

        ra = run_config("A) C12 baseline (naive)", ctrl_a, setup_A)
        rb = run_config("B) efficient comm", ctrl_b, setup_B)
        rc = run_config("C) smask_aware", ctrl_c, setup_C)

        if rank == 0:
            print(f"\n{'='*70}")
            print(f"SUMMARY — batch={args.batch_size}, gen={args.gen_length}")
            print(f"{'='*70}")
            print(f"{'Config':<30} {'Time':>8} {'Fwd':>5} {'ms/fwd':>8} {'vs A':>8}")
            print(f"{'-'*61}")
            ms_a = ra["ms_fwd"]
            for r in [ra, rb, rc]:
                delta = f"{(r['ms_fwd']/ms_a-1)*100:+.1f}%" if r != ra else "  —"
                print(f"{r['config']:<30} {r['time_s']:>7.2f}s {r['fwd']:>5.0f} "
                      f"{r['ms_fwd']:>7.2f} {delta:>8}")

            results = {"batch_size": args.batch_size, "gen_length": args.gen_length,
                       "A": ra, "B": rb, "C": rc,
                       "B_vs_A_pct": (rb["ms_fwd"]/ms_a-1)*100,
                       "C_vs_A_pct": (rc["ms_fwd"]/ms_a-1)*100}
            out = REPO_ROOT / "codex_coding" / "results" / "smask_aware_dp2_results.json"
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n  Saved to {out}")

    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
