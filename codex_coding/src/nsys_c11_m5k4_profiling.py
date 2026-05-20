#!/usr/bin/env python3
"""
v0.1.15.10 — nsys profiling for C11-M5-K4 AGR, 4-GPU TP, batch=512

MoE-focused component breakdown with NVTX markers + CUDA Event timers.
Profiles under AllToAll (allgather_reducescatter) backend.

Usage:
  # Timing-only (fast, no nsys):
  CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter \
    torchrun --nproc_per_node=4 codex_coding/src/nsys_c11_m5k4_profiling.py

  # With nsys capture (rank 0 only):
  CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter \
    nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx,nccl \
    -o codex_coding/results/c11_m5k4_agr_b512 \
    torchrun --nproc_per_node=4 codex_coding/src/nsys_c11_m5k4_profiling.py
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


# ================================================================
# CUDA Event Timer
# ================================================================
class ComponentTimer:
    def __init__(self):
        self._stack = []
        self.data = defaultdict(list)

    def start(self, tag):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._stack.append((tag, ev))

    def end(self):
        end_ev = torch.cuda.Event(enable_timing=True)
        end_ev.record()
        tag, start_ev = self._stack.pop()
        self.data[tag].append((start_ev, end_ev))

    def summarize(self):
        torch.cuda.synchronize()
        stats = {}
        for tag, pairs in self.data.items():
            times = [s.elapsed_time(e) for s, e in pairs]
            stats[tag] = {
                'count': len(times),
                'total_ms': sum(times),
                'avg_ms': sum(times) / len(times) if times else 0,
            }
        return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gen-length", type=int, default=64)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    from dinfer.model.modeling_llada2_moe import _moe_forward_with_context
    from transformers import AutoTokenizer, AutoConfig
    from test_fused_eb_triton import fused_routing
    from test_m_skip_sweep import MSkipEBController
    from baseline_optimizations import apply_all_optimizations

    if args.batch_size <= 128:
        from test_heteval128 import PROMPTS
    else:
        from test_heteval512 import PROMPTS

    # --- Distributed init ---
    vllm_dist.init_distributed_environment(world_size, rank, "env://", local_rank, "nccl")
    vllm_dist.initialize_model_parallel(world_size, backend="nccl")

    alltoall_backend = os.environ.get("VLLM_ALL2ALL_BACKEND", "")
    if rank == 0:
        print("=" * 80)
        print(f"nsys Profiling — C11-M5-K4 AGR, {world_size}-GPU TP")
        print(f"  batch={args.batch_size}, gen={args.gen_length}")
        print(f"  AllToAll backend: {alltoall_backend or 'AllReduce (no backend)'}")
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
        apply_all_optimizations(model)

        # Initialize AllToAll buffers if backend is set
        if alltoall_backend:
            from vllm.distributed import prepare_communication_buffer_for_model
            prepare_communication_buffer_for_model(model)
            if rank == 0:
                print(f"  AllToAll buffers initialized")

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
        input_ids = torch.stack(padded, dim=0).to(device)

        if rank == 0:
            print(f"  Input shape: {input_ids.shape}")
            mem_gb = torch.cuda.memory_allocated(device) / 1e9
            print(f"  GPU memory: {mem_gb:.1f} GB")

        # --- EB controller ---
        ctrl = MSkipEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)

        # --- Install NVTX + CUDA Event instrumentation (rank 0 only) ---
        timer = ComponentTimer() if rank == 0 else None
        hooks = []

        def hook_mod(mod, tag):
            if timer is None:
                return
            def pre(m, inp):
                torch.cuda.nvtx.range_push(tag)
                timer.start(tag)
            def post(m, inp, out):
                timer.end()
                torch.cuda.nvtx.range_pop()
            hooks.append(mod.register_forward_pre_hook(pre))
            hooks.append(mod.register_forward_hook(post))

        # Hook non-MoE components (reference)
        emb = getattr(model.model, 'word_embeddings',
                      getattr(model.model, 'embed_tokens', None))
        if emb:
            hook_mod(emb, "Embedding")

        for li, layer in enumerate(model.model.layers):
            attn = layer.attention if hasattr(layer, 'attention') else layer.self_attn
            hook_mod(attn, f"Attention_L{li}")
            hook_mod(layer.input_layernorm, f"RMSNorm_pre_L{li}")
            hook_mod(layer.post_attention_layernorm, f"RMSNorm_post_L{li}")

            if not hasattr(layer.mlp, 'gate'):
                hook_mod(layer.mlp, f"DenseMLP_L{li}")
                continue

            # --- MoE: patch forward with component timing ---
            moe = layer.mlp
            gate = moe.gate
            g_bias = gate.expert_bias
            g_rsf = gate.routed_scaling_factor
            g_tk, g_ng, g_tkg = gate.top_k, gate.n_group, gate.topk_group

            # Routing patch: EB + K=4 + timing
            def make_routing_fn(bias, rsf, tk, ng, tkg, layer_i, cc, tmr):
                def fn(hidden_states, gating_output, topk, renormalize):
                    if cc is not None and tmr is not None:
                        tmr.start(f"eb_total_L{layer_i}")
                        torch.cuda.nvtx.range_push(f"EB_L{layer_i}")
                    s_mask = cc.get_s_mask(layer_i, gating_output, bias) if cc is not None else None
                    if cc is not None and tmr is not None:
                        torch.cuda.nvtx.range_pop()
                        tmr.end()

                    if tmr is not None:
                        tmr.start(f"routing_L{layer_i}")
                        torch.cuda.nvtx.range_push(f"routing_L{layer_i}")
                    w, idx = fused_routing(gating_output, bias, rsf,
                                           s_mask=s_mask, K=4, ng=ng, tkg=tkg)
                    if tmr is not None:
                        torch.cuda.nvtx.range_pop()
                        tmr.end()
                    return w.to(gating_output.dtype), idx
                return fn

            gate.routing = make_routing_fn(
                g_bias, g_rsf, g_tk, g_ng, g_tkg, li, ctrl, timer)

            # MoE block forward patch with timing
            def make_moe_fwd(blk, layer_i, tmr):
                def fwd(hidden_states):
                    bsz, seq_len, h = hidden_states.shape
                    hs_flat = hidden_states.view(-1, h)

                    # shared_experts
                    if blk.config.num_shared_experts is not None:
                        if tmr is not None:
                            tmr.start(f"shared_L{layer_i}")
                            torch.cuda.nvtx.range_push(f"shared_L{layer_i}")
                        res = blk.shared_experts(hs_flat)
                        if tmr is not None:
                            torch.cuda.nvtx.range_pop()
                            tmr.end()
                    else:
                        res = None

                    # gate.get_logits
                    if tmr is not None:
                        tmr.start(f"getlogits_L{layer_i}")
                        torch.cuda.nvtx.range_push(f"getlogits_L{layer_i}")
                    logits = blk.gate.get_logits(hs_flat)
                    if tmr is not None:
                        torch.cuda.nvtx.range_pop()
                        tmr.end()

                    # fwd_impl (routing + MK dispatch + fused_experts + MK combine)
                    if tmr is not None:
                        tmr.start(f"fwd_impl_L{layer_i}")
                        torch.cuda.nvtx.range_push(f"fwd_impl_L{layer_i}")
                    y = _moe_forward_with_context(blk.experts, hs_flat, logits)
                    if tmr is not None:
                        torch.cuda.nvtx.range_pop()
                        tmr.end()

                    if res is not None:
                        y = y + res
                    y = y.view(bsz, seq_len, h)
                    return y
                return fwd

            moe.forward = make_moe_fwd(moe, li, timer)

        hook_mod(model.model.norm, "FinalRMSNorm")
        hook_mod(model.lm_head, "LMHead")

        # --- Warmup ---
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

        # Reset EB controller
        def reset():
            ctrl.prev_N.clear(); ctrl.K_init.clear()
            ctrl.cold_count = 0; ctrl.hot_count = 0
            ctrl.eb_calls = 0; ctrl.eb_skips = 0
            ctrl._bufs.clear(); ctrl.k_init_history.clear()
            ctrl.s_mask_cache.clear(); ctrl.pop_cache.clear()
            ctrl._fwd_in_block.clear(); ctrl._block_idx.clear()

        if rank == 0:
            print("\nWarmup...")
        reset()
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=args.gen_length,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd, "
                  f"cold={ctrl.cold_count}, hot={ctrl.hot_count}")
            # Clear warmup data
            timer.data.clear()

        # --- Profiled run ---
        if rank == 0:
            print("\nProfiled run...")
        reset()
        dllm = make_dllm()
        torch.cuda.synchronize()
        dist.barrier()

        # nsys capture range
        torch.cuda.cudart().cudaProfilerStart()
        t0 = time.perf_counter()
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=args.gen_length,
                              block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        torch.cuda.cudart().cudaProfilerStop()
        dist.barrier()

        total_time = t1 - t0
        total_fwd = dllm.diff_iteration.num_forwards

        if rank == 0:
            print(f"  Done: {total_time:.3f}s, {total_fwd} fwd, "
                  f"{total_time*1000/total_fwd:.2f} ms/fwd, "
                  f"cold={ctrl.cold_count}, hot={ctrl.hot_count}")

            # --- Analyze ---
            raw = timer.summarize()
            total_ms = total_time * 1000

            agg = defaultdict(lambda: {'count': 0, 'total_ms': 0.0})
            for tag, s in raw.items():
                if tag.startswith("eb_total_L"):     comp = "EB_total"
                elif tag.startswith("shared_L"):     comp = "shared_experts"
                elif tag.startswith("getlogits_L"):  comp = "gate_getlogits"
                elif tag.startswith("routing_L"):    comp = "routing"
                elif tag.startswith("fwd_impl_L"):   comp = "fwd_impl"
                elif tag.startswith("Attention_L"):  comp = "Attention"
                elif tag.startswith("RMSNorm_pre_L"):  comp = "RMSNorm_pre"
                elif tag.startswith("RMSNorm_post_L"): comp = "RMSNorm_post"
                elif tag.startswith("DenseMLP_L"):   comp = "DenseMLP"
                elif tag == "Embedding":    comp = "Embedding"
                elif tag == "FinalRMSNorm": comp = "FinalRMSNorm"
                elif tag == "LMHead":       comp = "LMHead"
                else: continue
                agg[comp]['count'] += s['count']
                agg[comp]['total_ms'] += s['total_ms']

            # Derived
            eb_ms = agg['EB_total']['total_ms']
            routing_ms = agg['routing']['total_ms']
            fwd_impl_ms = agg['fwd_impl']['total_ms']
            # fwd_impl includes: routing + EB + MK(dispatch+experts+combine)
            mk_internal_ms = max(0, fwd_impl_ms - eb_ms - routing_ms)

            moe_total = (agg['shared_experts']['total_ms'] +
                         agg['gate_getlogits']['total_ms'] + fwd_impl_ms)
            non_moe = (agg['Attention']['total_ms'] + agg['RMSNorm_pre']['total_ms']
                       + agg['RMSNorm_post']['total_ms'] + agg['DenseMLP']['total_ms']
                       + agg['Embedding']['total_ms'] + agg['FinalRMSNorm']['total_ms']
                       + agg['LMHead']['total_ms'])
            instrumented = moe_total + non_moe
            gap = total_ms - instrumented

            # Print
            print(f"\n{'='*80}")
            print(f"COMPONENT BREAKDOWN — C11-M5-K4 AGR (batch={args.batch_size})")
            print(f"  Wall-clock: {total_time:.3f}s | Fwd: {total_fwd} | "
                  f"ms/fwd: {total_ms/total_fwd:.2f}")
            print(f"  EB cold: {ctrl.cold_count} | hot: {ctrl.hot_count}")
            print(f"{'='*80}")

            rows = [
                ("=== MoE (L1-L19) ===", None),
                ("  shared_experts", agg['shared_experts']['total_ms']),
                ("  gate_getlogits", agg['gate_getlogits']['total_ms']),
                ("  EB_total", eb_ms),
                ("  routing (Triton)", routing_ms),
                ("  MK_internal *", mk_internal_ms),
                ("    (= fwd_impl - EB - routing)", None),
                ("    (= AGR dispatch + fused_experts + AGR combine)", None),
                ("  fwd_impl (total)", fwd_impl_ms),
                ("  --- MoE subtotal ---", moe_total),
                ("", None),
                ("=== Non-MoE (reference) ===", None),
                ("  Attention (incl TP AR)", agg['Attention']['total_ms']),
                ("  RMSNorm", agg['RMSNorm_pre']['total_ms'] + agg['RMSNorm_post']['total_ms']),
                ("  DenseMLP (L0)", agg['DenseMLP']['total_ms']),
                ("  Embedding", agg['Embedding']['total_ms']),
                ("  LMHead", agg['LMHead']['total_ms']),
                ("  FinalRMSNorm", agg['FinalRMSNorm']['total_ms']),
                ("  --- Non-MoE subtotal ---", non_moe),
                ("", None),
                ("Instrumented total", instrumented),
                ("Gap (Python/decoder)", gap),
            ]

            print(f"\n  {'Component':<40s} {'Total(ms)':>10s} {'ms/fwd':>8s} "
                  f"{'%wall':>7s}")
            print(f"  {'-'*68}")
            for label, ms in rows:
                if ms is None:
                    print(f"  {label}")
                else:
                    pct = ms / total_ms * 100
                    per_fwd = ms / total_fwd
                    print(f"  {label:<40s} {ms:>10.1f} {per_fwd:>8.2f} "
                          f"{pct:>6.1f}%")

            # Per-layer fwd_impl breakdown
            print(f"\n  Per-layer fwd_impl (ms, top 5 heaviest):")
            layer_fwd = {}
            for tag, s in raw.items():
                if tag.startswith("fwd_impl_L"):
                    li = int(tag.split("_L")[1])
                    layer_fwd[li] = s['total_ms']
            for li, ms in sorted(layer_fwd.items(), key=lambda x: -x[1])[:5]:
                print(f"    L{li}: {ms:.1f}ms ({ms/total_fwd:.2f} ms/fwd)")

            # Save
            results = {
                'config': 'C11_M5_K4_AGR',
                'batch_size': args.batch_size,
                'gen_length': args.gen_length,
                'world_size': world_size,
                'total_time_s': total_time,
                'total_fwd': total_fwd,
                'ms_per_fwd': total_ms / total_fwd,
                'components': {k: v for k, v in agg.items()},
                'mk_internal_ms': mk_internal_ms,
                'moe_total_ms': moe_total,
                'non_moe_ms': non_moe,
                'gap_ms': gap,
            }
            out_path = REPO_ROOT / "codex_coding" / "results" / "c11_m5k4_agr_profiling.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n  Saved to {out_path}")

        # Cleanup
        for h in hooks:
            h.remove()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
