#!/usr/bin/env python3
"""
TV6 Patch Module — Reuses bench_bsp_moe_dp2.py's G-path and TV6 functions directly.

Applies BSP-G AttnRS SP + EB fused routing (K=4) + TV6 compact dispatch/combine.

Usage:
    from tv6_patch import apply_tv6
    apply_tv6(model, decoder)

Requires:
    - vllm distributed already initialized (tp, dp, ep groups)
    - model already loaded and on device
    - baseline_optimizations.apply_all_optimizations() already called
    - Environment: DINF_SKIP_LOGITS_FLOAT=1 DINF_SP_LM_HEAD=1
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist
import triton.language as tl

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID = 156895

# --- Import DIRECTLY from bench script (module-level functions) ---
from bench_bsp_moe_dp2 import (
    make_bsp_g_attn_rs_decoder_forward,
    make_attention_reduce_scatter_forward,
    _set_experts_sequence_parallel,
    BSPMSkipController,
    COMPONENT_TIMER,
    SPHiddenState,
    AttnSPResult,
)
from dinfer.model.modeling_llada2_moe import _BSPGSPHiddenState
from test_fused_eb_triton import fused_routing


# =====================================================================
# Main entry point
# =====================================================================

def apply_tv6(model, decoder, vllm_cfg=None):
    """
    Apply TV6 (BSP-G + EB K=4 + compact dispatch/combine) to a loaded model.
    Reuses bench_bsp_moe_dp2.py's original G-path functions for exact parity.
    """
    from vllm.distributed import (
        get_ep_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_kernel, try_get_optimal_moe_config,
    )
    from vllm import _custom_ops as vllm_ops

    if vllm_cfg is None:
        from vllm.config import get_current_vllm_config
        vllm_cfg = get_current_vllm_config()

    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    ep_group = get_ep_group()
    ep_world = ep_group.world_size
    dp_size = vllm_cfg.parallel_config.data_parallel_size
    REFRESH_M = 5

    # DP=1: vllm doesn't init all2all_manager (only inits when dp>1).
    # BSP-G needs dispatch/combine via EP group, so init it manually.
    if dp_size <= 1 and ep_group.device_communicator is not None:
        comm = ep_group.device_communicator
        if comm.all2all_manager is None:
            from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
            comm.all2all_manager = AgRsAll2AllManager(comm.cpu_group)

    # --- Identify model components ---
    moe_blocks = [m for _n, m in model.named_modules()
                  if m.__class__.__name__ == "LLaDA2MoeSparseMoeBlock"]
    sparse_decoder_layers = [m for _n, m in model.named_modules()
                             if m.__class__.__name__ == "LLaDA2MoeDecoderLayer"
                             and hasattr(m, 'mlp')
                             and hasattr(m.mlp, 'experts')]

    # --- EB Controller (reuse bench script's BSPMSkipController) ---
    num_moe_layers = len(moe_blocks)
    eb_ctrl = BSPMSkipController(
        num_layers=num_moe_layers, K_target=40, q_major=1.0)

    # --- Routing (EB + fused_routing K=4) ---
    gi = 0
    for _n, m in model.named_modules():
        if m.__class__.__name__ != "LLaDA2MoeGate":
            continue
        b, r, ng, tkg, li = m.expert_bias, m.routed_scaling_factor, m.n_group, m.topk_group, gi

        def mk(bb, rr, nn, gg, layer_i, cc):
            def fn(hs, go, topk, renorm):
                sm = cc.get_s_mask(layer_i, go, bb)
                w, idx = fused_routing(go, bb, rr, s_mask=sm, K=4, ng=nn, tkg=gg)
                return w.to(go.dtype), idx
            return fn

        m.routing = mk(b, r, ng, tkg, li, eb_ctrl)
        gi += 1

    # --- BSP-G: set experts to SP mode ---
    for blk in moe_blocks:
        _set_experts_sequence_parallel(blk.experts, True, tp_size)

    # --- BSP-G: patch attention RS + decoder forward (REUSE bench script originals) ---
    last_sparse_idx = len(sparse_decoder_layers) - 1
    for i, dl in enumerate(sparse_decoder_layers):
        dl.attention.forward = make_attention_reduce_scatter_forward(
            dl.attention, "TV6", i)
        dl.forward = make_bsp_g_attn_rs_decoder_forward(
            dl, i, eb_ctrl, vllm_cfg, "TV6",
            is_last_sparse=(i == last_sparse_idx))

    # --- TV6 state ---
    _tv6 = {
        'moe_cache': {},
        'compute_mask_sp': None,
        'null_mask_sp': None,
        'compute_indices_sp': None,
        'null_indices_sp': None,
        'compact_sizes': None,
        'prev_decoded_sp': None,
        'step': 0,
        'bufs': {},
    }

    # --- TV6: patch experts.forward_impl per MoE block ---
    for i, blk in enumerate(moe_blocks):
        lid = i
        exp_obj = blk.experts
        orig_impl = blk.experts.forward_impl

        def _make_tv6_impl(orig_fn, lid, exp_obj):
            def _run_full_moe(self, hidden_states, router_logits):
                """Full MoE forward with dispatch/combine, used for fallback in DP=1."""
                ctx = get_forward_context()
                N_sp = hidden_states.shape[0]
                _created_meta = False
                if ctx.dp_metadata is None:
                    from vllm.forward_context import DPMetadata
                    ctx.dp_metadata = DPMetadata(
                        max_tokens_across_dp_cpu=torch.tensor([N_sp]),
                        num_tokens_across_dp_cpu=torch.tensor([N_sp]),
                    )
                    _created_meta = True
                full_sizes = [N_sp] * ep_world
                ctx.dp_metadata.local_sizes = full_sizes
                hs_g, rl_g = ep_group.dispatch(
                    hidden_states, router_logits, self.is_sequence_parallel)

                N_g = hs_g.shape[0]
                hidden_dim = hs_g.shape[1]
                topk_weights, topk_ids, _ = FusedMoE.select_experts(
                    hidden_states=hs_g, router_logits=rl_g,
                    use_grouped_topk=self.use_grouped_topk, top_k=self.top_k,
                    renormalize=self.renormalize, topk_group=self.topk_group,
                    num_expert_group=self.num_expert_group,
                    custom_routing_function=self.custom_routing_function,
                    scoring_func=self.scoring_func,
                    routed_scaling_factor=self.routed_scaling_factor,
                    e_score_correction_bias=self.e_score_correction_bias,
                    indices_type=getattr(self, 'topk_indices_dtype', None),
                )
                top_k_num = topk_ids.shape[1]
                N = exp_obj.w13_weight.shape[1]
                K = hidden_dim
                n_pairs = N_g * top_k_num
                bufs = _tv6['bufs']
                if bufs.get('config') is None:
                    bufs['config'] = try_get_optimal_moe_config(
                        exp_obj.w13_weight.size(), exp_obj.w2_weight.size(),
                        top_k_num, hs_g.dtype, N_g, block_shape=None)
                config = bufs['config']
                cache13 = torch.empty(n_pairs * max(N, K), device=hs_g.device, dtype=hs_g.dtype)
                cache1 = cache13[:n_pairs * N].view(N_g, top_k_num, N)
                cache3 = cache13[:n_pairs * K].view(N_g, top_k_num, K)
                cache2 = torch.empty(n_pairs, N // 2, device=hs_g.device, dtype=hs_g.dtype)
                sorted_token_ids, expert_ids, ntp = moe_align_block_size(
                    topk_ids, config['BLOCK_SIZE_M'],
                    self.global_num_experts, self.expert_map)
                ct = tl.bfloat16
                invoke_fused_moe_kernel(
                    hs_g, exp_obj.w13_weight, cache1, None, None, None,
                    topk_weights, sorted_token_ids, expert_ids, ntp,
                    self.apply_router_weight_on_input, top_k_num,
                    config, compute_type=ct,
                    use_fp8_w8a8=False, use_int8_w8a8=False,
                    use_int8_w8a16=False, use_int4_w4a16=False,
                    per_channel_quant=False)
                torch.ops._C.silu_and_mul(cache2, cache1.view(-1, N))
                invoke_fused_moe_kernel(
                    cache2, exp_obj.w2_weight, cache3, None, None, None,
                    topk_weights, sorted_token_ids, expert_ids, ntp,
                    not self.apply_router_weight_on_input, 1,
                    config, compute_type=ct,
                    use_fp8_w8a8=False, use_int8_w8a8=False,
                    use_int8_w8a16=False, use_int4_w4a16=False,
                    per_channel_quant=False)
                out_g = torch.empty(N_g, K, device=hs_g.device, dtype=hs_g.dtype)
                vllm_ops.moe_sum(cache3.view(N_g, top_k_num, K), out_g)
                y_sp = ep_group.combine(out_g, self.is_sequence_parallel)
                ctx.dp_metadata.local_sizes = None
                if _created_meta:
                    ctx.dp_metadata = None
                return y_sp

            def tv6_forward_impl(self, hidden_states, router_logits):
                compute_mask_sp = _tv6['compute_mask_sp']

                if compute_mask_sp is None:
                    if dp_size > 1:
                        y = orig_fn(hidden_states, router_logits)
                    else:
                        y = _run_full_moe(self, hidden_states, router_logits)
                    if isinstance(y, tuple):
                        y = y[1]
                    _tv6['moe_cache'][lid] = y.detach().clone()
                    return y

                ctx = get_forward_context()
                N_sp = hidden_states.shape[0]
                hidden_dim = hidden_states.shape[1]
                compute_idx_sp = _tv6['compute_indices_sp']
                hs_compute_sp = hidden_states[compute_idx_sp]
                rl_compute_sp = router_logits[compute_idx_sp]

                if ctx.dp_metadata is None:
                    from vllm.forward_context import DPMetadata
                    ctx.dp_metadata = DPMetadata(
                        max_tokens_across_dp_cpu=torch.tensor([N_sp]),
                        num_tokens_across_dp_cpu=torch.tensor([N_sp]),
                    )
                ctx.dp_metadata.local_sizes = _tv6['compact_sizes']
                hs_g, rl_g = ep_group.dispatch(
                    hs_compute_sp, rl_compute_sp, self.is_sequence_parallel)

                N_ct = hs_g.shape[0]
                topk_weights, topk_ids, _ = FusedMoE.select_experts(
                    hidden_states=hs_g, router_logits=rl_g,
                    use_grouped_topk=self.use_grouped_topk, top_k=self.top_k,
                    renormalize=self.renormalize, topk_group=self.topk_group,
                    num_expert_group=self.num_expert_group,
                    custom_routing_function=self.custom_routing_function,
                    scoring_func=self.scoring_func,
                    routed_scaling_factor=self.routed_scaling_factor,
                    e_score_correction_bias=self.e_score_correction_bias,
                    indices_type=getattr(self, 'topk_indices_dtype', None),
                )

                top_k_num = topk_ids.shape[1]
                N = exp_obj.w13_weight.shape[1]
                K = hidden_dim
                n_pairs = N_ct * top_k_num

                bufs = _tv6['bufs']
                if bufs.get('config') is None:
                    bufs['config'] = try_get_optimal_moe_config(
                        exp_obj.w13_weight.size(), exp_obj.w2_weight.size(),
                        top_k_num, hs_g.dtype, N_ct, block_shape=None)
                config = bufs['config']

                if bufs.get('_n_pairs') != n_pairs:
                    bufs['_n_pairs'] = n_pairs
                    bufs['cache13'] = torch.empty(
                        n_pairs * max(N, K), device=hs_g.device, dtype=hs_g.dtype)
                    bufs['cache2'] = torch.empty(
                        n_pairs, N // 2, device=hs_g.device, dtype=hs_g.dtype)
                    bufs['compact_output'] = torch.empty(
                        N_ct, K, device=hs_g.device, dtype=hs_g.dtype)

                cache1 = bufs['cache13'][:n_pairs * N].view(N_ct, top_k_num, N)
                cache3 = bufs['cache13'][:n_pairs * K].view(N_ct, top_k_num, K)
                cache2 = bufs['cache2']

                sorted_token_ids, expert_ids, ntp = moe_align_block_size(
                    topk_ids, config['BLOCK_SIZE_M'],
                    self.global_num_experts, self.expert_map)

                ct = tl.bfloat16
                invoke_fused_moe_kernel(
                    hs_g, exp_obj.w13_weight, cache1,
                    None, None, None, topk_weights,
                    sorted_token_ids, expert_ids, ntp,
                    self.apply_router_weight_on_input, top_k_num,
                    config, compute_type=ct,
                    use_fp8_w8a8=False, use_int8_w8a8=False,
                    use_int8_w8a16=False, use_int4_w4a16=False,
                    per_channel_quant=False)

                torch.ops._C.silu_and_mul(cache2, cache1.view(-1, N))

                invoke_fused_moe_kernel(
                    cache2, exp_obj.w2_weight, cache3,
                    None, None, None, topk_weights,
                    sorted_token_ids, expert_ids, ntp,
                    not self.apply_router_weight_on_input, 1,
                    config, compute_type=ct,
                    use_fp8_w8a8=False, use_int8_w8a8=False,
                    use_int8_w8a16=False, use_int4_w4a16=False,
                    per_channel_quant=False)

                compact_output = bufs['compact_output']
                vllm_ops.moe_sum(cache3.view(N_ct, top_k_num, K), compact_output)

                y_compute_sp = ep_group.combine(
                    compact_output, self.is_sequence_parallel)
                ctx.dp_metadata.local_sizes = None
                if dp_size <= 1:
                    ctx.dp_metadata = None

                if lid in _tv6['moe_cache'] and _tv6['moe_cache'][lid].shape[0] == N_sp:
                    y_sp = _tv6['moe_cache'][lid]
                else:
                    y_sp = torch.zeros(N_sp, hidden_dim,
                                       device=hidden_states.device, dtype=hidden_states.dtype)
                    _tv6['moe_cache'][lid] = y_sp

                y_sp.index_copy_(0, compute_idx_sp, y_compute_sp)
                return y_sp

            return types.MethodType(tv6_forward_impl, blk.experts)

        blk.experts.forward_impl = _make_tv6_impl(orig_impl, lid, exp_obj)

    # --- TV6: block_init wrapper ---
    orig_block_init = decoder.block_init

    def block_init_tv6(block_x, block_id):
        _tv6['moe_cache'].clear()
        _tv6['step'] = 0
        _tv6['prev_decoded_sp'] = None
        _tv6['compute_mask_sp'] = None
        _tv6['null_mask_sp'] = None
        _tv6['compute_indices_sp'] = None
        _tv6['null_indices_sp'] = None
        _tv6['compact_sizes'] = None
        eb_ctrl.note_block_start(int(block_id))
        return orig_block_init(block_x, block_id)

    decoder.block_init = block_init_tv6

    # --- TV6: model.forward wrapper ---
    orig_model_forward = model.forward

    def model_forward_tv6(input_ids=None, *args, **kwargs):
        if input_ids is not None:
            step = _tv6['step']
            is_mask = (input_ids == MASK_ID).view(-1)
            chunk = is_mask.shape[0] // tp_size
            sp_start = tp_rank * chunk
            decoded_sp = ~is_mask[sp_start:sp_start + chunk]

            prev_dec = _tv6['prev_decoded_sp']
            if (prev_dec is not None
                    and prev_dec.shape[0] == decoded_sp.shape[0]
                    and step % REFRESH_M != 1):
                null_mask_sp = decoded_sp & prev_dec
            else:
                null_mask_sp = None

            if null_mask_sp is not None and null_mask_sp.any():
                compute_mask_sp = ~null_mask_sp
                _tv6['compute_mask_sp'] = compute_mask_sp
                _tv6['null_mask_sp'] = null_mask_sp
                _tv6['compute_indices_sp'] = compute_mask_sp.nonzero(as_tuple=True)[0]
                _tv6['null_indices_sp'] = null_mask_sp.nonzero(as_tuple=True)[0]

                n_compute_sp = _tv6['compute_indices_sp'].shape[0]
                n_tensor = torch.tensor(
                    [n_compute_sp], device=decoded_sp.device, dtype=torch.int64)
                gathered_counts = [
                    torch.empty(1, device=decoded_sp.device, dtype=torch.int64)
                    for _ in range(ep_world)]
                dist.all_gather(gathered_counts, n_tensor, group=ep_group.device_group)
                _tv6['compact_sizes'] = torch.cat(gathered_counts).tolist()
            else:
                _tv6['compute_mask_sp'] = None
                _tv6['null_mask_sp'] = None
                _tv6['compute_indices_sp'] = None
                _tv6['null_indices_sp'] = None
                _tv6['compact_sizes'] = None
        else:
            _tv6['compute_mask_sp'] = None

        result = orig_model_forward(input_ids, *args, **kwargs)

        if input_ids is not None:
            _tv6['prev_decoded_sp'] = decoded_sp
            _tv6['step'] += 1
        return result

    model.forward = model_forward_tv6

    return {
        'eb_ctrl': eb_ctrl,
        'tv6_state': _tv6,
        'orig_block_init': orig_block_init,
        'orig_model_forward': orig_model_forward,
    }
