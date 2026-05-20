"""
Shared baseline optimizations for LLaDA2.0-mini.

Usage:
    from baseline_optimizations import apply_all_optimizations
    n_rms, n_fa = apply_all_optimizations(model)
"""

import torch
import torch.nn.functional as F


def apply_fused_rmsnorm(model):
    """Fuse layer-level RMSNorm (skip attention-internal QK norms)."""
    from vllm.model_executor.layers.layernorm import rms_norm as vllm_rms_norm
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeRMSNorm
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, LLaDA2MoeRMSNorm):
            if "query_layernorm" in name or "key_layernorm" in name:
                continue
            w, eps = module.weight, module.variance_epsilon
            def _mk(ww, ee):
                def _f(hs): return vllm_rms_norm(hs, ww, ee)
                return _f
            module.forward = _mk(w, eps)
            count += 1
    return count


def apply_flash_attn_classic(model):
    """Replace SDPA with classic flash-attn 2.8 + QK norm before transpose."""
    from flash_attn import flash_attn_func
    from vllm.model_executor.layers.layernorm import rms_norm as vllm_rms_norm
    from dinfer.model.modeling_llada2_moe import (
        LLaDA2MoeSdpaAttention, apply_rotary_pos_emb,
    )
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, LLaDA2MoeSdpaAttention):
            continue

        def make_fa_forward(attn_mod):
            q_norm_w = attn_mod.query_layernorm.weight
            q_norm_eps = attn_mod.query_layernorm.variance_epsilon
            k_norm_w = attn_mod.key_layernorm.weight
            k_norm_eps = attn_mod.key_layernorm.variance_epsilon

            def fa_forward(
                hidden_states: torch.Tensor, attention_mask=None,
                position_ids=None, past_key_value=None,
                output_attentions: bool = False, use_cache: bool = False,
                position_embeddings=None, cache_position=None,
                replace_position=None, **kwargs,
            ):
                bsz, q_len, _ = hidden_states.size()
                # C11: num_heads is already local (TP-partitioned)
                num_heads = attn_mod.num_heads
                num_kv_heads = attn_mod.num_key_value_heads
                head_dim = attn_mod.head_dim

                qkv, _ = attn_mod.query_key_value(hidden_states)
                qkv = qkv.view(bsz, q_len, num_heads + 2 * num_kv_heads, head_dim)
                query_states, key_states, value_states = qkv.split(
                    [num_heads, num_kv_heads, num_kv_heads], dim=-2)

                query_states = vllm_rms_norm(query_states, q_norm_w, q_norm_eps)
                key_states = vllm_rms_norm(key_states, k_norm_w, k_norm_eps)

                query_states = query_states.transpose(1, 2)
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin)

                if past_key_value is not None:
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, attn_mod.layer_idx,
                        replace_position)
                if use_cache:
                    past_key_value = (key_states, value_states)

                q_fa = query_states.transpose(1, 2)
                k_fa = key_states.transpose(1, 2)
                v_fa = value_states.transpose(1, 2)

                if attention_mask is not None:
                    from dinfer.model.modeling_llada2_moe import repeat_kv
                    num_kv_groups = num_heads // num_kv_heads
                    k_exp = repeat_kv(key_states, num_kv_groups).contiguous()
                    v_exp = repeat_kv(value_states, num_kv_groups).contiguous()
                    q_c = query_states.contiguous()
                    am = attention_mask.bool()
                    if am.dim() == 3:
                        am = am.unsqueeze(1)
                    attn_output = F.scaled_dot_product_attention(
                        q_c, k_exp, v_exp, attn_mask=am, dropout_p=0.0,
                        is_causal=False)
                    attn_output = attn_output.transpose(1, 2).contiguous()
                else:
                    attn_output = flash_attn_func(
                        q_fa.contiguous(), k_fa.contiguous(),
                        v_fa.contiguous(), causal=False)

                attn_output = attn_output.reshape(bsz, q_len, -1)
                attn_output, _ = attn_mod.dense(attn_output)
                return attn_output, None, past_key_value
            return fa_forward

        module.forward = make_fa_forward(module)
        count += 1
    return count


def apply_all_optimizations(model):
    """Apply all baseline optimizations: fused RMSNorm + classic flash-attn."""
    n_rms = apply_fused_rmsnorm(model)
    n_fa = apply_flash_attn_classic(model)
    return n_rms, n_fa
