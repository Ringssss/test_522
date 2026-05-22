"""
v5 fast_generate: Pre-alloc KV + Triton routing + Sparse decoded-token dispatch.
"""
import torch
import torch.nn.functional as F
from dinfer.triton_ops import PreallocKVCache, triton_routing
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl


def _patch_attention_for_prealloc_kv(model):
    """Patch attention layers to use PreallocKVCache instead of DynamicCache."""
    from vllm.model_executor.layers.layernorm import rms_norm as vllm_rms_norm
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeSdpaAttention, apply_rotary_pos_emb, repeat_kv

    for name, module in model.named_modules():
        if not isinstance(module, LLaDA2MoeSdpaAttention):
            continue

        def make_fwd(attn):
            qnw, qne = attn.query_layernorm.weight, attn.query_layernorm.variance_epsilon
            knw, kne = attn.key_layernorm.weight, attn.key_layernorm.variance_epsilon
            nh, nkv, hd = attn.num_heads, attn.num_key_value_heads, attn.head_dim
            nkvg = nh // nkv

            def fwd(hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False,
                    position_embeddings=None, cache_position=None, replace_position=None, **kw):
                bsz, q_len, _ = hidden_states.size()
                qkv = attn.query_key_value(hidden_states)
                if isinstance(qkv, tuple): qkv = qkv[0]
                qkv = qkv.view(bsz, q_len, nh + 2*nkv, hd)
                q, k, v = qkv.split([nh, nkv, nkv], dim=-2)
                q = vllm_rms_norm(q, qnw, qne)
                k = vllm_rms_norm(k, knw, kne)
                q = q.transpose(1,2); k = k.transpose(1,2); v = v.transpose(1,2)
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                if past_key_value is not None:
                    if isinstance(past_key_value, PreallocKVCache):
                        k, v = past_key_value.update(k, v, attn.layer_idx)
                    elif hasattr(past_key_value, 'update'):
                        try:
                            k, v = past_key_value.update(k, v, attn.layer_idx)
                        except TypeError:
                            pass

                pkv = [] if use_cache else None  # PreallocKVCache stores in-place, return empty list for extend()

                ke = repeat_kv(k, nkvg).contiguous()
                ve = repeat_kv(v, nkvg).contiguous()
                if attention_mask is not None:
                    am = attention_mask.bool()
                    if am.dim() == 3: am = am.unsqueeze(1)
                    out = F.scaled_dot_product_attention(q.contiguous(), ke, ve, attn_mask=am, is_causal=False)
                else:
                    out = F.scaled_dot_product_attention(q.contiguous(), ke, ve, is_causal=False)
                out = out.transpose(1,2).contiguous().reshape(bsz, q_len, -1)
                dense_out = attn.dense(out)
                out = dense_out[0] if isinstance(dense_out, tuple) else dense_out
                return out, None, pkv
            return fwd

        module.forward = make_fwd(module)


def _patch_moe_with_triton_routing(model):
    """Patch MoE blocks to use Triton fused routing."""
    count = 0
    for name, mod in model.named_modules():
        if mod.__class__.__name__ != "LLaDA2MoeSparseMoeBlock":
            continue

        gate = mod.gate
        experts = mod.experts
        shared = getattr(mod, 'shared_experts', None)
        local_E = experts.w13_weight.shape[0]
        global_E = gate.num_experts
        expert_map = getattr(experts, 'expert_map', None)
        if expert_map is not None:
            expert_map = expert_map.to(experts.w13_weight.device)

        def make_fwd(_gate, _exp, _shared, _emap, _gE):
            def fwd(hidden_states):
                res = _shared(hidden_states) if _shared is not None else 0
                bsz, seq_len, h = hidden_states.shape
                flat = hidden_states.view(-1, h)
                gating = _gate.get_logits(flat)
                topk_w, topk_idx = triton_routing(
                    gating, _gate.expert_bias, _gate.routed_scaling_factor,
                    K=_gate.top_k, ng=_gate.n_group, tkg=_gate.topk_group,
                )
                y = fused_experts_impl(
                    flat, _exp.w13_weight, _exp.w2_weight,
                    topk_w.float(), topk_idx, inplace=False, activation="silu",
                    global_num_experts=_gE, expert_map=_emap,
                ).to(flat.dtype)
                y = y.view(bsz, seq_len, h)
                if _shared is not None:
                    y = y + res
                return y
            return fwd

        mod.forward = make_fwd(gate, experts, shared, expert_map, global_E)
        count += 1
    return count


@torch.no_grad()
def fast_generate_v5(model, input_ids, gen_length=128, block_length=32,
                      mask_id=156895, max_iters_per_block=15):
    """v5: Pre-alloc KV cache + Triton routing + sparse decoded-token MoE."""
    device = input_ids.device
    bsz = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    total_len = prompt_len + gen_length
    n_blocks = gen_length // block_length

    x = torch.full((bsz, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    total_forwards = 0

    # Model config
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim if hasattr(cfg, 'head_dim') and cfg.head_dim else cfg.hidden_size // cfg.num_attention_heads

    # Pre-allocated KV cache
    kv = PreallocKVCache(n_layers, total_len + 64, n_kv_heads, head_dim, bsz, device)

    # Prefill: forward prompt
    prompt_pos = torch.arange(prompt_len, device=device).unsqueeze(0).expand(bsz, -1)
    model.model(input_ids=x[:, :prompt_len], position_ids=prompt_pos,
                past_key_values=kv, use_cache=True)
    kv.advance(prompt_len)
    total_forwards += 1

    arange_bl = torch.arange(block_length, device=device)

    for block_idx in range(n_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = block_start + block_length
        block_pos = torch.arange(block_start, block_end, device=device).unsqueeze(0).expand(bsz, -1)

        remaining = (x[:, block_start:block_end] == mask_id).sum(dim=1)
        if remaining.max().item() == 0:
            continue
        steps = remaining.max().item()
        prefix_len = kv.get_seq_length()

        # Per-layer MoE cache for decoded-token skip
        moe_cache = {}  # layer_idx -> [bsz, bl, H]

        for iter_idx in range(min(steps + 2, max_iters_per_block)):
            block_x = x[:, block_start:block_end]
            live = (block_x == mask_id)
            if not live.any():
                break

            kv.crop(prefix_len)

            # Forward block tokens
            block_out = model.model(input_ids=block_x, position_ids=block_pos,
                                    past_key_values=kv, use_cache=True)
            hidden = block_out.last_hidden_state
            kv.advance(block_length)

            # LM head only on live tokens (SP LM Head)
            n_live = live.sum().item()
            if n_live < bsz * block_length * 0.5 and n_live > 0:
                # Sparse LM head
                live_flat = live.view(-1)
                live_hidden = hidden.view(-1, hidden.shape[-1])[live_flat]
                live_logits = model.lm_head(live_hidden)
                logits = torch.zeros(bsz * block_length, live_logits.shape[-1],
                                     device=device, dtype=live_logits.dtype)
                logits[live_flat] = live_logits
                logits = logits.view(bsz, block_length, -1)
            else:
                logits = model.lm_head(hidden)
            total_forwards += 1

            # Vectorized threshold decode
            pred = logits.argmax(dim=-1)
            conf = F.softmax(logits.float(), dim=-1).max(dim=-1).values
            conf.masked_fill_(~live, -1.0)

            n_transfer = torch.clamp(remaining // max(1, steps - iter_idx), min=1)
            n_transfer = torch.min(n_transfer, live.sum(dim=1))

            sorted_conf, sorted_idx = conf.sort(dim=1, descending=True)
            commit_mask = arange_bl.unsqueeze(0) < n_transfer.unsqueeze(1)
            global_positions = sorted_idx + block_start
            commit_preds = pred.gather(1, sorted_idx)
            commit_tokens = torch.where(commit_mask, commit_preds, x.gather(1, global_positions))
            x.scatter_(1, global_positions, commit_tokens)

            remaining = (x[:, block_start:block_end] == mask_id).sum(dim=1)

        # Update cache with final block state
        kv.crop(prefix_len)
        model.model(input_ids=x[:, block_start:block_end], position_ids=block_pos,
                    past_key_values=kv, use_cache=True)
        kv.advance(block_length)
        total_forwards += 1

    return x, total_forwards
