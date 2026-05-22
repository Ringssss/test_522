"""
High-performance batched diffusion generation with KV cache.

Uses transformers DynamicCache directly (bypasses dInfer KVCache).
Forward only current block tokens (32 instead of 210).
"""
import torch
import torch.nn.functional as F
from transformers import DynamicCache


@torch.no_grad()
def fast_generate_with_kvcache(model, input_ids, gen_length=128, block_length=32,
                                mask_id=156895, max_iters_per_block=15):
    """Batched diffusion generation with DynamicCache KV cache."""
    device = input_ids.device
    bsz = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    total_len = prompt_len + gen_length
    n_blocks = gen_length // block_length

    x = torch.full((bsz, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    total_forwards = 0

    # Prefill: forward prompt to build KV cache
    prompt_pos = torch.arange(prompt_len, device=device).unsqueeze(0).expand(bsz, -1)
    kv_cache = DynamicCache()
    prompt_out = model.model(
        input_ids=x[:, :prompt_len],
        position_ids=prompt_pos,
        past_key_values=kv_cache,
        use_cache=True,
    )
    total_forwards += 1

    for block_idx in range(n_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = block_start + block_length
        block_pos = torch.arange(block_start, block_end, device=device).unsqueeze(0).expand(bsz, -1)

        remaining = (x[:, block_start:block_end] == mask_id).sum(dim=1)
        if remaining.max().item() == 0:
            continue
        steps = remaining.max().item()

        # Save KV cache length at block start
        prefix_len = kv_cache.get_seq_length()

        for iter_idx in range(min(steps + 2, max_iters_per_block)):
            block_x = x[:, block_start:block_end]
            if (block_x == mask_id).sum() == 0:
                break

            # Trim KV cache back to prefix (remove previous block iteration's KV)
            for layer in kv_cache.layers:
                layer.crop(prefix_len)

            # Forward only block tokens with KV cache prefix
            block_out = model.model(
                input_ids=block_x,
                position_ids=block_pos,
                past_key_values=kv_cache,
                use_cache=True,
            )
            hidden = block_out.last_hidden_state
            logits = model.lm_head(hidden)  # [bsz, block_length, V]
            total_forwards += 1

            # Vectorized threshold decode
            live = (block_x == mask_id)
            if not live.any():
                break

            pred = logits.argmax(dim=-1)
            conf = F.softmax(logits.float(), dim=-1).max(dim=-1).values
            conf.masked_fill_(~live, -1.0)

            n_transfer = torch.clamp(remaining // max(1, steps - iter_idx), min=1)
            n_transfer = torch.min(n_transfer, live.sum(dim=1))

            sorted_conf, sorted_idx = conf.sort(dim=1, descending=True)
            for b in range(bsz):
                n = n_transfer[b].item()
                if n > 0:
                    x[b, block_start + sorted_idx[b, :n]] = pred[b, sorted_idx[b, :n]]

            remaining = (x[:, block_start:block_end] == mask_id).sum(dim=1)

        # After block complete: trim KV to prefix, then forward final block to update cache
        for layer in kv_cache.layers:
            layer.crop(prefix_len)

        final_block = x[:, block_start:block_end]
        _ = model.model(
            input_ids=final_block,
            position_ids=block_pos,
            past_key_values=kv_cache,
            use_cache=True,
        )
        total_forwards += 1

    return x, total_forwards


@torch.no_grad()
def fast_generate_no_kvcache(model, input_ids, gen_length=128, block_length=32,
                              mask_id=156895, max_iters_per_block=15):
    """Baseline: forward full sequence every iteration (for comparison)."""
    device = input_ids.device
    bsz = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    total_len = prompt_len + gen_length
    n_blocks = gen_length // block_length

    x = torch.full((bsz, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    pos_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(bsz, -1)

    total_forwards = 0

    for block_idx in range(n_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = block_start + block_length
        remaining = (x[:, block_start:block_end] == mask_id).sum(dim=1)
        steps = remaining.max().item()
        if steps == 0:
            continue

        for iter_idx in range(min(steps + 2, max_iters_per_block)):
            block_x = x[:, block_start:block_end]
            if (block_x == mask_id).sum() == 0:
                break

            logits = model(x[:, :block_end], position_ids=pos_ids[:, :block_end],
                           use_cache=False).logits[:, block_start:block_end]
            total_forwards += 1

            live = (block_x == mask_id)
            pred = logits.argmax(dim=-1)
            conf = F.softmax(logits.float(), dim=-1).max(dim=-1).values
            conf.masked_fill_(~live, -1.0)

            n_transfer = torch.clamp(remaining // max(1, steps - iter_idx), min=1)
            n_transfer = torch.min(n_transfer, live.sum(dim=1))

            sorted_conf, sorted_idx = conf.sort(dim=1, descending=True)
            for b in range(bsz):
                n = n_transfer[b].item()
                if n > 0:
                    positions = sorted_idx[b, :n]
                    x[b, block_start + positions] = pred[b, positions]

            remaining = (x[:, block_start:block_end] == mask_id).sum(dim=1)

    return x, total_forwards
