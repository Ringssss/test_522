"""
High-performance batched diffusion generation with KV cache.
v2: CUDA Graph + fully vectorized commit (no Python per-sample loop).
"""
import torch
import torch.nn.functional as F
from transformers import DynamicCache


@torch.no_grad()
def fast_generate_with_kvcache(model, input_ids, gen_length=128, block_length=32,
                                mask_id=156895, max_iters_per_block=15):
    """Batched diffusion generation with KV cache + vectorized decode."""
    device = input_ids.device
    bsz = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    total_len = prompt_len + gen_length
    n_blocks = gen_length // block_length

    x = torch.full((bsz, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    total_forwards = 0

    # Prefill
    prompt_pos = torch.arange(prompt_len, device=device).unsqueeze(0).expand(bsz, -1)
    kv_cache = DynamicCache()
    prompt_out = model.model(input_ids=x[:, :prompt_len], position_ids=prompt_pos,
                             past_key_values=kv_cache, use_cache=True)
    total_forwards += 1

    # Pre-compute block position ids (constant)
    block_pos_cache = {}
    for bi in range(n_blocks):
        bs = prompt_len + bi * block_length
        block_pos_cache[bi] = torch.arange(bs, bs + block_length, device=device).unsqueeze(0).expand(bsz, -1)

    # Pre-allocate decode buffers
    arange_bl = torch.arange(block_length, device=device)

    for block_idx in range(n_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = block_start + block_length
        block_pos = block_pos_cache[block_idx]

        remaining = (x[:, block_start:block_end] == mask_id).sum(dim=1)
        if remaining.max().item() == 0:
            continue
        steps = remaining.max().item()

        prefix_len = kv_cache.get_seq_length()

        for iter_idx in range(min(steps + 2, max_iters_per_block)):
            block_x = x[:, block_start:block_end]
            if (block_x == mask_id).sum() == 0:
                break

            # Trim KV to prefix
            for layer in kv_cache.layers:
                layer.crop(prefix_len)

            # Forward only block tokens
            block_out = model.model(input_ids=block_x, position_ids=block_pos,
                                    past_key_values=kv_cache, use_cache=True)
            logits = model.lm_head(block_out.last_hidden_state)
            total_forwards += 1

            # ═══ Fully vectorized threshold decode (no Python loop) ═══
            live = (block_x == mask_id)  # [bsz, bl]
            if not live.any():
                break

            pred = logits.argmax(dim=-1)  # [bsz, bl]
            conf = F.softmax(logits.float(), dim=-1).max(dim=-1).values  # [bsz, bl]
            conf.masked_fill_(~live, -1.0)

            n_transfer = torch.clamp(remaining // max(1, steps - iter_idx), min=1)
            n_transfer = torch.min(n_transfer, live.sum(dim=1))  # [bsz]

            # Sort by confidence and build commit mask — fully vectorized
            sorted_conf, sorted_idx = conf.sort(dim=1, descending=True)
            # commit_mask[b, j] = True if j < n_transfer[b]
            commit_mask = arange_bl.unsqueeze(0) < n_transfer.unsqueeze(1)  # [bsz, bl]

            # Gather the positions to commit and the predicted tokens
            commit_positions = sorted_idx  # [bsz, bl] — positions sorted by confidence
            commit_preds = pred.gather(1, commit_positions)  # tokens at those positions

            # Apply commits via scatter — no Python loop
            # For each (b, j) where commit_mask is True:
            #   x[b, block_start + commit_positions[b, j]] = commit_preds[b, j]
            global_positions = commit_positions + block_start  # [bsz, bl]
            # Only scatter where mask is true
            commit_tokens = torch.where(commit_mask, commit_preds, x.gather(1, global_positions))
            x.scatter_(1, global_positions, commit_tokens)

            remaining = (x[:, block_start:block_end] == mask_id).sum(dim=1)

        # Update cache with final block
        for layer in kv_cache.layers:
            layer.crop(prefix_len)
        final_block = x[:, block_start:block_end]
        _ = model.model(input_ids=final_block, position_ids=block_pos,
                        past_key_values=kv_cache, use_cache=True)
        total_forwards += 1

    return x, total_forwards


@torch.no_grad()
def fast_generate_with_kvcache_cudagraph(model, input_ids, gen_length=128, block_length=32,
                                          mask_id=156895, max_iters_per_block=15):
    """KV cache + CUDA Graph captured forward."""
    device = input_ids.device
    bsz = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    total_len = prompt_len + gen_length
    n_blocks = gen_length // block_length

    x = torch.full((bsz, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    total_forwards = 0

    # Prefill
    prompt_pos = torch.arange(prompt_len, device=device).unsqueeze(0).expand(bsz, -1)
    kv_cache = DynamicCache()
    model.model(input_ids=x[:, :prompt_len], position_ids=prompt_pos,
                past_key_values=kv_cache, use_cache=True)
    total_forwards += 1

    arange_bl = torch.arange(block_length, device=device)

    # Static buffers for CUDA graph
    static_block_x = torch.zeros(bsz, block_length, dtype=torch.long, device=device)
    static_block_pos = torch.zeros(bsz, block_length, dtype=torch.long, device=device)
    graph = None
    static_hidden = None
    static_logits = None

    for block_idx in range(n_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = block_start + block_length
        block_pos = torch.arange(block_start, block_end, device=device).unsqueeze(0).expand(bsz, -1)

        remaining = (x[:, block_start:block_end] == mask_id).sum(dim=1)
        if remaining.max().item() == 0:
            continue
        steps = remaining.max().item()
        prefix_len = kv_cache.get_seq_length()

        # Try to capture CUDA graph for this block shape
        if graph is None:
            static_block_pos.copy_(block_pos)
            static_block_x.copy_(x[:, block_start:block_end])
            for layer in kv_cache.layers:
                layer.crop(prefix_len)

            # Warmup
            model.model(input_ids=static_block_x, position_ids=static_block_pos,
                        past_key_values=kv_cache, use_cache=True)
            for layer in kv_cache.layers:
                layer.crop(prefix_len)

            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = model.model(input_ids=static_block_x, position_ids=static_block_pos,
                                      past_key_values=kv_cache, use_cache=True)
                    static_hidden = out.last_hidden_state
                    static_logits = model.lm_head(static_hidden)
                torch.cuda.synchronize()
            except Exception as e:
                graph = None

        for iter_idx in range(min(steps + 2, max_iters_per_block)):
            block_x = x[:, block_start:block_end]
            if (block_x == mask_id).sum() == 0:
                break

            for layer in kv_cache.layers:
                layer.crop(prefix_len)

            if graph is not None:
                static_block_x.copy_(block_x)
                static_block_pos.copy_(block_pos)
                graph.replay()
                logits = static_logits
            else:
                out = model.model(input_ids=block_x, position_ids=block_pos,
                                  past_key_values=kv_cache, use_cache=True)
                logits = model.lm_head(out.last_hidden_state)
            total_forwards += 1

            live = (block_x == mask_id)
            if not live.any():
                break

            pred = logits.argmax(dim=-1)
            conf = F.softmax(logits.float(), dim=-1).max(dim=-1).values
            conf.masked_fill_(~live, -1.0)

            n_transfer = torch.clamp(remaining // max(1, steps - iter_idx), min=1)
            n_transfer = torch.min(n_transfer, live.sum(dim=1))

            sorted_conf, sorted_idx = conf.sort(dim=1, descending=True)
            commit_mask = arange_bl.unsqueeze(0) < n_transfer.unsqueeze(1)
            commit_positions = sorted_idx
            commit_preds = pred.gather(1, commit_positions)
            global_positions = commit_positions + block_start
            commit_tokens = torch.where(commit_mask, commit_preds, x.gather(1, global_positions))
            x.scatter_(1, global_positions, commit_tokens)

            remaining = (x[:, block_start:block_end] == mask_id).sum(dim=1)

        # Update cache
        for layer in kv_cache.layers:
            layer.crop(prefix_len)
        _ = model.model(input_ids=x[:, block_start:block_end], position_ids=block_pos,
                        past_key_values=kv_cache, use_cache=True)
        total_forwards += 1

    return x, total_forwards


@torch.no_grad()
def fast_generate_no_kvcache(model, input_ids, gen_length=128, block_length=32,
                              mask_id=156895, max_iters_per_block=15):
    """Baseline: forward full sequence every iteration."""
    device = input_ids.device
    bsz = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    total_len = prompt_len + gen_length
    n_blocks = gen_length // block_length

    x = torch.full((bsz, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    pos_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(bsz, -1)
    arange_bl = torch.arange(block_length, device=device)
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
            commit_mask = arange_bl.unsqueeze(0) < n_transfer.unsqueeze(1)
            commit_positions = sorted_idx
            commit_preds = pred.gather(1, commit_positions)
            global_positions = commit_positions + block_start
            commit_tokens = torch.where(commit_mask, commit_preds, x.gather(1, global_positions))
            x.scatter_(1, global_positions, commit_tokens)

            remaining = (x[:, block_start:block_end] == mask_id).sum(dim=1)

    return x, total_forwards
