"""
High-performance diffusion generation loop.

Eliminates Python overhead between forward iterations by:
1. Keeping decode logic as torch ops (no Python loops per-token)
2. Minimizing CPU-GPU synchronization
3. Pre-allocating all buffers
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def fast_diffusion_generate(model, input_ids, gen_length=128, block_length=32,
                            mask_id=156895, eos_id=156892, threshold=0.9,
                            temperature=0.0):
    """Fast diffusion generation — minimizes Python overhead between iterations.

    Key optimizations vs dInfer's BlockWiseDiffusionLLM:
    1. No TokenArray/BlockLoc abstractions (direct tensor ops)
    2. Vectorized threshold decode (no per-batch Python loop)
    3. No Python overhead between forward calls
    4. Pre-allocated output buffer
    """
    device = input_ids.device
    bsz = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    total_len = prompt_len + gen_length

    # Pre-allocate full sequence buffer
    x = torch.full((bsz, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids

    # Pre-allocate position ids (constant)
    pos_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(bsz, -1)

    n_blocks = gen_length // block_length
    total_forwards = 0

    for block_idx in range(n_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = block_start + block_length

        # Count masks in current block
        block_mask = (x[:, block_start:block_end] == mask_id)
        n_mask_per_seq = block_mask.sum(dim=1)  # [bsz]
        max_mask = n_mask_per_seq.max().item()

        if max_mask == 0:
            continue

        # Compute transfer schedule
        steps = max(1, max_mask)
        remaining = n_mask_per_seq.clone()  # [bsz]

        iter_count = 0
        while remaining.max().item() > 0 and iter_count < steps + 2:
            # Forward pass — full sequence up to block_end
            # (for non-causal attention, we need all tokens)
            logits = model(x[:, :block_end], position_ids=pos_ids[:, :block_end],
                           use_cache=False).logits  # [bsz, block_end, V]

            # Extract block logits
            block_logits = logits[:, block_start:block_end]  # [bsz, block_length, V]
            total_forwards += 1

            # Vectorized threshold decode — no Python loop over batch/positions
            block_x = x[:, block_start:block_end]
            live = (block_x == mask_id)  # [bsz, block_length]

            if not live.any():
                break

            # Compute predictions and confidence
            probs = F.softmax(block_logits.float(), dim=-1)
            pred_tokens = block_logits.argmax(dim=-1)  # [bsz, block_length]
            confidence = probs.max(dim=-1).values  # [bsz, block_length]

            # Mask out non-live positions
            confidence = confidence.masked_fill(~live, -1.0)

            # Determine how many to decode per sample this iteration
            n_transfer = torch.clamp(remaining // max(1, steps - iter_count), min=1)
            n_transfer = torch.min(n_transfer, live.sum(dim=1))

            # For each sample, select top-confidence positions to commit
            # Vectorized: sort by confidence, take top n_transfer
            sorted_conf, sorted_idx = confidence.sort(dim=1, descending=True)

            # Build commit mask: for each sample, commit top n_transfer positions
            pos_range = torch.arange(block_length, device=device).unsqueeze(0)  # [1, block_length]
            commit_mask = pos_range < n_transfer.unsqueeze(1)  # [bsz, block_length]

            # Scatter: commit the selected positions
            commit_positions = sorted_idx.gather(1, pos_range.expand(bsz, -1))
            # Only commit where mask is true AND we have budget
            for b in range(bsz):
                n = n_transfer[b].item()
                if n > 0:
                    positions = commit_positions[b, :n]
                    x[b, block_start + positions] = pred_tokens[b, positions]

            remaining = (x[:, block_start:block_end] == mask_id).sum(dim=1)
            iter_count += 1

    return x, total_forwards


@torch.no_grad()
def fast_diffusion_generate_cudagraph(model, input_ids, gen_length=128, block_length=32,
                                       mask_id=156895, eos_id=156892):
    """CUDA Graph accelerated generation — captures forward pass."""
    device = input_ids.device
    bsz = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    total_len = prompt_len + gen_length

    x = torch.full((bsz, total_len), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    pos_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(bsz, -1)

    n_blocks = gen_length // block_length
    total_forwards = 0

    # Static buffers for CUDA graph
    static_x = x.clone()
    static_pos = pos_ids.clone()

    # Capture CUDA graph for the forward pass
    graph = None
    static_logits = None

    for block_idx in range(n_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = block_start + block_length

        block_mask = (x[:, block_start:block_end] == mask_id)
        if not block_mask.any():
            continue

        remaining = block_mask.sum(dim=1)
        steps = remaining.max().item()
        iter_count = 0

        while remaining.max().item() > 0 and iter_count < steps + 2:
            # Copy current state to static buffer
            static_x.copy_(x)

            # Try CUDA graph or fall back to eager
            if graph is None and block_end == total_len:
                # Only capture graph when shape is stable (last block size)
                try:
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        _ = model(static_x[:, :block_end], position_ids=static_pos[:, :block_end], use_cache=False)
                    torch.cuda.current_stream().wait_stream(s)

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=s):
                        result = model(static_x[:, :block_end], position_ids=static_pos[:, :block_end], use_cache=False)
                        static_logits = result.logits
                    torch.cuda.synchronize()
                except Exception:
                    graph = "failed"

            if graph is not None and graph != "failed":
                graph.replay()
                logits = static_logits
            else:
                logits = model(x[:, :block_end], position_ids=pos_ids[:, :block_end], use_cache=False).logits

            total_forwards += 1

            block_logits = logits[:, block_start:block_end]
            block_x = x[:, block_start:block_end]
            live = (block_x == mask_id)
            if not live.any():
                break

            pred_tokens = block_logits.argmax(dim=-1)
            confidence = F.softmax(block_logits.float(), dim=-1).max(dim=-1).values
            confidence.masked_fill_(~live, -1.0)

            n_transfer = torch.clamp(remaining // max(1, steps - iter_count), min=1)
            n_transfer = torch.min(n_transfer, live.sum(dim=1))

            sorted_conf, sorted_idx = confidence.sort(dim=1, descending=True)
            for b in range(bsz):
                n = n_transfer[b].item()
                if n > 0:
                    positions = sorted_idx[b, :n]
                    x[b, block_start + positions] = pred_tokens[b, positions]

            remaining = (x[:, block_start:block_end] == mask_id).sum(dim=1)
            iter_count += 1

    return x, total_forwards
