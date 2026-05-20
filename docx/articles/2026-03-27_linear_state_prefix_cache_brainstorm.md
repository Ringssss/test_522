# Linear State Prefix/ Radix Cache Brainstorm

## 1. Purpose

This document records the current understanding of:

- how full-attention prefix cache and radix cache work,
- how hybrid linear-attention models differ,
- what `linear state` means in this project context,
- and what a stronger research direction could look like for linear-state-aware prefix caching.

The discussion is grounded in:

- the local Qwen3.5-35B-A3B run on SGLang,
- the current SGLang implementation of `RadixCache`, `MambaRadixCache`, and `HiMambaRadixCache`,
- and the user's proposed CPU pinned-memory linear-state cache pool idea.

## 2. Verified Runtime Facts from the Current Local Run

Online server used:

- `http://127.0.0.1:31000`

Model:

- `/home/wuhang/models/Qwen3.5-35B-A3B`

Observed server-side configuration:

- `disable_radix_cache = False`
- `disable_chunked_prefix_cache = False`
- `page_size = 1`
- `mamba_scheduler_strategy = no_buffer`
- full-attention backend = `fa3`
- linear-attention backend = `triton`

Observed memory summary from the active server:

- weight memory per rank: `32.81 GB`
- KV cache per rank: `15.39 GB`
- CUDA graph memory per rank: `2.84 GB`
- token capacity per rank: `1,613,942`

Observed linear-state memory allocation in logs:

- `conv_state`: about `0.32 GB`
- `ssm_state`: about `13.54 GB`

This is the practical basis for the comparisons below.

## 3. Full-Attention Prefix Cache and Radix Cache

### 3.1 Prefix cache in plain language

For full attention, if two requests share the same token prefix, then the KV cache produced by the prefill of that prefix is identical.

Therefore:

- the later request does not need to recompute the shared prefix,
- it can directly reuse the KV cache produced by the earlier request,
- and only compute the uncached suffix.

### 3.2 Radix cache as the implementation

In SGLang, `RadixCache` is the data structure that implements this reuse.

Conceptually:

- keys are token prefixes,
- nodes represent cached spans,
- and the longest common prefix between two requests corresponds to a shared path in the radix tree.

The ordinary full-attention path:

1. matches the request prefix in the radix tree,
2. returns the physical KV indices of the matched prefix,
3. skips recomputation for those tokens,
4. and inserts newly computed KV segments back into the tree.

Relevant code:

- `/home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/radix_cache.py`

Important practical detail:

- prefix reuse depends on `page_size`
- when `page_size > 1`, only page-aligned prefixes can be matched
- when `page_size = 1`, token-level reuse is possible

This is documented in:

- `/home/wuhang/wuhang/linear_wh/sglang/docs/advanced_features/attention_backend.md`

## 4. Why Hybrid Linear-Attention Models Are Different

For hybrid linear-attention models such as Qwen3.5, prefix reuse cannot rely on KV cache alone.

The reason is that full-attention layers and linear-attention layers do not preserve the same type of reusable intermediate state.

For full-attention layers:

- reusable intermediate = KV cache over token history

For linear-attention layers:

- reusable intermediate = recurrent or state-space-like summary state at a prefix boundary

In this project discussion, this recurrent state is referred to as:

- `linear state`

So for hybrid models, reusing only full-attention KV is insufficient. The runtime must also know whether the prefix boundary has a reusable linear-state snapshot.

## 5. What Linear State Means Here

The current SGLang implementation for hybrid SSM/GDN-style models stores two major state components:

- `conv_state`
- `temporal/ssm_state`

These are allocated in:

- `/home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/memory_pool.py`

Their shape and per-request memory cost are defined through:

- `/home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/configs/mamba_utils.py`

Most important conceptual distinction:

- KV cache is token-local history storage
- linear state is a snapshot of the recurrent system after processing a prefix

That means linear state is not naturally split the same way KV spans are split.

## 6. Why Linear State Is Coarser Than KV Cache

For the current local Qwen3.5 run, rough per-rank numbers are:

- full-attention KV cost per token: about `10 KB / token / rank`
- linear-state snapshot cost per request state slot: about `30.5 MB / snapshot / rank`

This means:

- one linear-state snapshot is roughly comparable to more than `3000` full-attention prefix tokens worth of KV cache

This is the key reason the caching strategy should differ:

- KV cache can be stored densely
- linear-state snapshots should be much sparser and much more selective

## 7. Current SGLang Support for Hybrid Linear State Reuse

The system already distinguishes hybrid linear-attention models from ordinary full-attention models.

Scheduler behavior:

- full-attention-only models use `RadixCache`
- hybrid SSM models use `MambaRadixCache`
- hierarchical host-storage variants use `HiMambaRadixCache`

Relevant scheduler entry:

- `/home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/managers/scheduler.py`

Qwen3-Next documentation also explicitly states:

- SGLang supports `MambaRadixCache`
- default mode: `no_buffer`
- optimized mode: `extra_buffer`

Relevant doc:

- `/home/wuhang/wuhang/linear_wh/sglang/docs/basic_usage/qwen3.md`

## 8. What `MambaRadixCache` Changes Relative to `RadixCache`

### 8.1 Extra payload on tree nodes

Ordinary `RadixCache` nodes carry:

- token key span
- KV indices span

`MambaRadixCache` nodes additionally carry:

- optional `mamba_value` / linear-state snapshot

### 8.2 Matching is more conservative

For full-attention radix cache, the longest token prefix match is enough.

For `MambaRadixCache`, the reusable prefix is effectively bounded by the latest prefix point that has a valid linear-state checkpoint.

So:

- token-prefix match can be longer,
- but state-safe reusable match can be shorter.

This is why the implementation tracks:

- `best_value_len`
- `mamba_branching_seqlen`

### 8.3 Linear state cannot be split like KV

In ordinary `RadixCache`, an internal node split can naturally split the KV span.

In `MambaRadixCache`, the new split internal node explicitly has:

- no `mamba_value`

Meaning:

- linear-state snapshots do not behave like per-token spans
- they are boundary snapshots, not divisible ranges

### 8.4 Copy-on-write semantics

When a request matches a cached linear-state snapshot, the runtime copies that state into request-local state storage before continuing execution.

This is effectively:

- state reuse with copy-on-write semantics

That is necessary because the running request will mutate the local state while decoding or extending.

## 9. Evaluation of the Proposed CPU Linear-State Cache Pool Idea

### 9.1 Core proposal

The user's idea is:

- keep hot linear-state snapshots on GPU,
- maintain a larger CPU-side linear-state pool in pinned memory,
- store additional checkpoints more densely for earlier prefix positions,
- and on a GPU miss:
  - first try CPU state cache,
  - then fall back to recomputation if CPU also misses.

### 9.2 Why this idea is strong

This idea is well aligned with the actual properties of linear state:

- linear-state snapshots are too expensive to store densely on GPU
- but they may still be much cheaper to fetch than to recompute a long suffix
- early prefix checkpoints are more likely to be globally reused

This means the proposal is directionally correct.

### 9.3 What is still missing for a research-quality design

A simple “GPU miss -> CPU fetch” policy is not enough.

The stronger version should instead decide among:

- GPU hit
- CPU host hit
- recompute from a previous checkpoint

based on a cost comparison.

Also, the design should treat the reusable object as:

- hybrid prefix checkpoint

rather than:

- linear state only

because the full-attention KV portion is still required.

## 10. Stronger Research Form of the Idea

The research-grade version of the proposal would look like:

### 10.1 Hybrid prefix checkpoint object

Each reusable checkpoint should include:

- token prefix boundary
- full-attention KV span metadata
- linear-state snapshot metadata
- branch and alignment metadata

### 10.2 Two-level checkpoint residency

- hot GPU resident checkpoints
- larger CPU host resident checkpoints

Optionally later:

- storage-backed cold tier

### 10.3 Checkpoint placement policy

Not only:

- dense near position 0
- sparse later

but:

- value-aware checkpoint selection based on estimated reuse and recomputation cost

### 10.4 Runtime restore policy

At a miss, choose among:

- GPU hit
- host fetch
- recompute

instead of always preferring host fetch.

### 10.5 Optional state compression

Host-side state caching becomes much more attractive if one also considers:

- bf16 host snapshots
- fp8 host snapshots
- asymmetric compression of temporal state and conv state
- delta or low-rank encoding between adjacent checkpoints

## 11. Why This Could Be ASPLOS-Like

This direction becomes more like a systems paper if it is framed as:

- a heterogeneous cache problem,
- for hybrid linear-attention models,
- where reusable state is sparse, coarse-grained, and non-splittable,
- unlike ordinary token-level KV cache reuse.

The strongest paper angle is likely not:

- “we store more state on CPU”

but:

- “we design a checkpointed prefix cache for hybrid linear-attention serving, with heterogeneous residency, placement, and restore policies.”

## 12. Main Open Research Questions

The current open questions worth exploring next are:

1. Which prefix positions should become state checkpoints?
2. How should checkpoint density vary with:
   - position,
   - observed reuse,
   - branch frequency,
   - and recompute cost?
3. When is host fetch cheaper than recompute?
4. Should linear-state checkpoints and KV spans share a joint policy or separate budgets?
5. How much of the host-side state can be compressed without harming correctness or restore cost?
6. Can asynchronous prefetch hide host-to-device latency for likely future prefix hits?

## 13. Suggested Next Step

The next best follow-up artifact would be:

- a design draft for a `Hybrid Prefix Checkpoint Cache`

including:

- cache object definition,
- GPU/CPU tier separation,
- checkpoint placement policy,
- restore policy,
- and evaluation plan.

## 14. Key File References

Most relevant local files:

- `/home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/radix_cache.py`
- `/home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/mamba_radix_cache.py`
- `/home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/hi_mamba_radix_cache.py`
- `/home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/memory_pool.py`
- `/home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/configs/mamba_utils.py`
- `/home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/server_args.py`
- `/home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/managers/scheduler.py`
- `/home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/managers/schedule_batch.py`
- `/home/wuhang/wuhang/linear_wh/sglang/docs/basic_usage/qwen3.md`
- `/home/wuhang/wuhang/linear_wh/sglang/docs/advanced_features/attention_backend.md`
