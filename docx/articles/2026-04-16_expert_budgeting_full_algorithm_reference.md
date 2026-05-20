# Expert Budgeting: Full Algorithm Reference

> Purpose: Self-contained reference for algorithm-level optimization review.
> Model: LLaDA2.0-mini (DeepSeek-style MoE, 256 experts, top_k=8, 20 layers, layer 0 dense, layers 1-19 MoE)
> Date: 2026-04-16

---

## 1. Parameters & Dimensions

| Symbol | Value | Description |
|--------|-------|-------------|
| N | ~1024 (varies) | Number of tokens per forward (batch_size x block_length) |
| E | 256 | Total number of routed experts |
| K | 8 | top_k (experts selected per token) |
| H | 1536 | hidden_size |
| I | 1536 | moe_intermediate_size (per expert) |
| n_group | 8 | Number of expert groups for group-limited topk |
| topk_group | 4 | Number of groups selected per token |
| routed_scaling_factor | 2.5 | Weight scaling after normalization |
| K_target | 40 | Expert Budgeting: initial popularity budget |
| quality_floor | 0.85 | Expert Budgeting: minimum per-token weight coverage |
| top_p | 0.75 | Top-p pruning threshold (combined with SHARED_RATE) |
| SHARED_RATE | 0.419 | Fraction of total MoE output from shared experts |
| ROUTING_RATE | 0.581 | Fraction from routed experts (1 - SHARED_RATE) |
| num_shared_experts | 2 | Number of shared experts |

### Expert Weight Dimensions

| Weight | Shape | Description |
|--------|-------|-------------|
| gate.weight | [E, H] = [256, 1536] | Gate linear projection |
| gate.expert_bias | [E] = [256] | Expert selection bias (initialized to 0) |
| experts.w13_weight | [E, 2*I, H] = [256, 3072, 1536] | Fused gate_proj + up_proj |
| experts.w2_weight | [E, H, I] = [256, 1536, 1536] | down_proj |

---

## 2. Baseline MoE Forward (No Expert Budgeting)

Source: `LLaDA2MoeSparseMoeBlock.forward()` (modeling_llada2_moe.py:639)

```python
def forward(self, hidden_states):     # hidden_states: [B, S, H]
    res = self.shared_experts(hidden_states)   # [B, S, H] — shared MLP, unchanged
    bsz, seq_len, h = hidden_states.shape
    hidden_states_flat = hidden_states.view(-1, h)  # [N, H]

    # Step 1: Get raw logits
    router_logits = self.gate.get_logits(hidden_states_flat)  # [N, E]

    # Step 2: Route + compute (inside forward_impl)
    y = self.experts.forward_impl(
        hidden_states=hidden_states_flat,     # [N, H]
        router_logits=router_logits)          # [N, E]
    # forward_impl internally calls:
    #   custom_routing_function(hs, router_logits, top_k, renormalize)
    #   → gate.routing(hs, router_logits, top_k, renormalize)
    #   → topk_weights [N, K], topk_ids [N, K]
    #   then calls fused_experts kernel
    # Output: y [N, H]

    y = y.view(bsz, seq_len, h)              # [B, S, H]
    y = y + res                                # [B, S, H]
    return y
```

**Total operations: get_logits → routing → fused_experts.**

---

## 3. Expert Budgeting Hook Forward (Proven Version)

Source: `BudgetingPlusTopPController.hook_forward()` (expert_budgeting_e2e.py:204)

This replaces the entire `LLaDA2MoeSparseMoeBlock.forward()` via hook.

```python
def hook_forward(self, moe_mod, layer_idx, hidden_states):
    # hidden_states: [B, S, H]
    bsz, seq_len, h = hidden_states.shape
    hs_flat = hidden_states.view(-1, h)           # [N, H]
    N = hs_flat.shape[0]

    # ---- Step A: Shared experts (unchanged) ----
    shared_res = moe_mod.shared_experts(hidden_states)  # [B, S, H]

    # ---- Step B: gate.get_logits (baseline also has this) ----
    gate_logits = moe_mod.gate.get_logits(hs_flat)  # [N, E=256]

    # ---- Step C: gate() full call (EXTRA — for topk_idx/topk_weight) ----
    topk_idx, topk_weight, _ = moe_mod.gate(hs_flat)
    # topk_idx: [N, K=8] int64 — original top-8 expert indices per token
    # topk_weight: [N, K=8] — normalized + scaled routing weights

    # ---- Step D: compute_active_set (CORE ALGORITHM) ----
    S_mask = compute_active_set(
        gate_logits,     # [N, E]
        topk_idx,        # [N, K]
        topk_weight,     # [N, K]
        K_target=40,
        quality_floor=0.85,
        top_p=0.75)
    # S_mask: [E=256] bool — True for active experts

    # ---- Step E: Mask logits ----
    masked_logits = gate_logits.clone()            # [N, E]
    masked_logits[:, ~S_mask] = float('-inf')      # non-S experts → -inf

    # ---- Step F: Re-route with masked logits (EXTRA) ----
    topk_weight_new, topk_idx_new = moe_mod.gate.routing(
        hs_flat, masked_logits, moe_mod.gate.top_k, True)
    # topk_weight_new: [N, K=8] — new weights, only S experts selected
    # topk_idx_new: [N, K=8] — new indices within S

    # ---- Step G: Top-p pruning (EXTRA) ----
    sorted_w, sort_order = topk_weight_new.sort(dim=1, descending=True)
    total_routing = topk_weight_new.sum(dim=1, keepdim=True)    # [N, 1]
    needed_frac = (0.75 - 0.419) / 0.581                        # = 0.570
    threshold = needed_frac * total_routing                      # [N, 1]
    cumsum = sorted_w.cumsum(dim=1)                              # [N, K]
    enough = cumsum >= threshold                                 # [N, K] bool
    enough[:, -1] = True
    cutoff = enough.float().argmax(dim=1) + 1                   # [N] — per-token # of experts

    rank_pos = torch.arange(K, device=...).unsqueeze(0)          # [1, K]
    keep_sorted = rank_pos < cutoff.unsqueeze(1)                 # [N, K] bool
    pruning_mask = torch.zeros_like(topk_weight_new, dtype=torch.bool)
    pruning_mask.scatter_(1, sort_order, keep_sorted)            # [N, K] bool

    kept_sum = (topk_weight_new * pruning_mask.float()).sum(dim=1, keepdim=True)
    orig_sum = topk_weight_new.sum(dim=1, keepdim=True)
    scale = orig_sum / (kept_sum + 1e-8)                         # [N, 1]
    new_weights = topk_weight_new * pruning_mask.float() * scale # [N, K]
    # Pruned tokens have weight=0. Avg ~4 non-zero per token.

    # ---- Step H: fused_experts kernel ----
    routed_y = fused_experts(
        hidden_states=hs_flat,                 # [N, H]
        w1=moe_mod.experts.w13_weight,         # [E, 2*I, H]
        w2=moe_mod.experts.w2_weight,          # [E, H, I]
        topk_weights=new_weights,              # [N, K]
        topk_ids=topk_idx_new,                 # [N, K]
        inplace=False)
    # routed_y: [N, H]

    routed_y = routed_y.view(bsz, seq_len, h)   # [B, S, H]
    out = routed_y + shared_res                   # [B, S, H]
    return out
```

---

## 4. Internal Functions — Complete Code

### 4.1 gate.get_logits()

```python
def get_logits(self, hidden_states):
    # hidden_states: [N, H]
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])  # [N, H]
    logits = F.linear(
        hidden_states.type(torch.float32),     # cast bf16 → fp32
        self.weight.type(torch.float32))       # weight: [E, H]
    # logits: [N, E] fp32
    return logits
```

### 4.2 gate.forward() (gate.__call__)

```python
def forward(self, hidden_states):
    # hidden_states: [N, H]
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])     # [N, H]
    logits = F.linear(
        hidden_states.type(torch.float32),
        self.weight.type(torch.float32))       # [N, E] — SAME as get_logits (redundant!)

    scores = torch.sigmoid(logits.float()).type_as(logits)              # [N, E]

    scores_for_routing = scores + self.expert_bias                      # [N, E]
    _, topk_idx = self.group_limited_topk(scores_for_routing)           # [N, K]

    scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits) # [N, K]

    topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)  # [N, K] normalized
    topk_weight = topk_weight * self.routed_scaling_factor               # [N, K] scaled

    return topk_idx, topk_weight, logits
    # topk_idx: [N, K] int64
    # topk_weight: [N, K] fp32 (or bf16)
    # logits: [N, E] fp32
```

### 4.3 gate.group_limited_topk()

```python
def group_limited_topk(self, scores):
    # scores: [N, E=256]
    num_tokens, _ = scores.size()

    # Step 1: Compute group scores (top-2 per group, summed)
    group_scores = scores.view(num_tokens, self.n_group, -1)  # [N, 8, 32]
    group_scores = group_scores.topk(2, dim=-1)[0]            # [N, 8, 2]
    group_scores = group_scores.sum(dim=-1)                    # [N, 8]

    # Step 2: Select top groups
    group_idx = torch.topk(
        group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [N, 4]
    group_mask = torch.zeros_like(group_scores)                     # [N, 8]
    group_mask.scatter_(1, group_idx, 1)                            # [N, 8] binary

    # Step 3: Expand group mask to expert mask
    score_mask = (
        group_mask.unsqueeze(-1)                       # [N, 8, 1]
        .expand(num_tokens, self.n_group,
                self.num_experts // self.n_group)       # [N, 8, 32]
        .reshape(num_tokens, -1)                        # [N, 256]
    )

    # Step 4: Mask non-selected groups, then topk within selected groups
    masked_scores = scores.masked_fill(
        ~score_mask.bool(), float('-inf'))               # [N, 256]
    probs, top_indices = torch.topk(
        masked_scores, k=self.top_k, dim=-1)             # [N, K=8], [N, K=8]

    return probs, top_indices
    # probs: [N, K] — scores of selected experts (with -inf for masked)
    # top_indices: [N, K] — expert indices
```

### 4.4 gate.routing()

```python
def routing(self, hidden_states, gating_output, topk, renormalize):
    # hidden_states: [N, H] (unused in computation, passed for interface compatibility)
    # gating_output: [N, E] — pre-computed logits (possibly masked)

    scores = torch.sigmoid(gating_output.float()).type_as(gating_output)  # [N, E]
    # If gating_output has -inf entries: sigmoid(-inf) = 0.0

    scores_for_routing = scores + self.expert_bias               # [N, E]
    _, topk_idx = self.group_limited_topk(scores_for_routing)    # [N, K]

    scores = torch.gather(
        scores, dim=1, index=topk_idx).type_as(gating_output)   # [N, K]

    topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)  # [N, K]
    topk_weight = topk_weight * self.routed_scaling_factor                # [N, K]

    return topk_weight, topk_idx
    # topk_weight: [N, K]
    # topk_idx: [N, K]
```

### 4.5 compute_active_set()

```python
def compute_active_set(gate_logits, topk_idx, topk_w, K_target, quality_floor,
                       top_p=0.75):
    """
    Input:
        gate_logits: [N, E=256] fp32 — raw gate logits
        topk_idx:    [N, K=8]  int64 — original top-8 expert indices per token
        topk_w:      [N, K=8]  fp32  — original routing weights (normalized+scaled)
        K_target:    int             — initial budget (e.g. 40)
        quality_floor: float         — min coverage ratio (e.g. 0.85)
        top_p:       float           — top-p threshold (e.g. 0.75)

    Output:
        S_mask: [E=256] bool — True for active experts (|S| typically 128-141)
    """
    N = gate_logits.shape[0]
    device = gate_logits.device

    # ======== Phase 1: Popularity-based initial S ========

    gate_w = torch.softmax(gate_logits.float(), dim=-1)  # [N, E]
    popularity = gate_w.sum(dim=0)                        # [E]
    _, pop_order = popularity.sort(descending=True)        # [E] indices
    S_mask = torch.zeros(NUM_EXPERTS, dtype=torch.bool, device=device)  # [E]
    S_mask[pop_order[:K_target]] = True                    # top-40 by popularity

    # ======== Phase 2: Per-token k_budget computation ========

    # How many of the original top-8 experts each token needs (based on top-p)
    sorted_rw, sort_order = topk_w.sort(dim=1, descending=True)  # [N, K]
    total_routing = topk_w.sum(dim=1, keepdim=True)               # [N, 1]
    needed_frac = (top_p - SHARED_RATE) / ROUTING_RATE             # scalar: 0.570
    threshold = needed_frac * total_routing                        # [N, 1]
    cumsum = sorted_rw.cumsum(dim=1)                               # [N, K]
    enough = cumsum >= threshold                                   # [N, K] bool
    enough[:, -1] = True                                           # at least all K
    k_budgets = (enough.float().argmax(dim=1) + 1).int()          # [N] int

    # Original coverage weight (how much weight is covered by k_budget experts)
    sorted_idx = topk_idx.gather(1, sort_order)                    # [N, K] expert indices sorted by weight
    original_gate_vals = gate_w.gather(1, sorted_idx)              # [N, K] gate_w values
    positions = torch.arange(TOP_K_ORIG, device=device).unsqueeze(0)  # [1, K]
    topp_mask = positions < k_budgets.unsqueeze(1)                 # [N, K] bool
    original_weight = (original_gate_vals * topp_mask.float()).sum(dim=1)  # [N]

    # ======== Phase 3: Iterative safety check (max 30 iterations) ========

    for iter_i in range(30):
        # Current S indices
        s_indices = S_mask.nonzero(as_tuple=True)[0]          # [|S|] int

        # For each token, sort S experts by gate_w, take top k_budget
        s_gate_w = gate_w[:, s_indices]                        # [N, |S|]
        s_sorted_w, _ = s_gate_w.sort(dim=1, descending=True) # [N, |S|]

        s_positions = torch.arange(
            s_sorted_w.shape[1], device=device).unsqueeze(0)   # [1, |S|]
        s_topp_mask = s_positions < k_budgets.unsqueeze(1)     # [N, |S|] bool
        covered = (s_sorted_w * s_topp_mask.float()).sum(dim=1) # [N]

        # Coverage ratio
        safe = original_weight > 1e-8                           # [N] bool
        cov_ratio = torch.where(
            safe,
            covered / original_weight.clamp(min=1e-8),
            torch.ones(N, device=device))                       # [N]

        # Check violations
        violated = (cov_ratio < quality_floor) & safe           # [N] bool
        if not violated.any():
            break  # All tokens satisfied — early exit

        # Find best missing expert for each violated token
        violated_idx = violated.nonzero(as_tuple=True)[0]       # [V] int
        v_gate = gate_w[violated_idx]                            # [V, E]
        v_gate_masked = v_gate.clone()                           # [V, E]
        v_gate_masked[:, S_mask] = -1                            # exclude current S
        best_missing = v_gate_masked.argmax(dim=1)               # [V] int

        # Add missing experts to S
        for e in best_missing.unique():           # Python loop + GPU→CPU sync
            S_mask[e.item()] = True               # .item() triggers sync

    return S_mask  # [E] bool, |S| typically 128-141
```

### 4.6 Top-p Pruning (inline in hook_forward, Step G)

```python
# Input: topk_weight_new [N, K=8], from gate.routing with masked logits
# Output: new_weights [N, K=8], with ~4 non-zero entries per token

sorted_w, sort_order = topk_weight_new.sort(dim=1, descending=True)  # [N, K]

total_routing = topk_weight_new.sum(dim=1, keepdim=True)      # [N, 1]
needed_frac = (top_p - SHARED_RATE) / ROUTING_RATE              # 0.570
threshold = needed_frac * total_routing                         # [N, 1]

cumsum = sorted_w.cumsum(dim=1)                                 # [N, K]
enough = cumsum >= threshold                                    # [N, K] bool
enough[:, -1] = True
cutoff = enough.float().argmax(dim=1) + 1                      # [N] int — experts to keep

# Build pruning mask in original order
rank_pos = torch.arange(K, device=...).unsqueeze(0)             # [1, K]
keep_sorted = rank_pos < cutoff.unsqueeze(1)                    # [N, K] bool (sorted order)
pruning_mask = torch.zeros_like(topk_weight_new, dtype=torch.bool)  # [N, K]
pruning_mask.scatter_(1, sort_order, keep_sorted)                # [N, K] bool (original order)

# Renormalize: scale kept weights so their sum = original sum
kept_sum = (topk_weight_new * pruning_mask.float()).sum(dim=1, keepdim=True)  # [N, 1]
orig_sum = topk_weight_new.sum(dim=1, keepdim=True)                           # [N, 1]
scale = orig_sum / (kept_sum + 1e-8)                                           # [N, 1]
new_weights = topk_weight_new * pruning_mask.float() * scale                   # [N, K]
# Pruned entries have weight = 0.
```

### 4.7 fused_experts (interface only)

```python
def fused_experts(
    hidden_states: torch.Tensor,    # [N, H] bf16 — token hidden states
    w1: torch.Tensor,               # [E, 2*I, H] bf16 — fused gate_proj + up_proj
    w2: torch.Tensor,               # [E, H, I] bf16 — down_proj
    topk_weights: torch.Tensor,     # [N, K] fp32 — routing weights (0 = skip)
    topk_ids: torch.Tensor,         # [N, K] int64 — expert indices
    inplace: bool = False,
) -> torch.Tensor:                  # [N, H] bf16 — routed output
    """
    For each token i, for each selected expert j in topk_ids[i]:
      expert_out = w2[j] @ activation(w1[j] @ hidden_states[i])
      output[i] += topk_weights[i, j] * expert_out

    Implementation: vllm Triton kernel with moe_align_block_size.
    Memory-bound on H100: bottleneck is loading expert weights from HBM.
    Kernel time ~ 18.4μs (fixed) + 1.13μs × (num_unique_active_experts).
    """
```

---

## 5. Profiling Data

Measured on optimized baseline (max_unroll=4 + fused RMSNorm + flash-attn 2.8.3).
HetEval-32 config, per MoE layer per forward (19 MoE layers, ~262 forwards).

| Step | What | Avg (μs) | % | In Baseline? |
|------|------|----------|---|------------|
| A | shared_experts | 602 | 9.6% | Yes |
| B | gate.get_logits | 227 | 3.6% | Yes |
| C | gate() full call | 381 | 6.1% | **No** (redundant F.linear + full routing) |
| D | compute_active_set | 4,234 | 67.3% | **No** (core algorithm + safety loop) |
| E | mask logits | 93 | 1.5% | **No** (clone + fill) |
| F | gate.routing | 292 | 4.6% | Replaces baseline's internal routing |
| G | top-p pruning | 279 | 4.4% | **No** (Layer 1 optimization) |
| H | fused_experts | 184 | 2.9% | Yes (kernel itself, benefits from fewer experts) |
| | **TOTAL** | **6,291** | **100%** | |

**Baseline MoE per layer ≈ 1,065μs.** Hook version = 6,291μs (**5.9x slower**).

### compute_active_set internal breakdown (Step D, 4,234μs)

| Phase | Description | Estimated |
|-------|-------------|-----------|
| D-1 | softmax + popularity + initial S | ~400μs |
| D-2 | k_budget + original_weight | ~600μs |
| D-3 | Safety loop (avg 2.1 iters × ~1,500μs) | ~3,150μs |
| | Including: fancy indexing [N, \|S\|], sort, GPU→CPU sync in Python for-loop | |

---

## 6. Computation Flow Diagram

```
Baseline MoE Forward:
  hidden_states [N,H]
    │
    ├── shared_experts ──────────────────────────── res [B,S,H]
    │
    ├── gate.get_logits ─── logits [N,E] ─── forward_impl ──┐
    │                                         (internal:      │
    │                                          routing →       │
    │                                          fused_experts)  │
    │                                                          ↓
    └──────────────────────────────────────────── y [N,H] + res → output


Expert Budgeting Hook Forward:
  hidden_states [N,H]
    │
    ├── A: shared_experts ───────────────────────── res [B,S,H]
    │
    ├── B: gate.get_logits ─── logits [N,E] ───────────────────┐
    │                                                            │
    ├── C: gate() ─── topk_idx [N,K], topk_w [N,K] ──┐        │
    │   (redundant F.linear!)                          │        │
    │                                                   ↓        │
    ├── D: compute_active_set ───── S_mask [E] ────────┤        │
    │   (softmax, popularity, safety loop)              │        │
    │                                                   │        ↓
    ├── E: mask logits ─── masked_logits [N,E] ────────┘───────┘
    │   (clone + fill -inf)
    │
    ├── F: gate.routing(masked_logits) ─── topk_w_new [N,K], topk_idx_new [N,K]
    │
    ├── G: top-p pruning ─── new_weights [N,K] (with ~4 non-zero per token)
    │
    ├── H: fused_experts(hs, w1, w2, new_weights, topk_idx_new)
    │
    └── routed_y [N,H] + res → output [B,S,H]
```

---

## 7. Identified Redundancies & Optimization Opportunities

### 7.1 Redundant F.linear in Step C

`gate.get_logits()` (Step B) and `gate()` (Step C) both compute:
```python
logits = F.linear(hidden_states.float(), self.weight.float())  # [N, E]
```
This is the same [N, 1536] × [256, 1536]^T matrix multiplication, computed twice.

**Opportunity:** `gate.forward()` already returns `logits` as its third output. If called instead of `get_logits`, we get logits + topk_idx + topk_weight in one call, eliminating Step B entirely.

### 7.2 compute_active_set uses softmax, gate uses sigmoid

- `compute_active_set`: `gate_w = softmax(gate_logits)` for popularity
- `gate.forward / routing`: `scores = sigmoid(gate_logits)` for routing

These are different normalizations of the same logits. The popularity ranking is likely similar under both (softmax and sigmoid are both monotonic), but they produce different absolute values. This means `original_weight` (computed from softmax-domain values) and the routing weights (sigmoid-domain) are in different scales.

**Question for expert:** Could the safety constraint use sigmoid scores directly, avoiding the softmax computation?

### 7.3 Safety loop has GPU→CPU sync

```python
for e in best_missing.unique():    # .unique() → GPU tensor
    S_mask[e.item()] = True        # .item() → GPU→CPU sync per expert
```

Each `.item()` forces GPU→CPU synchronization, stalling the GPU pipeline. With ~5-15 unique experts per iteration and ~2 iterations, this is ~10-30 sync points per layer.

**Opportunity:** Replace with vectorized:
```python
S_mask[best_missing.unique()] = True  # single GPU operation, no .item()
```

### 7.4 Fancy indexing in safety loop

Each iteration does:
```python
s_gate_w = gate_w[:, s_indices]         # [N, |S|] — |S| changes each iteration
s_sorted_w, _ = s_gate_w.sort(dim=1)   # [N, |S|] — sort varies with |S|
```

Dynamic `|S|` (growing from 40 to ~141) causes varying tensor shapes per iteration, preventing any kernel fusion or caching.

### 7.5 gate.routing (Step F) repeats sigmoid + group_limited_topk

Step F calls `gate.routing(masked_logits)` which internally does:
```python
scores = sigmoid(masked_logits)         # [N, E] — sigmoid again
group_limited_topk(scores + bias)       # full group selection again
gather + normalize                      # weight computation
```

The sigmoid was already computed in `compute_active_set` (as softmax, but sigmoid would also be available from Step C). The group_limited_topk is a complex multi-step operation.

**Question for expert:** If we already have `topk_idx_new` from a simplified routing within S, can we avoid the full group_limited_topk? Or can we share the sigmoid computation between steps?

### 7.6 Top-p pruning (Step G) largely overlaps with k_budget in Step D

Step D computes `k_budgets` (how many experts each token needs) using exactly the same top-p logic as Step G. The difference:
- Step D: computed on original routing weights (from all 256 experts)
- Step G: computed on re-routed weights (from S experts only)

**Question for expert:** Are these two computations truly independent, or could they be unified?

### 7.7 Summary: Minimum required computation

If all redundancies were eliminated, the minimum computation would be:

```
1. gate(hs_flat) — one call: logits [N,E] + topk_idx [N,K] + topk_w [N,K]
2. Determine S from logits (popularity + safety)
3. Mask logits[:, ~S] = -inf
4. Re-route within S → new topk_idx, topk_weight
5. Optional top-p → new_weights
6. fused_experts(hs, w1, w2, new_weights, new_topk_idx)
```

The core question: **Can steps 2-5 be done with significantly fewer operations than the current implementation?**
