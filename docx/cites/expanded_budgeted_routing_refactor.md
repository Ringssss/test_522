# Expanded Top-(K+M) Budgeted Routing 重构说明（LLaDA-MoE 2.0 mini / dLLM）

> 目标：在保留原始 reroute + final top-p trimming 逻辑的前提下，用 **expanded top-(K+M)** 替代原来的 dense `compute_active_set()` 安全循环，从而降低开销，同时保留“原 top-k 外但在 restricted reroute 中可能上浮”的近邻 expert 表达能力。

---

## 1. 最终确定的设计原则

这次重构的核心原则只有三条：

1. **候选专家空间** 用 `top-(K+M)`  
   不再把候选空间限制死在原始 top-k 上。这样可以保留：
   - 原 top-k 内的主专家
   - 原 top-k 外但分数接近、在 restricted reroute 中可能上浮的近邻专家

2. **所有分数统一使用 expanded weight**  
   所有核心量都在同一个语义空间里定义：
   - 初始 expert 重要性
   - token coverage
   - token 缺口
   - batch add expert 的打分  
   全部基于 `expanded weight`

3. **质量阈值只锚定 expanded 前 K 个的总质量，不锚定全部 K+M**
   对每个 token：
   - 先构造 `top-(K+M)` 的 expanded weight
   - 再只取其中按 expanded weight 排序后的前 K 个
   - 用这前 K 个的总质量定义质量阈值  
   即：
   \[
   r_i = quality\_floor \cdot \sum_{j=1}^{K} W^{expanded}_{i,(j)}
   \]
   这样：
   - 候选空间扩大了
   - 但质量目标不会因为多引入 M 个专家而被机械抬高

---

## 2. 与原算法相比，哪些地方变了，哪些地方不变

## 保留不变
- `shared_experts(hidden_states)`
- 最后的 `gate.routing(masked_logits, ...)`
- 最后的 top-p pruning / final trim
- `fused_experts(...)`

## 删除/替换
- 删除原来的 dense `compute_active_set()`
  - 不再使用 dense `softmax(gate_logits)` popularity
  - 不再使用 dense safety loop
  - 不再每轮在 `gate_w[:, S]` 上重排前缀
- 删除 `gate.get_logits()` + `gate()` 双调用
  - 改成一次拿到所有必要信息

---

## 3. 术语定义

### 3.1 维度
- `N = batch_size * seq_len`
- `E = 256`：总 routed experts 数
- `K = 8`：原始 top-k
- `M`：扩展出来的额外候选 expert 数，例如 2 / 4
- `K_ext = K + M`

### 3.2 关键张量
- `gate_logits`: `[N, E]`
- `scores_full`: `[N, E]`，定义为 `sigmoid(gate_logits)`  
- `topkm_idx`: `[N, K_ext]`，每个 token 的 top-(K+M) expert id
- `topkm_weight`: `[N, K_ext]`，expanded weight
- `S_mask`: `[E] bool`，本层本轮 active experts

---

## 4. Expanded weight 的定义

> 必须和 routing 语义一致。不要再切回 softmax popularity 空间。

对每个 token：

1. 计算 `scores_full = sigmoid(gate_logits)`
2. 取 `scores_full` 的 top-(K+M)：
   - `topkm_idx`
   - `topkm_score`
3. 在这 `K+M` 个 score 内做归一化
4. 再乘原始 `routed_scaling_factor`

公式：

\[
W^{expanded}_{ij}
=
\frac{s_{ij}}{\sum_{t=1}^{K+M} s_{it} + \epsilon}
\cdot routed\_scaling\_factor
\]

其中：
- `s_ij` 是 token `i` 在其 top-(K+M) 第 `j` 个 expert 上的 sigmoid score

注意：
- `topkm_weight` 的总和固定为 `routed_scaling_factor`
- 在当前模型里通常是 `2.5`

---

## 5. 整体 forward 流程

## Step 0. Shared experts
先执行 shared experts，不改：

```python
shared_res = moe_mod.shared_experts(hidden_states)   # [B, S, H]
```

---

## Step 1. 一次拿到 gate logits
不要再重复算线性层。建议直接取 logits，然后自行构造 expanded 信息：

```python
gate_logits = moe_mod.gate.get_logits(hs_flat)       # [N, E]
scores_full = torch.sigmoid(gate_logits.float())     # [N, E]
```

> 说明：  
> 原来的 `gate()` 只返回 top-k，不够用。  
> 现在我们需要 top-(K+M)，所以直接从 logits / sigmoid score 自己做扩展候选提取更自然。  
> 后面的 final reroute 仍然调用现有 `moe_mod.gate.routing(...)`。

---

## Step 2. 构造 top-(K+M) expanded candidates
对 `scores_full` 取前 `K+M` 个：

```python
topkm_score, topkm_idx = torch.topk(scores_full, k=K_ext, dim=1)
topkm_weight = topkm_score / (topkm_score.sum(dim=1, keepdim=True) + 1e-20)
topkm_weight = topkm_weight * routed_scaling_factor
```

得到：
- `topkm_idx [N, K_ext]`
- `topkm_weight [N, K_ext]`

这套 weight 是后面所有分数的统一基底。

---

## Step 3. 定义每个 token 的质量阈值 `r_i`
### 3.1 对 expanded weight 降序排序
如果 `topk` 已经返回有序，可直接使用；如果不保证，显式排序：

```python
sorted_w, sort_order = topkm_weight.sort(dim=1, descending=True)
sorted_idx = topkm_idx.gather(1, sort_order)
```

### 3.2 质量阈值
只用 expanded 前 K 个的总和定义阈值：

\[
r_i = quality\_floor \cdot \sum_{j=1}^{K} W^{expanded}_{i,(j)}
\]

实现：

```python
r = quality_floor * sorted_w[:, :K].sum(dim=1)    # [N]
```

注意：
- 这里不用全部 `K+M` 的总和
- 这里也不再切回原始 top-k weight
- 这保证了“候选扩展”和“质量线定义”在同一个分数体系里

---

## Step 4. 构造初始 active set
对每个 expert，累加它在所有 token 的 expanded weight：

\[
popularity(e)=\sum_i \sum_{j=1}^{K+M} \mathbf{1}[topkm\_idx_{ij}=e]\cdot topkm\_weight_{ij}
\]

实现：

```python
popularity = torch.zeros(E, device=topkm_weight.device, dtype=topkm_weight.dtype)
popularity.scatter_add_(0, topkm_idx.reshape(-1), topkm_weight.reshape(-1))

_, pop_order = popularity.sort(descending=True)
S_mask = torch.zeros(E, dtype=torch.bool, device=topkm_weight.device)
S_mask[pop_order[:K_target]] = True
```

说明：
- 初始 S 不再只看原始 top-k support
- top-k 外但在 expanded 中反复出现的近邻 expert，有机会更早进入 S

---

## Step 5. 计算当前 coverage / gap
对 token `i`，当前 coverage 定义为：

\[
c_i(S)=\sum_{e \in S \cap top\text{-}(K+M)} W^{expanded}_{ie}
\]

实现：

```python
in_S = S_mask[topkm_idx]                                # [N, K_ext]
c = (topkm_weight * in_S.float()).sum(dim=1)           # [N]
d = (r - c).clamp_min(0.0)                             # [N]
```

其中：
- `r`: 质量阈值
- `d`: 当前缺口

满足条件的 token：

```python
satisfied = d <= 0
sat_ratio = satisfied.float().mean()
```

---

## Step 6. Batch add experts（批量补 expert）
停止条件：
- `sat_ratio >= q_major`
- 或达到 `max_add_rounds`

### 6.1 候选边
只看当前未满足 token 的 `top-(K+M)` 边：

```python
unsat = ~satisfied
edge_mask = unsat.unsqueeze(1).expand_as(topkm_idx)

edge_expert = topkm_idx[edge_mask]       # [M_edges]
edge_weight = topkm_weight[edge_mask]    # [M_edges]
edge_token  = token_ids[edge_mask]       # [M_edges]
```

再去掉已在 `S` 中的 expert。

### 6.2 候选 expert 的打分
推荐用混合分数：

\[
Score(e)=\alpha H(e)+\beta G(e)
\]

其中：

#### (1) 缺口收益
\[
G(e)=\sum_{i \in U} \min(W^{expanded}_{ie}, d_i)
\]

#### (2) 直接过线收益
\[
H(e)=\sum_{i \in U} \mathbf{1}[c_i + W^{expanded}_{ie} \ge r_i]
\]

实现示意：

```python
G = torch.zeros(E, device=topkm_weight.device, dtype=topkm_weight.dtype)
H = torch.zeros(E, device=topkm_weight.device, dtype=topkm_weight.dtype)

gap_gain = torch.minimum(edge_weight, d[edge_token])
hit_gain = (c[edge_token] + edge_weight >= r[edge_token]).to(topkm_weight.dtype)

G.scatter_add_(0, edge_expert, gap_gain)
H.scatter_add_(0, edge_expert, hit_gain)

Score = alpha * H + beta * G
Score[S_mask] = -1e30
```

### 6.3 每轮加入一批 expert
```python
_, add_order = Score.sort(descending=True)
new_experts = add_order[:batch_add_size]
valid = Score[new_experts] > 0
new_experts = new_experts[valid]
S_mask[new_experts] = True
```

建议：
- `batch_add_size = 8` 或 `16`
- `max_add_rounds = 2` 或 `3`

### 6.4 重算 coverage / gap
回到 Step 5 重新计算，直到：
- 大多数 token 达标
- 或轮数耗尽

---

## Step 7. 在 S 内 reroute
这一步沿用原逻辑：

```python
masked_logits = gate_logits.masked_fill(~S_mask.unsqueeze(0), float('-inf'))
topk_weight_new, topk_idx_new = moe_mod.gate.routing(
    hs_flat, masked_logits, moe_mod.gate.top_k, True
)
```

说明：
- reroute 仍然是原始 `top_k=8`
- 仍然由现有 gate 的 `routing(...)` 实现
- 这样能最大限度减少对后半段执行逻辑的干扰

---

## Step 8. 保留原 final top-p pruning
虽然前面已经重写了 active set 逻辑，但 final trim 先保留，用作稳定兜底。

仍然按原逻辑：
1. 对 `topk_weight_new` 排序
2. 找到达到 target fraction 的最小前缀
3. 其余位置清零
4. 再做 renorm

这一块先不改。

---

## Step 9. Fused experts
不改：

```python
routed_y = fused_experts(
    hidden_states=hs_flat,
    w1=moe_mod.experts.w13_weight,
    w2=moe_mod.experts.w2_weight,
    topk_weights=new_weights,
    topk_ids=topk_idx_new,
    inplace=False
)
```

最后 reshape，并加回 `shared_res`。

---

## 6. 完整伪代码骨架

```python
def expanded_budgeted_forward(
    moe_mod,
    hidden_states,
    K_target=40,
    K=8,
    M=4,
    quality_floor=0.85,
    q_major=0.95,
    batch_add_size=8,
    max_add_rounds=3,
    alpha=1.0,
    beta=0.5,
):
    bsz, seq_len, h = hidden_states.shape
    hs_flat = hidden_states.view(-1, h)
    N = hs_flat.shape[0]
    E = moe_mod.gate.num_experts
    K_ext = K + M
    routed_scaling_factor = moe_mod.gate.routed_scaling_factor

    # Step 0
    shared_res = moe_mod.shared_experts(hidden_states)

    # Step 1
    gate_logits = moe_mod.gate.get_logits(hs_flat)         # [N, E]
    scores_full = torch.sigmoid(gate_logits.float())       # [N, E]

    # Step 2
    topkm_score, topkm_idx = torch.topk(scores_full, k=K_ext, dim=1)   # [N, K_ext]
    topkm_weight = topkm_score / (topkm_score.sum(dim=1, keepdim=True) + 1e-20)
    topkm_weight = topkm_weight * routed_scaling_factor

    sorted_w, sort_order = topkm_weight.sort(dim=1, descending=True)
    sorted_idx = topkm_idx.gather(1, sort_order)

    # Step 3
    r = quality_floor * sorted_w[:, :K].sum(dim=1)        # [N]

    # Step 4
    popularity = torch.zeros(E, device=topkm_weight.device, dtype=topkm_weight.dtype)
    popularity.scatter_add_(0, topkm_idx.reshape(-1), topkm_weight.reshape(-1))

    _, pop_order = popularity.sort(descending=True)
    S_mask = torch.zeros(E, dtype=torch.bool, device=topkm_weight.device)
    S_mask[pop_order[:K_target]] = True

    token_ids = torch.arange(N, device=topkm_weight.device).unsqueeze(1).expand(N, K_ext)

    # Step 5-6
    for _ in range(max_add_rounds):
        in_S = S_mask[topkm_idx]                                   # [N, K_ext]
        c = (topkm_weight * in_S.float()).sum(dim=1)               # [N]
        d = (r - c).clamp_min(0.0)                                 # [N]

        satisfied = d <= 0
        sat_ratio = satisfied.float().mean()
        if sat_ratio >= q_major:
            break

        unsat = ~satisfied
        edge_mask = unsat.unsqueeze(1).expand_as(topkm_idx)

        edge_expert = topkm_idx[edge_mask]
        edge_weight = topkm_weight[edge_mask]
        edge_token = token_ids[edge_mask]

        keep = ~S_mask[edge_expert]
        edge_expert = edge_expert[keep]
        edge_weight = edge_weight[keep]
        edge_token = edge_token[keep]

        if edge_expert.numel() == 0:
            break

        G = torch.zeros(E, device=topkm_weight.device, dtype=topkm_weight.dtype)
        H = torch.zeros(E, device=topkm_weight.device, dtype=topkm_weight.dtype)

        gap_gain = torch.minimum(edge_weight, d[edge_token])
        hit_gain = (c[edge_token] + edge_weight >= r[edge_token]).to(topkm_weight.dtype)

        G.scatter_add_(0, edge_expert, gap_gain)
        H.scatter_add_(0, edge_expert, hit_gain)

        Score = alpha * H + beta * G
        Score[S_mask] = -1e30

        _, add_order = Score.sort(descending=True)
        new_experts = add_order[:batch_add_size]
        valid = Score[new_experts] > 0
        new_experts = new_experts[valid]
        if new_experts.numel() == 0:
            break

        S_mask[new_experts] = True

    # Step 7
    masked_logits = gate_logits.masked_fill(~S_mask.unsqueeze(0), float('-inf'))
    topk_weight_new, topk_idx_new = moe_mod.gate.routing(
        hs_flat, masked_logits, moe_mod.gate.top_k, True
    )

    # Step 8
    new_weights = final_top_p_trim(topk_weight_new)

    # Step 9
    routed_y = fused_experts(
        hidden_states=hs_flat,
        w1=moe_mod.experts.w13_weight,
        w2=moe_mod.experts.w2_weight,
        topk_weights=new_weights,
        topk_ids=topk_idx_new,
        inplace=False
    )

    routed_y = routed_y.view(bsz, seq_len, h)
    out = routed_y + shared_res
    return out
```

---

## 7. 关键实现决策总结

## 必须做
- 候选扩展到 `top-(K+M)`
- 所有分数统一用 `expanded weight`
- 阈值 `r_i` 只取 expanded 前 K 个的总和再乘 `quality_floor`
- 初始 `S` 用 expanded popularity
- batch add experts 也用 expanded weight

## 暂时不要动
- `gate.routing(...)`
- final top-p trimming
- `fused_experts(...)`

---

## 8. 为什么这版比之前讨论过的方案更好

这版的优点：

1. **比纯 top-k sparse 更强**
   - 保留了 top-k 外近邻 expert
   - 能表达“restricted reroute 中会上浮”的替代专家

2. **比 full dense 原算法便宜很多**
   - 不需要 dense softmax popularity
   - 不需要 dense safety loop
   - 不需要每轮在 `gate_w[:, S]` 上重排排序

3. **比“主集合 + 外环”更统一**
   - 不需要人为区分两类 expert 身份
   - 只区分：
     - 候选空间：`K+M`
     - 质量目标锚点：expanded 前 K 个

4. **比“候选和阈值用两套 weight”更自洽**
   - 所有核心量都在 expanded weight 空间里定义
   - 不会出现尺度不一致问题

---

## 9. 推荐的第一版超参数

建议先从下面这组开始：

- `K = 8`
- `M = 4`
- `K_target = 40`
- `quality_floor = 0.85`
- `q_major = 0.95`
- `batch_add_size = 8`
- `max_add_rounds = 2`
- `alpha = 1.0`
- `beta = 0.5`

如果你担心 expanded 前 K 总和使阈值略偏松，可以先观察再决定要不要加补偿系数，而不是一开始就改。

---

## 10. 重构时的代码替换建议

建议按下面顺序改代码：

1. **先把 `hook_forward()` 改成只拿一次 logits**
   - 从 `get_logits()` 出发
   - 自己构造 `top-(K+M)` expanded 候选

2. **先替换原 `compute_active_set()`**
   - 改成：
     - expanded popularity 初始化 `S`
     - coverage / gap
     - batch add experts

3. **保留 reroute + final top-p + fused_experts 不动**
   - 这样第一版最稳

4. **先验证以下指标**
   - 最终 `|S|` 分布
   - reroute 后非零 experts 数
   - token 满足率 `sat_ratio`
   - 端到端质量变化

---

## 11. 一句话总结

最终重构版算法可以概括为：

> **用 top-(K+M) 的 expanded weight 统一定义候选、coverage 和 expert 打分；用 expanded 前 K 个的总质量乘 `quality_floor` 定义每个 token 的质量阈值；用 batch add experts 构造 active set；然后沿用原 reroute、final top-p 和 fused_experts 执行。**

这就是当前最适合落地重构的一版。
