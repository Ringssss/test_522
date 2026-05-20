# dLLM KV Cache 原理与 Fusion 优化方向 Brainstorm

## 1. 这篇文档要回答什么

这篇文档记录了一次围绕 dLLM 推理 infra 优化的深度讨论，聚焦以下问题：

1. AR 模型的 KV cache 为什么能缓存？causal mask 如何保证历史 token 的 KV 跨层不变？
2. dLLM 打破了哪些条件？哪些 KV 是可以不变的，哪些必须重算？
3. 去噪循环的数学原理是什么？decode 步骤能否用连续近似替代以实现 fusion？
4. 如果把 K 步去噪做成纯 GPU 行为并利用分层存储加速，主要的挑战和可行路径是什么？

本文基于 `lib_cite/dInfer` 代码分析与第一性原理推导。

---

## 2. AR 模型 KV Cache 的第一性原理

### 2.1 核心命题

**在 causal mask + token 不可变的条件下，任意位置 i 在任意层 l 的 K_i^l 和 V_i^l 一旦计算出来就永远不变。**

### 2.2 逐层归纳证明

**第 0 层输入（Embedding 层）：**

```
h_i^{input} = Embed(token_i) + PosEmbed(i)
```

`token_i` 一旦生成就不变，`i` 也不变，所以 `h_i^{input}` 是常量。

**第 0 层 Self-Attention（causal mask）：**

```
Q_i = W_Q · h_i^{input}
K_j = W_K · h_j^{input}    (j = 0, 1, ..., i)    ← causal: 只看 j ≤ i
V_j = W_V · h_j^{input}    (j = 0, 1, ..., i)

Attn_i = softmax(Q_i · K_{0:i}^T / √d) · V_{0:i}
```

causal mask 保证 position i 只看 position 0..i。因此：

- 当后续生成 token_{i+1}, token_{i+2}, ... 时，位置 i 的 attention 计算完全不感知这些新 token。
- `Attn_i` 只依赖 `h_0^{input}, h_1^{input}, ..., h_i^{input}`。
- 这些全是常量 → `Attn_i` 是常量。

**第 0 层输出：**

```
h_i^{L0_out} = LayerNorm(h_i^{input} + Attn_i) → 经过 FFN → h_i^{L0_final}
```

全是常量的函数 → 常量。

**归纳：第 l 层**

假设第 l-1 层的输出 `h_i^{L(l-1)_final}` 对所有 j ≤ i 都是常量。那么第 l 层：

```
K_i^l = W_K^l · h_i^{L(l-1)_final}    ← 常量的线性变换 = 常量
V_i^l = W_V^l · h_i^{L(l-1)_final}    ← 常量

Attn_i^l = softmax(Q_i^l · K_{0:i}^{l,T} / √d) · V_{0:i}^l
```

causal mask 保证 attention 只看 0..i，全是常量 → `Attn_i^l` 是常量 → `h_i^{Ll_final}` 是常量。

**QED：对任意层 l 的任意位置 i，`K_i^l` 和 `V_i^l` 一旦算出就永不改变。**

### 2.3 KV Cache 成立的两个充分条件

| 条件 | 含义 | AR 是否满足 |
|------|------|------------|
| 条件 1：token 不可变 | 生成后不改写 | 是 |
| 条件 2：注意力不看未来 | causal mask 保证 position i 的计算只依赖 0..i | 是 |

这两个条件一起，让"过去"与"未来"完全解耦。这是 KV cache 成立的根基。

---

## 3. dLLM 打破了什么

dLLM 同时破坏了上述两个条件。

### 3.1 破坏条件 1：token 会被改写

在 diffusion 去噪中，同一位置的 token 会从 MASK 被替换为真实 token：

```
Step 0:  [prompt] [MASK] [MASK] [MASK] [MASK] [MASK]
Step 1:  [prompt] [MASK]  the   [MASK] [MASK] [MASK]
Step 2:  [prompt]  is     the   [MASK] [MASK]  dog
Step 3:  [prompt]  is     the   quick  brown   dog
```

位置 1 从 MASK → is，`h_1^{input}` 变了 → 所有层的 `K_1^l, V_1^l` 全变了。

### 3.2 破坏条件 2：bidirectional attention

dLLM 用 full attention（或 block-constrained full attention），位置 i 能看到所有位置：

```
Attn_i^l = softmax(Q_i^l · K_{0:N}^{l,T} / √d) · V_{0:N}^l
                                ^^^^
                              注意是 0:N，不是 0:i
```

即使位置 i 自身的 token 没有改变，只要序列中任何其他位置 j 的 token 变了：

```
token_j 变了
→ h_j^{input} 变了
→ K_j^{L0}, V_j^{L0} 变了
→ Attn_i^{L0} 变了（因为 i 能看到 j）  ← AR 模型中如果 j > i，causal mask 会挡住
→ h_i^{L0_final} 变了
→ K_i^{L1}, V_i^{L1} 变了
→ ... 向上层层传播
```

### 3.3 结论

**在 full attention 下，改变任何一个 token，所有位置在所有层的 KV 理论上全部失效。**

这就是 dLLM KV cache 困难的根本原因。

---

## 4. dLLM 中哪些 KV 可以不变——三层稳定性光谱

虽然理论上全都失效，但工程上存在不同层次的"稳定域"。

### 4.1 稳定域 1：Block-Causal Mask 下已完成 block 的 KV（精确不变）

dInfer 的 BlockDiffusionLLM 用的是 block-level causal mask：

```
Block 0:  [可看 Block0]  [看不到 Block1]  [看不到 Block2]
Block 1:  [可看 Block0]  [可看 Block1]    [看不到 Block2]
Block 2:  [可看 Block0]  [可看 Block1]    [可看 Block2]
```

Block 0 内部是 full attention，但 Block 0 看不到 Block 1, 2, ...。

因此：当 Block 0 解完后，其 token 全部确定且永不改写；Block 1 解码时改写的 token 不会影响 Block 0 的 attention → **Block 0 的 KV 在所有层上是精确不变的，可以安全缓存**。

代码位置：`generate_uniform.py:937`

```python
block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.model.device))
bd_attn_mask = block_mask.repeat_interleave(block_length, dim=0).repeat_interleave(block_length, dim=1)
```

这就是 `BlockDiffusionPrefixCacheManager`（`utils.py:499`）能成立的原因。它不是近似，而是在 block-causal mask 下精确成立。

### 4.2 稳定域 2：当前 block 内"已解码 token"的 KV（不稳定）

假设 Block 1 内部正在去噪：

```
Step 1:  [MASK]  cat   [MASK] [MASK]
Step 2:   the    cat   [MASK] [MASK]
```

在 Step 2 中，位置 cat 的 token 没变，但 block 内是 full attention。位置 the 从 MASK 变成了 the，所以：

```
h_{cat}^{input} 没变
但 Attn_{cat}^{L0} 变了（因为 K_{the}^{L0} 变了，且 cat 能看到 the）
→ h_{cat}^{L0_final} 变了
→ K_{cat}^{L1}, V_{cat}^{L1} 变了
```

**即使一个 token 自身没有改变，只要同 block 内其他 token 改变了，它的 KV 就失效了。**

### 4.3 稳定域 3：Vicinity 近似——"远处的变化影响很小"

虽然理论上同 block 内任何变化都传播到全 block，但实际上 attention 有注意力衰减——距离远的位置权重通常较低。

这就是 `VicinityCacheIteration`（`generate_uniform.py:671`）的理论基础：只刷新当前 block 左右 `prefix_look` / `after_look` 窗口的 KV，远处的 KV 用旧值近似。

这不是精确的，是一个可控的近似。误差会累积，所以需要周期性全量刷新来校正。

### 4.4 总结

```
精确不变 ◄────────────────────────────────────────────► 每步都变

已完成 block 的 KV          远距离位置的 KV              当前 block 内
(block-causal mask          (vicinity 近似,             被改写位置的
 精确保证不变)               可控误差)                   邻近 KV
                                                       (每步必须重算)
```

---

## 5. 去噪循环的数学原理

### 5.1 正向过程（加噪）

LLaDA 类 dLLM 用离散 absorbing diffusion。正向过程：

```
q(x_t | x_0) = (1-t)·δ(x_t = x_0) + t·δ(x_t = [MASK])
```

在时间 t，每个 token 有概率 t 被替换成 [MASK]，概率 (1-t) 保持原值。

### 5.2 逆向过程（去噪）

模型 `p_θ(x_0 | x_t)` 的目标：给定当前的部分 mask 序列 x_t，预测每个位置的原始 token。

一步去噪对每个位置 i：

```
如果 x_t[i] ≠ MASK:  x_{t-Δt}[i] = x_t[i]        // 已确定，不动
如果 x_t[i] = MASK:
    x̂₀[i] = argmax_v p_θ(v | x_t)                  // 候选 token
    c[i]   = max_v p_θ(v | x_t)                     // 置信度
    如果 c[i] ≥ τ:  x_{t-Δt}[i] = x̂₀[i]           // 接受
    否则:           x_{t-Δt}[i] = MASK               // 保留 mask
```

### 5.3 张量形式表达

设 x_t 为当前 token 序列 [B, L]，E 为 embedding matrix [V, D]，M 为 mask 标记 `M[i] = (x_t[i] == MASK_ID)`。

一步去噪的精确张量形式：

```
logits    = Transformer(Embed(x_t))              # [B, L, V]
p         = softmax(logits, dim=-1)              # [B, L, V]
x̂₀        = argmax(p, dim=-1)                    # [B, L]     ← 离散！
c         = gather(p, dim=-1, index=x̂₀)          # [B, L]
accept    = M ∧ (c ≥ τ)                          # [B, L]     ← 离散！
x_{t-Δt}  = where(accept, x̂₀, x_t)              # [B, L]     ← 离散跳变！
```

三处离散操作（argmax、threshold comparison、where）让这一步无法被 fuse 进 forward。

### 5.4 dInfer 中的具体实现

核心采样逻辑在 `parallel_strategy.py:353-385` 的 `get_transfer_index_threshold`：

```python
@torch.compile(dynamic=True)
def get_transfer_index_threshold(logits, temperature, mask_index, x, mask_id, threshold, ...):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)              # argmax 取候选 token
    p = F.softmax(logits.to(torch.float32), dim=-1)           # 计算概率
    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1))    # 取候选 token 的置信度

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -inf)

    # 动态阈值：取 max(confidence) - ε 和 threshold 的较小值
    actual_threshold = torch.max(confidence, dim=1)[0] - 1e-5
    actual_threshold = actual_threshold.clamp(-1000, threshold).unsqueeze(-1)
    transfer_index = confidence >= actual_threshold             # 高于阈值的全部接受
    return x0, transfer_index
```

decode 的回写在 `ThresholdParallelDecoder.decode()`（`parallel_strategy.py:404-425`）：

```python
x[:, block_start:block_end] = torch.where(transfer_index, x0, curr_x)
broadcast_if_needed(x.data)
```

---

## 6. 连续近似：消除离散跳变以实现 Fusion

### 6.1 核心思路

**如果把 decode 的输出从离散 token id 变成连续 embedding，就可以消除离散跳变，让两步 forward 之间变成连续映射。**

### 6.2 三个近似替换

#### 近似 1：Soft Argmax → Top-K 加权 Embedding

离散版：

```
x̂₀ = argmax(logits)           # 离散 token id
h_next = E[x̂₀]                 # embedding lookup，不可微
```

连续近似（Top-K 稀疏版，避免 O(B·L·V·D) 的全 vocab matmul）：

```
topk_probs, topk_ids = torch.topk(softmax(logits), k=K, dim=-1)    # [B, L, K]
topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)     # 重归一化
topk_embeds = E[topk_ids]                                           # [B, L, K, D]
h_soft = (topk_probs.unsqueeze(-1) * topk_embeds).sum(dim=-2)      # [B, L, D]
```

K=4 时成本只有全 vocab 的 4/128K ≈ 0.003%。在高置信度下（p_max > 0.9），top-1 就占绝大部分权重。

#### 近似 2：Soft Threshold → Sigmoid

离散版：

```
accept = confidence >= threshold       # 硬切：0 或 1
```

连续近似：

```
accept_soft = σ((confidence - threshold) / β)     # β 控制锐度
```

β → 0 时退化为硬阈值。β = 0.05 左右已经非常接近硬阈值。

#### 近似 3：Soft Update → 加权混合

离散版：

```
h_next = where(accept, E[x̂₀], E[x_t])
```

连续近似：

```
h_next = accept_soft ⊙ h_soft + (1 - accept_soft) ⊙ h_current
```

### 6.3 组合：连续版一步去噪

```python
def soft_denoise_step(h_current, x_current, model, E, threshold, temperature, beta, mask_id):
    # Forward: 直接用 embedding 而不是 token id
    logits = model.forward_from_embedding(h_current)        # [B, L, V]

    # Soft argmax (top-K sparse)
    p = softmax(logits / temperature, dim=-1)
    topk_p, topk_ids = torch.topk(p, k=4, dim=-1)
    topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)
    h_predicted = (topk_p.unsqueeze(-1) * E[topk_ids]).sum(-2)  # [B, L, D]

    # Confidence
    confidence = topk_p[:, :, 0]                                 # top-1 概率

    # Soft threshold
    mask = (x_current == mask_id).float()
    accept = mask * torch.sigmoid((confidence - threshold) / beta)

    # Soft update
    h_next = accept.unsqueeze(-1) * h_predicted + (1 - accept.unsqueeze(-1)) * h_current

    # Hard token for mask tracking (只用于判断收敛)
    x_hard = torch.where(accept > 0.5, topk_ids[:, :, 0], x_current)

    return h_next, x_hard
```

全部操作都是 matmul / elementwise / topk，没有任何离散跳变。

### 6.4 K 步展开为可编译计算图

```python
@torch.compile(dynamic=False)
def fused_K_steps(h_init, x_init, model, E, threshold, temperature, beta, mask_id, K):
    h, x = h_init, x_init
    for step in range(K):
        logits = model.forward_from_embedding(h)
        p = F.softmax(logits / temperature, dim=-1)
        topk_p, topk_ids = torch.topk(p, k=4, dim=-1)
        topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True)
        h_pred = (topk_p.unsqueeze(-1) * E[topk_ids]).sum(-2)
        conf = topk_p[:, :, 0]
        mask = (x == mask_id).float()
        accept = mask * torch.sigmoid((conf - threshold) / beta)
        h = accept.unsqueeze(-1) * h_pred + (1 - accept.unsqueeze(-1)) * h
        x = torch.where(accept > 0.5, topk_ids[:, :, 0], x)
    return h, x
```

K 步执行期间零 host 交互，中间不需要 GPU→CPU sync 检查收敛。

### 6.5 与 dInfer 已有 IterationSmooth 的关系

dInfer 的 IterationSmooth（`generate_uniform.py:558-621`）已做了类似的事情：

```python
mask_index = (x.data == decoder.mask_id)
self.inputs_embeds = self.h2e(x.data, mask_index, output.logits, iter_cont_weight)
```

`h2e` 在做 soft embedding 混合。但它仍然是先做 hard decode（`decoder.decode` 改写 x），再用 soft embedding 做下一步的输入。**hard decode 和 soft embedding 是串行的两步，没有真正融合。**

上面的 soft denoise step 把这两步合成一步：hard decode 变成 soft decode 的一个副产品，而不是一个前置依赖。

### 6.6 精度与 Trade-off

1. **精度**：soft 近似在高置信度下几乎无损（top-1 prob > 0.9 时混合 embedding ≈ 纯 top-1 embedding），低置信度下会引入 embedding 层面的模糊。需要实验验证最终生成质量。
2. **终止条件**：仍需判断 block 解完。可以用 soft 判断：当所有 masked 位置的 accept_soft > 0.99 时视为收敛，放在 K 步后统一做一次。
3. **与 block-causal mask 兼容性**：soft embedding 不改变 attention mask 结构，block-causal 的 KV cache 精确缓存依然成立。

---

## 7. 纯 GPU 执行 + 分层存储加速：挑战分析

### 7.1 GPU 分层存储概览

```
                容量        带宽           延迟
Register     ~256KB/SM     ~无限          0 cycles
SRAM(共享)   ~192KB/SM     ~19 TB/s      ~5 cycles      ← FlashAttention 的核心利用层
L2 Cache     ~40MB(全局)   ~5 TB/s       ~50 cycles
HBM          ~80GB         ~2-3 TB/s     ~300 cycles    ← 所有大 tensor 住在这里
```

FlashAttention 的核心：不把 [L, L] 的 attention matrix 写回 HBM，而是在 SRAM 里分 tile 计算、用 online softmax 保持正确性，最后只把 [L, D] 的 output 写回 HBM。省了 O(L²) 的 HBM 读写。

### 7.2 挑战 1：循环体内是完整的 Transformer Forward

K 步 loop body 不是一个简单算子，而是：

```python
for step in range(K):
    for layer in range(N_layers):          # 32-80 层
        Q, K, V = W_qkv @ h               # matmul: [B,L,D] @ [D,3D]
        attn_out = FlashAttn(Q, K, V, KV_cache)
        h = h + attn_out
        h = h + FFN(h)                     # matmul: [B,L,D]→[B,L,4D]→[B,L,D]
    logits = W_lm_head @ h                 # matmul: [B,L,D] @ [D,V]
    # soft decode ...
```

FlashAttention 之所以能做，是因为 fuse 的是**单个算子**，有特定数学结构（可分块、可 online softmax）。

而这里要 fuse 的是 `N_layers × (3 matmul + attention + FFN) + LM head + soft decode`。这些操作的数据流完全不同，不能跨操作做 tile。

**h 的大小：**

```
h: [1, 32, 4096] = 256 KB (bf16)
一个 SM 的 SRAM: 192 KB

→ h 本身就放不进一个 SM 的 SRAM
```

跨层保持 h 在 SRAM 不现实。如果 tile over D（如 D=4096 分成 32 个 128 tile），QKV projection 变成 partial sum 需要 reduce，复杂度远超 FlashAttention。

### 7.3 挑战 2：Model Weights 是带宽的绝对大头

每步 forward 需从 HBM 读取所有模型权重：

```
每层权重: W_q + W_k + W_v + W_o + W_up + W_gate + W_down
        ≈ 4D² + 12D² = 16D²
        = 16 × 4096² × 2 bytes = 512 MB/layer (bf16)

32 层 = 16 GB

K=4 步 → 64 GB 权重读取

HBM 带宽 ~2.5 TB/s → 64 GB / 2.5 TB/s ≈ 25 ms（纯权重读取的下界）
```

**不管 soft decode 怎么 fuse，权重读取的 25ms 都逃不掉。这是绝对下界。**

FlashAttention 不涉及大量模型权重（只涉及 Q、K、V 这些中间 activation），所以不受此限制。

### 7.4 挑战 3：LM Head + Softmax + TopK——最值得 Flash 化的部分

当前流程：

```
logits = h @ W_lm_head^T     # [B,L,D] @ [D,V] = [B,L,V]  → 写 HBM: B×L×V×2 bytes
p = softmax(logits)           # 读 HBM + 写 HBM: B×L×V×2 bytes
topk_p, topk_ids = topk(p)   # 读 HBM: B×L×V×2 bytes
```

V=128K, L=32, B=1 时：

```
logits 大小: 32 × 128K × 2 = 8 MB
三次 HBM 读写: 8 × 3 = 24 MB
K=4 步: 96 MB
```

可以用类似 FlashAttention 的 tiling 策略消除这个中间 tensor：

```
FlashSoftmax + TopK:
  对 V 维度分 tile，每个 tile 大小 V_tile (如 1024)
  for v_start in range(0, V, V_tile):
      logit_tile = h @ W_lm_head[:, v_start:v_start+V_tile]^T   # SRAM 内
      用 online softmax 更新 running max, running sum              # SRAM 内
      用 running top-K heap 维护全局 top-K                         # SRAM 内

  最终得到: topk_p [B, L, K], topk_ids [B, L, K]
  → 只写回 B×L×K×2 bytes ≈ 微不足道
```

**省掉每步 24 MB 的 HBM 读写。**

### 7.5 挑战 4：Soft Decode 的 Embedding Lookup

```python
h_pred = (topk_p.unsqueeze(-1) * E[topk_ids]).sum(-2)    # [B, L, D]
```

`E[topk_ids]` 是 gather 操作：从 [V, D] 的 embedding matrix 取 K 行。这是**随机访问模式**，对 HBM 不友好。

但在 LLaDA 类模型中，LM head 权重和 embedding matrix 做了 weight tying（`W_lm_head == E`），所以可以在上面的 FlashSoftmax kernel 内部：当更新 top-K 时，同时把对应 embedding 行取出做加权累加。`E[topk_ids]` 的 gather 变成 `E[:, v_start:v_start+V_tile]` 的顺序读取，完美 coalesce。

### 7.6 挑战 5：KV Cache 的读取模式

K 步 loop 中，每步 attention 都要读取 prefix KV cache：

```
prefix_kv: [N_layers, 2, B, H, prefix_len, head_dim]
假设 prefix_len=512, 32 层: = 512 MB
K=4 步读 4 次: 2 GB 读取
```

在 block-causal mask 下，prefix KV 是 read-only 的。但放不进 L2（512 MB >> 40 MB）。

逐层看：每层的 prefix_kv = 512 MB / 32 = 16 MB，接近 L2 容量。在 K 步同层复用间可能有部分 L2 命中，但不可靠。

当前 block 的 KV per head：[2, 1, 1, 32, 128] = 16 KB → 可以放进 SRAM。FlashAttention 已经在这么做了。

**结论：prefix KV 的重复读取不可避免，但 FlashAttention 已以最优方式处理。**

### 7.7 挑战 6：torch.compile 的能力边界

```
能做:
  - fuse 连续的 element-wise ops（softmax、sigmoid、where、乘加）
  - 消除中间 tensor 的 HBM 写回（kernel fusion）
  - 把 K 步展开成静态图

做不到:
  - 不会重新实现 matmul（调用 cuBLAS）
  - 不会做跨 matmul 的 tiling（不做 FlashAttention 级别优化）
  - 不会把整个 transformer 编译成一个 kernel
```

所以 torch.compile 能帮 fuse 的是：LM_head 之后的 softmax + topk + embedding_mix + sigmoid + where → 一个 kernel。但 Transformer 各层 matmul 之间的 fusion 做不到。

---

## 8. 可行性总结与优先级

### 8.1 可行性分级

```
高可行，确定收益:
  1. Flash LM-Head Kernel
     把 LM_head + softmax + topK + embedding_gather fuse 成一个 kernel
     消除 [B,L,V] 中间 tensor
     每步省 ~24 MB HBM 读写, K 步省 ~100 MB

  2. torch.compile K 步 soft decode
     把 K 步的 element-wise 操作编译成连续图
     消除 host 交互和 Python overhead

  3. CUDA Graph capture K 步
     把 K 步 (forward + soft_decode) 作为 graph replay
     消除 kernel launch overhead

有价值但需要定制 kernel:
  4. Flash Soft-Decode Kernel
     把 LM_head 尾部与 soft decode 全流程写成 Triton kernel
     类似 FlashAttention 的 V 维度 tiling

  5. 当前 block KV 的 persistent SRAM 缓存
     每步 forward 中当前 block 的 QKV 投影结果留在 SRAM
     per-head 16KB，放得下

不可行或收益极低:
  6. 跨 Transformer 层的 tiling（h 留在 SRAM 跨层）
     h = 256KB > SRAM 192KB，且 matmul 需完整 D 维度
     不可行

  7. 权重跨步复用
     权重读取是每步 ~16 GB 的固定开销
     无法避免，是绝对下界

  8. prefix KV cache 跨步 SRAM 缓存
     per-layer 16 MB >> SRAM
     放不下
```

### 8.2 最有价值的单点突破：Flash LM-Head

结合挑战 3 + 挑战 4 的融合机会。利用 weight tying（LM head 权重 == embedding matrix）。

Triton 伪码：

```python
@triton.jit
def flash_soft_decode(h_ptr, W_ptr, h_out_ptr, x_out_ptr,
                      threshold, beta, mask_id, x_ptr,
                      B, L, D, V, K_top):
    # 每个 program 处理一个 (b, l) 位置
    b, l = program_id(0), program_id(1)

    # 从 HBM 读 h[b,l,:] 到 SRAM
    h_local = load(h_ptr[b, l, :])     # [D] in SRAM
    x_local = load(x_ptr[b, l])         # scalar

    # Online softmax + topK over V, tiled
    running_max = -inf
    running_sum = 0
    topk_vals = [-inf] * K_top          # 在 register 里
    topk_ids  = [0] * K_top

    for v_start in range(0, V, V_TILE):
        # 从 HBM 流式读权重 tile（weight tying: W == E）
        W_tile = load(W_ptr[v_start:v_start+V_TILE, :])   # [V_TILE, D]

        # 计算 logit tile: h_local @ W_tile^T
        logit_tile = dot(h_local, W_tile)                   # [V_TILE] in SRAM

        # online softmax 更新
        tile_max = max(logit_tile)
        new_max = max(running_max, tile_max)
        running_sum = running_sum * exp(running_max - new_max) + sum(exp(logit_tile - new_max))
        running_max = new_max

        # 更新 topK（register 里的小堆）
        update_topk(topk_vals, topk_ids, logit_tile, v_start)

    # 归一化 topK probabilities
    for k in range(K_top):
        topk_vals[k] = exp(topk_vals[k] - running_max) / running_sum

    # 加权 embedding（复用 W，因为 weight tying）
    h_pred = zeros(D)
    for k in range(K_top):
        W_row = load(W_ptr[topk_ids[k], :])  # 从 HBM 读一行
        h_pred += topk_vals[k] * W_row

    # Soft threshold + update
    conf = topk_vals[0]
    is_mask = (x_local == mask_id)
    accept = is_mask * sigmoid((conf - threshold) / beta)

    h_new = accept * h_pred + (1 - accept) * h_local
    x_new = topk_ids[0] if accept > 0.5 else x_local

    # 写回 HBM
    store(h_out_ptr[b, l, :], h_new)
    store(x_out_ptr[b, l], x_new)
```

**核心优势：完全不写 [B,L,V] 回 HBM，且 W 的读取可以跨 (b,l) 位置通过 L2 cache 自然复用。**

---

## 9. 其他 Infra 方向（来自前序讨论）

### 9.1 经典数据结构思路

| 方向 | 思路 | 适用场景 |
|------|------|---------|
| Dirty Bitmap | 给每个 cache position 维护 dirty bit，只重算 dirty segment 的 KV | 减少无效 full refresh |
| Ring Buffer | 固定大小 KV buffer + offset，O(1) 滑动窗口 | vicinity cache 加速 |
| Copy-on-Write | 已完成 block 的 KV 做快照，按需 copy-write | 跨请求 KV 共享 |
| Block-Indexed Pool | dLLM 版 PagedAttention，block 粒度内存池，page table 间接寻址 | 内存利用率 |

### 9.2 循环优化思路

| 方向 | 思路 | 收益 |
|------|------|------|
| Fuse forward+decode | 消除 host 往返 | 每步省 ~0.15ms |
| Fixed-K graph replay | CUDA graph capture K 步连续执行 | 消除 launch overhead |
| Adaptive K + async check | 异步终止检查，隐藏 sync latency | 进一步减少 bubble |
| Speculative overlap | decode_N 与 forward_{N+1} 重叠执行 | 后期成功率高 |

### 9.3 decode 策略优化

| 方向 | 思路 | 收益 |
|------|------|------|
| 动态阈值调度 | 早期低阈值激进解码、后期高阈值保守 | 减少 step 数 |
| confidence + locality 联合 | 不只看全局阈值，还看局部结构 | 提高解码均衡性 |
| M2T + T2T 混合 | 先 unmask 再有限编辑修正 | 兼顾速度和质量 |

---

## 10. 总结

dLLM 推理优化是一个 **step reduction × per-step cost reduction** 的乘积优化问题。

在 per-step cost 层面，最有杠杆的 infra 点是：

1. **Flash LM-Head**：消除 [B,L,V] 中间 tensor 的 HBM 读写，是当前唯一可以用类 FlashAttention tiling 思路做出显著收益的算子级优化。
2. **Soft decode + torch.compile/CUDA graph**：消除离散跳变和 host 交互，让 K 步成为连续的 GPU 执行。
3. **Dirty-bit driven selective cache refresh**：把 cache 刷新从 block 粒度推到 position 粒度。

模型权重的 HBM 读取（每步 ~16 GB）是不可突破的硬限，因此减少 step 数（decode 策略优化）往往比减少 per-step cost 更有杠杆。最优方案是两者联动。
