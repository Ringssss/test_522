# dLLM MoE 推理行为分析与优化探索：完整实验报告

> 日期：2026-04-12
> 阶段：v0.1-init-project（v0.1.13）
> 模型：LLaDA2.0-mini
> 硬件：NVIDIA H100 80GB HBM3
> 目标读者：AI infra / 系统方向研究者，用于学术讨论

---

## 1. 研究背景

### 1.1 什么是 Diffusion LLM (dLLM)

Diffusion LLM 是面向离散 token 序列的迭代去噪生成模型。与 AR（自回归）模型逐 token 生成不同，dLLM：

- 先将待生成区域初始化为 MASK token
- 反复执行 transformer forward（通常 ~12 次/block）
- 每轮根据置信度选择一部分 token 从 MASK → 真实 token
- 直到整个 block 完全去噪

**Semi-autoregressive 变体（Block Diffusion）**：将生成区切成 block（如 32 tokens），block 间顺序处理，block 内并行去噪。这是当前最实用的 dLLM serving 形态。

### 1.2 实验模型配置

```
模型: LLaDA2.0-mini
架构: DeepSeek-style MoE Transformer
hidden_size:          2048
num_hidden_layers:    20 (Layer 0 dense FFN, Layer 1-19 MoE)
num_attention_heads:  16
num_key_value_heads:  4
num_experts:          256
num_experts_per_tok:  8 (top_k=8)
n_group:              8 (grouped routing)
topk_group:           4
num_shared_experts:   1
moe_intermediate_size: 512
routed_scaling_factor: 2.5
vocab_size:           156928
mask_id:              156895
```

### 1.3 推理框架

使用 dInfer 框架 + vllm 后端：
- `BlockDiffusionLLMAttnmask`（BD-attnmask）：block-causal attention mask，无 KV cache，batch=1 baseline
- `BlockDiffusionLLM`：KV cache prefix，batch>1 支持
- MoE kernel：vllm `fused_moe_kernel`（Triton 实现）
- 解码策略：`ThresholdParallelDecoder`，threshold=0.90

### 1.4 Baseline 性能

| 路径 | batch=1 tok/s | fwd/s | forward 次数 |
|------|-------------|-------|-------------|
| BD-attnmask (no-cache) | 75-77 | 26-27 | 48 |
| BD+IterSmooth (short prompt) | 87.3 | 26.8 | 42 |
| BD+cache (long prompt, batch=32) | 1672 | 21.3 | 58 |

---

## 2. 三个核心 Insight：dLLM MoE 的独特行为

### 2.1 Insight A：MASK token 的 expert routing 集中现象

#### 2.1.1 观察

dLLM 每步输入的 block（32 tokens）中包含已解码 token 和 MASK token。MASK tokens 全部共享同一个 embedding (id=156895)。经过 attention 后获得上下文，但起点相同。进入 MoE gate 时，MASK tokens 的 hidden state 高度相似（尤其在浅层），导致 routing 集中到少数 experts。

**AR 中不存在这个现象**——所有 token 都是语义丰富的，routing 近似均匀。

#### 2.1.2 实验数据（异质 batch=8, temperature=0.7）

**Routing Entropy（per-layer, 平均值）**：

| Layer | MASK entropy | Decoded entropy | Ratio |
|-------|-------------|----------------|-------|
| 0 (浅) | 4.74 | 7.35 | **0.645** |
| 1 | 5.00 | 7.35 | 0.681 |
| 5 | 5.49 | 6.49 | 0.846 |
| 9 | 5.68 | 6.42 | 0.884 |
| 14 | 5.66 | 6.60 | 0.857 |
| 18 (深) | 5.69 | 6.57 | 0.866 |

**Active expert count（per-layer, 平均值）**：

| Layer | MASK active experts | Decoded active experts |
|-------|-------------------|----------------------|
| 0 (浅) | **64** / 256 | 215 / 256 |
| 1 | **76** / 256 | 214 / 256 |
| 5 | 81 / 256 | 147 / 256 |
| 9 | 92 / 256 | 152 / 256 |
| 18 (深) | 89 / 256 | 155 / 256 |

**Expert 负载不均衡（Gini 系数，batch=8, 首次迭代）**：

| Layer | MASK Gini | Decoded Gini |
|-------|-----------|-------------|
| 0 | **0.937** | 0.656 |
| 5 | **0.913** | 0.689 |
| 9 | 0.886 | 0.700 |
| 14 | 0.903 | 0.717 |
| 18 | **0.891** | 0.686 |

**关键结论**：浅层 MASK routing Gini > 0.93，仅激活 64-76/256 experts，routing entropy 仅为 decoded 的 65%。

#### 2.1.3 跨设置稳健性

| 设置 | MASK/Dec entropy ratio | 结论 |
|------|----------------------|------|
| batch=1, temp=0, 同质 | 0.70 | 成立 |
| batch=8, temp=0.7, 异质 | **0.65** | 成立，更极端 |
| batch=32, temp=0.7, 异质 | 0.66 | 成立 |

### 2.2 Insight B：跨迭代的 MoE 计算冗余

#### 2.2.1 观察

一个 block 内需要 ~12 次迭代才能完成解码。每次迭代，所有 32 个位置都过 MoE（19 层 × 1 次/层）。但已解码位置的 token 没有改变，其 MoE 输入、routing、expert 输出大概率与上一步几乎一致。

**AR 中不存在这个现象**——每步只处理 1 个全新 token。

#### 2.2.2 Routing 稳定性（已解码位置，跨迭代 change rate %）

数据来自异质 batch=8, temperature=0.7：

| Layer | Decoded routing change % | MASK routing change % |
|-------|--------------------------|-----------------------|
| 0 | **12.2%** | 60.4% |
| 4 | **27.1%** | 84.1% |
| 9 | **26.2%** | 89.0% |
| 14 | **25.9%** | 87.1% |
| 18 | **25.7%** | 72.1% |
| 平均 | **~24%** | ~82% |

**已解码位置的 routing 在 76% 的情况下保持不变。**

#### 2.2.3 MoE Output 相似度（已解码位置，跨迭代 cosine similarity）

| Layer | Decoded cos sim | MASK cos sim |
|-------|----------------|-------------|
| 0 | **0.995** | 0.968 |
| 4 | **0.986** | 0.904 |
| 9 | **0.981** | 0.780 |
| 14 | **0.975** | 0.833 |
| 18 | **0.991** | 0.942 |
| 平均 | **0.98** | 0.85 |

**已解码位置的 MoE output cosine similarity 高达 0.97-0.99。**

#### 2.2.4 总冗余率

| 设置 | 总冗余率 |
|------|---------|
| batch=1, 同质 | 30% |
| batch=8, 异质 | **49%** |
| batch=32, 异质 | **52%** |

冗余率定义：跨迭代中，已解码位置且 routing 未变化的 MoE 计算占总 MoE 计算的比例。

**迭代级冗余率演化（batch=8, 异质）**：

| 迭代 | 平均 MASK 数 | 冗余率 |
|------|------------|--------|
| 0 | 32.0 | 0% |
| 1 | 29.8 | 11.2% |
| 3 | 26.3 | 17.1% |
| 6 | 21.9 | 29.8% |
| 9 | 17.5 | 36.2% |
| 12 | 11.6 | 53.2% |
| 15 | 7.6 | 58.7% |

冗余率随迭代单调递增——因为越到后期已解码位置越多。

### 2.3 Insight C：dLLM 的天然批量效应

#### 2.3.1 观察

| 模型 | batch=1 时 MoE 处理的 token 数 |
|------|------------------------------|
| AR decode | 1 token/step |
| dLLM decode | **32 tokens/step**（block_size=32）|

dLLM 在 batch=1 时的 MoE GPU 利用率 ≈ AR 在 batch=32 时。

#### 2.3.2 影响

- dLLM 天然更能喂饱 MoE kernel
- 但在大 batch 下更容易进入 compute-bound
- batch=32 时 dLLM 的 MoE 处理 1024 tokens/step（32 × 32），all-to-all 通信量是 AR 的 32×

### 2.4 多卡 EP 视角下的影响放大

| Insight | 单卡影响 | 多卡 EP 影响 |
|---------|---------|------------|
| A (routing集中) | 少数 expert 过载，多数空闲 | **跨 GPU 负载不均 + all-to-all 流量不均** |
| B (跨迭代冗余) | ~50% 冗余计算 | **省 ~50% compute + 省 ~50% all-to-all 通信** |
| C (天然批量) | GPU 利用率高 | **all-to-all volume = AR 的 block_size 倍** |

---

## 3. MoE 内部结构分析

### 3.1 Shared vs Routed Expert 分解

每个 MoE block = shared_experts(all tokens) + fused_moe(routed, top_k=8)。

**Routed/Shared 量级比（per-layer）**：

| Layer | Shared norm | Routed norm | Ratio |
|-------|------------|------------|-------|
| 0 | 0.76 | 0.88 | 1.15 |
| 5 | 8.68 | 6.58 | **0.76** |
| 10 | 17.4 | 29.3 | **1.68** |
| 14 | 89.7 | 83.2 | 0.93 |
| 17 | 301.7 | 97.3 | **0.32** |
| 18 | 613.4 | 199.2 | **0.32** |

**关键结论**：routed 贡献在大多数层不可忽略（ratio 0.32-1.68），不能简单用 shared-only 近似。

### 3.2 近似质量

| 近似方式 | Cosine sim 范围 | 可行性 |
|---------|----------------|--------|
| Shared-only（忽略 routed） | 0.57-0.95 | **不可行** |
| v1 cache（full output，1-step-old） | 0.87-0.98 | 更好但仍有误差 |

### 3.3 逐层 Ablation

**单层缓存替换（每步全量计算，仅替换一层输出）**：

| 层 | 结果 | Forward 数 | Match % |
|---|------|-----------|---------|
| L1-L17 | exact match | 48-52 | **100%** |
| **L18** | **失败** | 50 | **19%** |
| L19 | exact match | 47 | **100%** |

**18/19 个 MoE 层在单独替换时安全，唯一敏感层是 Layer 18。**

**周期性刷新**：

| 刷新频率 N | 结果 | Cache 使用率 |
|-----------|------|------------|
| N=0 (永不刷新) | 失败 (63% match) | 100% |
| N=2 | exact match | 50% |
| N=3 | exact match | 67% |
| N=5 | exact match | 80% |
| N=10 | exact match | 90% |

**注意**：这些"安全"结果的前提是每步都做了全量计算——cache 始终是 1-step-old 的（新鲜的）。这和"跳过计算"完全不同。

---

## 4. Stable Cache 优化尝试与系统性失败

### 4.1 方案演进

| 版本 | 策略 | 结果 |
|------|------|------|
| v1 | 缓存 stable 位置的 full MoE output，跳过计算 | 失败 |
| v2a | 只缓存 routed 部分，shared 始终新鲜 | 失败 |
| v2b | 加周期性全量刷新 (N=2) | 失败 |
| v2c | 排除敏感层 L18 | 失败 |

### 4.2 多层组合 Ablation——致命发现

| 组合 | 单独安全？ | 组合结果 | Match % |
|------|-----------|---------|---------|
| L1-3 | 每层 ✓ | ✓ | 100% |
| L6-13 | 每层 ✓ | ✓ | 100% |
| L1-5 | 每层 ✓ | **✗** | 52% |
| L1-3 + L19 | 每层 ✓ | **✗** | — |
| L6-13 + L19 | 每层 ✓ | **✗** | — |
| L1-3 + L6-13 | 每层 ✓ | **✗** | — |
| 隔层(L1,3,5,...) | 每层 ✓ | **✗** | 20% |
| 全部(除L18) | 每层 ✓ | **✗** | 28% |

**结论：单层 safe ≠ 多层组合 safe。** 误差通过 attention 在层间传播，19 层串联后超过容错边界。

### 4.3 根本原因链

```
1. 跳过 stable 位置的 MoE 计算
   → cache 无法被 fresh output 更新
2. cache 越来越旧（N-step-old，N 随迭代增长）
3. 旧 cache 作为下一层 attention 输入
   → 下一层 hidden state 偏移
4. 偏移在 19 层串联中累积
   → logits 大幅偏移
   → decoder 行为改变
   → forward 次数增加
```

### 4.4 根本矛盾

> **跳过计算 ↔ 保持 cache fresh，二者不可兼得。**
>
> 要保持 cache fresh 必须做全量计算（消除误差），但那就不省计算了。
>
> Ablation 实验中的"安全"是建立在"每步都做全量计算后再替换输出"的前提上，和真正"跳过计算"有本质区别。

---

## 5. Padding-Free MoE Kernel 优化尝试

### 5.1 背景：当前 MoE kernel 的 padding 浪费

vllm 的 `fused_moe_kernel` 使用 `moe_align_block_size` 将每个 expert 的 token 数对齐到 `BLOCK_SIZE_M`（H100 autotuned 为 16）。

**Padding 量化**：

| batch | tokens | 有效 pairs | padding 后 | padding % |
|-------|--------|-----------|-----------|----------|
| 1 | 32 | 256 | 2,496 | **90%** |
| 4 | 128 | 1,024 | 3,472 | 70% |
| 8 | 256 | 2,048 | 4,864 | 58% |
| 16 | 512 | 4,096 | 10,112 | 60% |
| 32 | 1,024 | 8,192 | 20,992 | 61% |

### 5.2 Perfect-packing 理论上界实验

通过构造完美填充的 routing（每个 expert 恰好填满 BLOCK_SIZE_M）模拟 padding-free 上界：

| batch | uniform routing | perfect-packed | 加速比 | 19 层可省 |
|-------|----------------|---------------|--------|----------|
| 1 | 0.537ms | 0.335ms | **1.60x** | 3.84ms |
| 4 | 0.706ms | 0.345ms | **2.05x** | 6.86ms |
| 8 | 0.740ms | 0.481ms | **1.54x** | 4.92ms |
| 32 | 0.796ms | 0.529ms | **1.51x** | 5.08ms |

### 5.3 Routing 集中度对 kernel 性能的影响

| Routing 模式 | Active experts | Kernel time (batch=1) | vs uniform |
|-------------|---------------|----------------------|-----------|
| Uniform (168 experts) | 168 | 0.548ms | baseline |
| MASK-concentrated (111 experts) | 111 | 0.434ms | **-20.8%** |
| Extreme (10 experts) | 10 | 0.339ms | **-38.1%** |

**MASK routing 集中让 kernel 天然更快**（fewer active experts → fewer padding blocks → faster）。

### 5.4 Forward 时间拆分

| 组件 | 时间 (no-cache, batch=1, ~192 tokens) | 占比 |
|------|--------------------------------------|------|
| **Routed MoE (fused_experts)** | **13.9ms** | **37%** |
| Shared expert (dense FFN) | 0.8ms | 2% |
| Gate (router) | 0.3ms | 1% |
| Attention + Norm + Embedding + LM head | 22.0ms | 60% |
| **Total forward** | **~37ms** | 100% |

### 5.5 实现与结果

基于 X-MoE (SC 2025) 的 PFT 方案实现了完整的 padding-free MoE pipeline：
- PFT dispatch（sort + histogram，无 padding）
- Triton gather/scatter kernels
- Persistent grouped GEMM kernel（grid=NUM_SM=132）
- BF16 支持

**正确性**：cosine sim = 0.999994，**全部 PASS**

**性能**：

| batch | baseline (vllm) | padding-free | speedup |
|-------|----------------|-------------|---------|
| 1 | 0.53ms | 0.84ms | **0.64x（慢）** |
| 4 | 0.72ms | 1.08ms | 0.66x |
| 8 | 0.74ms | 1.11ms | 0.67x |
| 32 | 0.80ms | 1.22ms | 0.66x |

### 5.6 性能分析：为什么 padding-free 反而更慢

**瓶颈分解**：

| 组件 | 时间 |
|------|------|
| dispatch (sort+histogram) | 0.12ms |
| gather (triton) | 0.09ms |
| **grouped GEMM w1** | **0.43ms** |
| activation | 0.02ms |
| **grouped GEMM w2** | **0.43ms** |
| scatter | 0.11ms |
| **Total** | **~0.84ms** |

**根因**：我们的 grouped GEMM kernel 单次 0.43ms，而 vllm 的 fused kernel 把 w1+act+w2+reduce 全做完只需 0.53ms。差距来自：

1. **vllm kernel 有 H100 专用 autotuning**（从 JSON config 文件加载精细调优的 tile/warp/stages 参数）
2. **Padding 的 GEMM 计算几乎免费**——kernel 用 mask 把 padding 行屏蔽为 0，`tl.dot` 对 0 值不耗时间
3. **真正的开销是 weight loading**——不管有没有 padding，每个 active expert 都要从 HBM 加载权重矩阵

**算术强度分析**：batch=1 时每个 expert 平均 2.5 tokens，GEMM 为 [2.5, 2048]×[2048, 512]。算术强度 ≈ 2.5×2048×512 / (2.5×2048 + 2048×512 + 2.5×512) / 2 ≈ **1.5 MAC/byte**。H100 的 compute/bandwidth 平衡点 ≈ 295 MAC/byte。**完全 memory-bound，padding 的额外 compute 相对于 weight loading 微不足道。**

### 5.7 关键结论

> **对于 LLaDA2.0-mini（256 experts, moe_intermediate=512）在 H100 上的推理，MoE kernel 的瓶颈不是 padding 浪费的计算，而是 expert weight 的 HBM 带宽。消除 padding 无法超越已经高度优化的 vllm baseline kernel。**

---

## 6. 其他已验证的性能数据

### 6.1 BD + IterSmooth 结合实验

| 路径 | 短 prompt tok/s | forward 数 | vs baseline |
|------|---------------|-----------|-------------|
| BD-attnmask (baseline) | 77.4 | 48 | — |
| **BD+IS high_w (w=0.5)** | **87.3** | **42** | **+12.8%** |
| BD+IS default (w=0.3) | 79.9 | 46 | +2.8% |

**性能差异 100% 来自 forward 次数**，per-forward 时间不变。

### 6.2 KV Cache 优化与 Batch Scaling

**4 路径 × batch scaling（长 prompt tok/s）**：

| batch | A: no-cache | B: IS-nocache | C: cache-opt | D: IS+cache |
|-------|------------|--------------|-------------|------------|
| 1 | **78** | 75 | 63 | 58 |
| 8 | 371 | 391 | **625** | 470 |
| 32 | 460 | 668 | **1672** | 1616 |

**交叉点：batch≥8 长 prompt 时 cache 路径大幅领先**（batch=32: 1672 vs 460, 3.67x）。

### 6.3 fused_moe Kernel 时间随 token 数的变化

| tokens | pairs | kernel time | us/pair |
|--------|-------|-------------|---------|
| 8 | 64 | 0.346ms | 5.4 |
| 32 | 256 | 0.549ms | 2.1 |
| 128 | 1024 | 0.719ms | 0.7 |
| 512 | 4096 | 0.767ms | 0.2 |
| 1024 | 8192 | 0.800ms | 0.1 |

从 64 pairs 到 8192 pairs（128x），kernel 时间只从 0.346ms 涨到 0.800ms（2.3x）。**GPU 在小 token 数时严重空闲。**

---

## 7. 已排除的方向

| 方向 | 尝试 | 结论 | 原因 |
|------|------|------|------|
| BW 框架 + IterSmooth/VicinityCache | v0.1.10 | 无法超越 BD-attnmask | BW 的全序列 full attention 开销太大 |
| MoE output 直接 reuse (Stable Cache) | v0.1.13 | 不可行 | 跳过计算 ↔ 保持 cache fresh 矛盾 |
| 多层 MoE 组合缓存 | v0.1.13 | 不可行 | 单层 safe ≠ 多层 safe，误差跨层传播 |
| Shared-only 近似 | v0.1.13 | 不可行 | Cosine sim 仅 0.57-0.95，routed 不可忽略 |
| Padding-free grouped GEMM | v0.1.13 | 无法超越 baseline | Weight loading 是真正瓶颈，不是 padding compute |
| Compact expert set | v0.1.13 | 更慢 | Weight gather copy 开销 + 缺少 autotuning |

---

## 8. 仍然成立的优化机会

### 8.1 减少 forward 次数（已验证有效）

IterSmooth 已证明 forward 48→42（-12.8%），对应 77→87 tok/s。更激进的解码策略可能进一步减少。**这是乘性收益——少一次 forward 就少一次完整的 20 层计算。**

### 8.2 三个 Insight 作为 Characterization 贡献

三个 insight 在所有测试设置（同质/异质 batch, temperature=0/0.7, cache/no-cache）下全部稳健成立。作为 observation 贡献，它们量化了 dLLM MoE 与 AR MoE 的根本差异，是发表系统论文的基础。

### 8.3 多卡 EP 场景（未实验）

Insight A + B 在多卡 EP 下的影响被放大：
- A: MASK routing 集中 → 跨 GPU 负载不均 + all-to-all 热点
- B: 跨迭代冗余 → 省 all-to-all 通信量
- RBD (X-MoE 的 Redundancy-Bypassing Dispatch) + dLLM MASK routing 可能有协同效应

---

## 9. 相关工作交叉分析

### 9.1 与 13 篇 MoE 系统论文的关系

| 论文 | 会议 | 与我们的关联 |
|------|------|------------|
| **EARTH** (ASPLOS'26) | entropy-aware prefetch + result reuse | Insight A (entropy差异) + Insight B (result reuse)；但 EARTH 解决搬运瓶颈，我们是计算瓶颈 |
| **X-MoE** (SC'25) | padding-free + RBD for DeepSeek-style MoE | 架构同类（256 experts, top_k=8）；PFT 已验证但性能不及 vllm autotuned kernel |
| **Diff-MoE** (SC'25) | differential cache hierarchy | 我们有更强的 iteration locality（因果确定性） |
| **Samoyeds** (EuroSys'25) | structured sparsity kernel | MASK routing 集中 = activation-side structured sparsity |
| **MoE-APEX** (ASPLOS'26) | adaptive precision offloading | 精度层次化思路可迁移到 stable/active 位置 |
| **LAER-MoE** (ASPLOS'26) | load-adaptive re-layout | Insight A 导致的负载不均在多卡下需要动态重布局 |

### 9.2 EARTH 的可借鉴思想

| 思想 | EARTH 的应用 | 我们的场景 |
|------|------------|-----------|
| 重要性分层 | expert 按 sensitivity 分 important/moderate/unimportant | 位置按 token 状态分 MASK/decoded/stable |
| 渐进式近似 | base-only → base+delta → full | 我们尝试了 all-or-nothing（skip 或 full），渐进式未充分探索 |
| 离线校准阈值 | gating weight threshold 控制精度损失 <1% | 我们的 ablation 是静态的，未做自适应阈值 |

**关键认知**：EARTH 解决的是 expert 搬运瓶颈（88% 时间在搬），我们的场景是计算/带宽瓶颈。技术不能直接套用，但分层/渐进思想有价值。

---

## 10. 开放问题（供讨论）

1. **52% 的跨迭代冗余能否被安全利用？** 直接 skip 不行（误差累积），有没有其他方式（如近似、低精度、adaptive threshold）可以利用？

2. **MASK routing 集中这个性质有没有 kernel 层面的利用方式？** 浅层 MASK 只激活 64/256 experts，但当前 kernel 的瓶颈是 weight loading 而非 compute。有没有办法减少 weight loading？

3. **跨迭代的 expert weight caching** 是否可行？同一个 block 迭代 ~12 次，每次经过相同的 19 层 MoE，expert weights 相同。H100 L2 cache 50MB，单层 expert weights ~2MB×256 = 512MB，远超 L2。但如果只缓存 hot experts 呢？

4. **减少 forward 次数**是否是比 MoE kernel 优化更有前途的方向？IterSmooth 已验证 -12.8%，更激进的解码策略（dynamic threshold、logit margin、block-level collective decision）能否进一步减少？

5. **多卡 EP + dLLM 的 MASK routing 特性** 是否能形成一个独立的研究方向？Insight A 在多卡下导致 all-to-all 流量不均，X-MoE 的 RBD 在 dLLM 中可能有更大收益。

---

## 11. 附录：环境与工具

```
torch==2.8.0+cu128
transformers==5.3.0
vllm==0.10.2
triton==3.4.0
GPU: NVIDIA H100 80GB HBM3 (132 SMs, 3.35 TB/s HBM)
模型路径: /home/wuhang/models/LLaDA2.0-mini
```

### 数据文件位置

| 文件 | 内容 |
|------|------|
| `codex_coding/results/moe_routing_analysis_hetero_batch.json` | 异质 batch routing 分析（Insight A+B+C 完整数据） |
| `codex_coding/results/moe_decomp_analysis_results.json` | Shared/routed 分解分析 |
| `codex_coding/results/moe_layer_ablation_refresh_results.json` | 逐层 ablation + 刷新实验 |
| `codex_coding/results/moe_multilayer_ablation_results.json` | 多层组合 ablation |
| `codex_coding/results/stable_cache_benchmark_results.json` | Stable cache 性能/正确性 |
| `codex_coding/results/batch_4paths_benchmark_results.json` | 4 路径 batch scaling |
| `codex_coding/results/padding_free_moe_benchmark_results.json` | Padding-free kernel 性能 |
