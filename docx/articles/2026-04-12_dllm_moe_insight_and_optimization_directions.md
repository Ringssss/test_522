# dLLM MoE 推理的独特行为与优化方向

> 日期：2026-04-12
> 阶段：v0.1-init-project
> 前置文档：
> - `docx/articles/2026-04-09_diffusion_llm_inference_and_acceleration_landscape.md`（dLLM 推理综述）
> - `docx/cites/moe_systems_survey.md`（MoE 系统论文调研）
> - `code_building/process_docs/v0.1-init-project/v0.1.12-kv_cache_optimization_and_batch_scaling.md`（batch scaling 实验数据）

---

## 1. 背景与动机

从 profiling 数据看，LLaDA2.0-mini（256 experts, top_k=8, 20 layers）在 cache 路径上，MoE fused kernel 是 forward 中最大的单一开销（约 50% CUDA time）。

现有 MoE 系统论文（Klotski, EARTH, Diff-MoE, X-MoE 等）全部面向 AR 推理范式。dLLM 的迭代去噪结构在 MoE 层引入了三种 AR 中不存在的独特行为，这些行为既是效率损失的来源，也是优化的机会。

### LLaDA2.0-mini MoE 关键参数

```
num_experts:          256
num_experts_per_tok:  8 (top_k=8)
n_group:              8
topk_group:           4 (grouped routing: 8 groups, select top 4 groups, then top 8 experts)
num_shared_experts:   1 (所有 token 都过的 shared expert)
moe_intermediate_size: 512
hidden_size:          2048
num_hidden_layers:    20 (layer 0 dense, layer 1-19 MoE)
routed_scaling_factor: 2.5
```

---

## 2. 三个核心 Insight

### Insight A：MASK token 的 expert routing 集中现象

**观察**：dLLM 每步输入的 block（32 tokens）中包含已解码 token 和 MASK token。MASK token 全部共享同一个 embedding (id=156895)。经过 attention 后获得上下文，但起点相同。

**推测**：
- 进入 MoE gate 时，同一层的多个 MASK token 的 hidden state 高度相似（尤其在浅层）
- MASK tokens 倾向于 route 到相同的一小组 experts
- 造成 expert 负载严重不均衡：少数 expert 被淹没，多数 expert 空闲
- fused_moe kernel 效率下降（假设 tokens 大致均匀分布到 experts，不均衡导致 warp divergence）

**逐层演化推测**：
- Layer 1-5（浅层 MoE）：mask 特征主导 → routing 极度集中
- Layer 10+（深层 MoE）：上下文稀释 mask 特征 → routing 趋向正常分布
- Layer 19（最终层）：routing 最接近语义 token 的分布

**与 AR 的差异**：AR decode 中不存在 MASK token，所有 token 都是语义丰富的，routing 近似均匀随机。

### Insight B：跨迭代的 MoE 计算冗余

**观察**：dLLM 一个 block 内需要多次迭代（典型 ~12 次）才能完成解码。每次迭代，所有 32 个位置都过 MoE（20 层 × 每层 1 次）。但已解码位置的 token 没有改变，其 MoE 输入、routing、expert 输出大概率与上一步几乎一致。

**冗余率估算**：
```
假设一个 block 平均 12 次迭代，每步解码 ~2.7 个新 token：
- 总 MoE 计算: 12 步 × 32 tokens = 384 次/层
- 有效新计算: ~32 次/层（每个位置只需算一次 "最终版本"）
- 冗余率上界: 384/32 - 1 = 91%
```

实际冗余率取决于：已解码位置虽然 token 不变，但 attention 输出会因其他位置变化而微变 → MoE input 有轻微变化 → routing 和 output 可能有小幅偏移。需要实验验证实际的 routing 稳定性和 output 相似度。

**与 AR 的差异**：AR 中每步只处理 1 个全新 token，不存在"反复计算同一位置"的现象。这种跨迭代冗余是 dLLM 特有的。

### Insight C：dLLM 的天然批量效应

**观察**：
```
AR decode:   每请求 1 token/step → batch=32 时 MoE 处理 32 tokens
dLLM decode: 每请求 32 token/step → batch=1 时 MoE 就处理 32 tokens, batch=32 时处理 1024 tokens
```

**影响**：
- dLLM 在 batch=1 时的 MoE GPU 利用率 ≈ AR 在 batch=32 时
- dLLM 天然比 AR 更能喂饱 MoE kernel
- 但在大 batch 下更容易进入 compute-bound（1024 tokens → expert GEMM 已经 reasonable size）
- 最优 batch size 可能比 AR 更小

---

## 3. 多卡视角下的影响放大

在 Expert Parallel (EP) 部署下，上述三个 insight 的影响进一步放大：

### Insight A 在多卡下 → 通信热点 + GPU 负载不均

```
单卡: MASK tokens 集中到少数 expert → 那些 expert 的 GEMM 更大，但 GPU 总利用率还行
多卡: MASK tokens 集中到少数 expert → 这些 expert 所在的 GPU 过载
      → 其他 GPU 空等
      → all-to-all 流量不均衡（大量 token 涌向少数 GPU）
```

### Insight B 在多卡下 → 省通信比省计算更值钱

```
单卡: 跳过已解码位置的 MoE → 省 ~50% compute
多卡: 跳过已解码位置的 MoE → 省 ~50% compute + 省 ~50% all-to-all 通信
```

跳过已解码位置的 expert dispatch 意味着这些位置的 token **根本不需要参与 all-to-all**。这是纯粹的通信节省，不需要任何近似或精度损失。

多卡收益结构 = 节省 MoE compute + 节省 all-to-all volume + 减少 GPU 间等待。

### Insight C 在多卡下 → all-to-all volume 压力

```
AR decode:   batch=32 → 32 tokens 参与 all-to-all
dLLM decode: batch=32 → 1024 tokens 参与 all-to-all（32x）
```

dLLM 天然的大 token 批量意味着每步 all-to-all 通信量是 AR 的 block_size 倍。Insight B 的跳过机制直接削减这个量。

---

## 4. 与现有 MoE 系统论文的交叉分析

### 第一档：核心思路可直接借用

**EARTH (ASPLOS 2026)** — 最相关
- 提出 entropy-aware speculative prefetch + result reuse
- 与我们的 Insight B 高度对齐
- 关键差异：EARTH 面向 AR（token 间相似性弱，复用机会有限）；我们面向 dLLM（同一位置跨迭代反复计算，复用机会巨大）
- EARTH 的 entropy 概念与 Insight A 关联：MASK tokens → 低 routing entropy → 高度可预测 → reuse 可靠

**X-MoE (SC 2025)** — 技术可迁移
- 面向 DeepSeek-style 大 top_k 架构（LLaDA2.0-mini 恰好是 top_k=8）
- padding-free execution + redundancy-bypassing dispatch
- 与 Insight A 关联：MASK tokens 集中 route → dispatch 阶段冗余 → 可 bypass

**Diff-MoE (SC 2025)** — 缓存思路可扩展
- 区分 global locality 和 temporal locality
- 我们有更强的 "iteration locality"：同一位置跨迭代 route 到相同 expert（因果确定，非统计相关）

### 第二档：多卡场景下变得相关

**LAER-MoE (ASPLOS 2026)** — load-adaptive expert re-layout
- 如果 MASK routing 不均衡可预测，可提前做 placement/replication

**ScheMoE (EuroSys 2024)** — task scheduling + all-to-all
- 推理也可建 task graph，dispatch/load/compute/gather 联合调度
- dLLM 的跳过机制可以减少 dispatch task 数量

**PopFetcher (ATC 2025)** — popularity-based expert-wise prefetch
- expert popularity 在 dLLM 中有更强结构（MASK → 集中 → 可预测）
- 可适配到跨迭代的 expert usage 预测

### 第三档：概念启发

| 论文 | 有价值的概念 | 适用条件 |
|------|-------------|---------|
| Klotski (ASPLOS 2025) | expert 相关性 prefetch | expert 需要 offload 时 |
| Samoyeds (EuroSys 2025) | dual structured sparsity kernel | 模型权重本身是 sparse 时 |
| MoE-APEX (ASPLOS 2026) | adaptive precision offloading | 边缘/内存受限环境 |
| KTransformers (SOSP 2025) | CPU/GPU hybrid inference | 单卡/少卡低并发环境 |
| CoServe (ASPLOS 2025) | dependency-aware scheduling | 多模型协作系统 |

### 交叉矩阵

```
                    EARTH          X-MoE          Diff-MoE       LAER-MoE     ScheMoE
                 (result reuse)  (bypass dispatch) (diff cache)  (load adapt)  (task sched)

Insight A        entropy 区分     bypass MASK 的    MASK =         MASK 导致    MASK popularity
(routing集中)    reuse 可靠度     冗余 dispatch     确定性 locality 跨GPU负载    跨层可预测
                 [单+多卡]       [单+多卡]         [单+多卡]      不均[多卡]   [多卡]

Insight B        复用已解码位置   跳过已解码位置    iteration 级    已解码位置   已解码位置
(跨迭代冗余)     MoE 输出         的 all-to-all     differential   不参与       不需要跨GPU
                 [单+多卡]       [多卡核心]        caching        rebalance    dispatch
                                                   [单+多卡]      [多卡]      [多卡核心]

Insight C        批量效应改变     大 token 批量下   batch-aware     token数量    all-to-all
(天然批量)       reuse 收益       dispatch 效率     cache policy   放大不均衡   volume=32x AR
                 [单卡]          更关键[单+多卡]    [单+多卡]      [多卡]      [多卡核心]
```

---

## 5. 最有论文潜力的方向

### 方向定位："Iteration-Aware Selective MoE for Diffusion LLM Inference"

核心论点：AR 推理中的 MoE 每步处理全新 token，每次 expert 计算都是必要的。dLLM 推理中的 MoE 反复处理同一批位置，大量 expert 计算是冗余的。

### 与现有工作的差异化

```
vs EARTH:    EARTH 在 AR 中做 result reuse，收益有限（token 间差异大）
             我们在 dLLM 中做 result reuse，收益巨大（同一位置跨迭代不变）

vs X-MoE:    X-MoE 避免 padding 冗余（空间维度）
             我们避免迭代冗余（时间维度）

vs Diff-MoE: Diff-MoE 基于统计 locality 做缓存
             我们基于因果确定性做计算跳过——不是"可能命中"，而是"一定命中"
```

### 论文故事线框架

```
1. 观察层: dLLM 的迭代结构在 MoE 层产生三种 AR 不存在的独特行为
   → MASK routing 集中、跨迭代冗余、天然大批量

2. 影响层: 这些行为在单卡和多卡下都造成效率损失
   → 单卡: ~50% 冗余计算
   → 多卡: + 冗余 all-to-all + load imbalance

3. 方法层: Iteration-Aware Selective MoE
   → 单卡: 跳过已解码位置的 expert 计算，复用上一步结果
   → 多卡: 同时跳过 dispatch 和 all-to-all

4. 评估层:
   → 单卡 H100: LLaDA2.0-mini（当前可做）
   → 多卡 EP: 更大模型（可做 simulation 或后续实验）
```

---

## 6. 待验证的关键假设与实验设计

### 实验 1: MASK vs 已解码 token 的 routing 分布

**目的**: 验证 Insight A（MASK routing 集中）

**方法**: hook gate 输出，记录每步每层每位置的 topk_idx，按 mask/decoded 分桶统计 expert load distribution。计算 per-layer 的 entropy、active expert count、max/avg load ratio。

**预期**: 浅层 MASK entropy 远低于 decoded；随层加深差距缩小。

### 实验 2: 跨迭代的 routing 稳定性

**目的**: 验证 Insight B（已解码位置 routing 不变）

**方法**: hook gate 输出，记录同一位置在连续迭代中的 topk_idx。计算 routing change rate = |iter[t] 的 topk_idx != iter[t-1] 的 topk_idx| / total。

**预期**: 已解码位置 routing change rate ≈ 0；MASK 位置 change rate 较高但浅层可能也有规律。

### 实验 3: 跨迭代的 MoE output 相似度

**目的**: 验证 result reuse 是否安全

**方法**: hook MoE block output，对同一位置跨迭代计算 cosine similarity。分别统计 mask 和 decoded 位置。

**预期**: 已解码位置 cosine sim > 0.99；MASK 位置 cosine sim 较低但可能也 > 0.9。

**实施**: 三个实验可用一个 hook 脚本同时收集，一次 generate() 调用即可。

---

## 7. 其他备选优化方向（较小但可能有用）

### 7.1 预分配 KV cache buffer + 消除 block 边界重建

已在前期分析中定位：block 边界的 F.pad/consolidate/rebuild 开销可通过预分配 contiguous buffer 消除。

预期收益：~10ms/forward（占比取决于 batch size）。这是工程优化，不构成独立论文方向，但可以作为系统实现的一部分。

### 7.2 CUDA Graph capture for within-block iterations

block 内的 forward 形状固定（32 tokens），适合 CUDA Graph 捕获和重放。可消除 Python 循环 overhead 和 kernel launch latency。

当前受限：torch.compile 在 torch 2.8.0 + LLaDA2MoE 上触发 InductorError。CUDA Graph 可能可行但需要验证。

### 7.3 MoE kernel 的 MASK-aware 优化

如果实验验证 MASK routing 确实高度集中，可以设计 MASK-aware fused_moe kernel：
- 对 MASK tokens 做 batched routing（一次 gate 计算，结果广播）
- 或对 MASK tokens 使用 approximate expert（共享一个 "average expert" 的输出）

这需要较深的 kernel 开发，但如果 routing 集中度足够高，收益可能很显著。

---

## 8. 实验验证结果与 Stable Cache 失败分析（2026-04-12 补充）

### 8.1 三个 Insight 的实证验证

全部三个 insight 在异质 batch + 随机采样下稳健成立：

| 指标 | batch=1 同质 | batch=8 异质 | batch=32 异质 |
|------|-----------|------------|-------------|
| MASK/Dec entropy ratio | 0.70 | 0.65 | 0.66 |
| Dec routing change % | 19-48% | 12-27% | 10-27% |
| Dec output cosine sim | 0.96-0.99 | 0.97-0.99 | 0.97-0.99 |
| 总冗余率 | 30% | 49% | 52% |

### 8.2 Shared vs Routed 分解

- routed/shared 量级比在多数层 0.72-1.68（routed 不可忽略）
- shared-only 近似 cosine sim 仅 0.57-0.95 → **不可行**
- v1 缓存（full output）cosine sim 0.87-0.98 → 更好的近似基础

### 8.3 逐层 Ablation

- 单层缓存：18/19 MoE 层可安全缓存（唯一敏感层 Layer 18）
- 周期性刷新 N=2~10 全部 exact match

但这是 **"每步全量计算 + 只替换输出"** 的结果，cache 始终 fresh。

### 8.4 Stable Cache 失败的根本原因

v1、v2 全部失败。根因分析：

```
ablation（exact match）:  每步都完整计算 → cache 始终 1-step-old（新鲜）→ 替换输出安全
实际跳过计算:             跳过 stable 位置 → cache 不被更新 → 越来越旧 → 误差累积 → 崩溃
```

**跳过计算和保持 cache fresh 是矛盾的。** 要保持 cache fresh 必须做全量计算，但那就不省计算了。

### 8.5 多层组合 Ablation 的致命发现

```
单独安全: L1-3 ✓, L6-13 ✓, L19 ✓
任意组合: L1-3+L19 ✗, L6-13+L19 ✗, L1-3+L6-13 ✗, L1-3+L6-13+L19 ✗
```

**单层 safe ≠ 多层组合 safe。** 每层的 "safe" 是建立在其他层全部精确的前提上。一旦多层同时缓存，误差叠加超过容错边界。

### 8.6 EARTH 论文的启发

EARTH 解决的是 expert 搬运瓶颈（不是计算瓶颈），但其三个设计思想可借鉴：
1. 重要性分层：不是所有东西都需同等对待
2. 渐进式近似：不做 all-or-nothing
3. 离线校准阈值：控制精度损失在可接受范围内

### 8.7 结论与下一步方向

"跳过 stable 位置 MoE 计算" 在当前 transformer 架构下 **不可行**。三个 insight 作为 observation 成立，但直接转化为 compute-skipping 有根本性困难。

需要转向其他方向：
- 减少 forward 次数（IterSmooth 已证明 -12.8%）
- MoE 精度降级（对 stable 位置用低精度计算）
- Block 调度优化（dynamic batching）
- 以 insight 作为 characterization 贡献
