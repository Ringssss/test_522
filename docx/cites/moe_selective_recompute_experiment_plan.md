# MoE Selective Recompute Risk Proxy：完整实验设计文档

> 版本：v0.1  
> 目的：为 dLLM / MoE 场景下的 **selective recompute（选择性重算）** 提供一套可直接执行的研究与工程方案。  
> 适用对象：研究者、系统工程师、代码代理（AI coding agent）、实验执行同学。  
> 语言：中文  
> 输出形式：可直接作为后续代码建设、实验记录、结果判断的总纲。

---

# 1. 研究背景

## 1.1 问题背景

在 dLLM（Diffusion LLM）或 block diffusion 模型中，一个 block 内的 token 不是一次性生成完成，而是经过多轮迭代去噪（denoising iterations）逐步从 `MASK` 变成最终 token。  
与 AR（autoregressive）模型不同，dLLM 在每一轮中会同时处理 block 内多个位置，因此天然具有并行性；但与此同时，它也带来了一个很突出的推理代价：

- 同一个 block 会被重复前向多次；
- 已经“稳定”的 token 在后续迭代中仍然继续经过完整模型；
- 在 MoE（Mixture-of-Experts）模型里，这意味着 routed experts 可能被反复调用。

这会形成显著的 **跨迭代计算冗余**。

在已有观察中，常能看到以下现象：

- 一部分 token 已经在多轮中保持高置信、top-1 不变；
- 一部分 token 的 routing（top-k experts）在相邻迭代中几乎不变；
- 一部分层的 MoE 输出在跨迭代中相似度很高。

这些现象提示我们：

> 也许不是每一轮、每一层、每一个 token 都需要重新计算 routed experts。

但简单粗暴地“跳过 stable token 的 MoE 计算”通常会失败，因为误差会在层间传播并最终改变解码行为。

---

## 1.2 现有困难

朴素方案通常采用类似以下规则：

- `MASK` token：总是重算；
- 已经稳定的 token：直接复用上一次的 routed output，不重算。

这种规则的问题在于它只依据 **token 表面状态** 做决策，而没有判断：

> 这个 token 在当前层的 routed MoE 输出，是否已经“陈旧（stale）”到必须刷新。

换句话说，`stable token != safe to reuse routed output`。

这就导致：

- 某些 token 虽然已经 decode，但当前层 hidden state 仍在漂移；
- 某些层虽然单层误差小，但多层串联后误差会放大；
- 某些 token 的 routed experts 没变，但权重变了；
- 某些 token 的局部误差很小，但会在后续 attention / residual / deeper MoE 中放大。

因此，真正的问题不是“stable token 要不要跳过”，而是：

> **我们能否找到一个低成本信号（proxy），来预测：如果复用上一次 routed experts 的结果，这一次是否会把误差放大到不可接受？**

---

## 1.3 与 Elastic-Cache 的关系

Elastic-Cache 的核心思想不是“所有不重要 token 都跳过”，而是：

- 找到反映 stale risk 的低成本信号；
- 用这些信号决定 **什么时候刷新（when to refresh）**；
- 再用层级信息决定 **从哪里开始刷新（where to refresh）**。

这给 MoE selective recompute 的重要启发是：

> **我们不应该只按 token 类别做跳过，而应该按 stale risk 做动态刷新。**

因此，本项目的研究目标是构造一个 **MoE-specific stale-risk predictor**。

---

# 2. 研究目标与核心问题

## 2.1 总目标

设计一套实验体系，找到一个或一组 **低成本信号 proxy**，它能够预测：

> 在 step `t`、layer `l`、token `i` 上，如果复用上一次 routed experts 的输出，而不做 fresh routed compute，误差是否会在后续层中被放大到不可接受。

然后基于这个 proxy，构建：

- per-token selective recompute
- per-layer selective recompute
- sentinel-triggered refresh
- layer-boundary refresh

等真正可部署的策略。

---

## 2.2 核心问题

本项目要回答四个问题：

### Q1. 哪些低成本信号最能预测 skip risk？
例如：

- hidden drift
- gate logits drift
- top-k expert overlap
- routing weight drift
- token confidence
- layer / step 信息

### Q2. skip risk 是否具有明显的 layer-wise 差异？
即：

- 浅层是否更容易安全复用？
- 深层是否更容易放大误差？
- 是否存在个别敏感层必须强制 fresh？

### Q3. skip risk 是否具有 token-state 差异？
例如：

- `MASK`
- newly decoded
- stable decoded
- highly stable decoded

是否需要不同阈值和不同 policy？

### Q4. 少量 sentinel token 能否代表大部分 token 的 refresh 需求？
这决定后续能不能做低成本、工程上可行的 refresh trigger。

---

# 3. 核心思想与方法概述

## 3.1 从 heuristic 转向 risk prediction

旧思路是 heuristic：

- stable token -> reuse
- MASK token -> recompute

新思路是 prediction：

- 根据低成本 proxy 预测 stale risk
- stale risk 低 -> reuse
- stale risk 高 -> refresh

因此，问题从“设计规则”转变为“设计预测器”。

---

## 3.2 两阶段研究路线

整个项目分为两个阶段：

### 阶段 I：Characterization（刻画与分析）
目的：

- 建立单点 skip 的反事实数据集；
- 定量测量误差如何传播；
- 筛选最有效的低成本 proxy。

这阶段不追求 speedup，主要追求 **因果清晰的风险评估**。

### 阶段 II：Policy（策略化）
目的：

- 将阶段 I 找到的 proxy 转化为真正的 selective recompute 规则；
- 测速、测精度、测 forward 次数；
- 比较不同 refresh granularity 的工程实用性。

本文件重点覆盖阶段 I，同时给出阶段 II 的接口与落地方式。

---

# 4. 术语与数学定义

为了后续实现、日志采集和 AI 对话统一，这里先定义术语。

## 4.1 基本索引

记：

- `t`：denoising iteration / diffusion step
- `l`：MoE layer index
- `i`：token index within current block

一个基础样本记为：

\[
x = (t, l, i)
\]

表示：在第 `t` 轮、第 `l` 层、第 `i` 个 token 位置上的一次潜在复用决策。

---

## 4.2 基本变量

### 4.2.1 pre-MoE hidden state

记当前层进入 MoE 前的 hidden state 为：

\[
h_{t,l,i} \in \mathbb{R}^d
\]

其中 `d` 是 hidden dimension。

它反映该 token 在当前 step、当前层进入 MoE 前的表示。

---

### 4.2.2 gate logits

MoE gate 输出每个 expert 的打分：

\[
g_{t,l,i} \in \mathbb{R}^{E}
\]

其中 `E` 是 experts 数量。

通常会从中选择 top-k experts：

\[
\mathcal{K}_{t,l,i} = \text{TopK}(g_{t,l,i})
\]

---

### 4.2.3 routing weights

对 top-k experts 的归一化权重记为：

\[
w_{t,l,i} \in \mathbb{R}^{k}
\]

其中 `k` 是每个 token 选择的 experts 数量。

---

### 4.2.4 shared / routed output

MoE 输出通常分为：

- shared path output
- routed experts output

记：

\[
y^{shared}_{t,l,i}
\]

\[
y^{routed}_{t,l,i}
\]

最终 MoE block 输出为：

\[
y^{moe}_{t,l,i} = y^{shared}_{t,l,i} + y^{routed}_{t,l,i}
\]

---

### 4.2.5 cached routed output

如果复用上一轮的 routed experts 结果，则记缓存值为：

\[
\tilde{y}^{routed}_{t,l,i} = y^{routed}_{t-1,l,i}
\]

选择性复用时，MoE 输出变为：

\[
\tilde{y}^{moe}_{t,l,i} = y^{shared}_{t,l,i} + \tilde{y}^{routed}_{t,l,i}
\]

问题是：

\[
\tilde{y}^{moe}_{t,l,i} \approx y^{moe}_{t,l,i}
\]

是否成立，以及这个局部误差会不会在后续传播中被放大。

---

## 4.3 drift 的数学定义

### 4.3.1 cosine drift

对任意两个向量 `a, b`，其 cosine similarity 定义为：

\[
\cos(a,b)=\frac{a^\top b}{\|a\| \|b\|}
\]

对应的 drift 可写作：

\[
d_{cos}(a,b)=1-\cos(a,b)
\]

当 `d_cos` 越小，表示两个状态越相似。

---

### 4.3.2 relative L2 drift

\[
d_{relL2}(a,b)=\frac{\|a-b\|_2}{\|a\|_2 + \epsilon}
\]

用于度量相对变化强度。

---

### 4.3.3 top-k overlap

若当前 step 与上一步的 top-k experts 集合分别是：

\[
\mathcal{K}_{t,l,i}, \mathcal{K}_{t-1,l,i}
\]

则重叠率定义为：

\[
overlap_{topk}=\frac{|\mathcal{K}_{t,l,i}\cap \mathcal{K}_{t-1,l,i}|}{k}
\]

这是一个非常便宜且直观的 routing 稳定性指标。

---

### 4.3.4 routing weight drift

若 routing weights 分别为 `w_t, w_{t-1}`，则可以计算：

- cosine
- L1
- KL / JS divergence

例如 L1 drift：

\[
d_{L1}(w_t,w_{t-1}) = \|w_t-w_{t-1}\|_1
\]

---

### 4.3.5 logits KL divergence

若最终输出 logits 对应概率分布为 `p` 和 `q`，则：

\[
D_{KL}(p \| q)=\sum_j p_j \log \frac{p_j}{q_j}
\]

它衡量 skip 造成的最终输出分布偏移程度。

---

# 5. 真正要预测的对象：Skip Risk

## 5.1 为什么不能只看局部误差

我们真正关心的是：

> “如果复用上一轮 routed experts 的结果，而不 fresh compute，这个误差会不会在后续层、最终 logits 甚至解码轨迹中被放大？”

因此，单层局部误差只是中间指标，不是最终目标。

---

## 5.2 三层风险定义

### Level A：局部输出风险
衡量当前层 MoE 输出是否已经明显偏离：

- `local_routed_cos`
- `local_moe_cos`
- `local_moe_relL2`

### Level B：传播风险
衡量该局部误差是否影响后续层：

- next-layer hidden drift
- final-layer hidden drift
- final logits KL
- final top-1 changed?

### Level C：行为风险
衡量 skip 是否改变解码行为：

- token accepted step 是否变化
- 最终 token 是否变化
- block 完成所需 forward 次数是否变化

---

## 5.3 建议的风险标签

### 回归标签
对每个样本 `(t,l,i)` 记录：

- `local_out_error`
- `final_hidden_error`
- `final_logits_kl`
- `accepted_iter_shift`
- `output_changed`

### 分类标签
定义：

- `safe`
- `borderline`
- `unsafe`

例如：

若满足任一条件，则 `unsafe`：

- final top-1 changed
- final logits KL > ε1
- accepted iteration shift > ε2

若全部极小，则 `safe`。  
介于中间者记为 `borderline`。

---

# 6. 实验总框架

# 6.1 基线运行（Full Fresh Run）

对一批 prompts 正常执行完整推理，不做任何 selective recompute。  
在运行过程中记录每个 `step × layer × token` 的所有中间变量。

需要记录：

- token state
- hidden state
- gate logits
- top-k experts
- routing weights
- shared output
- routed output
- final moe output
- final logits
- accept status
- token confidence / margin

这是全套分析的参考基线。

---

## 6.2 单点反事实干预（Single-Site Counterfactual Intervention）

对于一个样本 `(t,l,i)`：

- 保持其它 token、其它层全部 fresh；
- 只在该层该 token 上，用上一轮的 routed output 替代 fresh routed output：

\[
y^{routed}_{t,l,i} \leftarrow y^{routed}_{t-1,l,i}
\]

然后继续向后执行这一轮 forward，记录误差传播结果。

这是整个项目最关键的实验设计。  
因为只有单点干预，才能保证：

- 因果清晰
- 标签明确
- 能训练和筛选 proxy

---

## 6.3 为什么必须单点干预

如果一开始就多层、多 token 同时 skip，会有三个问题：

1. 误差来源混合，无法归因
2. 无法判断到底哪种 proxy 真在预测风险
3. 无法得到 clean 的 safe/unsafe 标签

因此必须先做 **single-site intervention**。

---

# 7. 数据采样方案

单点干预很贵，所以必须有设计地采样。

## 7.1 覆盖设置

建议至少覆盖这些主设置：

- batch = 1 / 8 / 32
- temperature = 0 / 0.7
- homogeneous / heterogeneous prompts
- no-cache path 作为主线

如果资源有限，优先：

- batch = 8
- temperature = 0.7
- heterogeneous

因为这个设置通常最接近真实使用场景，也最容易暴露风险。

---

## 7.2 token 分层采样

每个 `step × layer` 里，不要全量干预所有 token。  
按 token 类型分层采样：

- 2 个 MASK token
- 2 个 newly decoded token
- 2 个 stable decoded token
- 2 个 highly stable decoded token

其中 highly stable decoded 可定义为：

- 连续 N 轮 top-1 未变
- 且 confidence 较高
- 且 margin 较大

---

## 7.3 层采样

第一轮不必全层铺开。建议先选：

- 浅层：L1 / L3
- 中层：L6 / L10 / L13
- 深层：L17 / L18 / L19

因为这些层足以覆盖：

- early representation mixing
- middle computation
- late sensitive layers

如果已有证据表明某层特别敏感，应单独重点观测。

---

## 7.4 step 采样

建议按迭代阶段采样：

- early steps：1–3
- middle steps：4–8
- late steps：9–15

因为不同阶段：

- token 稳定比例不同
- MASK 比例不同
- 风险结构可能不同

---

# 8. 候选低成本信号设计

本节定义待筛选的 proxy 候选集。

## 8.1 A 类：token-state signals

这些最便宜，用作 baseline。

### 候选项
- token type：MASK / newly decoded / stable decoded
- 连续多少轮 top-1 不变
- 当前 token confidence
- top1-top2 margin
- 过去两轮 confidence 变化量
- token accept status 是否连续稳定

### 用途
验证“只靠 token state 是否足够”。

---

## 8.2 B 类：pre-MoE hidden signals

### 候选项
- hidden cosine drift
- hidden relative L2 drift
- hidden norm drift
- hidden principal projection drift（可选）

### 用途
衡量进入 MoE 前的状态变化是否能预测 stale risk。

---

## 8.3 C 类：gate / routing signals

这类最贴近 MoE 本体，是重点。

### 候选项
- gate logits cosine
- gate logits relative L2
- gate logits KL / JS
- top-k expert overlap
- top-k rank displacement
- routing weight cosine
- routing weight L1 drift
- routing entropy drift
- routing Gini drift

### 解释
如果 routed experts 的集合和权重都几乎没变，那么复用 routed output 可能是安全的；  
如果这些量变化明显，则 stale risk 更高。

---

## 8.4 D 类：cheap output proxy

这些比 pure routing 稍贵，但仍便宜于 full routed compute。

### 候选项
- shared fresh output 的变化量
- cached routed output 与 fresh gate 的匹配残差
- routed output 的低维 sketch 相似度
- final moe output 的 cheap estimate

### 用途
更直接地近似最终误差。

---

## 8.5 E 类：上下文/层级信号

### 候选项
- layer id
- step id
- 当前 block 的 MASK ratio
- token 与最近稳定 token 的距离
- 当前层是否敏感层
- batch size
- temperature

### 用途
这些不是直接 drift，但几乎一定会影响判据阈值。

---

# 9. 日志与数据结构设计

为了后续 AI 和代码系统能直接工作，这里定义建议日志结构。

## 9.1 样本级记录格式

每条样本（一个 `(t,l,i)`）建议存成一行 JSON / parquet record，字段包括：

```json
{
  "sample_id": "...",
  "prompt_id": "...",
  "batch_size": 8,
  "temperature": 0.7,
  "step": 6,
  "layer": 10,
  "token_idx": 14,
  "token_state": "stable_decoded",
  "token_confidence": 0.992,
  "token_margin": 0.845,
  "stable_len": 4,

  "mask_ratio_block": 0.41,

  "hidden_cos_prev": 0.9981,
  "hidden_rel_l2_prev": 0.014,

  "gate_cos_prev": 0.9968,
  "gate_rel_l2_prev": 0.022,
  "gate_topk_overlap_prev": 0.875,
  "gate_weight_cos_prev": 0.991,
  "gate_weight_l1_prev": 0.053,
  "routing_entropy": 1.27,
  "routing_entropy_delta": -0.05,

  "shared_out_norm": 12.4,
  "cached_routed_norm": 8.1,

  "intervention_type": "reuse_prev_routed_only",

  "local_moe_cos": 0.985,
  "local_moe_rel_l2": 0.041,
  "next_hidden_cos": 0.981,
  "final_hidden_cos": 0.945,
  "final_logits_kl": 0.023,
  "final_top1_changed": false,
  "accepted_iter_shift": 0,
  "final_output_changed": false,

  "risk_label": "safe"
}
```

---

## 9.2 推荐文件组织

```text
experiments/
  proxy_risk_prediction/
    configs/
    raw_logs/
      full_fresh_runs/
      single_site_interventions/
    processed/
      sample_table.parquet
      train.csv
      val.csv
      test.csv
    analysis/
      notebooks/
      figures/
    reports/
```

---

# 10. 分析流程：如何找到最有效的 proxy

## 10.1 单变量分析

对于每一个候选 signal：

- 画 signal vs unsafe rate
- 画 signal vs final logits KL
- 画 signal vs accepted_iter_shift
- 分 layer / token_state 分桶画图

### 目的
看是否存在：

- 强单调性
- 明显阈值
- 某些层特有的行为模式

例如：

- `top-k overlap > 0.875` 在浅层几乎都 safe
- 但在深层仍有明显 unsafe

这会直接提示需要 layer-aware policy。

---

## 10.2 排序能力评估

对每个候选 signal 计算：

- ROC-AUC（预测 safe/unsafe）
- PR-AUC（尤其看 unsafe detection）
- Spearman correlation（对 final_logits_kl）
- calibration curve

### 重点
真正要看的是：

> 在固定很低 false-negative rate 的前提下，哪个 proxy 能留下最多可 reuse 样本。

因为最危险的是：

- 预测 `safe`
- 但实际上 `unsafe`

---

## 10.3 小组合分析

先用简单、可解释的模型：

- Logistic regression
- Shallow decision tree
- Rule list

输入建议从这些开始：

- layer id
- token type
- gate cosine
- top-k overlap
- routing weight cosine
- token margin

### 为什么不用复杂黑盒
因为最终需要的是：

- 可解释阈值
- 易于系统实现
- 能转化为推理规则

---

## 10.4 推荐的风险分数形式

最终判据可写成一个简单 score：

\[
R_{t,l,i} = f(
\text{layer},\ 
\text{token\_state},\ 
d_{gate},\ 
overlap_{topk},\ 
d_{w},\ 
margin
)
\]

如果：

\[
R_{t,l,i} < \tau
\]

则 `reuse`，否则 `refresh`。

其中 `f` 可以是：

- 线性函数
- 逻辑回归
- 人工规则

后期可压缩为静态阈值表。

---

# 11. 关键消融实验设计

## Ablation 1：哪类信号最能预测风险

比较：

- token-state only
- hidden only
- routing only
- hidden + routing
- routing + layer
- routing + layer + confidence

指标：

- ROC-AUC
- PR-AUC
- FNR under fixed threshold
- safe precision
- 可 reuse recall

### 目标
回答：

> 是 hidden 更重要，还是 routing 更重要，还是必须二者结合？

---

## Ablation 2：layer-wise 风险结构

按层比较：

- 浅层
- 中层
- 深层
- 敏感层

### 目标
回答：

- 是否存在 “浅层可激进复用 / 深层需保守刷新” 的模式？
- 是否应强制某些层永远 fresh？

---

## Ablation 3：不同 token 组的风险结构

比较：

- MASK
- newly decoded
- stable decoded
- highly stable decoded

### 目标
回答：

- 哪类 token 最适合 selective recompute？
- 是否需要 group-specific threshold？

---

## Ablation 4：单点安全是否能迁移到组合策略

当单点 proxy 已筛出后，再测：

- 单层多 token 同时 reuse
- 多层单 token reuse
- 多层多 token reuse

### 目标
验证：

> 单点判据是否在组合情况下仍具有预测能力。

这一步很关键，因为误差传播可能导致“单点 safe ≠ 组合 safe”。

---

## Ablation 5：per-token vs sentinel-triggered

比较三种 refresh granularity：

- per-token refresh
- per-layer sentinel-triggered refresh
- layer-boundary refresh

### 目标
回答：

- 哪种最稳？
- 哪种工程开销最低？
- 哪种最适合实际部署？

---

# 12. Sentinel 设计：如何借鉴 Elastic-Cache 的精神

## 12.1 为什么需要 sentinel

逐 token 判断虽然最细，但有三个问题：

- 开销大
- noisy
- 实现复杂

因此应尝试用少量 token 作为 **sentinel**，代表一整层或一类 token 的 stale risk。

---

## 12.2 候选 sentinel

建议优先选：

- 当前 block 内 highest-confidence stable tokens
- 连续多轮 top-1 不变的 tokens
- margin 最大的 stable tokens

这些 token 最有可能在跨迭代中最稳定，因此适合作为保守监测对象。

---

## 12.3 sentinel 触发规则示例

例如：

若某层的所有 sentinel 都满足：

- gate cosine > 0.995
- top-k overlap >= 0.875
- token margin > m

则该层 stable decoded tokens 全部 `reuse`；  
否则该层 `refresh`。

这类规则非常接近 Elastic-Cache 的方法论：  
不是全量监测，而是用少量保守 proxy 触发刷新。

---

# 13. 从 Characterization 到真正 Policy 的转化方式

当你筛出 proxy 后，下一步是把它变成可执行策略。

## 13.1 策略 S1：Layer-aware routed refresh

规则：

- gate：总是 fresh
- shared：总是 fresh
- routed：仅对 stable decoded token 允许 reuse
- 深层或敏感层强制 fresh

### 优点
- 稳
- 简单
- 易实现

---

## 13.2 策略 S2：Sentinel-triggered refresh

规则：

- 每层选 1–4 个 sentinel
- 计算其 gate/routing drift
- 若 drift 低，则 stable tokens routed reuse
- 否则 refresh

### 优点
- 很像 Elastic-Cache
- 成本低
- 适合动态刷新

---

## 13.3 策略 S3：Layer-boundary refresh

规则：

- 定义边界层 `l*`
- `l < l*`：允许 reuse
- `l >= l*`：统一 fresh

边界层可由 proxy 动态决定，也可以离线固定。

### 优点
- 故事 clean
- 实现成本低
- 易于形成论文式系统方案

---

# 14. 推荐的最小可行实验（MVP）

如果资源有限，建议先做这个版本。

## 14.1 设置
- batch = 8
- heterogeneous prompts
- temperature = 0.7
- no-cache path
- 只看 stable decoded token
- 层：L3, L10, L17, L18
- step：2, 6, 10, 14

## 14.2 候选信号
先只采：

- hidden cosine drift
- gate cosine drift
- top-k overlap
- routing weight cosine
- token margin
- stable_len
- layer id
- step id

## 14.3 单点干预
只做：

- `reuse_prev_routed_only`
- 其它全部 fresh

## 14.4 标签
记录：

- final logits KL
- final top-1 changed
- accepted_iter_shift

定义：

- unsafe if `top1_changed == True`
- or `accepted_iter_shift > 0`
- or `final_logits_kl > ε`

## 14.5 首个 proxy 规则
先试：

- logistic regression
- 或人工规则：

```text
if layer <= L*
and token_state == stable_decoded
and gate_cos_prev >= a
and gate_topk_overlap_prev >= b
and token_margin >= c:
    reuse
else:
    refresh
```

---

# 15. 结果判断标准

## 15.1 Characterization 成功标准

若能得到：

1. 一个或一组明显优于 token-state heuristic 的 proxy
2. 能解释 layer-wise 风险差异
3. 能在低 FNR 下保留大量 safe reuse 样本

则阶段 I 成功。

---

## 15.2 Policy 成功标准

将 proxy 转为 selective recompute 策略后，若能满足：

- 正确性明显优于旧 heuristic
- speedup 仍保留
- forward 次数不显著恶化
- 规则开销低、实现可行

则阶段 II 成功。

---

# 16. 预期图表清单

建议输出以下图表，供 AI 和研究讨论直接使用。

## 16.1 Characterization 图
- `top-k overlap vs unsafe rate`
- `gate cosine vs unsafe rate`
- `hidden cosine vs unsafe rate`
- 分 layer 的 violin / histogram
- 分 token_state 的 unsafe rate 柱状图
- `final_logits_kl` 的层级热图
- `accepted_iter_shift` 的 step-layer 热图

## 16.2 Proxy 排名图
- 单变量 ROC-AUC 排名
- 单变量 PR-AUC 排名
- 固定 FNR 下的 reusable recall 排名

## 16.3 Policy 图
- exact match vs speedup
- logits KL vs reused ratio
- accepted_iter_shift vs reused ratio
- layer-boundary policy 的 Pareto 曲线

---

# 17. 推荐的工程拆分

## 17.1 数据采集模块
功能：

- 运行 full fresh baseline
- 导出 per-step/per-layer/per-token 日志
- 支持单点 counterfactual intervention

## 17.2 特征提取模块
功能：

- 从日志中提取 hidden / gate / routing / token-state 特征
- 生成训练表

## 17.3 风险分析模块
功能：

- 计算 safe/unsafe 标签
- 做单变量分析
- 做逻辑回归 / 小树模型
- 导出阈值建议

## 17.4 策略执行模块
功能：

- 将阈值规则植入运行时
- 实现 per-token / sentinel / layer-boundary refresh

---

# 18. 给后续 AI / 代码代理的明确任务指令

下面这段可直接交给代码代理。

## 18.1 第一阶段任务
1. 实现 full fresh run logging  
2. 实现 single-site intervention framework  
3. 导出 `(step, layer, token)` 级别样本表  
4. 计算候选 proxy 特征  
5. 计算 risk labels  
6. 输出 proxy 排名和图表

## 18.2 第二阶段任务
1. 选取最优 proxy 组合  
2. 将其压缩成简单 threshold rule  
3. 实现三种 policy：
   - per-token
   - sentinel-triggered
   - layer-boundary
4. 比较速度、质量、forward 次数、reuse ratio

---

# 19. 最核心的研究结论模板

若实验顺利，最终应该形成类似这样的结论：

> 我们发现，token 是否 stable 不能充分预测 routed experts output 的 stale risk；  
> 相比之下，基于 gate/routing drift 的信号，尤其是 `top-k overlap + gate cosine + layer id`，能够更有效地预测 skip risk。  
> 同时，skip risk 具有明显的 layer-wise 非均匀性：浅层更适合 reuse，深层和敏感层需要更保守的 refresh。  
> 基于此，我们提出一种 layer-aware / sentinel-triggered 的 MoE selective recompute 机制，在保持正确性的同时减少 routed expert 的重复计算。

---

# 20. 一句话总结

本项目的本质，不是在找“哪个 token 更稳定”，而是在建立一个：

> **MoE routed-path stale-risk prediction framework**

它用反事实单点干预来构造真值，用 hidden/gate/routing 等低成本信号来筛选 proxy，最终把 selective recompute 从粗暴 heuristic 变成可解释、可验证、可部署的动态刷新策略。

---

# 附录 A：建议的风险标签阈值模板

下面是一个初始模板，具体数值需后续校准：

- `safe`
  - final_top1_changed = False
  - accepted_iter_shift = 0
  - final_logits_kl < 1e-3

- `borderline`
  - final_top1_changed = False
  - accepted_iter_shift <= 1
  - 1e-3 <= final_logits_kl < 1e-2

- `unsafe`
  - final_top1_changed = True
  - 或 accepted_iter_shift > 1
  - 或 final_logits_kl >= 1e-2

---

# 附录 B：建议的首个规则模板

```text
For stable decoded tokens only:

if layer in SAFE_REUSE_LAYERS
and gate_cos_prev >= A
and topk_overlap_prev >= B
and routing_weight_cos_prev >= C
and token_margin >= D:
    reuse previous routed output
else:
    refresh routed output
```

后续可扩展为：

- sentinel-triggered
- layer-boundary
- dynamic thresholds by step group

---

# 附录 C：建议的首批研究问题列表

1. hidden drift 和 routing drift，谁更能预测 skip risk？
2. top-k overlap 是否已经足够好，还是必须结合 gate weight drift？
3. layer id 是否必须进入判据？
4. MASK token 是否需要和 stable decoded token 完全分开建模？
5. sentinel token 是否能代表整个 layer 的 refresh 需求？
6. 单点 safe 的 proxy 能否迁移到组合 reuse 场景？
7. 最终最 practical 的粒度是 per-token、per-layer 还是 boundary-based？

---

# 结束语

这份文档的目标不是直接给出某个最终优化方案，而是建立一套 **统一、可执行、可扩展的研究管线**。  
一旦这套管线跑通，后续无论你是继续沿着 Elastic-Cache 风格的 refresh 机制走，还是结合 TEAM / routing consistency / memory-bound 分析走，都将拥有坚实的实验基础。
