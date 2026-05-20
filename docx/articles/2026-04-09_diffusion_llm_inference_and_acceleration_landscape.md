# diffusion LLM 的推理流程与加速技术综述

## 1. 这篇文档要回答什么

这篇文档聚焦三个问题：

1. 什么是 diffusion LLM（dLLM）；
2. 它的推理流程与普通自回归（AR）模型到底差在哪里；
3. 当前本地可见的主流实践里，dLLM 的加速技术主要有哪些，后续更值得优先优化什么。

本文基于本地 `lib_cite/dInfer`、`lib_cite/sglang` 代码与文档梳理，并结合官方一手资料校对当前生态状态。

## 2. 什么是 diffusion LLM

diffusion LLM 可以理解成“面向离散 token 序列的迭代去噪生成模型”。

与 AR 模型按 `t=1,2,3...` 逐 token 只向右生成不同，dLLM 通常会：

- 先把待生成区域整体初始化为 `mask token`；
- 再反复执行若干轮 transformer forward；
- 每一轮根据模型当前对所有 masked 位置的预测置信度，选择一部分 token 一次性从 `mask -> token`；
- 直到一个 block 或整个生成区被完全去噪。

从本地代码看，这个逻辑在 `dInfer` 中非常清晰：

- `TokenArray` 会把生成区初始化成全 `mask_id`；
- `BlockWiseDiffusionLLM.generate()` 按 block 迭代；
- `ThresholdParallelDecoder` 根据置信度阈值并行解码多个 token；
- `BlockDiffusionLLM` 进一步只计算当前 block 所需上下文，减少“未来全 mask token”带来的无效计算。

## 3. dLLM 的主推理流程

### 3.1 基础流程

一个典型 dLLM 推理流程可以抽象成：

1. 输入 prompt；
2. 在生成区补满 `mask_id`；
3. 将生成区切成一个或多个 block；
4. 对当前 block 做一次 forward，得到该 block 各位置对词表的 logits；
5. 依据某种并行解码规则，选出一批最该先确定的 token；
6. 把这些位置从 `mask` 改成真实 token；
7. 如果 block 还没解完，则继续下一轮 forward；
8. 当前 block 解完后，切换到下一个 block；
9. 全部 block 解完或遇到 EOS 后结束。

### 3.2 在 dInfer 中的落地

`dInfer` 的实现把这个流程拆成四层：

- model
- diffusion iteration manager
- decoder
- KV-cache manager

这也是它 README 里明确写出的框架分层。

从代码结构上看：

- `BlockWiseDiffusionLLM` 负责 block-by-block 外层流程；
- `BaseDiffusionIteration` / `BlockDiffusionIteration` 负责一次 diffusion step 的 forward 与 cache 协调；
- `ThresholdParallelDecoder` / `HierarchyDecoder` / `CreditThresholdParallelDecoder` 负责“本轮究竟解哪些 token”；
- `KVCache` 及其变体负责 prefix/dual/window 等 cache 复用策略。

### 3.3 在 SGLang 中的落地

SGLang 已经把 text dLLM 做成 `srt.dllm` 子系统，而不是简单外接脚本。

它的基本做法是：

- 由 `DllmConfig` 按模型架构选择 `mask_id` 和默认 `block_size`；
- 由 `tp_worker` 初始化具体 dLLM algorithm；
- 由 scheduler 的 `SchedulerDllmMixin` 管理 dLLM 请求的 prefill/decode phase；
- 每次 forward 后由算法对象直接修改 block 内 token，再决定是否继续下一轮。

当前本地代码里，SGLang 原生提供两类算法：

- `LowConfidence`
- `JointThreshold`

其中 `JointThreshold` 明确分成：

- M2T：mask-to-token
- T2T：token-to-token editing

这意味着它不是“只会 unmask”，而是允许在去噪结束后做有限编辑修正。

## 4. dLLM 与普通 AR 模型的核心区别

## 4.1 生成顺序不同

AR：

- 严格左到右；
- 每步只新增 1 个 token；
- 单步依赖前缀，天然因果。

dLLM：

- 一轮可以同时决定多个 token；
- 生成顺序不是固定时间轴，而是“按置信度逐步凝固”；
- 同一 block 内多个位置可以并行被确定。

## 4.2 注意力结构不同

AR 的默认注意力是 causal attention。

dLLM 更接近 full attention 或 block-constrained attention。本地 `modeling_llada.py` 和 `modeling_fused_lladamoe.py` 都明确把 diffusion 模式下的 causal 行为关掉，改成非因果 / full attention 默认。

这点非常关键：

- AR 的 KV cache 逻辑天然成立，因为 token 不会改写过去；
- dLLM 在去噪中会反复改写当前 block，因此 cache 不再是“只追加、不回头”的简单 append 语义。

## 4.3 KV cache 语义不同

AR 的 KV cache 是 append-only。

dLLM 的 cache 往往需要：

- 局部重算；
- 指定 replace_position 回写；
- 跨 block 更新；
- 有时还要在某些 step 强制整段刷新。

所以 dLLM 的 cache 管理比 AR 更像“增量近似 + 周期校正”，而不是单纯历史复用。

## 4.4 延迟构成不同

AR 的主要目标通常是：

- 单 token decode latency 最小化；
- prefix cache 命中最大化；
- batch scheduler 尽量保持 GPU 饱和。

dLLM 的核心目标变成：

- 每轮能解掉尽可能多的 token；
- 每轮 forward 不要算太多无效位置；
- cache 刷新不要过频也不能失真太大；
- block 切分、窗口大小、阈值调度三者要共同最优。

所以 dLLM 优化更像“step reduction + per-step cost reduction”的乘积优化。

## 5. 当前主流 dLLM 加速技术分类

这里把当前可见实践分成六层。

### 5.1 并行解码算法

这是 dLLM 最核心、也最特有的一层。

#### A. 置信度阈值并行解码

代表：

- `dInfer` 的 `ThresholdParallelDecoder`
- SGLang 的 `LowConfidence`

原理：

- 对当前 block 每个 masked 位置取 argmax token；
- 计算该 token 的概率 / 置信度；
- 一次性接受所有高于阈值的位置；
- 如果一轮没有任何位置超过阈值，至少强制接受 1 个位置，避免停滞。

优点：

- 实现简单；
- 质量/速度权衡直观；
- 非常适合工程化和批处理。

缺点：

- 阈值固定时，早期可能过保守，后期可能过激进；
- 无法显式处理“去噪结束后还需要小范围修正”的情况。

#### B. JointThreshold / M2T + T2T 编辑

代表：

- SGLang `JointThreshold`

原理：

- 先做 mask-to-token；
- 当 mask 清空后，允许 token-to-token 继续编辑若干步；
- 用 `edit_threshold`、`max_post_edit_steps` 和重复惩罚控制编辑阶段。

优点：

- 兼顾“先快速成型”和“后精修”；
- 对重复和局部错误更友好。

缺点：

- 需要更多 step；
- 工程上要更谨慎地处理终止条件和额外 forward。

#### C. 层级式 / 分段式并行解码

代表：

- `dInfer` 的 `HierarchyDecoder`

原理：

- 不只是按全局 top-k 或单阈值，而是按 masked segment / 局部结构选位置；
- 有的实现会强制每个 masked 段至少推进 1 个 token；
- 从而避免某一部分过快凝固、另一部分长期停滞。

它更像“平衡各局部区域去噪速度”的策略。

### 5.2 block 化与半自回归化

这是当前 text dLLM 实践非常重要的一层。

#### A. Block-wise diffusion

代表：

- `BlockWiseDiffusionLLM`
- `BlockDiffusionLLM`

原理：

- 把长生成拆成多个 block；
- 每次只对当前 block 做 diffusion；
- 前面 block 视为稳定上下文，后面 block 暂不真正计算。

意义：

- 避免整个生成区全长 full attention；
- 把问题从“全序列迭代去噪”压缩成“局部块内迭代去噪”；
- 非常接近工程上能落地的 dLLM 形态。

#### B. Semi-autoregressive dLLM

本质上就是 block 级别的“先前后、块内并行”。

它既保留了 dLLM 的并行去噪能力，又借用了 AR 的前缀稳定性，因此是当前最现实的 serving 形态之一。

### 5.3 cache 近似与选择性重算

这是 dLLM 第二关键层。

#### A. Prefix cache

原理：

- 假设 block 前缀已稳定；
- 只重算当前 block 及其之后的局部输入；
- 复用前缀 KV。

适用：

- 当前 block 改动主要不影响远前缀表示；
- 实现成本较低。

#### B. Dual cache

原理：

- 对当前 block 单独维护替换位置；
- 允许对局部 block 做 inplace cache rewrite；
- 比 prefix cache 更贴近“当前块持续被改写”的真实语义。

#### C. Vicinity cache / windowed cache

代表：

- `VicinityCacheDiffusionLLM`

原理：

- 每轮不是整段重算，也不是只看当前 block；
- 而是只刷新当前 block 左右一个窗口 `prefix_look` / `after_look`；
- 相当于只在局部感受野内做较精细的状态校正。

这是一个非常值得重视的工程折中：

- 比全量刷新便宜很多；
- 又比纯 prefix 近似更稳。

#### D. 周期性全量刷新

原理：

- 大多数 step 用近似 cache；
- 每隔若干 step 做一次 full refresh；
- 用来修正近似累计误差。

这个思路在 `generate_cache.py` 里体现得很直接。

### 5.4 连续嵌入平滑（Iteration Smoothing）

代表：

- `IterationSmoothDiffusionLLM`
- `IterSmoothWithVicinityCacheDiffusionLLM`

原理：

- 对仍然不确定的位置，不只用离散 token embedding；
- 还用当前 logits 对 embedding matrix 做 soft mixture；
- 再按 `iter_cont_weight` 把连续 embedding 混入下一轮输入。

直觉上，这相当于：

- 当 token 还没最终定下来时，不要过早硬离散化；
- 让模型在若干步内保留“软假设”。

它的潜在作用是：

- 降低早期误判的硬承诺；
- 让后续 step 更平滑；
- 在相同步数下提高最终质量，或者在相同质量下减少 step。

这类方法是 dLLM 很有代表性的“模型算法协同优化”。

### 5.5 运行时 / kernel / 并行层优化

这些并非 dLLM 独有，但现在已经是主流框架的标配加速层。

#### A. torch.compile + CUDA Graph

代表：

- `dInfer/decoding/diffusion_runner.py`

原理：

- 让重复 shape 的 diffusion step 进入编译图和 graph replay；
- 降低 Python / launch overhead。

因为 dLLM 会在相同 block size 上做很多轮重复 forward，所以这一层比在普通 AR decode 中更容易吃到收益。

#### B. 动态 batching

代表：

- `BlockDiffusionLLM.dynamic_batching_generate`
- SGLang 的 dLLM scheduler mixin

原理：

- 不同请求处于不同 block / 不同 cache capacity 时，按可兼容长度重新分批；
- 让还活跃的序列继续填满 batch。

这对 dLLM 很重要，因为不同请求的“剩余 masked token 数”和“当前 block 位置”差异更大。

#### C. TP / EP / SP / custom all-reduce

代表：

- dInfer 接入 SGLang distributed utilities
- SGLang 自身的 TP/DP/EP 基础设施

原理：

- 用成熟的张量并行、专家并行、序列并行和通信优化来支撑 dLLM 的重复 forward。

#### D. MoE 专项优化

代表：

- LLaDA2-flash / LLaDA-MoE
- fused MoE kernels
- expert-parallel runtime

对于大规模 MoE dLLM，MoE kernel 与通信往往是第一瓶颈之一。

### 5.6 量化

当前主流实践里，量化已经不是可选项，而是后续大模型 dLLM serving 的必备层。

从本地仓库和官方文档看：

- dInfer README 已经明确支持量化版 LLaDA2-mini / flash；
- SGLang 的通用 serving 体系支持 FP8 / INT8 等量化路径。

注意：

- dLLM 的量化不能只看单次 forward 精度；
- 还要看“多轮迭代误差是否会累积放大”。

所以 dLLM 的量化评估应更关注：

- step 数是否变多；
- 阈值策略是否需要重调；
- 最终质量是否在相同步数下下降。

## 6. 当前框架格局的一个务实判断

基于这次本地与官方资料核对，我对当前 text dLLM serving 生态的判断是：

### 6.1 真正“把 text dLLM 当一等公民”的框架，目前最值得看的是 dInfer 和 SGLang

原因：

- `dInfer` 更像一个针对 dLLM 的算法实验场 + 高性能推理框架；
- `SGLang` 则已经把 dLLM 算法接进正式 scheduler / runtime / serving API。

### 6.2 通用 AR serving 框架目前还不是 text dLLM 的原生主战场

更准确地说：

- 有些通用 backend 可以被 dLLM 框架借来承载部分模型或部分 kernel；
- 但“原生 text dLLM 算法 + scheduler + cache 语义”的完整闭环，目前主要还是 dInfer / SGLang 这条线更清晰。

### 6.3 当前最新实践并不是只卷 kernel，而是三层联动

当前更成熟的方向是：

1. 解码算法减少 step；
2. cache 策略减少每 step 成本；
3. runtime 把剩余成本压到接近硬件上限。

单纯只做 kernel，而不改 step 数和 cache 语义，收益通常不够大。

## 7. 我建议你后续优先优化的方向

如果你的目标是做出有区分度、且更容易形成论文或系统亮点的工作，我建议优先级如下。

### 优先级 1：围绕“每轮能解多少 token”做算法-系统协同

具体可以做：

- 动态阈值调度，而不是固定 threshold；
- confidence + locality 联合决策，而不是纯全局阈值；
- mask-to-token 和 token-to-token 的混合策略；
- 基于 step map 的自适应 block 内推进策略。

原因：

- step reduction 往往是 dLLM 最大杠杆；
- 这类优化同时影响质量和速度，更有研究价值。

### 优先级 2：围绕 cache 误差与重算边界做选择性刷新

具体可以做：

- 从 prefix / dual / vicinity cache 继续向“自适应窗口刷新”推进；
- 用置信度、block 边界变化量、hidden drift 来决定是否 full refresh；
- 研究 cross-block update 的必要条件，而不是固定更新。

原因：

- 这层直接决定 per-step cost；
- 也是 dLLM 与 AR 最大的系统差异点之一。

### 优先级 3：把 dLLM scheduler 真正做成一等公民

具体可以做：

- 面向 dLLM 的 batch formation；
- 按 block stage / cache length / mask density 分层调度；
- 把“当前剩余 mask 数”纳入优先级；
- 减少不同请求在多轮去噪中的尾部浪费。

如果你后续想走更偏系统的路线，这一层很关键。

### 优先级 4：在大 MoE dLLM 上做量化与 kernel 联合优化

这条线更偏工程性能，但也很强：

- FP8 / INT8 在多轮去噪中的误差传递；
- MoE expert dispatch 与 block diffusion 的耦合；
- compiled graph + MoE + blockwise cache 的联合 shape 设计。

如果目标是高吞吐 serving，这一层收益会非常现实。

## 8. 一个明确建议

如果你现在只能选一个主攻方向，我建议：

**先做“自适应 cache 刷新 + 动态并行解码阈值”的联动优化。**

原因是：

- 它同时触及 step reduction 和 per-step cost reduction；
- 与 `dInfer` 现有 prefix / dual / vicinity / iteration smoothing 代码天然衔接；
- 比只做 kernel 更容易做出“方法 + 系统实现 + 实验曲线”三位一体的成果；
- 也比单纯复刻 SGLang `LowConfidence` / `JointThreshold` 更容易形成你自己的新点。

更具体地说，可以把下一阶段目标定成：

- 预测“这一轮哪些位置该解、哪些 cache 区域该刷新、刷新范围多大”；
- 用统一控制器联合决定 decode 和 refresh；
- 再用 step 数、TPS、质量退化三条线去评估。

这会比把这些决策拆散做，更像一个完整研究问题。
