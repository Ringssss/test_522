# MoE 系统论文调研说明书（近两年系统会议，含一级/二级文章）

> 范围说明  
> - 会议优先覆盖：OSDI / SOSP / ASPLOS / EuroSys / SC / PPoPP / ATC  
> - 时间窗口：以 2024–2026 已公开论文为主，其中 2026 单独标注  
> - 一级文章：直接面向 **MoE 推理 / 部署 / 执行**  
> - 二级文章：主要面向 **训练、通信、容错、重布局**，但显式利用了 MoE 的结构性质，且对推理具有迁移价值

---

## 一、总览：一级文章与二级文章

### 一级文章（直接做 MoE 推理/部署/执行）
1. **Klotski** — ASPLOS 2025  
2. **Samoyeds** — EuroSys 2025  
3. **KTransformers** — SOSP 2025  
4. **Diff-MoE** — SC 2025  
5. **MoE-APEX** — ASPLOS 2026  
6. **EARTH** — ASPLOS 2026  
7. **CoServe** — ASPLOS 2025（边界论文，CoE/非标准 transformer-MoE，但系统问题高度同构）

### 二级文章（训练/通信/容错/布局，但可迁移到推理）
8. **ScheMoE** — EuroSys 2024  
9. **FSMoE** — ASPLOS 2025  
10. **MoC-System** — ASPLOS 2025  
11. **PopFetcher** — USENIX ATC 2025  
12. **X-MoE** — SC 2025  
13. **LAER-MoE** — ASPLOS 2026  

---

# 二、一级文章逐篇完整解读

## 1. Klotski — ASPLOS 2025
**Klotski: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline**

### 背景
MoE 让模型参数量显著增加，但每个 token 只激活少数几个 expert，因此理论计算量没有和参数量同步爆炸。真正的问题变成：**专家参数太大，显存装不下，只能频繁从 CPU 内存甚至更低层存储加载**。  
对于 dense LLM，多 batch 往往意味着更长的计算时间，可以覆盖更多权重加载；但在 MoE 中，batch 变大也会带来更多被激活的 expert，导致加载开销同步上升，于是 pipeline bubble 变得更加严重。

### 所解决的问题
Klotski 关注的不是“如何做一般的 offload”，而是：  
**当 expert 的 I/O 时间往往比 expert 计算时间还长时，怎样通过多 batch 组织、执行顺序优化和预取，让 I/O 与 compute 尽可能重叠，从而减少推理中的空泡？**

### 所用的技术
Klotski 的核心是 **expert-aware multi-batch pipeline**。  
它并不是按请求自然顺序执行，而是把多个 batch 合在一起看，根据 expert 激活模式来决定排程。论文将 expert 粗分为：
- **hot experts**：高频命中，计算时间更长，更值得常驻或优先保留
- **cold experts**：低频命中，计算时间短，但每次 miss 的加载代价大

围绕这个观察，Klotski 提出：
1. **Constraint-sensitive I/O-compute planner**  
   - 根据硬件带宽、计算速度、模型层结构和不同 batch 下的激活模式，求更优的 I/O 与 compute 交错方式
2. **Correlation-aware expert prefetcher**  
   - 利用不同 expert 激活之间的相关性，提前预取更可能被下一个 batch 或下一层命中的 expert

### 达成的效果
论文报告最高可达到 **85.12× throughput improvement**。  
这个收益的来源不是单一 kernel 提升，而是吃掉了原本非常严重的 I/O 等待和 pipeline bubble。

### 什么时候应该想到这篇论文
当你看到这些症状时，应优先想到 Klotski：
- GPU 利用率不高，但 I/O 非常繁忙
- batch 增大后吞吐没有提升，甚至更差
- 少量热门 expert 经常命中，而冷门 expert 偶发 miss 拉长尾
- 本层算完后，下一层 expert 还没加载到位

### 对推理加速的启发
- MoE 推理优化应该围绕 **expert 激活分布 + I/O/compute overlap** 进行全局调度
- 可以做 **热/冷 expert 分层缓存**
- 可以做 **跨 batch 的统一调度**
- 可以把 **expert correlation** 用到 prefetch 中

---

## 2. Samoyeds — EuroSys 2025
**Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores**

### 背景
许多 MoE 系统论文主要解决“expert 放不下”和“如何调度 expert”，但 Samoyeds 指出：  
**即使 expert 已经在设备上，MoE 的实际计算仍可能低效。**  
尤其在硬件开始提供 Sparse Tensor Cores 时，如果系统只利用参数稀疏，而没有同时利用激活稀疏，就很难真正压榨稀疏硬件的能力。

### 所解决的问题
**如何在 MoE 执行阶段，同时利用 weight sparsity 和 activation sparsity，让稀疏硬件真正发挥作用？**

### 所用的技术
Samoyeds 的主要工作在于执行底座：
1. 同时支持 **weight sparsity** 和 **activation sparsity**
2. 为 MoE 设计专门的 **sparse data format**
3. 提出针对 Sparse Tensor Cores 的 **sparse-sparse matrix multiplication kernel**
4. 做系统级优化，让 memory access、数据布局、kernel 调用都更适合双侧结构化稀疏

### 达成的效果
论文报告：
- kernel level 最多 **1.99×**
- model level 最多 **1.58×**
- 平均最大 batch size 提升 **4.41×**

### 什么时候应该想到这篇论文
- expert 已经能常驻显存，I/O 不是主瓶颈，但算得还是慢
- 稀疏硬件已经有了，MoE 仍然跑不快
- batch 稍微一放大就受限于内存与 kernel inefficiency
- 你已经做了缓存/调度，仍然怀疑 compute-side 低效

### 对推理加速的启发
- MoE 优化不只在 routing / loading，也在 **kernel / sparse execution**
- 如果你做编译器、kernel、runtime，这篇是关键参考
- 对结构化稀疏专家、剪枝专家尤其有价值

---

## 3. KTransformers — SOSP 2025
**KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models**

### 背景
MoE 每次只激活少量 expert，因此天然适合把部分参数和计算迁移到 CPU/DRAM 中。  
但已有 hybrid 方案常见问题是：
- CPU 自身算得不够快
- CPU/GPU 同步和切换开销太高

于是“理论上可以 hybrid”，但实际效果不理想。

### 所解决的问题
**怎样把 CPU 从“慢速参数仓库”提升为“真正参与 MoE 推理的计算层”，并让 CPU/GPU 协同不会被同步开销毁掉？**

### 所用的技术
1. **AMX-specialized kernels**  
   - 面向现代 CPU 的 AMX 指令优化 expert 相关计算
2. **Asynchronous CPU-GPU task scheduling**  
   - 减少 CPU/GPU 间的等待与切换开销
3. 硬件映射思路  
   - GPU 负责高并行、高带宽路径  
   - CPU 更积极承担一部分稀疏 expert 计算

### 达成的效果
论文报告：
- prefill 提升 **4.62–19.74×**
- decoding 提升 **1.25–4.09×**

### 什么时候应该想到这篇论文
- 只有单卡/少量 GPU，但 CPU 内存很大
- 本地部署、低并发、长上下文
- 想让 CPU 真正参与计算，而不是只做 offload
- 你愿意用更复杂 runtime 换更低部署成本

### 对推理加速的启发
- attention path 和 expert path 可以映射到不同硬件
- 热 expert 放 GPU，冷 expert 放 CPU
- CPU 可以作为主动计算层，而不是被动内存层

---

## 4. Diff-MoE — SC 2025
**Diff-MoE: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching**

### 背景
batch 做大以后，MoE 推理不一定更高效。  
原因是：更多请求会激活更多 expert，从而带来更重的缓存和通信压力。  
Diff-MoE 指出，现有系统没有充分利用 expert 访问中的 locality，因此 batch 场景下 cache policy 效果不佳。

### 所解决的问题
**如何根据 expert 的不同复用模式设计差异化的缓存层次和优先级策略，而不是用一套统一 cache policy？**

### 所用的技术
论文识别了两种 locality：
- **global locality**：跨更长时间、更多 batch 的长期热门 expert
- **temporal locality**：短时间 burst 中反复命中的 expert

围绕这两个性质，提出 **priority-driven differential cache hierarchy**：
- 对长期热门 expert 和短期 burst expert 采用不同管理策略
- 通过优先级驱动的分层缓存，适应 batched inference 的真实 expert reuse 模式

### 达成的效果
核心收益来自 batched 场景下的缓存命中率和整体执行效率提升，论文重点展示了这种 differential cache hierarchy 能有效缓解批推理瓶颈。

### 什么时候应该想到这篇论文
- 单请求还行，一做 batch 就掉速
- 存在少量长期热门 expert 和一批 burst 热门 expert
- 你已经做了 cache，但 miss pattern 明显有时间结构
- 你怀疑问题在 cache policy，而不是“有没有 cache”

### 对推理加速的启发
- session-aware / prefix-aware 的 expert cache
- 热 expert 长驻，burst expert 短驻
- cache policy 从 LFU/LRU 升级为 workload-aware priority

---

## 5. MoE-APEX — ASPLOS 2026
**MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading**

### 背景
在边缘设备或显存受限场景里，MoE 依然面临 expert 太大、放不下的问题。  
已有 offload 系统通常只做二元决策：expert 在不在高层内存。  
MoE-APEX 提出更细的思路：**expert 不只分“在/不在”，还可以分“以什么精度在”。**

### 所解决的问题
**如何在 memory-constrained environment 下，以更低代价加载 offloaded experts，并尽量不影响模型效果？**

### 所用的技术
三层级自适应管理：
1. **Token-level dynamic expert loading**
2. **Layer-level adaptive prefetching**
3. **Sequence-level cost-aware caching**

最大亮点是：
- 对 cache-miss expert 不一定总加载 full precision
- 可以根据 token 重要性、层敏感度、序列代价，决定使用更低精度版本

### 达成的效果
论文强调通过 adaptive precision management 映射 MoE 的自然层级结构，在边缘设备上显著改善 offloading 成本与推理效率。

### 什么时候应该想到这篇论文
- 设备内存/显存极其有限
- 冷 expert 很多，偶发 miss 代价大
- full-precision expert 加载不划算
- 不同层/不同 token 对精度敏感度不同

### 对推理加速的启发
- expert 可有多级状态：高精度驻留 / 低精度驻留 / miss 时低精度快速加载
- 精度层次可和缓存层次结合
- 对 edge/embedded MoE 推理特别重要

---

## 6. EARTH — ASPLOS 2026
**EARTH: An Efficient MoE Accelerator with Entropy-Aware Speculative Prefetch and Result Reuse**

### 背景
MoE 的 memory bottleneck 在 accelerator 层面尤其突出。  
如果 expert fetch 仍是主瓶颈，那么除了增加带宽，还可以思考：
- 能否根据 router 的不确定性进行 speculative prefetch
- 能否复用过去的结果，而不只是缓存参数

### 所解决的问题
**在路由结果存在不确定性的情况下，如何做高收益、低浪费的 speculative prefetch，并探索结果复用？**

### 所用的技术
1. **Entropy-aware speculative prefetch**
   - 根据路由熵决定预取是否值得做
2. **Result reuse**
   - 不只缓存参数，也尝试复用部分执行结果或模式
3. Accelerator-level orchestration
   - 让预取、执行与复用更紧密耦合

### 达成的效果
论文目标是实现更接近“memory 与 computation 完全重叠”的效果，并比简单预取显著更优。

### 什么时候应该想到这篇论文
- 普通 prefetch 效果有限，误取代价高
- router 有时稳定、有时不稳定
- 某些 prompt/模式反复触发相似 expert 组合
- 你想从“缓存参数”进一步走到“缓存结果”

### 对推理加速的启发
- 把 router uncertainty 正式纳入系统设计
- 统计路由熵，而不是只统计命中率
- 从参数缓存走向结果复用/模式复用

---

## 7. CoServe — ASPLOS 2025
**CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory**

### 背景
CoServe 处理的是 CoE（Collaboration-of-Experts），不是标准 transformer-MoE。  
但系统层面的问题非常接近：
- 专家很多
- 内存有限
- 请求依赖的 expert 组合不同
- expert switching 代价高

### 所解决的问题
**在有限内存下，如何为依赖不同 expert 组合的请求安排执行顺序和 expert 驻留，从而减少 switching 开销？**

### 所用的技术
1. **Dependency-aware request scheduling**
   - 按请求所需 expert 依赖关系安排执行顺序
2. **Dependency-aware expert management**
   - 按依赖关系管理不同 expert 的驻留与切换

### 达成的效果
核心收益来自减少 expert switching 开销，提高有限内存环境中的 expert 协作推理效率。

### 什么时候应该想到这篇论文
- 多个 expert model / 模块协同，而非单标准 MoE 层
- 请求依赖 expert 组合差异大
- 性能损失主要来自切换，而不是单次计算
- 多层内存结构中 switching thrashing 很严重

### 对推理加速的启发
- 可以按 expert dependency 对请求聚类
- scheduler 可显式最小化 expert set 变化
- 对多模型协作系统或 agent system 价值高

---

# 三、二级文章逐篇完整解读

## 8. ScheMoE — EuroSys 2024
**ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling**

### 背景
MoE 训练中 token 需要动态路由到不同 GPU 上的 expert，导致大量 all-to-all 通信。  
通信既动态又昂贵，严重拖累扩展效率。很多系统的 compute/comm overlap 不足，节点内与节点间带宽也没有被充分利用。

### 所解决的问题
**如何把 MoE 训练中的通信与计算任务显式抽象成 task，并做统一调度，从而提升整体扩展效率？**

### 所用的技术
1. **Generic scheduling framework**
   - 抽象 communication/computation tasks 并做统一调度
2. **Novel all-to-all collective**
   - 同时更好利用 intra-node 与 inter-node 带宽
3. **Extensible integration**
   - 方便接入新的 all-to-all 方案与压缩机制

### 达成的效果
在 32-GPU 集群上，相比 Tutel 和 FasterMoE 提升约 **9%–30%**。

### 什么时候应该想到这篇论文
- 多机扩展一上去就被 all-to-all 卡死
- 你要实验不同通信压缩方法，但系统不够可插拔
- 你感觉瓶颈在“整体流水和任务编排”

### 对推理加速的启发
- 推理也可以拆成 dispatch / load / compute / gather 多段 task graph
- 显式做 compute-comm overlap
- 多机 serving 中，token dispatch 与 expert gather 也适合分层调度

---

## 9. FSMoE — ASPLOS 2025
**FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models**

### 背景
MoE 训练性能受 routing、communication、expert compute、parallelism 共同影响。  
不同框架和不同 MoE 变种实现差异大，手工特化很难长期维护。

### 所解决的问题
**如何用统一抽象支持多种 MoE routing / 系统实现，并通过 online profiling 驱动调度，获得接近最优的训练执行？**

### 所用的技术
1. **Unified abstraction + online profiling**
2. **Communication-computation co-scheduling**
3. **Adaptive gradient partitioning + adaptive pipelining**

### 达成的效果
- 配置化实验最高约 **1.42×**
- 相对 DeepSpeed-MoE / Tutel 提升约 **1.18×–1.22×**
- 在真实模型上最高可到 **1.19×–3.01×**

### 什么时候应该想到这篇论文
- 你要支持很多不同的 MoE 实现
- 最优策略强依赖 routing 方式
- 不想手工为每个模型写调度逻辑
- 你想要通用 runtime，而不是单点优化

### 对推理加速的启发
- 推理 runtime 也可做在线 profiling
- 前端模型接口与后端执行策略解耦
- 用 profiling 决定 cache / prefetch / pipeline 策略

---

## 10. MoC-System — ASPLOS 2025
**MoC-System: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training**

### 背景
MoE 训练里，虽然每次只激活少量 expert，但模型总参数量极大。  
因此直接沿用 dense model checkpoint 方法，开销会非常高。

### 所解决的问题
**如何利用“MoE 每次只用少量 expert”这一性质，降低 checkpoint 的存储与性能开销？**

### 所用的技术
1. **Partial Experts Checkpointing (PEC)**
   - 只保存部分 experts
2. **Fully sharded checkpointing**
3. **Two-level asynchronous checkpointing management**

### 达成的效果
- checkpoint overhead 最多降低 **98.9%**
- 训练迭代加速最多 **5.12×**
- 准确率保持可比甚至略有提升

### 什么时候应该想到这篇论文
- expert 数量极大，但活跃度分布极不均衡
- 冷 expert 占资源很多
- 你想做部分物化、部分持久化或冷热分层

### 对推理加速的启发
- 并非所有 expert 都值得长期驻留或完整复制
- 可做 partial persistence / partial replication / partial materialization
- 冷热 expert 差异化管理可迁移到缓存和精度管理

---

## 11. PopFetcher — USENIX ATC 2025
**PopFetcher: Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch**

### 背景
PopFetcher 观察到：  
MoE 中的 expert selection 不是完全随机的，expert popularity 和 correlation 在相邻层、连续阶段中有稳定性和偏斜。

### 所解决的问题
**能否根据 expert popularity 的变化趋势，提前预取下一阶段更可能被需要的 expert，从而减少训练中的通信和等待？**

### 所用的技术
1. 识别 **skewed and correlated expert patterns**
2. 使用 **lightweight sliding-window** 预测 popularity
3. 在当前非-MoE 计算阶段提前预取下一层热门 expert
4. 利用 idle links 进行更有价值的传输

### 达成的效果
核心收益来自利用 popularity-guided 预取缓解训练中的 expert 通信与等待成本。

### 什么时候应该想到这篇论文
- expert 使用分布明显偏斜
- 连续 token / 相邻层 expert 模式有相关性
- 你系统里存在可利用的空闲 I/O / network 窗口
- miss penalty 很高，但不是完全不可预测

### 对推理加速的启发
- 会话级、序列级 expert popularity 预测
- prefix-aware 或 domain-aware 的预取
- 在 attention 或其他非 expert 计算阶段预拉 expert

---

## 12. X-MoE — SC 2025
**X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms**

### 背景
X-MoE 面向的是 DeepSeek-style、新型 expert-specialized MoE：
- 更细粒度 experts
- 更大的 top-k
- 更强的 specialization

这会导致新的系统瓶颈：
- activation memory overhead 更重
- A2A 通信冗余更多
- 跨平台 kernel 效率问题更突出

### 所解决的问题
**如何在 HPC 平台，尤其是非 NVIDIA 平台上，为新一代 expert-specialized MoE 提供可扩展训练系统？**

### 所用的技术
1. **Padding-free sparse MoE training pipeline**
2. **Hierarchical redundancy-bypassing dispatch**
3. **Hybrid parallelism with sequence-sharded MoE blocks**

### 达成的效果
- 在 Frontier、1024 个 AMD GPU 上训练到 **545B 参数**
- 同等预算下可训练模型规模提升约 **10×**
- 吞吐最高提升 **1.42×**

### 什么时候应该想到这篇论文
- 你研究的是 DeepSeek-style、大 top-k、细粒度 experts
- activation overhead 比参数更痛
- 通信冗余来自 dispatch 方式本身
- 平台不是传统单 NVIDIA 栈

### 对推理加速的启发
- padding-free execution 可直接迁到推理
- redundancy-bypassing dispatch 对大 top-k 推理很重要
- sequence-sharding 可启发长上下文推理中的 activation 管理

---

## 13. LAER-MoE — ASPLOS 2026
**LAER-MoE: Load-Adaptive Expert Re-layout for Efficient Mixture-of-Experts Training**

### 背景
动态 routing 往往导致热点 expert 过载。  
传统做法可能依赖 auxiliary loss、丢 token、固定复制 expert，但都不够灵活或代价较高。  
LAER-MoE 的核心思想是：**expert placement 不应该是静态常量。**

### 所解决的问题
**如何在训练期间根据负载动态重布局 expert，并控制随之而来的通信和恢复开销？**

### 所用的技术
1. **Fully Sharded Expert Parallel (FSEP)**
   - expert 参数完全按设备切分
   - 训练中按需恢复部分 expert
2. 更细粒度通信调度
3. **Load balancing planner**
   - 联合决定 expert re-layout 与 token routing

### 达成的效果
论文在 A100 集群上报告最高 **1.69×** 加速。

### 什么时候应该想到这篇论文
- 少数 expert 长期成为热点
- 静态 placement 适应不了实时流量
- 你想复制或迁移热门 expert，但不能代价太大
- 你想让 routing 与 placement 联合优化

### 对推理加速的启发
- 热 expert 可动态复制到更多 GPU
- 可做 load-adaptive expert placement
- 在多机 serving 中，可做热点时段临时 re-layout

---

# 四、问题到技术的反查索引

## 1. 如果你的主要症状是“显存装不下 expert，系统大量在等加载”
优先看：
- **Klotski**：multi-batch overlap + expert-aware scheduling
- **MoE-APEX**：adaptive precision offloading
- **EARTH**：speculative prefetch + result reuse

## 2. 如果你的主要症状是“cache 做了，但 batch 一上来还是掉速”
优先看：
- **Diff-MoE**：differential cache hierarchy
- **PopFetcher**：popularity-based prefetch

## 3. 如果你的主要症状是“I/O 不是唯一瓶颈，expert 算子本身也不快”
优先看：
- **Samoyeds**：kernel / sparse execution
- **X-MoE**：新型大 top-k 架构下的 padding-free / dispatch 优化

## 4. 如果你的主要症状是“GPU 不够，但 CPU 内存很多，且并发不高”
优先看：
- **KTransformers**：CPU/GPU hybrid inference

## 5. 如果你的主要症状是“多机扩展差，A2A 通信压垮系统”
优先看：
- **ScheMoE**：task scheduling + all-to-all
- **FSMoE**：profiling-driven runtime
- **X-MoE**：hierarchical redundancy-bypassing dispatch

## 6. 如果你的主要症状是“少数热门 expert 长期过载”
优先看：
- **LAER-MoE**：load-adaptive expert re-layout
- **Diff-MoE**：如果问题更偏 cache residency
- **PopFetcher**：如果问题更偏预测/预取

## 7. 如果你的主要症状是“不是所有 expert 都值得长期保存/复制/高精度驻留”
优先看：
- **MoC-System**：partial checkpoint / persistence 思路
- **MoE-APEX**：precision hierarchy
- **Diff-MoE**：cache hierarchy

---

# 五、超完整对照表

## 5.1 总对照表（摘要版）

| 论文 | 会议/年份 | 层级 | 主要问题 | 核心技术 | 主要收益 | 最适合的场景 | 对推理的直接启发 |
|---|---|---|---|---|---|---|---|
| Klotski | ASPLOS 2025 | 一级 | I/O 比 compute 更慢，pipeline bubble 严重 | expert-aware multi-batch pipeline、I/O-compute planner、correlation-aware prefetch | 最高 85.12× throughput | expert offload 严重、batch 增大反而更差 | 热冷 expert 分层、跨 batch 调度、预测式 prefetch |
| Samoyeds | EuroSys 2025 | 一级 | expert 已到位但算子执行低效 | dual structured sparsity、sparse data format、sparse-sparse kernel | kernel 1.99×、model 1.58×、batch ceiling 4.41× | 稀疏硬件可用、compute-side inefficiency 明显 | 结构化稀疏 MoE kernel、编译器/runtime 优化 |
| KTransformers | SOSP 2025 | 一级 | GPU 不够、CPU/GPU 协同低效 | AMX-specialized kernels、async CPU-GPU scheduling | prefill 4.62–19.74×；decoding 1.25–4.09× | 单机/少卡、本地部署、低并发 | CPU 作为主动计算层，异构路径拆分 |
| Diff-MoE | SC 2025 | 一级 | batched inference 下 cache/comm 失效 | differential expert caching、priority-driven hierarchy | 缓解批推理瓶颈 | batch 请求混合、长期热点 + burst 热点并存 | workload-aware cache、session/prefix-aware residency |
| MoE-APEX | ASPLOS 2026 | 一级 | edge/小内存环境下 offloading 成本高 | adaptive precision offloading、token/layer/sequence 自适应 | 优化边缘 MoE inference | 边缘设备、显存/内存极小 | precision hierarchy + cache hierarchy 联合设计 |
| EARTH | ASPLOS 2026 | 一级 | 路由不确定导致普通 prefetch 浪费大 | entropy-aware speculative prefetch、result reuse | 更强 overlap、更少 fetch stall | 想做预取，但 router 不稳定 | 把 router entropy 和结果复用引入推理系统 |
| CoServe | ASPLOS 2025 | 一级(边界) | 多 expert 协作下 switching 成本高 | dependency-aware scheduling、dependency-aware expert mgmt | 降低 switching 开销 | 多模型/多专家协作系统 | 按 dependency 聚类请求、最小化 expert set 变化 |
| ScheMoE | EuroSys 2024 | 二级 | MoE 训练中 all-to-all 与任务流水低效 | task scheduling、novel all-to-all、可扩展框架 | 9%–30% | 多机训练/多机 dispatch | 推理也可建 task graph，做 compute-comm overlap |
| FSMoE | ASPLOS 2025 | 二级 | 不同 MoE 实现差异大，难统一优化 | unified abstraction、online profiling、adaptive pipeline | 1.18×–3.01× | 通用 runtime / 多变种 MoE | profiling-driven serving runtime |
| MoC-System | ASPLOS 2025 | 二级 | checkpoint 开销被巨大参数量拖垮 | Partial Experts Checkpointing、fully sharded checkpoint、async mgmt | checkpoint overhead 降 98.9%，训练迭代最高 5.12× | expert 热度不均、想做分级持久化 | 冷热 expert 差异化 persistence / replication |
| PopFetcher | ATC 2025 | 二级 | expert future demand 可预测但未被利用 | popularity-based expert-wise prefetch、sliding window | 缓解训练中的通信/等待 | 存在强偏斜与相关性 | 会话级 expert popularity 预测与预取 |
| X-MoE | SC 2025 | 二级 | DeepSeek-style 新 MoE 的 activation/dispatch 成新瓶颈 | padding-free pipeline、redundancy-bypassing dispatch、sequence-sharding | 545B/1024 GPU；吞吐最高 1.42× | 大 top-k、细粒度 expert、新平台 | padding-free 推理、redundancy-bypassing dispatch |
| LAER-MoE | ASPLOS 2026 | 二级 | 热点 expert 负载长期不均 | FSEP、load-adaptive expert re-layout、planner | 最高 1.69× | 热点 expert 显著失衡 | load-adaptive placement、动态复制热点 expert |

## 5.2 详细对照表（按“问题特征 → 技术 → 迁移思路”组织）

| 论文 | 你会观察到的系统症状 | 作者认为的根因 | 论文中的关键技术 | 论文实际解决了什么 | 迁移到推理时可直接借用什么 | 更适合借鉴的方法层级 |
|---|---|---|---|---|---|---|
| Klotski | GPU 利用率不高但 I/O 很忙；batch 大了反而吞吐差 | expert load 比 compute 更慢，且 batch 增大带来更多 activated experts | multi-batch pipeline、I/O-compute planner、correlation-aware prefetch | 吃掉 I/O bubble、提高 overlap | 热/冷 expert 分层、跨 batch 调度、expert 相关性预取 | runtime / scheduler |
| Samoyeds | expert 已在显存中但执行仍低效；稀疏硬件没吃满 | 只利用单侧稀疏，MoE kernel 设计不适配稀疏硬件 | dual structured sparsity、专用 sparse format、sparse-sparse kernel | 提高 expert 计算效率与 batch ceiling | 稀疏执行引擎、结构化稀疏专家 kernel | kernel / compiler / runtime |
| KTransformers | GPU 显存不够但 CPU 内存充足；低并发环境成本敏感 | CPU 计算弱 + CPU/GPU 同步开销高 | AMX kernels、async CPU-GPU scheduling | 让 CPU 真正参与 expert inference | attention/expert 路径分离，热冷 expert 异构映射 | system runtime / deployment |
| Diff-MoE | 单请求可以，batch 变大后 cache/miss 突然恶化 | 不同 expert 的 reuse pattern 被统一 cache policy 混淆 | differential caching、priority hierarchy | 提升批推理下 cache 效率 | session-aware / prefix-aware cache、长短热点分离 | cache policy / runtime |
| MoE-APEX | cold expert miss 导致长尾；full precision 载入太贵 | offload 只做 in/out 二元决策，没利用精度层次 | adaptive precision loading、layer prefetch、sequence-level cache | 以更低加载代价完成 offload | 低精度 expert 副本、精度层次缓存 | memory hierarchy / runtime |
| EARTH | speculative prefetch 常误取；router 稳定性不一 | 没把路由不确定性纳入预取决策 | entropy-aware prefetch、result reuse | 提高预取收益并减少浪费 | router entropy 驱动的预取/复用策略 | accelerator / advanced runtime |
| CoServe | 多 expert/模型切换非常频繁，switch thrashing 严重 | 请求依赖 expert 组合差异大，但调度没感知依赖 | dependency-aware request scheduling、expert management | 降低 switching 开销 | 按 dependency 聚类请求、最小化 expert set 变化 | serving scheduler |
| ScheMoE | 多机 A2A 一上来就崩；流水大量等待 | compute 与 communication 没有被统一编排 | task scheduling、novel all-to-all | 提升扩展效率 | 推理 task graph，dispatch/load/compute/gather 联合调度 | distributed runtime |
| FSMoE | 不同 MoE 变种要分别调优，维护成本极高 | 缺乏统一抽象和 profiling-driven 策略选择 | unified abstraction、online profiling、adaptive pipeline | 支持多 routing / 多实现的统一优化 | 在线 profiling 决定 serving 策略 | general runtime framework |
| MoC-System | 冷 expert 很多但都被“等价对待”；存储/持久化浪费大 | 没有利用 expert 活跃度不均这一结构特征 | partial experts checkpointing、async checkpoint | 以选择性方式管理 experts | 冷热 expert 差异化持久化/副本/物化 | storage / persistence policy |
| PopFetcher | expert 命中分布明显偏斜；短窗口内有规律 | expert popularity 和 correlation 是可预测的 | sliding-window popularity prediction、expert-wise prefetch | 提前预取未来高需求 expert | prefix/session/domain aware prefetch | predictor / prefetcher |
| X-MoE | DeepSeek-style 大 top-k 架构下 activation 爆炸、dispatch 冗余高 | 新型 expert-specialized MoE 改变了系统主瓶颈 | padding-free pipeline、redundancy-bypassing dispatch、sequence-sharding | 支撑新型 MoE 可扩展训练 | padding-free 推理、针对大 top-k 的 dispatch 设计 | runtime / distributed system |
| LAER-MoE | 少数 expert 长期热点，静态 placement 顶不住 | expert placement 被当成静态常量 | FSEP、load-adaptive re-layout | 让专家布局随负载变化 | 热 expert 动态复制/迁移 | placement / load balancer |

## 5.3 按“可迁移到推理的价值”排序

### 第一档：最容易直接迁移到推理
1. Klotski  
2. Diff-MoE  
3. KTransformers  
4. MoE-APEX  
5. EARTH  
6. PopFetcher  
7. LAER-MoE  

### 第二档：需要系统性变形后迁移
8. Samoyeds  
9. FSMoE  
10. ScheMoE  
11. X-MoE  
12. MoC-System  
13. CoServe

---

# 六、总判断：从这些论文中抽出的五条技术主线

## 1. MoE 优化首先是访问模式优化
关键不是先优化 kernel，而是先判断：
- 哪些 expert 热
- 热度怎么演化
- 热度能不能预测
- 热点是长期还是短期 burst

代表论文：
- Klotski
- Diff-MoE
- PopFetcher
- LAER-MoE

## 2. MoE 运行时需要层次化资源管理
资源层次不仅是 HBM/DRAM/SSD，也包括：
- 精度层次
- cache 层次
- placement 层次
- persistence 层次

代表论文：
- MoE-APEX
- Diff-MoE
- MoC-System

## 3. 新型 MoE 架构会重写系统最优解
DeepSeek-style、大 top-k、细粒度 expert 的出现，意味着：
- activation overhead 更重要
- dispatch redundancy 更严重
- 传统 Mixtral/Switch 风格系统假设可能失效

代表论文：
- X-MoE

## 4. 通用 runtime 会越来越重要
系统不可能长期靠一篇篇特化论文拼起来，最终会走向：
- profiling-driven
- task-graph-driven
- strategy-adaptive runtime

代表论文：
- FSMoE
- ScheMoE

## 5. MoE 推理会越来越异构
MoE 的未来不是单一 GPU 方案，而会是：
- CPU/GPU 协同
- 专用 accelerator
- 软硬件协同设计

代表论文：
- KTransformers
- EARTH

---

# 七、建议的阅读与使用顺序

## 7.1 如果你要找“最直接可落地的推理优化”
1. Klotski  
2. Diff-MoE  
3. KTransformers  
4. MoE-APEX  
5. EARTH  

## 7.2 如果你要找“虽然是训练，但很可能能迁到推理”
1. PopFetcher  
2. LAER-MoE  
3. FSMoE  
4. ScheMoE  
5. X-MoE  
6. MoC-System  

## 7.3 如果你要做“通用 MoE 推理系统”
建议组合式阅读：
- 访问模式与缓存：Klotski + Diff-MoE + PopFetcher
- 通用 runtime：FSMoE + ScheMoE
- 异构部署：KTransformers
- 新型架构适配：X-MoE
- 动态布局：LAER-MoE
- 精度与层次资源：MoE-APEX + MoC-System
- 高级预取与复用：EARTH

---

# 八、可转交摘要（一页版）

如果一个同事只看一页，请把这段给他：

MoE 系统优化的主矛盾不再只是“模型太大”，而是“专家访问模式高度不均、动态、可预测但难以管理”。  
近两年的系统会议表明，MoE 推理/训练优化的核心方向有五类：

1. **基于 expert 热度与相关性的调度/预取**  
   - Klotski、PopFetcher、EARTH  
2. **基于长期热点/短期 burst 的缓存层次设计**  
   - Diff-MoE  
3. **基于硬件差异的异构执行**  
   - KTransformers  
4. **面向新型 DeepSeek-style MoE 的 padding-free / redundancy-bypassing dispatch**  
   - X-MoE  
5. **expert placement / precision / persistence 的层次化管理**  
   - LAER-MoE、MoE-APEX、MoC-System  

当你遇到的系统问题表现为：
- 等 expert → 看 Klotski / MoE-APEX / EARTH
- batch 一大就差 → 看 Diff-MoE / PopFetcher
- GPU 不够但 CPU 很多 → 看 KTransformers
- 多机 A2A 很重 → 看 ScheMoE / FSMoE / X-MoE
- 热点 expert 长期过载 → 看 LAER-MoE
- 冷热 expert 明显不均 → 看 MoC-System / Diff-MoE / MoE-APEX

---

# 九、备注
这份文档是基于公开可检索的会议目录、论文摘要与可公开访问的论文页面整理而成，重点是形成系统化的“问题 → 技术 → 迁移到推理”的分析框架，方便后续继续深挖与复现。
