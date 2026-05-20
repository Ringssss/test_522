# EARTH 论文解析与复现笔记

## 一、论文解析（新增总览部分）

### 1. 论文背景

这篇论文讨论的是 **MoE（Mixture-of-Experts）大模型推理中的“数据搬运瓶颈”**。  
MoE 的直觉优势是：虽然总参数量很大，但每个 token 只激活少量 expert，因此计算量比同等参数规模的稠密模型低。但论文指出，**实际系统瓶颈并不在计算，而在 expert 权重的加载与搬运**。在作者对 Qwen3-30B-A3B（INT8）在 DDR5-6400 系统上的分析中，**expert fetching 占总执行时间约 88%**，而 gating、计算、聚合加起来不到 12%。这说明 MoE 推理的主矛盾是“如何更快把 expert 权重搬进来”，而不是“如何把矩阵乘法再优化一点”。fileciteturn0file0L54-L59

论文进一步指出，MoE 层中 routed experts 数量极多，往往远超 FLOPs 等价的稠密模型，因此即便每次只激活 top-k experts，总体内存占用和运行时访存压力仍然极大。作者将目标明确定位为：**在资源受限硬件上，让 MoE 推理更高效地完成 expert offloading、prefetch 和执行重叠。** fileciteturn0file0L35-L49

---

### 2. 论文动机

论文的动机来自两个核心观察。

#### 2.1 观察一：不同 expert 对误差的敏感度不同

作者对 Qwen1.5-MoE-A2.7B 的所有 experts 做了噪声扰动实验，比较扰动前后的输出差异，发现 **expert sensitivity 在同层内部也存在很大差异**：有些 expert 对噪声非常鲁棒，有些则非常脆弱。图 4 显示，不同 expert 的抗噪性明显不同，这意味着：**不是所有 expert 都必须始终保持同样高的表示精度。** fileciteturn0file0L143-L151

这给出了一个关键启发：  
可以按 expert 的“重要性 / 敏感性”来决定是否使用完整精度表示，从而减少不必要的加载量。

#### 2.2 观察二：prefetch 有用，但受片上容量限制

论文还分析了 prefetch budget 对 expert loading 的影响。图 6 表明，随着预取窗口增大，每一步需要临时加载的 experts 数量会下降；例如在 DeepSeek-V2-Lite-Chat 上，prefetch window 从 6 增加到 16，平均需要加载的 experts 数从 5.22 降到 4.09。说明 **prefetch 本身是有效的**。但问题在于，完整 expert 太大，片上 buffer 太小，无法容纳足够多的候选 experts。fileciteturn0file0L171-L180

因此作者想解决的其实是一个双重矛盾：

1. 想要提高预取命中率，就希望一次缓存更多 experts；
2. 但完整 expert 太大，缓存空间根本不够；
3. 如果简单低比特压缩，又会损伤精度。

---

### 3. 论文核心思路

这篇论文的核心思路可以概括成一句话：

> **把 expert 权重拆成“高信息量的 base”和“低位修正的 delta”，优先预取 base，仅在必要时补 delta；同时利用 delta 的重复模式做查表复用，进一步减少带宽开销。**

这套设计不是单点优化，而是 **编码方式、预取策略、模式复用、硬件流水线调度** 的联合设计。论文把整个方案命名为 **EARTH**，强调它是一个 **hardware-software co-design**。fileciteturn0file0L8-L17

---

### 4. 技术方案解析

## 4.1 Dual-Entropy Encoding：把 INT8 expert 切成 base + delta

论文首先提出 **dual-entropy encoding**。对标准 INT8 quantized expert 的每个 8-bit 权重 \(w\)，将其拆成：

\[
w = (w_b << 4) + w_d
\]

其中：

- \(w_b\)：高 4 位 base
- \(w_d\)：低 4 位 delta

作者的理由是：**高位部分承载大部分有效信息，低位更多是精修项**。因此，把 expert 拆成 base 和 delta 后，可以先只传输更关键的 base，再决定是否取 delta。论文还给出了一般化表示：

\[
x = \sum_{i=0}^{b-1} s(c_i)\cdot 2^i
\]

其中 \(c=(c_0,c_1,\dots,c_{b-1})\) 是 b-bit code，且

- 当 \(c_i=1\) 时，\(s(c_i)=+1\)
- 当 \(c_i=0\) 时，\(s(c_i)=-1\)

作者认为这种表示有两个优点：  
一是保留了适合部分重建与组合计算的结构；  
二是便于跨 bit-width 一致扩展。fileciteturn0file0L189-L201

论文特别强调 4/4 split 的硬件友好性：

- 对齐到 1 byte 访问边界，避免非对齐访存带来的开销；
- 更匹配常见 PE 数据通路宽度，减少额外转换代价。fileciteturn0file0L222-L227

---

## 4.2 Delta-aware Speculative Prefetch：只预取 base，delta 按重要性决定

在 dual-entropy encoding 之上，论文提出了 **delta-aware speculative prefetching**。  
核心思想是：**预测未来可能会被选中的 experts，但只提前加载它们的 base 部分，而不是完整 expert。**

执行时分三种情况：

### 情况 A：prefetch hit，且 expert 很重要
如果预测对了，而且实际 gating weight 较高，那么这个 expert 对结果重要，就再去加载它的 delta，恢复高保真表示。fileciteturn0file0L234-L246

### 情况 B：prefetch hit，但 expert 不重要
如果预测对了，但 gating weight 很低，就直接只用 base 来计算，不再加载 delta。这样可以直接节省带宽。fileciteturn0file0L239-L246

### 情况 C：prefetch miss
如果预测错了，则只补加载“正确 expert 的 base”。因为 speculative 阶段本来就没有预取 delta，所以 miss penalty 被压低了。fileciteturn0file0L243-L249

更关键的是，论文说这个 gating weight threshold 是 **offline calibrated** 的，目标是让跳过 delta 后的精度损失不超过 1%。fileciteturn0file0L250-L252

---

## 4.3 新瓶颈迁移：优化完 base 之后，delta 变成新的瓶颈

论文不是做到这一步就结束了。作者进一步分析发现：当 base-loading 被 speculative prefetch 显著优化后，**delta loading 反而成为新的主瓶颈**。在他们的 profiling 中，delta part 仍然占总 load time 的 **60% 以上**。fileciteturn0file0L255-L263

这说明系统瓶颈不是被消灭了，而是发生了转移。  
EARTH 的真正亮点，是它继续向前追问：**怎么再把 delta 也压下去？**

---

## 4.4 Pattern Reuse：delta 可以查表复用

作者观察到一个非常关键的现象：在编码后的 experts 中，很多 **\<base, delta\> 模式高度重复**。  
图 8 显示，delta pattern 的重复率很高；特别是某些 base 模式对应的 delta 候选非常集中。例如对 base = `<1111>` 的情况，92.86% 只落在两种可能的 pattern 上。并且 dominant patterns 的数量很少，通常只是几十种，且与模型规模无关。fileciteturn0file0L265-L276

因此作者构造了一个 LUT：

- 输入键：base
- 输出值：预测 delta

运行时再按 expert 的重要性分层处理：

- **高重要性 expert**：总是从 DRAM 取真实 delta，保证精度；
- **低重要性 expert**：完全不取 delta，只用 base；
- **中等重要性 expert**：不去 DRAM 取 delta，而是用 base 查 LUT，复用预测 delta。fileciteturn0file0L309-L319

这使 EARTH 形成了一套非常清晰的分级策略：

- 关键 expert：高保真
- 不关键 expert：低成本
- 中间 expert：查表猜测

本质上是在“精度需求”和“带宽成本”之间做分层优化。

---

### 5. 硬件架构解析

EARTH 不是纯算法，而是一个完整 accelerator 设计。图 9 给出了总体架构：包括 gating module、token buffer、weight buffer、compute core（16 个 PEs）、weight dispatcher、activation dispatcher、output collector 等模块。论文明确说，设计目标是在受限带宽条件下持续保持较高的 PE utilization。fileciteturn0file0L282-L296

#### 5.1 Memory subsystem
weight buffer 中的 experts 采用 **bit-interleaved format** 存储，并分成 \(W_{base}\) 和 \(W_{delta}\)。运行时只取 \(W_{base}\) 和必要的 LUT entry，delta 解码尽量延后到真正需要的时候。fileciteturn0file0L296-L303

#### 5.2 Dataflow
论文采用 **output-stationary dataflow**：

- weights 用 unicast，因为每个 expert 不同；
- activations 用 broadcast，因为多个 expert 共享相同输入 token。fileciteturn0file0L322-L330

这点对复现很重要：  
它说明 EARTH 的收益不只是“压缩了数据”，而且是和 **数据流组织方式** 绑定的。

#### 5.3 Match-and-action schedule
图 11 中，EARTH 进一步采用 **match-and-action** 执行流程：

- **match 阶段**：根据预取到的 base 和 LUT 判断能否恢复 / 复用 delta；
- **action 阶段**：再把重建后的权重送到 PE 执行。fileciteturn0file0L352-L359

这套调度的核心价值是：  
把 **预取、解码、计算** 尽量重叠起来，减少 pipeline bubble，从而把理论上的数据搬运优化真正落到端到端 latency 上。图 11 的时间线对比说明，单纯 prefetch 或 bit-interleaving 仍然会在 delta fetch 处卡住，而完整 EARTH 通过 match-and-action 把这个问题进一步缓解了。fileciteturn0file0L338-L349

---

### 6. 论文结果解析

#### 6.1 准确率 / 加载量权衡

论文在 Mixtral、Qwen、DeepSeek 三类代表性 MoE 模型上评估了不同重要性 expert 配比下的结果。总体趋势非常清楚：

- 当 **80%–90% 的重要 experts 保持高保真** 时，
- 模型指标下降很小，
- 但 expert loading 需求可以减少 **20% 以上**。fileciteturn0file0L396-L404

以 DeepSeek 为例，在 80% important / 10% moderate / 10% unimportant 配置下，Load Reduction 达到 **42.5%**，而 Rouge-L 只从 28.7 / 25.5 小幅下降到 28.5 / 25.1。fileciteturn0file0L383-L389

论文对这个现象的解释也很合理：

- Mixtral 的 expert 更大，因此 base-only prefetch 带来的收益更明显；
- DeepSeek 的长序列场景里 delta pattern 重复更强，因此 speculative reuse 更有效。fileciteturn0file0L396-L404

#### 6.2 端到端加速

论文最硬的结果是：EARTH 在三种代表性 MoE 模型上实现了 **1.56×–2.10× 的端到端 speedup**。fileciteturn0file0L405-L413

表 5 中给出更细的结果：

- **Mixtral**：ideal 2.32×，practical 2.10×，达到理想值的 90.5%，overlap 89%
- **Qwen**：ideal 2.21×，practical 2.06×，达到理想值的 93.2%，overlap 91%
- **DeepSeek**：ideal 1.83×，practical 1.72×，达到理想值的 94.0%，overlap 86% fileciteturn0file0L415-L423

这说明 EARTH 不是只在局部减少了 load time，而是真正把 **memory transfer 与 compute overlap 做起来了**。

论文还定义了一个 compute-transfer overlap ratio：

\[
r_{overlap}=1-\frac{T_{total}}{T_{stall}}
\]

其中：

- \(T_{total}\)：实际端到端延迟
- \(T_{stall}\)：如果 expert-fetch 完全无法被掩盖时的理论延迟

这个指标本质上是在衡量：**EARTH 究竟把多少“原本会暴露出来的 stall”隐藏掉了。** fileciteturn0file0L406-L413

#### 6.3 能耗与面积

EARTH 不只是更快，也更省能耗。论文在 DeepSeek-V2-Lite-Chat workload 上报告：

- EARTH 的总能耗是 AdapMoE 的 **0.59×**
- 相比 EdgeMoE，能耗降低 **21.54%**
- 其中 DRAM energy 降到 **0.57×**，SRAM energy 降到 **0.68×**。fileciteturn0file0L424-L432

面积方面，总 chip footprint 为 **27.52 mm²**，其中：

- PE array：77.55%
- accumulators：15.26%
- LUTs + control：只有 6.08% fileciteturn0file0L433-L437

这很关键，因为它说明 LUT / 控制逻辑并没有引入离谱的额外开销。

---

### 7. 这篇论文的真正价值

我认为这篇论文的价值可以总结成四点。

#### 7.1 它抓住了 MoE 推理最核心的系统瓶颈
很多工作还在优化矩阵乘法、并行调度或泛化的 accelerator datapath，但 EARTH 直指 **expert fetch 是主要耗时来源**。论文首页和图 1 已经把这个问题定性得很清楚。fileciteturn0file0L35-L49

#### 7.2 它不是“压缩”而已，而是“编码—预取—复用—调度”一体化设计
EARTH 不是单纯地把 INT8 变成 INT4，而是做了：

- dual-entropy encoding
- delta-aware prefetch
- pattern reuse
- match-and-action pipeline scheduling

这是一条完整链路，而不是一个 isolated trick。fileciteturn0file0L8-L17

#### 7.3 它考虑了精度与带宽的可控折中
它没有强迫所有 experts 都低精度，而是引入 importance-aware 的分层策略。这样更像系统设计，而不是纯粹做 aggressive approximation。

#### 7.4 它有较强工程落地意味
论文使用了 RTL 实现、Synopsys Design Compiler 综合、CACTI 建模、cycle-accurate simulation，并明确给出 28nm、250MHz、HBM2E 配置。说明它不是概念验证，而是完整 accelerator paper。fileciteturn0file0L360-L378

---

### 8. 局限性与复现时要保持警惕的点

#### 8.1 依赖 expert sensitivity 的显著异质性
如果目标 MoE 模型里 experts 的鲁棒性差异没有论文里这么明显，那么“重要 / 中等 / 不重要”的分层收益可能会下降。fileciteturn0file0L143-L151

#### 8.2 依赖 delta pattern redundancy
如果 \<base, delta\> 模式重复度不高，那么 LUT reuse 的收益会变小。论文自己的结果也显示不同模型收益有明显差异。fileciteturn0file0L265-L276

#### 8.3 硬件收益依赖实现细节
EARTH 的加速并不只来自 algorithm，还来自：

- bit-interleaved memory layout
- FIFO-style refresh
- output-stationary dataflow
- overlapping schedule

如果复现时只模拟“base/delta 拆分”而不实现这些调度与数据流细节，结果通常会低很多。fileciteturn0file0L322-L330

---

### 9. 是否开源

就论文正文而言，我没有看到明确的 GitHub、artifact、repository 或 supplementary code 链接。论文提供的是 DOI、参考文献和 CC BY 4.0 的论文许可，但这并不等于代码已开源。就 PDF 本身能确认的信息来说，**无法确认其实现已经开源，至少论文正文没有给出公开仓库地址。** fileciteturn0file0L1-L17

---

## 二、复现时必须记住的关键结论

### 1. EARTH 解决的不是“计算瓶颈”，而是“数据搬运瓶颈”
- 在代表性配置里，expert fetching 占总执行时间约 88%。
- 所以优化目标是减少 expert loading latency 和提升 overlap，而不是单独追求 PE 算力。fileciteturn0file0L54-L59

### 2. 论文把 MoE offloading 的瓶颈拆成两个核心变量
根据论文给出的模型，负载时间取决于：
- 单 expert 的平均加载时间 \(\bar t\)
- 预取成功数 \(R\) 或预取预算 \(p\)

因此复现必须同时覆盖：
- 减小每个 expert 的加载代价
- 提高在给定 buffer 下的 prefetch 命中率。fileciteturn0file0L126-L138

### 3. expert sensitivity 是方案成立的基础前提
- 不同 expert 对噪声容忍度差异很大；
- 只有确认这种异质性存在，importance-aware 表示才有意义。fileciteturn0file0L143-L151

### 4. 高位信息密度更高，因此可以先传 base
- dual-entropy encoding 的前提是高位更有信息量；
- 所以 speculative prefetch 先传 base，不先传 delta。fileciteturn0file0L171-L180

### 5. speculative prefetch 只解决了一半问题
- 优化完 base 后，delta 仍占 60%+ load time；
- 所以必须继续做 delta-aware reuse。fileciteturn0file0L255-L263

### 6. delta reuse 成立依赖重复模式
- \<base, delta\> 的 dominant patterns 数量很少；
- LUT reuse 的核心价值是用极小的表替代大量 delta DRAM 访问。fileciteturn0file0L265-L276

### 7. 真正的系统收益来自联合优化
最终加速不是任何一个 trick 单独带来的，而是：
- dual-entropy encoding
- speculative prefetch
- delta reuse
- match-and-action scheduling
共同产生的。fileciteturn0file0L443-L449

---

## 三、复现时必须记住的关键公式

## 1. Expert loading time

论文定义：

\[
T_{load}=(k-R)\bar t
\]

其中：
- \(k\)：top-k 激活 expert 数
- \(R\)：成功预取的 expert 数
- \(\bar t\)：单个 expert 平均加载时间

这个公式的意义是：  
**真正需要临时加载的 expert 数 = 实际需要的 \(k\) 个 - 预取命中的 \(R\) 个。**

---

## 2. Expected loading time

在 routed experts 从 \(n\) 个 experts 中均匀采样、prefetch \(p\) 个 expert 的假设下，期望加载时间为：

\[
\mathbb E[T_{load}] = k\left(1-\frac{p}{n}\right)\bar t
\]

这个公式非常重要，因为它把论文动机精确拆成了两个方向：

- 减小 \(\bar t\)：让单个 expert 更轻、更快加载；
- 增大 \(p\)：让同样 buffer 下能预取更多 experts。fileciteturn0file0L126-L138

EARTH 正是围绕这两点设计的：
- 用 base/delta 降低有效加载成本；
- 用更小的 base 提高可预取 expert 数。

---

## 3. Dual-entropy encoding 的核心公式

### 3.1 Base-delta 分解
\[
w=(w_b << 4)+w_d
\]

这里 \(w_b\) 是高 4 位，\(w_d\) 是低 4 位。  
这是复现 EARTH 数据格式的最核心公式。fileciteturn0file0L189-L194

### 3.2 一般化 bit-level 表示
\[
x = \sum_{i=0}^{b-1} s(c_i)\cdot 2^i
\]

其中
\[
s(c_i)=
\begin{cases}
+1, & c_i=1 \\
-1, & c_i=0
\end{cases}
\]

这个公式主要用于说明其 sliceable / homomorphic 结构。  
如果你只是做最小系统复现，它的重要性低于上面的 base-delta 分解；  
但如果你要严格复现编码逻辑，就必须保留它。fileciteturn0file0L194-L201

---

## 4. Compute-transfer overlap ratio

论文给出：

\[
r_{overlap}=1-\frac{T_{total}}{T_{stall}}
\]

其中：
- \(T_{total}\)：实际端到端时延
- \(T_{stall}\)：如果所有 expert-fetch stall 都不被隐藏时的理论时延

这个公式是你复现系统级收益时必须实现的指标之一，因为论文表 5 里所有 “接近 ideal” 的说法都依赖这个 overlap 指标。fileciteturn0file0L406-L413

---

## 四、复现时需要拆解的 5 个模块

## 模块 1：Expert sensitivity profiling
目标：
- 对每个 layer、每个 expert 测试对扰动的敏感度；
- 获得重要 / 中等 / 不重要 expert 的划分依据。

最直接做法：
- 对 quantized INT8 experts 注入 additive noise；
- 比较扰动前后输出余弦距离；
- 统计不同 expert 的 sensitivity ranking。fileciteturn0file0L143-L151

你至少要复现出：
- sensitivity 在 expert 间高度不均衡；
- 同层内也存在明显差异。

---

## 模块 2：Dual-entropy encoding
目标：
- 实现 INT8 → base(4bit) + delta(4bit) 的拆分；
- 设计对应的存储格式。

必须验证：
- 拆分后能正确重建原始权重；
- 单独使用 base 推理时误差可控但不为零；
- 4/4 split 的存储和调度确实比“直接另存 INT4/INT8 两套副本”更合理。fileciteturn0file0L189-L201

---

## 模块 3：Delta-aware speculative prefetch
目标：
- 用 routing history 预测下一 token 可能使用的 experts；
- 只预取这些 experts 的 base；
- 在实际 routing 结果出来后，再决定是否取 delta。

你要复现三种分支：
1. hit + important → fetch delta
2. hit + unimportant → skip delta
3. miss → load right base only fileciteturn0file0L234-L249

还要实现一个离线校准的 gating threshold，使得跳过 delta 的精度下降不超过论文设定范围（文中说不超过 1%）。fileciteturn0file0L250-L252

---

## 模块 4：Pattern reuse / LUT
目标：
- 统计 \<base, delta\> 对的频次；
- 为频繁模式构建 LUT；
- 让 moderate-importance experts 通过查表获得 predicted delta。

你必须至少验证两件事：
- dominant patterns 数量很少；
- 通过 LUT 取代真实 delta 加载可以明显降低 DRAM traffic。fileciteturn0file0L265-L276

---

## 模块 5：Timeline / overlap simulation
目标：
- 比较以下 4 条时间线：
  1. Simple MoE
  2. MoE + Prefetch
  3. MoE + Bit Interleaving
  4. Full EARTH

最少要测：
- total latency
- prefetch hit rate
- delta load ratio
- overlap ratio
- end-to-end speedup fileciteturn0file0L338-L349

因为论文真正有说服力的不是单个模块，而是完整时间线对比。

---

## 五、关键决策点与实验注意事项

### 1. 不要把“important expert”理解成静态 expert identity
论文里的重要性是运行时由 gating weight 驱动的，不是说某几个 expert 永久重要。  
复现时一定要区分：
- offline sensitivity profiling
- runtime gating importance

它们相关，但不是同一个概念。

### 2. 不要只做 base/delta 格式而忽略 prefetch 逻辑
如果你只是拆了格式，但没有实现 speculative prefetch，那么你只能看到“表示层”收益，看不到论文的主要系统收益。

### 3. 不要只做 prefetch 而忽略 delta reuse
因为论文已经证明：base 优化之后，delta 才是新的瓶颈。  
如果没有 LUT reuse，你的结果大概率只能接近 “bit-interleaving” 而不是 “full EARTH”。fileciteturn0file0L255-L263

### 4. 不要只看 load reduction，还要看 accuracy
EARTH 的价值不在于“尽可能减少加载”，而在于“以尽量小的精度代价减少加载”。  
因此实验表必须同时给：
- Load Red.
- Rouge-L / QA accuracy / perplexity

### 5. CER 是一个非常重要的横轴
论文多次用 Cached Expert Ratio（CER）分析在不同缓存预算下的表现。  
你的复现实验最好至少覆盖 10%、30%、50% 三档。fileciteturn0file0L392-L395

---

## 六、最小复现路径（推荐）

如果你不是要做完整 RTL 级复现，而是想快速验证论文主张，建议按这个顺序：

### 第一步：做软件级 trace-driven 模拟
先不做硬件 RTL，直接在 Python / C++ 中用 token routing trace 模拟：
- top-k routing
- prefetch buffer
- hit/miss
- base / delta 加载量
- LUT reuse

目标是先验证：
- load reduction 是否成立
- prefetch hit rate 是否提升
- delta reuse 是否有效

### 第二步：加入 accuracy proxy
在真实模型上做小规模实验：
- 用 base-only、base+real-delta、base+LUT-delta 三种模式比较输出差异；
- 统计不同重要性阈值下的精度变化。fileciteturn0file0L396-L404

### 第三步：实现 timeline simulator
模拟论文图 11 的四类执行时间线，测：
- total latency
- overlap ratio
- ideal vs practical speedup

### 第四步：再决定是否进入硬件级实现
如果前面三步已经支持论文主要结论，再决定是否继续做：
- memory layout
- PE dataflow
- RTL / cycle-accurate simulation

---

## 七、最终结果检查清单

如果你说自己“复现了 EARTH”，至少要能回答下面这些问题：

### A. 背景与动机
- 你是否证明了 expert fetching 是主要瓶颈？
- 你是否证明了 expert sensitivity 存在显著差异？
- 你是否证明了 prefetch 受 buffer 限制？

### B. 编码与表示
- 你是否正确实现了 INT8 → base/delta 分解？
- 你是否验证了 base-only 会有一定误差，但不是灾难性失真？

### C. 预取逻辑
- 你是否实现了只预取 base 的 speculative prefetch？
- 你是否实现了 hit / miss / importance 三分支？

### D. Reuse 机制
- 你是否统计了 \<base, delta\> pattern redundancy？
- 你是否真正用 LUT 替代了一部分 delta fetch？

### E. 系统收益
- 你是否测了 hit rate、load reduction、latency、overlap ratio、speedup？
- 你是否至少复现出“prefetch 优于 naive，full EARTH 优于 only bit-interleaving”的趋势？

如果上述问题里有两三项还回答不了，那通常说明还只是“部分复现”。

---

## 八、一页纸速记版

### 论文一句话
EARTH 通过把 expert 拆成 base + delta，先预取 base，再按 expert 重要性决定加载或复用 delta，并用专门硬件调度把预取、解码、计算重叠起来，从而显著减少 MoE 推理中的权重搬运成本。

### 最重要的三个结论
1. MoE 瓶颈主要是 expert fetching，不是 compute。  
2. expert 有敏感度差异，因此可以重要性分层。  
3. delta pattern 有重复性，因此可以 LUT reuse。  

### 最重要的四个公式
\[
T_{load}=(k-R)\bar t
\]

\[
\mathbb E[T_{load}] = k\left(1-\frac{p}{n}\right)\bar t
\]

\[
w=(w_b << 4)+w_d
\]

\[
r_{overlap}=1-\frac{T_{total}}{T_{stall}}
\]

### 最重要的三个模块
1. base/delta encoding  
2. delta-aware speculative prefetch  
3. pattern reuse + match-and-action scheduling  

### 最终看什么指标
- accuracy / Rouge-L / perplexity
- load reduction
- prefetch hit rate
- overlap ratio
- end-to-end speedup

---

## 九、给你的实际建议

如果你的目标是“论文级复现”，不要一上来就做 RTL。  
最稳妥的路线是：

1. 先做 trace-driven simulator，验证论文主逻辑；
2. 再做 accuracy 验证，确认 important / moderate / unimportant 分层是可行的；
3. 最后再做硬件级数据流和时间线模拟。

这样最容易定位：到底是编码不对、阈值不对、prefetch 不对，还是 reuse 不对。
