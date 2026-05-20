# EARTH 复现文档

*An Efficient MoE Accelerator with Entropy-Aware Speculative Prefetch and Pattern Reuse*  
面向复现的关键结论、关键公式、实现要点与实验检查清单

---

## 先记住这 6 句话

1. **EARTH 要解决的主瓶颈不是 MoE 计算，而是 expert weight fetching。**
2. 它的主策略不是“把所有 expert 都压得更狠”，而是把 INT8 权重拆成高位 `base` + 低位 `delta`。
3. 预取时只预取 `base`，因此同样的片上 buffer 可以容纳更多候选 expert，显著提升预取命中率。
4. `delta` 不再统一加载，而是依据 gating weight / expert importance 做按需加载或查表复用。
5. 复现时最重要的不是把所有硬件细节完全重做，而是先重建：**load-time model、dual-entropy format、prefetch policy、delta reuse policy、评价协议**。
6. 论文的主结果是：在基本保持精度的同时，显著降低数据搬运，并实现 **1.56×–2.10×** 端到端加速。

---

## 1. 复现时必须记住的关键结论

- **K1. 真正的瓶颈是 expert 搬运，占总时间的大头。**
- **K2. 两个决定性能的核心量是：单 expert 加载时间与可预取 expert 数量。**
- **K3. 不同 expert 的敏感度不同，允许做差异化保真度管理。**
- **K4. 高位 bits 信息量更高，因此可以拆成 `base / delta`。**
- **K5. 只预取 `base` 是 EARTH 的关键工程杠杆。**
- **K6. 完成 `base` 侧优化后，`delta` 会成为新瓶颈。**
- **K7. `delta` 并非完全随机，存在大量可复用的 `<base, delta>` 模式。**
- **K8. 复现时必须把“精度–加载量–速度”三者一起看。**

### 关键结论 1：fetching 是主瓶颈

- 论文 profiling 表明，在代表性 MoE 推理流程中，**expert fetching 占总执行时间约 88%**，gating、专家计算、聚合合计不足 12%。
- 因此复现 EARTH 的第一原则，是先复现“数据搬运占主导”的实验结论。
- 建议先做一个无任何 EARTH 优化的 baseline timeline，把一个 token / step 的时间拆成：`gate / fetch / compute / aggregate` 四部分。

### 关键结论 2：性能由两个量主导

- 论文把 expert loading time 建模为两部分共同作用：每次 miss 的代价有多大，以及预取到底命中了多少。
- 所有复现工作都可以归结为两个方向：
  - 降低单个 expert 的有效加载代价
  - 提升可驻留 / 可预取的 expert 数量与命中率

### 关键结论 3：expert sensitivity 高度异质

- 作者通过对 `Qwen1.5-MoE-A2.7B` 的 expert 注入扰动，测量输出相似度变化，发现不同 expert 的脆弱性差异很大，甚至同一层内部也明显不同。
- 这直接支撑了“不要统一精度处理所有 expert”的设计动机。
- 如果你复现算法部分，至少要能重现“某些 expert 对扰动明显更敏感”的排序特征，哪怕数值不完全一致。

### 关键结论 4：高位信息主导，适合拆成 `base / delta`

- 论文认为高位 bits 承载了主要信息，低位 bits 更像精修项。
- 因此 `INT8 → 4-bit base + 4-bit delta` 既保留了可组合性，又给预取和按需补全创造了空间。
- 复现时要特别注意：论文强调这种 4/4 切分不仅是信息论上的观察，也是硬件上对齐与数据通路友好的选择。

### 关键结论 5：只预取 `base` 会抬高有效 buffer 容量

- 这是最值得记住的一条工程经验：不要把“预取完整 expert”当作默认目标，而要把“先确保更多候选 expert 的 `base` 在片上”当作目标。
- 原因很简单：同样的 buffer 预算下，`base` 更小，能容纳更多 expert，因此命中率提升非常明显。

### 关键结论 6：`delta` 是第二阶段瓶颈

- 在 `base` 预取做好之后，剩余的主要代价会转移到 `delta` 加载。
- 换句话说，复现 EARTH 不能只做 `base` 预取，否则通常只能拿到部分收益。

### 关键结论 7：`delta` 有模式复用机会

- 作者统计发现，许多 `<base, delta>` 模式重复出现，而且 dominant patterns 的数量不大。
- 这给出了 LUT-based reuse 的依据：对中等重要性的 expert，不必去 DRAM 取真实 `delta`，而可以用 `base` 去查表预测一个 `delta`。

### 关键结论 8：EARTH 的核心不是单个点优化，而是链路协同

- 完整 EARTH = `dual-entropy encoding + delta-aware speculative prefetch + pattern reuse + 匹配硬件数据流与调度`
- 只单独复现其中一个模块，往往不足以接近论文的完整速度收益。

---

## 2. 复现时最重要的关键公式

| 编号 | 公式 | 你要如何理解 | 复现用途 |
|---|---|---|---|
| Eq.1 | `T_load = (k - R) · t̄` | 一次推理中需要临时加载的专家数 = 实际激活专家数 `k` 减去成功预取的专家数 `R`；每个专家平均加载代价是 `t̄`。 | 用来拆解“为什么慢” |
| Eq.2 | `E[T_load] = k · (1 - p/n) · t̄` | 当 routed experts 近似从 `n` 个专家中均匀采样，且可预取 `p` 个时，期望加载时间与 `p/n` 成反比。 | 用来证明“多驻留 / 多预取”为什么有效 |
| Eq.3 | `w = (w_b << 4) + w_d` | INT8 权重拆为高 4 位 `base` 与低 4 位 `delta`；先用 `base` 保住主体信息，`delta` 作为补偿项。 | 用来实现 dual-entropy format |
| Eq.4 | `x = Σ_{i=0}^{b-1} s(c_i) · 2^i` | 论文给出的 b-bit 码表示，用于说明该编码在加法/乘法上的可组合性；其中 `s(c_i)=+1` 或 `-1`。 | 用来理解为何部分计算可组合 |
| Eq.5 | `r_overlap = 1 - T_total / T_stall` | 衡量计算与数据传输重叠得有多好；越接近 1，说明 fetch stalls 被隐藏得越充分。 | 用来复现实验中的 overlap 指标 |

### 每个公式的复现解释

#### 公式 1：`T_load = (k - R) · t̄`

- `k` 是当前 token 真正需要计算的 expert 数（top-k）。
- `R` 是其中已经被成功预取到片上的 expert 数。
- `t̄` 是平均加载一个 expert 的时间，通常与专家大小、带宽、访存延迟、传输调度有关。
- 复现建议：先记录每 step 的 `k、R、平均加载字节数、有效带宽`，从 trace 中反推 `t̄`，再校验公式趋势是否成立。

#### 公式 2：`E[T_load] = k · (1 - p/n) · t̄`

- `p` 是预取窗口中可容纳的专家数量，`n` 是当前层 routed experts 总数。
- 如果 `p` 增大，`p/n` 增大，则期望 miss 数减少，因此加载时间下降。
- 这个公式虽然基于均匀采样假设，不一定精确拟合真实 gating，但非常适合作为设计直觉与 ablation 的解释工具。
- 复现建议：画出不同 `p` 下的平均 miss experts per step 与实际 `T_load` 曲线，再对照该模型看趋势一致性。

#### 公式 3：`w = (w_b << 4) + w_d`

- `w_b` 是高位 `base`，`w_d` 是低位 `delta`。
- 工程上要注意 `signed/unsigned` 的处理、量化零点和 scale 的位置，以及 reconstruct 后是否和原 INT8 权重严格一致。
- 若只做算法仿真，可以先在整数域精确重建；若做硬件复现，则要明确 `bit packing、alignment 和 memory layout`。

#### 公式 4：`x = Σ s(c_i) · 2^i`

- 这部分不是简单的“再写一遍二进制分解”，而是在说明这种表示利于部分和组合。
- 论文强调它保留 `addition / multiplication` 的 homomorphism，因此 `base` 计算与 `delta` 计算可以后续拼起来。
- 复现建议：若你不重做硬件 datapath，可以将其视作一种理论支撑；但若做 accelerator 仿真，就应验证 `base-only、base+delta、LUT-predicted-delta` 三种路径的输出拼接是否一致。

#### 公式 5：`r_overlap = 1 - T_total / T_stall`

- `T_total` 是实际总延迟。
- `T_stall` 是假设所有 expert-fetch stalls 都完全暴露、没有被计算覆盖时的理论延迟。
- 该指标用来衡量调度与预取是不是把访存空泡藏起来了。
- 复现建议：不要只报告 `speedup`，最好同时报告 `overlap ratio`，这样能解释 EARTH 为什么接近 ideal speedup。

---

## 3. 论文方法拆成可复现的 5 个模块

### 模块 A：Baseline profiling

- 输入：选定的 MoE 模型、量化后的 expert weights、推理 trace
- 输出：每 step 的 `gate / fetch / compute / aggregate` 时间拆分
- 最低要求：确认 fetching 是主要瓶颈，并测得不同层、不同模型下的 expert loading 时间

### 模块 B：Dual-Entropy Encoding

- 把 INT8 expert 拆为 4-bit `base` 与 4-bit `delta`
- 保证 reconstruct correctness：`base+delta` 恢复后的权重与原 INT8 一致
- 输出新的存储格式，以及以 `base / delta` 为基本对象的 memory layout

### 模块 C：Delta-aware Speculative Prefetch

- 用 routing history 预测下一步可能激活的 expert
- 仅预取 `base` 到片上 `FIFO / cache / simulated buffer`
- 待真实 gating 结果出来后，按重要性决定：加载真实 `delta / 跳过 delta / 准备走 LUT reuse`

### 模块 D：Pattern-based Delta Reuse

- 统计训练 / 推理过程中高频 `<base, delta>` 模式
- 构建 LUT：`base` 作为 key，`delta` 作为 value（或候选 value）
- 对 moderate-importance experts 使用 speculative delta reuse

### 模块 E：Evaluation protocol

- 精度侧：`Rouge-L、QA accuracy、MMLU` 或等价任务指标
- 系统侧：`load reduction、prefetch hit rate、latency、speedup、overlap ratio、energy/area（若做硬件）`
- ablation 侧：`无预取、naive prefetch、speculative prefetch、bit interleaving、full EARTH`

---

## 4. 复现中的关键决策点

| 决策点 | 建议 |
|---|---|
| D1. 你复现的是“算法趋势”还是“硬件结果” | 如果只是验证论文思想，可先在 Python / simulator 中实现 `weight split、prefetch、reuse policy`；如果要接近论文结果，则需要 cycle-accurate timing 与 memory model。 |
| D2. 重要 / 中等 / 不重要 expert 如何划分 | 论文基于 gating weight threshold 离线校准，目标是跳过 delta 后精度下降不超过 1%。复现时可以把阈值 sweep 成曲线，而不是一开始就固定死。 |
| D3. 预取预测器如何实现 | 论文主要利用 routing history；你可以先做最简单的 `last-step / recent-frequency predictor`，只要能体现 speculative prefetch 的趋势。 |
| D4. LUT 大小与候选模式数 | 不要一开始追求复杂模型。先统计高频模式覆盖率，确认“少量模式覆盖大部分 delta”是否成立，再决定 LUT 规模。 |
| D5. buffer policy | 论文图示采用 FIFO-style refresh。复现时最好至少对比 `FIFO` 与 naive `cache/queue`，验证 hit-rate 提升。 |
| D6. 评估数据长度 | 论文对 `CNN/DM` 与 `LongBench` 使用固定长度 traces。复现时要避免 trace 长度不一致导致速度和命中率不可比。 |

---

## 5. 推荐的最小复现路径（按优先级）

1. **Step 1：** 重建 baseline profile，证明 fetch 是主要瓶颈。
2. **Step 2：** 实现 `INT8→(base, delta)` 的拆分与精确重建。
3. **Step 3：** 实现只预取 `base` 的 speculative prefetch，先不做 delta reuse。
4. **Step 4：** 引入 importance thresholds，区分 `high / low experts`，对 low experts 跳过 delta。
5. **Step 5：** 加入 LUT-based delta reuse，只作用于 moderate experts。
6. **Step 6：** 报告“精度–加载量–速度”三张曲线，而不只是一组点。
7. **Step 7：** 若继续深入，再做 `memory model / cycle-accurate / RTL` 映射。

---

## 6. 你最终至少应复现出的结果

### 结果 A：趋势性结论

- 随着可预取 expert 数 `p` 上升，平均加载需求下降。
- 只预取 `base` 比预取完整 expert 有更高 hit rate / 更低有效加载时间。
- 如果仅做普通低比特截断，精度更容易明显下降；EARTH 格式能在更高近似比例下保持较好精度。

### 结果 B：性能指标

- 端到端 `speedup` 相对强 baseline 明显提升。
- `overlap ratio` 显著高于 naive prefetch。
- tight memory budget 下提升更明显。

### 结果 C：精度指标

- 保留 80%–90% 的重要 experts 时，精度损失应接近可忽略。
- 对中等重要性 expert 使用 delta reuse 后，应看到更优的 `accuracy–throughput tradeoff`。

---

## 7. 复现检查清单

- [ ] 是否已验证 original INT8 权重能被 `base+delta` 无损重建？
- [ ] 是否区分了 `base 命中、delta 命中、prefetch miss` 三种情况？
- [ ] 是否显式记录了每 step 的 `top-k experts、预测 experts、命中 experts`？
- [ ] 是否做了 `gating threshold sweep`，而不是只报一个阈值？
- [ ] 是否统计了 `<base, delta>` 模式覆盖率，而不是直接假设 LUT 有效？
- [ ] 是否把 latency 拆成 `fetch / compute / overlap`，而不仅是总时间？
- [ ] 是否在不同 buffer capacity 下测试过 hit rate？
- [ ] 是否做了 ablation：`baseline、naive prefetch、speculative prefetch、EARTH full`？
- [ ] 是否检查过长序列任务与短序列任务表现差异？
- [ ] 是否把所有图表都固定在相同 trace 长度与评价协议下？

---

## 8. 一页纸速记版

1. 主瓶颈：MoE 不是算得慢，而是 expert fetch 慢。
2. 两大变量：单次加载代价 `t̄`、可预取专家数 `p`。
3. 核心模型：`T_load = (k - R)t̄`；`E[T_load] = k(1-p/n)t̄`。
4. 核心表示：`w = (w_b << 4) + w_d`。
5. 核心策略：先预取 `base`，再按重要性加载 / 复用 `delta`。
6. 核心假设：高位更有信息量；`delta` 有重复模式。
7. 核心收益：同样 buffer 下预取更多 expert、减少真实 DRAM 访问、提高 overlap。
8. 核心指标：`accuracy、load reduction、prefetch hit rate、latency、speedup、r_overlap`。

---

## 附注

本文件依据你提供的论文 PDF 内容整理，目标是帮助你做“面向复现”的信息压缩。  
它重点保留了论文中的关键结论、关键公式、关键实现逻辑和实验检查点，便于你后续拆成代码任务或实验任务。
