# MoE 推理优化：Expert 使用模式分析与静态化加速探索

> **用途**：供与算法专家讨论 fused_experts 静态化/持久化加速方向  
> **日期**：2026-04-19  
> **阶段**：v0.1.15.8l（Expert Budgeting + K_B_v3 + batch=128 M-sweep + Profiling + Expert Usage Pattern Analysis）

---

## 1. 系统概况

### 1.1 模型

- **LLaDA2.0-mini**（DeepSeek-style MoE）
- 256 experts, top_k=8, n_group=8, topk_group=4, group_size=32
- 20 层 (L0 dense, L1-L19 MoE)
- hidden_size=2048, moe_intermediate_size=512
- 每个 expert 权重: w13 [1024, 2048] + w2 [2048, 512] = **6.29MB** (bf16)
- 全模型 MoE 权重: 256 × 6.29MB × 19 layers = **30.7GB**

### 1.2 推理框架

- **Block Diffusion**: semi-autoregressive, 32-token blocks, threshold-based parallel decoding
- 每个 block 内迭代去噪直到所有 token confidence > threshold (0.90)
- 典型 forward 次数: batch=128 下 ~278-285 fwd (gen_length=256)

### 1.3 硬件

- NVIDIA H100 80GB HBM3
- HBM bandwidth: 3.35 TB/s
- L2 cache: 50MB
- 单个 expert 权重 6.29MB > L2 可容纳量

### 1.4 标准测试场景

| 参数 | 值 |
|------|-----|
| batch_size | 128 |
| gen_length | 256 |
| block_length | 32 |
| threshold | 0.90 |
| temp | 0 (timing) / 0.7 (quality) |
| q_major | 1.0 (100% token 覆盖) |
| M (S_mask 刷新周期) | 5 |

---

## 2. 当前优化架构

### 2.1 MoE Layer 的完整 Forward 流程

```
Per MoE layer forward (L1-L19):

  ① shared_experts(hidden_states)                    ← 固定 MLP，不涉及 expert 选择
  ② gate.get_logits(hs_flat) → logits [N, 256]      ← 1 个 cuBLAS GEMM
  ③ Expert Budgeting (EB):
     冷路径 (block 首次 forward):
       zero_init → K_A_cold → K_B_v3 → 16×(K_C+K_D) → S_mask [256]
       质量保证: q_major=1.0 → 100% token 满足 quality_floor=0.70 覆盖率
     热路径 (fwd_in_block % 5 == 0):
       K_A + K_B_v3 → 刷新 S_mask (popularity top-K_init)
       无质量保证，纯 popularity 投票
     热路径 (其余轮):
       直接返回缓存的 S_mask，零计算
  ④ fused_routing(logits, bias, S_mask) → topk_ids [N, 8], topk_weights [N, 8]
     1个Triton kernel，融合: sigmoid → group_limited_topk → AND S_mask → top-8 → gather → normalize
  ⑤ fused_experts(hs, w1[256,...], w2[256,...], topk_weights, topk_ids)
     vllm 的 Triton kernel，auto-tuned for E=256, N=512, H100
     只加载 topk_ids 中出现的 unique expert 的权重
  ⑥ y + shared_res
```

### 2.2 S_mask 的工作原理

S_mask 是一个 `[256]` 的 0/1 向量，标记哪些 expert 是当前 block 的候选集。

- S_mask **不改变 weight tensor 大小**（仍然是 `w[256, ...]`）
- S_mask 在 fused_routing kernel 内部作为 AND 过滤器：`expert_active = group_selected AND S_mask`
- 效果：topk_ids 中只出现 S_mask=1 的 expert ID → fused_experts 只加载这些 expert 的权重
- 典型 |S_mask| ≈ 161 (avg), range [118, 168]

### 2.3 当前性能

| 配置 | batch | Time(s) | Fwd | ms/fwd | vs C0 |
|------|------:|------:|----:|------:|------:|
| C0: 原始 baseline (eager) | 32 | 8.79 | 262 | 33.56 | — |
| C5: +fused routing Triton kernel | 32 | 7.70 | 272 | 28.32 | **-12.4%** |
| C5 | 128 | 12.42 | 278 | 44.67 | — |
| **C10-M5: +EB (M=5, K_B_v3)** | **128** | **12.07** | **282** | **42.80** | **-2.8% vs C5** |

Baseline 优化（已叠加）：max_unroll=4 + fused RMSNorm + flash-attn 2.8.3 = +18.4%

输出质量：128 个 prompt 全部验证 PASS（5 个可验证 prompt 逻辑正确 + 20 个随机抽样对比 + 128 个自动预筛零 FLAG）。

---

## 3. 组件级 Profiling (batch=128, gen_length=64, 91 fwd)

### 3.1 C5 vs C10-M5 组件对比

| 组件 | C5 (ms) | C10-M5 (ms) | Delta | %instr(C5) |
|------|------:|------:|------:|------:|
| **fused_experts** | **1550.2** | **1362.4** | **-187.8 (-12.1%)** | **43.2%** |
| **Attention** | **1054.1** | **1051.9** | **-2.2** | **29.4%** |
| LMHead | 326.2 | 325.1 | -1.1 | 9.1% |
| gate_getlogits | 252.2 | 250.7 | -1.4 | 7.0% |
| shared_experts | 108.0 | 107.0 | -1.0 | 3.0% |
| routing (Triton) | 76.6 | 90.1 | +13.5 | 2.1% |
| **EB_total** | **0** | **84.6** | **+84.6** | — |
| RMSNorm (pre+post) | 142.4 | 142.5 | +0.1 | 4.0% |
| DenseMLP (L0) | 40.2 | 40.0 | -0.2 | 1.1% |
| Gap (Python/decoder) | 413.0 | 409.8 | -3.2 | — |

**净收益**：fused_experts 节省 187.8ms - EB 开销 84.6ms = **103.2ms** → 对应 E2E -2.8%

### 3.2 EB 子 kernel 细分

| Sub-kernel | Total (ms) | Per-call | 占 EB |
|------------|------:|------:|------:|
| **cold_batchadd (K_C+K_D 循环)** | **42.5** | **746μs** | **50.2%** |
| hot_update (K_A+K_B_v3) | 23.8 | 74μs (323 calls) | 28.1% |
| cold_K_A | 6.5 | 114μs (57 calls) | 7.7% |
| hot_skip (cache return) | 1.8 | 1.3μs (1349 calls) | 2.1% |
| cold_K_B (v3, tl.sort) | 0.3 | 5μs (57 calls) | 0.4% |
| cold_zero_init | 0.2 | 4μs (57 calls) | 0.2% |

### 3.3 Profiling 关键结论

- fused_experts 仍是最大组件（43.2%），但 **batch=128 下 Attention (29.4%) 和 LMHead (9.1%) 占比显著增大**（batch=32 时分别是 20.9% 和 2.8%）
- EB 的最大开销是**冷路径 batchadd** (42.5ms, 占 EB 的 50%)
- hot_skip（M=5 跳过轮）开销几乎为零（1.3μs/call）
- routing 因 S_mask 检查变慢了 +13.5ms (+17.6%)

---

## 4. Expert 使用模式数据 (★ 核心采集结果)

### 4.1 整体统计

| 指标 | 值 | 含义 |
|------|-----|------|
| n_unique_experts avg | 160.9 | 实际被使用的 expert 数 |
| \|S_mask\| avg | 161.2 | 候选 expert 数 |
| **unique / \|S_mask\|** | **99.8%** | **候选 expert 几乎全部被使用** |
| count per expert avg | 212.3 | 每个 expert 处理的 token 数（跨所有 fwd 平均） |
| count_std / count_mean | ~1.08 | 分布不均匀（有超级热和冷 expert） |
| count_max | 1464 | 最热 expert 的 token 数 |
| top10_concentration | 26.3% | 最热 10 个 expert 占所有 pair 的比例 |
| adj_expert_jaccard avg | 0.968 | 相邻 forward 的 active expert set Jaccard |
| adj_count_correlation avg | 0.985 | 相邻 forward 的 per-expert token count Pearson 相关 |
| adj_top1_overlap avg | 0.825 | 相邻 forward 中保持同一 top-1 expert 的 token 比例 |
| cross_window_jaccard avg | 0.686 | 跨 M=5 窗口（S_mask 更新时）的 expert set Jaccard |

### 4.2 M=5 窗口内时间演化 (★ 核心模式)

| fwd_in_block | adj_expert_jaccard | adj_count_corr | adj_top1_overlap | top10_conc |
|------:|------:|------:|------:|------:|
| 0 (冷启动) | 0.686 | 0.730 | — | 0.324 |
| 1 | 0.971 | 0.852 | — | 0.408 |
| 2-4 (窗口内) | 0.982~0.995 | 0.997~0.998 | 0.63~0.68 | 0.36~0.39 |
| **5 (M=5 更新)** | **0.663** | 0.970 | 0.61 | 0.327 |
| **6-9 (窗口内)** | **1.0000** | **0.998** | **0.70~0.75** | 0.28~0.31 |
| **11-14** | **1.0000** | **0.998** | **0.78~0.82** | 0.24~0.26 |
| **21-24** | **1.0000** | **0.999** | **0.90~0.93** | 0.22 |
| **26-28** | **1.0000** | **0.9995** | **0.96~0.97** | 0.22 |
| 31 (block 末尾) | 1.0000 | 0.9999 | 0.99 | 0.21 |

**数据揭示的核心静态性模式**：

1. **Expert set 在 M=5 窗口内完全不变 (Jaccard=1.0)**：每 5 轮 forward 使用完全相同的 ~161 个 expert。
2. **Per-expert token count 极度稳定 (corr>0.998)**：每个 expert 的 workload 几乎不变。
3. **Token-expert 映射逐步收敛到近乎固定 (top1_overlap: 0.63→0.97)**：block 后期 97% 的 token 保持同一 top-1 expert。
4. **Token 分布从集中到均匀 (top10_conc: 0.41→0.21)**：block 后期 token 在 expert 间更均匀分布。
5. **跨 block 冷启动时 ~31% expert 更换 (cross_window_jaccard=0.686)**：S_mask 重建导致较大变化，但 69% expert 保持。

### 4.3 By Layer

| 层 | n_unique | adj_J | adj_corr | top1_overlap | top10_conc |
|---:|------:|------:|------:|------:|------:|
| L0 | 167.5 | 0.961 | 0.964 | 0.914 | 0.292 |
| L4 | 157.7 | 0.966 | 0.987 | 0.809 | 0.283 |
| L9 | 154.5 | 0.966 | 0.988 | 0.779 | 0.269 |
| L14 | 162.4 | 0.972 | 0.989 | 0.837 | 0.224 |
| L17 | 150.4 | 0.969 | 0.987 | 0.850 | 0.303 |
| L18 | 160.8 | 0.975 | 0.987 | 0.843 | 0.294 |

层间差异不大。L17 的 n_unique 最低（150.4），说明某些层的 expert 使用更集中。

---

## 5. D1 Token Skip → Expert Elimination 分析

### 5.1 背景

D1 (Token Temporal Reuse) 是 v0.1.14 的历史方向：用 token 的输出置信度（LM head softmax max prob）预测 stable token，跳过其 MoE 计算。

假设：如果被跳过的 token 恰好是某些 cold expert 的"唯一客户"，这些 expert 就不需要加载权重 → 将 token-level skip 转化为 expert-level memory saving。

### 5.2 实验结果

| Threshold | SkipRate | Unique(Before) | Unique(After) | Eliminated | HBM saved |
|------:|------:|------:|------:|------:|------:|
| 0.50 | 63.1% | 161.1 | 160.0 | 1.0 | 0.6% |
| 0.60 | 57.1% | 161.1 | 160.6 | 0.5 | 0.3% |
| 0.70 | 51.8% | 161.1 | 160.7 | 0.3 | 0.2% |
| 0.80 | 46.2% | 161.1 | 160.8 | 0.2 | 0.1% |
| 0.90 | 39.7% | 161.1 | 160.9 | 0.2 | 0.1% |
| 0.95 | 34.8% | 161.1 | 161.0 | 0.1 | 0.1% |

**结论：即使跳过 63% token，平均只消除 1 个 expert (0.6% HBM saving)。D1 → expert elimination 在 batch=128 下不成立。**

### 5.3 Stable Token ↔ Expert 热度相关性

| Expert 热度 (token count) | avg stable_fraction |
|------:|------:|
| <5 (最冷) | 0.470 |
| 5-20 | 0.432 |
| 20-50 | 0.418 |
| 50-200 | 0.462 |
| >200 (最热) | 0.399 |

Stable token 轻微偏好 cold expert (47.0% vs 39.9%)，但相关性太弱，无法产生有意义的 elimination。

### 5.4 根因

batch=128 下 N≈12800~45000 tokens，每层 pairs = N×8 = 102400~360000，分布在 ~161 个 expert 上。即使移除 63% token，剩余 token 仍然覆盖几乎所有 expert → 没有 expert 的 count 能降到 0。

---

## 6. Weight Compaction 探索

### 6.1 思路

将 active expert 权重从 `w[256, ...]` 紧缩到 `w_compact[161, ...]`，remap expert IDs 到连续 0-160。

### 6.2 结果

| 配置 | Latency | vs Original |
|------|------:|------:|
| Original [256, auto-tuned] | 1.771ms | baseline |
| Compact [158, default config] | 3.198ms | **+80.6%** |
| Compact [120] | 3.174ms | +79.2% |
| Compact [200] | 3.210ms | +81.3% |
| Sweep [256] (= original) | 1.841ms | +3.9% |

**结论：Compact 慢了 +80%，但不是 compact 本身的问题，而是 kernel 配置问题。**

### 6.3 根因：vllm fused_moe 的 auto-tune 依赖

- vllm 为 `E=256, N=512, H100` 提供了 auto-tuned kernel 配置（优化的 BLOCK_M/N/K, num_stages, num_warps）
- 对 E=158 等非标准值，没有配置文件 → 回退到保守的 default config → 性能暴跌
- S_mask 方案不受影响，因为它**不改变 E**（weight tensor 仍然是 [256, ...]），vllm 正常使用 E=256 的 tuned config
- S_mask 的收益来自：topk_ids 中只出现 ~161 个 unique ID → kernel 只加载这些 expert 的权重 → 减少 HBM 带宽

### 6.4 关键认知

**当前 S_mask + E=256 tuned kernel 已经是高效的组合**：tuned kernel 在 E=256 空间内高效运作，S_mask 通过控制 topk_ids 内容来减少实际 weight loading，不需要物理紧缩 tensor。

---

## 7. 其他已排除的方向

| 方向 | 结果 | 根因 |
|------|------|------|
| CUDA Stream Overlap (EB \|\| fused_experts) | 慢 +16~20% | fused_experts memory-bound (86% HBM 利用率)，K_A 争抢 HBM 带宽 |
| torch.compile | 失败 | DynamicCache 重编译 + Triton kernel graph break |
| pair-level 裁剪 (top-p) | 51.5% pair 节省，kernel 仅快 7% | 瓶颈是 unique expert weight loading，不是 GEMM 计算量 |
| 跨 forward 权重缓存 | 不可能 | 单个 expert 6.29MB >> H100 L2 50MB |
| D1 token skip → expert elimination | 最多消除 1 个 expert | batch=128 下 token coverage 太密集 |

---

## 8. fused_experts 的瓶颈本质

- **完全 memory-bound**：HBM 带宽利用率 86%，接近 H100 峰值 (3.35 TB/s)
- 每层每 forward：~161 experts × 6.29MB = ~1.01GB weight loading
- 唯一被验证有效的加速手段：**减少 unique active expert 数量**（S_mask: 256→161 → fused_experts -12.1%）
- vllm E=256 tuned kernel 在稀疏使用下已高效（不加载不在 topk_ids 中的 expert）
- 计算量不是瓶颈：128 tokens × 8 experts = 1024 pairs → 平均 6.4 tokens/expert → GEMM 利用率低但计算量极小

---

## 9. 开放问题：静态化/持久化加速方向

### 9.1 我们发现的可利用的"静态性"

在 M=5 窗口内（5 个连续 forward），以下特性被实验验证：

| 特性 | 量化 | 持续时间 |
|------|------|---------|
| Expert set 完全固定 | Jaccard = 1.0000 | 5 fwd (M=5 窗口) |
| Per-expert token count 稳定 | Pearson corr > 0.998 | 5 fwd |
| Token-expert top-1 映射稳定 | overlap 0.63 → 0.97 (随 block 收敛) | 渐进稳定 |
| Token count 分布结构性稳定 | std/mean ≈ 1.08, 不随 fwd 变化 | 整个 block |
| S_mask 本身跨 block 69% 保持 | cross_window Jaccard = 0.686 | 跨 block |

### 9.2 核心问题

**这些"静态性/可预测性"有没有办法转化为 fused_experts 的加速？**

fused_experts 的瓶颈是 HBM weight loading。我们已经通过 S_mask 减少了 unique expert 数（256→161）。进一步减少的空间有限（unique/|S_mask|=99.8%，几乎没有冗余 expert）。

### 9.3 我们想到但尚未验证的方向

1. **"固定执行路径"类 CUDA Graph 思路**：M=5 窗口内 expert set 完全不变，如果能固定 token-expert grouping（容忍微小近似），是否可以 capture 执行路径，消除 per-forward 的 dispatch/grouping 开销？
   - 障碍：fused_moe 内部的 token grouping 每轮因 routing 变化而不同（虽然变化很小）

2. **针对"已知 expert set + 稳定 workload"的专用 Triton kernel**：当前 vllm kernel 是通用的（任意 E, 任意 token 分布），如果 expert set 和大致 workload 已知，能否做更激进的 tiling/scheduling 优化？

3. **跨 forward 的增量计算**：97% token 保持同一 top-1 expert，hidden_states 也在逐步收敛。是否有算法变换能利用 `output_{t+1} ≈ f(output_t, Δhidden)` 来减少计算？
   - 障碍：expert 内部有 SiLU 非线性激活

4. **token count 分布不均匀的利用**：最热 expert 1464 tokens vs 最冷 1-2 tokens。是否有分层处理策略（热 expert 大 tile, 冷 expert 小 tile 或合并处理）？

5. **其他我们没想到的方向**？

### 9.4 约束条件

- 不能改变输出质量（已验证 128 prompt 全 PASS 的 C10-M5 配置为基准）
- 不改变 fwd 次数（282 fwd 为基准，允许 ±2）
- 硬件固定为 H100 80GB
- vllm fused_moe kernel 可以替换为自定义 Triton kernel，但需要保持功能等价

---

## 附录 A: M=5 窗口内 Expert 使用模式完整数据

```
fib  n_unique  adj_J    adj_corr  top1_ovlp  top10_conc  records
  0    160.9   0.6859    0.7302        —       0.3240     171
  1    156.6   0.9711    0.8515        —       0.4080     171
  2    158.8   0.9816    0.9972    0.6310      0.3891     171
  3    159.5   0.9922    0.9982    0.6640      0.3737     171
  4    160.0   0.9948    0.9984    0.6789      0.3586     171
  5    161.0   0.6626    0.9703    0.6132      0.3269     171
  6    161.0   1.0000    0.9980    0.7039      0.3127     171
  7    161.0   1.0000    0.9982    0.7203      0.3002     171
  8    161.0   1.0000    0.9982    0.7407      0.2887     171
  9    161.0   1.0000    0.9981    0.7536      0.2781     171
 10    161.0   0.9091    0.9961    0.7340      0.2684     171
 11    161.0   1.0000    0.9978    0.7753      0.2592     171
 12    161.0   1.0000    0.9978    0.7922      0.2511     171
 13    161.0   1.0000    0.9977    0.8038      0.2439     171
 14    161.0   1.0000    0.9977    0.8189      0.2378     171
 15    161.0   0.9264    0.9948    0.7928      0.2325     171
 16    161.0   1.0000    0.9979    0.8463      0.2285     171
 17    161.0   1.0000    0.9978    0.8579      0.2254     171
 18    161.0   1.0000    0.9981    0.8683      0.2231     171
 19    161.0   1.0000    0.9981    0.8797      0.2219     171
 20    161.0   0.9386    0.9945    0.8462      0.2208     171
 21    161.0   1.0000    0.9985    0.9014      0.2204     171
 22    161.0   1.0000    0.9987    0.9095      0.2206     171
 23    161.0   1.0000    0.9990    0.9216      0.2211     171
 24    161.0   1.0000    0.9991    0.9331      0.2218     171
 25    161.0   0.9378    0.9942    0.8782      0.2224     171
 26    161.0   1.0000    0.9995    0.9563      0.2230     171
 27    161.0   1.0000    0.9994    0.9599      0.2238     171
 28    161.0   1.0000    0.9997    0.9723      0.2247     171
 29    163.5   1.0000    0.9998    0.9782      0.2178     152
 30    163.5   0.9591    0.9969    0.9196      0.2184     152
 31    166.9   1.0000    0.9999    0.9923      0.2092      95
```

## 附录 B: D1 Token Skip → Expert Elimination 完整数据

```
Threshold  SkipRate  Unique(B)  Unique(A)  Eliminated  HBM_saved
    0.50     63.1%     161.1     160.0        1.0       0.6%
    0.60     57.1%     161.1     160.6        0.5       0.3%
    0.70     51.8%     161.1     160.7        0.3       0.2%
    0.80     46.2%     161.1     160.8        0.2       0.1%
    0.85     43.3%     161.1     160.9        0.2       0.1%
    0.90     39.7%     161.1     160.9        0.2       0.1%
    0.95     34.8%     161.1     161.0        0.1       0.1%

Stable token ↔ Expert hotness (threshold=0.90):
     Band   #entries  avg_stable_frac  avg_count
       <5      1934          0.4696        2.5
     5-20      9851          0.4317       12.5
    20-50     57508          0.4180       38.4
   50-200    477616          0.4617      111.0
     >200    261000          0.3986      417.7
```

## 附录 C: Weight Compaction Micro-bench 完整数据

```
配置                               avg(ms)  std(ms)  p50(ms)  w_shape
A) Original [256, auto-tuned]      1.771    0.043    1.785    [256, 1024, 2048]
B) Compact [158, default config]   3.198    0.012    3.196    [158, 1024, 2048]

Sweep:
  [120, default]                   3.174    0.033    3.158    [120, 1024, 2048]
  [140, default]                   3.183    0.013    3.183    [140, 1024, 2048]
  [180, default]                   3.206    0.056    3.200    [180, 1024, 2048]
  [200, default]                   3.210    0.017    3.208    [200, 1024, 2048]
  [230, default]                   3.206    0.017    3.202    [230, 1024, 2048]
  [256, auto-tuned]                1.841    0.042    1.854    [256, 1024, 2048]

Compaction cost: 0.78ms (one-time)
Remap cost: 0.020ms/forward

根因: vllm only has auto-tuned config for E=256.
      Non-standard E falls back to default config → +80% slower.
      S_mask avoids this by keeping E=256 tensor, only controlling topk_ids content.
```

## 附录 D: 组件级 Profiling 完整数据

```
=== C5 (batch=128, 91 fwd) ===
  fused_experts:   1550.2ms  17.035ms/fwd  43.2%
  Attention:       1054.1ms  11.583ms/fwd  29.4%
  LMHead:           326.2ms   3.585ms/fwd   9.1%
  gate_getlogits:   252.2ms   2.771ms/fwd   7.0%
  shared_experts:   108.0ms   1.186ms/fwd   3.0%
  routing:           76.6ms   0.842ms/fwd   2.1%
  RMSNorm_post:      70.9ms   0.779ms/fwd   2.0%
  RMSNorm_pre:       71.5ms   0.786ms/fwd   2.0%
  DenseMLP (L0):     40.2ms   0.442ms/fwd   1.1%
  residual:          31.4ms   0.345ms/fwd   0.9%
  Instrumented:    3587.8ms                100.0%
  Gap:              413.0ms

=== C10-M5 (batch=128, 91 fwd) ===
  fused_experts:   1362.4ms  14.971ms/fwd  39.0%
  Attention:       1051.9ms  11.559ms/fwd  30.1%
  LMHead:           325.1ms   3.573ms/fwd   9.3%
  gate_getlogits:   250.7ms   2.755ms/fwd   7.2%
  shared_experts:   107.0ms   1.175ms/fwd   3.1%
  routing:           90.1ms   0.990ms/fwd   2.6%
  EB_total:          84.6ms   0.930ms/fwd   2.4%
    cold subtotal:   49.5ms (zero_init 0.2 + K_A 6.5 + K_B 0.3 + batchadd 42.5)
    hot subtotal:    25.6ms (update 23.8 + skip 1.8)
  RMSNorm_post:      71.0ms   0.780ms/fwd   2.0%
  RMSNorm_pre:       71.5ms   0.786ms/fwd   2.0%
  DenseMLP (L0):     40.0ms   0.439ms/fwd   1.1%
  residual:          31.5ms   0.346ms/fwd   0.9%
  Instrumented:    3492.1ms                100.0%
  Gap:              409.8ms
```
