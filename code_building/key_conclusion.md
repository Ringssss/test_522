# Key Conclusions

## Living Insight Ledger

- 方向选择前优先查看 `/home/wuhang/wuhang/dllm_wh/docx/context_index/04_insight_ledger.md`。
- 该文件是当前 dLLM / MoE / EB insight 的 canonical 活文档，维护每条 insight 的机制、数据来源、状态和优化含义。
- 新实验如果改变了 insight 的状态，应优先更新该 ledger，再在本文件追加最高层结论。

## 2026-04-27 BSP-MoE 结论

- BSP-MoE 机制已在 C12 标准配置验证成立：
  - `dp=2,tp=4,ep=8,batch=512,block_length=32`
  - 每个 DP rank `local_bs=256`
  - block forward 的 MoE token 数是 `N_dp=256*32=8192`
  - BSP 后每个 TP rank 处理 `N_sp=2048` tokens，padding 为 0
- BSP-MoE 保持 EB/S_mask 路径语义：
  - baseline 与 BSP 的 path counts 完全一致：`prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`
  - 说明 block clock、cold/hot_update/hot_skip 触发机制没有被 BSP 改写
- BSP-MoE 的 dispatch payload 节省是真实的：
  - baseline dispatch payload `826.877 MB/fwd`
  - BSP dispatch payload `206.719 MB/fwd`
  - 这说明 TP 内 MoE token 冗余确实被去掉
- 当前 Python monkey-patch 版本只得到小幅端到端收益：
  - no-timing e2e: baseline `20.1285s / 75.675 ms/fwd`
  - BSP `19.8475s / 74.615 ms/fwd`
  - delta `-1.40%`
- 当前 blocker 不是理论 work reduction 不成立，而是实现和 collective layout 开销：
  - BSP component timing 中 combine 从 `3.587` 增到 `8.284 ms/fwd`
  - 新增 TP all-gather `2.618 ms/fwd`
  - component timing run 端到端反而 `+6.50%`
- BSP 输出质量肉眼检查未见 BSP 特有语义退化，但 forward-check 非 bitwise identical：
  - layers 0/9/18 的 path counts 一致
  - 典型 `abs_mean≈5e-4~1e-3`，`rel_max≈0.1%~0.7%`
  - 后续质量判断仍应坚持人工语义检查，而不是字符串匹配
- 方向判断：
  - BSP 比 Scheme3 standalone 更像高价值系统方向，因为它触及 TP group 内 shared/gate/native MoE 的冗余
  - 但下一步必须转向 native SP/MoE 集成或 collective layout profiling，不能继续在 Python monkey-patch 上叠小优化

## 2026-04-09

- `docx/` 当前应被视为分层文档库，而不是统一笔记目录：
  - 根目录负责当前阶段入口、恢复材料和稳定指南；
  - `cites/` 负责论文定位、理论框架和 POC 范围；
  - `articles/` 负责专题技术分析和阶段性技术归档；
  - `context_index/` 负责恢复导航和最小阅读路径；
  - `plans/` 预留给正式计划稿和可执行设计稿。
- 当前 `docx/` 中存在较多历史 `linear_wh` 绝对路径引用；它们可以继续作为上下文参考，但在真正执行命令、查代码或改文件前，必须先核实当前 live workspace 是否已经迁移到 `dllm_wh`。
- diffusion LLM 的工程本质不是“换一种模型就完了”，而是“step 数优化”和“每 step 计算范围优化”同时成立才会真正快。
- 当前本地代码与官方资料共同表明，text dLLM 最重要的加速层主要有六类：
  - 并行解码算法（threshold / hierarchy / M2T+T2T）
  - block-wise / semi-autoregressive diffusion
  - cache 近似与选择性重算（prefix / dual / vicinity / periodic refresh）
  - 连续 embedding 平滑（iteration smoothing）
  - runtime / batching / CUDA graph / compile / TP-EP-SP
  - 量化与 MoE kernel 优化
- 就当前生态成熟度看，`dInfer` 与 `SGLang` 是最值得直接对标的两条 text dLLM 推理路线：
  - `dInfer` 更偏 dLLM 专用算法与推理框架；
  - `SGLang` 更偏把 dLLM 纳入正式 runtime 和 scheduler。
- 如果下一阶段只能优先做一条线，最值得先攻的是“自适应 cache 刷新 + 动态并行解码阈值”的协同优化，而不是单独卷 kernel。
- 当前环境虽然能解析到本地 `dInfer` 源码路径，但 `import dinfer` 会因为缺失可选 serving backend（当前观察到的是 `vllm`）而失败；因此“最小可运行 demo”更适合采用自包含脚本，只依赖 `torch + transformers`。
- 当前环境已经可以正式跑通 dInfer + `LLaDA2.0-mini`：
  - 路线采用 `LLaDA2MoeModelLM + ThresholdParallelDecoder + BlockDiffusionLLM`
  - 配置为 `tp=1`、`prefix cache`、`block diffusion`、`threshold=0.95`
  - 实测 load time `8.9566 s`，generation time `1.9186 s`，吞吐 `38.05 tokens/s`
- 跑通正式链路前，实际需要处理三类兼容问题：
  - 缺失 `vllm`
  - `transformers` API 演化导致的 `is_torch_fx_available` / default RoPE 不兼容
  - 环境里的 `flash_attn` namespace 与 dInfer 旧代码假设不一致
- 历史归档中的正式 dInfer 成功运行仍然有效，但它已经不能直接代表当前 live 环境的可复验状态。
- 当前 live 环境在恢复后重新验证时出现了更明确的回归事实：
  - `sglang` 单独导入正常
  - 裸 `import dinfer` 会以 `std::bad_alloc` 退出
  - 缩短参数重跑 formal script 也会以 `std::bad_alloc` 退出
- 当前 live 环境与历史成功归档至少存在一个已确认漂移：
  - 历史记录中的 `torch` 为 `2.8.0+cu128`
  - 当前环境中的 `torch` 为 `2.9.1+cu128`
- 因此当前项目状态不应再描述为“已经可以直接继续 benchmark sweep”，而应描述为：
  - 有真实的历史成功结果
  - 但当前 live 环境需要先恢复可复验性
- 之前记录里的 `outlines/outlines_core` 冲突不是当前最新环境快照下的主要问题，因为当前 `outlines_core` 已是 `0.1.26`。
- 当前更合理的下一步不是直接扩 benchmark，而是先做环境回归定位：
  - 缩小 `dinfer` 顶层导入崩溃的触发链
  - 对齐当前与历史成功时的关键版本差异
  - 先恢复一条可复验 formal route，再继续实验扩展
- `std::bad_alloc` 崩溃的根因已确认定位：
  - vllm 0.10.2 的编译扩展 (`_C.abi3.so`) 与 torch 2.9.1 的 C++ ABI 不兼容
  - 崩溃路径：`dinfer/__init__.py` -> `serving.py` -> `from vllm import distributed` -> 加载 `_C.abi3.so` -> crash
  - 与 dinfer 的 patch、sglang、flash_attn 均无关
- 回退 torch 到 2.8.0+cu128 后，formal path 完全恢复：
  - 复验结果与历史归档一致（73 tokens / 47 forwards / 37.77 tokens/s）
  - 生成文本完全一致
- 关键教训：vllm 的编译 `.so` 文件与 torch 版本强绑定，torch 大版本升级必须同步重装或升级 vllm
- 当前项目状态已回到"可以继续做 benchmark sweep"
- 当前 best baseline 已建立：
  - 配置：`use_bd=True`, `cache=prefix`, `threshold=0.90`, `block_length=32`, `gen_length=128`
  - 性能：**70.6 tok/s**, 26.3 fwd/s, 51 forwards
- 关键发现：单次 forward 速度恒定 (~26 fwd/s)，性能差异完全由 forward 次数（step count）决定
  - threshold 0.90 -> 51 forwards；threshold 0.99 -> 78 forwards
  - 这意味着减少 step count 是最大的吞吐杠杆
- `torch.compile` 在 torch 2.8.0 + LLaDA2MoE 上触发 InductorError（`decompose_auto_functionalized` 阶段），当前不可用
  - 这是一个已知限制，后续 torch 升级可能修复
- 后续缓存优化方向：减少每步 forward 中的无效重算（block 内已确定 token 的 KV 复用）
- 全路径对比后确认：**BlockDiffusionLLMAttnmask（全重算、无 KV cache）是当前最快路径**
  - 短 prompt (55 tok): 76.8 tok/s；长 prompt (662 tok): 75.2 tok/s
  - 比 prefix cache 版本快 8-30%，fwd/s 几乎不受 prompt 长度影响
- prefix cache 在长 prompt 下 fwd/s 从 26.3 降到 19.7，cache 管理的 Python/显存拷贝开销反而成为瓶颈
- IterSmooth（迭代平滑）和 VicinityCache（邻域缓存）尚未适配 LLaDA2MoE：
  - IterSmooth 的 `h2e` 函数有 dtype 不匹配（float vs bfloat16），需要 cast 修复
  - VicinityCache 的 RoPE position embedding 尺寸与窗口输入不匹配
- 当前最终 baseline：BlockDiffusionLLMAttnmask, threshold=0.90, **75-77 tok/s**
- 后续优化最有价值的方向：在 attnmask 路径基础上，对 prompt + 已完成 block 引入选择性 KV 缓存，只重算因当前 block 变化而需要更新的 KV，兼顾精确度和效率

## 2026-04-10

- IterSmooth dtype bug 已修复：`modeling_llada2_moe.py:284`，`prob.to(self.W_e.dtype) @ self.W_e`
- VicinityCache position/replace_position bug 已修复：`generate_uniform.py:704-707` 和 `795-801`，覆盖 replace_position 为窗口范围并传显式 position_ids
- **IterSmooth 和 VicinityCache 在 BW 框架下均无法超过 BD-attnmask baseline**：
  - 短 prompt: BD-attnmask 78.1 tok/s vs IterSmooth 最好 56.9 tok/s
  - 长 prompt: BD-attnmask 75.7 tok/s vs BW 最好 43.5 tok/s
- IterSmooth 在 BW+长 prompt 下反而 **大幅增加** forward 次数（80→102），soft embedding 对未来 mask block 的 logits 做加权是噪声注入
- VicinityCache 没有减少 forward 次数，窗口化 KV 管理开销反而拖慢 fwd/s
- **根本瓶颈**：BW 系列 forward 次数（62-102）远多于 BD（48），来自"模型看到未来全 mask block"导致的置信度下降
- 下一步已设计但未实现的方案：**BD + IterSmooth 结合**——在 BD-attnmask 路径上只对当前 block 的 mask 位置做 soft embedding，保留 BD 的截断视野和低 forward 次数，同时用 soft embedding 进一步提升解码器置信度
- dInfer 中 prefix cache 和 dual cache 在 BlockDiffusionLLM 下实际等价（BlockDiffusionPrefixCacheManager 覆盖了 get_key_values，始终返回 block 范围）
- BlockDiffusionIteration 中 `kv_cache.update(output.past_key_values)` 每步全量重建 cache 是 BD+cache 路径的核心瓶颈（TODO 注释已承认不必要）
- generate_cache.py 中的"周期性刷新"思路（`cache_update_tag("fix_iter")`）是更合理的 cache 更新策略，但未适配到当前框架

- **BD + IterSmooth 结合方案已实现并测试**：
  - 在 BD-attnmask 路径上用 `inputs_embeds` 代替 `input_ids`，只对当前 block 的 mask 位置做 soft embedding
  - 新增 `IterSmoothBlockDiffusionIteration`、`IterSmoothBlockDiffusionLLMAttnmask`（no-cache）、`IterSmoothBlockDiffusionLLMCache`（cache）三个类
- **短 prompt 结论：BD+IterSmooth 成功超越 baseline +13%**
  - BD+IS high_w (cont_weight=0.5): 87.3-87.8 tok/s，forward 48→42
  - 这是首次在 BD 路径上实现 forward 次数的显著降低
- **长 prompt 结论：IterSmooth 在长上下文中无效甚至有害**
  - BD+IS high_w: 70.6-71.4 tok/s，forward 48→51（退化 -6.6%）
  - BD+IS default: 62.8 tok/s，forward 48→58（退化 -17.3%）
  - 原因推测：长上下文中模型条件已足够强，soft embedding 引入轻微噪声干扰收敛
- **KV cache 在 BD 路径上是确定的负优化**（第三次确认）：
  - cache 路径 fwd/s 从 ~26-27 骤降到 ~19-21（cache 管理开销压倒了计算节省）
  - BD+IS+cache 长 prompt 只有 49-50 tok/s，比 baseline 慢 34%
- **threshold_decay 是必要的**：无 decay 时 iter_threshold 恒为 1.0，每步最多解 1 token，forward 暴增到 116-121
- **h2e 额外开销可忽略**：fwd/s 恒定 ~26-27，与 baseline 无差异。O(block_len × V × d) 的 matmul 在 H100 上开销极小
- 性能差异 100% 来自 forward 次数——这一结论在所有测试中持续成立
- **dInfer 的 attention 是 transformers 库级别实现（eager/flash_attn/sdpa），不是 vllm 的 Paged Attention**
  - KV cache 管理用 `F.pad` 扩展、`slice_scatter` 回写、`torch.cat` 追加——与 transformers DynamicCache 等价
  - dLLM 需要 KV 原地改写（diffusion 迭代）而非 append-only，与 Paged Attention 的页表设计冲突
  - 这是 cache 路径始终为负优化的深层系统原因
- **Decoder 的 full softmax 可用 logsumexp 替代，但收益可忽略（~0.016%）**——真正的优化杠杆是解码策略本身（forward 次数）
- 替代解码策略方向（尚未实现）：logit margin 阈值、top-k 决策、block 级联合判断

## 2026-05-03 EPLB 因果链分析 + Kernel Micro-benchmark

- **模型参数修正**：hidden_size=2048（不是 2560），moe_intermediate_size=512，per-expert weight=6.29MB，FLOPs/token=6.29M
- **K=4 确认**：fused_routing 返回 K=4（覆盖模型 topk=8），T_avg = batch × 0.5
- **Grouped topk 消融**：去掉后负载 CV 不变（0.074→0.071），性能微快 1%。Grouped topk 不是隐式负载均衡器
- **Kernel Sweep (batch=128→2048)**：
  - batch=512 (T=256): memory-bound，kernel ~26.7 ms/fwd
  - 转换点 ~batch=900 (T≈450)
  - batch=2048 (T=1024): compute-bound，kernel ~78.8 ms/fwd
- **EPLB 在 compute-bound 下仍无效**：batch=2048 EPLB ON spread 从 1.26→7.24 ms（恶化），combine 变慢 100% 来自等待
- **Per-layer per-GPU timing**：实际 per-forward 不均衡成本 = Σ qa_gap = 0.93 ms (batch=512) / 3.14 ms (batch=2048)
- **因果分析**：per-layer Pearson r = +0.93（token count ↔ kernel time 强正相关）。Dampening = 0.06x（50% token 不均衡→仅 3% timing 不均衡）
- **Kernel 分布无关性**：kernel_time = f(total_pairs_per_GPU)，不依赖 pairs 在 experts 间的分布（Active Expert Sweep 验证）
- **2D Sweep (pairs × active_experts)**：低 pairs 时 active experts 有 10-23% 影响（weight loading 可见），高 pairs 时 <6%
- **Tiling Config 发现**：E=32/34/36 的 FusedMoE config 完全复制自 E=256，未独立 auto-tune。E=62/128/512 有不同最优 config
- **EPLB 无效的完整因果链**：统计平滑 (32 experts/GPU) + kernel 分布无关性 + memory-bound dampening (0.06x) + redundant experts 增大 spread → 净负优化
- **优化优先级**：(1) TEAM decoded-skip 减少 dispatch/combine token 数 ~18ms (24%); (2) tiling config auto-tune; (3) dispatch/combine 通信优化

## 2026-05-04 Tiling Config Auto-Tune + Dispatch/Combine 根因分析

- **Tiling config auto-tune 完成**：实际推理 M=16384（AgRs AllGather 全量 tokens），high-M grid search 仅 2-3% micro-bench 改善，端到端无可测改善。E=32 config 在实际 M 下已近最优
- **Pre-filter block waste 排除**：移除 87.5% early-exit blocks 反而退化 1.8%。Early-exit 在 H100 上零成本
- **小 tile config (M_eff=256) 排除**：block waste overhead (grid 8x 膨胀) 超过 tiling 收益，退化 2.8%
- **Joint Stats 联合数据**：pairs 跨 GPU 3-5x 不均衡但 kernel time 仅 5-7% 差异 (dampening 0.10x)。热点 GPU 完全稳定（确定性）
- **Dampening vs batch size**：0.10x (b512) → 0.11x (b1024) → 0.14x (b2048)，远不到线性
- **Dispatch+Combine 完整分解**：21.7 ms = NCCL 通信 7.8ms (36%) + straggler 等待 4.5ms (21%) + framework 开销 8.7ms (42%)
- **Combine straggler 分离**：插入 sync+barrier 后 combine 从 12.7ms 降到 8.2ms，4.5ms 转移到 barrier
- **NCCL micro-bench**：单次 AG 9MB = 0.249ms (253 GB/s)。fp8 半 payload 3.2x faster
- **torch.compile**：vllm 0.11.0 下不再 InductorError，但有 graph breaks + forward context 兼容问题
- **优化优先级更新**：(1) CUDA Graph ~8.7ms (11%); (2) fp8 通信 ~5ms (6.7%); (3) TEAM ~18ms (24%)

## 2026-04-11

- **Lazy cache update（跳过每步 DiffusionKVCacheManager.update）验证为无损**——forward 次数和生成文本与 existing cache 完全一致
- **Per-step manager.update() 不是 cache 路径慢的根因**——去掉后 fwd/s 无变化（短 27.3→27.3，长 19.8→19.8）
- **Per-layer KVCache.update() 的 slice_scatter 是短 prompt 上 cache 慢 2% 的原因**——inplace write 消除后差距从 2.2% 缩小到 0.3%
- **长 prompt cache 路径慢 23% 的根因是 GPU 利用率不足**——batch=1 时 32 tokens 喂不饱 H100 的 132 个 SM（MoE FFN、SDPA、LM head 并行度均不足）
- **大 batch 下 no-cache 路径因 O(batch×seq²) 计算崩塌**——长 prompt batch=32 fwd/s 从 26.3 暴降到 5.3
- **cache 路径 fwd/s 在大 batch 下始终稳定 ~20-22**——不随 batch 增大崩塌
- **交叉点**：短 prompt batch=16、长 prompt batch=8 时 cache 路径吞吐量开始超过 no-cache
- **batch=32 长 prompt：cache-opt 吞吐 1672 tok/s vs no-cache 460 tok/s（3.67x）**
- **对 serving 场景的关键结论**：batch=1 优化无 cache 单步速度；batch>=8 优化 cache 路径降低每步计算量。两者互补而非替代
- **IterSmooth 在大 batch 长 prompt 下 forward 次数显著降低**（52→36），可能在高并行度下更有效

## 2026-04-12

- **dLLM MoE 推理与 AR 存在三个核心结构性差异**，这些差异在所有现有 MoE 系统论文中尚未被识别或利用：
  - **Insight A: MASK token routing 集中** — MASK tokens 共享相同 embedding，经 attention 后 hidden state 仍高度相似（尤其浅层），预期 route 到高度集中的 expert 子集，造成 load imbalance 和 fused_moe kernel 效率下降
  - **Insight B: 跨迭代 MoE 计算冗余** — 一个 block 内 ~12 次迭代，每次所有 32 位置都过 MoE，但已解码位置 token 未变，其 MoE routing 和 output 大概率稳定；冗余率上界估计 ~91%
  - **Insight C: 天然批量效应** — dLLM 每步处理 block_size=32 tokens/请求（AR 只处理 1），dLLM batch=1 的 MoE 利用率 ≈ AR batch=32
- **多卡 EP 视角下这些 insight 影响更大**：
  - Insight A → MASK routing 集中导致跨 GPU 负载不均和 all-to-all 流量不均衡
  - Insight B → 跳过已解码位置不仅省 compute 还省 all-to-all 通信（多卡收益 > 单卡）
  - Insight C → dLLM 每步 all-to-all 通信量是 AR 的 block_size 倍
- **最有论文潜力的方向: "Iteration-Aware Selective MoE"** — 跳过已解码位置的 expert 计算，复用上一步结果；多卡下同时省 dispatch 和 all-to-all
- **与现有工作的差异化**：vs EARTH (AR result reuse, 收益有限) / vs X-MoE (空间 padding 冗余) / vs Diff-MoE (统计 locality 缓存) — 我们基于因果确定性做时间维度的计算跳过
- **三个待验证假设**（需实验确认）：
  1. MASK vs 已解码 token 的 routing 分布差异（per-layer entropy）
  2. 已解码位置跨迭代的 routing 稳定性（change rate ≈ 0?）
  3. 已解码位置跨迭代的 MoE output 相似度（cosine sim > 0.99?）

- **MoE routing 实证数据（异质 batch=32, temperature=0.7）**：
  - Insight A 稳健：MASK/Decoded entropy ratio 0.65-0.66，不随 batch 变化
  - Insight B 大 batch 更好：decoded routing change rate 10-27%（batch=32）vs 19-48%（batch=1）
  - 冗余率 51.5%（batch=32），后期迭代 80-95%
  - Output cosine sim 0.97-0.99 for decoded positions（所有设置）
- **Shared vs Routed 分解**：
  - routed/shared 量级比在多数层为 0.72-1.68（routed 不可忽略）
  - shared-only 近似 cosine sim 仅 0.57-0.95（不可行）
  - v1 缓存（full output）cosine sim 0.87-0.98（更好的近似基础）
- **逐层 ablation 关键发现**：
  - 单层缓存：18/19 MoE 层可安全缓存，唯一敏感层 Layer 18
  - 但这是建立在"其他层全量计算"前提上的——单层 safe ≠ 多层组合 safe
- **多层组合 ablation 根本性结论**：
  - 单独安全的层组合后全部失败（L1-3+L19 ✗, L6-13+L19 ✗, 等等）
  - 误差在层间传播太快，任何两个不相邻的 safe 层段组合都超过容错边界
  - 结论：**"跳过 stable 位置 MoE 计算"在当前 transformer 架构下不可行**
- **Stable Cache 优化方案的完整尝试与结论**：
  - v1（全层缓存）：token match 21%，完全失败
  - v2（Layer 18 防火墙 + 刷新）：仍然失败，根因是 cache 变旧后误差累积
  - ablation 中的 exact match 是因为每步都做全量计算（cache 始终 fresh）——不代表跳过计算安全
  - 根本矛盾：跳过计算 ↔ 保持 cache fresh，二者不可兼得
- **EARTH 论文启发**：
  - EARTH 解决的是 expert 搬运瓶颈（88% 时间在搬），我们的问题是计算瓶颈
  - 但其分层思想（重要/中等/不重要）、渐进式近似、离线校准阈值有借鉴价值
  - 可能的方向：不追求跳过计算，而是降低计算精度（如混合精度 MoE）或减少 forward 次数
- **下一步方向需重新评估**：
  - 路径 1：放弃 MoE 层计算跳过，转向减少 forward 次数（IterSmooth 已证明可行 -12.8%）
  - 路径 2：探索 MoE 层内的精度降级（如对 stable 位置用低精度 expert 计算）
  - 路径 3：转向 block 调度/batching 层面的优化（dynamic batching, block-level scheduling）
  - 路径 4：以 MoE routing insight 作为 characterization 贡献，不依赖于 compute-skipping

## 2026-04-13

- **Padding-Free MoE Kernel 无法超越 vllm baseline**：
  - 基于 X-MoE (SC 2025) PFT 方案实现完整 padding-free pipeline，正确性通过（cosine sim = 0.999994）
  - 性能反而更慢：batch=1 0.84ms vs baseline 0.53ms（慢 1.58x）
  - 根因：MoE kernel 瓶颈是 expert weight HBM loading，不是 padding compute
  - Per-expert GEMM 算术强度 ~1.5 MAC/byte，H100 平衡点 ~295，完全 memory-bound
  - padding 的额外计算几乎免费（tl.dot 对 mask=0 不耗时间），消除 padding 无法减少 weight loading
  - vllm kernel 有 H100 专用 autotuning（从 JSON config 加载精细调优参数），我们的 grouped GEMM 缺少
- **Forward 时间拆分**：
  - Routed MoE (fused_experts): 13.9ms (37%)
  - Attention + Norm + Embedding + LM head: 22.0ms (60%)
  - Shared expert + Gate: 1.1ms (3%)
  - MoE 不是唯一瓶颈，Attention 占比更大
- **v0.1.13 总体结论**：
  - 三个 Insight 作为 characterization 贡献成立（所有设置下稳健）
  - "跳过计算"路径（Stable Cache）因误差累积不可行
  - "消除 padding"路径（Padding-free kernel）因 memory-bound 无法超越 baseline
  - 仍有效的方向：减少 forward 次数（乘性收益）、多卡 EP 场景（Insight A+B 影响放大）、EARTH 分层/渐进思想

## 2026-04-13 ~ 2026-04-14 (v0.1.14 MoE Selective Recompute Risk Proxy)

### 方向一：Token-level Temporal Reuse

- **token_confidence (AUC=0.900) 是最强的 MoE skip risk 预测器**，token_margin (0.895) 紧随其后
- routing 信号（gate_cos=0.696, topk_overlap=0.634）预测力弱于 token 状态信号
- 特征集 ablation: token_state_only (0.887) ≈ all_signals (0.894)，routing 信号几乎无附加价值
- **单点安全 ≠ 组合安全**：全量组合复用时误差放大 5-37 倍（通过 attention 层间传播）
- **Layer-range 非均匀性**：中层(L8-L11)最安全，深层(L14-18)最危险（直接影响 logits，无后续修正机会）
- **Drift guard (shared_cos) 能解锁更激进的 margin 阈值**：margin>0.70+drift<0.02 比 margin>0.95 无 guard 更优
- 两个备选配置：Config-S (margin>0.90, L4-14) / Config-D (margin>0.70, L4-14, drift<0.02)
- 理论节省 ~8%（仅覆盖 stable tokens）

### 方向二：Expert-level Adaptive Pruning (top-p)

- **shared expert 贡献 41.9%，routing expert 贡献 58.1%** — shared 提供了强 baseline
- **LLaDA2.0-mini 的 MoE routing 高度冗余**：只保留 top-4 expert（39% routing 权重）就能保持质量
- **自适应 top-p 完胜 fixed top-k**：
  - top-p=0.80: ΔFwd=0, 42.6% expert 节省（零损失配置）
  - top-p=0.75: ΔFwd=-2, 51.5% expert 节省（decoder 收敛更快）
  - fixed top-4: ΔFwd=+1, 50% expert 节省（不如 top-p=0.75 用更少 expert 但更好结果）
- **去掉低权重 expert 的噪声反而帮助 decoder 收敛** — 这是一个反直觉但强有力的发现

### 工程发现

- **Python monkey-patching hook 开销 > 计算节省**（batch=32 下慢 15-20%）
- 需要把逻辑内联到原生代码中才能兑现收益
- fused_moe kernel 对 weight=0 的 expert 仍然 dispatch（不会自动跳过）
- 真正省计算需要 kernel 层面支持或改变 top_k 参数

## 2026-04-14 (v0.1.15 D1×D2 耦合优化)

### 耦合架构验证

- **D2（top-p）作为底座 + D1（temporal reuse）作为叠加层** 的架构已完成实验验证
- 耦合决策逻辑：stable token → 全跳（D1）；非 stable → top-p pruning（D2）

### 假设验证结果

- **假设 B（proxy 有效性）：成立** — reuse rate 从 22.1% 降到 20.9%（比率 0.95），proxy 在 pruned 环境下仍有效
- **假设 C（误差独立性）：被打破** — D1 独立 ΔFwd=-4, D2 独立 ΔFwd=-2, 预期叠加 -6, 实际耦合 +4（偏差 +10）
  - 两种近似的误差方向相关，通过 attention 层间传播后超线性放大
  - 这是计划中预测的"误差共振"场景

### 耦合仍有实用价值

- 误差放大有界（ΔFwd=+2 到 +4），输出文本仍连贯正确
- 最佳安全耦合配置：
  - **D1+D2:tp75_m90**: ΔFwd=+2, expert savings=61.5%
  - **D1+D2:tp80_m70_d02**: ΔFwd=+2, expert savings=55.0%
- 即使多 2-4 次 forward，绝对 expert 计算节省约 60%

### Pareto 前沿

- **质量优先**：D2:tp75 单独（51.5% 节省, ΔFwd=-2）
- **均衡方案**：D1+D2:tp75_m90（61.5% 节省, ΔFwd=+2）
- **最大节省**：D1+D2:tp70_m70_d02（66.2% 节省, ΔFwd=+4）

### 理论计算节省的全局图景

| 配置 | Expert 节省 | ΔFwd | 适用场景 |
|------|------------|------|---------|
| D2:tp75 单独 | 51.5% | -2 | 最安全，单独即可 |
| D1+D2:tp75_m90 | 61.5% | +2 | 愿接受微小质量退化 |
| D1+D2:tp70_m70_d02 | 66.2% | +4 | 追求最大节省 |

## 2026-04-14 (v0.1.15.2 Step 1: Kernel Micro-Benchmark)

### 门槛测试结论：情况 γ-worst

- **weight=0 完全不省 kernel 时间**：B/A ≈ 1.00（所有 token 数下）
  - 原因：fused_moe_kernel 的 weight 乘法在 GEMM 循环之后（line 468-472），所有 expert weight loading 和 GEMM 先完成，最后才乘 weight
  - moe_align_block_size 只看 topk_ids 不看 weights，grid size 不变
- **物理裁剪到 top-4 效果也很有限**：
  - N=32: C/A=0.976（几乎无效）
  - N=256: C/A=0.849（省 15%）
  - N=1024: C/A=0.929（省 7%）
- **根因：kernel 完全 memory-bound，瓶颈是 expert weight HBM loading**
  - 减少 token-expert pairs 不减少 unique active expert 数（N=1024 下 top-4 仍有 245/256 active）
  - 每个 active expert 的 weight 都要从 HBM 加载（w1=4MB + w2=2MB per expert）
  - 只有大幅减少 active expert 数才能省 weight loading
  - top-1 在 N=1024 下降到 170/256 active → D/A=0.650（省 35%）
- **对 top-p pruning 的含义**：
  - top-p=0.75 平均保留 5 个 expert，但 active expert 数几乎不变（247/256 vs 248/256）
  - 因此 top-p 的 51.5% "expert 节省"在 wall-clock 上接近 0
  - 51.5% 节省的是 GEMM FLOPs，但 kernel 是 memory-bound，FLOPs 不是瓶颈

## 2026-04-27 (v0.1.15.12k BSP-MoE nsys collective profiling)

- **BSP-MoE 的理论节省真实存在，但当前 monkey-patch 路径被 collective 开销吃掉**：
  - 完整 C12 component timing 中，dispatch payload 从 `826.877 MB/fwd` 降到 `206.719 MB/fwd`，下降 `75.00%`
  - 短 nsys trace 中，Device-to-Device memcpy 总量从 `41222.3 MB` 降到 `21307.7 MB`，下降 `48.31%`
  - dense GEMM rankmax ms/fwd 从 `0.441` 降到 `0.213`，下降 `51.74%`
- **BSP 的主要 blocker 是 NCCL Reduce/AllGather 放大**：
  - 短 nsys trace 中，NCCL kernels rankmax ms/fwd 从 `7.993` 增加到 `17.114`，增加 `114.12%`
  - `NCCL_AllGather` rankmax ms/fwd 从 `1.837` 增加到 `5.962`，增加 `224.53%`
  - `NCCL_Reduce` rankmax ms/fwd 从 `2.381` 增加到 `7.573`，增加 `218.05%`
  - `NCCL_AllGather` count 从 `9120` 增加到 `18392`，符合 BSP 路径中 AgRs dispatch all-gather 加显式 TP all-gather 输出收集的行为
- **AgRs combine 变慢与 nsys 证据对齐**：
  - 完整 C12 component timing 中，`moe.combine` 从 `3.587 ms/fwd` 增加到 `8.284 ms/fwd`，增加 `130.95%`
  - 短 nsys trace 中 NCCL Reduce/AllGather 同时放大，因此 combine 慢不是单次测量噪声
- **vLLM AgRs group 切换解释了 collective 变化**：
  - `AgRsAll2AllManager.dispatch/combine` 在 `is_sequence_parallel=False` 时使用 `get_dp_group()`
  - 在 `is_sequence_parallel=True` 时使用 `get_ep_group()`
  - BSP 把 MoE 计算转入 sequence-parallel FusedMoE 后，collective 从 baseline DP/TP reduce 结构变成 EP group AgRs + 显式 TP output all-gather，导致收益转移为更重通信
- **下一步方向结论**：
  - 不应继续在 Python monkey-patch 层叠加 Scheme3 等小优化
  - 优先做原生 BSP/SP MoE integration、combine/gather 融合或重排、以及让 MoE 输出保持 sequence-parallel 更久，目标是减少新增 NCCL AllGather/Reduce

## 2026-04-27 (v0.1.15.12m M1/M2/M3 BSP-DelayGather)

- **当前实测最优是 M1+M2 的保守 delayed gather 路径**：
  - C12 no-timing baseline A: `20.227s / 76.04 ms/fwd`
  - B BSP-MoE: `19.898s / 74.80 ms/fwd`，`-1.63%`
  - C BSP-DelayGather: `19.777s / 74.35 ms/fwd`，`-2.22%`
  - D BSP-DelayGather-M3EPReduce: `19.946s / 74.98 ms/fwd`，`-1.39%`
- **M2 保守路线有效但收益仍小**：
  - C 相比 B 进一步降低 `native_forward/quant_apply/combine`
  - 但仍保留显式 TP all-gather 和较重 AgRs dispatch/combine，因此只有约 `1.023x` 加速
- **M3 三路径接口可行，但当前 hook 位置不兑现收益**：
  - D 的 `path_counts` 在 8 rank 一致：`prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`
  - D 的 `ep_reduce_calls=1862`、`ep_reduce_mb=1.907` per rank
  - 说明 cold/hot_skip/hot_update 兼容性成立
  - 但因为 routing 仍在 FusedMoE dispatch 之后，M3 只是新增 expert-wise pop 同步，没有减少 token-level hidden/router logits AgRs 通信
- **源码下沉优先级**：
  - 优先下沉 C 路径：attention full layout，MoE 前 chunk，MoE + residual 局部操作保持 SP，attention 前 gather
  - M3 暂缓作为收益主线，只保留接口验证；要兑现 M3，必须把 EB pop combine 移到 token-level AgRs gather 之前或更深的 routing 边界

## 2026-04-28 (v0.1.15.12n BSP C+ / C++ Upper-Bound)

- **当前 BSP 路线最优实测从 C 提升到 E/C+**：
  - 两轮 C12 no-timing 平均 baseline A: `75.73 ms/fwd`
  - C BSP-DelayGather: `74.14 ms/fwd`，`-2.10%`
  - E BSP-CrossLayerSP: `71.80 ms/fwd`，`-5.20%`
  - F BSP-AllReduceFullProbe: `72.46 ms/fwd`，`-4.32%`
- **E/C+ 的收益原理**：
  - 继续延长 SP layout 生命周期，把 hidden state 跨 sparse decoder layer 保持 SP。
  - attention 输入前才 gather，最后 model norm 前保证 full tensor。
  - E 没有继续降低通信 payload：dispatch 仍 `206.719 MB/fwd`，TP gather 仍 `165.375 MB/fwd`。
  - 因此 E 的收益主要来自 layout 生命周期延长带来的 native/quant/combine 与调度开销下降，而不是单纯减少字节量。
- **E/C+ 是下一步源码下沉主线**：
  - 它比此前 C 保守 delayed gather 多兑现约 `3.1` 个百分点。
  - A-F path counts 全部一致：`prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`，说明不需要改变 EB/s_mask 算法。
  - 小规模 quality smoke 未见灾难性语义崩坏，但生产结论前仍需完整质量集。
- **F/C++ 是 collective 设计证据，不是 production 方案**：
  - F 通过跳过 native combine 和显式 TP gather，把 `native_forward` 降到 `29.195 ms/fwd`，说明 combine+gather 融合/重排有真实 upside。
  - 但 F 的 raw EP full all-reduce payload 达 `1311.020 MB/fwd`，`ep_full_all_reduce=9.467 ms/fwd`。
  - 所以 production 方向应是 hierarchical collective 或 fused reduce_scatterv+gather，避免 raw full-buffer all-reduce。
- **M3 继续保持接口验证定位**：
  - hot_update `pop[E]` all-reduce 兼容性已成立。
  - 真正收益仍要等 EB pop combine 前移到 token-level AgRs gather 之前；不要在 M1/M2/E 下沉前单独改变 EB 算法。

## 2026-04-28 (v0.1.15.12o BSP-G vLLM SP-Parity)

- **BSP-G 是当前最强的可搬运 BSP 路线**：
  - 两轮 C12 no-quality 平均 baseline A: `76.03 ms/fwd`
  - E/C+ BSP-CrossLayerSP: `71.63 ms/fwd`，`-5.78%`
  - G BSP-G-AttnReduceScatterSP: `69.53 ms/fwd`，`-8.55%`
  - G 比 E 继续快 `2.93%`，约 `2.10 ms/fwd`
- **G 的收益原理是 attention output 直接进入 SP layout**：
  - G 把 attention output projection 的 TP all-reduce/full layout 输出改成 TP reduce-scatter/SP layout 输出
  - 后续 residual、post-attention norm、MoE 直接在 SP layout 上执行
  - component timing 中 `moe.bsp_chunk` 从 E 的 `0.872 ms/fwd`、count `5320` 降到 G 的 `0.003 ms/fwd`、count `266`
- **G 不是减少 MoE payload，而是移动同步边界并延长 SP 生命周期**：
  - G 新增 `attn.tp_reduce_scatter=5.020 ms/fwd`
  - G 新增 `attn_rs_payload=661.502 MB/fwd`
  - G 的 MoE dispatch payload 仍为 `206.719 MB/fwd`，TP gather payload 仍为 `165.375 MB/fwd`
  - 所以 BSP-H/F2 的 combine+gather 融合/重排问题仍未解决
- **EB/s_mask 与 BSP-G 兼容**：
  - C12 A/B/C/D/E/G/F path counts 全部一致：`prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`
  - 不需要单独改变 EB 算法
- **下一步优先级调整**：
  - BSP-G 应优先于 BSP-H/F2 做源码级 feature flag 设计
  - 先完成 attention reduce-scatter + SPHiddenState layout metadata 的原生表达和完整质量验证
  - BSP-H/F2 继续保留为后续 hierarchical/fused collective 方向，不在当前阶段提前启动

## 2026-04-28 (v0.1.15.12p BSP-G2 vLLM SP-Parity Bundle)

- **BSP-G2 已完成实验闭环，但不是新的性能台阶**：
  - C12 no-quality: A `75.428 ms/fwd`，E `71.816`，G `69.676`，G2 `69.661`，F `72.380`
  - G2 vs A 为 `-7.646%`
  - G2 vs G 只快 `0.015 ms/fwd`，属于测量噪声内持平
  - 所以性能参考仍应以 G 为主，而不是把 G2 视作比 G 更快的新路线
- **G2 的有效贡献是源码组织边界更干净**：
  - G2 让 attention path 同时负责 attention-input gather 和 attention-output reduce-scatter
  - component timing 中，G 的 `moe.tp_all_gather=2.662 ms/fwd,count=5054`
  - G2 变为 `attn.input_all_gather=2.573 ms/fwd,count=4788`，残留 `moe.tp_all_gather=0.141 ms/fwd,count=266`
  - 这证明 vLLM/SP-parity 的 boundary ownership 搬运成功
- **G2 没有减少通信字节或同步点**：
  - G 的 `tp_gather_payload=165.375 MB/fwd`
  - G2 变成 `attn_input_gather_payload=156.671 MB/fwd` 加 `tp_gather_payload=8.704 MB/fwd`
  - G/G2 都有 `attn_rs_payload=661.502 MB/fwd` 和 `dispatch_payload=206.719 MB/fwd`
  - 所以 G2 是 bucket 迁移/组织等价，不是 BSP-H/F2 那类 collective 融合
- **component timing 下 G2 反而慢于 G**：
  - G `78.149 ms/fwd`
  - G2 `79.786 ms/fwd`
  - 差值 `+1.637 ms/fwd`
  - 这更支持“G2 是组织方案，不是性能方案”的判断
- **EB/s_mask 与 G2 兼容**：
  - C12 e2e 和 component timing 中 A/E/G/G2/F 全部保持 `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`
  - 小 batch smoke 中 E 仍出现 `hot_skip=893`，但 C12 正式 invariant 正常；记录为小 batch artifact，不影响 C12 结论
- **下一步判断**：
  - 若目标是“现在能吃的性能”，继续以 G 为 measured-best
  - 若目标是“源码下沉时边界更清晰”，G2 的 ownership 设计可作为参考
  - 不应因为 G2 结果直接进入 BSP-H/F2；真正下一步必须减少/fuse 同步点或继续延长 SP lifetime，而不是单纯迁移 gather 归属

## 2026-04-28 (v0.1.15.12q vLLM SP-Parity Inventory)

- **新增持续确认清单**：
  - `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12q-vllm_sp_parity_inventory.md`
- **核心原则**：
  - 不能因为完成 G/G2 就声称 vLLM SP-parity 全部搬完。
  - 必须逐项确认 vLLM 中已有的 SP-MoE、communication backend、compilation SP、residual scattered、CUDA graph/static-size 支持。
- **优先确认项**：
  - LLaDA2 shared expert / dense MLP 在 SP 下是否还有 TP 冗余。
  - 当前环境可用的 vLLM MoE communication backend 及其 SP 支持。
  - vLLM compilation-level sequence parallelism pass 是否能迁移或手工等价。
  - source landing 时 residual scattered metadata、static sizes、CUDA graph 约束。
- **当前判断**：
  - G 仍是 measured-best 性能路径。
  - G2 是 ownership parity，不是性能点。
  - 下一轮讨论/分析应围绕 inventory 表逐项确认，而不是直接开始新代码建设。

## 2026-04-28 (v0.1.15.12r BSP-G3 to G7 vLLM SP Completion)

- **G3 / VSP-06 结论**：
  - vLLM DeepSeek/Llama4 的 shared expert SP 处理核心是 `disable_tp=is_sequence_parallel`，避免 TP Linear 的结果 all-reduce。
  - 当前 LLaDA2-mini 的 shared expert 和 dense-only MLP 都是 replicated `nn.Linear`，不是 TP Linear。
  - BSP-G 已经在 `hs_sp` 上运行 shared expert，因此没有遗漏一个可直接搬的 shared expert 性能点。
- **G4 source-downshift 结论**：
  - 首版源码下沉目标应是 measured-best BSP-G，而不是 G2。
  - G 的核心收益是 attention `RowParallelLinear` 输出从 TP all-reduce/full layout 改为 token-axis reduce-scatter/SP layout。
  - G2 的 attention-input gather ownership 更干净，但没有新性能收益，只作为 source organization 参考。
- **G5 AsyncTP 结论**：
  - vLLM `SequenceParallelismPass` / `AsyncTPPass`、`torch.ops.vllm.*`、`torch.ops.symm_mem.fused_*` 在当前环境可用。
  - 但当前 BSP-G monkey-patch benchmark 没进入 vLLM compile pass manager，所以没有实际吃到 AsyncTP fusion。
  - BSP-G/G2 源码形态概念上接近 `GEMM+ReduceScatter` / `AllGather+GEMM`，source landing 时应保留这种图形态。
- **G6 backend 结论**：
  - AgRs `allgather_reducescatter` 是当前默认和已有性能参考。
  - FlashInfer all2all capability 存在，是唯一较低风险的后续 isolated smoke backend。
  - DeepEP 当前 ABI-broken：`deep_ep._C` 缺 `ncclTeamWorld`，不能测试。
  - PPLX 未安装；FlashInfer Cutlass fused MoE 不是当前 bf16 BSP-G 的直接通信 drop-in。
- **G7 source landing 约束**：
  - source landing 必须显式携带 SP layout metadata 和原始 `N`，不能只靠 tensor shape 推断。
  - static/cudagraph sizes 必须是 TP multiple；C12 `N=8192,tp=4,N_sp=2048` 是安全形状。
  - 避免 forward-time module/buffer mutation，不能延续 monkey-patch 式 per-forward 状态修改。
- 所有后续 source/backend 实验仍必须保持 C12 path counts：`prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`。

## 2026-04-30 (v0.1.15.15 E=34 tuned-path unblock phase-1)

- 已通过补齐 `E=34,N=512` 配置文件消除 P5 C 路径的 default/fallback MoE config 警告，确认运行时命中新增配置文件。
- 但“直接复用 E=32 模板到 E=34”在当前环境未带来性能改善：
  - A: `144.381 ms/fwd`
  - C: `157.488 ms/fwd`
  - C vs A: `+9.078%`（较前次 `+5.689%` 更差）
- 因此当前性能负收益不再可归因于“缺少 E34 config 文件”，而是“E34 首版参数不匹配实际算子路径/负载特征”。
- 下一步应将 E34 路径视作独立调优对象（轻量 sweep），并与 P1/P5 联合编排设计并行推进；不能仅凭“fallback消除”判断 P5 已具备正收益。

## 2026-04-30 (v0.1.15.15 P1/P5 联合编排首版)

- P1/P5 已在源码层打通首版联合编排：不再需要“replica 下忽略 non-identity external map”的临时保护。
- 本次实现保持 `physical_to_logical` canonical 语义不变，仅重排 physical id 的 rank 归属（base 受 P1 控制，冗余按 round-robin）。
- A/B/C 短测结果（`batch=128,gen=32,tp=4,world=8`）：
  - A: `140.744 ms/fwd`
  - B: `137.330 ms/fwd`（相对 A `-2.425%`）
  - C: `155.162 ms/fwd`（相对 A `+10.244%`，相对 B `+12.984%`）
- 对比旧 C（e34cfg）：
  - `157.488 -> 155.162 ms/fwd`，仅 `-1.477%` 小幅改善，未改变“C 仍慢于 A/B”的结论。
- A/B/C path counts 一致（`19/38/893/209`），说明 EB 语义与 path schedule 未被破坏。
- 结论：当前主收益仍来自 P1；P5 的收益兑现仍受其他开销主导，下一步要做 replica 开销拆解与 EB-aware 副本质量优化，而不是继续改 EB 核心语义。

## 2026-04-30 (v0.1.15.15 C 慢因子归因拆解)

- 已完成 6 组同口径归因矩阵（A0/C0/C1/B0/C2/C3），并保持 path counts 全一致（`19/38/893/209`）。
- 归因结论明确：
  - `E=34` 形状本身只贡献小幅开销（约 `+2.46% ~ +2.72%`）。
  - runtime EPLB logical->physical map 路径贡献主要额外开销（约 `+8.56% ~ +13.75%`）。
- 因此“C 慢”的主导项不是 E34 config 缺失或 E34 形状，而是 runtime map 机制/实现开销。
- 优先级建议更新：
  - 先优化 runtime map 路径（含副本命中质量与 remap 开销），
  - E34 config 微调作为次优先级并行项即可。

## 2026-04-30 (v0.1.15.15 runtime remap record gating)

- 在 C3（P1+replica）口径下，若保持 logical->physical 映射不变，仅把 runtime EPLB record 从 `full` 降到 `cold_only`，两次复测平均可回收约 `5.03%` 时延（`151.750 -> 144.117 ms/fwd`）。
- 同口径下 `off`（全程 map-only、无 record）平均仅回收 `0.70%`（`151.750 -> 150.684 ms/fwd`），显著弱于 `cold_only`。
- 六次 run path counts 全一致（`19/38/893/209`），说明该收益来源于 runtime EPLB record 策略而非 EB 语义漂移。
- 工程含义：保守主线可优先推进“cold 保留完整 record，hot 减少/关闭 record”下沉实现，风险低且收益显著。

## 2026-04-30 (v0.1.15.15 cold-gated rearrange script PoC)

- cold-gated rearrange PoC 已完成接口修复并可稳定触发：
  - `proxy.expert_weights` 改为每层 `(w13_weight, w2_weight)` 后，`rearrange` 不再失败；
  - smoke 成功触发 1 次，单次重排成本约 `3.2~3.3s`。
- 在 C3 正式口径（`batch=128,gen=32,tp=4,world=8,P1+replica,record_mode=cold_only`）下：
  - base: `149.587 ms/fwd`
  - rg8: `201.459 ms/fwd`（`+34.677%`）
  - rg16: `201.898 ms/fwd`（`+34.970%`）
- 三组 path counts 全一致（`19/38/893/209`），说明语义未漂移；性能退化来自重排固定成本本身，而非 EB 路径变化。
- 结论：
  - 当前阶段不应把 cold-gated rearrange 放入在线默认路径；
  - 继续以 `record_mode=cold_only` 作为主线收益兑现方案；
  - 重排仅保留为离线/更低频窗口策略，或后续等待更低成本的 native 执行路径再复评。


## 2026-04-30 (v0.1.15.15 runtime map impl 优化)

- 在 C3（P1+replica，`record_mode=cold_only`）口径下，runtime EPLB map 的实现方式本身是可兑现收益点。
- 将 map 从 vLLM 基线实现替换为 `flat-index eager` 后，严格串行两轮复测结果：
  - r1: `147.146 -> 136.931 ms/fwd`（`-6.94%`）
  - r2: `152.264 -> 141.412 ms/fwd`（`-7.13%`）
  - mean: `149.705 -> 139.171 ms/fwd`（`-7.04%`, `-10.53 ms/fwd`）
- `flat_compile` 在当前环境是负优化（smoke 明显慢于 vLLM），当前不适合作为主线。
- 在 `flat_eager` 新路径下，hot-replica ids（top16）未带来叠加收益，反而退化到 `149.572 ms/fwd`；说明当前阶段应优先押注 map 内核实现/下沉，而不是 replica id 先验。
- 语义不变性保持：所有正式对比 run 的 path counts 均为 `prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`。
- 现阶段最稳妥主线：
  - 保留 `record_mode=cold_only`
  - 推进 `flat-eager` map 逻辑的 source-level 下沉
  - 暂不默认启用 hot-replica ids
  - 后续再评估能否进一步逼近 `B0(P1,E32)=132.759 ms/fwd`。

## 2026-04-30 (v0.1.15.15 runtime map source-downshift 完成)

- 已完成从“脚本 monkey-patch”到“dInfer 源码策略接口”的下沉闭环：
  - `modeling_llada2_moe.py` 新增 `configure_eplb_runtime_map_policy / set_eplb_runtime_route_path / use_eplb_runtime_map_policy`。
  - `collect_eb_heteval512_laws.py` 不再内置 map patch，只负责设置策略与 route-path。
- C3（P1+replica，`record_mode=cold_only`）串行两轮复测：
  - r1: `vllm=155.228`, `flat_eager=143.272`, `-7.70%`
  - r2: `vllm=154.932`, `flat_eager=139.569`, `-9.91%`
  - mean: `155.080 -> 141.421 ms/fwd`, `-8.81%`
- 语义与负载不变性同时满足：
  - path counts 四组全一致：`prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`
  - `eplb_load_balance_diag` 的 `overall_skew / layer_skew` 完全一致。
- 结论：
  - 收益来源依然是 runtime map 实现开销降低，而不是负载重分配或 EB 语义变化；
  - `flat_eager + cold_only` 可作为 dInfer 主线默认候选，后续优化应集中在继续压缩 runtime map/record 常驻开销与低成本 remap 策略。

## 2026-04-30 (v0.1.15.15 native gate 可用性 + tensor_cache A/B)

- **当前运行态 vLLM 不支持 `should_record_tensor` 路径门控**：
  - `native_record_gate` 统计在所有 run 中均为 `available=False,total_calls=0`。
  - 因此通过 `native_record_gate_mode` 做 cold/hot 记录门控在当前环境不可用，不应继续作为主线。
- **`flat_eager` 的 runtime tensor cache 在该 workload 为负优化**：
  - 严格 C3 A/B（两轮）：
    - `tensor_cache=off`: `139.116`、`138.536`，mean `138.826 ms/fwd`
    - `tensor_cache=on`: `172.328`、`146.131`，mean `159.230 ms/fwd`
  - `off` 相比 `on` 平均快 `14.70%`。
- **主线配置更新**：
  - `flat_eager + cold_only + tensor_cache=off`。
  - 相比旧 `flat_eager` 均值（`141.421`）再降约 `1.83%`。
  - 相比 `vllm` 均值（`155.080`）当前总回收约 `11.71%`。
- **语义不变性维持**：
  - 所有关键对比组 path counts 保持 `prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`。

## 2026-04-30 (v0.1.15.15 runtime fastpath 去动态化)

- 在 `flat_eager + cold_only + tensor_cache=off` 主线下，进一步做“热路径去动态化”（专用 `_patched_fast` + `path_is_cold` 布尔态）可稳定兑现额外收益。
- 同版本可控 A/B（`DINF_EPLB_RUNTIME_FASTPATH`）两轮结果：
  - `off` mean: `141.071 ms/fwd`
  - `on` mean: `136.021 ms/fwd`
  - 改善：`-5.050 ms/fwd`（`-3.58%`）
- 相比上一轮主线最优（cacheab-off mean `138.826`），新实现再降 `-2.805 ms/fwd`（`-2.02%`）。
- 四组 path counts 全一致 `19/38/893/209`，可判定收益来自 runtime map/record Python 开销压降，而非 EB 语义变化。

## 2026-04-30 (v0.1.15.15 runtime fastpath-v2 负优化回退)

- 在 fastpath-v1 基础上追加“局部张量模板缓存”（`pos_indices/ones`）并未带来收益，反而显著退化。
- C3 两轮：`142.370 / 146.753`，mean `144.562 ms/fwd`，相对 fastpath-v1 mean `136.021` 变慢 `+6.279%`。
- path counts 仍保持 `19/38/893/209`，排除语义漂移。
- 该优化已回退；回退后校验 `136.527 ms/fwd`，恢复到 fastpath-v1 主线水平。

## 2026-04-30 (v0.1.15.15 sparse-rr map fastpath 负优化回退)

- 在 `flat_eager` map core 试验“仅对 `replica_count>1` 的 token 执行 rr/mod，单副本 token 直接 slot-0”并未兑现收益。
- C3 A/B（`batch=128,gen=32,tp=4,world=8,P1+replica,cold_only,fastpath=on`）:
  - `sparse_rr=off` mean: `141.113 ms/fwd`
  - `sparse_rr=on` mean: `145.297 ms/fwd`
  - 结果为 `+2.97%` 退化。
- 四组 run path counts 均保持 `prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`，说明退化来自实现开销而非 EB 语义变化。
- 已回退该实现；回退校验 `eb_eplb_sparse_rr_revertcheck_c3_on_r1_20260430` 为 `134.599 ms/fwd`，与 `routeopt_revertcheck`（`134.636`）几乎一致，主线状态恢复正常。

## 2026-04-30 (v0.1.15.15 route no-op skip v2 收益确认)

- route 层仅跳过两处确定 no-op 调用（`native_record_gate_mode=off` + `eplb_cold_rearrange=False`）在当前 C3 主线口径下可稳定回收时延。
- A/B 结果：
  - `off` mean: `143.886 ms/fwd`
  - `on` mean: `136.512 ms/fwd`
  - 改善：`-5.125%`（`-7.374 ms/fwd`）
- 四轮 path counts 全一致：`prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`，说明收益来自 Python 调用开销压降而非 EB 语义变化。
- 与此前失败的 routeopt-v1 区别：本版不改 route-path setter 形态，仅去掉确定 no-op 调用，因此风险更低、效果更稳。

## 2026-04-30 (v0.1.15.15 route no-op skip v2 稳定性复证修正)

- 对 `route_skip_noop v2` 做交错 3x3 复证后，均值收益未复现：
  - `off mean/std`: `140.919 ± 5.048 ms/fwd`
  - `on mean/std`: `140.753 ± 1.928 ms/fwd`
  - `on vs off`: `-0.118%`（`-0.167 ms/fwd`）
- 六组 path counts 全一致（`19/38/893/209`），排除语义漂移。
- 结论修正：
  - 该优化目前仅体现“轻微且不稳定”的边际收益，不满足主线收益兑现阈值（>=2%）。
  - 暂不作为独立收益点对外宣称；可保留开关，后续在更大样本/更长窗口下再验证。

## 2026-04-30 (v0.1.15.15 EPLB step 主线缺口确认)

- 通过 `window/step` 面扫 + step-hook 诊断确认：
  - 当前 dInfer 主线并未自然推进 `EplbState.step()`；
  - 因此仅调整 `window_size/step_interval` 不会带来真实的 EPLB 重排行为变化。
- 强制 per-forward 调用 step 后可见 `Rearranging experts ...`，但带来两类风险：
  - 显著性能回退（smoke 从 `161.462` 上升到 `198~217 ms/fwd`）；
  - 存在 `AssertionError()`，稳定性不足。
- 结论：
  - 下一阶段必须采用“保守接入策略”而非直接开启重排：
    - `cold` 触发
    - 最小间隔
    - fail-open 降级（异常时回落到 map-only）
  - 先保证可控与稳定，再做性能收益优化。

## 2026-04-30 (v0.1.15.15 EPLB step/rearrange 断言与成本结论)

- 断言根因已被精确定位并复现：
  - `vllm/distributed/eplb/rebalance_execute.py:292`
  - `assert len(expert_weights) == num_moe_layers`
- 在实验桥接 MoE 运行时契约（`moe_layers/expert_weights/set_eplb_state`）后，断言可以清零。
- 成本拆分结论：
  - `step()` 壳本身不是主开销（大间隔、无重排时接近 baseline）；
  - 真正的大头是 `rearrange_expert_weights_inplace`（专家权重搬运/通信），可使时延从 `~163 ms/fwd` 抬升到 `~515 ms/fwd`。
- 因此下一阶段策略必须是：
  - 只在 `cold` 路径尝试；
  - 加最小间隔；
  - 失败/超时 fail-open 回退；
  - 明确禁止 per-forward 无条件重排。

## 2026-05-01 (v0.1.15.15 C12口径对齐：A/G vs EPLB)

- 已完成统一 C12 口径对齐（`batch=512,gen=256,tp=4,world=8,no-quality`）:
  - `A (bench)`: `75.836 ms/fwd`
  - `G (bench)`: `70.030 ms/fwd`
  - `B0 (eplb script, P1,E32)`: `127.408 ms/fwd`
  - `C3 (eplb script, P1+replica,E34)`: `187.862 ms/fwd`
- 相对 C12-A/G：
  - `B0 vs A`: `+68.00%`，`B0 vs G`: `+81.93%`
  - `C3 vs A`: `+147.72%`，`C3 vs G`: `+168.26%`
- path counts 不变性通过：四组均为 `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`。
- 结论：
  - 当前 EPLB 在 `collect_eb_heteval512_laws.py` 入口下的绝对时延不具备与 C12 A/G 直接竞争性；
  - 若目标是“EPLB 比 C12-G 更快”，下一阶段必须迁移到 bench 主路径进行最小观测负载的 source-level 接入与 A/B，而不是继续使用 law-probe 脚本做绝对性能比较。

## 2026-05-01 (v0.1.15.15 bench主路径 EPLB C12 A/B)

- `bench_bsp_moe_dp2.py` 主路径 EPLB 接入已跑通（修复 `last_path` 路由状态后无崩溃）。
- C12 口径 `bspg_source`（`batch=512,gen=256,tp=4,world=8`）off/on 对比：
  - `A`: `75.953 -> 78.671 ms/fwd`（`+3.58%`）
  - `E`: `71.989 -> 74.910 ms/fwd`（`+4.06%`）
  - `G`: `69.811 -> 72.804 ms/fwd`（`+4.29%`）
  - `GS`: `69.626 -> 72.884 ms/fwd`（`+4.68%`）
- path counts off/on 完全一致：
  - `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`
- 结论：
  - 当前 EPLB runtime（cold_only + flat_eager）在 bench 主路径仍呈稳定正开销（约 `+3.6%~+4.7%`），收益未兑现。

## 2026-05-01 (v0.1.15.15 开销归因矩阵：runtime 常驻税为主)

- 已完成 `OFF32 / ON32-off / ON32-cold / ON34-off / ON34-cold` 同口径分解。
- 关键结论（以 G 路径为代表）：
  - `OFF32 -> ON32-off` 增量最大（`+3.557 ms/fwd`），说明仅开启 runtime（即使 `record=off`）已引入主要税。
  - `ON32-off -> ON34-off` 未见正税（本轮为 `-0.776 ms/fwd`），E34 不是当前主矛盾。
  - `ON34-off -> ON34-cold` 仅 `+0.213 ms/fwd`，cold record 仅占小头。
- path counts 在 5 组实验中全部一致（`19/171/3933/931`），可排除语义漂移。
- 因此后续优化顺序应为：
  - 先回收 runtime enable 常驻开销；
  - 再优化 cold record 边际开销。

## 2026-05-01 (P1 static-only vs runtime tax)

- `OFF32 -> ON32-off` 的主增量是 `+3.557 ms/fwd`，而 `E34-disable-runtime` 只比 `OFF32` 慢 `+0.535 ms/fwd`。
- 这说明当前主矛盾不是 E34 形状，也不是 cold record，而是 `runtime-enable but record=off` 的 per-forward 常驻税。
- 因此更合理的主线是：
  - 静态 P1 placement 作为默认候选；
  - runtime EPLB 只保留为可选/诊断开关；
  - 后续只在需要时再讨论更深的 runtime 下沉或编译化。


## 2026-05-01 (P1 static-only C12 验证通过)

- 已在 bench C12 `bspg_source` 下验证 `P1 static-only`（`redundant=0 + disable_after_build`）：
  - A `+0.42%`，E `+0.31%`，G `+0.61%`，GS `+0.69%`（均相对 `OFF32`）。
- 对比 runtime-on 路径：
  - `ON32-off` 相对 `OFF32` 为 `+4.13% ~ +5.10%`；
  - 说明 runtime 常驻税仍是当前主要性能障碍。
- path counts 保持 `19/171/3933/931` 一致，语义未漂移。
- 工程结论：
  - 可以把 `P1 static-only` 作为当前默认主线；
  - runtime EPLB 保持可选开关，后续单独优化其常驻开销。

## 2026-05-01 (runtime identity map 常驻税回收)

- 在 `runtime-enable + redundant=0` 路径下，主税进一步被定位为：
  - `logical_replica_count==1` 且 map 为 identity 时，仍执行了每 token 的 runtime map 计算。
- 已在 `modeling_llada2_moe.py` 落地 identity passthrough：
  - 满足 `lrc==1 && l2p[...,0]==logical_id` 时直接透传 `topk_ids`；
  - 保留 `indices_type` 与 `cold_only` record 语义；
  - 不改 EB/s_mask 逻辑。
- 同版本 C12 A/B（`ON32-off`）收益稳定：
  - 关闭 passthrough：`A/E/G/GS = 79.179/75.245/73.442/73.114`
  - 开启 passthrough：`A/E/G/GS = 76.280/72.653/70.481/70.401`
  - 净收益：`-3.66% / -3.44% / -4.03% / -3.71%`
- 对 baseline `OFF` 的剩余差距显著收敛：
  - `ON32-off` 由 `+4~5%` 降到 `+0.35~0.94%`
  - `ON32-cold` 收敛到 `+0.62~1.28%`
- 诊断计数确认命中真实发生（C12, G）：
  - `identity_hits=5054/5054`（100%）
  - `cold_only` 下 `record_calls=171` 与 cold 次数一致。
- 结论更新：
  - EPLB 在 dInfer 当前主线并非“必然负收益”；
  - 主要问题是 runtime map 实现未区分 identity 场景；
  - 该问题已被工程性修复并兑现为可量化时延回收。

## 2026-05-01 (C12 双轴结论：性能与负载均衡需同时评估)

- 已在 `bench_bsp_moe_dp2.py` 正式接入负载均衡指标：
  - `ep_load_cv`
  - `ep_load_max_mean`
  - `ep_load_p95_p50`
  - 以及层加权统计 `layer_ep_load_cv_weighted`。
- C12 三组（`OFF / ON32-off / ON32-cold`, `batch=512,gen=256,tp=4,ep=8`）显示：
  - `ON32-off` 与 `ON32-cold` 相对 `OFF` 仍有 `+7%~+9%` 时延回退；
  - 同时 `ep_load_cv` 从 `~0.073~0.076` 上升到 `~0.214~0.215`（约 `+185%~+193%`），负载均衡明显恶化。
- `ON32-cold` 与 `ON32-off` 基本重合（仅 `record_calls` 差异），说明当前主问题不是 cold 记录开销，而是 runtime-on 的 mapping/placement 组合路径。
- 三组 path counts 始终一致（`19/171/3933/931`），因此这是性能路径问题而非 EB 语义漂移。

## 2026-05-01 (E=36 config 补齐后的结论更新)

- `E=36,N=512` 配置补齐后，ON32-off 已命中专用 config，不再走 default fallback。
- 相比补齐前 ON32-off，补齐后性能回收约 `3.2%~3.8%`（A/E/G/GS 一致）。
- 但相对 OFF，ON32-off 仍有 `+3.85%~+5.30%` 时延回退，说明 fallback 不是唯一瓶颈。
- 负载均衡差距仍显著：
  - OFF `ep_load_cv ~0.073~0.076`
  - ON32-off `ep_load_cv ~0.214~0.215`
- 因此当前应将问题拆成两层：
  1. 计算配置层（E36 fallback）已部分修复；
  2. runtime EPLB 映射/placement 层仍是主要剩余优化对象。

## 2026-05-01 (ON32 启动排布 weight-balance init 结论)

- 已实现 `ON32` 启动排布可切换：
  - `joint_p1_p5`（默认旧策略）
  - `weight_balance`（新策略，基于离线 expert-load prior）。
- 在不启 runtime rearrange 的前提下，C12 正式对比显示：
  - GS: `old=79.615 ms/fwd` vs `wb=80.081 ms/fwd`（`wb` 慢 `+0.585%`）
  - GS `ep_load_cv`: `old=0.053773` vs `wb=0.053029`（仅 `-1.38%`）
- 结论：
  - “仅替换初始化排布”为 `weight_balance` 当前没有带来端到端收益；
  - 排布本身不是当前主瓶颈，后续应继续优化 runtime map/select 或副本选择分布策略。

## 2026-05-01 (two-choice 局部性触发路线结论)

- 结论一：`two_choice_lb` 全程路径不适合当前主线（smoke 可达 `+15%~+47%` 退化），必须避免 hot 常驻执行。
- 结论二：按时间局部性改成 cold-only 触发后，机制与诊断都成立（C12 可见 `2.27M` multi-token load-aware 决策，`171` 次 cold decay/update）。
- 结论三：即便 cold-only，当前收益仍不足：
  - C12 相对 `flat_eager`：A `+1.31%`, E `+1.68%`, G `-0.73%`, GS `+0.83%`；
  - 负载指标仅微弱改善（GS `ep_load_cv 0.215393 -> 0.214857`）。
- 最终判断：保留 two-choice 为实验分支，不进入默认主线；主线继续 `flat_eager`。

## 2026-05-01 (Profiling 专项单次口径结论)

- 在单次 C12 profiling 下，`ON32-flat` 和 `ON32-cold` 仍显著慢于 `OFF`（GS `+6.36%` 与 `+5.07%`），`cold_only` 仅小幅缓解。
- `P1 static-only` 虽优于 runtime-on，但本次单测仍高于 OFF（GS `+3.65%`），提示当前 run-to-run 条件仍有扰动，需要进一步控制实验环境或增加 targeted profiling。
- 关键判断：四组通信 payload 体积一致，性能差异主要来自执行时延抬升（`native_forward/quant_apply/combine/dispatch`），而非字节量本身。
- 负载均衡指标在 ON32 模式下变差（GS `ep_load_cv` 从 `0.0305` 到 `0.0538`），并与性能回退方向一致，但不是唯一解释因子。
- 下一步优先级应是 rank-tail 与同步等待剖析（谁在等、等在哪个 collective/组件），而不是继续搬运或扩展 mapping 算法。

## 2026-05-01 (上下文压缩恢复入口刷新结论)

- `history-chat.txt` 已刷新为当前压缩恢复主入口，内容覆盖了：
  - 必读文档顺序；
  - 当前代码改动与实验状态；
  - path invariant 与关键性能结论；
  - 下一步待办与复现命令。
- 当前最关键执行导向不变：
  - 继续以 profiling/rank-tail 归因为主线；
  - 先补齐 `ON32-flat` 与 `P1-static` 的同口径 tail 对照；
  - 在此之前不单独改 EB/s_mask 语义。

## 2026-05-02 (EPLB 全栈评估与 per-layer 负载均衡 characterization)

### EPLB 优化全程结果

- vllm `balanced_packing` 修复了旧排布的负载恶化 bug（ep_load_cv 从 0.215 降到 0.054，优于 OFF 的 0.076）。
- 旧 `joint_p1_p5` 排布策略破坏了跨层 GPU 轮转（Rank 0/7 被饿死 60-68%），属功能性缺陷。
- vllm Triton fused map kernel 已集成，替代 6 个 Python tensor op 为 1 个 kernel launch。
- hot_skip 缓存方案有正确性风险（10-27% token 会被路由到错误 expert），已回退。
- 方案 B（replica_count=1 透传）正确但在当前配置下无额外性能收益（has_multi_replica=True）。
- **最终 EPLB 配置**：`native_class + vllm_balanced + cold_only + tensor_cache=off`，GS 路径 +2.2% vs OFF。
- 质量验证通过（heteval512 verifiable snippets 与 OFF 无可辨别差异）。

### Per-layer 负载均衡核心发现 (I20-I22)

- **I20**: 各 MoE 层的 256 个 expert 是完全独立的权重实体。推理逐层串行执行，每层末尾有 collective 同步屏障。因此 overall ep_load_cv（跨层平均）是误导性指标；真正决定性能的是 **per-layer per-GPU 不均衡**。
- **I21**: Per-layer expert 负载极度倾斜（CV~1.22，top10% expert 承担 37% token，max/min 比达 65000x）。但 19 层的 top-5 热 expert **完全不同**（83 个不同 expert 占 95 个位置，无任何 expert 出现在所有层 top-5 中）。
- **I22**: Per-layer token 不均衡（max/mean=1.50）的 wall-clock 影响被 memory-bound fused MoE 削弱到 <1%。50% 更多 token 仅造成 0.36 ms tail（43.76 ms 的 0.82%），因为 kernel 瓶颈是 expert weight HBM loading，不是 token GEMM。

### 方向判断更新

- EPLB 在当前配置下（256 experts / 8 GPU / batch=512 / memory-bound kernel）的 **wall-clock 天花板极低**（~1.2 ms，1.3% of e2e），被 runtime map 开销（+2.2%）抵消。
- 但 EPLB 价值会在以下条件下上升：更大 EP 规模（32-64 GPU，4-8 experts/GPU）、更小 batch（per-expert token 少，方差大）、compute-bound 化的 MoE kernel。
- 短期推荐：P1 static + vllm balanced expert_map（零 runtime 开销，改善初始均衡）。
- MASK vs decoded token 的路由差异分析（Insight I3 的细化）因 h2e hook 被 BlockDiffusion 框架绕过，未完成，需后续用不同 hook 策略补充。

## 2026-05-02 今日成果总结

### 成果清单

1. **vllm `balanced_packing` 集成**：修复了旧 `joint_p1_p5` 排布策略的功能性缺陷（ep_load_cv 从 0.215 恢复到 0.054），复用了 vllm 的成熟组件。
2. **Per-layer balanced placement（I23, free lunch）**：用真实 per-layer per-expert 负载数据 + vllm `balanced_packing`，为每层独立优化 expert-to-GPU 分配，19 层全部不同，per-layer GPU CV 从 0.14~0.20 降到 ~0.0000。零 runtime 开销。
3. **vllm Triton fused map kernel 集成**：将 6 个 Python tensor op 合并为 1 个 kernel launch。
4. **`_EplbNativeMapper` callable class**：替代 closure 实现，结构更清晰。
5. **`vllm_balanced` init placement mode**：新增环境变量 `DINF_EPLB_INIT_PLACEMENT_MODE=vllm_balanced` + `DINF_EPLB_VLLM_PHY2LOG_PATH`，支持加载预计算的 per-layer phy2log。
6. **Per-layer routing heatmap profiler**：`--routing-heatmap` 开关，采集 per-layer per-expert hit count。
7. **Insight I20-I23 归档**：形成了 dLLM MoE 负载均衡的完整认知体系。

### 关键认知建立

- **I20**: 各层 expert 独立，推理逐层串行 → overall cv 是误导指标，per-layer per-GPU 才是关键。
- **I21**: Per-layer expert 负载极度倾斜（CV~1.22），但热 expert 跨层完全不同（83 unique / 95 slots）。
- **I22**: Memory-bound 削弱效应实验确认：22~41% token 不均衡 → 仅 0.4~0.8% wall-clock 差异。
- **I23**: Per-layer balanced placement 是 free lunch（零开销 +0.3~0.6 ms 收益）。

### 保留的最优 EPLB 配置

- **静态方案（推荐主线）**：`vllm_perlayer_balanced_expert_map_ep8_20260502.pt`，E=32 无冗余，零 runtime 开销。
- **Runtime 方案（备选）**：`native_class` + `vllm_balanced` + `cold_only` + `tensor_cache=off`，+2.2% 开销，适用于需要 adaptive 能力的场景。
- Runtime EPLB 在更大 EP 规模（32-64 GPU）或 compute-bound kernel 条件下价值会显著上升。

### 性能基准状态

- 05/02 检测到 ~7 ms 环境退化（代码回退测试确认非代码原因）。
- 当前绝对基准：A=81.1, G=76.9, GS=76.9 ms/fwd。
- 所有相对结论（per-layer balanced +0.3~0.6 ms、EPLB runtime +2.2% 等）在同 session 对比中有效。
- 待机器环境恢复后重新校准绝对数字。

## 2026-05-02 (per-layer balanced placement 实验确认)

- **实验设计**：用 vllm `balanced_packing` + 真实 per-layer per-expert 负载数据生成 per-layer 独立优化的 expert_map（19 层全部不同），与原始 linear split（所有层相同）做零 runtime 开销的 clean A/B。
- **per-layer GPU 均衡改善极大**：OLD per-layer CV=0.14~0.20 → NEW CV≈0.0000（几乎完美均衡）。
- **wall-clock 改善很小**：GS 76.99→76.71 ms（-0.28 ms, -0.36%），A 81.28→80.67 ms（-0.61 ms, -0.75%）。
- **memory-bound 削弱效应得到实验确认**（I22 升级为 Experimentally confirmed）：22~41% 的 token 不均衡仅转化为 0.4~0.8% wall-clock 差异。
- **新增 I23**：per-layer balanced placement 是 free lunch——零 runtime 开销，只需离线 profiling + map 生成。
- **EPLB 方向最终判断**：当前配置下 runtime EPLB 不值得（开销 +2.2% > 收益 ~0.5%）。静态 per-layer balanced placement 是正确选择。Runtime EPLB 仅在更大 EP 规模或 compute-bound kernel 条件下才有意义。

## 2026-05-02 (系统级 Profiling + MASK Routing + Roofline + TEAM 搬运方案)

### GS 路径 Component Timing Profiling

- GS 76.6 ms/fwd 完整拆分：MoE kernel 24.7ms(32%), MoE dispatch+combine 14.5ms(19%), Attention ~20.5ms(27%), Attention RS+MoE gather 6.9ms(9%), Gate+shared+norms 7.3ms(10%), Other 2.7ms(3%)。
- MoE native_forward 内部：quant_apply 24.67ms(60%), dispatch 8.02ms(20%), combine 6.48ms(16%)。
- Rank-tail gap 仅 ~1.7 ms/fwd (2.2%)，rank 不均衡不是瓶颈。

### MASK Routing Per-Layer 分析 (I24)

- 修复了 MASK/decoded routing 拆分（model forward pre-hook 替代 h2e hook）。
- MASK routing 集中度呈深度依赖梯度：L0-L1 entropy ratio 0.65-0.69（极度集中），L4+ ratio 0.94-0.98（与 decoded 无差异）。
- Wall-clock 优化潜力极低（memory-bound + 仅 12.9% MASK + 仅 L0-L1 有效）。

### MoE Kernel Roofline 分析（正确版：block_length=32 + KV cache）

- dLLM 有 KV cache 时，每次 forward 只有 block_length=32 个 token 过 MoE（不是全序列）。
- Per-expert T_avg = 512 tokens, AI = 284, H100 balance point = 295 → 恰在转换点。
- 这解释了 I22 的 ~0.5% EPLB 收益。
- Batch 越小越 memory-bound，EPLB 效果越差。Batch 越大越 compute-bound，但 32 experts/GPU 的统计平滑仍限制 EPLB 天花板。

### Batch=1024 EPLB 实验

- GS OFF: 123.59 ms/fwd → GS ON-static: 123.11 ms/fwd (-0.4%)。与 batch=512 的 -0.4% 一致。
- 结论：EPLB 天花板由 per-GPU 统计平均（32 experts/GPU）决定，与 batch 无关。

### DeepEP Sparse Dispatch 实验

- DeepEP HT: 134.01 ms/fwd vs AgRs: 80.67 ms/fwd → **+66% 退化**。
- 单节点 NVLink ring collective 已是最优，DeepEP P2P sparse dispatch 开销更大。
- 结论：DeepEP 适用于跨节点 EP=32-64，不适用于单节点 8-GPU。

### EB 机制正确理解（从代码验证）

- q_major=1.0（100% token 覆盖），K_target=40（初始种子），K_init per-layer 不同。
- EB hot_skip 安全（只缓存候选集 mask，routing 仍用 fresh gate logits）。
- EB 收益来源：(a) 减少 unique active experts → 省 weight loading, (b) K=4 vs K=8 → dispatch 减半。

### TEAM 论文组件搬运方案（已批准，未实施）

- TEAM (arXiv:2602.08404) 核心：decoded-token skip（跳过已解码 token 的 MoE 计算）。
- 我们的 C12 中 87% tokens 是 decoded → 理论上可跳过 87% MoE 计算。
- 与我们之前失败的 stable cache 的关键区别：TEAM 只跳 decoded 位置（不再被预测），MASK 位置仍做完整 MoE → error bounded。
- EB 兼容设计：gate + s_mask 对全量 tokens 计算（保证质量），只有 fused_moe kernel 对 MASK-only tokens 执行。
- 预估收益：~15-18 ms/fwd (20-23%)。
- 方案详见 `/home/wuhang/.claude/plans/iridescent-churning-flamingo.md`。

## 2026-05-05 CUDA Graph + Profiling + SP LM Head

### CUDA Graph 结论

- CUDA Graph at batch=512 仅省 ~2 ms (2%)，因为 GPU pipeline 隐藏了 Python dispatch 开销
- batch=64 时可省 30%（GPU compute 轻，Python gap 暴露）
- torch.compile 不兼容 EP dispatch（inductor 不支持 .cpu() sync）
- **CUDA Graph 方向在 batch=512 下收益极低，已排除**

### 完整 GS Path 76.4 ms/fwd 分解

| 组件 | ms/fwd | 占比 |
|------|--------|------|
| MoE kernel | 20.6 | 26.9% |
| MoE combine | 11.6 | 15.2% |
| MoE dispatch | 8.7 | 11.4% |
| Attention 全链路 | 14.1 | 18.4% |
| LM head + logits.float | 11.5 | 15.0% |
| 其他 (shared, gate, norms) | 10.0 | 13.1% |

### nsys 关键发现

- GPU 利用率 71%，29% idle 来自 kernel launch gap（非 NCCL 等待）
- 单 stream，零 NCCL-compute overlap
- Compute 跨 rank 完全均衡 (CV<1%)

### 已验证的优化

- OPT-2 (skip logits.float): **-4.3 ms** → G: 70.2→65.9 ms/fwd
- SP LM Head (SP lm_head + local decode): **-7.7 ms** → G: 65.9→58.2 ms/fwd
- **累积: 70.2→58.2 ms/fwd (1.21x 加速)**

### 已排除的新方向

- fp8 dispatch: +1.5 ms 退化（NVLink 带宽充足，cast 开销 > 通信节省）
- Shared expert ∥ dispatch overlap: ~0 ms（GPU pipeline 已隐藏）

### 当前最优配置

- G path: **58.2 ms/fwd**, env: `DINF_SKIP_LOGITS_FLOAT=1 DINF_SP_LM_HEAD=1`
- tp=4, dp=2, ep=8, SP, EB K=4, flash_attn, fused_routing, fused_rmsnorm

## 2026-05-06 BSP-H 探索 + TEAM Decoded-Token Skip

### BSP-H 结论

- BSP-H 原始定义是 hierarchical collective（EP RS + TP AG 融合），不是 AllReduce 替换
- AllReduce 方案实测 GS 59.24 → H 57.08 ms (-2.17 ms)，但 AllReduce 本身仅省 0.5 ms（combine 11.17 vs combine+gather 11.67），主要收益来自消除 attention input TP AllGather
- TP attention 有结构性限制：RowParallelLinear 要求所有 GPU 处理相同 token，无法通过改变并行策略消除 TP AllGather
- Token-parallel attention（每 GPU full heads）理论省 7.5 ms 但大模型不泛用
- AsyncTP fused ops（symm_mem.fused_matmul_reduce_scatter 等）在 H100 C12 shape 下 2-6x 退化，全部关闭
- **结论：BSP-H 在 NVLink 8-GPU batch=512 配置下收益有限，方向暂搁置**

### TEAM Decoded-Token Skip 结论

- TEAM 核心思想：dLLM block diffusion 中 87% token 是已 decoded 的，其 MoE 计算对最终输出零贡献，可用缓存替代
- 与 Stable Cache (v0.1.13) 的关键区别：TEAM 只跳 decoded 位置，MASK 位置仍做完整 MoE → 误差只通过 attention 间接传播（二阶效应）
- 缓存在 decoded 那一刻冻结不更新，但 decoded 位置 cosine sim 0.97-0.99，threshold decoder 提供容错
- TEAM v1 (extract+pad+context)：native_forward -6.2 ms 但 per-layer overhead +9.4 ms → 净退化。根因是 38 次/fwd AllReduce（max_n_mask + DPMetadata）
- TEAM null expert 方案：expert_map[256]=-1，topk_ids[decoded]=256 → kernel skip → 零 Python overhead。kernel 独立测试通过（精确零）。但**完整流程质量崩坏**，cache-only 质量可接受 → bug 在 routing 修改本身。待 debug
- MoE kernel 是 memory-bound (I8/I25)：null expert 减少 pairs 但不减少 weight loading，kernel 反而因 560 extra skip blocks 慢 0.9 ms
- **结论：TEAM 机制可行（cache-only 质量通过），但高效实现需要解决 (a) null expert 质量 bug 或 (b) v1 的 per-layer overhead。核心矛盾：减少 dispatch 通信需要新 forward_context（有 overhead），减少 kernel 需要 null expert（有 bug）**

## 2026-05-07 TEAM Null Expert 质量排查 + MoE Cache 状态机 + Piggyback Dispatch

### Null Expert 质量排查完成

- TD7b 实验证明：null expert kernel 不影响 MASK token 输出（diff=0.000000）
- 质量崩坏根因：decoded=0 通过 residual→attention 跨层传播，间接污染 MASK 计算
- 解决方案：cache merge 补偿 decoded 位置 + M=5 周期刷新防止 cache staleness
- 质量结果：TD4 #0/#13 与 G 逐字一致，#19/#28 极轻微差异（满足 TEAM 可接受范围）

### Cross_block Forward 机制

- BlockDiffusionLLM (KV cache 路径) 在 Block 1+ 的首个 forward 处理 prev_block+curr_block（N_sp=4096）
- 目的：刷新前一 block 的 KV cache 以反映最终 decoded 状态
- 影响：MoE cache 的 prev_decoded 状态机需要处理 shape 不匹配（4096 vs 2048）

### Prev_decoded 状态机设计

- null_mask = decoded_sp & prev_decoded_sp & (step%M != 0)
- prev_decoded=None 时自动降级为 G 路径（处理 prefill/cross_block/block 边界）
- M=5 刷新：每 5 步全量计算刷新 cache，cache staleness 从整个 block (~27步) 降到最多 4 步
- WRITE 条件：(decoded AND step%M==0) OR (刚变成 decoded) → 正常计算 + 写 cache
- READ 条件：(decoded AND step%M!=0) AND (上一步就是 decoded) → null expert + 读 cache

### Piggyback Dispatch 方案

- decoded mask 搭载 router_logits 的额外列通过已有 dispatch AllGatherV 传输
- 零额外 NCCL collective（消除了独立 AllGather 的 deadlock 风险）
- gate.get_logits wrap: [N_sp, E] → [N_sp, E+1]
- routing function: 从 gating_output[:, -1] 提取 null mask

### TV3 Sparse Dispatch 排除

- Payload -81%（206→38 MB），但 Python overhead（alloc/cat/index_copy）+12.7 ms > NCCL 节省 10.6 ms
- 端到端无净收益，正确性也有 bug → 暂搁置
- 在 NVLink 高带宽 + batch=512 GPU pipeline 条件下，减少 payload 不如减少 Python 操作

### 当前 TD4 性能状态

- 质量：接近 G（M=5 刷新后重复词现象大幅减少）
- 性能：73.79 ms/fwd（比 G 慢 24%）
- 主要 overhead：cache clone 0.69ms×19层 + cache merge 2.63ms + piggyback torch.cat ~10ms
- 下一步：消除 per-layer overhead（预分配 buffer / in-place）或重新评估 cache-only 方案

## 2026-05-08 TEAM Sparse Kernel 多路径探索（TV4/TV5/TV4m）

### TV5 当前最优方案（+0.66% vs G）

- 利用 vllm `moe_align_block_size` C++ kernel 的 `expert_id >= num_experts → continue` 保护
- 实现：routing 后 `topk_ids.index_fill_(0, null_indices_gathered, 256)` + kernel 后 `c_out.index_fill_(0, null_indices_gathered, 0)`
- 不需要 null expert（expert 257）、不需要 expert_map 修改、不需要 gate wrap、不需要 extract/scatter
- 每层仅 4 个额外 kernel launch（2× index_fill_ + 2× cache merge）
- 59.71 ms/fwd, path counts 一致, 质量验证通过

### TV4 Buffer 优化版（+1.2% vs G）

- 核心：extract compact tokens → fused_experts(256 experts) → scatter back
- Buffer 优化：预分配 + index_select(out=) + 消除 per-layer zeros + int64 index 替代 boolean indexing
- kernel time 比 G 少 16 秒（compact silu/moe_sum/fused_moe），但 +418k launches 的 CPU overhead 完全吃掉节省
- 60.60 ms/fwd

### CPU Launch Overhead 是 Monkey-Patch 硬瓶颈

- nsys 证据：TV4 cudaLaunchKernel 比 G 多 2.3 秒 CPU 时间
- CUDA Event 证据：TV4 kernel phase 比 G 快 40%（0.642 vs 1.067 ms/layer）
- 理论 native 集成后 TV4 可达 ~51 ms/fwd (-16% vs G)
- Monkey-patch 层面无法兑现——Python interpreter 每次 op 间 ~6.6μs roundtrip 造成 GPU 空泡

### TV4m Mapped Kernel（理论 -16%, 未完成）

- 自定义 Triton kernel `_fused_moe_kernel_mapped`：加 input_map_ptr 参数，第一个 GEMM 从 FULL A 间接读
- silu_and_mul + moe_sum 在 compact cache 上操作（省 87%）
- crash: illegal memory access（sorted_token_ids padding sentinel 越界 input_map）
- 待调试：需要在 input_map load 时对 compact_tok 做更严格 bound check

### 已排除方向

- TV4c Phase 1 AllReduce：NVLink 带宽充足，无改善
- TV4c Phase 2 compact dispatch：vllm sp_local_sizes 断言不兼容
- Triton gather kernel：效率低于 PyTorch vectorized_gather (+6.3%)
- shuffle_rows (vllm C++)：需 int32 cast，略慢于 index_select (+2.3%)
- TD8 无 cache：质量严重崩坏（cache 对质量必要）

## 2026-05-09 TV4m Crash Fix + TV6 Compact Dispatch

### TV4m 修复结论
- TV4m crash ROOT CAUSE: `N = w13_weight.shape[2]`（K=2048）应为 `shape[1]`（N=1024）
- silu_and_mul 读取未初始化内存导致质量崩坏（forward 数 2x，重复词）
- 修复后 TV4m: **57.9 ms/fwd (-2.0% vs G)**，monkey-patch 阶段最优
- 调试方法论：standalone kernel test (`test_triton_mapped_kernel.py`) 隔离了 kernel 逻辑后，dump 真实数据精确定位

### TV6 Compact Dispatch/Combine
- vllm 原生 `all_gatherv`/`reduce_scatterv` 支持 variable-size collectives（pynccl 层验证通过）
- TV6 compact dispatch 省 NCCL AllGather -20.0s，compact combine 省 Reduce -3.3s（nsys 实测）
- TV6 GPU kernel 总节省 **-28.2s**（是 TV4m -13.0s 的 2.2 倍）
- 但端到端仅 **-1.3% vs G**（58.3 ms/fwd），因为 Python overhead 吃掉了 GPU 节省

### CPU Overhead 精确量化
- **B 类（隐式 sync）**：boolean indexing 内部调 `.nonzero()` 触发 per-layer GPU→CPU sync。修复（int indexing + tolist）后省 **-1.9 ms/fwd**
- **A 类（CPU 喂不饱 GPU）**：per-layer Python tensor ops 产生 5-50μs gap，compact kernel 太短（avg 36.7μs）GPU 等 CPU → 约 2.8s 增量 idle
- **in-place cache 优化**：直接修改 cache tensor（1 kernel/layer 代替 4 kernel/layer），省 **-0.7 ms/fwd**
- GPU 利用率：G=43.4%, TV4m=41.0%, TV6=36.8%

### 性能演进表
| 方案 | ms/fwd | vs G | 核心机制 |
|------|--------|------|---------|
| G 基线 | 59.1 | — | full MoE |
| TV5 (topk_ids skip) | 59.7 | +0.66% | 零 overhead 但 kernel 全量 |
| TV4m (mapped kernel) | 57.9 | -2.0% | compact kernel + input_map |
| TV6 (compact dispatch) | 58.3 | -1.3% | compact dispatch/combine + vllm原版kernel |

### 已排除方向
- TV4m-v2 (token_remap Triton kernel): crash/暂停，收益仅 ~0.3ms
- TV6 via quant_method.apply: +5.8%（sp_local_sizes 覆盖 + Python wrapper 开销）
- "Triton 散列读取 bug" 假设: 实际是 N 维度 indexing 错误

## 2026-05-10 论文动机图 + TV6 Patch + 性能/质量验证

### 动机图数据结论
- **Expert routing stability (Fig.3)**：block 内跨迭代 Jaccard > 0.94，top 31-51% experts 覆盖 90% routing → 支撑 EB 机制
- **MoE output stability (Fig.4)**：Decoded token cos_sim ≈ 0.98，MASK ≈ 0.68-0.76 → 支撑 decoded-skip cache
- **Cache staleness (Fig.5)**：cos_sim 从 1.0 衰减到 ~0.80 at step=20，M=5 处 avg=0.96 → 支撑 refresh interval 选择
- 以上规律在 GSM8K/HumanEval/MT-Bench 三个数据集上一致

### TV6 Patch 独立模块结论
- tv6_patch.py 从 bench 脚本成功解耦，性能完全对齐（58.36 vs 57.96 ms/fwd）
- 关键教训：必须直接 import bench 脚本的 `BSPMSkipController`（含 predict_path 等方法），不能自定义简化版
- 关键教训：不能直接 wrap `experts.forward_impl`（破坏 vllm forward context），必须 patch `_moe_forward_with_context` 全局函数

### 完整性能链路（batch=512, heteval512）
| 配置 | ms/fwd | fwd | 加速 |
|------|--------|-----|------|
| Vanilla (K=8) | 85.39 | 266 | — |
| A (EB K=4) | 71.53 | 266 | -16.2% |
| G (BSP-G) | 58.39 | 266 | -31.6% |
| TV6 (compact) | 58.36 | 266 | -31.7% |

- EB K=4 不改变 fwd count（266 = 266），不影响 decoding 行为
- TV6 加速随 batch 增大增强：batch=32 无加速，batch=512 -32.5%（Python overhead 为常数，收益与 batch 正比）

### 质量验证结论
- bench 框架肉眼检查：A/G/TV6 输出与 Vanilla 语义一致，质量无退化
- GSM8K 上 EB K=4 vs K=8 fwd 差距仅 3.7%（batch=512），数学答案正确
- lm-eval 标准 benchmark 因 TP 拓扑不兼容（eval_dinfer.py 用 TP=8 spawn，TV6 需要 tp=4 dp=2）暂搁置
- 正确的质量验证方式：bench 框架 + 肉眼检查 + 相同 prompt source

## 2026-05-11 Baseline Bench + 跨框架 Benchmark 对比

### 核心性能数据（GSM8K, pad=128, gen=256, block=32, threshold=0.90, TP=4, 4GPU）

| 引擎 | batch=32 ms/fwd (fwd) | batch=256 ms/fwd (fwd) |
|------|----------------------|----------------------|
| dInfer w/o cache | 77.43 (208) | OOM |
| dInfer w/ cache | 48.13 (204) | 98.92 (234) |
| SGLang flashinfer no-graph | 40.30 (231) | 待测（max_rr=256 OOM） |
| Ours Vanilla d1t4e4 | 40.26 (201) | 65.10 (235) |
| Ours TV6 d1t4e4 | 47.66 (234) | 49.18 (255) |
| Ours Vanilla d2t4e8 (8GPU) | 52.06 (205) | 58.82 (230) |
| Ours TV6 d2t4e8 (8GPU) | 53.55 (233) | 54.22 (253) |

### 关键结论

1. **同口径 4GPU 对比：Vanilla d1t4e4 比 baseline 快 16%**（flash_attn + fused_rmsnorm + OPT-2 + SP-LM Head 的纯净收益）
2. **TV6 d1t4e4 在 batch=256 比 baseline 快 50%**（49.18 vs 98.92 ms/fwd），ms/fwd 几乎不随 batch 增长
3. **小 batch 下 DP=2 比 DP=1 慢 29%**（AllToAll collective overhead 在 batch=32 下未被 amortize）
4. **SGLang flashinfer no-graph ≈ Our Vanilla**（40.30 vs 40.26 ms/fwd），CUDA Graph 贡献 2.4x，FlashInfer 贡献 4.4x
5. **Baseline 受限于 TP=4（kv_heads=4），batch=512 OOM，batch=256 no-cache OOM**

### TV6 DP=1 修复

- 三处改动使 TV6 在 DP=1 下正常工作：初始化 all2all_manager + 创建临时 DPMetadata + fallback 路径 _run_full_moe
- 修复后 fwd count 从 257 降到 234（fallback 路径输出不完整导致的多余迭代被消除）
- DP=2 数据未被破坏
