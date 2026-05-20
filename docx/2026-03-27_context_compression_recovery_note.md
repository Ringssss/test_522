【
上一轮信息：

1. 先看的开发规范仍然是：

- /home/wuhang/wuhang/linear_wh/docx/next_step.txt

关键规范总结：

- 当前阶段代号：v0.1-init-project
- 代码注释如果新增，必须用英文
- 测试脚本统一放到 /home/wuhang/wuhang/linear_wh/codex_coding/src
- 测试结果统一放到 /home/wuhang/wuhang/linear_wh/codex_coding/results
- 重要文档放到 /home/wuhang/wuhang/linear_wh/docx
- 过程文档放到 /home/wuhang/wuhang/linear_wh/code_building
- 每完成一个进展，都要同步到 /home/wuhang/wuhang/linear_wh/code_building/progress_diff_summary.md
- 同步时要带上【本轮命令】
- 如果感到盲目，就先搜代码、文档和配置，不要盲改

还要记住一个现场偏差：

- next_step.txt 里把 Triton 项目地址写成了 `/home/wuhang/wuhang/linear_wh/custom_pkg/triton`
- 但当前实际存在且已经被检查过的仓库是 `/home/wuhang/wuhang/linear_wh/triton`
- 以后如果需要改 Triton 代码，要先和用户确认究竟以哪个路径为准，不要自己假设

2. 这一轮之前，已经完成并归档的两条高层认知：

- 项目总体定位来自三篇 citation notes，核心是一个研究型 PoC：
  - 在 SGLang + Triton 上做 LLM 推理热点子图的 layout / kernel / execution 优化
- Triton `LinearLayout` 已经被梳理过：
  - 它的核心价值不是单独一种 layout 名字，而是把 layout 变成可组合、可比较、可约简的代数对象
  - 这条思路已经归档，不是当前唯一方向，只是保留为一条 warm thread

对应关键文档：

- /home/wuhang/wuhang/linear_wh/docx/cites/sc_paper_architecture.md
- /home/wuhang/wuhang/linear_wh/docx/cites/subgraph_layout_ir_theory.md
- /home/wuhang/wuhang/linear_wh/docx/cites/sglang_poc_build_plan.md
- /home/wuhang/wuhang/linear_wh/docx/articles/2026-03-27_linear_layout_and_subgraph_optimization.md

3. 刚刚完成的模型初始化和运行链路，核心成果是：

- 成功用本地 editable SGLang 跑起：
  - `/home/wuhang/models/Qwen3.5-35B-A3B`
- SGLang 项目路径是：
  - `/home/wuhang/wuhang/linear_wh/sglang`
- conda 环境是：
  - `linearllm`
- 机器硬件是：
  - 8 x H100 80GB
- profiling 工具是可用的：
  - `/usr/local/cuda-13.0/bin/ncu`
  - `/usr/local/cuda-13.0/bin/ncu-ui`
  - `/usr/local/cuda-13.0/bin/nsys`
  - `/usr/local/cuda-13.0/bin/nsys-ui`
  - `/home/wuhang/miniconda3/envs/linearllm/bin/py-spy`
  - `/usr/bin/htop`

4. 本轮在线服务已经真实跑通，不是只到初始化：

- 服务地址：
  - `http://127.0.0.1:31000`
- 模型：
  - `/home/wuhang/models/Qwen3.5-35B-A3B`
- 当前运行配置的关键点：
  - `tp=2`
  - `dtype=bfloat16`
  - `mem_fraction_static=0.8`
  - `attention_backend=fa3`
  - `linear_attn_backend=triton`
  - `disable_radix_cache=False`
  - `disable_chunked_prefix_cache=False`
  - `page_size=1`
  - `mamba_scheduler_strategy=no_buffer`

5. 当前已经记录下来的 baseline 结果概要：

- 单请求检查：
  - prompt: `用一句话介绍你自己。`
  - prompt tokens = 15
  - completion tokens = 64
  - elapsed = 1.644s
- online benchmark 1：
  - 80 prompts
  - random input = 256
  - random output = 64
  - max concurrency = 16
  - request throughput = 4.41 req/s
  - output throughput = 120.25 tok/s
  - mean TTFT = 475.54 ms
  - mean TPOT = 105.94 ms
- online benchmark 2：
  - 40 prompts
  - random input = 4096
  - random output = 128
  - max concurrency = 8
  - request throughput = 1.84 req/s
  - output throughput = 102.35 tok/s
  - mean TTFT = 421.12 ms
  - mean TPOT = 82.73 ms

注意：

- 这一轮没有把原始 benchmark stdout 直接重定向保存成独立原始日志文件
- 但已经把结果概要补存为：
  - `/home/wuhang/wuhang/linear_wh/codex_coding/results/2026-03-27_qwen35_online_baseline_summary.md`
- 下一轮不要再只把结果留在聊天里，必须把脚本和结果文件落到 `codex_coding/src` 和 `codex_coding/results`

6. 当前对模型结构的关键认识：

- Qwen3.5-35B-A3B 不是普通 dense decoder
- 它是：
  - hybrid attention
  - sparse MoE
  - shared expert
  - long context
  - MTP
- 语言骨干可概括为：
  - `10 × (3 × (Gated DeltaNet -> MoE) + 1 × (Gated Attention -> MoE))`

这意味着后续性能研究不能只盯 full attention，也不能只盯 MoE，而要把：

- linear attention
- periodic full attention
- MoE
- state / cache / transition overhead

作为一个 heterogeneous system 来看。

对应关键文档：

- /home/wuhang/models/Qwen3.5-35B-A3B/config.json
- /home/wuhang/models/Qwen3.5-35B-A3B/README.md
- /home/wuhang/wuhang/linear_wh/docx/articles/2026-03-27_qwen35_hybrid_attention_brainstorm.md

7. 运行里出现的最关键系统信号：

- 当前 SGLang 在 H100 上找不到这组 MoE 的优化 Triton config，日志明确说：
  - `(E=256, N=256, H100)` 的 Triton fused MoE config 缺失
  - 当前使用 default config，性能可能 sub-optimal

这个信号非常重要，因为它说明：

- 现在的 baseline 不是 fully tuned baseline
- 后面如果做性能分析，要区分：
  - 模型架构本身的问题
  - 当前 MoE kernel config 缺失带来的额外损失

8. 刚刚完成的另一条核心分析，是 prefix/radix cache、KV cache 和 linear state 的区别：

- full attention 中：
  - prefix cache 复用的是前缀对应的 KV cache
  - radix cache 是实现这一复用的前缀树结构
- hybrid linear-attention 模型中：
  - 不能只复用 KV
  - 还必须复用某个前缀边界上的 recurrent / linear state snapshot
- 在 SGLang 当前实现里：
  - full-attention-only 模型走 `RadixCache`
  - hybrid SSM/GDN 模型走 `MambaRadixCache`
  - 分层 host cache 版本走 `HiMambaRadixCache`

关键源码：

- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/radix_cache.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/mamba_radix_cache.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/hi_mamba_radix_cache.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/managers/scheduler.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/server_args.py

9. 当前最重要的缓存认知恢复点：

- `KV cache` 是按 token/page 增长的前缀历史
- `linear state` 是某个前缀边界上的状态快照
- KV 更适合密集缓存
- linear state 更适合稀疏 checkpoint

对当前 Qwen3.5 本地运行，粗略量级是：

- full-attn KV: 约 `10 KB / token / rank`
- linear-state snapshot: 约 `30.5 MB / snapshot / rank`

也就是说：

- 一个 linear-state snapshot 大约相当于 `3000+` 个 full-attention prefix token 的 KV

所以一个重要结论是：

- full-attention prefix cache 更像“最长 token 前缀命中”
- linear-state prefix cache 更像“最长可用状态边界命中”

10. 用户刚刚提出了一个很值得继续做的研究主思路：

- GPU 上保留热的 linear-state checkpoint
- CPU host/pinned memory 上保留更大、更冷的 linear-state checkpoint pool
- checkpoint 密度随 prefix position 变化，靠前更密、靠后更稀
- GPU miss 时先查 host state，再决定是否 fetch 或重算

当前对这个方案的评价是：

- 方向是对的
- 但如果要写成 ASPLOS 风格论文，不能只做“CPU 上多存一些 state”
- 更强的形式应该是：
  - hybrid prefix checkpoint object
  - GPU/CPU 多层级 state residency
  - checkpoint placement policy
  - fetch vs recompute vs GPU-hit cost model
  - 异步 prefetch / load-back
  - 可选的 state compression

这个方向已经被单独归档为：

- /home/wuhang/wuhang/linear_wh/docx/articles/2026-03-27_linear_state_prefix_cache_brainstorm.md

11. 当前最重要的 warm/hot threads：

- hot:
  - Qwen3.5 baseline and hybrid-attention optimization
  - benchmark persistence discipline
  - prefix/radix cache and linear-state-aware checkpoint design
- warm:
  - `LinearLayout -> subgraph planning`

12. 当前最重要的“不要重复做”的事：

- 不要重新从头搭环境
- 不要重新证明模型能跑起来
- 不要再把 benchmark 输出只留在会话里
- 不要把 full-attention KV cache 和 linear state 当成同一种东西来做缓存策略
- 不要直接把“CPU state pool”写成工程 patch 再说，要先明确论文级的问题定义和系统抽象

13. 失忆后建议的恢复顺序：

先看规范和当前压缩恢复说明：

- /home/wuhang/wuhang/linear_wh/docx/next_step.txt
- /home/wuhang/wuhang/linear_wh/docx/2026-03-27_context_compression_recovery_note.md

再看当前 stage / active threads：

- /home/wuhang/wuhang/linear_wh/docx/context_index/01_stage_summary.md
- /home/wuhang/wuhang/linear_wh/docx/context_index/02_active_threads.md
- /home/wuhang/wuhang/linear_wh/docx/context_index/03_current_required_process_docs.md

再看与当前任务最相关的技术文档：

- /home/wuhang/wuhang/linear_wh/docx/articles/2026-03-27_qwen35_hybrid_attention_brainstorm.md
- /home/wuhang/wuhang/linear_wh/docx/articles/2026-03-27_linear_state_prefix_cache_brainstorm.md
- /home/wuhang/wuhang/linear_wh/codex_coding/results/2026-03-27_qwen35_online_baseline_summary.md

再看过程文档：

- /home/wuhang/wuhang/linear_wh/code_building/process_docs/2026-03-20_development_consensus.md
- /home/wuhang/wuhang/linear_wh/code_building/process_docs/2026-03-20_project_scope_from_citations.md
- /home/wuhang/wuhang/linear_wh/code_building/process_docs/2026-03-27_qwen35_hybrid_attention_brainstorm_archive.md
- /home/wuhang/wuhang/linear_wh/code_building/process_docs/2026-03-27_linear_state_prefix_cache_brainstorm_archive.md

然后再回看关键源码：

- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/radix_cache.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/mamba_radix_cache.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/hi_mamba_radix_cache.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/memory_pool.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/configs/mamba_utils.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/managers/scheduler.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/managers/schedule_batch.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/server_args.py

最后再回看支持文档和模型卡：

- /home/wuhang/wuhang/linear_wh/sglang/docs/basic_usage/qwen3.md
- /home/wuhang/wuhang/linear_wh/sglang/docs/basic_usage/qwen3_5.md
- /home/wuhang/wuhang/linear_wh/sglang/docs/advanced_features/attention_backend.md
- /home/wuhang/models/Qwen3.5-35B-A3B/config.json
- /home/wuhang/models/Qwen3.5-35B-A3B/README.md

14. 最重要的恢复记忆句：

- 当前已经完成项目规范对齐、citation-based 研究定位、`LinearLayout` 高层分析、Qwen3.5 本地 SGLang 在线 baseline 跑通、hybrid attention 方向梳理，以及 full-attention prefix/radix cache 与 linear-state-aware cache 的差异分析；下一步不该重复环境和跑通验证，而该决定是否把“CPU/host 侧 linear-state checkpoint pool + checkpoint placement + fetch-vs-recompute policy”正式推进成一个 `Hybrid Prefix Checkpoint Cache` 研究设计。

这一轮：
请你先回顾一下上一轮我们做了什么，然后，这一轮你的任务如下：

1. 先重新阅读这些文档：

- /home/wuhang/wuhang/linear_wh/docx/next_step.txt
- /home/wuhang/wuhang/linear_wh/docx/2026-03-27_context_compression_recovery_note.md
- /home/wuhang/wuhang/linear_wh/docx/context_index/01_stage_summary.md
- /home/wuhang/wuhang/linear_wh/docx/context_index/02_active_threads.md
- /home/wuhang/wuhang/linear_wh/docx/context_index/03_current_required_process_docs.md
- /home/wuhang/wuhang/linear_wh/docx/articles/2026-03-27_qwen35_hybrid_attention_brainstorm.md
- /home/wuhang/wuhang/linear_wh/docx/articles/2026-03-27_linear_state_prefix_cache_brainstorm.md
- /home/wuhang/wuhang/linear_wh/codex_coding/results/2026-03-27_qwen35_online_baseline_summary.md

2. 然后优先回看这些关键代码：

- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/radix_cache.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/mamba_radix_cache.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/hi_mamba_radix_cache.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/mem_cache/memory_pool.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/configs/mamba_utils.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/managers/scheduler.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/managers/schedule_batch.py
- /home/wuhang/wuhang/linear_wh/sglang/python/sglang/srt/server_args.py

3. 重点恢复这几个认知：

- full-attention prefix/radix cache 复用的是前缀对应的 KV cache
- hybrid linear-attention 模型不能只复用 KV，还要复用前缀边界上的 linear-state snapshot
- 当前 SGLang 已经区分：
  - `RadixCache`
  - `MambaRadixCache`
  - `HiMambaRadixCache`
- 当前服务实际状态是：
  - `disable_radix_cache=False`
  - `page_size=1`
  - `mamba_scheduler_strategy=no_buffer`
- linear state 比 KV cache 粗粒度得多，单位 snapshot 的显存成本远高于单 token KV
- 当前最值得推进的研究主思路是：
  - `Hybrid Prefix Checkpoint Cache`
  - GPU/CPU 双层 state residency
  - checkpoint placement policy
  - fetch vs recompute cost model

4. 下一步不要重复做已经完成的事情：

- 不要重新从头搭环境
- 不要重新证明模型能跑起来
- 不要重新做同样的在线 benchmark 只是为了确认服务可用

5. 下一步如果继续，最合理的方向是：

- 把“CPU/host 侧 linear-state checkpoint pool”正式收束成一个设计草案
- 定义：
  - checkpoint object
  - GPU hot tier
  - CPU cold tier
  - placement strategy
  - restore strategy
  - evaluation plan
- 并明确这条线要不要发展成 ASPLOS 风格的系统论文问题

6. 但在那之前，先和用户确认下一轮要优先做哪一种：

- A. 直接写 `Hybrid Prefix Checkpoint Cache` 的详细设计文档
- B. 先做更细的源码级变量和执行路径梳理
- C. 先把 benchmark harness 和结果持久化链路补完整

得到用户同意后再继续。
】
