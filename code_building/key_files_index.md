# Key Files Index

## 2026-04-09

- 路径：`/home/wuhang/wuhang/dllm_wh/docx/00_docx_storage_and_lookup_guide.md`
  - 作用：说明 `docx/` 根目录、`cites/`、`articles/`、`context_index/`、`plans/` 的职责边界，以及重要文档的存放规则和查找顺序。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.1-docx_storage_and_lookup_guide_archive.md`
  - 作用：记录本次 `docx` 文档地图与查找指南的形成过程、验证范围、结果和本轮命令。
- 路径：`/home/wuhang/wuhang/dllm_wh/docx/articles/2026-04-09_diffusion_llm_inference_and_acceleration_landscape.md`
  - 作用：系统总结 diffusion LLM 的定义、推理流程、与 AR 的差异、当前 text dLLM 加速技术分层，以及建议优先推进的优化方向。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.2-diffusion_llm_acceleration_landscape_archive.md`
  - 作用：记录本轮对 `dInfer`、`SGLang` diffusion LLM 代码与官方资料的阅读过程、结论与本轮命令。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/dinfer_dllm_demo.py`
  - 作用：提供一个最小自包含的 dInfer 风格 diffusion LLM demo 脚本，可展示 block-wise threshold 去噪过程、生成文本和速度指标。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.3-dinfer_demo_script_archive.md`
  - 作用：记录 demo 脚本的设计约束、验证结果以及当前环境中 `dinfer` 可选依赖缺失的实际情况。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/run_dinfer_llada2_mini.py`
  - 作用：按 dInfer 官方 benchmark 路线正式跑 `LLaDA2.0-mini`，支持 `tp=1` 的真实模型加载、生成和测速。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/dinfer_llada2_mini_metrics.json`
  - 作用：保存本轮正式 `LLaDA2.0-mini` 实验的关键指标和生成文本。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/2026-04-09_dinfer_llada2_mini_formal_run.md`
  - 作用：汇总本轮正式实验配置、吞吐、forward 次数和生成输出。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.4-dinfer_llada2_mini_formal_run.md`
  - 作用：记录正式跑通 `LLaDA2.0-mini` 的环境修复、执行过程和结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/docx/2026-04-09_context_compression_recovery_note.md`
  - 作用：为上下文压缩后的后续 agent 提供高密度恢复说明，明确 must-read 文件、不要重复做的工作、当前环境状态以及下一步需要先向用户确认的分叉方向。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.5-context_compression_recovery_handoff.md`
  - 作用：记录本轮恢复说明与 `context_index` 刷新的形成过程、验证依据和本轮命令。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/2026-04-09_post_recovery_revalidation.md`
  - 作用：记录恢复后对当前 live 环境的最小复验结果，明确指出历史 formal run 仍然成立，但当前环境已不能直接复现。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.6-post_recovery_revalidation_and_env_drift.md`
  - 作用：记录本轮环境复验、版本漂移确认、`std::bad_alloc` 复现，以及为何下一步应先做回归定位而不是直接扩 benchmark。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/dinfer_llada2_mini_revalidation_metrics.json`
  - 作用：torch 回退至 2.8.0+cu128 后的 formal path 复验指标，确认结果与历史归档一致。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.7-env_regression_diagnosis_and_torch_rollback.md`
  - 作用：记录 `std::bad_alloc` 根因诊断（vllm 0.10.2 编译扩展与 torch 2.9.1 ABI 不兼容）、torch 回退修复过程和复验结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/sweep_dinfer_llada2_mini.py`
  - 作用：dInfer + LLaDA2.0-mini 的参数 sweep 脚本，用于建立性能 baseline。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/sweep_dinfer_llada2_mini_results.json`
  - 作用：threshold sweep 的完整结果 JSON，包含每个配置的 3 次计时数据。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.8-dllm_performance_sweep_and_baseline.md`
  - 作用：记录 sweep 过程、结果和 baseline 配置选定。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_all_paths.py`
  - 作用：全路径对比脚本，测试所有 dInfer 推理路径的吞吐。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_long_prompt.py`
  - 作用：长 prompt (662 tok) 对比脚本，含 IterSmooth 和 VicinityCache 路径。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.9-all_paths_benchmark_and_baseline.md`
  - 作用：记录全路径对比结果和最终 baseline 确定。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_itersmooth_vicinity.py`
  - 作用：IterSmooth / VicinityCache 修复后的全面对比脚本（含短/长 prompt 6 条路径）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/itersmooth_vicinity_benchmark_results.json`
  - 作用：IterSmooth / VicinityCache 对比的完整结果 JSON。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.10-itersmooth_vicinity_fix_and_benchmark.md`
  - 作用：记录 IterSmooth/VicinityCache 修复过程、benchmark 结果、BD+IterSmooth 结合方案设计。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_bd_itersmooth.py`
  - 作用：BD+IterSmooth 对比脚本，测试 BD-attnmask baseline vs BD+IterSmooth (no-cache/cache, default/high_w)。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bd_itersmooth_benchmark_results.json`
  - 作用：BD+IterSmooth 全路径对比的完整结果 JSON（含 cache 路径）。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.11-bd_itersmooth_combination.md`
  - 作用：记录 BD+IterSmooth 结合方案的实现、两轮 benchmark 结果和关键结论。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_lazy_cache.py`
  - 作用：Lazy cache + inplace write 对比脚本。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_batch_4paths.py`
  - 作用：4 路径（no-cache/IS-nocache/cache-opt/IS+cache）× batch=1,4,8,16,32 完整对比脚本。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/batch_4paths_benchmark_results.json`
  - 作用：4 路径 batch scaling 完整结果 JSON。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.12-kv_cache_optimization_and_batch_scaling.md`
  - 作用：记录 KV cache 优化（lazy+inplace）、batch>1 支持、4 路径 batch scaling 对比结果和关键结论。

## 2026-04-12

- 路径：`/home/wuhang/wuhang/dllm_wh/docx/articles/2026-04-12_dllm_moe_insight_and_optimization_directions.md`
  - 作用：系统分析 dLLM MoE 推理与 AR 的三个核心差异（MASK routing 集中、跨迭代冗余、天然批量），与 13 篇 MoE 系统论文的交叉分析，提出 "Iteration-Aware Selective MoE" 论文方向，包含多卡 EP 视角和待验证实验设计。
- 路径：`/home/wuhang/wuhang/dllm_wh/docx/cites/moe_systems_survey.md`
  - 作用：近两年系统会议（OSDI/SOSP/ASPLOS/EuroSys/SC/ATC）的 MoE 推理/训练系统论文调研，含 13 篇论文的逐篇解读、问题到技术反查索引和迁移到推理的优先级排序。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_profile_kernels.py`
  - 作用：torch.profiler 算子级 profiling 脚本，对 cache/no-cache 路径做 CUDA kernel 分类统计。

## 2026-04-13

- 路径：`/home/wuhang/wuhang/dllm_wh/docx/articles/2026-04-12_dllm_moe_complete_experiment_report.md`
  - 作用：面向学术讨论的完整实验报告，包含 dLLM 背景、三个 Insight 完整量化数据、MoE 内部结构分析、Stable Cache 失败分析、Padding-Free kernel 性能分析、与 13 篇论文交叉分析、5 个开放问题。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.13-dllm_moe_complete_experiment_report_archive.md`
  - 作用：v0.1.13 归档文档，记录本轮所有实验成果、结论和下一步方向。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/padding_free_moe.py`
  - 作用：基于 X-MoE PFT 方案的完整 padding-free MoE 实现（dispatch + gather + grouped GEMM + scatter + BF16）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_padding_free_moe.py`
  - 作用：Padding-free MoE kernel 的正确性验证和性能对比脚本。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/padding_free_moe_benchmark_results.json`
  - 作用：Padding-free kernel 性能数据（证明 memory-bound 下无法超越 vllm baseline）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/padding_free_correctness_check.json`
  - 作用：Padding-free kernel 正确性验证数据（cosine sim = 0.999994）。

## 2026-04-14

- 路径：`/home/wuhang/wuhang/dllm_wh/docx/plans/2026-04-14_v0.1.15_coupled_optimization_plan.md`
  - 作用：方向一（Token Temporal Reuse）× 方向二（Expert Adaptive Pruning）耦合优化实验计划。包含耦合架构（底座+叠加）、三个隐含假设、联合参数扫描方案、退路决策树（区分假设 A/B/C 被打破的应对策略）、Pareto frontier 分析框架和成功标准。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/moe_coupled_e2e.py`
  - 作用：D1×D2 耦合优化 E2E 实验脚本，包含 CoupledController（合并 temporal reuse + top-p pruning）、13 配置联合扫描、假设验证和质量检查。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/coupled_optimization_results.json`
  - 作用：耦合优化实验完整结果 JSON，包含 13 个配置的 ΔFwd、savings%、reuse%、avg_expert 数据。关键发现：假设 C 被打破（误差共振 +10），假设 B 成立（proxy 比率 0.95）。
- 路径：`/home/wuhang/wuhang/dllm_wh/docx/plans/2026-04-14_v0.1.15.2_wall_clock_realization_plan.md`
  - 作用：Wall-clock 收益兑现的完整里程碑计划。包含 7 个开销源清单（C1-C7）、3 步 profiling 方案（kernel micro-benchmark → 逐组件 profiling → hook overhead isolation）、决策分叉树（α/β/γ 三种情况对应不同工程路线）、成功标准和关键代码位置索引。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_fused_experts_weight_zero.py`
  - 作用：fused_experts kernel micro-benchmark，测试 weight=0 vs 物理裁剪 vs baseline 在不同 token 数下的 kernel 时间。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/fused_experts_weight_zero_benchmark.json`
  - 作用：kernel micro-benchmark 完整数据。关键结论：情况 γ-worst，weight=0 不省 kernel 时间，物理裁剪效果也有限（memory-bound）。

## 2026-04-27

- 路径：`/home/wuhang/wuhang/dllm_wh/history-chat.txt`
  - 作用：当前上下文压缩恢复说明。按用户模板记录上一轮信息和下一轮恢复任务，重点包含 Scheme3 8 卡结论、insight ledger、必读文档和下一轮“先讨论方向、不直接代码建设”的约束。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12i-context_compression_recovery_handoff.md`
  - 作用：记录本次上下文压缩恢复交接的归档过程、更新文件、恢复重点和本轮命令。
- 路径：`/home/wuhang/wuhang/dllm_wh/docx/context_index/01_stage_summary.md`
  - 作用：当前阶段摘要。已从早期 dInfer/LLaDA2 benchmark 阶段更新为 C12-AgRs Scheme3 8 卡验证后的 insight-led direction selection 阶段。
- 路径：`/home/wuhang/wuhang/dllm_wh/docx/context_index/02_active_threads.md`
  - 作用：当前活跃线程索引。最高优先级为 insight-led direction selection，Scheme3 standalone 降级，Sequence/TP、native active-expert reduction、block-stage-aware scheduler 保持 warm。
- 路径：`/home/wuhang/wuhang/dllm_wh/docx/context_index/03_current_required_process_docs.md`
  - 作用：当前最小必读过程文档集合。用于压缩后快速恢复到 Scheme3 / insight ledger 方向选择现场。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_bsp_moe_dp2.py`
  - 作用：BSP-MoE 独立 monkey-patch 验证脚本，支持 baseline/BSP compare、shape probe、forward-check、component timing 和 verifiable quality snippet 输出。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12j-bsp_moe_validation.md`
  - 作用：记录 BSP-MoE C12 8 卡机制验证、shape probe、forward-check、端到端结果、组件 timing、人工质量检查和方向结论。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_c12_8g_e2e_summary_20260427.json`
  - 作用：BSP-MoE C12 无 component timing 端到端 summary；baseline `20.1285s / 75.675 ms/fwd`，BSP `19.8475s / 74.615 ms/fwd`。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_c12_8g_component_summary_20260427.json`
  - 作用：BSP-MoE C12 component timing summary，记录 dispatch payload `826.877 -> 206.719 MB/fwd` 以及 combine / TP all-gather 开销。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_forward_check.json`
  - 作用：BSP-MoE 与 baseline MoE block 的层级 forward-check 结果，覆盖 layers 0/9/18 的 cold/hot_skip/hot_update 路径。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_shape_probe.json`
  - 作用：BSP-MoE shape probe JSON，确认 C12 下 block MoE token 数 `N_dp=8192`，BSP 后 `N_sp=2048`，无 padding。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/analyze_nsys_bsp.py`
  - 作用：BSP-MoE A/B nsys sqlite 分析脚本；按 `*.generate.run1` NVTX window 汇总 kernel category、collective split、memcpy、NVTX component，并生成 JSON/Markdown 报告。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/nsys_bsp_short_nvtx_analysis_20260427.md`
  - 作用：BSP-MoE 短 nsys trace 可读报告；核心结论是 BSP 降低 D2D/dense GEMM/部分 MoE wrapper 时间，但 NCCL `AllGather` 和 `Reduce` 放大超过 200%，吃掉收益。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/nsys_bsp_short_nvtx_analysis_20260427.json`
  - 作用：BSP-MoE 短 nsys trace 结构化结果；保留每类 kernel/collective/memcpy 的 rankmax、count、total_ms，便于后续脚本复用和对比更长 trace。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12k-bsp_nsys_collective_profiling.md`
  - 作用：记录 BSP-MoE nsys collective profiling 的目标、输入 trace、分析方法、A/B 表格、源码解释和后续方向建议。
- 路径：`/home/wuhang/wuhang/dllm_wh/history-chat.txt`
  - 作用：最新上下文压缩恢复说明，覆盖旧内容。重点恢复 BSP-MoE 机制验证、nsys collective profiling、关键数据、必读文件和下一轮“先讨论、不直接代码建设”的约束。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12l-context_compression_recovery_handoff_bsp_nsys.md`
  - 作用：记录本次 BSP/nsys 上下文压缩交接的目标、恢复重点、必读文件、不要遗忘的判断和本轮命令。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_m123_smoke_abcd_20260427.json`
  - 作用：M1/M2/M3 A/B/C/D smoke 结果；确认 D 路径 EP pop all-reduce 不再挂起，path_counts 在 8 rank 一致。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_m123_c12_e2e_abcd_20260427.json`
  - 作用：M1/M2/M3 C12 no-timing 端到端结果；C BSP-DelayGather 最优，`76.04 -> 74.35 ms/fwd`，约 `-2.22%`。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_m123_c12_component_abcd_20260427.json`
  - 作用：M1/M2/M3 C12 component timing 结果；记录 dispatch payload `826.877 -> 206.719 MB/fwd`，以及 C/D 在 `native_forward/quant_apply/combine` 上的差异。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12m-m123_bsp_delay_m3_experiment.md`
  - 作用：记录 M1/M2/M3 建设、D 路径 collective 修复、smoke/C12 e2e/component timing、以及下一步源码下沉建议。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_cplus_cxx_smoke_20260427.json`
  - 作用：BSP C+/C++ A/B/C/D/E/F smoke 结果；确认新增 E/F 路径可运行，path_counts 在 8 rank 一致。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_cplus_cxx_c12_e2e_20260427.json`
  - 作用：BSP C+/C++ 第一轮 C12 no-timing 结果；E `71.90 ms/fwd` 最优，F `72.41 ms/fwd` 次优。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_cplus_cxx_c12_e2e_repeat_20260428.json`
  - 作用：BSP C+/C++ C12 no-timing 复测结果；E `71.69 ms/fwd`，F `72.51 ms/fwd`，确认新路径收益稳定。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_cplus_cxx_c12_component_20260427.json`
  - 作用：BSP C+/C++ C12 component timing 结果；记录 E 的 SP norm/native/quant/combine 分解，以及 F 的 `ep_full_all_reduce=9.467 ms/fwd`、`ep_full_allreduce_payload=1311.020 MB/fwd`。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_cplus_cxx_quality_smoke_20260428.json`
  - 作用：BSP C+/C++ 小规模质量 smoke；E/F snippets 未见灾难性语义崩坏，但只作为 smoke 质量证据。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12n-cplus_cxx_bsp_upper_bound.md`
  - 作用：记录 BSP C+ / C++ upper-bound 实验的实现、C12 两轮 e2e、component timing、quality smoke 和源码下沉建议。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_smoke_20260428.json`
  - 作用：BSP-G A/B/C/D/E/G/F smoke 结果；确认 attention reduce-scatter SP 路径可运行且不挂起。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_c12_e2e_20260428.json`
  - 作用：BSP-G 第一轮 C12 no-quality 结果；G `69.55 ms/fwd`，优于 A `75.35` 和 E `71.67`。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_c12_e2e_repeat_20260428.json`
  - 作用：BSP-G C12 no-quality 复测结果；G `69.51 ms/fwd`，确认收益稳定。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_c12_component_20260428.json`
  - 作用：BSP-G component timing 结果；记录 `attn.tp_reduce_scatter=5.020 ms/fwd`、`attn_rs_payload=661.502 MB/fwd`、`moe.bsp_chunk` 降到 `0.003 ms/fwd`。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_quality_smoke_20260428.json`
  - 作用：BSP-G 小规模质量 smoke；G snippets 未见灾难性语义崩坏，但只作为 smoke 质量证据。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12o-bsp_g_vllm_sp_parity.md`
  - 作用：记录 BSP-G vLLM SP-parity 实验的实现、C12 两轮 e2e、component timing、quality smoke、环境 OOM 处理和下一步 source-downshift 建议。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg2_smoke_20260428.json`
  - 作用：BSP-G2 A/E/G/G2/F smoke 结果；确认 G2 attention-input gather ownership 迁移后不挂起。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg2_c12_e2e_20260428.json`
  - 作用：BSP-G2 C12 no-quality 结果；G2 `69.661 ms/fwd`，与 G `69.676 ms/fwd` 基本持平，A 为 `75.428 ms/fwd`。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg2_c12_component_20260428.json`
  - 作用：BSP-G2 component timing 结果；记录 `attn.input_all_gather=2.573 ms/fwd`、`moe.tp_all_gather` 降到 `0.141 ms/fwd`，证明 gather bucket 从 MoE wrapper 迁移到 attention input。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg2_quality_smoke_20260428.json`
  - 作用：BSP-G2 小规模质量 smoke；G2 snippets 与 G 基本一致，未见新增灾难性语义崩坏。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12p-bsp_g2_sp_parity_bundle.md`
  - 作用：记录 BSP-G2 vLLM SP-parity bundle 的代码状态、实验前归档、smoke/C12/component/quality 结果和最终判断。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12q-vllm_sp_parity_inventory.md`
  - 作用：持续跟踪 vLLM SP-parity / SP-MoE / sequence-parallelism 点位确认状态；直到每一项确认前，不认为 vLLM SP 相关可搬性能收益已经全部盘完。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12r-bsp_g3_to_g7_vllm_sp_completion.md`
  - 作用：记录 BSP-G3 到 BSP-G7 的连续确认结果：VSP-06 shared expert、BSP-G source-downshift 设计、VSP-10 AsyncTP pattern-hit、VSP-08 backend inventory、VSP-11/12 source landing 约束和最终下一步建议。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/probe_vllm_async_tp_patterns.py`
  - 作用：只读 probe，用于复查 vLLM sequence-parallel / AsyncTP 的 op/pass 环境前提，并扫描当前 BSP-G/G2 源码是否具备概念上的 `GEMM+ReduceScatter` / `AllGather+GEMM` pattern 形态。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/vllm_async_tp_pattern_probe_20260428.json`
  - 作用：AsyncTP pattern probe 结果；确认 vLLM pass/op 前提存在，`symm_mem.fused_*` 可注册，但当前 benchmark 未使用 vLLM compile pass manager。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/probe_vllm_moe_backends.py`
  - 作用：只读 backend inventory probe，检查 AgRs/Naive/DeepEP/PPLX/FlashInfer all2all/FlashInfer Cutlass 的模块、env 和 vLLM helper 可用性。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/vllm_moe_backend_inventory_20260428.json`
  - 作用：MoE backend inventory 结果；AgRs 默认可用，FlashInfer all2all capability 可用，DeepEP ABI-broken，PPLX absent，FlashInfer Cutlass 非当前首选。

## 2026-04-30

- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.15-eb_eplb_source_landing_execution.md`
  - 作用：持续记录 EB-aware EPLB source landing（P1~P5）主线；已新增 `E=34 tuned-path补齐` pre-archive 与实验结果回填。
- 路径：`/home/wuhang/miniconda3/envs/dllm/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/configs/E=34,N=512,device_name=NVIDIA_H100_80GB_HBM3.json`
  - 作用：为 P5 local experts=34 路径补齐首版 MoE config（由 E=32,N=512 模板复制），用于消除 default/fallback config 路径。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_p5_ab_a_linear_true_e34cfg_20260430_summary.json`
  - 作用：E34 配置补齐后的 A 组短实验 summary（`num_redundant_experts=0`，`ms/fwd=144.381`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_p5_ab_c_replica_true_e34cfg_20260430_summary.json`
  - 作用：E34 配置补齐后的 C 组短实验 summary（`num_redundant_experts=13`，`ms/fwd=157.488`，日志确认命中 E34 配置文件）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_p5_ab_a_linear_true_jointmap_20260430_summary.json`
  - 作用：P1/P5 联合编排首版后 A 组短实验 summary（`ms/fwd=140.744`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_p5_ab_b_p1_true_jointmap_20260430_summary.json`
  - 作用：P1/P5 联合编排首版后 B 组短实验 summary（P1 map, no replica，`ms/fwd=137.330`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_p5_ab_c_replica_true_jointmap_20260430_summary.json`
  - 作用：P1/P5 联合编排首版后 C 组短实验 summary（P1 map + P5 replica，`ms/fwd=155.162`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_factor_a0_linear_e32_20260430_summary.json`
  - 作用：C 慢因子归因矩阵 A0（`E=32`, linear）基线结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_factor_c0_linear_e34_nort_20260430_summary.json`
  - 作用：C 慢因子归因矩阵 C0（`E=34`, linear, skip runtime map）结果，用于估计纯 E34 形状开销。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_factor_c1_linear_e34_rt_20260430_summary.json`
  - 作用：C 慢因子归因矩阵 C1（`E=34`, linear, runtime map on）结果，用于估计 runtime map 增量开销。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_factor_b0_p1_e32_20260430_summary.json`
  - 作用：C 慢因子归因矩阵 B0（`E=32`, P1）结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_factor_c2_p1_e34_nort_20260430_summary.json`
  - 作用：C 慢因子归因矩阵 C2（`E=34`, P1, skip runtime map）结果，用于估计 P1 场景纯 E34 形状开销。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_factor_c3_p1_e34_rt_20260430_summary.json`
  - 作用：C 慢因子归因矩阵 C3（`E=34`, P1, runtime map on）结果，用于估计 P1 场景 runtime map 增量开销。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/collect_eb_heteval512_laws.py`
  - 作用：新增 runtime EPLB record gating 实验入口（`--eplb-runtime-record-mode {full,cold_only,off}`）与 route-path 感知 patch，用于验证“cold 保留 record、hot 降低 record”的收益。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_recordmode_c3_full_20260430_summary.json`
  - 作用：runtime record gating C3-r1（`full`）结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_recordmode_c3_coldonly_20260430_summary.json`
  - 作用：runtime record gating C3-r1（`cold_only`）结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_recordmode_c3_off_20260430_summary.json`
  - 作用：runtime record gating C3-r1（`off`）结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_recordmode_c3_full_r2_20260430_summary.json`
  - 作用：runtime record gating C3-r2（`full`）复测结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_recordmode_c3_coldonly_r2_20260430_summary.json`
  - 作用：runtime record gating C3-r2（`cold_only`）复测结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_recordmode_c3_off_r2_20260430_summary.json`
  - 作用：runtime record gating C3-r2（`off`）复测结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_coldrearrange_smoke_rg8_fix_20260430_summary.json`
  - 作用：cold-gated rearrange 修复后 smoke summary，确认 trigger 成功（`attempt=1,success=1`）及单次重排耗时量级（约 `3.2~3.3s`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_coldrearrange_c3_base_20260430_summary.json`
  - 作用：cold-gated rearrange C3 对照组（base）summary，作为 `record_mode=cold_only` 下无重排基线（`149.587 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_coldrearrange_c3_rg8_20260430_summary.json`
  - 作用：cold-gated rearrange C3 `gap=8` summary，观测在线重排成本对端到端时延的影响（`201.459 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_coldrearrange_c3_rg16_20260430_summary.json`
  - 作用：cold-gated rearrange C3 `gap=16` summary，用于与 `gap=8` 对比触发频率敏感性（`201.898 ms/fwd`）。

- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/collect_eb_heteval512_laws.py`
  - 作用：新增 runtime map 实现开关 `--eplb-runtime-map-impl`（`vllm/flat_compile/flat_eager`）和 `flat-index` 映射路径；用于在不改 EB 语义下优化 EPLB logical->physical map 开销。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_mapimpl_smoke_vllm_20260430_summary.json`
  - 作用：runtime map impl smoke 基线（`vllm`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_mapimpl_smoke_flatc_20260430_summary.json`
  - 作用：runtime map impl smoke `flat_compile` 结果（用于识别 compile 负优化）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_mapimpl_smoke_flate_20260430_summary.json`
  - 作用：runtime map impl smoke `flat_eager` 结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_mapimpl_c3_vllm_seq_20260430_summary.json`
  - 作用：C3 正式口径串行 r1 的 `vllm` map 结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_mapimpl_c3_flate_seq_20260430_summary.json`
  - 作用：C3 正式口径串行 r1 的 `flat_eager` map 结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_mapimpl_c3_vllm_seq_r2_20260430_summary.json`
  - 作用：C3 正式口径串行 r2 的 `vllm` map 复测结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_mapimpl_c3_flate_seq_r2_20260430_summary.json`
  - 作用：C3 正式口径串行 r2 的 `flat_eager` map 复测结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_mapimpl_c3_flate_hotrep_seq_20260430_summary.json`
  - 作用：`flat_eager` 路径下启用 hot-replica ids 的效果复核结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py`
  - 作用：runtime map source-downshift 主实现。新增 `configure_eplb_runtime_map_policy / set_eplb_runtime_route_path / use_eplb_runtime_map_policy`，将 `flat_eager + cold_only` 从脚本 patch 下沉到 dInfer 源码可控路径。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/collect_eb_heteval512_laws.py`
  - 作用：接入 dInfer 源码策略接口，删除脚本内 runtime map patch；`--eplb-runtime-map-impl` 收敛为 `{vllm, flat_eager}`。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_srcdown_c3_vllm_seq_20260430_summary.json`
  - 作用：源码下沉后 C3 串行 r1 的 `vllm + cold_only` 对照结果（`155.228 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_srcdown_c3_flate_seq_20260430_summary.json`
  - 作用：源码下沉后 C3 串行 r1 的 `flat_eager + cold_only` 结果（`143.272 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_srcdown_c3_vllm_seq_r2_20260430_summary.json`
  - 作用：源码下沉后 C3 串行 r2 的 `vllm + cold_only` 复测结果（`154.932 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_srcdown_c3_flate_seq_r2_20260430_summary.json`
  - 作用：源码下沉后 C3 串行 r2 的 `flat_eager + cold_only` 复测结果（`139.569 ms/fwd`）。

- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_nativegate_smoke_flate_20260430_summary.json`
  - 作用：native gate 可用性验证 smoke（`flat_eager+cold_only` 对照）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_nativegate_smoke_nativecold_20260430_summary.json`
  - 作用：native gate 可用性验证 smoke（`vllm+full+native(cold_only)`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_nativegate_c3_flate_20260430_summary.json`
  - 作用：native gate 可用性验证 C3 对照组（`145.457 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_nativegate_c3_nativecold_20260430_summary.json`
  - 作用：native gate 可用性验证 C3 实验组（`155.605 ms/fwd`）；并用于确认当前运行态 `native_record_gate` 未生效（`available=False`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py`
  - 作用：新增 runtime map 的 `tensor_cache` 可控开关（`on/off`）与默认策略，支持 A/B 对照。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/collect_eb_heteval512_laws.py`
  - 作用：新增 `--eplb-runtime-tensor-cache` 和 `--native-record-gate-mode` 参数；可输出 native gate 可用性诊断信息。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_cacheab_c3_off_r1_20260430_summary.json`
  - 作用：`tensor_cache=off` C3 A/B r1（`139.116 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_cacheab_c3_on_r1_20260430_summary.json`
  - 作用：`tensor_cache=on` C3 A/B r1（`172.328 ms/fwd`，显著回退）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_cacheab_c3_off_r2_20260430_summary.json`
  - 作用：`tensor_cache=off` C3 A/B r2（`138.536 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_cacheab_c3_on_r2_20260430_summary.json`
  - 作用：`tensor_cache=on` C3 A/B r2（`146.131 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py`
  - 作用：runtime fastpath 主实现更新。新增 `path_is_cold` 布尔状态、`DINF_EPLB_RUNTIME_FASTPATH` 开关，以及 `cold_only+flat_eager+cache=off` 专用 `_patched_fast` 路径，用于减少热路径动态分支/字符串判断开销。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_fastpathab_c3_off_r1_20260430_summary.json`
  - 作用：runtime fastpath A/B（`fastpath=off`）C3 r1 结果（`140.442 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_fastpathab_c3_off_r2_20260430_summary.json`
  - 作用：runtime fastpath A/B（`fastpath=off`）C3 r2 结果（`141.700 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_fastpathab_c3_on_r1_20260430_summary.json`
  - 作用：runtime fastpath A/B（`fastpath=on`）C3 r1 结果（`135.206 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_fastpathab_c3_on_r2_20260430_summary.json`
  - 作用：runtime fastpath A/B（`fastpath=on`）C3 r2 结果（`136.837 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_fastpath_c3_flateoff_r1_20260430_summary.json`
  - 作用：runtime fastpath 先验 r1（未加可控开关）结果（`140.578 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_fastpath_c3_flateoff_r2_20260430_summary.json`
  - 作用：runtime fastpath 先验 r2（未加可控开关）结果（`148.048 ms/fwd`，用于识别波动并触发可控 A/B）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_fastpathv2_c3_on_r1_20260430_summary.json`
  - 作用：fastpath-v2（局部缓存尝试）C3 r1 结果（`142.370 ms/fwd`，用于判定负优化）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_fastpathv2_c3_on_r2_20260430_summary.json`
  - 作用：fastpath-v2（局部缓存尝试）C3 r2 结果（`146.753 ms/fwd`，确认负优化稳定存在）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_fastpath_revertcheck_c3_on_r1_20260430_summary.json`
  - 作用：回退到 fastpath-v1 后的恢复性校验结果（`136.527 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_routeopt_c3_on_r1_20260430_summary.json`
  - 作用：route-layer overhead trim 尝试 run1 结果（`139.854 ms/fwd`，用于判定该路线不稳定/非持续收益）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_routeopt_c3_on_r2_20260430_summary.json`
  - 作用：route-layer overhead trim 尝试 run2 结果（`133.649 ms/fwd`，与 r1/r3 波动较大）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_routeopt_c3_on_r3_20260430_summary.json`
  - 作用：route-layer overhead trim 尝试 run3 结果（`139.408 ms/fwd`，配合 r1/r2 说明方差偏大）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_routeopt_revertcheck_c3_on_r1_20260430_summary.json`
  - 作用：route-layer 改动回退后的恢复性校验（`134.636 ms/fwd`，并确认 path counts `19/38/893/209`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_sparse_rr_c3_off_r1_20260430_summary.json`
  - 作用：sparse-rr A/B 的 `off` 组 r1（`141.610 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_sparse_rr_c3_off_r2_20260430_summary.json`
  - 作用：sparse-rr A/B 的 `off` 组 r2（`140.615 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_sparse_rr_c3_on_r1_20260430_summary.json`
  - 作用：sparse-rr A/B 的 `on` 组 r1（`143.156 ms/fwd`，负优化）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_sparse_rr_c3_on_r2_20260430_summary.json`
  - 作用：sparse-rr A/B 的 `on` 组 r2（`147.438 ms/fwd`，负优化）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_sparse_rr_revertcheck_c3_on_r1_20260430_summary.json`
  - 作用：sparse-rr 回退后的恢复性校验（`134.599 ms/fwd`），确认主线恢复并保持 path counts 不变。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_routenop_v2_c3_off_r1_20260430_summary.json`
  - 作用：route no-op skip v2 A/B 的 `off` 组 r1（`142.767 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_routenop_v2_c3_off_r2_20260430_summary.json`
  - 作用：route no-op skip v2 A/B 的 `off` 组 r2（`145.005 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_routenop_v2_c3_on_r1_20260430_summary.json`
  - 作用：route no-op skip v2 A/B 的 `on` 组 r1（`139.067 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_routenop_v2_c3_on_r2_20260430_summary.json`
  - 作用：route no-op skip v2 A/B 的 `on` 组 r2（`133.958 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_routenop_v2_val_off_r1_20260430_summary.json`
  - 作用：route no-op skip v2 稳定性复证（3x3）`off` 组 r1（`148.034 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_routenop_v2_val_off_r2_20260430_summary.json`
  - 作用：route no-op skip v2 稳定性复证（3x3）`off` 组 r2（`136.857 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_routenop_v2_val_off_r3_20260430_summary.json`
  - 作用：route no-op skip v2 稳定性复证（3x3）`off` 组 r3（`137.867 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_routenop_v2_val_on_r1_20260430_summary.json`
  - 作用：route no-op skip v2 稳定性复证（3x3）`on` 组 r1（`138.695 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_routenop_v2_val_on_r2_20260430_summary.json`
  - 作用：route no-op skip v2 稳定性复证（3x3）`on` 组 r2（`143.330 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_routenop_v2_val_on_r3_20260430_summary.json`
  - 作用：route no-op skip v2 稳定性复证（3x3）`on` 组 r3（`140.233 ms/fwd`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_wssweep_c3_w16_s16_r1_20260430_summary.json`
  - 作用：EPLB `window/step` 参数面扫（gen=32）基线结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_wssweep_c3_w64_s256_r1_20260430_summary.json`
  - 作用：EPLB `window/step` 参数面扫（gen=32）中频配置结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_wssweep_c3_w256_s1024_r1_20260430_summary.json`
  - 作用：EPLB `window/step` 参数面扫（gen=32）低频配置结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_wssweep_c3_w1000_s3000_r1_20260430_summary.json`
  - 作用：EPLB `window/step` 参数面扫（gen=32）超低频配置结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_wssweep_lgen_w16_s16_r1_20260430_summary.json`
  - 作用：EPLB 长窗口对照（gen=256，高频 step）结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_wssweep_lgen_w1000_s3000_r1_20260430_summary.json`
  - 作用：EPLB 长窗口对照（gen=256，低频 step）结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_force_step_smoke_r1_20260430_summary.json`
  - 作用：强制单步 `EplbState.step()` 诊断结果（验证 state 计数器可推进）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_stephook_off_smoke_r1_20260430_summary.json`
  - 作用：`step hook off` 对照；用于确认主线下 `EplbState.step` 未自然推进。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_stephook_on_smoke_fix3_r1_20260430_summary.json`
  - 作用：`step hook on` 实验；触发重排并记录性能回退与断言风险。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_stephook_on_smoke_fix3_r2_20260430_summary.json`
  - 作用：`step hook on` 复测；确认重排触发下性能回退现象可复现。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_stepdiag_off_smoke_r1_20260430_summary.json`
  - 作用：step/rearrange 诊断对照组；用于确认主线下 `EplbState.step` 不推进（`state_step` 固定）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_stepdiag_on_smoke_r1_20260430_summary.json`
  - 作用：step/rearrange 诊断初版；包含断言 traceback（`len(expert_weights)==num_moe_layers`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_stepdiag_on_fixlen_smoke_r1_20260430_summary.json`
  - 作用：修复 expert_weights 长度断言后的首轮结果；验证断言可清零但重排开销极大。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_stepdiag_on_fixview_smoke_r1_20260430_summary.json`
  - 作用：step 视图恢复修正后的重排开销验证结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_stepdiag_on_fixview_s256_smoke_r1_20260430_summary.json`
  - 作用：大 `step_interval`（不触发重排）下的 step 调用成本基准，用于和重排成本拆分。

- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results.json`
  - 作用：2026-05-01 C12 对齐基线（bench脚本）结果，含 A/E/G/F 的 `ms/fwd` 与 path counts；本轮用于对齐锚点（A=`75.836`，G=`70.030`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_c12align_b0_p1_e32_20260501_summary.json`
  - 作用：2026-05-01 C12 对齐下 EPLB-B0（P1,E32,cold_only,flat_eager）结果；`127.408 ms/fwd`，path counts `19/171/3933/931`。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_eplb_c12align_c3_p1_e34_20260501_summary.json`
  - 作用：2026-05-01 C12 对齐下 EPLB-C3（P1+replica,E34,cold_only,flat_eager）结果；`187.862 ms/fwd`，path counts `19/171/3933/931`。

- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_bspgsource_off_20260501.json`
  - 作用：bench 主路径 C12-off（`bspg_source`）结果，含 `A/E/G/GS` 的 `ms/fwd` 与 path counts。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_bspgsource_on_cold_flate_20260501.json`
  - 作用：bench 主路径 C12-on（EPLB runtime: `cold_only + flat_eager + redundant=13`）结果；用于与 off 做同口径 A/B。

- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_bspgsource_on32_offrecord_20260501.json`
  - 作用：开销归因矩阵 `ON32-off`（runtime on, redundant=0, record=off）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_bspgsource_on32_coldrecord_20260501.json`
  - 作用：开销归因矩阵 `ON32-cold`（runtime on, redundant=0, cold_only）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_bspgsource_on34_offrecord_20260501.json`
  - 作用：开销归因矩阵 `ON34-off`（runtime on, redundant=13, record=off）。

- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_bsp_moe_dp2.py`
  - 作用：新增 C12 双轴输出能力（每个 config 落盘 `load_balance_runs/load_balance_summary`），支持与时延同口径汇报 `ep_load_cv / ep_load_max_mean / ep_load_p95_p50`。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_lb2_off_20260501.json`
  - 作用：C12 OFF 双轴基线结果（含 A/E/G/GS 时延与负载均衡指标）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_lb2_on32_off_20260501.json`
  - 作用：C12 `ON32-off` 双轴结果（runtime on, redundant=32, record=off）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_lb2_on32_cold_20260501.json`
  - 作用：C12 `ON32-cold` 双轴结果（runtime on, redundant=32, record=cold_only）。

- 路径：`/home/wuhang/miniconda3/envs/dllm/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/configs/E=36,N=512,device_name=NVIDIA_H100_80GB_HBM3.json`
  - 作用：补齐 ON32（`E=36`）路径的 MoE kernel 配置，避免 default fallback 干扰。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_lb3_off_e36cfg_20260501.json`
  - 作用：E=36 配置补齐后的 C12 OFF 双轴对照结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_lb3_on32_off_e36cfg_20260501.json`
  - 作用：E=36 配置补齐后的 C12 ON32-off 双轴结果（用于和 lb2 对比回收幅度）。

- 路径：`/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py`
  - 作用：新增 ON32 启动排布可切换策略（`DINF_EPLB_INIT_PLACEMENT_MODE`）与 load-prior 接口（`DINF_EPLB_INIT_GLOBAL_EXPERT_LOAD_PATH`），支持 `weight_balance` 初始化排布构造。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_lb4_smoke_on32_wbinit_20260501.json`
  - 作用：`weight_balance` 初始化排布 smoke 验证结果（`batch=32,gen=32`）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_lb4_on32_oldinit_20260501.json`
  - 作用：C12 正式对照组（ON32-off + old `joint_p1_p5` 初始化排布）双轴结果。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_lb4_on32_wbinit_20260501.json`
  - 作用：C12 正式实验组（ON32-off + `weight_balance` 初始化排布）双轴结果。

- 路径：`/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py`
  - 作用：two-choice 局部性版实现与诊断修复（cold-only 触发、hot 回退、two-choice 诊断计数可见化、map-only 计数补齐、policy reset 不再抹除诊断）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_bsp_moe_dp2.py`
  - 作用：`two_choice_lb + record_mode=off` 时也下发 cold/hot route path 信号，确保 cold-only two-choice 真正触发。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_lb6_smoke_on32_flat_v3_20260501.json`
  - 作用：two-choice 实验对照 smoke（flat）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_lb6_smoke_on32_twochoice_v3_20260501.json`
  - 作用：two-choice 全程路径 smoke（确认显著负优化）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_lb6_smoke_on32_twochoice_coldonly_v2_20260501.json`
  - 作用：two-choice cold-only 路径 smoke（确认轻微负优化与诊断命中）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_lb6_on32_flat_v4_20260501.json`
  - 作用：C12 对照组（ON32 + flat_eager）用于和 two-choice cold-only 比较。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_lb6_on32_twochoice_coldonly_v2_20260501.json`
  - 作用：C12 two-choice cold-only 正式结果（含 twochoice 命中计数、decay/update 次数、双轴指标）。

- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_prof_lb7_off_20260501.json`
  - 作用：Profiling 专项 `OFF` 单次结果（含 component timing + load balance）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_prof_lb7_on32_flat_20260501.json`
  - 作用：Profiling 专项 `ON32-flat` 单次结果（runtime on, redundant=32, record=off）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_prof_lb7_on32_cold_20260501.json`
  - 作用：Profiling 专项 `ON32-cold` 单次结果（runtime on, redundant=32, record=cold_only）。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_prof_lb7_p1static_20260501.json`
  - 作用：Profiling 专项 `P1 static-only` 单次结果（build 后 runtime disable）。

- 路径：`/home/wuhang/wuhang/dllm_wh/history-chat.txt`
  - 作用：上下文压缩恢复主入口（2026-05-01 刷新版），包含当前任务阅读顺序、关键结论、待办项与复现命令；压缩后优先读取该文件恢复记忆。
- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.15-eb_eplb_source_landing_execution.md`
  - 作用：当前阶段主过程文档，已追加 `2026-05-01 22:07` 的压缩恢复归档记录（history-chat 覆盖动作与关键内容）。

## 2026-05-02

- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/vllm_balanced_phy2log_ep8_r32_20260501.pt`
  - 作用：vllm `balanced_packing` 生成的 ON32 per-layer phy2log map `[19, 288]`，用于 runtime EPLB 的 `vllm_balanced` placement mode。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/vllm_perlayer_balanced_expert_map_ep8_20260502.pt`
  - 作用：用真实 per-layer per-expert 负载数据 + vllm `balanced_packing` 生成的 E=32 per-layer balanced expert_map `[19, 256]`（19 层全部不同）。推荐的静态 EPLB 主线配置。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_hm11_off_20260502_heatmap.json`
  - 作用：Per-layer per-expert routing heatmap 数据（C12 OFF 基线），包含 19 层 × 256 expert 的 token hit count，用于负载分析和 expert_map 生成。
- 路径：`/home/wuhang/wuhang/dllm_wh/docx/context_index/04_insight_ledger.md`
  - 作用：新增 I20-I23（per-layer 独立性、expert 负载倾斜、memory-bound 削弱、per-layer balanced placement free lunch）。

## 2026-05-03

- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.16-eplb_causal_chain_and_kernel_microbench.md`
  - 作用：EPLB 因果链完整分析（11 组实验）+ kernel micro-benchmark + tiling config 发现。包含完整数据和纠错记录。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/active_expert_2d_sweep_20260503.json`
  - 作用：2D sweep (total_pairs × active_experts) 数据，证明 kernel 分布无关性和 weight loading 可见性。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_dp2_results_c12_causal_b512_20260503.json`
  - 作用：因果分析数据，含 per-rank per-layer token count 和 timing 矩阵，Pearson r=+0.93。

## 2026-05-04

- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.17-tiling_autotune_and_dispatch_combine_analysis.md`
  - 作用：Tiling config auto-tune（M=16384 已近最优）+ dispatch/combine 21.7ms 完整分解（NCCL 36%、straggler 21%、framework 42%）+ I28/I29 定义。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/autotune_fused_moe_e32.py`
  - 作用：E=32 FusedMoE tiling config grid search 脚本，384 configs sweep，验证 M=16384 下 config 近最优。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_active_experts.py`
  - 作用：Active expert sweep 脚本，验证 kernel_time = f(total_pairs) 不依赖 expert 分布。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/nccl_comm_bench.py`
  - 作用：NCCL 单次 AllGather micro-benchmark，测量 9MB payload = 0.249ms (253 GB/s)。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/nccl_overhead_isolation.py`
  - 作用：Dispatch/combine 开销分离实验，用 sync+barrier 拆分 NCCL 通信、straggler 等待、framework 三部分。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/profile_routing_heatmap.py`
  - 作用：Per-layer per-expert routing heatmap profiler，采集 MASK/decoded token 路由分布。

## 2026-05-05

- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.18-cudagraph_profiling_and_sp_lm_head.md`
  - 作用：CUDA Graph 全面探索（5 组实验，batch=512 仅省 2ms）+ GS path 完整 component timing 分解（76.4 ms/fwd）+ nsys 深度分析（71% GPU 利用率）+ OPT-2/4/5 验证 + SP LM Head 实现（-7.7ms）。定义 I30-I33。
- 路径：`/home/wuhang/wuhang/dllm_wh/docx/plans/2026-05-05_gs_path_optimization_roadmap.md`
  - 作用：GS path 优化路线图，含完整 profiling 数据、6 项优化方案（OPT-1~OPT-6）设计、实施优先级、已实测数据更新和验证框架。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/poc_cudagraph_moe.py`
  - 作用：MoE-only CUDA Graph PoC，验证单层 capture 仅省 0.93ms/19 层。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/poc_cudagraph_full_forward.py`
  - 作用：Full model forward CUDA Graph PoC，batch=64 省 30%，batch=512 OOM。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/poc_cudagraph_single_graph.py`
  - 作用：单图方案 PoC（预分配 KV cache + index_copy + SDPA mask），SDPA 引入 30ms 退化。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/poc_cudagraph_eb.py`
  - 作用：EB + CUDA Graph PoC，batch=512 下仅省 2ms。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/poc_torch_compile_native.py`
  - 作用：torch.compile 可行性测试，inductor 不兼容 EP dispatch 的 .cpu() sync。
- 路径：`/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_e2e_compare.py`
  - 作用：Baseline dInfer vs 优化版端到端对比脚本（未完成，baseline 环境不兼容）。

## 2026-05-06

- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.19-bsp_h_and_team_decoded_skip.md`
  - 作用：BSP-H AllReduce 探索（6 组实验）+ TEAM decoded-token skip 集成（v1 extract+pad、v2 null expert）。含 component timing 对比、null expert 质量崩坏 debug 状态、TEAM 安全性分析。

## 2026-05-07

- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.20-team_null_expert_quality_debug_and_optimization.md`
  - 作用：TEAM null expert 质量排查（TD1-TD7b 消融实验、TD7b 证明 kernel 正确性 diff=0）、cross_block forward 发现、prev_decoded 状态机设计（M=5 刷新）、piggyback dispatch 方案（decoded mask 搭载 AllGatherV）、TV3 sparse dispatch 探索（未成功）。TD4 最终状态：质量接近 G 但慢 24%。

## 2026-05-08

- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.21-team_sparse_kernel_tv4_tv5.md`
  - 作用：TEAM sparse kernel 多路径探索。TV4（extract→compact kernel→scatter, +1.2%）、TV5（topk_ids skip, +0.66%，当前最优）、TV4m（mapped Triton kernel, -16% 理论, crash 待调试）。nsys pipeline 分析证明 CPU launch overhead 是 monkey-patch 瓶颈。CUDA event 实验证明 TV4 kernel phase 比 G 快 40%。
- 路径：`/tmp/nsys_team/g_fresh_kern.csv`, `tv4_fresh_kern.csv`, `tv4_opt_kern.csv`, `tv5_kern.csv`
  - 作用：nsys kernel summary CSV 数据（G / TV4 / TV5 三方对比）

## 2026-05-09

- 路径：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.22-tv4m_crash_fix_and_tv6_compact_dispatch.md`
  - 作用：TV4m crash 修复（5 bug，root cause=N 维度 indexing 错误）+ TV6 compact dispatch/combine 完整实现（5 版本迭代）+ nsys 三方 profiling + CPU overhead 优化（sync 消除 + in-place cache）
- 路径：`codex_coding/src/test_triton_mapped_kernel.py`
  - 作用：TV4m Triton kernel 正确性 standalone test（合成 + 真实数据 dump）。用于隔离 kernel bug vs 集成 bug
- 路径：`codex_coding/src/test_compact_collective.py`
  - 作用：NCCL variable-size all_gatherv/reduce_scatterv 可行性测试（8 个测试）。验证 pynccl 支持动态 sizes
- 路径：`/tmp/nsys_team/g_v6cmp.sqlite`, `tv4m_v6cmp.sqlite`, `tv6d_v6cmp.sqlite`
  - 作用：G/TV4m/TV6 三方 nsys profiling sqlite 数据。含 kernel time、launch count、gap 分布分析

## 2026-05-10 论文实验文件

### 数据采集脚本
- `codex_coding/src/collect_routing_stability.py` — Fig.3 routing stability 数据采集（K=8 native, Jaccard + CDF）
- `codex_coding/src/collect_moe_output_stability.py` — Fig.4 MoE output stability 数据采集（Decoded vs MASK cos_sim）
- `codex_coding/src/collect_cache_staleness.py` — Fig.5 cache staleness 数据采集（per-step decay, baseline at t_decode+1）

### 绘图脚本
- `codex_coding/src/plt/fig3.py` — Jaccard heatmap + routing CDF（紧凑版 5.0×4.0）
- `codex_coding/src/plt/fig4.py` — Decoded vs MASK boxplot（1k 采样 + scatter）
- `codex_coding/src/plt/fig5_2.py` — Cache staleness per-step 折线图（quality threshold + M=5 标注）

### 结果数据
- `codex_coding/results/plt/routing_stability_data.json` — Fig.3 数据
- `codex_coding/results/plt/moe_output_stability_data.json` — Fig.4 数据（209.6 MB，含所有原始 cos_sim）
- `codex_coding/results/plt/cache_staleness_data.json` — Fig.5 数据（3 数据集 × 3 层 gap/step 级统计）

### 图片
- `codex_coding/results/plt/fig3_routing_stability_compact.png` — Fig.3 最终版
- `codex_coding/results/plt/fig4_moe_output_stability.png` — Fig.4 最终版
- `codex_coding/results/plt/fig5_2_cache_quality_per_step.png` — Fig.5 最终版

### TV6 Patch 模块
- `codex_coding/src/tv6_patch.py` — TV6 独立 patch（import bench 脚本函数，API: apply_tv6(model, decoder, vllm_cfg)）
- `codex_coding/src/bench_tv6_throughput.py` — TV6 throughput A/B benchmark（支持 --prompt-source gsm8k）
- `codex_coding/src/eval_tv6_quality.py` — lm-eval 质量评估（有 TP 拓扑兼容问题，备用）

### 过程文档
- `code_building/process_docs/v0.1-init-project/v0.1.15.23-paper_experiment_motivation_figures_and_tv6_patch.md` — 本轮完整实验记录

## 2026-05-11 Baseline Bench + 跨框架对比

| 文件路径 | 用途 |
|---------|------|
| `codex_coding/src/bench_baseline_dinfer.py` | Baseline dInfer throughput benchmark（TP=4, sdpa, prefix cache / no-cache） |
| `codex_coding/src/bench_sglang_dllm.py` | SGLang dLLM benchmark（sgl.Engine, LowConfidence, flashinfer） |
| `codex_coding/src/_sglang_bench_config.yaml` | SGLang threshold 配置文件 |
| `codex_coding/src/tv6_patch.py` | TV6 patch 模块（本轮加 DP=1 兼容性修复） |
| `lib_cite/sglang/.../low_confidence.py` | SGLang LowConfidence 算法（本轮加 _FWD_COUNTER） |
| `code_building/process_docs/v0.1-init-project/v0.1.15.24-paper_experiment_baseline_bench_and_cross_framework_comparison.md` | 本轮完整过程文档 |
