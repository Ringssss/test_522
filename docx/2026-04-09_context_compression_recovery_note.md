补充更新（2026-04-09 恢复后复验）：

- 下文记录的“正式 dInfer 已跑通”仍然是成立的历史事实，但已经不能直接等同于“当前 live environment 仍可立即复现”。
- 当前 live 环境复验结果是：
  - `torch` 现在读到的是 `2.9.1+cu128`
  - `sglang` 单独导入正常
  - 裸 `import dinfer` 会以 `std::bad_alloc` 退出
  - 缩短参数重跑 `/home/wuhang/wuhang/dllm_wh/codex_coding/src/run_dinfer_llada2_mini.py` 也会以 `std::bad_alloc` 退出
- 恢复后在讨论下一步前，请先补读：
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/2026-04-09_post_recovery_revalidation.md`
  - `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.6-post_recovery_revalidation_and_env_drift.md`

【
上一轮信息：

1. 当前项目仍然遵循的开发规范文件是：

- /home/wuhang/wuhang/dllm_wh/docx/next_step.txt

关键规范总结：

- 当前阶段代号：v0.1-init-project
- 当前项目根目录：/home/wuhang/wuhang/dllm_wh
- 代码注释如果新增，必须用英文
- 测试脚本统一放到 /home/wuhang/wuhang/dllm_wh/codex_coding/src
- 测试结果统一放到 /home/wuhang/wuhang/dllm_wh/codex_coding/results
- 重要文档统一放到 /home/wuhang/wuhang/dllm_wh/docx
- 过程文档统一放到 /home/wuhang/wuhang/dllm_wh/code_building/process_docs
- 每完成一个有效进展，都要同步到 /home/wuhang/wuhang/dllm_wh/code_building/progress_diff_summary.md
- 同步时必须带上【本轮命令】
- 形成关键归档文件时，要同步到 /home/wuhang/wuhang/dllm_wh/code_building/key_files_index.md
- 形成关键结论时，要同步到 /home/wuhang/wuhang/dllm_wh/code_building/key_conclusion.md
- 归档到 `docx/` 前，要按 /home/wuhang/wuhang/dllm_wh/docx/00_docx_storage_and_lookup_guide.md 判断落位
- 如果感到盲目，就先搜代码和文档，不要硬猜

2. 前几轮已经完成的核心工作：

- 已阅读并整理本地 `lib_cite/dInfer` 与 `lib_cite/sglang` 的 diffusion LLM 部分
- 已写出系统性技术综述：
  - /home/wuhang/wuhang/dllm_wh/docx/articles/2026-04-09_diffusion_llm_inference_and_acceleration_landscape.md
- 已形成最小自包含 demo 脚本：
  - /home/wuhang/wuhang/dllm_wh/codex_coding/src/dinfer_dllm_demo.py
- 已完成正式 dInfer 路线的真实实验脚本：
  - /home/wuhang/wuhang/dllm_wh/codex_coding/src/run_dinfer_llada2_mini.py

3. 刚刚完成的这一轮，最关键的事实是正式 dInfer 路线已经跑通：

- 真实使用的模型路径：
  - /home/wuhang/models/LLaDA2.0-mini
- 正式实验不是 fallback demo，而是 dInfer 官方 benchmark 风格路径：
  - `dinfer.model.LLaDA2MoeModelLM`
  - `ThresholdParallelDecoder`
  - `BlockDiffusionLLM`
  - `KVCacheFactory('prefix')`
- 当前正式实验脚本路径：
  - /home/wuhang/wuhang/dllm_wh/codex_coding/src/run_dinfer_llada2_mini.py

4. 这轮正式实验的真实结果已经保存：

- 指标 JSON：
  - /home/wuhang/wuhang/dllm_wh/codex_coding/results/dinfer_llada2_mini_metrics.json
- 结果摘要：
  - /home/wuhang/wuhang/dllm_wh/codex_coding/results/2026-04-09_dinfer_llada2_mini_formal_run.md

关键指标：

- model_path = /home/wuhang/models/LLaDA2.0-mini
- device = cuda:0
- tp_size = 1
- use_bd = true
- cache = prefix
- threshold = 0.95
- prompt_tokens = 55
- generated_tokens = 73
- num_forwards = 47
- load_time_sec = 8.9566
- generation_time_sec = 1.9186
- throughput = 38.05 tokens/s

5. 本轮得到的环境事实：

- `vllm==0.10.2` 已安装
- `sglang` 已安装，但当前是 editable 状态：
  - site-packages 记录显示：
    - Editable project location: /home/wuhang/wuhang/dllm_wh/lib_cite/sglang/python
- `sglang-kernel==0.4.1` 已安装
- `torch==2.8.0+cu128`
- `transformers==5.3.0`

6. 当前仍然不能丢失的关键补丁与现实约束：

- 为了让当前环境里的正式 dInfer 路线跑通，我对以下文件做了局部兼容修复：
  - /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py

这些兼容修复包括：

- `is_torch_fx_available` 缺失时的回退实现
- `flash_attn` 导入失败时的安全降级
- `rope_type=default` 在当前 `transformers` 版本下的默认 RoPE 参数计算回退

这些 patch 目前不要误删，因为它们直接支撑当前环境的正式实验成功。

7. 当前已验证和未完全规范化的地方要区分清楚：

已验证事实：

- 正式 dInfer 路线能真实加载 LLaDA2.0-mini 并完成一次生成
- demo 脚本和正式脚本都可用
- 当前环境已经足够继续做 benchmark 和参数 sweep

尚未完全规范化的地方：

- `sglang` 当前不是纯 wheel 安装，而是 editable 指向本地源码
- `dinfer` 与当前 `transformers` / `flash_attn` 版本之间仍有兼容补丁
- 裸 `import dinfer` 在某些新进程中出现过一次 `std::bad_alloc` 异常退出
- `vllm` 安装后存在一个 pip 级别冲突提示：
  - `outlines 0.1.11` 要求 `outlines_core==0.1.26`
  - 但 `vllm` 安装后带来了 `outlines_core 0.2.11`
- 这些问题当前不阻塞正式实验，但如果要长期稳定维护环境，后面需要专门清理

8. 下一轮不要重复做的事：

- 不要重新从头搭环境
- 不要重复安装 `vllm`、`sglang`、`sglang-kernel`，除非明确要做环境归一化
- 不要重新做“什么是 diffusion LLM”的基础阅读
- 不要重复写 fallback demo 作为主要实验入口
- 不要误删 `modeling_llada2_moe.py` 里的兼容 patch
- 不要把当前环境误当成“已经完全规范化的纯净环境”

9. 当前最值得保留的恢复记忆句：

- 现在最重要的事实不是“环境快搭好了”，而是“正式 dInfer 已经在 `/home/wuhang/models/LLaDA2.0-mini` 上跑通”；下一步不该继续折腾环境基础设施，而该在“继续做 benchmark sweep”和“是否先清理环境/补丁”之间与用户确认优先级。

这一轮：
请你先回顾一下上一轮我们做了什么，然后，这一轮你的任务如下：

1. 先重新阅读这些文档：

- /home/wuhang/wuhang/dllm_wh/docx/next_step.txt
- /home/wuhang/wuhang/dllm_wh/docx/00_docx_storage_and_lookup_guide.md
- /home/wuhang/wuhang/dllm_wh/docx/2026-04-09_context_compression_recovery_note.md
- /home/wuhang/wuhang/dllm_wh/docx/context_index/01_stage_summary.md
- /home/wuhang/wuhang/dllm_wh/docx/context_index/02_active_threads.md
- /home/wuhang/wuhang/dllm_wh/docx/context_index/03_current_required_process_docs.md
- /home/wuhang/wuhang/dllm_wh/docx/articles/2026-04-09_diffusion_llm_inference_and_acceleration_landscape.md

2. 然后优先阅读这些过程文档：

- /home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.2-diffusion_llm_acceleration_landscape_archive.md
- /home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.3-dinfer_demo_script_archive.md
- /home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.4-dinfer_llada2_mini_formal_run.md
- /home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.5-context_compression_recovery_handoff.md

3. 然后回看这些最关键的代码和结果文件：

- /home/wuhang/wuhang/dllm_wh/codex_coding/src/run_dinfer_llada2_mini.py
- /home/wuhang/wuhang/dllm_wh/codex_coding/src/dinfer_dllm_demo.py
- /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py
- /home/wuhang/wuhang/dllm_wh/codex_coding/results/dinfer_llada2_mini_metrics.json
- /home/wuhang/wuhang/dllm_wh/codex_coding/results/2026-04-09_dinfer_llada2_mini_formal_run.md

4. 重点恢复这几个认知：

- 当前项目已经不是“只做理论阅读”，而是已经进入 dInfer/LLaDA2.0-mini 的真实运行与验证阶段
- 正式 dInfer 路线已经跑通，不只是 fallback demo 能跑
- 当前 formal run 使用的是：
  - `LLaDA2MoeModelLM`
  - `ThresholdParallelDecoder`
  - `BlockDiffusionLLM`
  - `prefix cache`
  - `tp=1`
- 当前最重要的结果文件是：
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/dinfer_llada2_mini_metrics.json`
- 当前最重要的技术综述文件是：
  - `/home/wuhang/wuhang/dllm_wh/docx/articles/2026-04-09_diffusion_llm_inference_and_acceleration_landscape.md`
- 当前环境已经足够继续做实验，但还不是完全“无 patch、无影子路径、无冲突”的规范终态
- 当前不能丢的 patch 在：
  - `/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py`

5. 下一步不要重复做已经完成的事：

- 不要重复从头安装 `vllm`、`sglang`、`sglang-kernel`
- 不要重复做 `LLaDA2.0-mini` 的首次正式 bring-up
- 不要把主要精力重新放回“什么是 diffusion LLM”的基础介绍
- 不要在没有明确目标的情况下反复改环境版本

6. 下一步如果继续，最合理的方向只有两类，先和用户确认优先级再动手：

- 方向 A：继续实验
  - 扩展 benchmark 矩阵
  - 比较 `prefix` / `dual`
  - 比较 `use_bd=True` / `False`
  - sweep `threshold`
  - 提高 `gen_length`
  - 评估 `torch.compile`

- 方向 B：环境归一化
  - 判断是否要去掉 `modeling_llada2_moe.py` 中的兼容 patch
  - 判断是否要把 editable `sglang` 变成固定 wheel 安装
  - 判断是否要修掉 `outlines/outlines_core` 的 pip 冲突
  - 判断为何裸 `import dinfer` 在某些进程里会触发 `std::bad_alloc`

7. 但在那之前，先和用户确认：

- 下一轮是优先做 benchmark sweep，还是优先清理环境和补丁
- 如果做 benchmark，优先扫哪几个变量：
  - `cache`
  - `use_bd`
  - `threshold`
  - `gen_length`
  - `torch.compile`
- 如果做环境清理，是否允许在不影响当前可运行链路的前提下逐步回退 patch 并重新验证

8. 最重要的恢复记忆句：

- 当前最关键的状态是：`/home/wuhang/models/LLaDA2.0-mini` 已经通过正式 dInfer 路线跑通；下一步应该在“继续做性能实验”与“先把环境规范化”之间让用户拍板，而不是自己重新发散。
】
