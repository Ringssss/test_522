# Progress Diff Summary

## 2026-03-20

- Established the workspace development consensus before implementation.
- Confirmed `/home/wuhang/wuhang/linear_wh/triton` as the actual Git repository root.
- Standardized future durable records on `code_building/` and `codex_coding/`.
- Preserved historical empty directories `code-building/` and `codex-coding/` without renaming them.
- Initialized `code_building/process_docs/`, `codex_coding/src/`, and `codex_coding/results/`.
- Added the process record `code_building/process_docs/2026-03-20_development_consensus.md`.
- Read the three planning notes under `docx/cites/` and extracted the project definition.
- Added the process record `code_building/process_docs/2026-03-20_project_scope_from_citations.md`.

## 2026-03-27

- Archived the current `LinearLayout` investigation into `/home/wuhang/wuhang/linear_wh/docx/articles/2026-03-27_linear_layout_and_subgraph_optimization.md`.
- Created `/home/wuhang/wuhang/linear_wh/docx/articles/` as the standard path for new technical article archives.
- Created `/home/wuhang/wuhang/linear_wh/docx/context_index/` and initialized the document registry, stage summary, active threads, and current required process docs.
- Added the process record `/home/wuhang/wuhang/linear_wh/code_building/process_docs/2026-03-27_linearlayout_article_archive.md`.
- Added `/home/wuhang/wuhang/linear_wh/docx/articles/2026-03-27_qwen35_hybrid_attention_brainstorm.md` to archive the current Qwen3.5 hybrid-attention brainstorming.
- Added the process record `/home/wuhang/wuhang/linear_wh/code_building/process_docs/2026-03-27_qwen35_hybrid_attention_brainstorm_archive.md`.
- Updated the context index files to reflect the current focus on Qwen3.5 model initialization, hybrid attention, and benchmark-result persistence discipline.
- Added `/home/wuhang/wuhang/linear_wh/docx/articles/2026-03-27_linear_state_prefix_cache_brainstorm.md` to archive the current analysis of radix cache, linear state, and hybrid prefix checkpoint ideas.
- Added `/home/wuhang/wuhang/linear_wh/codex_coding/results/2026-03-27_qwen35_online_baseline_summary.md` to preserve the already observed Qwen3.5 serving baseline summary.
- Added `/home/wuhang/wuhang/linear_wh/docx/2026-03-27_context_compression_recovery_note.md` as a restart-safe recovery note for context compression.
- Added the process record `/home/wuhang/wuhang/linear_wh/code_building/process_docs/2026-03-27_linear_state_prefix_cache_brainstorm_archive.md`.

## 2026-04-09

- 新增 `/home/wuhang/wuhang/dllm_wh/docx/00_docx_storage_and_lookup_guide.md`，作为 `docx/` 根目录下的稳定文档地图与归档查找指南。
- 明确了 `docx/` 根目录、`cites/`、`articles/`、`context_index/`、`plans/` 的职责边界，以及新文档的建议落位规则。
- 在指南中补充了查找顺序，说明开发规范、阶段状态、活跃线程、技术专题和恢复材料各自应该先读哪类文件。
- 在指南中补充了路径风险提示，明确指出历史文档中的 `linear_wh` 路径不能直接当作当前 live path 使用。
- 更新 `/home/wuhang/wuhang/dllm_wh/docx/context_index/00_document_registry.md`，把新指南登记为高优先级 guide 文档。
- 新增过程归档 `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.1-docx_storage_and_lookup_guide_archive.md`。
- 更新了 `code_building/key_files_index.md` 和 `code_building/key_conclusion.md`，同步本次关键文件和结论。

### 本轮命令

- `find /home/wuhang/wuhang/dllm_wh/docx -maxdepth 2 -type d | sort`
- `find /home/wuhang/wuhang/dllm_wh/docx -maxdepth 2 -type f | sort`
- `sed -n '1,220p' /home/wuhang/wuhang/dllm_wh/docx/next_step.txt`
- `sed -n '1,220p' /home/wuhang/wuhang/dllm_wh/docx/2026-03-27_context_compression_recovery_note.md`
- `sed -n '1,220p' /home/wuhang/wuhang/dllm_wh/docx/context_index/00_document_registry.md`
- `sed -n '1,220p' /home/wuhang/wuhang/dllm_wh/docx/context_index/01_stage_summary.md`
- `sed -n '1,220p' /home/wuhang/wuhang/dllm_wh/docx/context_index/02_active_threads.md`
- `sed -n '1,220p' /home/wuhang/wuhang/dllm_wh/docx/context_index/03_current_required_process_docs.md`

- 新增 `/home/wuhang/wuhang/dllm_wh/docx/articles/2026-04-09_diffusion_llm_inference_and_acceleration_landscape.md`，系统梳理 diffusion LLM 的定义、推理流程、与 AR 的差异，以及当前 text dLLM 加速技术分层。
- 确认 `dInfer` 的主线是 block-wise diffusion + 并行阈值解码 + 多种 cache 近似与运行时优化，`SGLang` 的主线是 `srt.dllm` 子系统下的原生算法接入与专用调度。
- 归纳出当前最值得优先优化的方向，不是只做 kernel，而是联动优化“每轮解多少 token”和“每轮刷新多少 cache”。
- 新增过程归档 `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.2-diffusion_llm_acceleration_landscape_archive.md`。
- 更新了 `docx/context_index/00_document_registry.md`、`code_building/key_files_index.md` 和 `code_building/key_conclusion.md`，同步本轮新增技术文章与关键判断。

### 本轮命令

- `sed -n '1,260p' /home/wuhang/.codex/skills/planning-with-files/SKILL.md`
- `sed -n '1,240p' /home/wuhang/.codex/suss-skills/building-rules/SKILL.md`
- `rg -n -i "diffusion|dllm|denois|mask|mdlm|ladd|speculative|draft|verify|tree" /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer /home/wuhang/wuhang/dllm_wh/lib_cite/sglang`
- `sed -n '1,240p' /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/README.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/decoding/parallel_strategy.py`
- `sed -n '1,560p' /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/decoding/generate_uniform.py`
- `sed -n '1,220p' /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/decoding/diffusion_runner.py`
- `sed -n '1,420p' /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/decoding/generate_fastdllm.py`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/decoding/generate_cache.py`
- `sed -n '1,180p' /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/into_sglang/algorithm.py`
- `sed -n '850,980p' /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada.py`
- `sed -n '228,312p' /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe_sglang.py`
- `sed -n '820,980p' /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/tools/modeling_fused_lladamoe.py`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/lib_cite/sglang/docs/supported_models/text_generation/diffusion_language_models.md`
- `sed -n '1,240p' /home/wuhang/wuhang/dllm_wh/lib_cite/sglang/python/sglang/srt/dllm/config.py`
- `sed -n '1,240p' /home/wuhang/wuhang/dllm_wh/lib_cite/sglang/python/sglang/srt/dllm/algorithm/low_confidence.py`
- `sed -n '1,280p' /home/wuhang/wuhang/dllm_wh/lib_cite/sglang/python/sglang/srt/dllm/algorithm/joint_threshold.py`
- `sed -n '1,120p' /home/wuhang/wuhang/dllm_wh/lib_cite/sglang/python/sglang/srt/dllm/mixin/scheduler.py`
- `sed -n '360,470p' /home/wuhang/wuhang/dllm_wh/lib_cite/sglang/python/sglang/srt/managers/tp_worker.py`
- `sed -n '391,500p' /home/wuhang/wuhang/dllm_wh/lib_cite/sglang/python/sglang/srt/managers/schedule_policy.py`

- 新增 `/home/wuhang/wuhang/dllm_wh/codex_coding/src/dinfer_dllm_demo.py`，提供一个最小自包含的 dInfer 风格 diffusion LLM demo 脚本，可展示 block-wise threshold 去噪过程并输出测速指标。
- 该脚本只依赖 `torch + transformers`，避免了当前环境中 `import dinfer` 被缺失 `vllm` 依赖阻塞的问题。
- 新增 `/home/wuhang/wuhang/dllm_wh/codex_coding/results/2026-04-09_dinfer_dllm_demo_smoke_test.md`，保存语法检查、dry-run 和 dummy model smoke test 结果。
- 新增过程归档 `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.3-dinfer_demo_script_archive.md`。
- 更新了 `code_building/key_files_index.md` 和 `code_building/key_conclusion.md`，同步脚本位置和当前环境约束。

### 本轮命令

- `python - <<'PY' ... importlib.util.find_spec('dinfer') ... PY`
- `python - <<'PY' ... import dinfer ... PY`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/__init__.py`
- `sed -n '1,220p' /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/tests/test_llada.py`
- `rg -n "AutoModelForCausalLM|from_pretrained\\(|AutoTokenizer|BlockWiseDiffusionLLM\\(|ThresholdParallelDecoder\\(" /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/tests /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/README.md`
- `python -m py_compile /home/wuhang/wuhang/dllm_wh/codex_coding/src/dinfer_dllm_demo.py`
- `python /home/wuhang/wuhang/dllm_wh/codex_coding/src/dinfer_dllm_demo.py --model-path GSAI-ML/LLaDA-1.5 --dry-run`
- `python - <<'PY' ... dummy model smoke test ... PY`
- `find /mnt/infra/myx/models -maxdepth 2 -type d 2>/dev/null | rg 'LLaDA|llada|SDAR|sdar'`
- `find $HOME/.cache/huggingface -maxdepth 3 -type d 2>/dev/null | rg 'LLaDA|llada|SDAR|sdar'`

- 补装 `vllm==0.10.2`，解决正式 `dinfer` 顶层 import 对 `vllm` 的依赖。
- 确认本机存在 `/home/wuhang/models/LLaDA2.0-mini`，并用正式 dInfer benchmark 路线而不是 fallback demo 路线跑通真实实验。
- 新增正式实验脚本 `/home/wuhang/wuhang/dllm_wh/codex_coding/src/run_dinfer_llada2_mini.py`，采用 `LLaDA2MoeModelLM + ThresholdParallelDecoder + BlockDiffusionLLM`。
- 为了兼容当前环境，给 `modeling_llada2_moe.py` 增加了三个局部兼容修复：
  - `is_torch_fx_available` 缺失回退；
  - `flash_attn` 导入失败降级；
  - `rope_type=default` 的默认 RoPE 计算回退。
- 在 `cuda:0`、`tp=1`、`prefix cache`、`block diffusion`、`threshold=0.95`、`gen_length=64` 配置下正式跑通 `LLaDA2.0-mini`：
  - load time `8.9566 s`
  - generation time `1.9186 s`
  - throughput `38.05 tokens/s`
  - measured forwards `47`
- 新增结果文件：
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/dinfer_llada2_mini_metrics.json`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/2026-04-09_dinfer_llada2_mini_formal_run.md`
- 新增过程归档：
  - `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.4-dinfer_llada2_mini_formal_run.md`
- 新增 `/home/wuhang/wuhang/dllm_wh/docx/2026-04-09_context_compression_recovery_note.md`，作为当前项目新的上下文压缩恢复说明。
- 刷新了当前 `dllm_wh` 的 `context_index/00~03`，把旧的 `linear_wh` 现场替换成当前 diffusion LLM / dInfer / LLaDA2.0-mini 的真实状态。
- 新增过程归档：
  - `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.5-context_compression_recovery_handoff.md`
- 在恢复说明中明确了当前最重要的状态：
  - 正式 dInfer 已经跑通
  - 当前环境足够继续实验
  - 但还不是完全规范化终态
  - 下一步应先让用户在“benchmark sweep”和“环境规范化”之间确认优先级

### 本轮命令

- `sed -n '1,260p' /home/wuhang/.codex/suss-skills/recovery-handoff/SKILL.md`
- `sed -n '1,240p' /home/wuhang/.codex/suss-skills/building-rules/SKILL.md`
- `sed -n '1,240p' /home/wuhang/wuhang/dllm_wh/docx/next_step.txt`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/docx/context_index/00_document_registry.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/docx/context_index/01_stage_summary.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/docx/context_index/02_active_threads.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/docx/context_index/03_current_required_process_docs.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/docx/articles/2026-04-09_diffusion_llm_inference_and_acceleration_landscape.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.2-diffusion_llm_acceleration_landscape_archive.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.3-dinfer_demo_script_archive.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.4-dinfer_llada2_mini_formal_run.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/code_building/progress_diff_summary.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/code_building/key_files_index.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/code_building/key_conclusion.md`

### 本轮命令

- `pip install vllm==0.10.2`
- `PYTHONPATH=/home/wuhang/wuhang/dllm_wh/lib_cite/sglang/python:$PYTHONPATH python - <<'PY' ... import dinfer ... PY`
- `PYTHONPATH=/home/wuhang/wuhang/dllm_wh/lib_cite/sglang/python:$PYTHONPATH python - <<'PY' ... from dinfer.model import LLaDA2MoeModelLM ... PY`
- `python - <<'PY' ... from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS ... PY`
- `python - <<'PY' ... read /home/wuhang/models/LLaDA2.0-mini/config.json ... PY`
- `python -m py_compile /home/wuhang/wuhang/dllm_wh/codex_coding/src/run_dinfer_llada2_mini.py`
- `PYTHONPATH=/home/wuhang/wuhang/dllm_wh/lib_cite/sglang/python:$PYTHONPATH python /home/wuhang/wuhang/dllm_wh/codex_coding/src/run_dinfer_llada2_mini.py --model-path /home/wuhang/models/LLaDA2.0-mini --gen-length 64 --block-length 32 --threshold 0.95 --cache prefix --warmup-runs 1 --metrics-output /home/wuhang/wuhang/dllm_wh/codex_coding/results/dinfer_llada2_mini_metrics.json`

- 恢复后重新阅读了当前 `docx`、`context_index`、过程归档、正式脚本和结果文件，确认此前的 formal run 是历史真实成功，而不是凭空假设。
- 新增 `/home/wuhang/wuhang/dllm_wh/codex_coding/results/2026-04-09_post_recovery_revalidation.md`，记录当前 live 环境的复验结果。
- 新增过程归档 `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.6-post_recovery_revalidation_and_env_drift.md`。
- 当前复验确认：
  - `sglang` 单独导入正常
  - 裸 `import dinfer` 会以 `std::bad_alloc` 退出
  - 缩短参数重跑 `run_dinfer_llada2_mini.py` 也会以 `std::bad_alloc` 退出
- 当前环境与历史成功归档相比至少有一个明确漂移：
  - 历史记录中的 `torch` 为 `2.8.0+cu128`
  - 当前 live 环境读到的 `torch` 为 `2.9.1+cu128`
- 之前提到的 `outlines/outlines_core` 冲突在当前环境快照下已不再成立：
  - 当前 `outlines_core` 为 `0.1.26`
- 已更新：
  - `docx/context_index/00_document_registry.md`
  - `docx/context_index/01_stage_summary.md`
  - `docx/context_index/02_active_threads.md`
  - `docx/context_index/03_current_required_process_docs.md`
  - `docx/2026-04-09_context_compression_recovery_note.md`
  - `code_building/key_files_index.md`
  - `code_building/key_conclusion.md`
- 当前项目状态应修正为：
  - 归档中的正式 dInfer 成功运行仍然有效
  - 但当前 live 环境已经不能直接复验
  - 下一步更合理的是先做环境回归定位，再谈 benchmark sweep

### 本轮命令

- `sed -n '1,220p' /home/wuhang/wuhang/dllm_wh/docx/next_step.txt`
- `sed -n '1,240p' /home/wuhang/wuhang/dllm_wh/docx/00_docx_storage_and_lookup_guide.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/docx/2026-04-09_context_compression_recovery_note.md`
- `sed -n '1,220p' /home/wuhang/wuhang/dllm_wh/docx/context_index/01_stage_summary.md`
- `sed -n '1,220p' /home/wuhang/wuhang/dllm_wh/docx/context_index/02_active_threads.md`
- `sed -n '1,240p' /home/wuhang/wuhang/dllm_wh/docx/context_index/03_current_required_process_docs.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/docx/articles/2026-04-09_diffusion_llm_inference_and_acceleration_landscape.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.2-diffusion_llm_acceleration_landscape_archive.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.3-dinfer_demo_script_archive.md`
- `sed -n '1,300p' /home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.4-dinfer_llada2_mini_formal_run.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.5-context_compression_recovery_handoff.md`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/codex_coding/src/run_dinfer_llada2_mini.py`
- `sed -n '1,220p' /home/wuhang/wuhang/dllm_wh/codex_coding/src/dinfer_dllm_demo.py`
- `sed -n '1,280p' /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py`
- `cat /home/wuhang/wuhang/dllm_wh/codex_coding/results/dinfer_llada2_mini_metrics.json`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/codex_coding/results/2026-04-09_dinfer_llada2_mini_formal_run.md`
- `PYTHONPATH=/home/wuhang/wuhang/dllm_wh/lib_cite/sglang/python:$PYTHONPATH python - <<'PY' ... import torch, transformers; import dinfer, sglang ... PY`
- `python - <<'PY' ... importlib.metadata versions for vllm/sglang/sglang-kernel/outlines/outlines_core ... PY`
- `PYTHONPATH=/home/wuhang/wuhang/dllm_wh/lib_cite/sglang/python:$PYTHONPATH python - <<'PY' ... import sglang ... PY`
- `PYTHONPATH=/home/wuhang/wuhang/dllm_wh/lib_cite/sglang/python:$PYTHONPATH python - <<'PY' ... import dinfer ... PY`
- `python - <<'PY' ... print torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) ... PY`
- `PYTHONPATH=/home/wuhang/wuhang/dllm_wh/lib_cite/sglang/python:$PYTHONPATH python /home/wuhang/wuhang/dllm_wh/codex_coding/src/run_dinfer_llada2_mini.py --model-path /home/wuhang/models/LLaDA2.0-mini --gen-length 16 --block-length 16 --threshold 0.95 --cache prefix --warmup-runs 0 --metrics-output /home/wuhang/wuhang/dllm_wh/codex_coding/results/_tmp_reverify_dinfer_llada2_mini_metrics.json`

- 诊断了 v0.1.6 发现的 `import dinfer` -> `std::bad_alloc` 崩溃的根因：
  - 确认与 dinfer patch、sglang、flash_attn 均无关
  - 确认崩溃发生在 `vllm.distributed` -> `vllm._C.abi3.so` 加载阶段
  - 根因：vllm 0.10.2 的编译扩展与 torch 2.9.1 的 C++ ABI 不兼容
- 回退 torch 到 2.8.0+cu128（归档成功时的版本）
- 复验正式脚本 `run_dinfer_llada2_mini.py`：
  - formal path 完全恢复
  - 结果与历史归档一致：73 tokens / 47 forwards / 37.77 tokens/s
  - 生成文本完全一致
- 新增复验指标：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/dinfer_llada2_mini_revalidation_metrics.json`
- 新增过程归档：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.7-env_regression_diagnosis_and_torch_rollback.md`
- 当前项目状态回到"可以继续做 benchmark sweep"

### 本轮命令

- 恢复上下文后阅读了所有指定文档和代码文件
- `python3 -c "import torch; print(torch.__version__)"` (确认 2.9.1)
- `python3 -c "from vllm import distributed"` (确认崩溃)
- 逐步隔离：`vllm.distributed.utils` -> `vllm._C` (确认编译扩展 ABI 不兼容)
- `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128`
- `python3 -c "import dinfer; from dinfer.model import LLaDA2MoeModelLM"` (确认恢复)
- `python3 run_dinfer_llada2_mini.py --gen-length 64 --block-length 32 --threshold 0.95 --cache prefix --warmup-runs 1` (formal path 复验成功)

- 新增 sweep 脚本 `/home/wuhang/wuhang/dllm_wh/codex_coding/src/sweep_dinfer_llada2_mini.py`
- 在 `use_bd=True`, `cache=prefix`, `gen_length=128`, `block_length=32` 下扫描 threshold = 0.90 / 0.95 / 0.99
- 确认 `torch.compile` 在 torch 2.8.0 + LLaDA2MoE 上触发 InductorError，记录为已知限制
- 确定 best baseline：`threshold=0.90`, **70.6 tok/s**, 26.3 fwd/s, 51 forwards
- 关键发现：fwd/s 恒定 (~26)，性能差异完全由 forward 次数决定
- 新增结果文件：`/home/wuhang/wuhang/dllm_wh/codex_coding/results/sweep_dinfer_llada2_mini_results.json`
- 新增过程归档：`/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.8-dllm_performance_sweep_and_baseline.md`

### 本轮命令

- `python3 codex_coding/src/sweep_dinfer_llada2_mini.py`

- 新增全路径对比脚本 `bench_all_paths.py` 和长 prompt 对比脚本 `bench_long_prompt.py`
- 对比了 8 条推理路径（BlockDiffusionLLM prefix/attnmask、BlockWiseDiffusionLLM prefix/dual/none、IterSmooth、VicinityCache）
- 确认 `BlockDiffusionLLMAttnmask`（全重算、无 KV cache）是当前最快路径：
  - 短 prompt (55 tok): 76.8 tok/s
  - 长 prompt (662 tok): 75.2 tok/s
- 发现 prefix cache 在长 prompt 下 fwd/s 从 26.3 降到 19.7，cache 管理开销反而拖慢了 forward
- 发现 IterSmooth 和 VicinityCache 尚未适配 LLaDA2MoE（dtype / position 不兼容）
- 最终 baseline：BlockDiffusionLLMAttnmask, threshold=0.90, 75-77 tok/s
- 新增结果文件：
  - `codex_coding/results/all_paths_benchmark_results.json`
  - `codex_coding/results/long_prompt_benchmark_results.json`
- 新增过程归档：`code_building/process_docs/v0.1-init-project/v0.1.9-all_paths_benchmark_and_baseline.md`

### 本轮命令

- `python3 codex_coding/src/bench_all_paths.py`
- `python3 codex_coding/src/bench_long_prompt.py`

## 2026-04-10

- 修复了 IterSmooth 的 dtype bug：`modeling_llada2_moe.py:284`，`prob.to(self.W_e.dtype) @ self.W_e`
- 修复了 VicinityCache 的 position_ids / replace_position bug：`generate_uniform.py:704-707` 和 `795-801`
  - 根因：`DiffusionKVCacheManager.get_key_values` 对 prefix 类型返回 `(left_start, seq_len)` 而不是 `(left_start, right_end)`
  - 修复：覆盖 replace_position 为窗口范围，显式传入 position_ids
- 新增 benchmark 脚本 `codex_coding/src/bench_itersmooth_vicinity.py`，测试 6 条路径（BD-attnmask baseline + BW ref + IterSmooth×2 + VicinityCache + IterSmooth+Vicinity）
- **关键结论：IterSmooth 和 VicinityCache 在 BW 框架下均无法超过 BD-attnmask baseline**
  - BD-attnmask: 78.1 / 75.7 tok/s（短/长 prompt）
  - IterSmooth no cache 最好: 56.9 / 32.7 tok/s
  - IterSmooth 在长 prompt 下 forward 次数从 80 暴增到 102（soft embedding 对未来 mask block 是噪声注入）
- 系统梳理了 dInfer 中所有推理框架（BW / BD / BD-attnmask）和 cache 策略（prefix / dual / vicinity / none）的组合差异
  - BW 让模型看到未来 mask block → forward 次数多
  - BD 截断到当前 block → forward 次数少
  - BlockDiffusionPrefixCacheManager 下 prefix 和 dual 实际等价
  - BlockDiffusionIteration 每步全量重建 cache 是 BD+cache 的瓶颈
- 设计了 BD + IterSmooth 结合方案（尚未实现）：
  - 在 BD-attnmask 路径上用 `inputs_embeds` 代替 `input_ids`
  - 只对当前 block 的 mask 位置做 soft embedding，prompt/已完成 block 保持 discrete
  - `h2e` 计算量从 O(total_len) 降到 O(block_len)
  - 保留 BD 的 block-causal attention mask 和 threshold_decay

### 本轮命令

- 修改 `lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py:284`
- 修改 `lib_cite/dInfer/python/dinfer/decoding/generate_uniform.py:704-707, 795-801`
- `python3 codex_coding/src/bench_itersmooth_vicinity.py`（两次）

- 实现了 BD + IterSmooth 结合方案，新增三个类（不修改任何现有代码）：
  - `IterSmoothBlockDiffusionIteration`：在 BD 路径上用 inputs_embeds 代替 input_ids，只对当前 block 的 mask 位置做 soft embedding
  - `IterSmoothBlockDiffusionLLMAttnmask`：no-cache 版本
  - `IterSmoothBlockDiffusionLLMCache`：prefix cache 版本
- **短 prompt 上 BD+IterSmooth 成功超越 baseline**：
  - BD+IS high_w (cont_weight=0.5): **87.3-87.8 tok/s**，forward 从 48 降到 42，**+13% tok/s**
  - BD+IS default (cont_weight=0.3): 79.9 tok/s，forward 46，+2.8%
- **长 prompt 上 IterSmooth 反而增加 forward 次数**：
  - BD+IS high_w: 70.6-71.4 tok/s，forward 51，-6.6%
  - BD+IS default: 62.8 tok/s，forward 58，-17.3%
- **KV cache 在 BD 路径上是确定的负优化**（再次确认）：
  - cache 路径 fwd/s 从 ~26-27 降到 ~19-21
  - BD+IS+cache: 49-50 tok/s（长 prompt），比 baseline 慢 34%
- threshold_decay 是必要的，无 decay 时 forward 暴增到 116-121
- h2e 的额外开销可忽略：fwd/s 恒定 ~26-27，与 baseline 无差异
- 新增结果文件：`codex_coding/results/bd_itersmooth_benchmark_results.json`
- 新增 benchmark 脚本：`codex_coding/src/bench_bd_itersmooth.py`
- 新增过程归档：`code_building/process_docs/v0.1-init-project/v0.1.11-bd_itersmooth_combination.md`

### 本轮命令

- 修改 `generate_uniform.py`：新增 IterSmoothBlockDiffusionIteration、IterSmoothBlockDiffusionLLMAttnmask、IterSmoothBlockDiffusionLLMCache
- 修改 `dinfer/__init__.py`：导出新类
- 新建 `codex_coding/src/bench_bd_itersmooth.py`
- `python3 codex_coding/src/bench_bd_itersmooth.py`（两轮：no-cache + cache）

## 2026-04-12

- 完成了 cache 路径 fwd/s 低于 no-cache 的根因分析（代码追踪 + 理论推导）：
  - 识别出 block 边界的 F.pad/consolidate/rebuild 是主要非 forward 开销
  - 提出预分配 contiguous buffer + 消除 block 边界重建的优化方案
  - 分析了 Paged Attention 对 dLLM 的适用性：cache 管理思路可用，但 paged attention kernel 不适用（q_len=32 vs AR q_len=1，且 dLLM 需要 in-place rewrite 语义）
- 编写了 torch.profiler 算子级 profiling 脚本 `codex_coding/src/bench_profile_kernels.py`
- **核心新工作：dLLM MoE 推理行为分析**
  - 系统分析了 dLLM MoE 与 AR MoE 的三个核心差异：MASK routing 集中、跨迭代计算冗余、天然批量效应
  - 与 13 篇近两年 MoE 系统论文（EARTH, X-MoE, Diff-MoE, LAER-MoE, Klotski 等）做了交叉分析
  - 识别出 EARTH 的 result reuse + X-MoE 的 redundancy bypass 可以在 dLLM 的迭代结构下产生远比 AR 更强的版本
  - 补充了多卡 EP 视角：Insight B 在多卡下不仅省 compute 还省 all-to-all 通信
  - 确定了最有论文潜力的方向："Iteration-Aware Selective MoE for Diffusion LLM Inference"
- 新增技术文档归档：`docx/articles/2026-04-12_dllm_moe_insight_and_optimization_directions.md`
  - 包含三个 insight 的详细分析、多卡影响、与现有论文的差异化、论文故事线框架和三个待验证实验设计
- 更新了 `code_building/key_files_index.md`、`code_building/key_conclusion.md`

### 本轮命令

- 阅读 `history-chat.txt` 恢复上下文
- 阅读所有指定文档、过程文档、代码和结果文件
- 阅读 `docx/cites/moe_systems_survey.md`（13 篇 MoE 系统论文调研）
- 阅读 `modeling_llada2_moe.py` 中的 MoE 相关代码（LLaDA2MoeGate, LLaDA2MoeSparseMoeBlock, FusedMoE）
- 读取模型配置（num_experts=256, top_k=8, n_group=8, topk_group=4）
- 编写 `codex_coding/src/bench_profile_kernels.py`
- 新建 `docx/articles/2026-04-12_dllm_moe_insight_and_optimization_directions.md`

- **MoE routing 实证分析（4 组实验）**
  - 实验 1：batch=1 同质，no-cache 路径 — 验证三个 insight
  - 实验 2：batch=1 同质，cache 路径 — 对比路径无关性
  - 实验 3：batch=8/32 异质（8 个不同 prompt + temperature=0.7）— 验证 insight 在真实 serving 下是否成立
  - 结论：三个 insight 全部在所有设置下稳健成立，大 batch 下冗余率更高（51.5%）
- **Shared vs Routed 分解分析（实验 1+3）**
  - routed expert 和 shared expert 在多数层量级相当（ratio 0.72-1.68），跳过 routed 会丢失大量信息
  - shared-only 近似的 cosine sim 只有 0.57-0.95，远差于 v1 缓存（0.87-0.98）
  - 结论：shared-only 路径不可行，v1 缓存方式（完整 full output）是更好的近似基础
- **逐层 ablation + 周期性刷新（实验 2+4）**
  - 单层缓存：19 个 MoE 层中 18 个可安全缓存，唯一敏感层是 Layer 18
  - 周期性刷新：N=2 到 N=10 全部 exact match（前提：每步都做全量计算，只替换输出）
  - 关键发现：ablation 实验中 cache 始终是 fresh（每步都算了全量），所以才能 exact match
- **Stable Cache v1 实现与测试**
  - 新增 StableCacheBlockDiffusionIteration + StableCacheBlockDiffusionLLM
  - v1 结果：token match 21.1%，forward 49→87，比 baseline 慢 — 失败
  - 原因：误差跨 19 层串联累积，logits 偏移导致 decoder 置信度下降 → 更多 forward
- **Stable Cache v2（Layer 18 防火墙 + 刷新）实现与测试**
  - v2a：缓存 routed 部分 + fresh shared → 仍然失败（fresh shared + stale routed 不一致）
  - v2b：缓存完整 full output → 仍然失败（跳过计算后 cache 越来越旧，不像 ablation 中每步都刷新）
  - v2c：REFRESH_INTERVAL=2 → 仍然失败（即使只隔 1 步，跳过的那步 cache 也导致偏移）
- **多层组合 ablation（关键实验）**
  - 单独安全的层：L1-3 ✓, L6-13 ✓, L19 ✓
  - 任意组合都失败：L1-3+L19 ✗, L6-13+L19 ✗, L1-3+L6-13 ✗, L1-3+L6-13+L19 ✗
  - **根本结论：单层 ablation 的 "safe" 不能组合。误差在层间传播太快，多层同时缓存超过容错边界**
- **总体结论**
  - "跳过 stable 位置 MoE 计算"在当前架构下不可行（误差累积太快）
  - ablation 中的 "exact match" 是因为每步都做了全量计算（cache 始终 fresh），不代表跳过计算安全
  - 三个 MoE insight（MASK routing 集中、跨迭代冗余、天然批量）作为 observation 成立，但直接转化为 "跳过计算" 的优化路径有根本性困难
  - 需要转向其他优化方向（参考 EARTH 的分层/近似思路，或转向减少 forward 次数的算法优化）
- 新增/修改的文件：
  - `codex_coding/src/bench_moe_routing_analysis.py` — MoE routing profiling（batch=1, cache 路径）
  - `codex_coding/src/bench_moe_routing_hetero.py` — 异质 batch + 随机采样 routing profiling
  - `codex_coding/src/bench_moe_decomp.py` — shared/routed 分解分析
  - `codex_coding/src/bench_moe_layer_ablation.py` — 逐层 ablation + 周期性刷新
  - `codex_coding/src/bench_moe_multilayer_ablation.py` — 多层组合 ablation
  - `codex_coding/src/bench_stable_cache.py` — stable cache benchmark
  - `codex_coding/results/moe_routing_analysis_cache_results.json`
  - `codex_coding/results/moe_routing_analysis_hetero_batch.json`
  - `codex_coding/results/moe_decomp_analysis_results.json`
  - `codex_coding/results/moe_layer_ablation_refresh_results.json`
  - `codex_coding/results/moe_multilayer_ablation_results.json`
  - `codex_coding/results/stable_cache_benchmark_results.json`
  - `generate_uniform.py`：新增 StableCacheBlockDiffusionIteration, StableCacheBlockDiffusionLLM
  - `dinfer/__init__.py`：新增 StableCacheBlockDiffusionLLM 导出
  - `docx/articles/2026-04-12_dllm_moe_insight_and_optimization_directions.md`（MoE insight 技术归档）

### 本轮命令

- 多次运行 MoE routing/decomp/ablation profiling 脚本
- 多次运行 stable_cache benchmark（v1, v2a, v2b, v2c）
- 运行多层组合 ablation
- 修改 generate_uniform.py（StableCacheBlockDiffusionIteration v1→v2 迭代）
- 阅读 EARTH 论文分析文档
- 更新 docx/articles, key_files_index, key_conclusion, progress_diff_summary

## 2026-04-13 (v0.1.13 续)

- **X-MoE 开源代码深度分析**
  - 完整扫描 `/home/wuhang/wuhang/dllm_wh/lib_cite/X-MoE/` 代码库
  - 理解三个核心组件：PFT (Padding-Free Token dispatch)、Grouped GEMM (persistent kernel, grid=NUM_SM)、RBD (Redundancy-Bypassing Dispatch)
  - 确认 X-MoE 架构同类（256 experts, top_k=8, DeepSeek-style），代码可直接参考

- **Padding-Free MoE Kernel 实现与测试**
  - 基于 X-MoE PFT 方案实现完整 padding-free MoE pipeline
  - 包含：PFT dispatch (sort + histogram) + Triton gather/scatter kernels + persistent grouped GEMM + BF16 支持
  - 正确性验证通过：cosine sim = 0.999994
  - **性能结论：反而更慢**
    - batch=1: 0.84ms vs baseline 0.53ms（慢 1.58x）
    - batch=32: 1.22ms vs baseline 0.80ms（慢 1.53x）
  - 根因：MoE kernel 瓶颈是 expert weight HBM loading（算术强度 ~1.5 MAC/byte，H100 平衡点 ~295），完全 memory-bound
  - padding 的 GEMM 计算几乎免费（tl.dot 对 mask=0 不耗时间）
  - vllm kernel 有 H100 专用 autotuning，我们的 grouped GEMM 缺少

- **完整实验报告生成**
  - 生成面向学术讨论的完整报告：`docx/articles/2026-04-12_dllm_moe_complete_experiment_report.md`
  - 包含：dLLM 背景、三个 Insight 完整数据、MoE 内部结构分析、Stable Cache 失败分析、Padding-Free kernel 分析、与 13 篇论文交叉分析、5 个开放问题

- **v0.1.13 总结论**
  - 三个 Insight 作为 observation 贡献成立（所有设置下稳健）
  - 直接"跳过计算"路径（Stable Cache）有根本性困难（误差累积，单层 safe ≠ 多层 safe）
  - 消除 padding 路径（Padding-free kernel）因 memory-bound 无法超越 vllm baseline
  - 仍有效的方向：减少 forward 次数（IterSmooth -12.8%）、多卡 EP 场景、EARTH 分层思想

- 新增/修改的文件：
  - `codex_coding/src/padding_free_moe.py` — PFT dispatch + grouped GEMM + gather/scatter
  - `codex_coding/src/bench_padding_free_moe.py` — Padding-free kernel benchmark
  - `codex_coding/results/padding_free_moe_benchmark_results.json` — Padding-free 性能数据
  - `codex_coding/results/padding_free_correctness_check.json` — 正确性验证
  - `docx/articles/2026-04-12_dllm_moe_complete_experiment_report.md` — 完整实验报告
  - `code_building/process_docs/v0.1-init-project/v0.1.13-dllm_moe_complete_experiment_report_archive.md` — 归档文档
  - `lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py` — 新增 padding-free 分支（如已合入）

### 本轮命令

- 阅读 X-MoE 代码库（gating.py, kernels.py, metadata.py, experts.py, moe_v2.py）
- 阅读 vllm fused_moe 源码（fused_moe.py, moe_align_block_size.py, layer.py）
- 实现 padding_free_moe.py
- 运行 bench_padding_free_moe.py（正确性 + 性能测试）
- 生成完整实验报告
- 更新 history-chat.txt、progress_diff_summary、key_files_index、key_conclusion

## 2026-04-13 ~ 2026-04-14 (v0.1.14)

- **MoE Selective Recompute Risk Proxy — Phase I 完整 characterization**
- 两大优化方向均已完成实验验证：

### 方向一：Token Temporal Reuse (Config-S / Config-D)
- v0.1.14.1: Fresh run logging (30 steps × 19 layers × 8 batch × 32 tokens)
- v0.1.14.2: 单点反事实干预 (5681 samples, KV cache)
- v0.1.14.3-4: Proxy 信号分析 (token_confidence AUC=0.900)
- v0.1.14.5: 组合迁移预检 (全量复用 NO-GO, 放大 5-37x)
- v0.1.14.5.1-2: 污染传播检测 + Layer-range sweep (中层最安全)
- v0.1.14.6-7: 端到端验证 (hook 开销问题发现)
- v0.1.14.8: Threshold sweep + Drift guard 效果 (Config-S/D 确定)

### 方向二：Expert Adaptive Pruning (top-p)
- v0.1.14.9a: Routing weight 分布分析 (Q2 有绝对阈值 bug)
- v0.1.14.9b: Expert pruning 第一版 (gate hook no-op bug)
- v0.1.14.9c: 修复版 (shared_rate=41.9%, top-4 可用)
- v0.1.14.9d: Q2 修正 (正确全局 top-p 框架)
- v0.1.14.9e: 自适应 top-p e2e (top-p=0.75: 51.5% 节省, ΔFwd=-2)

### 关键结论
- top-p=0.80 零损失 (42.6% expert 节省), top-p=0.75 甚至提升质量 (-2 forward)
- Config-D (margin>0.70 + drift<0.02) 22.1% token reuse
- 两者正交可叠加，但 Python hook 开销 > 计算节省
- 下一步：内联到原生代码 + 解决 fused_moe per-token variable-k

### 新增文件
- codex_coding/src/moe_risk_logger.py ~ moe_topp_e2e.py (17 个实验脚本)
- codex_coding/results/proxy_risk_prediction/ (11 个结果文件)
- code_building/process_docs/v0.1-init-project/v0.1.14-moe_selective_recompute_risk_proxy.md
- docx/cites/moe_selective_recompute_experiment_plan.md (实验总纲)

### 本轮命令
- 运行 17 个实验脚本（详见 v0.1.14 过程归档）
- 更新 history-chat.txt, progress_diff_summary, key_conclusion, key_files_index

## 2026-04-14 ~ 2026-04-15 (v0.1.15)

- **MoE 优化收益兑现：从 Pareto 前沿到 Expert Budgeting**
- 三个阶段完成：耦合实验 → kernel 瓶颈定位 → Expert Budgeting 算法设计与验证

### v0.1.15.1: 方向一 × 方向二耦合实验
- 将 Token Reuse (D1) 和 Expert Pruning (D2) 叠加测试
- Pareto 前沿：D2:tp75 (51.5% pair 节省, ΔFwd=-2), D1+D2:tp75_m90 (61.5%, +2), D1+D2:tp70_m70 (66.2%, +4)
- 关键发现：误差共振存在（两者独立 ΔFwd 均为负，耦合后变正），但输出质量不受影响
- 计划文档：docx/plans/2026-04-14_v0.1.15_coupled_optimization_plan.md

### v0.1.15.2: Kernel 瓶颈精确定位
- **Kernel 源码分析**：fused_moe_kernel 的 weight 乘法在 GEMM 之后 (line 468-472)，weight=0 不跳过计算
- **Kernel micro-benchmark**：物理 top-k 减少 ([N,8]→[N,4]→[N,1]) kernel 时间恒定 ~0.30ms → kernel 是访存密集型
- **Monkey-patch vs True-skip 等价性**：bf16 数值差异导致不同 forward 数（129 vs 141），但输出质量等价。ΔFwd 不可靠。
- **Active Expert Count 分析**：51.5% pair 节省只转化为 10.7% unique expert 减少 → pair 节省无法高效转化为 HBM 节省
- 关键结论：真正瓶颈是 unique active expert 的 HBM weight 搬运，不是 GEMM 计算

### v0.1.15.3: Expert Budgeting 算法
- **核心思想**：Pre-routing 限制 active expert set S，所有 token 只能从 S 中选 expert
- **安全约束**：quality_floor (总权重覆盖率 ≥ 0.85) + k_budget 不增 + 循环补充 exception
- **Phase 1 离线分析**：K=150 时 |S|≈181, HBM 节省 18%
- **Phase 2 端到端验证**：K=150 ΔFwd=0 (零损失), K=80 ΔFwd=-8, 全部质量 PASS
- **Phase 2b 边界探测**：top-p=0.75 是质量硬边界 (tp=0.60 崩溃), K 无下限 (safety constraint 保护)
- **Phase 2c batch=8 K 极限**：K=60 Active=113, HBM 节省 49.2%, 质量 PASS
- **Phase 2d batch=32 K 极限**：K=20 Active=139, HBM 节省 36.9%, 质量 PASS (轻微瑕疵)
- **最终标准配置：K40+tp75+D1**：Active=141.2, HBM 节省 36.1%, 全部可验证 prompt PASS, 无瑕疵

### 最终标准配置参数卡
| 参数 | 值 |
|------|-----|
| K_target | 40 |
| quality_floor | 0.85 |
| top-p | 0.75 |
| D1 margin | 0.90 |
| D1 reuse layers | L4-L14 |
| D1 drift guard | 0.02 |
| batch=32 Active | 141.2/220.8 |
| HBM 节省 | 36.1% |
| ΔFwd | -2 |

### 待执行：Phase 3 wall-clock 验证
- 用标准配置 K40+tp75+D1，在 kernel 层面测真实 wall-clock speedup
- 需要 ncu profiling 精确定位 HBM 流量变化

### 新增关键文件
- codex_coding/src/moe_coupled_e2e.py — v0.1.15.1 耦合实验
- codex_coding/src/bench_fused_experts_topk.py — kernel micro-benchmark
- codex_coding/src/bench_monkey_vs_trueskip.py — 等价性验证
- codex_coding/src/bench_active_expert_count.py — active expert 分析
- codex_coding/src/collect_routing_data.py — routing 数据采集
- codex_coding/src/analyze_expert_budgeting.py — 离线可行性分析
- codex_coding/src/expert_budgeting_e2e.py — Phase 2 端到端验证
- codex_coding/src/expert_budgeting_boundary.py — Phase 2b 边界探测
- codex_coding/src/expert_budgeting_k_boundary.py — Phase 2c batch=8 K 极限
- codex_coding/src/expert_budgeting_batch32_boundary.py — Phase 2d batch=32 K 极限
- codex_coding/results/expert_budgeting_batch32_boundary.json — 最终结果数据
- docx/plans/2026-04-14_v0.1.15_coupled_optimization_plan.md — 耦合优化计划
- codex_coding/results/fused_experts_topk_benchmark.json — kernel benchmark 数据
- codex_coding/results/monkey_patch_vs_true_skip_comparison.json — 等价性验证数据
- codex_coding/results/active_expert_count_analysis.json — active expert 分析数据
- codex_coding/results/expert_budgeting_routing_data.pt — routing 原始数据 (1.1GB)
- codex_coding/results/expert_budgeting_feasibility.json — 离线可行性分析结果

### 本轮命令
- 运行 moe_coupled_e2e.py (Pareto 前沿实验)
- 运行 bench_fused_experts_topk.py (kernel micro-benchmark)
- 运行 bench_monkey_vs_trueskip.py (等价性验证)
- 运行 bench_active_expert_count.py (active expert 分析)
- 运行 collect_routing_data.py (采集 routing 数据)
- 运行 analyze_expert_budgeting.py (离线可行性分析)
- 运行 expert_budgeting_e2e.py (Phase 2 端到端)
- 运行 expert_budgeting_boundary.py (Phase 2b 边界)
- 运行 expert_budgeting_k_boundary.py (Phase 2c)
- 运行 expert_budgeting_batch32_boundary.py (Phase 2d)

## 2026-04-14 (v0.1.15.1)

- **方向一 × 方向二耦合优化实验计划（讨论阶段）**
- 确定耦合架构：方向二（top-p）作为底座（always-on），方向一（temporal reuse）作为叠加层
- 决策逻辑：stable token → 全跳（D1）；非 stable → top-p pruning（D2）
- 理论总节省估算：~62%（D1 覆盖 ~22% 位置 × 100% + D1 未覆盖 ~78% × 51.5%）
- 识别三个隐含假设：(A) top-p 是质量中性去噪 (B) proxy 跨条件有效 (C) 误差独立不共振
- 设计了退路决策树：通过同时观测 ΔFwd 和 reuse rate 区分哪个假设被打破
- 如果假设 C 被打破（误差共振），转向"cache 补偿"新架构（D1 为 D2 裁掉的 expert 提供近似）
- 如果假设 B 被打破（proxy 失效），在 pruned 环境下重新做 characterization
- 如果假设 A 被打破（top-p 非无损），转为联合误差预算分配框架
- 保底方案：即使所有耦合失败，D2 单独（top-p=0.75, 51.5% 节省, ΔFwd=-2）仍是强结果
- 新增计划文档：`docx/plans/2026-04-14_v0.1.15_coupled_optimization_plan.md`

### 本轮命令
- 阅读 history-chat.txt 恢复上下文
- 阅读所有指定文档（next_step.txt, docx guide, progress_diff_summary, experiment plan, complete report）
- 阅读 3 个关键实验脚本（moe_topp_e2e.py, moe_expert_pruning_fixed.py, moe_drift_guard_sweep.py）
- 阅读模型代码（modeling_llada2_moe.py:620-680, generate_uniform.py:198-440）
- 与用户讨论耦合方案，形成计划文档

### v0.1.15.1 实验执行

- 新增耦合实验脚本 `codex_coding/src/moe_coupled_e2e.py`
- 测试了 13 个配置（4 baselines + 2 D1-only + 7 coupled），每个 5 次 temp=0 + 质量检查 temp=0.7
- **D2 alone 结果完全复现 v0.1.14**：tp80=ΔFwd 0, tp75=-2, tp70=+1
- **D1 alone 结果完全复现 v0.1.14**：m90=-1, m70_d02=-4
- **关键发现：假设 C（误差独立性）被打破**
  - D1 独立 ΔFwd=-4, D2 独立 ΔFwd=-2, 预期叠加 ΔFwd=-6
  - 实际耦合 D1+D2:tp75_m70_d02 ΔFwd=+4, 偏差 +10 — 确认误差共振
- **假设 B（proxy 有效性）成立**：reuse rate 22.1% → 20.9%（比率 0.95）
- **但耦合仍有实用价值**：
  - D1+D2:tp75_m90: ΔFwd=+2, expert savings=61.5% (SAFE)
  - D1+D2:tp80_m70_d02: ΔFwd=+2, expert savings=55.0% (SAFE)
  - 即使多 2-4 次 forward，绝对 expert 计算节省约 60%
- **Pareto 前沿**：
  - 质量优先：D2:tp75 单独（51.5% 节省, ΔFwd=-2）
  - 节省优先：D1+D2:tp75_m90（61.5% 节省, ΔFwd=+2）
  - 最大节省：D1+D2:tp70_m70_d02（66.2% 节省, ΔFwd=+4）
- 输出质量检查：所有耦合配置的文本仍然连贯、正确
- 新增结果文件：`codex_coding/results/coupled_optimization_results.json`

### 本轮命令
- `python3 codex_coding/src/moe_coupled_e2e.py`（13 configs × 5 runs + quality check）

### v0.1.15.2 Wall-Clock 收益兑现计划（讨论阶段）

- 讨论并制定了 wall-clock 收益兑现的完整 profiling 方案
- 识别了 7 个开销源（C1-C7）：kernel weight=0 处理、gate routing、top-p 决策逻辑、D1 cache 管理、Python hook dispatch、fused_experts API 限制、内联不可消除开销
- 确定 Step 1（fused_experts kernel micro-benchmark）为门槛测试：
  - 测 weight=0 vs 物理裁剪 vs 正常 routing 的 kernel 时间
  - 判断结果属于 α（物理跳过）/ β（间接跳过）/ γ（不跳过）
- 设计了 Step 2（逐组件 profiling）和 Step 3（hook overhead isolation）
- 设计了决策分叉树：α→内联即可，β→评估是否足够，γ→物理裁剪/改 kernel/接受 forward 次数收益
- 新增计划文档：`docx/plans/2026-04-14_v0.1.15.2_wall_clock_realization_plan.md`

### 本轮命令
- 与用户讨论 wall-clock 兑现方案，形成里程碑计划文档

### v0.1.15.2 Step 1 执行

- 完成源码阅读准备工作：
  - 读取 autotuning JSON：按 M（token 数）索引，不按 top_k，公平可比
  - 读取 moe_align_block_size：只处理 topk_ids，不看 weights，weight=0 不改变 grid
  - 读取 fused_moe_kernel Triton 代码：**weight 乘法在 GEMM 循环之后（line 468-472）**，确认 weight=0 不跳过任何计算
- 新增 benchmark 脚本 `codex_coding/src/bench_fused_experts_weight_zero.py`
- **Step 1 结论：情况 γ-worst**
  - B/A ≈ 1.00：weight=0 完全不省 kernel 时间（所有 N 下 B/A > 0.986）
  - C/A = 0.85-0.98：物理裁剪到 top-4 效果也有限（N=1024 仅省 7%）
  - D/A = 0.62-1.08：物理裁剪到 top-1 在大 N 下省 35%，小 N 下反而更慢
  - **根因：kernel 完全 memory-bound，瓶颈是 expert weight HBM loading，不是 GEMM 计算量**
  - 减少 token-expert pairs 不能减少 unique active expert 数（N=1024 下 top-4 仍有 245/256 active experts）
  - 只有大幅减少 active expert 数（top-1: 170/256）才能显著减少 weight loading
- 新增结果文件：`codex_coding/results/fused_experts_weight_zero_benchmark.json`

### 本轮命令
- 阅读 vllm fused_moe 源码（fused_moe.py, moe_align_block_size.py, autotuning JSON）
- `python3 codex_coding/src/bench_fused_experts_weight_zero.py`

## 2026-04-15 ~ 2026-04-16 (v0.1.15.4)

- **Phase 3 Wall-Clock 验证 + Baseline 优化**
- 从 kernel 级验证到端到端 profiling 到 baseline 优化的完整流程

### v0.1.15.4a: routing_p 重构讨论
- 发现旧 top-p=0.75 算法依赖 SHARED_RATE=0.419（来自 v0.1.12 bench_moe_decomp.py 实测值），跨层不稳定
- 决定改用 **routing_p=0.6**：直接在 routing weight 上做 top-p，消除 SHARED_RATE 依赖
- 等价换算：旧 top_p=0.75 → routing_p≈0.57 → 新 routing_p=0.6（略保守，更安全）
- 后续代码中使用 EQUIV_TOP_P = 0.6 * ROUTING_RATE + SHARED_RATE = 0.768 传入旧接口

### v0.1.15.4b: ncu Profiling（★ 关键成果）
- 用 `sudo ncu` 对 fused_experts kernel 做了精确 HBM 流量测量
- 3 个 case：baseline (222 experts), EB_K40_rp60 (128 experts), restrict_60 (56 experts)
- **核心结果**：
  | Case | HBM Read (MB) | Duration (μs) |
  |------|---:|---:|
  | baseline (222) | 775.6 | 269.5 |
  | EB_K40_rp60 (128) | 433.5 | 163.2 |
  | restrict_60 (56) | 517.3 | 189.4 |
- Expert Budgeting: **HBM 减少 44%, kernel 时间减少 39%**
- kernel 带宽利用率 86% of H100 peak (2.88 TB/s)

### v0.1.15.4c: 端到端 Wall-Clock + torch.profiler
- 用 HetEval-32 配置跑端到端 timing
- Hook 版本慢 4 倍（10.7s → 42.3s），Python hook 开销淹没 kernel 收益
- torch.profiler 组件分解：**fused_moe_kernel 占 Self CUDA 的 43.2%**（此前估算 13.4% 偏低）

### v0.1.15.4d: nsys Full Profiling（Level 0 + Level 3）
- 用 nsys + NVTX markers 做完整推理组件分解
- Level 3 标记：Embedding, RMSNorm, Attention (QKV/OProj), MoE (Gate/Shared/Routed), LMHead, Decoder
- Level 0 标记：Iter_ModelForward, Iter_DecoderDecode, Runner_SelectUndecoded, etc.
- **关键发现**：
  - MoE Routed (fused_moe): 34.7% of wall-clock
  - Attention (含 QK norm + RoPE + Flash): 29.5%
  - RMSNorm: 10.5%
  - 未标记的"gap"主要是 Python dispatch + residual add (~4.8%)
  - Decoder 逻辑和 cache 管理 < 2%
  - Attention 内部：Flash kernel 只占 4%，75% 是 SDPA 调用约定引入的 overhead (contiguous×3 + repeat_kv + transpose)

### v0.1.15.4e: Baseline 优化（★★★ 核心成果 ★★★）
三项 baseline 优化叠加：

1. **maximum_unroll=4**（参数调优）→ 减少 Python loop overhead，+3.2%
2. **Fused RMSNorm**（monkey-patch vllm rms_norm kernel）→ 41 个 norm 模块从 7 kernel→1 kernel，+7%
   - 跳过 attention 内部的 query_layernorm / key_layernorm（4D 非 contiguous）
3. **Classic flash-attn 2.8.3**（替换 SDPA）→ 消除 contiguous×3 + repeat_kv + QK norm fuse，+8.5%
   - QK Norm 移到 transpose 前执行（contiguous → 可用 fused RMSNorm）
   - flash_attn_func 原生 (B,S,H,D) + GQA，不需要 repeat_kv
   - SDPA fallback：当 attention_mask 不为 None 时（prefill/cross-block）仍用 SDPA

**总加速：18.4%（旧 baseline 10.775s → 新 baseline 8.789s, 29.8 fwd/s）**
**HetEval-32 质量：5 个可验证 prompt 全 PASS**

### 尝试但未成功的优化
- **flash-attn-4 (CUTE DSL)**：比 SDPA 慢 5.5%（JIT 或 kernel 本身不如 cuDNN）
- **flashinfer**：micro-benchmark 快 2x，但 generate 中 20 层数值差异累积导致不收敛
- **FlashAttention-3 (SM90 kernel)**：需要从源码编译 hopper/ 目录，451 个 .cu 文件编译时间过长（~90min），已中断
- **CUDA Graph**：dInfer 的 block diffusion 有动态 batch size，不适用

### 当前已确认的注意力实现细节
- flash-attn 2.8.3 的 .so 只包含 SM80 kernel（FA2），无 SM90 kernel（FA3）
- 但 8.5% 加速来自消除 overhead，不是更快的 kernel
- PyTorch SDPA 的 cuDNN 后端（cudnn_sdp_enabled=True）可能已是 Hopper FA3 级别

### HetEval-32 标准评测配置
| 参数 | 值 |
|------|-----|
| 模型 | LLaDA2.0-mini (路径: /mnt/models/LLaDA2.0-mini) |
| batch_size | 32 |
| prompts | 32 条异质 prompt（含 5 个可验证任务） |
| gen_length | 256 |
| block_length | 32 |
| threshold | 0.90 |
| temp | 0 (timing) / 0.7 (quality) |
| cache | prefix, lazy=True, inplace=True |
| maximum_unroll | 4 |

### 新增关键文件
- codex_coding/src/bench_ncu_fused_experts.py — ncu profiling 脚本
- codex_coding/src/phase3_wallclock_profiling.py — 端到端 wall-clock + torch.profiler
- codex_coding/src/nsys_full_profiling.py — nsys Level 0+3 NVTX profiling
- codex_coding/src/optimized_baseline.py — max_unroll + fused RMSNorm
- codex_coding/src/optimized_baseline_v2.py — + flash-attn-4 (失败)
- codex_coding/src/optimized_baseline_v3.py — + classic flash-attn 2.8.3 (成功, ★当前最优)
- codex_coding/src/heteval32_verify_optimized.py — 输出质量验证
- codex_coding/results/phase3_wallclock_profiling.json — wall-clock + profiler 数据
- codex_coding/results/optimized_baseline_timing.json — v1 timing
- codex_coding/results/optimized_baseline_v2_timing.json — v2 timing
- codex_coding/results/baseline_level0_3.nsys-rep — nsys profiling 数据
- codex_coding/results/baseline_level0_3.sqlite — nsys SQLite 数据

### 本轮命令
- sudo ncu --metrics dram__bytes_read.sum,... bench_ncu_fused_experts.py
- python phase3_wallclock_profiling.py
- nsys profile --capture-range=cudaProfilerApi nsys_full_profiling.py
- python optimized_baseline.py / v2 / v3
- python heteval32_verify_optimized.py
- pip uninstall flash-attn-4; pip install flash-attn (2.8.3)

## 2026-04-16 ~ 2026-04-17 (v0.1.15.5)

- **Expert Budgeting 内联尝试 + 新算法设计 + 参数 sweep + 性能深度分析**

### v0.1.15.5a: nsys Profiling 新 Baseline 组件分解
- 创建轻量化 nsys profiling 脚本 (module-level hooks only, 无 forward 重写)
- 新 baseline 组件占比: **MoE 60.0%, Attention 36.2%, RMSNorm 3.1%**
- Python dispatch gap 几乎为零 → MoE 是唯一需要优化的目标
- 新增: nsys_optimized_baseline_profiling.py, optimized_baseline_level2.nsys-rep

### v0.1.15.5b: 简化版 Expert Budgeting Inline（失败）
- popularity top-S 方案, 6 tensor ops/层, zero Python loops
- **全部更慢**: S=128 慢 16.2%, S=80 慢 19%
- 质量也退化: S=96 出现乱码
- 原因: 6 ops × 19 层 × ~25μs/op = 2.85ms/forward 的 Python dispatch 远超 kernel 节省
- 新增: expert_budgeting_inline_simple.py

### v0.1.15.5c: 新算法 — Expanded Top-(K+M) Budgeted Routing
- 与算法专家讨论后设计的新算法, 替代原 dense compute_active_set
- 核心改进:
  - 候选空间从 [N, 256] 缩减为 [N, K+M=12]
  - 统一 expanded weight 分数体系 (sigmoid 空间)
  - batch-add 替代逐个补充 (vectorized, 无 GPU→CPU sync)
  - group_limited_topk 的 diversity 由 Step 7 final routing 保证
- 算法文档: docx/cites/expanded_budgeted_routing_refactor.md
- 新增: expanded_budgeted_routing_e2e.py, baseline_optimizations.py (共享模块)

### v0.1.15.5d: 参数 Sweep（三组实验）
- **Sweep A** (per_round_cap, QF=0.85): cap=1→|S|=122, cap=64→|S|=172
  - cap 越小 |S| 越紧凑, cap=8 是平衡点
- **Sweep B** (quality_floor, cap=8): QF=0.40→|S|=59(质量退化), QF=0.70→|S|=103(PASS)
  - QF=0.70 是甜点 (|S|=103, HBM 节省 53%, 质量 PASS)
- **Sweep C** (q_major, cap=8, QF=0.70): q_major=0.85→|S|=75(质量退化), q_major=0.95→|S|=103(PASS)
  - q_major=0.95 是质量硬边界
- **锁定配置: M=4, K_target=40, cap=8, QF=0.70, q_major=0.95, no top-p**
  - |S|=103, HBM 节省 53%, ΔFwd=+13, 质量 PASS
- 新增: expanded_budgeted_routing_sweep.py, expanded_budgeted_routing_qmajor_sweep.py
- 结果: expanded_budgeted_routing_e2e.json, expanded_budgeted_routing_sweep.json, expanded_budgeted_routing_qmajor_sweep.json

### v0.1.15.5e: 性能深度 Profiling
- Hook 版本端到端: 42.9s (baseline 8.8s, +381%, 4.8x 慢)
- Per-step sync 分析: batch-add 占 74.4%, F2 edge construction 最贵 (32.4%)
- **Baseline vs EB 同条件对比 (★ 关键发现)**:
  | 组件 | Baseline (μs) | EB (μs) | Δ |
  |------|-------------|---------|---|
  | shared_experts | 135 | 139 | +4 |
  | gate.get_logits | 119 | 118 | -0 |
  | forward_impl (routing+fused) | **1,199** | — | — |
  | EB routing + fused_experts | — | **992** | **-208** |
  | EB batch-add | — | 4,332 | +4,332 |
  - **compute path 真实节省 208μs/层 = 11.7% 端到端**
- 新增: expanded_budgeted_routing_profiling.py, baseline_vs_eb_profiling.py
- 结果: expanded_budgeted_routing_profiling.json, baseline_vs_eb_moe_profiling.json

### v0.1.15.5f: MoE 开销深度分析
- fused_experts 调用 Triton kernel **两次** (w1 gate+up, w2 down)
- 每次 kernel ~270μs (baseline), 固定开销 (moe_align + silu + alloc) ~290μs
- **kernel 占 fused_experts 的 65%, 固定开销占 35%**
- HBM 节省 53% → kernel 省 39% → fused_experts 省 25% → forward_impl 省 17% (逐层稀释)
- MoE 每层 ~28 个 CUDA op, 37% (sync) / 15% (wall-clock) 是 Python dispatch 固定开销
- **routing 12 个小 op 的 GPU 时间 (5-15μs) < CPU dispatch (25μs) → GPU 空闲等 CPU**
- Per-layer 分布: L0/L1 |S| 最大 (159-223), L17 最小 (99-160)

### v0.1.15.5g: 优化方向共识
- **优先级确定: 先用 torch.compile 削减框架开销, 再叠加 Expert Budgeting**
- torch.compile 预期: routing 12 ops 融合 → 省 200-350μs/层 → 4-7% 端到端
- compile + EB (预计算方案) 叠加预期: ~10% 端到端
- 待执行: torch.compile 实验

### 新增关键文件清单
- codex_coding/src/baseline_optimizations.py — ★ 共享 baseline 优化模块
- codex_coding/src/nsys_optimized_baseline_profiling.py — 轻量化 nsys profiling
- codex_coding/src/expert_budgeting_inline_simple.py — 简化版 EB inline (失败)
- codex_coding/src/expanded_budgeted_routing_e2e.py — 新算法 E2E 验证
- codex_coding/src/expanded_budgeted_routing_sweep.py — cap + QF sweep
- codex_coding/src/expanded_budgeted_routing_qmajor_sweep.py — q_major sweep
- codex_coding/src/expanded_budgeted_routing_profiling.py — 新算法 per-step profiling
- codex_coding/src/baseline_vs_eb_profiling.py — baseline vs EB 对比 profiling
- docx/cites/expanded_budgeted_routing_refactor.md — 新算法完整设计文档
- codex_coding/results/optimized_baseline_nvtx_summary.json — nsys 组件分解
- codex_coding/results/expert_budgeting_inline_simple.json — 简化版 EB 结果
- codex_coding/results/expanded_budgeted_routing_e2e.json — 新算法初始验证
- codex_coding/results/expanded_budgeted_routing_sweep.json — cap + QF sweep 数据
- codex_coding/results/expanded_budgeted_routing_qmajor_sweep.json — q_major sweep 数据
- codex_coding/results/expanded_budgeted_routing_profiling.json — 新算法 profiling 数据
- codex_coding/results/baseline_vs_eb_moe_profiling.json — BL vs EB 对比数据

### 本轮命令
- python nsys_optimized_baseline_profiling.py (timing + nsys)
- python expert_budgeting_inline_simple.py (简化版 EB, 失败)
- python expanded_budgeted_routing_e2e.py (新算法初始验证)
- python expanded_budgeted_routing_sweep.py (cap + QF sweep)
- python expanded_budgeted_routing_qmajor_sweep.py (q_major sweep)
- python expanded_budgeted_routing_profiling.py (per-step profiling)
- python baseline_vs_eb_profiling.py (BL vs EB 对比)

## 2026-04-18 (v0.1.15.6 ~ v0.1.15.7)

- **torch.compile 探索 + Triton Kernel 融合 + Expert Budgeting Triton 化**

### v0.1.15.6: torch.compile 探索（全部失败）

**v0.1.15.6a: Partial compile (Prism 方案)**
- 参考 Fast-dLLM / SGLang / Prism 三个框架的优化策略
- 尝试 torch.compile(model) + torch._dynamo.disable on MoE submodules
- **结果：+22.8% 更慢** (8.75s → 10.75s, fwd 262→274)
- 根因：dInfer DynamicCache 的 `len(past_key_value.layers)` 变化触发 Dynamo 反复重编译
- 重编译超过 limit(8) 后 fallback to eager，compile overhead 全部是净损失

**v0.1.15.6b: MoE-only compile**
- 只编译 MoE block forward (避开 KV cache 问题)
- **结果：+10.6% 更慢** (8.76s → 9.69s, fwd 262→274)
- 根因：fused_moe Triton kernel 造成 graph break → break 点 overhead 超过 routing fusion 节省

**结论：torch.compile 对包含 Triton custom kernel 的 MoE 路径在 PyTorch 2.8.0 下无效**

### v0.1.15.7: Fused MoE Routing Triton Kernel（★★★ 核心突破）

**v0.1.15.7a: Fused routing kernel 实现**
- 手写 Triton kernel 融合 gate.routing 的 12 个 post-GEMM ops
  (sigmoid → group_limited_topk → gather → normalize → scale)
- Micro-bench: 4x speedup (0.188ms → 0.048ms), IDs 精确匹配, cosine sim = 1.0
- **E2E: -12.4% vs C0 (8.79s → 7.70s, 272 fwd), 质量 5/5 PASS**
- Per-forward: 33.56 → 28.32 ms/fwd (-15.6%)
- 新增: test_fused_routing.py

**v0.1.15.7b: Fused routing + Expert Budgeting (Plan C per-block popularity)**
- 在 fused routing 上叠加 EB：per-block 计算 S_mask, 跨迭代复用
- K_target sweep: 120, 100, 80
- **结果：EB 未带来额外加速** (fwd 272→285, per-fwd 仅省 0.18ms)
- 根因 1: S_mask 限制 routing → fwd 次数增加 (+13) 抵消节省
- 根因 2: fused_experts kernel 节省远低于 ncu 预期 (0.18ms vs 预期 3.7ms)
- 新增: test_fused_routing_eb.py

**v0.1.15.7c: Plan A — Per-Forward Warm-Start Batch-Add（算法验证）**
- 每次 forward 做热启动 batch-add (上次 |S| 作为 K_init)
- **关键发现：fwd 次数控制住了 (274, 仅 +2 vs C5 的 272)**
- 热启动极有效：99.9% 零轮 batch-add
- **但 Python dispatch 开销 12.55ms/fwd (24 ops + 2 GPU→CPU sync / 层)**
- 总时间 11.38s (+29.4% vs C0) — 算法正确但实现太慢
- 新增: test_plan_a_warmstart.py

**v0.1.15.7d: Plan A Optimized — Cold/Hot Dual Path**
- 冷启动 (block 首次 fwd): 完整 batch-add 带质量保证
- 热启动 (block 后续 fwd): popularity top-K_init, 跳过 coverage check 和 sync
- Block 边界检测: N > prev_N (prefix 增长 = 新 block)
- 初版 cold=342 (检测 bug), 修复后 cold=171 (正确)
- 热启动仍慢：12 Python dispatch / 层无法被 GPU pipeline 隐藏
- 新增: test_plan_a_optimized.py

**v0.1.15.7e: Fused EB Triton Kernels — Stage 1+2（当前最新）**
- 4 个 Triton kernel:
  - K_A: per-token sigmoid + topk(12) + normalize + atomic popularity
  - K_B_v2: sort popularity → S_mask + zero pop (1 kernel 替代 3 Python ops)
  - K_C: per-token coverage check + batch-add scoring (冷启动)
  - K_D_v2: sat check + expert selection + S update + zero buffers (冷启动)
- 热启动: 2 Triton dispatch/层, 0 sync, 0 .zero_()
- 冷启动: 35 Triton dispatch/层, 0 sync (GPU-side early exit)
- **结果: 8.622s, -1.9% vs C0, 276 fwd, 质量 5/5 PASS**
- 但仍比 C5 (7.70s) 慢 11.9%，EB 的 kernel 节省不足以覆盖 EB 自身计算
- 新增: test_fused_eb_triton.py

### 当前最优结果
- **C5 (Fused routing only): 7.704s, -12.4% vs C0 (8.792s)**
- Fwd: 272 (vs C0 262, tie-breaking 差异)
- 质量: HetEval-32 5/5 PASS

### 新增关键文件
- codex_coding/src/test_partial_compile.py — torch.compile 实验
- codex_coding/src/test_moe_compile.py — MoE-only compile 实验
- codex_coding/src/test_fused_routing.py — ★ fused routing Triton kernel
- codex_coding/src/test_fused_routing_eb.py — fused routing + EB (Plan C)
- codex_coding/src/test_plan_a_warmstart.py — Plan A Python warm-start
- codex_coding/src/test_plan_a_optimized.py — Plan A cold/hot dual path
- codex_coding/src/test_fused_eb_triton.py — ★ fused EB Triton kernels (4 kernels)
- codex_coding/results/partial_compile_results.json
- codex_coding/results/moe_compile_results.json
- codex_coding/results/fused_routing_results.json
- codex_coding/results/fused_routing_eb_results.json
- codex_coding/results/plan_a_warmstart_results.json
- codex_coding/results/plan_a_optimized_results.json
- codex_coding/results/fused_eb_triton_results.json

### 本轮命令
- CUDA_VISIBLE_DEVICES=4 python test_partial_compile.py (torch.compile C1-C3)
- CUDA_VISIBLE_DEVICES=4 python test_moe_compile.py (MoE compile C4)
- CUDA_VISIBLE_DEVICES=4 python test_fused_routing.py (fused routing C5)
- CUDA_VISIBLE_DEVICES=4 python test_fused_routing_eb.py (EB Plan C)
- CUDA_VISIBLE_DEVICES=4 python test_plan_a_warmstart.py (Plan A Python)
- CUDA_VISIBLE_DEVICES=4 python test_plan_a_optimized.py (Plan A cold/hot)
- CUDA_VISIBLE_DEVICES=4 python test_fused_eb_triton.py (fused EB Triton)

## 2026-04-19 (v0.1.15.8)

- **EB 热路径优化系列实验 + 大 batch 验证 + S_mask 稳定性分析**

### v0.1.15.8a: 组件级 Profiling (C5 vs C10)
- 编写 nsys_c5_c10_profiling.py，NVTX + CUDA event 双计时
- 在 gen_length=64, batch=32 下分别 profiling C5 和 C10
- **C5 组件分布**: fused_experts 50.6%, Attention 20.9%, routing 7.9%, shared 7.7%
- **C10 vs C5 关键发现**:
  - fused_experts 逐层确实变快 -14.5%（排除 L7 异常）
  - 但 EB hot 总开销 551.8ms >> fused_experts 节省 206.3ms
  - EB hot per call = 289μs (K_A ~100μs + K_B_v2 ~120μs + Python ~40μs)
  - K_B_v2 的 256 次串行 iterative argmax 是 EB 的主要瓶颈
- **结论**: EB 方向对，但 hot path 开销太大

### v0.1.15.8b-d: K_init 修正 + K_A/K_B 优化实验
- **v0.1.15.8b (C10-v2)**: K_init 从 168 修正为 cold path 实际 |S|（~91）+ torch.topk 替换 K_B_v2
  - fwd 从 89 增加到 92 → |S| 太紧
- **v0.1.15.8c (C10-v3)**: K_init=actual |S| + 原版 K_B_v2
  - fwd=91, ms/fwd 从 31.96 降到 31.21 → K_init 修正有效但 fwd +2
  - 加 2×cap margin (K_init=|S|+16): fwd=91, 改善有限
- **v0.1.15.8d (C10-v4)**: K_A_hot_lite (去 sigmoid / count-based / weight-based) → 均导致 fwd 增加
- **结论**: K_A 的 sigmoid + normalize 是必要的，不能简化

### v0.1.15.8e: 算法同学方案 — K_A_v2 (local_pop) + K_B_v3 (tl.sort)
- 参考 docx/cites/warmstart_triton_final.md 的方案
- K_B_v3 (tl.sort): micro-bench 1.65x 加速 (72μs → 43μs)，Jaccard=1.0
- K_A_v2 (local_pop): 反而更慢（寄存器压力），BLOCK_N 越大越慢
- E2E: K_A_v2 + K_B_v3 反而比 K_A + K_B_v2 略慢（K_A_v2 劣化抵消 K_B_v3 改善）

### v0.1.15.8f (C10-v6): 原版 K_A + K_B_v3 (tl.sort) — 最优 kernel 组合
- 原版 K_A（per-token atomic，已证明够快）+ K_B_v3（tl.sort，1.65x）
- gen_length=64: 2.884s vs C5 2.542s → +13.5%
- **确认为最优 kernel 组合**（K_A 不动 + K_B 用 tl.sort）

### v0.1.15.8g: S_mask 稳定性数据采集 (HetEval-32)
- 在 gen_length=256 下采集每次 hot forward 的 S_mask
- **q_major=0.95**: Adjacent Jaccard 0.920, vs Cold 0.596, M=2 安全窗口 31.8%
- **q_major=1.0**: Adjacent Jaccard 0.936, vs Cold 0.630, M=2 安全窗口 51.7%
- Block 后期 (fwd_in_block > 20) Jaccard 达 0.96-0.98
- **结论**: q_major=1.0 显著改善稳定性，分段策略可行

### v0.1.15.8h: 1-Forward Delay 质量验证
- 模拟 overlap：hot path 返回上一次 S_mask，同时计算新 S_mask
- **q_major=1.0 + delay: fwd=272，和 C5 完全一致！** 质量 5/5 PASS
- q_major=0.95 + delay: fwd=282 → 不可用
- **结论**: q_major=1.0 是 delay/skip 策略的前提

### v0.1.15.8i: CUDA Stream Overlap 实验
- 用 torch.cuda.Stream 把 K_A+K_B 放到 eb_stream 和 fused_experts 并行
- **结果: 反而更慢** (9.292s vs 模拟 8.728s)
- 根因: fused_experts 是 memory-bound，K_A 在 eb_stream 上争抢 HBM 带宽
- 改进: 去掉 .copy_() 用双缓冲指针 swap → 9.023s（改善 270ms 但仍比模拟慢）
- **结论: overlap 方向因 HBM 竞争不可行**

### v0.1.15.8j: M-skip Sweep (batch=32, q_major=1.0) ★★★
- M=1 到 M=∞ 全部 5/5 质量 PASS
- fwd 稳定在 270-274（波动 ±2）
- **M=∞ (cold-only): 7.998s vs C5 7.693s → +4.0%**
- EB 的 "地板" = +4% (cold path + s_mask routing 开销)
- 在 batch=32 下 EB 始终不如 C5

### v0.1.15.8k: HetEval-128 大 batch 验证 ★★★★★
- 构建 128 个异质 prompt (HetEval-128)
- **batch=128, M=∞ (cold-only): 12.134s vs C5 12.412s → -2.2% ★**
- **C10 首次超过 C5！** ms/fwd 从 44.65 降到 42.57 (-4.7%)
- fwd 增加 +7 (285 vs 278)，但 per-fwd 节省超过 fwd 惩罚
- |S| avg=160.5 (min=118, max=168)
- 质量 5/5 PASS
- **关键 insight: EB 收益随 batch size 增长，大 batch 下优势更明显**

### 当前最优结果
- **batch=32: C5 (fused routing only) = 7.70s, -12.4% vs C0** ← 小 batch 最优
- **batch=128: C10-M5 (fused routing + EB M=5, K_B_v3) = 12.07s, -2.8% vs C5, ~-15.0% vs C0** ← 大 batch 最优

### v0.1.15.8l: K_B_v3 全局升级 + batch=128 M-sweep ★★★
- K_B_v3 (tl.sort) 写入 test_fused_eb_triton.py 作为默认 K_B，全局替换 K_B_v2
- 同步更新 test_heteval128.py, test_m_skip_sweep.py, test_m_skip_b128.py
- HetEval-128 M=∞ 升级前后对比：零回归（fwd 285 不变，|S| 160.5 不变，质量 5/5 PASS）
- **batch=128 M-sweep**: M=5 是最优（12.068s, -2.8% vs C5），比 M=∞ (-2.2%) 多 0.6% 收益
- M=5 的周期性 S_mask 刷新减少 fwd（285→282），是关键 +0.6% 来源
- **新最优: C10-M5 batch=128 = 12.068s, 282 fwd, 42.80 ms/fwd**

### 待执行任务
1. **在 batch=128 下做组件级 profiling**（C5 vs C10-M5），确认下一个优化目标
2. 冷路径 K_C+K_D batch-add 循环开销确认
3. 基于 profiling 结果决定进一步优化方向

### 新增关键文件
- codex_coding/src/nsys_c5_c10_profiling.py — ★ 组件级 profiling (C5 vs C10)
- codex_coding/src/test_c10_v2.py — K_init fix + torch.topk
- codex_coding/src/test_c10_v3.py — K_init fix + K_B_v2
- codex_coding/src/test_c10_v4.py — K_A_hot_lite 实验
- codex_coding/src/test_c10_v5.py — K_A_v2 (local_pop) + K_B_v3
- codex_coding/src/test_c10_v6.py — ★ 原版 K_A + K_B_v3 (最优 kernel 组合)
- codex_coding/src/collect_smask_stability.py — S_mask 稳定性采集
- codex_coding/src/test_delay_simulation.py — 1-forward delay 模拟
- codex_coding/src/test_c10_overlap.py — CUDA stream overlap
- codex_coding/src/test_m_skip_sweep.py — ★ M-skip sweep (batch=32)
- codex_coding/src/test_heteval128.py — ★★★ HetEval-128 大 batch 验证
- codex_coding/src/test_m_skip_b128.py — ★★★ M-skip sweep at batch=128
- codex_coding/src/test_triton_stream.py — Triton stream 兼容性验证
- docx/cites/warmstart_triton_final.md — 算法同学 Triton 优化方案
- codex_coding/results/c5_component_profiling.json
- codex_coding/results/c10_component_profiling.json
- codex_coding/results/c10_v2_comparison.json ~ c10_v6_comparison.json
- codex_coding/results/s_mask_stability_qm095.json / qm10.json
- codex_coding/results/delay_simulation_results.json
- codex_coding/results/c10_overlap_results.json
- codex_coding/results/m_skip_sweep_results.json
- codex_coding/results/m_skip_sweep_b128_results.json — ★★★ batch=128 M-sweep
- codex_coding/results/heteval128_results.json

### 本轮命令
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python nsys_c5_c10_profiling.py --config c5/c10
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python test_c10_v2.py ~ test_c10_v6.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python collect_smask_stability.py --q-major 0.95/1.0
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python test_delay_simulation.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python test_c10_overlap.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python test_m_skip_sweep.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python test_heteval128.py

## 2026-04-19 ~ 2026-04-20 (v0.1.15.8l~m)

- **batch=128 组件级 Profiling + Expert 使用模式分析 + mask_chaos 探索 + MAX_ROUNDS 修正**

### v0.1.15.8l: batch=128 组件级 Profiling

- 新增 nsys_b128_profiling.py，C5 vs C10-M5, gen_length=64
- C5 组件分布: fused_experts 43.2%, Attention 29.4%, LMHead 9.1%, gate_getlogits 7.0%
- C10-M5: fused_experts -187.8ms (-12.1%), EB 开销 84.6ms → 净收益 103.2ms
- EB 子 kernel: cold_batchadd 42.5ms (50.2%), hot_update 23.8ms, cold_K_A 6.5ms
- 关键发现: batch=128 下 Attention (29.4%) 和 LMHead (9.1%) 占比显著增大（batch=32 时分别 20.9% 和 2.8%）

### v0.1.15.8l: Expert 使用模式采集

- 新增 collect_expert_usage.py, batch=128, M=5, gen_length=256
- M=5 窗口内: expert set Jaccard=1.0（完全不变）, per-expert count corr>0.998
- token-expert top1_overlap: 0.63→0.97（随 block 收敛）
- unique/|S_mask| = 99.8%（候选 expert 几乎全被使用）
- cross_window_jaccard = 0.686（冷启动 31% expert 更换）

### v0.1.15.8l: Weight Compaction 验证（已排除）

- 新增 bench_weight_compaction.py
- Compact [158] 比 Original [256] 慢 +80%
- 根因: vllm 只有 E=256 auto-tuned kernel config, 非标准 E 回退 default
- S_mask 在 E=256 空间内已足够高效

### v0.1.15.8l: D1 Token Skip → Expert Elimination（已排除）

- 新增 collect_d1_expert_elimination.py
- 即使跳过 63% token，平均只消除 1 个 expert (0.6% HBM saving)
- batch=128 下 token coverage 太密集

### v0.1.15.8m: MAX_ROUNDS 修正（★ 已合入 test_fused_eb_triton.py）

- MAX_ROUNDS: 16 → 27（40 + 27×8 = 256，理论完备）
- K_init: 改为 actual_s = int(b['s_mask'].sum().item())
- K_C kernel 已有 sat_flag 提前退出（无 CPU sync）
- 诊断（diag_smask_rounds.py）: avg 17.2 轮, max=23, 68.4% 冷启动需要 >16 轮
- |S| avg=168, L0/L1 需要 200+, L17 最少 ~151
- C10-M5 性能验证: 12.02s, -3.3% vs C5（和旧版一致）

### v0.1.15.8m: Routing 分散度分析

- 新增 collect_routing_dispersion.py
- 浅层(L0-L11): 低置信 token max_sigmoid 低于高置信 (差 0.15-0.40)
- 深层(L15-L18): **反转！** 低置信 token max_sigmoid 反而更高
- 解读: 深层对不确定 token routing 更集中（尝试"搞清楚"token）

### v0.1.15.8m: mask_chaos 计划（★★★ 重要负面结论，方向已关闭）

- 发现 vllm fused_experts 使用 **inplace=True**
  - forward_impl 原地修改 hs_flat
  - 之前 torch.where 实验的 "5/5 PASS" 全部是假阳性
  - skip token 实际得到 routed_output × rsf，不是 original_hs × rsf
- 修正后验证（test_mask_chaos_skip_v3.py）: identity skip 在所有 threshold 下崩溃
- Random routing (test_mask_chaos_routing.py): conf<0.3 勉强 4/5, conf≥0.5 退化
- Fixed routing (top-8 popular): 和 random 类似
- Top-1 concentration (test_mask_chaos_top1.py): 2/5（unique expert 减少 10.5% 但质量差）
- 两级 skip (test_two_level_skip.py): Tier2 增量仅 1.6%-4.4%
- **结论: 低置信 token 必须经过 expert 非线性计算，不可跳过/简化。方向关闭。**

### v0.1.15.8m: 给算法专家的分析文档

- 新增 docx/articles/2026-04-19_moe_expert_usage_pattern_and_static_acceleration.md
- 包含: 系统概况、profiling 数据、expert 使用模式、已排除方向、开放问题
- 等待算法专家反馈

### 当前最优配置

| Config | batch | Time(s) | Fwd | ms/fwd | vs C5 |
|--------|------:|------:|----:|------:|------:|
| C5 | 128 | 12.42 | 278 | 44.69 | — |
| **C10-M5** | **128** | **12.02** | **280** | **42.91** | **-3.3%** |

### 新增关键文件

- codex_coding/src/nsys_b128_profiling.py — batch=128 组件级 profiling
- codex_coding/src/collect_expert_usage.py — expert 使用模式
- codex_coding/src/collect_d1_expert_elimination.py — D1 expert elimination
- codex_coding/src/bench_weight_compaction.py — weight compaction micro-bench
- codex_coding/src/collect_routing_dispersion.py — routing 分散度
- codex_coding/src/diag_smask_rounds.py — S_mask 轮次诊断
- codex_coding/src/test_mask_chaos_skip_v3.py — identity skip 修正版
- codex_coding/src/test_mask_chaos_routing.py — random/fixed routing
- codex_coding/src/test_mask_chaos_top1.py — top-1 concentration
- codex_coding/src/test_two_level_skip.py — 两级 skip
- codex_coding/src/dump_c10_outputs.py — 128 条输出 dump
- docx/articles/2026-04-19_moe_expert_usage_pattern_and_static_acceleration.md — ★★★ 算法专家文档
- code_building/process_docs/v0.1-init-project/v0.1.15.8m-mask_chaos_and_kernel_fixes.md — 本轮过程文档

### 本轮命令

- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python nsys_b128_profiling.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python collect_expert_usage.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python collect_d1_expert_elimination.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python bench_weight_compaction.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python collect_routing_dispersion.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python diag_smask_rounds.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python test_mask_chaos_skip_v2.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python test_mask_chaos_skip_v3.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python test_mask_chaos_routing.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python test_mask_chaos_top1.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python test_two_level_skip.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python test_heteval128.py
- CUDA_VISIBLE_DEVICES=4 conda run -n dllm python dump_c10_outputs.py

## 2026-04-20 ~ 2026-04-21 (v0.1.15.8n)

- **ncu 深度 Profiling + Tiling Config 实验 + 8-GPU EP 验证**

### v0.1.15.8n: ncu Profiling (两轮)

- GEMM-1: BW=2307 GB/s (68.9%), GEMM-2: BW=1851 GB/s (55.3%), Occupancy=12.5%
- S_mask 效果验证：ncu bytes 和理论 avg=165.6 吻合
- GEMM-2 pipeline 效率低：K=512 → 62.5% (vs GEMM-1 87.5%)

### v0.1.15.8n: Tiling Config 实验（5 组全部失败，方向关闭）

- vllm auto-tuned (128,256,64,w8,s4) 是此 workload 最优

### v0.1.15.8n: 8-GPU EP 首次验证

- C5 8-GPU: 11.14s (-11.6%), C10-M5 8-GPU: 11.43s (-5.9%)
- E=32 缺 auto-tuned config → 需要 AllToAll EP 适配（下一阶段）

### v0.1.15.8n: M=5 Skip 行为验证

- 1368 次 skip hash 全部一致（0 failures），质量通过

### 待执行：补全 ncu 深度诊断（coalescing, bank conflict, stall reason）

## 2026-04-22 (v0.1.15.8n 续)

- **ncu 深度诊断完成 + Routing TopK 压缩实验 + 原版 dInfer 对比**

### v0.1.15.8n: ncu 深度诊断（3 轮补充采集）

- Bank conflicts: **零**（Triton swizzle 完美）
- Coalescing: 15.08 / 12.67 sec/req → vec-8 bf16 loads，**正常**（非瓶颈）
- Warp stall 主导因素: **pipeline sync (barrier 24% + wait 14% = 38%)** > HBM latency (long_scoreboard 26%) > occupancy (12.5%, 2 warps/scheduler)
- 30% BW gap 根因: ~18% pipeline sync + ~12% HBM latency hiding 不足 + ~5% throttle + ~3% GEMM-2 bubble
- **结论: fused_moe_kernel 内部已接近 vllm auto-tuned 最优点，后续优化转向 kernel 外部**
- 注意: `smsp__average_warp_latency_issue_stalled_*.pct` 对 Triton JIT kernel 返回 0，需用 `smsp__warps_issue_stalled_*.avg` (raw counter)

### v0.1.15.8n: Routing TopK 压缩实验

- C10-M5-K4 (batch=128): 11.28s, 282 fwd, 40.00 ms/fwd, **-8.5% vs C10-M5-K8**
- Forward count 完全不变（282），topk=4 不影响 threshold 收敛
- C10-M5-K4 (batch=64): 8.88s, 277 fwd, 32.06 ms/fwd, -0.9% vs K8（batch 小时增量提升有限）

### v0.1.15.8n: 原版 dInfer 公平对比

- 原版 dInfer (batch=64): 15.51s, 278 fwd, **55.80 ms/fwd**
- C10-M5-K4 (batch=64): 8.88s, 277 fwd, **32.06 ms/fwd**, **-42.5%**
- C10-M5-K4 (batch=128): 11.28s, 282 fwd, **40.00 ms/fwd**
- 吞吐量提升: 4.13 → 11.35 prompts/s (**2.75x**)

### 本轮命令

- sudo /usr/local/cuda-13.0/bin/ncu --metrics coalescing,bank_conflicts,stall (3 轮深度采集)
- CUDA_VISIBLE_DEVICES=4 python bench_routing_topk.py (topk 实验, batch=128 和 64)
- PYTHONPATH=baseline_dInfer/python CUDA_VISIBLE_DEVICES=4 python bench_baseline_dinfer.py

## 2026-04-23 ~ 2026-04-24 (v0.1.15.9)

- **dLLM DP + AllToAll EP 适配（Phase 1: Naive 后端）**

### v0.1.15.9: AllReduce EP → AllToAll EP

- 将 dInfer 多卡推理从 AllReduce EP (dp=1, tp=N) 升级到 DP AllToAll EP (dp=N, tp=1)
- 这是全新工程方向——dLLM block diffusion + AllToAll EP，vllm/sglang 均未实现

### 关键发现

- **dInfer 的 Attention 不做 TP 分片**：`LLaDA2MoeAttention.tp_size=1` 硬编码，用原生 `nn.Linear`
  - AllReduce EP (tp=4) 下 4 卡各自独立算相同 128 prompts 的完整 attention = **3/4 纯冗余**
  - DP AllToAll (dp=4) 下各卡算不同 32 prompts = **零冗余**
- **dLLM + AllToAll EP 的 forward 对齐问题是全新挑战**：
  - AR 推理每步 1 次 forward，调度器保证对齐
  - dLLM 的 3 层嵌套循环（block 内/block 间/prefill）forward 次数由数据决定
  - 不同 DP rank 处理不同数据 → forward 次数不对齐 → AllToAll collective 死锁
  - vllm 没有 dLLM，sglang 的 dLLM 不支持 AllToAll EP → **无先例可参考**

### 解决的 5 个技术问题

1. **vllm DP 初始化端口调整**：两阶段 init（dp=1 初始化 torch.distributed → dp=N 创建 groups）
2. **ForwardContext 缺失**：在 _moe_forward_with_context() 中注入 DPMetadata
3. **三层 forward 对齐**：unroll=1 + all_reduce(MAX) + early_stop all_reduce(MIN)
4. **★★★ 输出乱码根因**：`modeling_fused_olmoe.py` line 51 无条件 monkey-patch tensor_model_parallel_all_reduce 为 world group all_reduce → combine 后再次 all_reduce → 输出 × dp_size
5. **NCCL init hang**：GPU 0 被占用时 NCCL P2P 探测阻塞（系统环境限制，非代码问题）

### 4 卡性能结果（batch=128, gen=256, HetEval-128）

| Config | Time(s) | Fwd | ms/fwd | vs 1-GPU | 内存/卡 |
|--------|---------|-----|--------|----------|---------|
| C5 1-GPU | 12.42 | 278 | 44.69 | — | ~15GB |
| C5 AllReduce 4-GPU | 11.59 | 281 | 41.24 | -7.7% | ~15GB |
| C5 DP AllToAll 4-GPU | 15.72 | 283 | 55.55 | +24.3% | **9.7GB** |

- Naive AllToAll 通信开销太大（broadcast × dp_size），总时间比单卡还慢
- 但内存节省 35%，框架正确性已验证（质量 5/5 PASS）
- 需要 DeepEP/pplx 后端才能体现 AllToAll 性能优势

### 修改的文件

- modeling_llada2_moe.py: monkey-patch 条件化 + EP rank + ForwardContext
- modeling_fused_olmoe.py: ★★★ 去掉无条件 all_reduce patch（质量 bug 根因）
- generate_uniform.py: 三层 forward 对齐同步 + unroll=1
- parallel_strategy.py: broadcast_if_needed TP group fix
- bench_multi_gpu_dp.py (新建): DP AllToAll EP 主测试脚本

### 本轮命令

- CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 bench_multi_gpu_dp.py --batch-size 128 --gen-length 256
- CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 bench_multi_gpu.py (AllReduce EP 对照)
- 多个诊断脚本（diag_prefill_logits.py, _diag_dp_groups.py 等）

## 2026-04-24 ~ 2026-04-25 (v0.1.15.10)

- **DeepEP 安装验证 + V1 HT 质量修复 + V2 ElasticBuffer 探索 + V1 优化尝试 + Attention TP 方向确认**

### v0.1.15.10: DeepEP V1/V2 集成与多卡通信优化探索

### DeepEP 安装

- 安装 DeepEP V2 (v2.0.0) 从源码编译，依赖：NCCL 2.30.4 + NVSHMEM 3.3.20
- 升级 pip nvidia-nccl-cu12 从 2.27.3 → 2.30.4
- git submodule update --init (fmt 子模块)

### DeepEP V1 HT via vllm 质量修复（★★★ 最关键的 bug fix）

- **根因**：`bench_multi_gpu_dp.py` 缺少 `prepare_communication_buffer_for_model(model)` 调用
  → DeepEP 的 FusedMoEModularKernel 未初始化 → MoE 层完全无跨卡通信 → 输出空/乱
- **修复**：模型加载后加一行：
  ```python
  from vllm.distributed import prepare_communication_buffer_for_model
  prepare_communication_buffer_for_model(model)
  ```
- 修复后质量 5/5 PASS，逐层 norm 精确匹配 naive

### DeepEP V1 HT 真实性能（修复后）

| Config | Time(s) | Fwd | ms/fwd | vs 单卡 | vs naive |
|--------|---------|-----|--------|---------|----------|
| C5 1-GPU | 12.42 | 278 | 44.69 | — | — |
| C5 AllReduce 4-GPU | 11.59 | 281 | 41.24 | -7.7% | — |
| C5 DP naive 4-GPU | 15.72 | 283 | 55.55 | +24.3% | — |
| C5 DP DeepEP V1 HT 4-GPU | **14.80** | **284** | **52.12** | +16.6% | -5.8% |

- DeepEP V1 HT 比 naive 快 5.8%，但仍比单卡慢 16.6%，比 AllReduce 慢 27%
- 原因：V1 每层 MoE 需 get_dispatch_layout (CPU-GPU sync) + dispatch + combine = ~2ms/层 × 19 层 = ~38ms 通信

### DeepEP V2 ElasticBuffer 探索

- **2 卡 V2 成功**：逐层 norm 精确匹配 naive，num_sms=22（EP=2）
- **4 卡 V2 阻塞**：NCCL GIN 不可用
  - 原因：GPU 驱动 580.65.06 的 NCCL 不支持 GIN（ginType=NONE）
  - IMEX daemon 无法启动（IMEX 580.126 vs 驱动 580.65 版本不匹配）
  - 修改了 DeepEP nccl.cu:87 的无条件 GIN assert 为条件判断
  - EP_DISABLE_GIN=1 可跳过 GIN，但 4 卡 generate 循环崩溃（单次 forward 正确）
- **结论**：V2 在当前系统上 4 卡不可用，需升级驱动到 580.126+

### V1 优化尝试（借鉴 V2 思想）

| 优化 | 结果 | 原因 |
|------|------|------|
| async combine | 无效 (14.97 vs 14.80) | 瓶颈在 GPU 通信时间，不在 CPU 阻塞 |
| 减少 SM (20→16) | 无效 (14.92) | 通信 kernel 需要足够 SM 驱动 NVLink |
| 减少 SM (20→10) | 更慢 (15.28) | SM 不足导致通信更慢 |
| async layout | 失败 | V1 C++ assert 阻止 previous_event 链接 |

- **V1 HT 在当前系统的 14.80s 就是性能天花板**

### pplx 后端调查

- pplx_kernels 在 PyPI 不存在，是 Perplexity 内部私有包
- 最新 vllm 已移除 pplx 后端（回退到 allgather_reducescatter）
- 此路不通

### AllReduce vs AllToAll 最终对比

| | AllReduce 4 卡 | DeepEP AllToAll 4 卡 |
|---|---|---|
| 性能 | **11.59s** (快 22%) | 14.80s |
| 内存 | ~15 GB | **~9.7 GB** (省 35%) |
| Attention | 4x 冗余 | 零冗余 |
| 通信 | 1× AllReduce/层 (~18ms) | 2× dispatch+combine/层 (~38ms) |
| 适合场景 | 当前最优 | 大模型/多节点/内存受限 |

### ★★★ 下一步方向确认：Attention TP 化

**核心发现**：vllm 的最优 MoE 配置是 tp=N + ep=N（同组 GPU），Attention 用 TP 分 heads，MoE 用 EP 分 experts。dInfer 的 attention 不支持 TP（tp_size=1 硬编码，nn.Linear）是性能瓶颈的根本原因。

**目标配置（tp=4, ep=4, dp=1）**：
- Attention: TP=4，每卡处理 1/4 heads → 计算 1/4 + 小量 AllReduce
- MoE: EP=4 + AllReduce → 计算 1/4 + AllReduce
- 预期 ~34 ms/fwd（vs 当前 AllReduce 41.24）

**改动范围**：
- Q/K/V 投影：nn.Linear → QKVParallelLinear/ColumnParallelLinear
- O 投影：nn.Linear → RowParallelLinear（自带 TP AllReduce）
- 去掉 tp_size=1 硬编码
- num_heads/num_kv_heads 按 tp_size 切分
- Weight loading 适配 TP shard

### 修改/新建的文件

- bench_multi_gpu_dp.py: 添加 prepare_communication_buffer_for_model + V2/V1_OPT 支持
- deepep_v2_pf.py (新建): V2 ElasticBuffer PrepareAndFinalize
- deepep_v1_optimized_pf.py (新建): V1 async combine + SM 可调
- diag_deepep_quality.py (新建): 逐层 norm 对比诊断
- diag_deepep_independent_quality.py (新建): 独立质量评估
- diag_deepep_raw_tokens.py (新建): 原始 token 分析
- DeepEP/csrc/kernels/backend/nccl.cu: GIN assert 改为条件判断

### 本轮命令

- CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_ALL2ALL_BACKEND=deepep_high_throughput torchrun --nproc_per_node=4 bench_multi_gpu_dp.py
- CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 diag_deepep_quality.py (多种 backend 对比)
- CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 diag_deepep_independent_quality.py
- USE_DEEPEP_V2=1 EP_DISABLE_GIN=1 ... (V2 ElasticBuffer 测试)
- USE_V1_OPT=1 V1_NUM_SMS=10/16 ... (V1 优化测试)

## 2026-04-25 ~ 2026-04-26 (v0.1.15.11)

- **C11: TP Attention + C12: dp=2 AllToAll EP + Insight 驱动优化**

### C11: TP Attention 实现 (tp=4 ep=4 dp=1, 4-GPU)

- nn.Linear → QKVParallelLinear + RowParallelLinear（参考 Qwen3MoeAttention per-head QK norm）
- LMHead → ColumnParallelLinear(gather_output=True)
- baseline_optimizations.py: 删 //tp_size + tuple 解包
- HetEval-512 batch=512: C11-M5-K4 AllReduce = 104.35 ms/fwd, C11-M5-K4 AGR = 104.39 ms/fwd
- AGR ≈ AllReduce（dp=1 时 dispatch 被跳过）

### C12: dp=2 tp=4 ep=8 (8-GPU 真正 AllToAll)

- TP Group 0: GPU 0-3, TP Group 1: GPU 4-7, EP: 全 8 GPU (32 experts/GPU)
- C12 naive: 74.76 ms/fwd, 25.7 p/s
- C12 + popularity placement: 73.50 ms/fwd (-1.7%)
- C12 DeepEP V1 HT: 133 ms/fwd (弃用)

### Insight 验证 (E1-E5)

- 3 datasets × 2 batch sizes × 19 layers 完整验证矩阵
- A: S_mask M-forward 静态（Jaccard=1.0 within skip window）✓
- B: Block-start routing 集中（9/19 层显著，C12 MoE 慢 45%）✓
- C: Expert popularity data-dependent（跨 dataset 0/19 一致）✓
- D: S_mask 通信包络（ratio 1.04-1.68）✓
- 否定: mask routing 非模型常量（跨 dataset Jaccard=0.19）

### Insight 驱动优化

- 方案 1 (shared expert overlap): 负优化 ✗
- 方案 2 (popularity placement): +1.7% ✓
- 方案 3 (S_mask 通信 schedule 缓存): ★未实现，最有潜力
- 方案 4a (block-start proactive replication): 前提不成立 ✗
- 方案 5 (LMHead TP): -4.3% ✓

### 新增关键文件

- codex_coding/src/bench_dp2_tp4_ep8.py — C12 主测试
- codex_coding/src/bench_dp2_expert_placement.py — popularity placement
- codex_coding/src/bench_dp2_block_start.py — block-start timing
- codex_coding/src/nsys_c11_m5k4_profiling.py — C11 组件 profiling
- codex_coding/src/collect_expert_load_dist.py — expert 负载分布
- codex_coding/src/collect_expert_load_temporal.py — 时序负载
- codex_coding/src/compare_eb_load_dist.py — C5 vs EB 负载对比
- codex_coding/src/validate_insights.py — insight 验证
- codex_coding/src/test_heteval512.py — HetEval-512 prompt 集
- code_building/process_docs/v0.1-init-project/v0.1.15.11-c11_c12_tp_attention_and_insight_optimization.md

### 本轮命令

- CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 bench_multi_gpu.py --batch-size 512 (C11 各配置)
- CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter torchrun --nproc_per_node=4 bench_multi_gpu.py --batch-size 512
- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 bench_dp2_tp4_ep8.py --batch-size 512
- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 bench_dp2_expert_placement.py
- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 bench_dp2_block_start.py
- CUDA_VISIBLE_DEVICES=4 python validate_insights.py
- CUDA_VISIBLE_DEVICES=4 python collect_expert_load_dist.py
- CUDA_VISIBLE_DEVICES=4 python collect_expert_load_temporal.py
- CUDA_VISIBLE_DEVICES=4 python compare_eb_load_dist.py
- CUDA_VISIBLE_DEVICES=4 python nsys_c11_m5k4_profiling.py

## 2026-04-26 (v0.1.15.12)

- **方案 3: S_mask 通信 Schedule 缓存 — Route-Before-Dispatch 实现**

### v0.1.15.12a: C12 组件级 Profiling

- 新增 `nsys_c12_profiling.py`，对 C12 (dp=2 tp=4 ep=8) 做组件级计时
- 精确测量 dispatch/combine 通信占比：
  - Dispatch (naive_multicast): 550.2 ms (9.8% of wall)
  - Combine (all_reduce+slice): 446.7 ms (7.9% of wall)
  - **D+C 合计: 996.9 ms, 17.7% of wall, 13.47 ms/fwd, 0.709 ms/layer/fwd**
- 关键发现：通信占比 17.7%，远超预估的 <5%
- 原因：vllm 0.10.2 的 NaiveAll2AllManager 使用逐 rank broadcast（非 all_gather），效率低
- MoE total (hot): 2613.9 ms (46.3%), non-MoE: 2331.2 ms (41.3%)

### v0.1.15.12b: SMaskAware Route-Before-Dispatch 实现

- 新增 `bench_smask_aware_dp2.py`，实现 S_mask-aware 通信优化
- **核心改变**：在 hot path 中先本地 routing (256 tokens, cached S_mask)，再 dispatch routing 结果
  - Baseline: dispatch(hidden[N×2048] + router_logits[N×256]) → route on 512 tokens
  - SMaskAware: route on 256 tokens → dispatch(hidden[N×2048] + topk_ids[N×4] + topk_weights[N×4])
- 收益来源：
  1. dispatch 通信量减少 10.6% (skip router_logits broadcast)
  2. routing 计算量减半 (256 vs 512 tokens)
  3. 跳过 FusedMoE 内部 routing，直接调用 fused_experts
- **实现**：monkey-patch SparseMoeBlock.forward，hot path 直接调用 naive_multicast + fused_experts

### v0.1.15.12b: 性能验证

| Config | batch=512 gen=256 | ms/fwd | Fwd | vs Baseline |
|--------|-------------------|--------|-----|-------------|
| C12 baseline | 19.932s | 74.93 | 266 | — |
| **C12 SMaskAware** | **19.213s** | **72.23** | **266** | **-3.6%** |

- Forward 次数完全一致 (266)，S_mask 缓存不影响质量收敛
- 质量正常，输出文本语义正确
- 每 forward 节省 2.70 ms，与 profiling 预估 (~2-3 ms) 吻合
- 论文贡献：dLLM 独有特性 (S_mask 静态) 被系统利用，AR MoE 无法复现

### Level 2 (Selective Token Dispatch) 分析

- 分析表明 Level 2 额外收益有限：
  - fused_experts kernel 是 weight-bound，减少 token 不减少 weight HBM loading
  - NVLink 带宽充裕，dp=2 通信量 ~1.2MB/层
  - Level 2 复杂度 (~500行) 不值得 <0.5 ms/fwd 额外收益
- 结论：Level 1 已是该方向最优实现

### 新增关键文件

- codex_coding/src/nsys_c12_profiling.py — C12 组件级 profiling
- codex_coding/src/bench_smask_aware_dp2.py — ★ S_mask-aware route-before-dispatch
- codex_coding/results/c12_component_profiling.json — profiling 数据
- codex_coding/results/smask_aware_dp2_results.json — 性能对比数据

### 本轮命令

- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=naive torchrun --nproc_per_node=8 nsys_c12_profiling.py
- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=naive torchrun --nproc_per_node=8 bench_smask_aware_dp2.py --batch-size 512 --gen-length 256 --mode both

### v0.1.15.12c: vllm 升级 0.10.2 → 0.11.0

- pip install vllm==0.11.0（torch==2.8.0 兼容确认）
- 新增 AgRsAll2AllManager: dispatch 用 all_gatherv（1 次 NCCL），combine 用 reduce_scatterv（1 次）
- 旧 NaiveAll2AllManager: dispatch 用逐 rank broadcast（4 次），combine 用 all_reduce + slice（1 次）
- 每层 MoE NCCL 调用从 6 次降到 3 次（含 TP all_reduce）
- 需要 `sys.modules['deep_ep'] = None` 屏蔽 DeepEP NCCL 符号不兼容
- 缺少 E=32,N=512 auto-tuned MoE kernel config（WARNING 但不影响正确性）
- **C12-AgRs baseline: 75.60 ms/fwd**（与旧 naive 74.76 基本持平）

### v0.1.15.12d: 方案 3 重写（基于 vllm 0.11.0 AgRs 后端）

- 新建 bench_scheme3_dp2.py，使用 vllm 0.11.0 的 dp_group.all_gatherv + reduce_scatterv
- all_gatherv 接口测试通过：支持 3 tensor 异构 list + int32 dtype
- 需要 ForwardContext 中 dp_metadata.sp_local_sizes(sp_size) 上下文
- 单层 MoE forward 测试通过（6 次 forward, cold/skip 切换正确, 内存稳定 5.6GB）
- **generate() 循环极慢/卡住** — 根因分析中

### v0.1.15.12d: 卡住问题分析

可能根因：
1. skip path 的 set_forward_context 内部 DPMetadata.make 每次做 all_reduce → 每层额外 1 次 NCCL
2. 或 E=32 缺少 auto-tuned config 导致 fused_experts 极慢
3. 或 cold/skip path 之间 NCCL collective 序列不一致

解决方向：
- a) 预缓存 dp_metadata（cold path 记录 sizes, skip path 复用）
- b) 绕过 set_forward_context，直接用 dp_group.all_gatherv 手动传 sizes
- c) 检查 E=32 kernel config 影响

### v0.1.15.12e: Sequence Parallelism 分析

- 发现 C12 TP group 内 4 GPU 在 gate+routing+shared 存在 3/4 冗余
- SP 可消除：Attention AllReduce → ReduceScatter + AllGather
- 预估收益 ~9%，比方案 3 route-before-dispatch (~2-3%) 更大
- 和方案 3 正交可叠加
- 实现复杂度高（需改 attention 输出 + MoE 输入/输出 + dispatch group）

### 新增关键文件（本轮追加）

- codex_coding/src/bench_scheme3_dp2.py — ★ 方案 3 v2 主测试（vllm 0.11.0 AgRs 后端）
- codex_coding/results/dp2_tp4_ep8_benchmark.json — C12-AgRs baseline: 75.60 ms/fwd
- code_building/process_docs/v0.1-init-project/v0.1.15.12-vllm_upgrade_and_scheme3_design.md — 本轮完整过程

### 本轮命令（追加）

- pip install vllm==0.11.0
- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter torchrun --nproc_per_node=8 bench_dp2_tp4_ep8.py --batch-size 512 --gen-length 256
- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter torchrun --nproc_per_node=8 bench_scheme3_dp2.py --batch-size 512 --gen-length 64

### v0.1.15.12f: Scheme3 native-topk monkey-patch 实验

- 在 `codex_coding/src/bench_scheme3_dp2.py` 中新增 `B2) C12-Scheme3-native-topk`
- 实现方式：
  - `SparseMoeBlock.forward` 只在 skip path 计算 local `topk_w/topk_ids`
  - monkey-patch 单个 `FusedMoE.forward_impl`，在 DP dispatch 阶段传 `hidden_states + compact_topk_payload`
  - monkey-patch `FusedMoE.select_experts`，识别 compact payload 后直接返回预计算 topk，跳过 routing 重算
  - cold / hot-update 保持原生 `router_logits` 路径
- 修复两个实验问题：
  - 显式包 `set_forward_context`，避免 `_moe_forward_with_context` 的宽泛 `except` 吞掉真实错误后无 context fallback
  - compact payload 解包按 payload 宽度恢复实际 topk，避免 vLLM config `top_k` 与实验 `K=4` 不一致导致 shape mismatch

#### 4-GPU fallback 实验：C12-AgRs-g4 / dp=2,tp=2,ep=4,batch=256

| Config | End-to-end | Fwd | ms/fwd | vs A |
|--------|------------|-----|--------|------|
| A) C12-AgRs baseline | 15.74s | 266 | 59.18 | — |
| B) C12-Scheme3 blockpatch | 16.73s | 266 | 62.88 | +6.3% |
| B2) C12-Scheme3-native-topk | 17.31s | 266 | 65.06 | +9.9% |

路径计数一致：

| Path | Count |
|------|-------|
| prefill_fallback | 19 |
| cold | 171 |
| hot_skip | 3933 |
| hot_update | 931 |
| guard_fallback | 0 |

质量肉眼检查：

| Prompt | Observation |
|--------|-------------|
| #0 average speed | A/B/B2 输出一致，推导过程正确，片段截断在最终公式前 |
| #8 quadratic | A/B/B2 输出一致，有重复啰嗦，但方向正确 |
| #13 logic | A/B/B2 输出一致，逻辑推导基本正确 |
| #19 Fibonacci | A/B/B2 输出一致，先出现错误代码片段，随后给出正确 iterative 版本 |
| #28 planets | A/B/B2 输出一致，顺序基本正确，Jupiter 类型表述不严谨 |

结论：

- B2 native-topk 路径功能跑通，且没有观察到 Scheme3 特有质量退化
- 但 B2 性能仍为负收益，比 A 慢 +9.9%，比旧 B 也慢
- 这说明当前收益没有被 vLLM 原生路径兑现；主要开销很可能来自 Python monkey-patch 包装、compact payload 打包/解包、额外 `select_experts` 绕行，以及 dispatch 输入虽然缩小但整体 MoE kernel/collective 固定开销占主导
- 下一步不应继续扩大 B2 实现，而应做组件级 timing：local routing、pack/unpack、dispatch、quant_method.apply、combine、TP all_reduce，确认通信节省是否被固定开销完全淹没

### 本轮命令（追加）

- CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter torchrun --nproc_per_node=4 codex_coding/src/bench_scheme3_dp2.py --batch-size 256

### v0.1.15.12g: Scheme3 8-GPU 标准配置复验

- 在 8 卡标准 C12 配置上重跑 Scheme3：
  - `dp=2,tp=4,ep=8`
  - `batch_size=512`
  - `gen_length=256`
  - `block_length=32`
  - `VLLM_ALL2ALL_BACKEND=allgather_reducescatter`
- 先跑组件 timing 并保存到 `codex_coding/results/scheme3_dp2_component_timing_8g.json`
- 再跑无 instrumentation 的端到端实验并保存到 `codex_coding/results/scheme3_dp2_results.json`
- 保存完整终端日志：
  - `codex_coding/results/scheme3_dp2_8g_component_timing_20260427_034930.log`
  - `codex_coding/results/scheme3_dp2_8g_e2e_20260427_035313.log`

#### 8-GPU 端到端结果（无 component timing）

| Config | End-to-end | Fwd | ms/fwd | vs A |
|--------|------------|-----|--------|------|
| A) C12-AgRs baseline | 20.12s | 266 | 75.62 | — |
| B) C12-Scheme3 route-before-dispatch | 21.18s | 266 | 79.61 | +5.3% |
| B2) C12-Scheme3-native-topk | 21.64s | 266 | 81.35 | +7.6% |

路径计数完全一致：

| Path | Count |
|------|-------|
| prefill_fallback | 19 |
| cold | 171 |
| hot_skip | 3933 |
| hot_update | 931 |
| guard_fallback | 0 |

#### 8-GPU 组件 timing 结果

| Component | A ms/fwd | B ms/fwd | B2 ms/fwd |
|-----------|---------:|---------:|----------:|
| shared | n/a | 2.024 | 2.029 |
| gate_logits | n/a | 4.900 | 4.905 |
| skip_decision | n/a | 5.354 | 5.422 |
| get_s_mask | n/a | 0.341 | 0.343 |
| local_routing | n/a | 2.628 | 2.615 |
| pack_topk | n/a | n/a | 0.858 |
| dispatch | 5.698 | 5.250 | 5.175 |
| select_experts | n/a | n/a | 3.407 |
| unpack_topk | n/a | 0.123 | 0.177 |
| fused_experts | n/a | 12.407 | n/a |
| quant_apply | 20.638 | 6.795 | 21.802 |
| combine | 3.751 | 3.541 | 5.026 |
| TP all_reduce | 4.648 | 4.675 | 6.028 |
| native_forward | n/a | 11.192 | 39.875 |

Dispatch payload:

| Config | MB/fwd | vs A |
|--------|------:|-----:|
| A) C12-AgRs baseline | 826.877 | — |
| B) C12-Scheme3 route-before-dispatch | 705.753 | -14.6% |
| B2) C12-Scheme3-native-topk | 706.721 | -14.5% |

#### 质量肉眼检查

| Prompt | Observation |
|--------|-------------|
| #0 average speed | A/B/B2 可见推导正确：总路程 480km，分段时间 3h/4h；部分日志片段在最终平均速度前截断 |
| #8 quadratic | A/B/B2 方向正确，能找到 -2/-3 因子，对应根 2/3；输出偏啰嗦但可接受 |
| #13 logic | A/B/B2 正确推出 B、C 为真；有轻微文字噪声，D 的后续讨论在日志片段中截断 |
| #19 Fibonacci | A/B/B2 iterative 代码可接受，前几项可见结果正确；后续列表被截断 |
| #28 planets | A/B/B2 行星顺序和可见类型大体正确；最新日志中 Uranus 周期写成 54.0 年是错误，Neptune 行被截断；该问题三组共享，不是 Scheme3 特有退化 |

结论：

- Scheme3 期望的 routing-logits dispatch payload 节省确实发生：约 `826.9 -> 706 MB/fwd`
- 但 wall-clock dispatch 只节省约 `0.45-0.52 ms/fwd`
- 新增 `gate_logits + skip_decision + local_routing` 等开销明显大于通信节省
- B2 虽复用 native-topk 路径，但 `pack_topk/select_experts/native_forward/combine/TP all_reduce` 开销更高，端到端更慢
- 当前 C12-AgRs 下不应继续扩大 B/B2 monkey-patch 路径；若继续 Scheme3，应压低 Python/per-layer overhead，否则优先转向 Sequence Parallelism

### 新增关键文件（本轮追加）

- code_building/process_docs/v0.1-init-project/v0.1.15.12g-scheme3_8g_timing_and_quality.md — 8 卡标准配置 timing、质量检查与结论
- codex_coding/results/scheme3_dp2_component_timing_8g.json — 8 卡组件 timing JSON
- codex_coding/results/scheme3_dp2_results.json — 8 卡无 instrumentation 端到端 JSON
- codex_coding/results/scheme3_dp2_8g_component_timing_20260427_034930.log — 8 卡组件 timing stdout
- codex_coding/results/scheme3_dp2_8g_e2e_20260427_035313.log — 8 卡端到端 stdout

### 本轮命令（追加）

- nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter torchrun --nproc_per_node=8 codex_coding/src/bench_scheme3_dp2.py --batch-size 512 --gen-length 256 --num-runs 1 --component-timing
- cp codex_coding/results/scheme3_dp2_results.json codex_coding/results/scheme3_dp2_component_timing_8g.json
- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter torchrun --nproc_per_node=8 codex_coding/src/bench_scheme3_dp2.py --batch-size 512 --gen-length 256 --num-runs 2

### v0.1.15.12h: Insight Ledger 活文档归档

- 新增 `docx/context_index/04_insight_ledger.md`，作为 dLLM / MoE / EB 方向选择前的 canonical insight ledger。
- 该 ledger 归纳当前 18 条 insight，包括：
  - forward count 是 dLLM 最大杠杆；
  - MASK routing 集中；
  - stable MoE output 直接复用失败；
  - S_mask block 内稳定；
  - EB 收益随 batch 增大；
  - fused MoE 受 expert weight HBM loading 限制；
  - Scheme3 routing-logits 通信节省真实但 standalone 优先级降低；
  - TP/parallelism 结构优化是高价值方向。
- 在 `docx/context_index/00_document_registry.md` 登记 `04_insight_ledger.md` 为 high priority living index。
- 在 `code_building/key_conclusion.md` 顶部新增 ledger 入口说明，要求方向选择前优先查看该文件。
- 新增过程归档 `code_building/process_docs/v0.1-init-project/v0.1.15.12h-insight_ledger_archive.md`。

### 新增关键文件（本轮追加）

- docx/context_index/04_insight_ledger.md — dLLM / MoE / EB insight 活文档
- code_building/process_docs/v0.1-init-project/v0.1.15.12h-insight_ledger_archive.md — 本次 insight ledger 归档过程记录

### 本轮命令（追加）

- sed -n '1,220p' /home/wuhang/.codex/suss-skills/building-rules/SKILL.md
- sed -n '1,260p' docx/context_index/00_document_registry.md
- tail -n 80 code_building/key_conclusion.md
- tail -n 120 code_building/progress_diff_summary.md
- ls -1 docx/context_index

### v0.1.15.12i: 上下文压缩恢复交接归档

- 覆盖写入 `history-chat.txt`，按照用户指定模板生成本轮上下文恢复说明。
- 恢复说明重点记录：
  - 当前阶段仍为 `v0.1-init-project`；
  - 当前工作已从早期 dInfer/LLaDA2 baseline 转到 dLLM / MoE / EB insight-led direction selection；
  - Scheme3 C12-AgRs 8 卡标准实验已经完成；
  - A/B/B2 端到端结果分别为 `75.62 / 79.61 / 81.35 ms/fwd`；
  - Scheme3 routing-logits dispatch payload saving 真实存在，但 standalone wall-clock 收益太小；
  - 五条 verifiable prompt 没有观察到 Scheme3 特有质量退化；
  - 下一轮应优先阅读 `04_insight_ledger.md`，先讨论方向选择，不要直接代码建设。
- 刷新 `docx/context_index/01_stage_summary.md`：
  - 当前阶段更新为 Scheme3 8 卡验证后的方向选择阶段。
- 刷新 `docx/context_index/02_active_threads.md`：
  - 最高优先级线程更新为 insight-led direction selection；
  - Scheme3 standalone 降优先级；
  - Sequence/TP、native active-expert reduction、block-stage-aware scheduler 保持 warm。
- 刷新 `docx/context_index/03_current_required_process_docs.md`：
  - 最小必读过程文档集合更新为当前 Scheme3 / insight ledger 相关文档。
- 刷新 `docx/context_index/00_document_registry.md`：
  - 登记 `history-chat.txt`、当前恢复索引、Scheme3 结果和 insight ledger。
- 新增过程文档 `code_building/process_docs/v0.1-init-project/v0.1.15.12i-context_compression_recovery_handoff.md`。

### 新增关键文件（本轮追加）

- history-chat.txt — 当前上下文压缩恢复说明，下一轮恢复首要入口之一
- docx/context_index/01_stage_summary.md — 当前阶段摘要，已切到 Scheme3 后的方向选择阶段
- docx/context_index/02_active_threads.md — 当前活跃线程，最高优先级为 insight-led direction selection
- docx/context_index/03_current_required_process_docs.md — 当前最小必读过程文档集合
- code_building/process_docs/v0.1-init-project/v0.1.15.12i-context_compression_recovery_handoff.md — 本轮恢复交接归档过程记录

### 本轮命令（追加）

- sed -n '1,220p' /home/wuhang/.codex/suss-skills/archive-progress/SKILL.md
- sed -n '1,220p' /home/wuhang/.codex/suss-skills/building-rules/SKILL.md
- sed -n '1,220p' /home/wuhang/.codex/suss-skills/recovery-handoff/SKILL.md
- sed -n '1,260p' docx/next_step.txt
- sed -n '1,260p' docx/context_index/04_insight_ledger.md
- tail -220 code_building/key_conclusion.md
- tail -260 code_building/progress_diff_summary.md
- sed -n '1,260p' code_building/process_docs/v0.1-init-project/v0.1.15.12g-scheme3_8g_timing_and_quality.md
- sed -n '1,220p' code_building/process_docs/v0.1-init-project/v0.1.15.12h-insight_ledger_archive.md
- sed -n '1,260p' code_building/process_docs/v0.1-init-project/v0.1.15.12-vllm_upgrade_and_scheme3_design.md
- sed -n '1,220p' code_building/process_docs/v0.1-init-project/v0.1.15.11-c11_c12_tp_attention_and_insight_optimization.md
- git status --short

### v0.1.15.12j: BSP-MoE 8-GPU C12 monkey-patch 验证

- 新增独立脚本 `codex_coding/src/bench_bsp_moe_dp2.py`，用于验证 Block Sequence Parallel MoE。
- 未修改 dInfer / vLLM 源码；BSP 只 monkey-patch `LLaDA2MoeSparseMoeBlock.forward`。
- BSP 机制：
  - flatten `[bsz, seq_len, h] -> [N, h]`
  - 用 vLLM `sequence_parallel_chunk` 在 TP group 内切 token
  - 只在 TP-local token shard 上跑 shared expert、gate logits 和 native `FusedMoE.forward_impl`
  - 设置 `experts.is_sequence_parallel=True` 和 `sp_size=tp_size`
  - 用 `tensor_model_parallel_all_gather(..., dim=0)` 收集输出并裁剪 padding
  - routing 仍留在 native FusedMoE 内部，保持 EB/S_mask global semantics

#### C12 shape probe

| Item | Value |
|------|------:|
| batch_size | 512 |
| dp/tp/ep | 2 / 4 / 8 |
| local_bs per DP rank | 256 |
| block_length | 32 |
| block MoE N_dp | 8192 |
| BSP N_sp per TP rank | 2048 |
| padding | 0 |

结论：

- 之前担心的 “每 rank 只有 64 tokens” 忽视了 `block_length=32`。
- 真实 MoE token 数是 `local_bs * block_length = 256 * 32 = 8192`，按 TP=4 后每 rank `2048`，足够验证 BSP。

#### Forward-check

| Layer | Path counts | abs_mean | rel_max |
|------:|-------------|---------:|--------:|
| 0 | baseline/BSP 一致 | `~5.5e-4` | `0.15%-0.67%` |
| 9 | baseline/BSP 一致 | `~8.4e-4~9.0e-4` | `0.29%-0.70%` |
| 18 | baseline/BSP 一致 | `~9.6e-4~1.0e-3` | `0.32%-0.71%` |

结论：

- BSP 功能正确但不是 bitwise identical。
- 长迭代输出允许出现小文本差异，因此质量必须肉眼语义检查。

#### 8-GPU C12 端到端结果（无 component timing）

| Config | End-to-end | Fwd | ms/fwd | vs A |
|--------|-----------:|----:|-------:|-----:|
| A) C12-AgRs baseline | 20.1285s | 266 | 75.675 | — |
| B) C12-BSP-MoE | 19.8475s | 266 | 74.615 | -1.40% |

路径计数一致：

| Path | Count |
|------|------:|
| prefill_fallback | 19 |
| cold | 171 |
| hot_skip | 3933 |
| hot_update | 931 |

#### 8-GPU C12 component timing

| Component | A ms/fwd | B ms/fwd | Delta |
|-----------|---------:|---------:|------:|
| moe.bsp_chunk | n/a | 0.858 | +0.858 |
| moe.shared | 2.042 | 3.376 | +1.334 |
| moe.gate_logits | 4.916 | 2.705 | -2.211 |
| moe.native_forward | 40.343 | 37.796 | -2.547 |
| moe.dispatch | 5.756 | 7.822 | +2.066 |
| moe.quant_apply | 20.510 | 20.105 | -0.405 |
| moe.combine | 3.587 | 8.284 | +4.697 |
| moe.tp_all_reduce | 4.718 | n/a | -4.718 |
| moe.tp_all_gather | n/a | 2.618 | +2.618 |

Payload:

| Payload | A MB/fwd | B MB/fwd | Delta |
|---------|---------:|---------:|------:|
| dispatch_payload | 826.877 | 206.719 | -75.0% |
| tp_gather_payload | n/a | 165.375 | +165.375 |

结论：

- BSP 预期的 dispatch payload 大幅下降真实发生。
- 但当前 combine 和 TP all-gather 开销吃掉大部分收益。
- component timing run 端到端为 A `78.50 ms/fwd`、B `83.60 ms/fwd`，该结果包含 timing overhead，不作为最终速度结论。

#### 质量肉眼检查

| Prompt | Observation |
|--------|-------------|
| #0 average speed | A/B 都可见总路程 480km、分段时间 3h/4h；最终答案在片段中截断，无 BSP 特有错误可见 |
| #8 quadratic | A/B 都正确找到 `-2,-3`，B 可见 `(x-2)(x-3)`；无 BSP 特有退化 |
| #13 logic | A/B 都正确推出 B/C true；D 后续截断 |
| #19 Fibonacci | A 给出正确 iterative；B 先给错误代码后自我修正为正确 iterative，有局部瑕疵但不构成语义崩坏 |
| #28 planets | A 行星顺序基本正确但 Uranus 写成 `54.0` 年错误；B 可见 Uranus `~84` 年正确；无 BSP 特有退化 |

总体结论：

- BSP-MoE 机制成立，且比 Scheme3 standalone 更像系统优化方向，因为它触及 TP group 内 shared/gate/native MoE 的 token 冗余。
- 当前 monkey-patch 版本只得到 `-1.40%` 小幅无 timing 端到端收益，不能作为强系统胜利。
- 真正 blocker 是 collective layout 和实现开销，特别是 AgRs combine 与新增 TP all-gather。
- 下一步若继续 BSP，应做 native SP/MoE integration 或 nsys 级 collective layout profiling，不应继续在 Python monkey-patch 层叠加 Scheme3 等小优化。

### 新增关键文件（本轮追加）

- codex_coding/src/bench_bsp_moe_dp2.py — BSP-MoE 独立 monkey-patch 验证脚本
- code_building/process_docs/v0.1-init-project/v0.1.15.12j-bsp_moe_validation.md — BSP-MoE 机制、实验和结论归档
- codex_coding/results/bsp_moe_c12_8g_e2e_summary_20260427.json — C12 无 timing 端到端 summary
- codex_coding/results/bsp_moe_c12_8g_component_summary_20260427.json — C12 component timing summary
- codex_coding/results/bsp_moe_forward_check.json — BSP vs baseline MoE block forward-check
- codex_coding/results/bsp_moe_shape_probe.json — BSP shape probe JSON

### 本轮命令（追加）

- sed -n '1,240p' docx/next_step.txt
- sed -n '1,260p' codex_coding/src/bench_bsp_moe_dp2.py
- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter torchrun --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --mode shape --num-runs 1
- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter torchrun --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --mode forward-check --num-runs 1
- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter torchrun --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 32 --mode shape --num-runs 1
- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter torchrun --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --mode compare --num-runs 2
- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_ALL2ALL_BACKEND=allgather_reducescatter torchrun --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --mode compare --num-runs 1 --component-timing
- python - <<'PY'
import json
from pathlib import Path
for name in [
    'bsp_moe_c12_8g_e2e_summary_20260427.json',
    'bsp_moe_c12_8g_component_summary_20260427.json',
]:
    data = json.loads(Path('codex_coding/results/' + name).read_text())
    print(name, data.get('delta_pct'))
PY

### v0.1.15.12k: BSP-MoE nsys collective profiling

- 新增 nsys sqlite 分析脚本 `codex_coding/src/analyze_nsys_bsp.py`。
- 生成短 trace A/B 分析报告：
  - `codex_coding/results/nsys_bsp_short_nvtx_analysis_20260427.json`
  - `codex_coding/results/nsys_bsp_short_nvtx_analysis_20260427.md`
- 使用已有短 nsys trace：
  - A baseline: `nsys_bsp_A_short_nvtx_8g_20260427.sqlite`
  - B BSP: `nsys_bsp_B_short_nvtx_8g_20260427.sqlite`
- 分析方法：
  - 只统计 `*.generate.run1` NVTX global window 内的 CUDA kernel/memcpy/runtime。
  - NVTX component 用 rank/thread 维度 rank-max。
  - kernel 按 NCCL、vLLM cross-device reduce、MoE fused、dense GEMM、attention、routing/topk、D2D memcpy 等分类。

#### nsys 短 trace 端到端

| Metric | A baseline | B BSP | B vs A |
|--------|-----------:|------:|-------:|
| log time | 4.176s | 4.680s | +12.07% |
| log ms/fwd | 69.60 | 76.72 | +10.23% |
| NVTX rankmax generate | 3895.928 ms | 4406.343 ms | +13.10% |

说明：nsys 有 profiling overhead，绝对数不和无 profiling C12 e2e 混比；但同条件 A/B 方向可信。

#### collective split 关键结果

| Collective | A rankmax ms/fwd | B rankmax ms/fwd | B vs A |
|------------|-----------------:|-----------------:|-------:|
| NCCL_Reduce | 2.381 | 7.573 | +218.05% |
| NCCL_AllGather | 1.837 | 5.962 | +224.53% |
| NCCL_AllReduce | 3.768 | 4.648 | +23.35% |
| vLLM_cross_device_reduce | 16.276 | 4.687 | -71.20% |

结论：

- BSP 确实降低了 baseline 中一部分 vLLM cross-device TP reduce。
- 但显著放大了 NCCL `AllGather` 和 `Reduce`。
- AllGather count 从 `9120` 增加到 `18392`，与 BSP 路径中 native AgRs all-gather 加上显式 TP all-gather 输出收集一致。

#### 其他开销变化

| Item | A | B | B vs A |
|------|--:|--:|-------:|
| NCCL kernels rankmax ms/fwd | 7.993 | 17.114 | +114.12% |
| dense GEMM rankmax ms/fwd | 0.441 | 0.213 | -51.74% |
| MoE fused kernel rankmax ms/fwd | 1.147 | 1.110 | -3.20% |
| Device-to-Device memcpy total MB | 41222.3 | 21307.7 | -48.31% |

结论：

- BSP 不是完全没有省：D2D memcpy、dense GEMM、外层 native MoE wrapper 都有下降。
- 但 collective 增量大于这些节省，特别是 NCCL Reduce/AllGather。

#### 和完整 C12 component timing 对齐

完整 C12 component timing 中：

| Component | A ms/fwd | B ms/fwd | B vs A |
|-----------|---------:|---------:|-------:|
| moe.dispatch | 5.756 | 7.822 | +35.89% |
| moe.combine | 3.587 | 8.284 | +130.95% |
| moe.tp_all_gather | n/a | 2.618 | n/a |
| dispatch_payload | 826.877 MB/fwd | 206.719 MB/fwd | -75.00% |

因此本轮给出的更精确解释是：

- BSP 的 dispatch payload 节省真实存在。
- 但当前 monkey-patch 把 TP 内 MoE token 冗余转移成更重的 collective sequence。
- AgRs combine 变慢不是偶然噪声，短 nsys trace 的 NCCL Reduce/AllGather 放大与 component timing 对齐。

### 新增关键文件（本轮追加）

- codex_coding/src/analyze_nsys_bsp.py — nsys BSP A/B sqlite 分析脚本
- codex_coding/results/nsys_bsp_short_nvtx_analysis_20260427.md — nsys 短 trace A/B 可读表格报告
- codex_coding/results/nsys_bsp_short_nvtx_analysis_20260427.json — nsys 短 trace A/B 完整结构化结果
- code_building/process_docs/v0.1-init-project/v0.1.15.12k-bsp_nsys_collective_profiling.md — 本轮 profiling 过程和结论归档

### 本轮命令（追加）

- python -m py_compile codex_coding/src/analyze_nsys_bsp.py
- python codex_coding/src/analyze_nsys_bsp.py
- sed -n '1,260p' codex_coding/results/nsys_bsp_short_nvtx_analysis_20260427.md
- sed -n '1,170p' /home/wuhang/miniconda3/envs/dllm/lib/python3.10/site-packages/vllm/distributed/device_communicators/all2all.py
- sed -n '1940,2075p' /home/wuhang/miniconda3/envs/dllm/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/layer.py

### v0.1.15.12l: BSP/nsys 上下文压缩恢复交接

- 按用户模板覆盖写入 `history-chat.txt`。
- 新增过程文档 `code_building/process_docs/v0.1-init-project/v0.1.15.12l-context_compression_recovery_handoff_bsp_nsys.md`。
- 本次恢复说明的目标是让压缩后的 agent 迅速接回当前状态：
  - BSP-MoE 机制验证已完成。
  - nsys collective profiling 已完成。
  - 当前结论是 BSP 的理论节省真实存在，但 monkey-patch 路径把收益转移成更重的 NCCL AllGather/Reduce。
  - 下一步应先和用户讨论 native BSP/SP MoE integration、延迟 all-gather/保持 SP layout 更久、combine/gather 融合或重排，不要直接代码建设。

### 新增关键文件（本轮追加）

- history-chat.txt — 新的上下文压缩恢复说明，覆盖旧内容
- code_building/process_docs/v0.1-init-project/v0.1.15.12l-context_compression_recovery_handoff_bsp_nsys.md — 本轮上下文交接过程文档

### 本轮命令（追加）

- sed -n '1,220p' docx/next_step.txt
- tail -220 code_building/progress_diff_summary.md
- tail -140 code_building/key_conclusion.md
- tail -120 code_building/key_files_index.md
- apply_patch 覆盖写入 history-chat.txt
- apply_patch 新增 code_building/process_docs/v0.1-init-project/v0.1.15.12l-context_compression_recovery_handoff_bsp_nsys.md

### v0.1.15.12m: M1/M2/M3 BSP-DelayGather 与 EB 三路径实验

- 在 `codex_coding/src/bench_bsp_moe_dp2.py` 中补齐 A/B/C/D 四路对比：
  - A: baseline
  - B: BSP-MoE
  - C: BSP-DelayGather
  - D: BSP-DelayGather-M3EPReduce
- D 路径 hot_update 改为显式 vLLM EP group `pop[E]` all-reduce，并增加全 rank controller 诊断。
- 修复一次 D smoke 的 orphan-rank/hung-collective 风险：默认 `dist.all_reduce(pop)` 改成 `get_ep_group().device_group` 后，A/B/C/D smoke 与 C12 均通过。

#### C12 no-timing 结果

| Config | time_s | fwd | ms/fwd | vs A |
|---|---:|---:|---:|---:|
| A baseline | 20.227 | 266 | 76.04 | - |
| B BSP-MoE | 19.898 | 266 | 74.80 | -1.63% |
| C BSP-DelayGather | 19.777 | 266 | 74.35 | -2.22% |
| D BSP-DelayGather-M3EPReduce | 19.946 | 266 | 74.98 | -1.39% |

所有配置 path counts 一致：`prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`。D 路径 `ep_reduce_calls=1862`，`ep_reduce_mb=1.907` per rank。

#### Component timing 归因

- A dispatch payload `826.877 MB/fwd`。
- B/C/D dispatch payload 均为 `206.719 MB/fwd`，确认 BSP payload 下降 `75%`。
- C 相对 B 改善主要来自 `moe.native_forward 38.240 -> 35.227`、`moe.quant_apply 20.161 -> 18.705`、`moe.combine 8.520 -> 6.916`。
- D 不优于 C：`moe.native_forward 38.715`、`moe.quant_apply 20.655`、`moe.combine 8.801`，说明当前 M3 hook 只验证接口兼容，未减少 token-level AgRs 通信。

#### 新增关键文件

- codex_coding/results/bsp_moe_m123_smoke_abcd_20260427.json
- codex_coding/results/bsp_moe_m123_c12_e2e_abcd_20260427.json
- codex_coding/results/bsp_moe_m123_c12_component_abcd_20260427.json
- code_building/process_docs/v0.1-init-project/v0.1.15.12m-m123_bsp_delay_m3_experiment.md

#### 本轮命令（追加）

- python -m py_compile codex_coding/src/bench_bsp_moe_dp2.py
- torchrun --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --mode compare --num-runs 1 --no-quality
- torchrun --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --mode compare --num-runs 1 --no-quality
- torchrun --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --mode compare --num-runs 1 --component-timing --no-quality

### v0.1.15.12n: BSP C+ / C++ Upper-Bound 实验

- 在 `codex_coding/src/bench_bsp_moe_dp2.py` 中补齐 A/B/C/D/E/F 六路对比：
  - E: `C12-BSP-CrossLayerSP`，把 SP hidden state 保持跨 sparse decoder layer 边界，attention 前再 gather。
  - F: `C12-BSP-AllReduceFullProbe`，保留 EP dispatch 和本地 expert compute，但用 EP full all-reduce probe 替代 native combine+TP gather。
- 修复 F probe 中 `dp_rank` 闭包作用域问题，并扩展 compare summary、JSON 输出和 component timing。
- 仍不修改 dInfer/vLLM 源码，不引入 Scheme3 payload 改造，不改变 EB/s_mask 算法。

#### C12 no-timing 两轮结果

| Config | Run1 ms/fwd | Repeat ms/fwd | Avg ms/fwd | Avg vs A |
|---|---:|---:|---:|---:|
| A baseline | 75.61 | 75.86 | 75.73 | - |
| B BSP-MoE | 74.59 | 74.64 | 74.62 | -1.47% |
| C BSP-DelayGather | 74.14 | 74.14 | 74.14 | -2.10% |
| D BSP-DelayGather-M3EPReduce | 74.63 | 74.45 | 74.54 | -1.57% |
| E BSP-CrossLayerSP | 71.90 | 71.69 | 71.80 | -5.20% |
| F BSP-AllReduceFullProbe | 72.41 | 72.51 | 72.46 | -4.32% |

所有 C12 配置 path counts 一致：`prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`。

#### Component timing 归因

- E 的 dispatch payload 仍为 `206.719 MB/fwd`，TP gather payload 仍为 `165.375 MB/fwd`，说明 E 的收益不是新的 payload 减少，而是 SP layout 生命周期延长带来的 native/quant/combine 与调度开销下降。
- E component：`native_forward=35.322`、`quant_apply=18.873`、`combine=7.267 ms/fwd`，基本保持 C 路径的低开销。
- F component：`native_forward=29.195`、`ep_full_all_reduce=9.467 ms/fwd`，但 `ep_full_allreduce_payload=1311.020 MB/fwd`。
- F 证明 combine+gather 融合/重排方向有潜力，但 raw full all-reduce payload 太大，不应作为 production 方案。

#### 质量 smoke

- 小规模 `batch=32,gen=32` snippets 显示 E/F 没有灾难性语义崩坏。
- 仍需完整质量集才能做生产质量结论。

#### 新增关键文件

- codex_coding/results/bsp_moe_cplus_cxx_smoke_20260427.json
- codex_coding/results/bsp_moe_cplus_cxx_c12_e2e_20260427.json
- codex_coding/results/bsp_moe_cplus_cxx_c12_e2e_repeat_20260428.json
- codex_coding/results/bsp_moe_cplus_cxx_c12_component_20260427.json
- codex_coding/results/bsp_moe_cplus_cxx_quality_smoke_20260428.json
- code_building/process_docs/v0.1-init-project/v0.1.15.12n-cplus_cxx_bsp_upper_bound.md

#### 本轮命令（追加）

- python3 -m py_compile codex_coding/src/bench_bsp_moe_dp2.py
- torchrun --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --mode compare --num-runs 1 --no-quality
- torchrun --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --mode compare --num-runs 1 --no-quality
- torchrun --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --mode compare --num-runs 1 --component-timing --no-quality
- torchrun --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --mode compare --num-runs 1

### v0.1.15.12o: BSP-G vLLM SP-Parity 实验

- 在 `codex_coding/src/bench_bsp_moe_dp2.py` 中新增 `G) C12-BSP-G-AttnReduceScatterSP`。
- G 将 attention output projection 从 TP all-reduce/full layout 输出改成 TP reduce-scatter/SP layout 输出。
- G 之后 residual、post-attention norm、MoE 直接消费 SP hidden state，进一步延长 SP layout 生命周期。
- 仍不修改 dInfer/vLLM 源码，不引入 Scheme3 payload 改造，不改变 EB/s_mask 算法。

#### C12 no-quality 两轮结果

| Config | Run1 ms/fwd | Repeat ms/fwd | Avg ms/fwd | Avg vs A |
|---|---:|---:|---:|---:|
| A baseline | 75.35 | 76.70 | 76.03 | - |
| B BSP-MoE | 74.48 | 74.77 | 74.62 | -1.85% |
| C BSP-DelayGather | 73.98 | 75.79 | 74.89 | -1.50% |
| D BSP-DelayGather-M3EPReduce | 74.56 | 80.54 | 77.55 | +2.01% |
| E BSP-CrossLayerSP | 71.67 | 71.59 | 71.63 | -5.78% |
| G BSP-G-AttnReduceScatterSP | 69.55 | 69.51 | 69.53 | -8.55% |
| F BSP-AllReduceFullProbe | 72.28 | 72.34 | 72.31 | -4.90% |

所有 C12 配置 path counts 一致：`prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`。

#### Component timing 归因

- G 把 `moe.bsp_chunk` 从 E 的 `0.872 ms/fwd`、count `5320` 降到 `0.003 ms/fwd`、count `266`。
- G 新增 `attn.tp_reduce_scatter=5.020 ms/fwd`，`attn_rs_payload=661.502 MB/fwd`。
- G 的 MoE dispatch payload 仍为 `206.719 MB/fwd`，TP gather payload 仍为 `165.375 MB/fwd`，与 E 相同。
- 结论：G 的收益来自 attention output 直接进入 SP layout、减少反复 full/SP chunk，而不是减少 MoE combine/gather 字节。
- component timing 的绝对 e2e 排序受 instrumentation 扰动；机制归因有效，最终速度以 no-quality e2e 为主。

#### Quality smoke

- 小规模 `batch=32,gen=32` snippets 显示 G 没有灾难性语义崩坏。
- 仍需完整质量集才能做生产质量结论。

#### 环境问题

- 第一次 component timing 在 A baseline prefill OOM；原因是旧 benchmark PIDs 报告每卡占用约 `56-58 GB`。
- 清理/释放后重跑成功；该问题记录到 `.learnings/ERRORS.md` 的 `ERR-20260428-001`。

#### 新增关键文件

- codex_coding/results/bsp_moe_bspg_smoke_20260428.json
- codex_coding/results/bsp_moe_bspg_c12_e2e_20260428.json
- codex_coding/results/bsp_moe_bspg_c12_e2e_repeat_20260428.json
- codex_coding/results/bsp_moe_bspg_c12_component_20260428.json
- codex_coding/results/bsp_moe_bspg_quality_smoke_20260428.json
- code_building/process_docs/v0.1-init-project/v0.1.15.12o-bsp_g_vllm_sp_parity.md

#### 本轮命令（追加）

- python3 -m py_compile codex_coding/src/bench_bsp_moe_dp2.py
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --num-runs 1 --mode compare --no-quality
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --num-runs 1 --mode compare --no-quality
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --num-runs 1 --mode compare --component-timing --no-quality
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --num-runs 1 --mode compare

### v0.1.15.12p: BSP-G2 vLLM SP-Parity Bundle

- 在 `codex_coding/src/bench_bsp_moe_dp2.py` 中新增 `G2) C12-BSP-G2-SPParityBundle`。
- G2 新增 `SPAttentionInput` carrier，让 attention path 接收 SP-normalized hidden，并在 attention 内部做 input all-gather。
- G2 继续用 attention output reduce-scatter 返回 SP layout，让 residual、post-attention norm、MoE 保持 SP。
- E 和 G 均保持不变；G2 只作为独立 compare config。
- 新增 reduced matrix：
  - `--config-set bspg2`: A/E/G/G2/F
  - `--config-set aeg2f`: A/E/G2/F
  - `--config-set aeggf`: A/E/G/G2/F
- 仍不修改 dInfer/vLLM 源码，不引入 Scheme3 payload 改造，不改变 EB/s_mask 算法。
- 按新规范在实验开始前创建过程文档，并在每个实验阶段结束后立即更新归档。

#### C12 no-quality 结果

| Config | ms/fwd | vs A | Path counts |
|---|---:|---:|---|
| A baseline | 75.428 | - | `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931` |
| E BSP-CrossLayerSP | 71.816 | -4.789% | 同 A |
| G BSP-G-AttnReduceScatterSP | 69.676 | -7.626% | 同 A |
| G2 BSP-G2-SPParityBundle | 69.661 | -7.646% | 同 A |
| F BSP-AllReduceFullProbe | 72.380 | -4.042% | 同 A |

G2 与 G 只差 `0.015 ms/fwd`，属于测量噪声内持平。

#### Component timing 归因

- G2 成功把 attention-input gather 从 MoE/wrapper bucket 迁移到 attention bucket：
  - G: `moe.tp_all_gather=2.662 ms/fwd,count=5054`
  - G2: `attn.input_all_gather=2.573 ms/fwd,count=4788`
  - G2 residual: `moe.tp_all_gather=0.141 ms/fwd,count=266`
- G2 payload 只是迁移而非减少：
  - G: `tp_gather_payload=165.375 MB/fwd`
  - G2: `attn_input_gather_payload=156.671 MB/fwd` + `tp_gather_payload=8.704 MB/fwd`
- G/G2 共同开销仍在：
  - `attn_rs_payload=661.502 MB/fwd`
  - `dispatch_payload=206.719 MB/fwd`
- component instrumentation 下 G2 比 G 慢：
  - G `78.149 ms/fwd`
  - G2 `79.786 ms/fwd`

#### 质量 smoke

- 小规模 `batch=32,gen=32` snippets 显示 G2 与 G 基本一致，未见新增灾难性语义崩坏。
- E 在小 batch 下再次出现 `hot_skip=893`，但 C12 正式 invariant 正常，记录为小 batch artifact。

#### 最终判断

- G 仍是 measured-best 性能路径。
- G2 是 source-organization/SP-parity 路径：attention 同时 owning input gather 和 output reduce-scatter。
- G2 不减少通信字节或同步点，因此不能替代 BSP-H/F2。
- 下一步若继续做性能，应聚焦减少/fuse synchronization 或继续延长 SP layout 生命周期，而不是单纯迁移 gather bucket。

#### 新增关键文件

- codex_coding/results/bsp_moe_bspg2_smoke_20260428.json
- codex_coding/results/bsp_moe_bspg2_c12_e2e_20260428.json
- codex_coding/results/bsp_moe_bspg2_c12_component_20260428.json
- codex_coding/results/bsp_moe_bspg2_quality_smoke_20260428.json
- code_building/process_docs/v0.1-init-project/v0.1.15.12p-bsp_g2_sp_parity_bundle.md

#### 本轮命令（追加）

- python3 -m py_compile codex_coding/src/bench_bsp_moe_dp2.py
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --num-runs 1 --mode compare --config-set bspg2 --no-quality
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --num-runs 1 --mode compare --config-set bspg2 --no-quality
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --num-runs 1 --mode compare --config-set bspg2 --component-timing --no-quality
- torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --num-runs 1 --mode compare --config-set bspg2

### v0.1.15.13: EB HetEval512 Law Probe

- 新增观测型脚本 `codex_coding/src/collect_eb_heteval512_laws.py`，在不改变 EB 路由策略的前提下记录压缩后的 `S_mask`、EB top4、no-EB top4/top8、per-layer expert histogram 和 request grouping 摘要。
- 按新规范在实验开始前创建过程文档 `code_building/process_docs/v0.1-init-project/v0.1.15.13-eb_heteval512_law_probe.md`，实验结束后在同一归档补充结果和结论。
- HetEval512/C12 配置确认：
  - `batch=512,gen=256,block=32`
  - `dp=2,tp=4,ep=8`
  - `threshold=0.90`
  - EB: `K=8,topk=4,K_target=40,q_major=1.0,skip_m=5`
- 全量运行 path counts 与历史 C12 一致：
  - `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`
- 主结果：
  - EB top4 unique active experts mean `197.98`
  - no-EB top4 unique active experts mean `251.22`
  - no-EB top8 unique active experts mean `254.59`
  - EB vs no-EB top4 active expert reduction `21.19%`
- 层差异明显：
  - 最强压缩层：L17 `25.17%`, L7 `24.90%`, L9 `24.17%`, L16 `24.02%`
  - 最弱压缩层：L1 `13.73%`, L0 `14.24%`
- `S_mask` 稳定性：
  - adjacent Jaccard mean `0.9862`, p50/p90/p95 为 `1.0`
  - previous-call same-size set coverage `0.9658`
  - previous-block cold coverage `0.9328`
- 控制实验：
  - EB `S_mask` coverage `0.9681`
  - random same-size set coverage `0.7733`
  - offline global-popularity same-size set coverage `0.9704`
- 关键结论：
  - EB 不是随机裁剪，active-expert reduction 真实存在。
  - HetEval512 大 batch 下存在很强的全局热门 expert 结构；后续应探索 `global prior + EB correction`。
  - EB 减少 active experts，但 linear placement 下 EP load skew 略变差：EB `3.118` vs no-EB top4 `2.970`，说明 active-expert reduction 和 load balancing 是两个目标。
  - 简单 request centroid grouping 不够好；scheduler 需要 set-similarity / load-aware 指标。

#### 新增关键文件

- `codex_coding/src/collect_eb_heteval512_laws.py`
- `codex_coding/results/eb_heteval512_laws_20260428.log`
- `codex_coding/results/eb_heteval512_laws_20260428_rank0.json.gz`
- `codex_coding/results/eb_heteval512_laws_20260428_rank4.json.gz`
- `codex_coding/results/eb_heteval512_laws_20260428_summary.json`
- `codex_coding/results/eb_heteval512_laws_20260428_extended_controls.json`
- `code_building/process_docs/v0.1-init-project/v0.1.15.13-eb_heteval512_law_probe.md`

#### 本轮命令（追加）

- `python3 -m py_compile codex_coding/src/collect_eb_heteval512_laws.py`
- `torchrun --standalone --nproc_per_node=8 codex_coding/src/collect_eb_heteval512_laws.py --batch-size 32 --gen-length 32 --output-prefix eb_heteval512_laws_smoke_20260428`
- `torchrun --standalone --nproc_per_node=8 codex_coding/src/collect_eb_heteval512_laws.py --batch-size 512 --gen-length 256 --output-prefix eb_heteval512_laws_20260428`

### v0.1.15.15: EB-aware EPLB Source Landing (E=34 tuned-path unblock phase-1)

- 按新规范先在主过程文档追加 pre-archive，再执行实验并回填结果。
- 新增 vLLM fused_moe 配置文件：
  - `/home/wuhang/miniconda3/envs/dllm/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/configs/E=34,N=512,device_name=NVIDIA_H100_80GB_HBM3.json`
  - 首版直接复用 `E=32,N=512` 的 tile 参数。
- 复跑 P5 A/C（`batch=128,gen=32,tp=4,dp=2`）：
  - A: `eb_eplb_p5_ab_a_linear_true_e34cfg_20260430_summary.json`
    - `144.381 ms/fwd`
  - C: `eb_eplb_p5_ab_c_replica_true_e34cfg_20260430_summary.json`
    - `157.488 ms/fwd`
  - C vs A: `+9.078%`（更慢）
- 日志确认：
  - C 路径命中 `E=34,N=512` 配置文件，未再出现 default MoE config fallback warning。
- 结论：
  - fallback 问题已消除；
  - 但 E32 模板直接迁移到 E34 在当前环境未带来性能改善，反而扩大 C 对 A 的劣化；
  - 下一步应进行 E34 配置轻量 sweep，并并行推进 P1/P5 联合编排设计。

#### 本轮命令（追加）

- `ls/rg/cat` 查询 vLLM fused_moe config 目录与 E32 配置内容
- `torchrun --nproc_per_node=8 codex_coding/src/collect_eb_heteval512_laws.py --batch-size 128 --gen-length 32 --tp-size 4 --world-size 8 --trace-rank-mode tp0 --output-prefix eb_eplb_p5_ab_a_linear_true_e34cfg_20260430 --eplb-runtime-redundant-experts 0 --eplb-runtime-window-size 16 --eplb-runtime-step-interval 16`
- `MASTER_PORT=29531 torchrun --nproc_per_node=8 codex_coding/src/collect_eb_heteval512_laws.py --batch-size 128 --gen-length 32 --tp-size 4 --world-size 8 --trace-rank-mode tp0 --output-prefix eb_eplb_p5_ab_c_replica_true_e34cfg_20260430 --eplb-runtime-redundant-experts 13 --eplb-runtime-window-size 16 --eplb-runtime-step-interval 16`

### v0.1.15.15: P1/P5 联合编排首版（同源 physical id 编排）

- 在 `modeling_llada2_moe.py` 落地联合编排首版：
  - 保持 canonical `physical_to_logical` 不变；
  - `pid<256` 按 P1 map 决定 rank 归属；
  - 冗余 `pid>=256` 按 round-robin 分配到 EP ranks；
  - 去掉“replica 开启时忽略 non-identity external map”的临时保护。
- 新增不变性检查：
  - 每 rank local slot 数量匹配；
  - local physical id 无重复、无越界；
  - 每层全局 physical 分配保持双射；
  - 每 logical 的全局副本计数与 canonical 期望一致。
- `py_compile` 通过后复跑 A/B/C（`batch=128,gen=32,tp=4,world=8`）：
  - A: `eb_eplb_p5_ab_a_linear_true_jointmap_20260430_summary.json` -> `140.744 ms/fwd`
  - B: `eb_eplb_p5_ab_b_p1_true_jointmap_20260430_summary.json` -> `137.330 ms/fwd`
  - C: `eb_eplb_p5_ab_c_replica_true_jointmap_20260430_summary.json` -> `155.162 ms/fwd`
- 对比：
  - B vs A: `-2.425%`（P1 仍正收益）
  - C vs A: `+10.244%`（仍慢）
  - C vs B: `+12.984%`（仍慢）
  - C(new jointmap) vs C(old e34cfg): `-1.477%`（小幅改善，但不足以转正）
- 路径计数一致：
  - A/B/C 均为 `prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`
  - 说明 EB 语义和 path schedule 未被本次改造破坏。

#### 本轮命令（追加）

- `python3 -m py_compile /home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py`
- `MASTER_PORT=29540 torchrun --nproc_per_node=8 codex_coding/src/collect_eb_heteval512_laws.py --batch-size 128 --gen-length 32 --tp-size 4 --world-size 8 --trace-rank-mode tp0 --output-prefix eb_eplb_p5_ab_a_linear_true_jointmap_20260430 --eplb-runtime-redundant-experts 0 --eplb-runtime-window-size 16 --eplb-runtime-step-interval 16`
- `DINF_EPLB_EXPERT_MAP_PATH=/home/wuhang/wuhang/dllm_wh/codex_coding/results/coact_expert_map_ep8_20260429.pt MASTER_ADDR=127.0.0.1 MASTER_PORT=29541 torchrun --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29541 --nproc_per_node=8 codex_coding/src/collect_eb_heteval512_laws.py --batch-size 128 --gen-length 32 --tp-size 4 --world-size 8 --trace-rank-mode tp0 --output-prefix eb_eplb_p5_ab_b_p1_true_jointmap_20260430 --eplb-runtime-redundant-experts 0 --eplb-runtime-window-size 16 --eplb-runtime-step-interval 16`
- `DINF_EPLB_EXPERT_MAP_PATH=/home/wuhang/wuhang/dllm_wh/codex_coding/results/coact_expert_map_ep8_20260429.pt MASTER_PORT=29532 torchrun --nproc_per_node=8 codex_coding/src/collect_eb_heteval512_laws.py --batch-size 128 --gen-length 32 --tp-size 4 --world-size 8 --trace-rank-mode tp0 --output-prefix eb_eplb_p5_ab_c_replica_true_jointmap_20260430 --eplb-runtime-redundant-experts 13 --eplb-runtime-window-size 16 --eplb-runtime-step-interval 16`

### v0.1.15.15: C 慢因子归因拆解（E34 vs runtime EPLB map）

- 在 `collect_eb_heteval512_laws.py` 增加 `--skip-runtime-eplb-state` 开关：
  - 保留 constructor 期 EPLB physical capacity（`E=34`）；
  - 跳过 runtime `build_and_set_eplb_runtime_state` 时显式 `set_eplb_runtime_state(enable_eplb=False)`，确保前向不走 logical->physical remap。
- 完整 6 组矩阵（`batch=128,gen=32,tp=4,world=8`）：
  - A0: `eb_eplb_factor_a0_linear_e32_20260430` -> `138.862 ms/fwd`
  - C0: `eb_eplb_factor_c0_linear_e34_nort_20260430` -> `142.277 ms/fwd`
  - C1: `eb_eplb_factor_c1_linear_e34_rt_20260430` -> `161.840 ms/fwd`
  - B0: `eb_eplb_factor_b0_p1_e32_20260430` -> `132.759 ms/fwd`
  - C2: `eb_eplb_factor_c2_p1_e34_nort_20260430` -> `136.368 ms/fwd`
  - C3: `eb_eplb_factor_c3_p1_e34_rt_20260430` -> `148.046 ms/fwd`
- 归因结论：
  - 纯 `E=34` 形状开销约 `+2.46% ~ +2.72%`（`C0-A0`、`C2-B0`）。
  - runtime EPLB map 增量开销约 `+8.56% ~ +13.75%`（`C3-C2`、`C1-C0`）。
  - 当前 C 路径慢因子主导项是 runtime map，不是 E34 config 缺失或 E34 形状本身。
- 语义稳定性：
  - 六组 path counts 全一致 `19/38/893/209`，EB 语义与 path schedule 未变化。

#### 本轮命令（追加）

- `python3 -m py_compile codex_coding/src/collect_eb_heteval512_laws.py`
- `MASTER_PORT=29550 torchrun --nproc_per_node=8 codex_coding/src/collect_eb_heteval512_laws.py --batch-size 128 --gen-length 32 --tp-size 4 --world-size 8 --trace-rank-mode tp0 --output-prefix eb_eplb_factor_a0_linear_e32_20260430 --eplb-runtime-redundant-experts 0 --eplb-runtime-window-size 16 --eplb-runtime-step-interval 16`
- `MASTER_PORT=29551 torchrun --nproc_per_node=8 codex_coding/src/collect_eb_heteval512_laws.py --batch-size 128 --gen-length 32 --tp-size 4 --world-size 8 --trace-rank-mode tp0 --output-prefix eb_eplb_factor_c0_linear_e34_nort_20260430 --eplb-runtime-redundant-experts 13 --eplb-runtime-window-size 16 --eplb-runtime-step-interval 16 --skip-runtime-eplb-state`
- `MASTER_PORT=29552 torchrun --nproc_per_node=8 codex_coding/src/collect_eb_heteval512_laws.py --batch-size 128 --gen-length 32 --tp-size 4 --world-size 8 --trace-rank-mode tp0 --output-prefix eb_eplb_factor_c1_linear_e34_rt_20260430 --eplb-runtime-redundant-experts 13 --eplb-runtime-window-size 16 --eplb-runtime-step-interval 16`
- `DINF_EPLB_EXPERT_MAP_PATH=/home/wuhang/wuhang/dllm_wh/codex_coding/results/coact_expert_map_ep8_20260429.pt MASTER_PORT=29553 torchrun --nproc_per_node=8 codex_coding/src/collect_eb_heteval512_laws.py --batch-size 128 --gen-length 32 --tp-size 4 --world-size 8 --trace-rank-mode tp0 --output-prefix eb_eplb_factor_b0_p1_e32_20260430 --eplb-runtime-redundant-experts 0 --eplb-runtime-window-size 16 --eplb-runtime-step-interval 16`
- `DINF_EPLB_EXPERT_MAP_PATH=/home/wuhang/wuhang/dllm_wh/codex_coding/results/coact_expert_map_ep8_20260429.pt MASTER_PORT=29554 torchrun --nproc_per_node=8 codex_coding/src/collect_eb_heteval512_laws.py --batch-size 128 --gen-length 32 --tp-size 4 --world-size 8 --trace-rank-mode tp0 --output-prefix eb_eplb_factor_c2_p1_e34_nort_20260430 --eplb-runtime-redundant-experts 13 --eplb-runtime-window-size 16 --eplb-runtime-step-interval 16 --skip-runtime-eplb-state`
- `DINF_EPLB_EXPERT_MAP_PATH=/home/wuhang/wuhang/dllm_wh/codex_coding/results/coact_expert_map_ep8_20260429.pt MASTER_PORT=29555 torchrun --nproc_per_node=8 codex_coding/src/collect_eb_heteval512_laws.py --batch-size 128 --gen-length 32 --tp-size 4 --world-size 8 --trace-rank-mode tp0 --output-prefix eb_eplb_factor_c3_p1_e34_rt_20260430 --eplb-runtime-redundant-experts 13 --eplb-runtime-window-size 16 --eplb-runtime-step-interval 16`

### v0.1.15.15: runtime remap record gating（cold-first）实验

- 在 `collect_eb_heteval512_laws.py` 新增 `--eplb-runtime-record-mode {full,cold_only,off}`。
- 通过 runtime patch 验证“保留 logical->physical 映射，但按 path 控制 record”的可行性：
  - `full`: 原生 map+record
  - `cold_only`: cold map+record，hot map-only
  - `off`: map-only
- C3 口径（`batch=128,gen=32,tp=4,world=8`）两次复测均显示：
  - `full` mean `151.750 ms/fwd`
  - `cold_only` mean `144.117 ms/fwd`
  - `off` mean `150.684 ms/fwd`
  - `cold_only` 相对 `full`：`-5.03%`
  - `off` 相对 `full`：`-0.70%`
- 不变性：
  - 六个 run path counts 均一致：`prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`
  - EB 语义与路径调度保持不变。
- 阶段结论：
  - 当前主线可优先采用 `cold_only` 思路（保守语义 + 可观收益）；
  - 下阶段建议把该策略下沉到 dInfer/vLLM 源码实现，避免实验层 monkey-patch + compile 的运行时噪声。

### v0.1.15.15: cold-gated rearrange（脚本层 PoC）矩阵

- 修复了 cold-gated rearrange PoC 的接口对齐问题：
  - `proxy.expert_weights` 改为 vLLM `rearrange` 期望的每层 `(w13_weight, w2_weight)`；
  - 补齐 proxy metadata（`num_redundant_experts`、`num_local_physical_experts` 等）；
  - 异常日志升级为 `type+repr+traceback`，消除空字符串 fail reason。
- smoke `eb_eplb_coldrearrange_smoke_rg8_fix_20260430` 验证：
  - `attempts=1, success=1, fail=0`
  - 单次重排约 `3.2~3.3s`
- C3 正式矩阵（`batch=128,gen=32,tp=4,world=8,P1+replica,record_mode=cold_only`）：
  - `base`: `149.587 ms/fwd`
  - `rg8`: `201.459 ms/fwd`（vs base `+34.677%`）
  - `rg16`: `201.898 ms/fwd`（vs base `+34.970%`）
- 不变性：
  - 三组 path counts 一致：`prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`
  - rearrange 均 `fail=0`，8 rank triplet 一致。
- 结论：
  - 当前口径下 cold-gated rearrange 的一次性重排成本（约 `3.0~3.2s`）显著大于其潜在收益；
  - 暂不建议在线默认启用该策略，主线继续采用 `record_mode=cold_only`，并将重排留在离线/更低频窗口再评估。

#### 本轮命令（追加）

- `python3 -m py_compile codex_coding/src/collect_eb_heteval512_laws.py`
- `MASTER_PORT=29660 torchrun --nproc_per_node=8 ... --output-prefix eb_eplb_coldrearrange_smoke_rg8_fix_20260430 --batch-size 64 --gen-length 8 --eplb-runtime-record-mode cold_only --eplb-cold-rearrange --eplb-cold-rearrange-min-gap 8`
- `MASTER_PORT=29661 torchrun --nproc_per_node=8 ... --output-prefix eb_eplb_coldrearrange_c3_base_20260430 --eplb-runtime-record-mode cold_only`
- `MASTER_PORT=29662 torchrun --nproc_per_node=8 ... --output-prefix eb_eplb_coldrearrange_c3_rg8_20260430 --eplb-runtime-record-mode cold_only --eplb-cold-rearrange --eplb-cold-rearrange-min-gap 8`
- `MASTER_PORT=29663 torchrun --nproc_per_node=8 ... --output-prefix eb_eplb_coldrearrange_c3_rg16_20260430 --eplb-runtime-record-mode cold_only --eplb-cold-rearrange --eplb-cold-rearrange-min-gap 16`


### v0.1.15.15: runtime map impl 优化（flat-index）与串行复测

- `collect_eb_heteval512_laws.py` 新增 `--eplb-runtime-map-impl {vllm,flat_compile,flat_eager}`，并把 runtime patch 扩展为 `record_mode + map_impl` 双维控制。
- 新增 `flat-index` 映射实现（`logical_to_physical_map.reshape(-1)[flat_idx]`），在不改 EB 语义的前提下替换原 `l2p[topk_ids].gather(...)` 逻辑。
- smoke（`batch=64,gen=8,P1+replica,record_mode=cold_only`）结果：
  - `vllm`: `194.055 ms/fwd`
  - `flat_compile`: `216.433 ms/fwd`（负优化）
  - `flat_eager`: `162.225 ms/fwd`（显著改善）
  - path counts 一致：`19/19/399/95`
- 正式口径（严格串行，`batch=128,gen=32,tp=4,world=8,P1+replica,record_mode=cold_only`）两轮复测：
  - r1: `vllm=147.146`, `flat_eager=136.931`，`-6.942%`
  - r2: `vllm=152.264`, `flat_eager=141.412`，`-7.127%`
  - mean: `149.705 -> 139.171 ms/fwd`，平均回收 `-7.036%`（`-10.534 ms/fwd`）
- 与历史锚点对比：
  - `B0(P1,E32)=132.759 ms/fwd`
  - `flat_eager C3 mean=139.171`，仍高于 B0 约 `+4.83%`，但相比此前 C3（约 148ms）明显收敛。
- 在新路径下复核 hot-replica（`DINF_EPLB_REPLICA_LOGICAL_IDS_PATH`）:
  - `flat_eager + hotrep = 149.572 ms/fwd`
  - 相比 `flat_eager r2=141.412` 回退 `+5.77%`
  - 仅相对 `vllm r2=152.264` 轻微改善 `-1.77%`
- 阶段结论：
  - 当前可兑现收益主线是 runtime map 实现优化（`flat_eager`），而不是继续调 hot replica id。
  - `flat_compile` 在当前环境不适合，后续应优先推进 eager/native 下沉。

#### 本轮命令（追加）

- `python3 -m py_compile codex_coding/src/collect_eb_heteval512_laws.py`
- `torchrun --master_port=29720 --nproc_per_node=8 ... --output-prefix eb_eplb_mapimpl_smoke_vllm_20260430 --eplb-runtime-map-impl vllm`
- `torchrun --master_port=29721 --nproc_per_node=8 ... --output-prefix eb_eplb_mapimpl_smoke_flatc_20260430 --eplb-runtime-map-impl flat_compile`
- `torchrun --master_port=29712 --nproc_per_node=8 ... --output-prefix eb_eplb_mapimpl_smoke_flate_20260430 --eplb-runtime-map-impl flat_eager`
- `torchrun --master_port=29740 --nproc_per_node=8 ... --output-prefix eb_eplb_mapimpl_c3_vllm_seq_20260430 --eplb-runtime-map-impl vllm`
- `torchrun --master_port=29741 --nproc_per_node=8 ... --output-prefix eb_eplb_mapimpl_c3_flate_seq_20260430 --eplb-runtime-map-impl flat_eager`
- `torchrun --master_port=29742 --nproc_per_node=8 ... --output-prefix eb_eplb_mapimpl_c3_vllm_seq_r2_20260430 --eplb-runtime-map-impl vllm`
- `torchrun --master_port=29743 --nproc_per_node=8 ... --output-prefix eb_eplb_mapimpl_c3_flate_seq_r2_20260430 --eplb-runtime-map-impl flat_eager`
- `DINF_EPLB_REPLICA_LOGICAL_IDS_PATH=.../eplb_hot_replica_ids_top16_20260430.pt torchrun --master_port=29744 --nproc_per_node=8 ... --output-prefix eb_eplb_mapimpl_c3_flate_hotrep_seq_20260430 --eplb-runtime-map-impl flat_eager`

### v0.1.15.15: runtime map source-downshift（dInfer源码落地）完成

- 已将 `flat_eager + cold_only` 从实验脚本 monkey-patch 下沉到 dInfer 源码接口：
  - `modeling_llada2_moe.py` 新增：
    - `configure_eplb_runtime_map_policy(...)`
    - `set_eplb_runtime_route_path(...)`
    - `use_eplb_runtime_map_policy(...)`
  - 脚本 `collect_eb_heteval512_laws.py` 仅负责设置 policy 与 route path，不再内置 map patch。
- C3 串行两轮复测（`batch=128,gen=32,tp=4,world=8,P1+replica,cold_only`）：
  - r1: `vllm=155.228`, `flat_eager=143.272`, `-7.70%`
  - r2: `vllm=154.932`, `flat_eager=139.569`, `-9.91%`
  - mean: `155.080 -> 141.421 ms/fwd`, `-8.81%`
- 不变性：
  - 4 组 path counts 均为 `prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`
  - `eplb_load_balance_diag`（`overall_skew/layer_skew`）保持一致，说明收益仍来自 map 实现开销而非负载变化。
- 与旧脚本 patch 对照：
  - 旧均值 delta `-7.04%`，源码下沉后为 `-8.81%`，收益方向与量级稳定保留。
- 当前状态：
  - runtime map 优化主线已完成“验证 -> 下沉 -> 复证”闭环，可作为后续 dInfer 主干能力继续迭代。

### v0.1.15.15: native gate 路线排除 + flat_eager A/B（tensor_cache）

- 新增 `native_record_gate_mode` 实验开关后确认：当前运行环境该路线不可用（`available=False,total_calls=0`），因为运行态 vLLM EPLB 版本不含 `should_record_tensor` 机制。
- 在不可用前提下，`vllm+full+native(cold_only)` C3 为 `155.605 ms/fwd`，显著慢于当前 `flat_eager+cold_only`（同轮 `145.457`）；该路线已停用。
- 在 `flat_eager+cold_only` 主线新增 `--eplb-runtime-tensor-cache {on,off}` 做严格 C3 A/B：
  - `off`: `139.116`、`138.536`（mean `138.826`）
  - `on`: `172.328`、`146.131`（mean `159.230`）
  - 结论：`tensor_cache=on` 是负优化（`+14.7%`），并伴随高抖动；默认值已回设为 `off`。
- 当前主线最优点更新为：
  - `flat_eager + cold_only + tensor_cache=off`
  - 相比旧 `flat_eager` 均值（`141.421`）再降约 `1.83%`；
  - 相比 `vllm` 均值（`155.080`）总体回收约 `11.71%`。
- 所有对比组 path counts 保持一致 `prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`，语义稳定。

### v0.1.15.15: flat_eager+cold_only 去动态化快路径（runtime fastpath）上线

- 在 `modeling_llada2_moe.py` 为 `record_mode=cold_only + map_impl=flat_eager + tensor_cache=off` 增加专用 `_patched_fast`，并引入 `path_is_cold` 布尔状态，去掉热路径中的字符串比较和多分支动态判断。
- 新增可控开关 `DINF_EPLB_RUNTIME_FASTPATH`（默认 `on`），用于同版本 `on/off` A/B 验证与回归隔离。
- C3 同口径四轮 A/B（严格串行）结果：
  - `off`: `140.442 / 141.700`，mean `141.071 ms/fwd`
  - `on`: `135.206 / 136.837`，mean `136.021 ms/fwd`
  - `on vs off`: `-5.050 ms/fwd`（`-3.58%`）
- 对旧最优（cacheab `off` 均值 `138.826`）：
  - 新主线再降 `-2.805 ms/fwd`（`-2.02%`）。
- 四组 path counts 全一致 `prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`，说明收益来自实现开销压降而非 EB 语义变化。
- 主线更新为：
  - `flat_eager + cold_only + tensor_cache=off + runtime_fastpath(on)`.

### v0.1.15.15: fastpath-v2（局部缓存）验证失败并回退

- 尝试在 `_patched_fast` 内加入局部张量模板缓存（`pos_indices/ones`），期望进一步减少构造开销。
- C3 两轮结果显著回退：
  - r1 `142.370`, r2 `146.753`, mean `144.562 ms/fwd`
  - 相比 fastpath-v1 均值 `136.021` 变慢 `+6.279%`。
- path counts 仍为 `19/38/893/209`，说明问题是实现开销负优化而非语义漂移。
- 已回退到 fastpath-v1；回退校验 `136.527 ms/fwd`，恢复到主线水平。
- 当前结论保持：
  - 不引入 fastpath-v2 局部缓存；
  - 保持 `flat_eager + cold_only + tensor_cache=off + runtime_fastpath(on)` 主线。

### v0.1.15.15: sparse-rr map fastpath（仅多副本位置取模）验证失败并回退

- 在 `flat_eager` map core 试验“仅对 `replica_count>1` 做 rr/mod，`replica_count==1` 直接 slot-0”。
- 通过 `DINF_EPLB_MAP_SPARSE_RR` 做可控 C3 A/B（`batch=128,gen=32,tp=4,world=8,P1+replica,cold_only,fastpath=on`）：
  - `off`: `141.610 / 140.615`，mean `141.113 ms/fwd`
  - `on`: `143.156 / 147.438`，mean `145.297 ms/fwd`
  - `on vs off`: `+2.97%`（负优化）
- 四组 path counts 全一致：`prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`，语义未漂移。
- 已回退该实现；回退后 `eb_eplb_sparse_rr_revertcheck_c3_on_r1_20260430` 为 `134.599 ms/fwd`，与 `routeopt_revertcheck`（`134.636`）一致，说明主线恢复正常。
- 阶段结论：
  - 不保留 sparse-rr；
  - 继续维持 `flat_eager + cold_only + tensor_cache=off + runtime_fastpath(on)` 主线。

### v0.1.15.15: route no-op skip v2（仅跳过确定 no-op 调用）收益确认

- 在 route 闭包中增加 `DINF_EPLB_ROUTE_SKIP_NOOP`，仅在以下条件跳过函数调用：
  - `native_record_gate_mode=off` 时跳过 `native_record_gate.update_from_path(...)`；
  - `eplb_cold_rearrange=False` 时跳过 `rearrange_driver.maybe_rearrange(...)`。
- 不改 `set_eplb_runtime_route_path`，不改 map 实现与 EB/s_mask 语义。
- C3 A/B（`batch=128,gen=32,tp=4,world=8,P1+replica,cold_only,flat_eager,cache=off,fastpath=on`）：
  - `off`: `142.767 / 145.005`，mean `143.886 ms/fwd`
  - `on`: `139.067 / 133.958`，mean `136.512 ms/fwd`
  - `on vs off`: `-5.125%`（`-7.374 ms/fwd`）
- 四轮 path counts 全一致：`prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`。
- 结论：
  - 该“no-op skip”版本可保留并进入主线；
  - 与此前失败的 routeopt-v1（含 bool-route setter）不同，本版不改路径上报行为，收益更稳定。

### v0.1.15.15: route no-op skip v2 3x3 稳定性复证（结论修正）

- 在同口径 `C3`（`batch=128,gen=32,tp=4,world=8,P1+replica,cold_only,flat_eager,cache=off,fastpath=on`）完成交错 3x3：
  - `off`: `148.034 / 136.857 / 137.867`，mean/std=`140.919 ± 5.048`
  - `on`: `138.695 / 143.330 / 140.233`，mean/std=`140.753 ± 1.928`
  - `on vs off`: `-0.167 ms/fwd`（`-0.118%`）
- 六组 path counts 全一致：`prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`。
- 结论修正：
  - 该优化在 3x3 复证下未达到可宣称的稳定收益阈值（此前计划阈值 >=2%）；
  - 暂不将其作为主线“收益兑现点”；
  - 可保留为可控开关，仅在后续更长窗口/更大样本下再复核。

### v0.1.15.15: EPLB step 路径诊断（发现主线缺口）

- 对 `window/step` 参数面扫后发现 `overall_skew` 基本不变；进一步诊断确认根因：
  - 当前 dInfer 主线路径未自然推进 `EplbState.step()`，因此 runtime EPLB 重排逻辑并未真正运行。
- 新增实验诊断开关（脚本层）：
  - `--eplb-force-step`（生成后强制一步）
  - `--eplb-step-per-forward`（每次 forward 后推进一步）
- 观测结果：
  - `step hook off`：`state_step` 不前进（`window_step=0,rearrangement_step=12`），`ms/fwd=161.462`
  - `step hook on`：出现 `Rearranging experts ...`，`state_step` 前进（`window_step=13`），但时延显著上升到 `198~217 ms/fwd`，并伴随 `AssertionError()`。
- 结论：
  - 不能直接把 per-forward step 粗暴下沉到主线；
  - 下一步改造方向应是“保守接入 + fail-open”：
    - `cold` 触发
    - 最小间隔
    - 异常回退到 map-only，确保可用性与语义稳定。

### v0.1.15.15: step/rearrange 断言根因定位与开销拆分

- 新增 step 诊断（traceback + state_step）后，已精确定位断言来源：
  - `vllm/distributed/eplb/rebalance_execute.py:292`
  - `assert len(expert_weights) == num_moe_layers`
- 通过实验桥接 MixtureOfExperts 运行时契约（补齐 `moe_layers/expert_weights/set_eplb_state` 并预注册），断言可清零：
  - `stepdiag_on_smoke`: `223.396 ms/fwd`, `errors=2`
  - `stepdiag_on_fixview`: `515.484 ms/fwd`, `errors=0`
- 开销拆分结果（关键）：
  - `step off`: `163.461 ms/fwd`
  - `step on + 大间隔(避免重排)`: `156.454 ms/fwd`（`step_interval=256`）
  - `step on + 可触发重排`: `515.484 ms/fwd`（`step_interval=16`，日志有 `Rearranging experts ...`）
- 结论：
  - 主要成本在 `rearrange_expert_weights_inplace`（专家权重搬运/通信），而非 `step()` 壳本身。
  - 下一阶段必须坚持保守接入策略：`cold + min_gap + fail-open`，禁止无条件 per-forward 重排。

### v0.1.15.15: C12 对齐复测（A/G vs EPLB）

- 按统一 C12 口径（`batch=512, gen=256, tp=4, world=8, no-quality`）完成对齐实验：
  - `bench_bsp_moe_dp2.py`（A/E/G/F）
  - `collect_eb_heteval512_laws.py`（B0: P1,E32；C3: P1+replica,E34）
- 当前机器实测：
  - A: `75.836 ms/fwd`
  - G: `70.030 ms/fwd`
  - B0: `127.408 ms/fwd`
  - C3: `187.862 ms/fwd`
- 相对变化：
  - `B0 vs A`: `+68.00%`
  - `B0 vs G`: `+81.93%`
  - `C3 vs A`: `+147.72%`
  - `C3 vs G`: `+168.26%`
- 四组 path counts 全一致：`prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`。
- 结论：
  - C12 对齐后，当前 EPLB 路线在 law-probe 脚本入口下仍显著慢于 C12 A/G；
  - 后续要兑现“EPLB 追平/超过 C12-G”，需要迁移到 bench 主路径做最小观测负载的 source-level A/B，而非在观察型脚本上继续比较绝对时延。

### v0.1.15.15: bench 主路径 EPLB A/B（C12, bspg_source）

- 完成 `bench_bsp_moe_dp2.py` 最小接入修复：
  - `BSPMSkipController` 增加 `last_path` 状态追踪并在 `get_s_mask` 内维护；
  - baseline 保持 `set_forward_context(...)` 包裹 `experts.forward_impl`。
- 新增 `--results-suffix`，用于保留 off/on 两组独立结果文件，不影响算法逻辑。
- C12 口径主路径 A/B（`batch=512,gen=256,tp=4,world=8,config-set=bspg_source,no-quality`）：
  - A: `75.953 -> 78.671 ms/fwd`（`+3.58%`）
  - E: `71.989 -> 74.910 ms/fwd`（`+4.06%`）
  - G: `69.811 -> 72.804 ms/fwd`（`+4.29%`）
  - GS: `69.626 -> 72.884 ms/fwd`（`+4.68%`）
- off/on path counts 完全一致：
  - `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`
- 结论：
  - bench 主路径已稳定接入 EPLB runtime，但当前实现仍存在约 `+3.6%~+4.7%` 的稳定开销，尚未兑现性能收益。

### v0.1.15.15: 开销归因矩阵（bench 主路径, C12）

- 完成 5 组同口径分解：
  - `OFF32`
  - `ON32-off`（runtime on, redundant=0, record=off）
  - `ON32-cold`
  - `ON34-off`（runtime on, redundant=13, record=off）
  - `ON34-cold`
- 以 G 路径为例的拆分结论：
  - `OFF32 -> ON32-off`: `+3.557 ms`（主开销）
  - `ON32-off -> ON34-off`: `-0.776 ms`（E34 非主矛盾）
  - `ON34-off -> ON34-cold`: `+0.213 ms`（cold record 边际）
  - `OFF32 -> ON34-cold`: `+2.993 ms`
- A/E/G/GS path counts 全一致（`19/171/3933/931`），语义稳定。
- 结论：
  - 当前主要开销来自“runtime enable 常驻路径”（record=off 仍显著变慢），不是 record 冷路径本身，也不是 E34 形状变化；
  - 后续优化优先级应先压 runtime 常驻税，再处理 record 冷路径边际税。

- 进一步确认 `disable_after_build` 已将静态 placement 与 runtime EPLB 分离，下一步应验证 `P1 static-only`（`redundant=0 + disable_after_build`）是否能逼近 baseline.


- 新增 C12 bench 验证：`P1 static-only`（`redundant=0 + disable_after_build`）相对 `OFF32` 仅有 `+0.31% ~ +0.69%` 轻微差距，path counts 保持 `19/171/3933/931`。
- 对照 `ON32-off`（runtime on, record off）仍为 `+4.13% ~ +5.10%`，再次确认主税来自 runtime 常驻路径。
- 当前工程主线可收敛为“静态 P1 默认 + runtime EPLB 可选诊断开关”。

### v0.1.15.15: C12 双轴结果（latency + load-balance）

- 在 `bench_bsp_moe_dp2.py` 落地了负载均衡指标输出：
  - 新增 `RuntimeLayerLoadCollector`（routing 回调处按层计数本 rank 路由负载）；
  - 每个配置结果新增 `load_balance_runs`、`load_balance_summary`。
- 双轴口径 C12 三组（`batch=512,gen=256,tp=4,world=8,num_runs=2,bspg_source,no-quality`）：
  - OFF: A/E/G/GS = `80.549/78.819/76.619/76.640 ms/fwd`
  - ON32-off: `88.094/84.602/82.673/82.463`（相对 OFF `+7.34%~+9.37%`）
  - ON32-cold: `88.043/84.623/82.716/82.435`（与 ON32-off 基本重合）
- 负载均衡指标（`ep_load_cv`）：
  - OFF: `~0.073~0.076`
  - ON32-off/cold: `~0.214~0.215`（较 OFF 增加 `+185%~+193%`）
- 不变性：
  - 三组 path counts 全一致 `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`。
- 结论：
  - 双轴观测能力已可复用；
  - 当前 `redundant=32` runtime-on 路径在本口径下同时造成“时延回退 + 负载均衡恶化”；
  - `cold_only` 与 `off` 差异极小，主问题不在 cold record，而在 runtime-on mapping/placement 路径。

### v0.1.15.15: two-choice 局部性触发（cold-only）验证结论

- 已补齐 two-choice 路径的关键工程缺口：
  - `record_mode=off + map_impl=two_choice_lb` 时，bench route 仍会下发 cold/hot path 信号；
  - runtime map diag 不再在 policy reset 时丢失，结果文件可直接看到 two-choice 实际命中统计。
- 新增局部性控制策略：
  - `DINF_EPLB_TWOCHOICE_COLD_ONLY=1`（默认）：hot 回退 `flat_eager`，cold 才做 two-choice + 本地计数更新。
- smoke 结果（`batch=32,gen=32`）：
  - 全程 two-choice：A `+47.12%`，E/G/GS `+15~17%`（显著负优化）；
  - cold-only two-choice：A/E/G/GS 仍 `+1.09%~+2.29%`（轻微负优化）。
- C12 结果（`batch=512,gen=256,num_runs=1`）：
  - 相对 flat：A `+1.31%`, E `+1.68%`, G `-0.73%`, GS `+0.83%`；
  - 无稳定净收益（仅 G 单点轻微改善）。
- C12 诊断确认算法真实生效（GS）：
  - `twochoice_total=18,055,168`
  - `twochoice_multi=2,275,511`
  - `twochoice_lb_applied=2,275,511`
  - `twochoice_decay_calls=171`, `twochoice_update_calls=171`
- 负载均衡指标仅微弱改善（GS `ep_load_cv 0.215393 -> 0.214857`），不足以覆盖管理开销。
- 结论：
  - “局部性触发”方向语义与工程都可行，但当前 two-choice 路径不具主线性能价值；
  - 主线保持 `flat_eager`，two-choice 作为实验分支保留即可。

### v0.1.15.15: Profiling 专项（单次口径）结果

- 按用户要求改为每组单次，完成 C12 四组 profiling：
  - `OFF`
  - `ON32-flat`（runtime on, redundant=32, record=off）
  - `ON32-cold`（runtime on, redundant=32, record=cold_only）
  - `P1 static-only`（build 后 `disable_after_build`）
- GS 端到端时延：
  - `OFF=87.840 ms/fwd`
  - `ON32-flat=93.426`（`+6.36%`）
  - `ON32-cold=92.295`（`+5.07%`）
  - `P1 static-only=91.046`（`+3.65%`）
- 四组 path counts 全一致：`prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`。
- GS 双轴负载：
  - OFF/P1: `ep_load_cv=0.0305`，`ep_load_max_mean=1.0408`
  - ON32-flat/cold: `ep_load_cv=0.0538`，`ep_load_max_mean=1.0710`
- 组件 rank-max（GS）表明回退主要来自执行时延抬升而非字节量：
  - `moe.native_forward / moe.quant_apply / moe.combine / moe.dispatch` 在 ON32 档位显著抬升；
  - `dispatch_payload/tp_gather_payload/attn_rs_payload` 四组保持一致。
- 结论：
  - 当前瓶颈不是“通信 payload 更多”，而是 runtime-on 路径导致关键执行段更慢；
  - `cold_only` 仅小幅缓解，说明冷路径记录不是主税；
  - 下一步应转向 rank-tail / 等待时间 / 同步点深剖，不再优先做搬运型改造。

### v0.1.15.15: 上下文压缩恢复说明刷新（history-chat 覆盖）

- 按开发规范完成“压缩恢复入口”重写，已覆盖：
  - `/home/wuhang/wuhang/dllm_wh/history-chat.txt`
- 新版恢复说明包含：
  - 必读文档顺序（规范/进展/结论/过程文档/结果文件）；
  - 当前主线代码变更点（`bench_bsp_moe_dp2.py` 的 rank-tail 扩展）；
  - 最新实验证据（`c12_prof_tail_off` vs `c12_prof_tail_on32_cold`）；
  - 关键不变性（path counts）与下一步待办（补齐 ON32-flat/P1-static tail 对照）；
  - 本轮关键命令与恢复后执行顺序。
- 本次仅做归档与恢复说明更新，不改算法语义，不改已有实验结果。

### 本轮命令

- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/docx/next_step.txt`
- `sed -n '1,260p' /home/wuhang/wuhang/dllm_wh/docx/00_docx_storage_and_lookup_guide.md`
- `sed -n '1,220p' /home/wuhang/wuhang/dllm_wh/code_building/progress_diff_summary.md`
- `tail -n 220 /home/wuhang/wuhang/dllm_wh/code_building/key_files_index.md`
- `tail -n 220 /home/wuhang/wuhang/dllm_wh/code_building/key_conclusion.md`
- `tail -n 220 /home/wuhang/wuhang/dllm_wh/progress.md`
- `tail -n 220 /home/wuhang/wuhang/dllm_wh/findings.md`
- `apply_patch` 覆盖 `/home/wuhang/wuhang/dllm_wh/history-chat.txt`

## 2026-05-03 EPLB 因果链分析 + Kernel Micro-benchmark

- 完成 EPLB 在 EP=8 下无效的完整因果链分析（11 组实验）
- Grouped topk 消融：不影响负载均衡
- K=4 确认 + 模型参数修正（hidden_size=2048）
- Kernel sweep (batch=128→2048)：确认 memory→compute-bound 转换点 ~batch=900
- EPLB 在 compute-bound (batch=2048) 下仍无效（spread 恶化 5x）
- Per-layer per-GPU timing：实际不均衡成本仅 0.93 ms (1.2%)
- 因果分析：per-layer r=+0.93（负载驱动），dampening=0.06x
- Active expert sweep + total pairs sweep：证明 kernel_time=f(total_pairs)，分布无关
- 2D sweep (pairs×active_experts)：低 pairs 时 weight loading 有 10-23% 影响
- Tiling config 分析：E=32 的 config 完全复制自 E=256，应做独立 auto-tune
- 结论：EPLB 原理上无法优化 kernel（只改分布不改总量），且实现反而恶化不均衡

### 本轮关键命令

- `torchrun --nproc-per-node=8 bench_bsp_moe_dp2.py --config-set global_topk`（grouped topk 消融）
- `torchrun --nproc-per-node=8 bench_bsp_moe_dp2.py --batch-size {128,256,512,1024,1280,2048} --component-timing --per-layer-timing`（kernel sweep + per-layer timing）
- `torchrun --nproc-per-node=8 bench_bsp_moe_dp2.py --mode active-expert-sweep`（2D sweep）
- `torchrun --nproc-per-node=8 bench_bsp_moe_dp2.py --batch-size 2048 --tp-size 2 --eplb-runtime-enable --eplb-runtime-redundant-experts 32 --eplb-runtime-disable-after-build`（EPLB compute-bound）

## 2026-05-04 Tiling Config Auto-Tune + Dispatch/Combine 根因分析

- Tiling config auto-tune for E=32,N=512：low-M (256-2048) 找到 1.31x kernel 改善，但实际推理 M=16384
- Instrumentation 确认实际 M=16384 (decode) / M=32768 (prefill)，原因是 AgRs AllGather
- High-M grid search (8192-32768)：E=32 config 在实际 M 下已近最优（仅 2-3%）
- 端到端 A/B：tiling config 改变无可测改善 (GS 76.39 → 76.51 ms/fwd)
- Test-C 小 tile (BSM=64 BSN=64)：退化 2.8%（block waste overhead > tiling 收益）
- Pre-filter block waste：退化 1.8%（early-exit 在 H100 零成本，zero-init 有开销）
- Joint stats 联合数据收集：per-layer per-GPU (pairs, active_experts, kernel_time)
  - pairs 跨 GPU 3-5x 不均衡，但 kernel time 仅 5-7% 差异
  - 热点 GPU 跨 forward 完全稳定（确定性）
- Dampening vs batch size：0.10x (b512) → 0.14x (b2048)，远不到线性
- Component timing 跨 batch size：负载不均衡总损失 ~1-2 ms/fwd，占比随 batch 下降
- NCCL micro-bench：AG 9MB 单次 0.249ms (253 GB/s)，fp8 半 payload 3.2x faster
- Overhead isolation：torch.empty 零成本，kernel gap +1ms，fp8 通信省 60%
- Combine straggler 分离：combine 12.7ms = straggler 4.5ms + NCCL 3.8ms + framework 4.4ms
- **完整分解：dispatch+combine 21.7ms = NCCL 7.8ms (36%) + straggler 4.5ms (21%) + framework 8.7ms (42%)**
- torch.compile 测试：vllm 0.11.0 不再 InductorError，但有 graph breaks + forward context 兼容问题
- I25-I27 加入 insight ledger

### 本轮关键命令

- `python autotune_fused_moe_e32.py --mode all --m-values 8192,16384,32768`（high-M grid search）
- `torchrun --nproc_per_node=8 bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --num-runs 2 --mode compare --config-set bspg_source --no-quality`（标准 A/B）
- `VLLM_TUNED_CONFIG_FOLDER=.../tuned_configs_testC torchrun ...`（小 tile A/B）
- `torchrun --nproc_per_node=8 bench_bsp_moe_dp2.py --collect-joint-stats --per-layer-timing --component-timing`（联合数据）
- `torchrun --nproc_per_node=8 nccl_overhead_isolation.py`（开销分离）
- `torchrun --nproc_per_node=8 bench_bsp_moe_dp2.py --component-timing --isolate-combine-wait`（straggler 分离）
- `torchrun --nproc_per_node=8 bench_bsp_moe_dp2.py --torch-compile`（torch.compile 测试）

## 2026-05-05 CUDA Graph 探索 + 完整 Profiling + SP LM Head 优化

- CUDA Graph MoE-only: capture 成功但仅省 1ms/fwd (2%) at batch=512
- CUDA Graph full forward (use_cache=False, batch=64): 30% 加速 (59.5→41.5ms)
- CUDA Graph full forward (batch=512): OOM（67+19 > 80 GB HBM）
- 单图方案（预分配 KV + index_copy + SDPA mask）: SDPA 引入 30ms 退化 > graph 收益
- EB + CUDA Graph at batch=512: 仅省 2ms → **CUDA Graph 方向排除**
- torch.compile: inductor 不兼容 EP dispatch (.cpu() sync) → **排除**
- **根因：batch=512 时 GPU pipeline 隐藏 Python 开销，CUDA Graph 收益上限 ~2ms**
- 完整 GS Path component timing: 首次揭示 76.4 ms 全分解（MoE 41ms + attn 14ms + LM 11.5ms + comm 8.1ms）
- nsys 深度分析: GPU 利用率 71%，单 stream 零 overlap，compute 跨 rank CV<1%
- OPT-2 (skip logits.float): **-4.3 ms (6.1%)** ← 已验证
- OPT-3 (selective LM head): -2.0 ms ← 低于预期
- OPT-4 (fp8 dispatch): +1.5 ms 退化 ← 排除（NVLink 带宽充足）
- OPT-5 (shared expert overlap): ~0 ms ← 排除（GPU pipeline 已隐藏）
- **SP LM Head (clean): -7.7 ms (11.7%)** ← 最大单项收益
- 累积: G path 70.2 → 58.2 ms/fwd (**1.21x 加速**)
- 质量验证通过（heteval512 verifiable prompts 正确）
- 优化路线图文档: docx/plans/2026-05-05_gs_path_optimization_roadmap.md
- 过程文档: code_building/process_docs/v0.1-init-project/v0.1.15.18-cudagraph_profiling_and_sp_lm_head.md

### 本轮关键命令

- `torchrun --nproc_per_node=8 poc_cudagraph_moe.py --batch-size 512`（MoE CUDA Graph）
- `torchrun --nproc_per_node=8 poc_cudagraph_full_forward.py --batch-size 64 --seq-len 64`（full forward graph）
- `torchrun --nproc_per_node=8 poc_cudagraph_eb.py --batch-size 512`（EB + graph）
- `torchrun --nproc_per_node=8 bench_bsp_moe_dp2.py --config-set bspg_source --component-timing`（完整 profiling）
- `TMPDIR=/home/wuhang/tmp_nsys nsys profile --trace=cuda,nvtx torchrun ...`（nsys）
- `DINF_SKIP_LOGITS_FLOAT=1 torchrun ... --no-quality`（OPT-2）
- `DINF_SKIP_LOGITS_FLOAT=1 DINF_SP_LM_HEAD=1 torchrun ... --no-quality`（OPT-2 + SP-LM）

## 2026-05-06 BSP-H 探索 + TEAM Decoded-Token Skip

- BSP-H AllReduce 方案：GS 59.24 → H 57.08 ms/fwd (-2.17 ms, -3.7%)，但收益主要来自消除 attention input TP AllGather，AllReduce 本身仅省 0.5 ms
- BSP-H 概念重审：原始定义是 hierarchical collective fusion，不是 AllReduce 替换。TP attention 结构性要求 token AllGather 无法消除
- TEAM 组件分析：dLLM decoded token 的 MoE 计算对最终输出零贡献，可用缓存替代。与之前失败的 Stable Cache 的关键区别：TEAM 不跳过 MASK 位置
- TEAM v1 (extract+pad+context)：skip_ratio=55.1% 但退化 9.4 ms，根因是 per-layer AllReduce×38 开销
- TEAM null expert 方案：扩展 expert_map[257] 加 null expert，kernel 验证通过（精确零），但完整流程质量崩坏，正在 debug
- Insight ledger 更新：I28-I33 补齐（tiling config、dispatch/combine 分解、CUDA Graph、GPU 利用率、SP LM Head、fp8/overlap 排除）
- Key files index 更新：2026-05-04 和 2026-05-05 section 补齐

### 本轮关键命令

- `torchrun --nproc_per_node=8 bench_bsp_moe_dp2.py --config-set bspg_h --batch-size 512 --gen-length 256 --num-runs 2 --no-quality`（BSP-H A/B）
- `torchrun --nproc_per_node=8 bench_bsp_moe_dp2.py --config-set bspg_team --batch-size 512 --gen-length 256 --num-runs 2 --no-quality`（TEAM A/B）
- `torchrun --nproc_per_node=8 bench_bsp_moe_dp2.py --config-set bspg_team --component-timing`（TEAM component）

## 2026-05-07 TEAM Null Expert 质量排查 + MoE Cache 状态机 + Piggyback Dispatch

- 完成 TD1-TD7b 系列消融实验，确认 V3(topk_ids[decoded]=256) 是质量崩坏唯一根因
- TD7b 关键实验：MASK 输出在 null expert vs 正常路由下精确一致(diff=0)，证明 kernel 不影响 MASK
- 质量崩坏机制定位：decoded=0 通过 residual→attention 跨层传播间接污染 MASK 计算
- Cross_block forward 发现：Block 1+ 首个 forward 处理 prev_block+curr_block (N_sp=4096)，用于 KV cache 刷新
- prev_decoded 状态机设计：null_mask = decoded & prev_decoded & (step%M!=0)，M=5 周期刷新
- Piggyback dispatch 方案：decoded mask 搭载 router_logits 额外列，零额外 NCCL collective
- NCCL deadlock 排查：is_hot 判断不一致 + dist.all_gather 固定大小要求 → 改为 piggyback 方案解决
- TV3 sparse dispatch/combine 尝试：payload -81% 但 Python overhead +12.7 ms > NCCL 节省 10.6 ms → 暂搁置
- TD4 最终状态：质量接近 G（#0/#13 逐字一致），性能 73.79 ms（比 G 慢 24%），overhead 来自 cache clone + torch.cat
- G 路径隔离确认：restore_blocks_and_experts 恢复 gate.get_logits + decoder.block_init

### 本轮关键命令

- `DINF_SKIP_LOGITS_FLOAT=1 DINF_SP_LM_HEAD=1 torchrun --nproc_per_node=8 bench_bsp_moe_dp2.py --config-set team_debug --profile-target bsp --batch-size 512 --gen-length 256 --num-runs 2`（TD4 perf A/B）
- `同上 --num-runs 1`（TD4 质量验证，带 heteval512 输出）
- `同上 --component-timing`（TD4 component timing）
- `同上 --config-set bspg_team`（TEAM cache-only 对比）

## 2026-05-08 TEAM Sparse Kernel 多路径探索

### 进展

- TD4 cache write 逻辑修正 + 删 `.any()`：72.93 ms/fwd (+23.6%)
- TD8 无 cache 实验：质量崩坏，确认 cache 必要
- TV4 sparse kernel 原版：65.69 ms/fwd (+11.3%)
- TV4 buffer 优化（预分配 + index_select out=）：60.60 ms/fwd (+1.2%)
- TV4c Phase 1 AllReduce（无改善，回退）
- TV4c Phase 2 compact dispatch（sp_local_sizes 不兼容，回退）
- **TV5 topk_ids skip：59.71 ms/fwd (+0.66%) — 当前最优**
- nsys 三方对比证明 CPU launch overhead 是瓶颈（+418k launches = +2.3s CPU）
- CUDA Event 实验证明 TV4 kernel phase 比 G 快 40%（native 集成理论 -16%）
- Triton gather kernel / shuffle_rows 均比 PyTorch index_select 慢（回退）
- TV4m mapped kernel（crash 待调试，理论最优 -16%）
- 发现 vllm moe_align_block_size C++ kernel 有 `expert_id >= num_experts → continue` 保护

### 关键结论

- TV5 利用 `topk_ids >= num_experts` 让 kernel 自动跳过 null tokens，零 extract/scatter
- NVLink 8-GPU batch=512 下通信不是瓶颈，kernel time 节省被 GPU pipeline 隐藏
- monkey-patch 性能天花板 ≈ G + 0.66%（CPU launch overhead 不可消除）
- native 集成可释放 TV4 的 16% 理论收益

### 本轮关键命令

- `DINF_SKIP_LOGITS_FLOAT=1 DINF_SP_LM_HEAD=1 torchrun --nproc_per_node=8 bench_bsp_moe_dp2.py --config-set team_tv4 --profile-target bsp --batch-size 512 --gen-length 256 --num-runs 2 --no-quality`
- `同上 --config-set team_tv5`（TV5 性能）
- `同上 --config-set team_g_only`（G 基线）
- `DINF_KERNEL_PHASE_TIMING=1 同上 --config-set team_kp --num-runs 1`（kernel phase timing）
- `nsys profile -o /tmp/nsys_team/tv4_fresh --trace=cuda,nvtx,osrt --cuda-memory-usage=false torchrun ...`（nsys profiling）

## 2026-05-08 ~ 2026-05-09 TV4m Crash Fix + TV6 Compact Dispatch

### TV4m Crash 修复
- 修复 5 个 bug：padding OOB、C stride 错误、B stride 反了、expert_ids=-1 未处理、**N=w13.shape[2]→shape[1] 维度索引笔误**（ROOT CAUSE）
- N 维度错误导致 silu_and_mul 读取未初始化内存 → 质量崩坏（forward 数 2x）
- 修复后 TV4m: **57.9 ms/fwd (-2.5% vs G)**，质量验证通过
- TV4m 预分配优化：per-layer alloc → 首次分配后复用，省 ~1ms

### TV4m-v2（暂停）
- full-layout moe_align + token_remap kernel
- Triton constexpr + padding 问题导致 crash，收益仅 ~0.3ms
- 暂停，代码保留

### TV6 Compact Dispatch/Combine
- vllm 原生支持 all_gatherv/reduce_scatterv（variable-size collectives）
- `test_compact_collective.py` 8 个测试全 PASS（含动态 sizes、5×19 层交替等）
- TV6 经历 5 个版本迭代（v1→v5），从 +5.8% 优化到 **-1.3% vs G**
- 关键优化：boolean→integer indexing 消除隐式 sync（-1.9ms）、in-place cache 消除 3 kernel/layer（-0.7ms）

### nsys Profiling 三方对比
- TV6 GPU kernel 端节省 **-28.2s**（是 TV4m -13.0s 的 2.2 倍）
- TV6 compact dispatch 省 AllGather -20.0s，compact combine 省 Reduce -3.3s
- 但 GPU idle 从 36.4s(G) 增加到 42.4s(TV6)——Python overhead 吃掉了 GPU 节省
- 5-50μs gap 增量 +2.8s（per-layer Python tensor ops）
- 200μs-1ms gap 增量 +2.0s（compact collective Python wrapper + 隐式 sync）

### 当前最优配置
- **TV4m: 57.9 ms/fwd (-2.0% vs G)** — monkey-patch 阶段最快
- **TV6: 58.3 ms/fwd (-1.3% vs G)** — compact dispatch，GPU 端收益更大，但 Python overhead 抵消
- 两者均通过质量验证（path counts 19/171/3933/931，verifiable prompts 正确）

### 本轮关键命令
- `torchrun --nproc_per_node=8 bench_bsp_moe_dp2.py --config-set team_tv4m --profile-target bsp --batch-size 512 --gen-length 256 --num-runs 2 --no-quality`
- `同上 --config-set team_tv6`
- `python test_triton_mapped_kernel.py --real`（standalone kernel test with dumped data）
- `torchrun --nproc_per_node=8 test_compact_collective.py`（NCCL variable-size test）
- `TMPDIR=/tmp/nsys_team nsys profile -o /tmp/nsys_team/tv6d_v6cmp -t cuda -w true --sample=none --cpuctxsw=none --trace-fork-before-exec=true torchrun ...`


## 2026-05-10 论文实验：动机图 + TV6 Patch + 质量验证

### 动机图数据采集与绘图
- 采集 expert routing stability 数据（Jaccard heatmap + CDF），绘制 Fig.3
- 采集 MoE output stability 数据（Decoded vs MASK cos_sim），绘制 Fig.4 箱型图
- 采集 cache staleness 数据（per-step cache quality decay），绘制 Fig.5 折线图
- 三张动机图完整支撑 EB + decoded-skip 机制的合理性

### TV6 Patch 独立模块
- 从 bench_bsp_moe_dp2.py 提取 TV6 为独立模块 tv6_patch.py
- 直接 import bench 脚本模块级函数（不重写），确保完全一致
- 性能验证：tv6_patch 58.36 ms/fwd vs bench 57.96 ms/fwd（差 0.4ms）

### 性能 A/B 验证（batch=512, heteval512）
- Vanilla (K=8): 85.39 ms/fwd, 266 fwd
- A (EB K=4): 71.53 ms/fwd, 266 fwd (-16.2%)
- G (BSP-G): 58.39 ms/fwd, 266 fwd (-31.6%)
- TV6 (compact): 58.36 ms/fwd, 266 fwd (-31.7%)
- fwd count 全部一致（266），EB K=4 不影响 decoding 行为

### 质量验证
- bench 框架肉眼检查：Vanilla / A / G / TV6 输出语义一致，质量无退化
- GSM8K batch=512：两者数学推理正确，fwd 差距 3.7%（K=4 vs K=8 routing 差异）
- TV6 加速随 batch 增大而增强：batch=32 无加速，batch=512 -32.5%
- lm-eval 路线因 TP 拓扑不兼容暂搁置

### 本轮关键命令
- `torchrun --nproc_per_node=8 codex_coding/src/collect_routing_stability.py --batch-size 32 --gen-length 256 --block-length 32`
- `torchrun --nproc_per_node=8 codex_coding/src/collect_moe_output_stability.py`
- `torchrun --nproc_per_node=8 codex_coding/src/collect_cache_staleness.py`
- `python codex_coding/src/plt/fig3.py && python codex_coding/src/plt/fig4.py && python codex_coding/src/plt/fig5_2.py`
- `DINF_SKIP_LOGITS_FLOAT=1 DINF_SP_LM_HEAD=1 torchrun --nproc_per_node=8 codex_coding/src/bench_tv6_throughput.py --batch-size 512 --gen-length 256 --num-runs 2 --no-quality`
- `同上 --baseline-only`（Vanilla K=8 baseline）
- `同上 --prompt-source gsm8k`（GSM8K 质量验证）

## 2026-05-11 Baseline Bench + 跨框架 Benchmark 对比

- 完成 baseline_dInfer 和修改版 dInfer 的模型文件差异分析：baseline 用 nn.Linear + 后置 tensor_parallel()，修改版用 QKVParallelLinear 原生 TP；baseline 最大 TP=4（kv_heads=4），不支持 DP>1
- 重写 `bench_baseline_dinfer.py`：torchrun 启动，TP=4，runtime shim is_torch_fx_available/ROPE_INIT_FUNCTIONS，支持 --pad-to/--no-cache/--prompt-source
- 新建 `bench_sglang_dllm.py`：sgl.Engine 离线模式，支持 --disable-cuda-graph/--attention-backend/--pad-to
- 给 `bench_tv6_throughput.py` 加 --pad-to 和 --tp-size 参数
- 修复 TV6 在 DP=1 下的兼容性（tv6_patch.py 三处改动：初始化 all2all_manager、创建临时 DPMetadata、fallback 路径 _run_full_moe）
- 在 low_confidence.py 加 _FWD_COUNTER 计数器用于获取 SGLang 的 fwd count
- 确认两边 Attention 实现：baseline 和修改版都初始化为 SdpaAttention；修改版 apply_all_optimizations 替换 forward 为 flash_attn（有 attn_mask 时 fallback 到 SDPA）；BlockDiffusionLLM prefix cache 路径下大部分 forward 不传 attn_mask → 走 flash_attn
- 确认 dInfer 无 intra-block batch shrinking（select_undecoded 空实现）；Fast-dLLM 有 block 间 batch shrinking；SGLang 有请求级动态调度
- 完成统一配置下四种引擎的 benchmark 数据采集（GSM8K, pad=128, gen=256, block=32, threshold=0.90, TP=4）

### 本轮关键命令

- `conda run -n dllm_base torchrun --nproc_per_node=4 codex_coding/src/bench_baseline_dinfer.py --batch-size {32,64,128,256} --gen-length 256 --num-runs 2 --no-quality --prompt-source gsm8k --pad-to 128`
- `conda run -n dllm_base torchrun --nproc_per_node=4 codex_coding/src/bench_baseline_dinfer.py --batch-size {32,64} --gen-length 256 --num-runs 2 --no-quality --prompt-source gsm8k --pad-to 128 --no-cache`
- `DINF_SKIP_LOGITS_FLOAT=1 DINF_SP_LM_HEAD=1 torchrun --nproc_per_node={4,8} codex_coding/src/bench_tv6_throughput.py --batch-size {32,256} --gen-length 256 --num-runs 2 --no-quality --prompt-source gsm8k --pad-to 128 --tp-size 4 --{baseline-only,tv6-only}`
- `conda run -n dllm_sglang python codex_coding/src/bench_sglang_dllm.py --batch-size 32 --gen-length 256 --num-runs 2 --no-quality --prompt-source gsm8k --pad-to 128 --tp-size 4 --disable-cuda-graph --attention-backend {flashinfer,torch_native}`
