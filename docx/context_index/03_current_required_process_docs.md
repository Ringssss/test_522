# Current Required Process Docs

Only the following process documents are required to continue the current task efficiently. This list is intentionally focused on the current Scheme3 / insight-led direction-selection stage.

## Required Docs

- `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12i-context_compression_recovery_handoff.md`
  - Required because it records the current compression handoff update and the refreshed recovery intent.
- `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12h-insight_ledger_archive.md`
  - Required because it records the creation of the canonical insight ledger and explains its role in direction selection.
- `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12g-scheme3_8g_timing_and_quality.md`
  - Required because it contains the decisive 8-GPU Scheme3 timing, dispatch payload, path count, and manual quality-check results.
- `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12-vllm_upgrade_and_scheme3_design.md`
  - Required because it explains vLLM 0.11.0 AgRs behavior, the Scheme3 route-before-dispatch design, and why the original monkey-patch path became suspect.
- `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.11-c11_c12_tp_attention_and_insight_optimization.md`
  - Required because it defines C11/C12, records C12 `dp=2,tp=4,ep=8`, and captures the E1-E5 insight validation that led to Scheme3 and later direction choices.
- `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.8n-ncu_profiling_tiling_and_multi_gpu.md`
  - Required because it records NCU/kernel evidence behind memory-bound fused MoE and the need to distinguish FLOP savings from wall-clock savings.
- `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.8-eb_hot_path_optimization_and_batch_scaling.md`
  - Required because it records the EB hot-path and batch-scaling evidence behind `S_mask` stability and EB value at larger batch.

## Must-Read Non-Process Files

- `/home/wuhang/wuhang/dllm_wh/docx/next_step.txt`
  - Current project rules and archive conventions.
- `/home/wuhang/wuhang/dllm_wh/history-chat.txt`
  - Paste-ready context recovery block for the next compressed session.
- `/home/wuhang/wuhang/dllm_wh/docx/context_index/04_insight_ledger.md`
  - Canonical insight ledger for direction selection.
- `/home/wuhang/wuhang/dllm_wh/code_building/key_conclusion.md`
  - Long-lived key conclusion file.
- `/home/wuhang/wuhang/dllm_wh/code_building/progress_diff_summary.md`
  - Incremental progress history and latest deltas.

## Not Included On Purpose

- Early dInfer/LLaDA2 process docs are omitted because they are historical and not needed for the immediate direction-selection discussion.
- Raw result files are omitted from this process-doc list; the decisive raw result paths are referenced from the required process docs and `history-chat.txt`.
- Older `linear_wh` context is irrelevant to this active project thread.
