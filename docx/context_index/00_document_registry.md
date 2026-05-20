# Document Registry

This registry lists the documents with real recovery or traceability value for the current `v0.1-init-project` stage. Use it as a lookup aid; process docs and result files remain the evidence source.

## Active Recovery Documents

| Path | Role | Topic | Priority | Key Conclusions |
| --- | --- | --- | --- | --- |
| `/home/wuhang/wuhang/dllm_wh/docx/next_step.txt` | stage_entry | Current stage and development rules | high | Defines `v0.1-init-project`, archive rules, command logging requirements, and directory conventions. |
| `/home/wuhang/wuhang/dllm_wh/history-chat.txt` | recovery_handoff | Current context compression recovery block | high | Paste-ready handoff for the next compressed session; emphasizes Scheme3 8-GPU conclusions and insight-led direction selection. |
| `/home/wuhang/wuhang/dllm_wh/docx/00_docx_storage_and_lookup_guide.md` | guide | `docx/` storage and lookup guidance | high | Defines what belongs in `docx/`, `articles/`, `context_index/`, and what belongs in `code_building/` or `codex_coding/`. |
| `/home/wuhang/wuhang/dllm_wh/docx/context_index/01_stage_summary.md` | context_index | Current stage summary | high | Current stage is direction selection after C12-AgRs Scheme3 8-GPU validation; next step is discussion, not code. |
| `/home/wuhang/wuhang/dllm_wh/docx/context_index/02_active_threads.md` | context_index | Active research/development threads | high | Tracks active direction selection, deprioritized Scheme3 standalone path, and warm SP/native-expert/scheduler threads. |
| `/home/wuhang/wuhang/dllm_wh/docx/context_index/03_current_required_process_docs.md` | context_index | Minimal required process-doc reading set | high | Lists the smallest sufficient process-doc set for recovery into the current Scheme3 / insight-led direction-selection stage. |
| `/home/wuhang/wuhang/dllm_wh/docx/context_index/04_insight_ledger.md` | living_index | dLLM / MoE / EB insight ledger | high | Canonical maintained table of supported/revised/rejected insights, source links, and candidate direction buckets. |
| `/home/wuhang/wuhang/dllm_wh/code_building/key_conclusion.md` | living_summary | Key project conclusions | high | Long-lived key conclusions; direction selection should first consult the insight ledger. |
| `/home/wuhang/wuhang/dllm_wh/code_building/progress_diff_summary.md` | progress_log | Incremental progress summary | high | Append-only progress deltas, including Scheme3 8-GPU results and insight ledger archive. |

## Current Process Documents

| Path | Role | Topic | Priority | Key Conclusions |
| --- | --- | --- | --- | --- |
| `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12i-context_compression_recovery_handoff.md` | process_doc | Current compression handoff update | high | Records the update to `history-chat.txt` and refreshed recovery indexes for the current direction-selection stage. |
| `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12h-insight_ledger_archive.md` | process_doc | Insight ledger archive | high | Records creation of `04_insight_ledger.md` as the canonical direction-selection ledger. |
| `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12g-scheme3_8g_timing_and_quality.md` | process_doc | Scheme3 8-GPU timing and quality | high | Confirms Scheme3 payload saving is real but standalone wall-clock benefit is negative under C12-AgRs; no Scheme3-specific quality degradation observed. |
| `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12-vllm_upgrade_and_scheme3_design.md` | process_doc | vLLM 0.11.0 AgRs and Scheme3 design | high | Explains AgRs AllToAll behavior, Scheme3 route-before-dispatch design, and early monkey-patch pitfalls. |
| `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.11-c11_c12_tp_attention_and_insight_optimization.md` | process_doc | C11/C12 and insight validation | high | Defines C11/C12, validates E1-E5 insights, and records C12 `dp=2,tp=4,ep=8` behavior. |
| `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.8n-ncu_profiling_tiling_and_multi_gpu.md` | process_doc | Kernel profiling and multi-GPU evidence | medium-high | Provides NCU/kernel evidence that fused MoE wall time is often expert-weight HBM-load bound. |
| `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.8-eb_hot_path_optimization_and_batch_scaling.md` | process_doc | EB hot path and batch scaling | medium-high | Records EB/S_mask stability and batch-size-dependent value. |

## Current Scripts and Results

| Path | Role | Topic | Priority | Key Conclusions |
| --- | --- | --- | --- | --- |
| `/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_scheme3_dp2.py` | runnable_script | Scheme3 A/B/B2 benchmark | high | Main script for C12-AgRs Scheme3 comparison and component timing. |
| `/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_dp2_tp4_ep8.py` | runnable_script | C12 baseline benchmark | high | C12 baseline runner for `dp=2,tp=4,ep=8`. |
| `/home/wuhang/wuhang/dllm_wh/codex_coding/results/scheme3_dp2_results.json` | result_summary | Scheme3 8-GPU end-to-end result | high | A `75.62 ms/fwd`, B `79.61 ms/fwd`, B2 `81.35 ms/fwd`. |
| `/home/wuhang/wuhang/dllm_wh/codex_coding/results/scheme3_dp2_component_timing_8g.json` | result_summary | Scheme3 8-GPU component timing | high | Dispatch payload drops from about `826.9` to `706 MB/fwd`, but dispatch time improves only about `0.45-0.52 ms/fwd`. |
| `/home/wuhang/wuhang/dllm_wh/codex_coding/results/scheme3_dp2_8g_component_timing_20260427_034930.log` | raw_log | Scheme3 component timing stdout | medium | Contains printed component timing and verifiable quality snippets. |
| `/home/wuhang/wuhang/dllm_wh/codex_coding/results/scheme3_dp2_8g_e2e_20260427_035313.log` | raw_log | Scheme3 end-to-end stdout | medium | Contains end-to-end run output and manual-quality snippets. |

## Historical High-Value Documents

| Path | Role | Topic | Priority | Key Conclusions |
| --- | --- | --- | --- | --- |
| `/home/wuhang/wuhang/dllm_wh/docx/articles/2026-04-12_dllm_moe_complete_experiment_report.md` | technical_article | dLLM MoE characterization and experiments | medium | Records MASK routing concentration, failed stable cache, padding-free MoE failure, and broader MoE insight context. |
| `/home/wuhang/wuhang/dllm_wh/docx/articles/2026-04-12_dllm_moe_insight_and_optimization_directions.md` | technical_article | dLLM MoE insight directions | medium | Early articulation of dLLM-specific MoE differences and optimization hypotheses. |
| `/home/wuhang/wuhang/dllm_wh/docx/cites/moe_systems_survey.md` | literature_notes | MoE systems survey | medium | Survey context for MoE systems, used as background rather than immediate recovery input. |

## Notes

- Historical documents may still mention early dInfer/LLaDA2 work. Treat them as background unless the user explicitly returns to that line.
- The current high-priority recovery path is Scheme3 8-GPU conclusion -> insight ledger -> next-direction discussion.
- Do not use this registry as a replacement for reading the required process docs when making technical claims.
