# Linear State Prefix Cache Brainstorm Archive

## Goal

Archive the current discussion about full-attention prefix cache, radix cache, linear state, and the proposed CPU/host-side linear-state checkpoint pool.

## Changes Made

- Added the technical article:
  - `/home/wuhang/wuhang/linear_wh/docx/articles/2026-03-27_linear_state_prefix_cache_brainstorm.md`
- Added a persistent summary of the current Qwen3.5 online serving baseline:
  - `/home/wuhang/wuhang/linear_wh/codex_coding/results/2026-03-27_qwen35_online_baseline_summary.md`
- Prepared a restart-safe recovery note to support upcoming context compression:
  - `/home/wuhang/wuhang/linear_wh/docx/2026-03-27_context_compression_recovery_note.md`
- Refreshed the archive index files to shift the active thread from generic hybrid-attention brainstorming toward prefix/radix cache and linear-state-aware checkpoint design.

## Verification

- Confirmed the article exists under `/home/wuhang/wuhang/linear_wh/docx/articles/`.
- Confirmed the baseline summary exists under `/home/wuhang/wuhang/linear_wh/codex_coding/results/`.
- Confirmed the recovery note exists under `/home/wuhang/wuhang/linear_wh/docx/`.
- The article captures:
  - ordinary full-attention radix-cache behavior,
  - hybrid linear-state caching differences,
  - local memory numbers,
  - and the current evaluation of the CPU/host checkpoint pool idea.

## Result

The current cache-focused research thread is now archived in durable documents, and the benchmark observations are no longer only present in transient chat context.

## Next Steps

- After context compression, recover from the new recovery note first.
- Then continue from the active question:
  - whether to formalize a `Hybrid Prefix Checkpoint Cache` design around GPU/CPU tiers, checkpoint placement, and fetch-vs-recompute policy.

## 本轮命令

- `sed -n '1,240p' /home/wuhang/.codex/suss-skills/recovery-handoff/SKILL.md`
- `find /home/wuhang/wuhang/linear_wh/code_building/process_docs -maxdepth 1 -type f | sort`
- `find /home/wuhang/wuhang/linear_wh/docx/articles -maxdepth 1 -type f | sort`
- `sed -n '1,220p' /home/wuhang/wuhang/linear_wh/code_building/progress_diff_summary.md`
- `sed -n '1,240p' /home/wuhang/wuhang/linear_wh/docx/next_step.txt`
- `sed -n '1,220p' /home/wuhang/wuhang/linear_wh/docx/context_index/00_document_registry.md`
- `sed -n '1,220p' /home/wuhang/wuhang/linear_wh/docx/context_index/01_stage_summary.md`
- `sed -n '1,220p' /home/wuhang/wuhang/linear_wh/docx/context_index/02_active_threads.md`
- `sed -n '1,220p' /home/wuhang/wuhang/linear_wh/docx/context_index/03_current_required_process_docs.md`
