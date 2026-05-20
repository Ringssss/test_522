# LinearLayout Article Archive

## Goal

Archive the current understanding of Triton `LinearLayout` and its relevance to subgraph-level model optimization into a durable technical article under `docx/articles/`.

## Changes Made

- Inspected the current archive layout under `/home/wuhang/wuhang/linear_wh/docx`.
- Confirmed there is a historical typo directory `/home/wuhang/wuhang/linear_wh/docx/acticles`.
- Created the standard archive directories:
  - `/home/wuhang/wuhang/linear_wh/docx/articles`
  - `/home/wuhang/wuhang/linear_wh/docx/context_index`
- Added the technical archive:
  - `/home/wuhang/wuhang/linear_wh/docx/articles/2026-03-27_linear_layout_and_subgraph_optimization.md`
- Refreshed the durable recovery index files:
  - `/home/wuhang/wuhang/linear_wh/docx/context_index/00_document_registry.md`
  - `/home/wuhang/wuhang/linear_wh/docx/context_index/01_stage_summary.md`
  - `/home/wuhang/wuhang/linear_wh/docx/context_index/02_active_threads.md`
  - `/home/wuhang/wuhang/linear_wh/docx/context_index/03_current_required_process_docs.md`

## Verification

- Confirmed the archive skill requirements and update rules from the local skill references.
- Confirmed the new article path and context index paths are under `/home/wuhang/wuhang/linear_wh/docx/`.
- This round produced documentation only. No code files, test scripts, or result artifacts were changed.

## Result

The current `LinearLayout` investigation is now archived as a reusable technical article with matching process documentation and recovery index files, so the discussion can resume from disk rather than from transient chat context.

## Next Steps

- Continue the design discussion from the archived article, focusing on how to define a subgraph-level layout state and legality relation.
- Decide whether the next round should stay at the research-design level or start mapping the idea into concrete code locations.

## 本轮命令

- `sed -n '1,220p' /home/wuhang/.codex/suss-skills/archive-progress/SKILL.md`
- `sed -n '1,220p' /home/wuhang/.codex/suss-skills/building-rules/SKILL.md`
- `find /home/wuhang/wuhang/linear_wh/docx -maxdepth 3 | sort`
- `find /home/wuhang/wuhang/linear_wh/code_building -maxdepth 3 -type f | sort`
- `sed -n '1,220p' /home/wuhang/.codex/suss-skills/archive-progress/references/archive_checklist.md`
- `sed -n '1,260p' /home/wuhang/.codex/suss-skills/archive-progress/references/file_update_rules.md`
- `find /home/wuhang/wuhang/linear_wh/docx/context_index -maxdepth 2 -type f | sort`
- `mkdir -p /home/wuhang/wuhang/linear_wh/docx/articles /home/wuhang/wuhang/linear_wh/docx/context_index`
