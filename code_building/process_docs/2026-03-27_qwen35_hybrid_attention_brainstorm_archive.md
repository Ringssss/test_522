# Qwen3.5 Hybrid Attention Brainstorm Archive

## Goal

Archive the current brainstorming about Qwen3.5-style hybrid attention optimization into a durable technical article under `docx/articles/`.

## Changes Made

- Reviewed the current archive index files under `/home/wuhang/wuhang/linear_wh/docx/context_index/`.
- Added the technical brainstorming article:
  - `/home/wuhang/wuhang/linear_wh/docx/articles/2026-03-27_qwen35_hybrid_attention_brainstorm.md`
- Refreshed the archive memory files to reflect the task shift from `LinearLayout` discussion to Qwen3.5 model-oriented initialization and hybrid-attention analysis.
- Recorded an explicit caveat in the article that the earlier runtime outputs were not yet persisted under `codex_coding/results`, and that future runs should store scripts and outputs there.

## Verification

- Confirmed the new article exists under `/home/wuhang/wuhang/linear_wh/docx/articles/`.
- Confirmed the article content captures:
  - verified model characteristics,
  - literature-direction summary,
  - hybrid-attention system characteristics,
  - optimization-space brainstorming,
  - and persistence caveats for later benchmark runs.
- No code files or test scripts were modified in this round.

## Result

The Qwen3.5 hybrid-attention discussion is now archived in a dedicated brainstorming document, and the project memory files now reflect the current research direction more accurately.

## Next Steps

- Build a reproducible benchmark harness under `/home/wuhang/wuhang/linear_wh/codex_coding/src`.
- Persist future online benchmark outputs and request samples under `/home/wuhang/wuhang/linear_wh/codex_coding/results`.
- Start profiling hybrid linear-attention layers, periodic full-attention layers, and MoE execution separately.

## 本轮命令

- `find /home/wuhang/wuhang/linear_wh/docx/articles -maxdepth 2 -type f | sort`
- `find /home/wuhang/wuhang/linear_wh/docx/context_index -maxdepth 2 -type f | sort`
- `sed -n '1,220p' /home/wuhang/wuhang/linear_wh/docx/context_index/00_document_registry.md`
- `sed -n '1,220p' /home/wuhang/wuhang/linear_wh/docx/context_index/01_stage_summary.md`
- `sed -n '1,220p' /home/wuhang/wuhang/linear_wh/docx/context_index/02_active_threads.md`
- `sed -n '1,220p' /home/wuhang/wuhang/linear_wh/docx/context_index/03_current_required_process_docs.md`
