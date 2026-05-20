# Project Scope from Citation Notes

## Goal

Infer the project's intended direction by reading the three planning notes under `docx/cites/`.

## Changes Made

- Read the three project notes:
  - `docx/cites/sc_paper_architecture.md`
  - `docx/cites/subgraph_layout_ir_theory.md`
  - `docx/cites/sglang_poc_build_plan.md`
- Extracted the common project definition:
  - This is a research-oriented systems project, not a generic product build.
  - The core target is subgraph-aware layout optimization for LLM inference.
  - The host serving system is SGLang.
  - The kernel substrate is Triton.
  - The main novelty is to optimize layouts jointly across hotspot subgraphs instead of per operator.
- Extracted the practical project boundary:
  - Focus on quantized FFN / expert FFN first, then attention preparation or epilogue chains.
  - Avoid full scheduler redesign, KV-cache redesign, distributed MoE redesign, or Triton core compiler forking.
  - Build a planner plus a controlled set of layout-specialized Triton kernel families.
- Extracted the intended output:
  - a working POC integrated into SGLang,
  - benchmark and profiling evidence,
  - and an SC-style systems paper narrative.

## Verification

- Confirmed the three files exist under `docx/cites/`.
- Read the front and back sections of each document.
- Cross-checked section headings to ensure the interpretation matches all three notes.

## Result

The project is best understood as a research POC for an SC-style paper: build a subgraph-level layout planning layer on top of SGLang and Triton to improve LLM inference by reducing layout mismatch, conversion overhead, and missed fusion opportunities in selected hotspot subgraphs.

## Next Steps

- Map the current `triton/` codebase and nearby integration points to this planned architecture.
- Identify whether the active implementation target is the planner side, kernel side, or benchmark/instrumentation side.

## 本轮命令

- `ls -la /home/wuhang/wuhang/linear_wh/docx/cites`
- `wc -l /home/wuhang/wuhang/linear_wh/docx/cites/*.md`
- `sed -n '1,260p' /home/wuhang/wuhang/linear_wh/docx/cites/sc_paper_architecture.md`
- `sed -n '1,260p' /home/wuhang/wuhang/linear_wh/docx/cites/subgraph_layout_ir_theory.md`
- `sed -n '1,260p' /home/wuhang/wuhang/linear_wh/docx/cites/sglang_poc_build_plan.md`
- `rg '^#' /home/wuhang/wuhang/linear_wh/docx/cites/sc_paper_architecture.md /home/wuhang/wuhang/linear_wh/docx/cites/subgraph_layout_ir_theory.md /home/wuhang/wuhang/linear_wh/docx/cites/sglang_poc_build_plan.md`
- `sed -n '261,360p' /home/wuhang/wuhang/linear_wh/docx/cites/sc_paper_architecture.md`
- `sed -n '261,430p' /home/wuhang/wuhang/linear_wh/docx/cites/subgraph_layout_ir_theory.md`
- `sed -n '261,460p' /home/wuhang/wuhang/linear_wh/docx/cites/sglang_poc_build_plan.md`
