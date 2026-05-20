# Learnings

Corrections, insights, and knowledge gaps captured during development.

**Categories**: correction | insight | knowledge_gap | best_practice

---

## [LRN-20260428-001] correction

**Logged**: 2026-04-28T17:44:45+08:00
**Priority**: low
**Status**: resolved
**Area**: docs

### Summary
Use `DInferCompileBackend`, not the typo `DInferConpileBackend`, for the dInfer/vLLM compile adapter.

### Details
The initial AsyncTP adapter/probe was named `DInferConpileBackend` during discussion. The user corrected the spelling to `DInferCompileBackend`. Source files and new probes were renamed to `compile`; historical result artifacts may still contain `conpile` in their filenames.

### Suggested Action
Use `dinfer.compile_backend.DInferCompileBackend` in new code and documentation. Treat `conpile` references as historical-only unless explicitly discussing the original artifact.

### Metadata
- Source: user_feedback
- Related Files: lib_cite/dInfer/python/dinfer/compile_backend.py, codex_coding/src/probe_dinfer_compile_backend_async_tp.py
- Tags: naming, async-tp, dinfer

---
