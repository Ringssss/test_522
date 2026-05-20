# Progress Log

## Session: 2026-04-09

### Phase 1: Requirements & Discovery
- **Status:** in_progress
- **Started:** 2026-04-09 17:49
- Actions taken:
  - Read `docx/next_step.txt` and `docx/00_docx_storage_and_lookup_guide.md` in earlier turns to align with project rules.
  - Activated `planning-with-files` and `building-rules` workflow for this research-heavy task.
  - Scanned local `dInfer` and `sglang` repositories for diffusion-related files and keywords.
  - Read `lib_cite/dInfer/README.md` and identified core modules for follow-up reading.
  - Confirmed `sglang` repo contains both text diffusion LLM support and separate multimodal diffusion stack.
- Files created/modified:
  - `/home/wuhang/wuhang/dllm_wh/task_plan.md` (created)
  - `/home/wuhang/wuhang/dllm_wh/findings.md` (created)
  - `/home/wuhang/wuhang/dllm_wh/progress.md` (created)

### Phase 2: Local Codebase Analysis
- **Status:** complete
- Actions taken:
  - Read dInfer decoding core: `parallel_strategy.py`, `generate_uniform.py`, `generate_cache.py`, `generate_fastdllm.py`.
  - Read dInfer-to-SGLang integration and model-side attention/cache code.
  - Read SGLang text diffusion docs plus `srt.dllm` config, algorithms, scheduler mixin, and worker integration.
  - Extracted a concrete acceleration taxonomy spanning algorithm, cache, runtime, and model-system co-design.
- Files created/modified:
  - `/home/wuhang/wuhang/dllm_wh/findings.md` (updated)
  - `/home/wuhang/wuhang/dllm_wh/progress.md` (updated)

### Phase 3: External Verification
- **Status:** complete
- Actions taken:
  - Cross-checked official/public sources for LLaDA, Fast-dLLM, dInfer, and SGLang diffusion-LLM status.
  - Verified that current official SGLang docs expose `LowConfidence` and `JointThreshold` as first-class dLLM algorithms.
  - Verified that current dInfer materials position it as an efficient inference framework for diffusion LLMs with LLaDA/LLaDA2 support.
- Files created/modified:
  - `/home/wuhang/wuhang/dllm_wh/findings.md` (updated)
  - `/home/wuhang/wuhang/dllm_wh/progress.md` (updated)

### Phase 4: Synthesis
- **Status:** complete
- Actions taken:
  - Wrote a reusable technical article under `docx/articles` summarizing dLLM definition, inference flow, acceleration techniques, and optimization priorities.
  - Drafted archive/process documentation and synced key conclusions into repository memory files.
- Files created/modified:
  - `/home/wuhang/wuhang/dllm_wh/docx/articles/2026-04-09_diffusion_llm_inference_and_acceleration_landscape.md` (created)
  - `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.2-diffusion_llm_acceleration_landscape_archive.md` (created)
  - `/home/wuhang/wuhang/dllm_wh/docx/context_index/00_document_registry.md` (updated)
  - `/home/wuhang/wuhang/dllm_wh/code_building/progress_diff_summary.md` (updated)
  - `/home/wuhang/wuhang/dllm_wh/code_building/key_files_index.md` (updated)
  - `/home/wuhang/wuhang/dllm_wh/code_building/key_conclusion.md` (updated)

### Phase 5: Demo Script
- **Status:** complete
- Actions taken:
  - Checked the current environment and confirmed that top-level `import dinfer` fails because optional backend dependencies such as `vllm` are missing.
  - Implemented a self-contained minimal dInfer-style demo script under `codex_coding/src/` using only `torch` and `transformers`.
  - Added CLI options for model path, prompt, trace logging, generation length, block length, threshold, and metrics output.
  - Verified the script with `py_compile`, a CLI dry-run, and a dummy-model smoke test.
  - Checked common local model cache locations and found no local LLaDA/LLaDA2/SDAR checkpoint for a real short-run validation.
- Files created/modified:
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/src/dinfer_dllm_demo.py` (created)
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/2026-04-09_dinfer_dllm_demo_smoke_test.md` (created)
  - `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.3-dinfer_demo_script_archive.md` (created)
  - `/home/wuhang/wuhang/dllm_wh/code_building/progress_diff_summary.md` (updated)
  - `/home/wuhang/wuhang/dllm_wh/code_building/key_files_index.md` (updated)
  - `/home/wuhang/wuhang/dllm_wh/code_building/key_conclusion.md` (updated)

### Phase 6: Formal dInfer Run
- **Status:** complete
- Actions taken:
  - Installed `vllm==0.10.2` to unblock the formal `dinfer` import path.
  - Verified that the local model `/home/wuhang/models/LLaDA2.0-mini` exists and can load its config.
  - Added local compatibility fixes in `modeling_llada2_moe.py` for current `transformers` and `flash_attn` behavior.
  - Implemented a formal run script using the official dInfer benchmark path with `LLaDA2MoeModelLM`, `ThresholdParallelDecoder`, and `BlockDiffusionLLM`.
  - Executed a real generation on `cuda:0` with `tp=1` and saved metrics/output.
- Files created/modified:
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/src/run_dinfer_llada2_mini.py` (created)
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/dinfer_llada2_mini_metrics.json` (created)
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/2026-04-09_dinfer_llada2_mini_formal_run.md` (created)
  - `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.4-dinfer_llada2_mini_formal_run.md` (created)
  - `/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py` (updated)
  - `/home/wuhang/wuhang/dllm_wh/code_building/progress_diff_summary.md` (updated)
  - `/home/wuhang/wuhang/dllm_wh/code_building/key_files_index.md` (updated)
  - `/home/wuhang/wuhang/dllm_wh/code_building/key_conclusion.md` (updated)

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Local search | diffusion keyword scan in `dInfer` and `sglang` | Locate text diffusion entry points | Located dInfer decoding modules and SGLang text-diffusion docs/scheduler hooks | ✓ |
| Source cross-check | Official repo/docs/paper lookup | Confirm latest public framework status | Confirmed dInfer + SGLang are the clearest current text-dLLM framework lines in checked official sources | ✓ |
| Script syntax | `python -m py_compile codex_coding/src/dinfer_dllm_demo.py` | No syntax errors | Passed | ✓ |
| CLI dry-run | `python codex_coding/src/dinfer_dllm_demo.py --model-path GSAI-ML/LLaDA-1.5 --dry-run` | Print resolved config and exit | Passed | ✓ |
| Dummy smoke test | Inline dummy model + script classes | Block-wise loop should terminate and return stats | Passed with 2 forwards / 4 generated tokens | ✓ |
| Formal dInfer run | `run_dinfer_llada2_mini.py` on `/home/wuhang/models/LLaDA2.0-mini` | Real model load + generation + timing | Passed on `cuda:0`, `tp=1`, `38.05 tokens/s` | ✓ |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-04-09 17:47 | Expected `sglang/srt/diffusion` directory not found by path lookup | 1 | Switched to full-repo search for diffusion docs, scheduler branches, and LLaDA references |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 5 delivery |
| Where am I going? | Final user-facing explanation and follow-up direction setting |
| What's the goal? | Explain dLLM concept, inference flow, acceleration landscape, and optimization directions |
| What have I learned? | dInfer gives the clearest local algorithm/cache variants; SGLang provides first-class runtime/scheduler support for text dLLM |
| What have I done? | Completed local analysis, external verification, and durable archival |

*Update after completing each phase or encountering errors*

---

## Session: 2026-04-27 BSP-MoE

### Phase B1: Recovery & Constraints
- **Status:** complete
- **Started:** 2026-04-27 06:49 CST
- Actions taken:
  - Restored handoff context after compaction.
  - Read planning-with-files instructions and detected existing planning files from an older dLLM research task.
  - Checked repo status; many modified/untracked files already exist and must not be reverted.
  - Checked GPU state; GPUs 0-7 are H100 80GB and currently idle.
  - Read `docx/next_step.txt`, the insight ledger, and Scheme3 C12 timing/quality archive.
- Key constraints:
  - Build an independent BSP-MoE monkey-patch experiment.
  - Start with shape probe before performance claims.
  - Preserve EB/S_mask global semantics.
  - Avoid Scheme3/topk/pruning/scheduler changes in the first BSP experiment.

### Phase B2: Source-Level Patch Design
- **Status:** complete
- Actions taken:
  - Read C12 baseline script and Scheme3 script.
  - Read LLaDA2 MoE block, gate, shared expert, vLLM SP helper, Qwen3-MoE SP pattern, AgRs all2all manager, and forward-context DP metadata.
  - Confirmed the minimal BSP patch point is `LLaDA2MoeSparseMoeBlock.forward`.
  - Confirmed EB/S_mask can remain globally computed if routing stays inside native `FusedMoE.forward_impl` after SP dispatch.
- Current decision:
  - Build an independent `bench_bsp_moe_dp2.py` with baseline, shape-probe, and BSP-MoE modes.
  - Do not modify dInfer or vLLM source files.

### Phase B3: Independent BSP Script
- **Status:** complete
- Actions taken:
  - Added `/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_bsp_moe_dp2.py`.
  - Implemented baseline, BSP-MoE, shape-probe, forward-check, component-timing, and manual quality snippet reporting modes.
  - Kept the experiment independent from Scheme3/topk compaction/expert pruning/scheduler changes.
- Files created/modified:
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_bsp_moe_dp2.py` (created)

### Phase B4: Experiments
- **Status:** complete
- Actions taken:
  - Ran 8-GPU smoke shape probe at `batch=32,gen=32`.
  - Ran 8-GPU forward-check comparing baseline MoE blocks with BSP-MoE blocks on layers 0/9/18.
  - Ran C12 shape probe at `batch=512,gen=32`.
  - Ran C12 no-timing e2e compare at `batch=512,gen=256,num_runs=2`.
  - Ran C12 component timing at `batch=512,gen=256,num_runs=1`.
- Key results:
  - C12 block shape: `local_bs=256`, `seq_len=32`, `N_dp=8192`, BSP `N_sp=2048`, no padding.
  - C12 no-timing e2e: baseline `20.1285s / 75.675 ms/fwd`; BSP `19.8475s / 74.615 ms/fwd`; delta `-1.40%`.
  - Path counts identical: `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`.
  - Dispatch payload reduced `826.877 -> 206.719 MB/fwd`, but component timing shows combine and TP all-gather overhead consume the benefit.
  - Manual quality check on five verifiable prompts did not show BSP-specific semantic degradation.
- Result files:
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_shape_probe_smoke_8g_20260427.log`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_compare_smoke_8g_component_20260427.log`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_forward_check_8g_20260427.log`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_forward_check.json`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_shape_probe_c12_8g_20260427.log`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_shape_probe.json`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_c12_8g_e2e_20260427.log`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_c12_8g_e2e_summary_20260427.json`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_c12_8g_component_timing_20260427.log`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_c12_8g_component_summary_20260427.json`

### Phase B5: Archival
- **Status:** complete
- Actions taken:
  - Added process archive `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12j-bsp_moe_validation.md`.
  - Updated `/home/wuhang/wuhang/dllm_wh/findings.md` with BSP experiment results.
  - Updated `/home/wuhang/wuhang/dllm_wh/task_plan.md` with completed BSP phases.
  - Synced project memory files under `code_building/`.

## Session: 2026-04-27 BSP-MoE nsys Profiling

### Phase N1-N3: Trace Analysis Tooling
- **Status:** complete
- Actions taken:
  - Confirmed BSP short NVTX trace finished and produced `.log`, `.nsys-rep`, and `.sqlite`.
  - Added `/home/wuhang/wuhang/dllm_wh/codex_coding/src/analyze_nsys_bsp.py`.
  - Script parses A/B logs, filters CUDA work to `*.generate.run1` NVTX window, and summarizes NVTX components, kernel categories, collectives, memcpy, runtime, and top kernels.
  - Ran `python -m py_compile codex_coding/src/analyze_nsys_bsp.py`.
  - Ran `python codex_coding/src/analyze_nsys_bsp.py`.
- Files created/modified:
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/src/analyze_nsys_bsp.py` (created)
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/nsys_bsp_short_nvtx_analysis_20260427.json` (created)
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/nsys_bsp_short_nvtx_analysis_20260427.md` (created)

### Phase N4: Interpretation
- **Status:** complete
- Key results:
  - Short nsys e2e: A `69.60 ms/fwd`, B `76.72 ms/fwd`, B slower by `+10.23%`.
  - NVTX rankmax generate: A `3895.928 ms`, B `4406.343 ms`, B slower by `+13.10%`.
  - NCCL rankmax: A `7.993 ms/fwd`, B `17.114 ms/fwd`, B higher by `+114.12%`.
  - NCCL AllGather: A `1.837 ms/fwd`, B `5.962 ms/fwd`, B higher by `+224.53%`; count `9120 -> 18392`.
  - NCCL Reduce: A `2.381 ms/fwd`, B `7.573 ms/fwd`, B higher by `+218.05%`.
  - D2D memcpy total: A `41222.3 MB`, B `21307.7 MB`, B lower by `-48.31%`.
  - Dense GEMM rankmax: A `0.441 ms/fwd`, B `0.213 ms/fwd`, B lower by `-51.74%`.
- Interpretation:
  - BSP does reduce some work, but current monkey-patch collective sequence dominates.
  - vLLM AgRs uses DP group for non-SP and EP group for SP, so BSP changes group/layout and adds explicit TP all-gather after native MoE.
  - This explains why full C12 component timing saw `moe.combine 3.587 -> 8.284 ms/fwd` and new `moe.tp_all_gather 2.618 ms/fwd`.

### Phase N5: Archival
- **Status:** complete
- Actions taken:
  - Added `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12k-bsp_nsys_collective_profiling.md`.
  - Updated `/home/wuhang/wuhang/dllm_wh/code_building/progress_diff_summary.md`.
  - Updated `/home/wuhang/wuhang/dllm_wh/code_building/key_files_index.md`.
  - Updated `/home/wuhang/wuhang/dllm_wh/code_building/key_conclusion.md`.
  - Updated planning files with BSP nsys findings.


### Phase M1-M3 Continuation: D-path smoke recovery
- **Status:** in_progress
- Actions taken:
  - Detected prior D-path smoke left 7 orphan rank processes on GPU1-7 while rank0 was gone, indicating an invalid hung collective state rather than a valid result.
  - Killed orphan rank processes and confirmed syntax of `codex_coding/src/bench_bsp_moe_dp2.py` still passes `py_compile`.
- Current diagnosis:
  - The likely risk is the new M3 hot-update `dist.all_reduce(pop)` call needing exact path/order consistency across all ranks.

- Smoke after patch:
  - `batch=32, gen=32, A/B/C/D` completed.
  - D no longer hangs after switching M3 hot-update to explicit vLLM EP process group.
  - Path counts are identical across 8 ranks for all configs.
  - D reports `ep_reduce_calls=418` and `ep_reduce_mb=0.428` per rank.
  - Smoke timing: A `51.58 ms/fwd`, B `55.71`, C `54.39`, D `54.11`; smoke is for liveness/invariants, not final speed.
  - Archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_m123_smoke_abcd_20260427.json`.

- C12 no-timing A/B/C/D result:
  - A baseline `20.227s`, `76.04 ms/fwd`.
  - B BSP-MoE `19.898s`, `74.80 ms/fwd`, `-1.6%` vs A.
  - C BSP-DelayGather `19.777s`, `74.35 ms/fwd`, `-2.2%` vs A.
  - D BSP-DelayGather-M3EPReduce `19.946s`, `74.98 ms/fwd`, `-1.4%` vs A.
  - All configs path_counts identical on 8 ranks: `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`.
  - D `ep_reduce_calls=1862`, `ep_reduce_mb=1.907` per rank.
  - Archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_m123_c12_e2e_abcd_20260427.json`.

- C12 component timing A/B/C/D result:
  - A `78.27 ms/fwd`; key components: native `40.322`, dispatch `5.801`, quant `20.575`, combine `3.514`, TP all-reduce `4.646`, dispatch payload `826.877 MB/fwd`.
  - B `83.93 ms/fwd`; dispatch payload drops to `206.719 MB/fwd`, but dispatch `8.318`, combine `8.520`, explicit TP gather `2.614` dominate.
  - C `80.79 ms/fwd`; improves over B via native `35.227`, quant `18.705`, combine `6.916`, but still pays dispatch/gather overhead.
  - D `84.44 ms/fwd`; M3 EP pop all-reduce is invariant-correct but raises native/quant/combine back near B-level and is not a wall-clock win in the current patch layer.
  - Archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_m123_c12_component_abcd_20260427.json`.

## Session: 2026-04-27/28 BSP C+ / C++ Upper-Bound Experiments

### Phase U1-U2: Wiring and smoke
- **Status:** complete
- Actions taken:
  - Recovered partial C+ / C++ edits in `/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_bsp_moe_dp2.py`.
  - Added E path `C12-BSP-CrossLayerSP`, carrying `SPHiddenState` across sparse decoder layer boundaries and gathering full layout only for attention input / final sparse layer output.
  - Added F path `C12-BSP-AllReduceFullProbe`, preserving EP dispatch + local expert compute but replacing combine+TP gather with an EP full all-reduce probe.
  - Fixed F closure scope by passing `dp_rank` explicitly.
  - Extended compare summary, JSON output, component timing, and byte accounting with E/F and `ep_full_allreduce_payload`.
  - Ran `python3 -m py_compile codex_coding/src/bench_bsp_moe_dp2.py`.
  - Ran 8GPU smoke `batch=32,gen=32,no_quality`; all A/B/C/D/E/F completed, and path counts were rank-consistent.
- Smoke result:
  - A `53.11 ms/fwd`; B `55.84`; C `54.06`; D `54.21`; E `54.64`; F `48.87`.
  - Smoke is liveness/invariant only; not final speed.
  - Archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_cplus_cxx_smoke_20260427.json`.

### Phase U3: C12 no-timing
- **Status:** complete
- Run 1 result:
  - A `20.112s`, `75.61 ms/fwd`.
  - B `19.840s`, `74.59 ms/fwd`, `-1.35%`.
  - C `19.721s`, `74.14 ms/fwd`, `-1.94%`.
  - D `19.852s`, `74.63 ms/fwd`, `-1.29%`.
  - E `19.127s`, `71.90 ms/fwd`, `-4.90%`.
  - F `19.260s`, `72.41 ms/fwd`, `-4.23%`.
  - Archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_cplus_cxx_c12_e2e_20260427.json`.
- Repeat result:
  - A `20.178s`, `75.86 ms/fwd`.
  - B `19.855s`, `74.64 ms/fwd`, `-1.60%`.
  - C `19.722s`, `74.14 ms/fwd`, `-2.26%`.
  - D `19.804s`, `74.45 ms/fwd`, `-1.85%`.
  - E `19.069s`, `71.69 ms/fwd`, `-5.49%`.
  - F `19.288s`, `72.51 ms/fwd`, `-4.41%`.
  - Archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_cplus_cxx_c12_e2e_repeat_20260428.json`.
- Two-run average:
  - A `75.73 ms/fwd`; C `74.14`; E `71.80`; F `72.46`.
  - E average speedup `-5.20%`; F average speedup `-4.32%`.
  - A-F C12 path counts were invariant-consistent: `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`.

### Phase U4: Component timing and quality smoke
- **Status:** complete
- Component timing:
  - A `78.92 ms/fwd`.
  - C `81.03 ms/fwd`, with `native_forward=35.110`, `quant_apply=18.757`, `combine=7.216`.
  - E `80.84 ms/fwd`, with `native_forward=35.322`, `quant_apply=18.873`, `combine=7.267`, plus SP norm buckets `input_norm_sp=0.385`, `post_attn_norm_sp=0.893`.
  - F `79.94 ms/fwd`, with `native_forward=29.195`, `dispatch=8.030`, `quant_apply=20.135`, `ep_full_all_reduce=9.467`.
  - E still reports dispatch payload `206.719 MB/fwd` and TP gather payload `165.375 MB/fwd`.
  - F reports `ep_full_allreduce_payload=1311.020 MB/fwd`, so it is a probe rather than a production-ready collective.
  - Archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_cplus_cxx_c12_component_20260427.json`.
- Quality smoke:
  - Ran small `batch=32,gen=32` A-F with snippets enabled.
  - E/F snippets remain coherent on verifiable prompts, with formatting variation versus baseline.
  - Archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_cplus_cxx_quality_smoke_20260428.json`.

### Phase U5: Archival
- **Status:** complete
- Actions taken:
  - Added process doc `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12n-cplus_cxx_bsp_upper_bound.md`.
  - Updated `/home/wuhang/wuhang/dllm_wh/task_plan.md`.
  - Updated `/home/wuhang/wuhang/dllm_wh/findings.md`.
  - Updated `/home/wuhang/wuhang/dllm_wh/progress.md`.
  - Synced project memory files under `/home/wuhang/wuhang/dllm_wh/code_building/`.

## Session: 2026-04-28 BSP-G vLLM SP-Parity Experiments

### Phase G1-G3: Wiring, smoke, C12 e2e
- **Status:** complete
- Actions taken:
  - Added `G) C12-BSP-G-AttnReduceScatterSP` to `/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_bsp_moe_dp2.py`.
  - G patches attention output projection from TP all-reduce/full-layout output into TP reduce-scatter/SP-layout output.
  - Decoder then performs residual, post-attention norm, and MoE on SP layout, preserving the existing EB/S_mask controller path.
  - Ran `python3 -m py_compile codex_coding/src/bench_bsp_moe_dp2.py` before smoke in the previous session.
  - Ran 8GPU smoke `batch=32,gen=32,no_quality`; A/B/C/D/E/G/F completed without hang.
  - Archived smoke to `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_smoke_20260428.json`.
  - Ran C12 no-quality compare `batch=512,gen=256,num_runs=1`.
  - Archived C12 e2e to `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_c12_e2e_20260428.json`.
- C12 e2e result:
  - A baseline `75.35 ms/fwd`.
  - B BSP-MoE `74.48 ms/fwd`, `-1.16%` vs A.
  - C BSP-DelayGather `73.98 ms/fwd`, `-1.82%` vs A.
  - D BSP-DelayGather-M3EPReduce `74.56 ms/fwd`, `-1.05%` vs A.
  - E BSP-CrossLayerSP `71.67 ms/fwd`, `-4.88%` vs A.
  - G BSP-G-AttnReduceScatterSP `69.55 ms/fwd`, `-7.70%` vs A and `-2.96%` vs E.
  - F BSP-AllReduceFullProbe `72.28 ms/fwd`, `-4.08%` vs A.
  - A/B/C/D/E/G/F all preserve path counts: `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`.
- Next:
  - Run C12 component timing for A/B/C/D/E/G/F to determine whether G's gain comes from replacing attention TP all-reduce, reducing layout conversion cost, or moving the synchronization boundary.

### Phase G4: Component timing attempt 1
- **Status:** failed_environment
- Result:
  - First C12 component-timing attempt failed before G, during A baseline prefill.
  - Error was `torch.OutOfMemoryError` allocating `14.39 GiB`; the error report showed old benchmark PIDs using about `56-58 GB` per GPU.
  - Follow-up `ps` found no live `bench_bsp_moe_dp2`/`torchrun` processes, and `nvidia-smi --query-compute-apps` returned no active compute apps after release.
- Interpretation:
  - This is an environment/stale-process memory failure, not a BSP-G correctness or collective failure.
- Next:
  - Rerun C12 component timing only after confirming GPUs are free.

### Phase G4-G5: Component timing, repeat, quality, archival
- **Status:** complete
- Actions taken:
  - Reran C12 component timing after confirming no active `bench_bsp_moe_dp2`/`torchrun` process and no active compute apps.
  - Archived component timing to `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_c12_component_20260428.json`.
  - Reran C12 no-quality e2e repeat and archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_c12_e2e_repeat_20260428.json`.
  - Ran small quality smoke with snippets and archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_quality_smoke_20260428.json`.
  - Added process doc `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12o-bsp_g_vllm_sp_parity.md`.
- C12 e2e repeat:
  - A `76.70 ms/fwd`.
  - B `74.77 ms/fwd`, `-2.52%` vs A.
  - C `75.79 ms/fwd`, `-1.19%` vs A.
  - D `80.54 ms/fwd`, `+5.00%` vs A.
  - E `71.59 ms/fwd`, `-6.67%` vs A.
  - G `69.51 ms/fwd`, `-9.38%` vs A.
  - F `72.34 ms/fwd`, `-5.69%` vs A.
  - All C12 path counts remained `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`.
- Two-run averages:
  - A `76.03 ms/fwd`.
  - E `71.63 ms/fwd`, `-5.78%` vs A.
  - G `69.53 ms/fwd`, `-8.55%` vs A and `-2.93%` vs E.
  - F `72.31 ms/fwd`, `-4.90%` vs A.
- Component attribution:
  - G reduces `moe.bsp_chunk` from E's `0.872 ms/fwd`, count `5320`, to `0.003 ms/fwd`, count `266`.
  - G adds `attn.tp_reduce_scatter=5.020 ms/fwd` and `attn_rs_payload=661.502 MB/fwd`.
  - G keeps MoE dispatch payload `206.719 MB/fwd` and TP gather payload `165.375 MB/fwd`, same as E.
  - Therefore G's gain is mainly from attention output directly entering SP layout and reducing repeated full/SP layout conversion, not from reducing MoE combine/gather bytes.
- Quality smoke:
  - G snippets remain coherent on three verifiable prompts, with local formatting differences.
  - This is only a smoke check; full quality set remains required before source landing.
- Decision:
  - BSP-G is the next source-downshift candidate before BSP-H/F2.
  - Do not alter EB/s_mask algorithm; C12 invariants already show compatibility.

## Session: 2026-04-28 BSP-G2 vLLM SP-Parity Bundle

### Phase G2-1: Code wiring
- **Status:** complete
- Actions taken:
  - Added `SPAttentionInput` carrier to `/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_bsp_moe_dp2.py`.
  - Added `make_attention_g2_sp_parity_forward(...)`, which accepts SP-normalized attention input, gathers to full inside patched attention, and returns reduce-scattered SP output.
  - Added `make_bsp_g2_sp_parity_decoder_forward(...)`, which keeps E/G residual/norm/MoE SP layout but moves the attention input gather boundary into attention.
  - Added setup path `G2) C12-BSP-G2-SPParityBundle`.
  - Added `--config-set` with reduced experiment matrices:
    - `bspg=A/E/G/F`
    - `bspg2=A/E/G/G2/F`
    - `aeg2f=A/E/G2/F`
    - `aeggf=A/E/G/G2/F`
  - Added component buckets `attn.input_all_gather` and `attn_input_gather_payload`.
  - Ran `python3 -m py_compile codex_coding/src/bench_bsp_moe_dp2.py`; it passed.
- Compatibility:
  - E remains unchanged and still uses its original decoder-layer-only `CrossLayerSP` path.
  - G remains unchanged and still uses decoder-side attention input gather plus attention output reduce-scatter.
  - G2 is isolated behind its own setup path and `restore_blocks_and_experts()` restores attention forwards before each config.
- Smoke:
  - GPUs were free before the run.
  - Ran `torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --num-runs 1 --mode compare --config-set bspg2 --no-quality`.
  - Archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg2_smoke_20260428.json`.
  - A `53.567 ms/fwd`, E `53.973`, G `53.618`, G2 `53.328`, F `49.212`.
  - G2 no-hang and matches A/G/F path counts: `prefill_fallback=19,cold=38,hot_skip=874,hot_update=209`.
  - E has small-smoke path-count difference: `hot_skip=893` instead of `874`; record as smoke-only signal and require C12 invariant check.
- Next:
  - Run C12 no-quality e2e with `--config-set bspg2` and archive immediately after completion.

### Phase G2-3: C12 e2e
- **Status:** complete
- Actions taken:
  - Confirmed no active `bench_bsp_moe_dp2`/`torchrun` process and no active GPU compute apps before the run.
  - Ran `torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --num-runs 1 --mode compare --config-set bspg2 --no-quality`.
  - Archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg2_c12_e2e_20260428.json`.
- Result:
  - A `75.428 ms/fwd`.
  - E `71.816 ms/fwd`, `-4.789%` vs A.
  - G `69.676 ms/fwd`, `-7.626%` vs A.
  - G2 `69.661 ms/fwd`, `-7.646%` vs A.
  - F `72.380 ms/fwd`, `-4.042%` vs A.
  - All C12 path counts match invariant: `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`.
- Interpretation:
  - G2 no-hang and preserves EB/s_mask behavior.
  - G2 is effectively tied with G; the `0.015 ms/fwd` difference is measurement noise.
  - This supports treating G2 as SP-boundary/code-organization parity, not a new speed step yet.
- Next:
  - Run C12 component timing and archive immediately.

### Phase G2-4: C12 component timing
- **Status:** complete
- Actions taken:
  - Confirmed no active benchmark process and no active GPU compute apps before the run.
  - Ran `torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --num-runs 1 --mode compare --config-set bspg2 --component-timing --no-quality`.
  - Archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg2_c12_component_20260428.json`.
- Topline:
  - A `78.468 ms/fwd`.
  - E `82.541 ms/fwd`, `+5.191%` vs A under component instrumentation.
  - G `78.149 ms/fwd`, `-0.406%` vs A.
  - G2 `79.786 ms/fwd`, `+1.680%` vs A and `+1.637 ms/fwd` vs G.
  - F `78.624 ms/fwd`, `+0.200%` vs A.
  - All path counts match C12 invariant.
- G vs G2 attribution:
  - G has `moe.tp_all_gather=2.662 ms/fwd`, count `5054`, `tp_gather_payload=165.375 MB/fwd`.
  - G2 has `attn.input_all_gather=2.573 ms/fwd`, count `4788`, `attn_input_gather_payload=156.671 MB/fwd`.
  - G2 reduces `moe.tp_all_gather` to `0.141 ms/fwd`, count `266`, `tp_gather_payload=8.704 MB/fwd`.
  - Both G and G2 have `attn.tp_reduce_scatter` around `4.3 ms/fwd` and identical `attn_rs_payload=661.502 MB/fwd`.
  - Both keep `moe.bsp_chunk=0.003 ms/fwd`, count `266`.
- Interpretation:
  - G2 successfully moves gather ownership from decoder/MoE wrapper into attention input.
  - G2 does not reduce communication bytes or synchronization points relative to G.
  - Current monkey-patch G2 is source-organization parity, not performance-positive versus G.
- Next:
  - Run small quality smoke with snippets and archive.

### Phase G2-4: Quality smoke
- **Status:** complete
- Actions taken:
  - Confirmed GPUs were free before the run.
  - Ran `torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --num-runs 1 --mode compare --config-set bspg2`.
  - Archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg2_quality_smoke_20260428.json`.
- Result:
  - A `52.840 ms/fwd`, path `prefill_fallback=19,cold=38,hot_skip=874,hot_update=209`.
  - E `53.939 ms/fwd`, path `prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`.
  - G `53.622 ms/fwd`, path `prefill_fallback=19,cold=38,hot_skip=874,hot_update=209`.
  - G2 `53.617 ms/fwd`, path `prefill_fallback=19,cold=38,hot_skip=874,hot_update=209`.
  - F `48.855 ms/fwd`, path `prefill_fallback=19,cold=38,hot_skip=874,hot_update=209`.
  - G2 visible snippets are coherent and match G closely on the visible prompts.
  - E repeats the small-batch `hot_skip=893` difference; C12 invariant remains normal, so this is recorded as small-batch artifact rather than a C12 blocker.

### Phase G2-5: Decision
- **Status:** complete
- Final decision:
  - G remains the measured best performance reference for BSP-G.
  - G2 is successful as a vLLM/SP-parity source-organization bundle: attention owns input gather and output reduce-scatter.
  - G2 does not produce additional measured speed because it migrates communication accounting rather than reducing bytes or sync points.
  - Do not start BSP-H/F2 from this result alone; next useful optimization must reduce/fuse synchronization or extend SP lifetime further, not merely move gather ownership.

## Session: 2026-04-28 vLLM SP-Parity Inventory

### Phase I1: Inventory archival
- **Status:** complete
- Actions taken:
  - Created `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12q-vllm_sp_parity_inventory.md`.
  - Archived a persistent table of vLLM existing SP components:
    - MoE token-axis chunk.
    - FusedMoE SP mode.
    - MoE output gather.
    - AgRs EP/DP group switch.
    - DP/SP token metadata.
    - shared expert / dense MLP SP handling.
    - FusedMoE no TP all-reduce under SP.
    - MoE communication backend options.
    - EPLB.
    - compilation-level sequence parallelism.
    - residual scattered runtime support.
    - CUDAGraph/static-size alignment.
  - Archived a separate table of our BSP-specific extensions:
    - cross-layer SP hidden lifecycle.
    - attention output reduce-scatter.
    - attention-input gather ownership.
    - EB/s_mask SP compatibility.
    - EB M3 allreduce_pop.
    - hierarchical collective / BSP-H/F2.
    - dLLM block-iteration SP lifecycle.
- Key rule:
  - Until every row in the inventory is confirmed, do not claim vLLM SP-parity is fully moved.
- Next:
  - Confirm `VSP-06`, `VSP-08`, `VSP-10`, `VSP-11/VSP-12`, then revisit BSP-H/F2.
## Session: 2026-04-28 BSP-G3 to BSP-G7 vLLM SP Completion

### Phase G3-0: Pre-archive
- **Status:** complete
- Actions taken:
  - Restored current plan/context from `task_plan.md`, `progress.md`, `findings.md`, and `v0.1.15.12q-vllm_sp_parity_inventory.md`.
  - Confirmed current worktree contains many historical experimental files and docs; no rollback will be performed.
  - Created `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12r-bsp_g3_to_g7_vllm_sp_completion.md` before running new verification work.
  - Appended BSP-G3 to BSP-G7 plan block to `task_plan.md`.
- Next:
  - Start G3 source confirmation for `VSP-06 shared expert / dense MLP SP handling`.

### Phase G3-1: VSP-06 source confirmation
- **Status:** complete
- Findings:
  - vLLM DeepSeek/Llama4 set `disable_tp=is_sequence_parallel` for shared expert/dense MLP TP Linear so SP mode avoids TP all-reduce.
  - LLaDA2 `LLaDA2MoeMLP` uses replicated `nn.Linear` for shared expert and dense-only MLP, not TP Linear.
  - `/mnt/models/LLaDA2.0-mini/config.json` has `num_hidden_layers=20`, `first_k_dense_replace=1`, and `num_shared_experts=1`; current model has 1 dense-only layer and 19 MoE layers with shared expert.
  - Current BSP-G paths call shared expert on `hs_sp`, so shared expert already runs on TP-local token shards.
- Conclusion:
  - `VSP-06` is not a missing performance component for current LLaDA2-mini BSP-G.
  - Source landing must preserve shared expert on SP-local tokens, but no extra vLLM `disable_tp` mechanism needs to be ported unless LLaDA2 shared/dense MLP later becomes TP Linear.
- Files updated:
  - `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12r-bsp_g3_to_g7_vllm_sp_completion.md`
  - `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12q-vllm_sp_parity_inventory.md`
  - `/home/wuhang/wuhang/dllm_wh/task_plan.md`
- Next:
  - Start G4 source-downshift design.

### Phase G4-1: BSP-G source-downshift design
- **Status:** complete
- Findings:
  - BSP-G monkey-patch boundary consists of `SPHiddenState`, attention output `AttnSPResult`, vLLM `sequence_parallel_chunk`, FusedMoE SP flag, attention o_proj token-axis reduce-scatter, and decoder-layer cross-layer SP lifecycle.
  - LLaDA2 attention uses `RowParallelLinear` for `dense`; BSP-G's core performance mechanism is replacing that output all-reduce with `tensor_model_parallel_reduce_scatter(..., dim=0)`.
  - G2 attention-input gather ownership is cleaner structurally but not performance-positive versus G.
- Decision:
  - First source-downshift target should be measured-best BSP-G, not G2.
  - Keep decoder-side attention input gather initially to match measured G.
  - Keep EB/s_mask unchanged and exclude Scheme3/M3 payload changes.
- Source landing design archived in:
  - `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12r-bsp_g3_to_g7_vllm_sp_completion.md`
- Next:
  - Start G5 AsyncTP / compilation-level SP pattern-hit verification.

### Phase G5-1: AsyncTP / compilation-level SP pattern-hit probe
- **Status:** complete
- Actions taken:
  - Read vLLM `compilation/sequence_parallelism.py`, `compilation/collective_fusion.py`, and `compilation/pass_manager.py`.
  - Added `/home/wuhang/wuhang/dllm_wh/codex_coding/src/probe_vllm_async_tp_patterns.py`.
  - Ran `python3 -m py_compile codex_coding/src/probe_vllm_async_tp_patterns.py`.
  - Ran `python3 codex_coding/src/probe_vllm_async_tp_patterns.py --output codex_coding/results/vllm_async_tp_pattern_probe_20260428.json`.
- Results:
  - `SequenceParallelismPass` and `AsyncTPPass` import under the same `deep_ep` guard used by the benchmark.
  - `torch.ops.vllm.all_reduce/reduce_scatter/all_gather`, `_C.rms_norm/fused_add_rms_norm`, and `symm_mem.fused_*` ops are registered after the relevant vLLM imports.
  - Source scan confirms BSP-G has conceptual `dense.quant_method.apply -> tensor_model_parallel_reduce_scatter` adjacency and G2 has conceptual `all_gather -> query_key_value` adjacency.
  - Current standalone benchmark does not use vLLM compile pass manager, so AsyncTP fusion is not active today.
  - Direct import without `sys.modules["deep_ep"] = None` can fail on `deep_ep._C` with missing `ncclTeamWorld`; this is an environment constraint to carry into G6.
- Conclusion:
  - VSP-10 is feasible as a future source-downshift/compiler layer, but not currently consumed by BSP-G monkey-patch experiments.
- Files:
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/src/probe_vllm_async_tp_patterns.py`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/vllm_async_tp_pattern_probe_20260428.json`
  - `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12r-bsp_g3_to_g7_vllm_sp_completion.md`
- Next:
  - Start G6 backend inventory.

### Phase G6-1: MoE backend inventory
- **Status:** complete
- Actions taken:
  - Read vLLM `distributed/device_communicators/all2all.py`, `cuda_communicator.py`, `config/parallel.py`, `envs.py`, `utils/flashinfer.py`, and FusedMoE backend selection in `fused_moe/layer.py`.
  - Added `/home/wuhang/wuhang/dllm_wh/codex_coding/src/probe_vllm_moe_backends.py`.
  - Ran `python3 -m py_compile codex_coding/src/probe_vllm_moe_backends.py`.
  - Ran `python3 codex_coding/src/probe_vllm_moe_backends.py --output codex_coding/results/vllm_moe_backend_inventory_20260428.json`.
- Results:
  - AgRs `allgather_reducescatter` is current default and is the backend used by existing BSP-G measurements.
  - Naive is available but debug-only.
  - PPLX is not installed.
  - DeepEP package spec exists but direct import fails on `undefined symbol: ncclTeamWorld`; under benchmark guard `has_deep_ep=false`.
  - FlashInfer all2all capability is available and is the only plausible low-risk backend smoke candidate.
  - FlashInfer Cutlass fused MoE capability exists, but env flag is off and vLLM's enable path has dtype/DP/device constraints; not the next BSP-G communication drop-in.
- Decision:
  - Keep AgRs as reference.
  - Do not test DeepEP until ABI is fixed.
  - Do not pursue PPLX without install/build approval.
  - Treat `VLLM_ALL2ALL_BACKEND=flashinfer_all2allv` as optional isolated future smoke, not part of current source-downshift.
- Files:
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/src/probe_vllm_moe_backends.py`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/vllm_moe_backend_inventory_20260428.json`
- Next:
  - Start G7 residual scattered/static-size/CUDA graph source landing constraints.

### Phase G7-1: Source landing constraints and final report
- **Status:** complete
- Actions taken:
  - Read vLLM residual scattered helper, GPU runner padding/slicing logic, CUDA graph capture-size filtering, and compiler wrapper mutation guard.
  - Updated `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12r-bsp_g3_to_g7_vllm_sp_completion.md` with G7 constraints and final summary.
  - Marked BSP-G3 to BSP-G7 task plan complete.
- Results:
  - vLLM considers residual scattered only when sequence parallelism is enabled, TP > 1, token count is TP-divisible, and token count is in compile sizes.
  - CUDA graph capture sizes are filtered to TP multiples when sequence parallelism is enabled.
  - Source landing must preserve explicit layout metadata and original token count; shape alone is insufficient.
  - C12 `N=8192`, `tp=4`, `N_sp=2048` is safe for the current standard shape.
- Final recommendation:
  - If source landing starts, land measured BSP-G first with flags/fallback and no EB/s_mask change.
  - Treat FlashInfer all2all smoke and AsyncTP integration as follow-up stages after source path correctness.

## Session: 2026-04-28 BSP-G Source / Backend / Compile Execution

### Phase S1-0: Pre-archive
- **Status:** complete
- Actions taken:
  - Restored current plan/context from `task_plan.md`, `progress.md`, `findings.md`, and `v0.1.15.12r-bsp_g3_to_g7_vllm_sp_completion.md`.
  - Confirmed current DeepEP install naming: `deep_ep_cpp` exists and `deep_ep` imports it from `deep_ep/buffer.py`; old `deep_ep._C` probe is not valid for this install.
  - Created `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12s-bsp_g_source_backend_compile_execution.md` before running new experiments.
  - Appended BSP-G source/backend/compile execution plan to `task_plan.md`.
- Next:
  - Add and run a standalone DeepEP distributed smoke script before any source landing.

### Phase S1-1: DeepEP distributed smoke
- **Status:** complete
- Actions taken:
  - Added `/home/wuhang/wuhang/dllm_wh/codex_coding/src/smoke_deepep_distributed.py`.
  - Ran `python3 -m py_compile codex_coding/src/smoke_deepep_distributed.py`; it passed.
  - Confirmed 8x H100 GPUs were idle before the run.
  - Ran `torchrun --standalone --nproc_per_node=8 codex_coding/src/smoke_deepep_distributed.py --tokens 1024 --hidden 256 --num-experts 8 --top-k 1 --output codex_coding/results/deepep_distributed_smoke_20260428.json`.
- Result:
  - Overall `ok=true`.
  - `deep_ep_cpp` extension exists and `deep_ep.Buffer.is_sm90_compiled()` is true on all ranks.
  - All 8 ranks initialized `deep_ep.Buffer`, completed dispatch/combine, and produced combined shape `[1024,256]`.
  - Identity round trip reported `max_abs_diff=0.0` and `mean_abs_diff=0.0` on all ranks.
- Conclusion:
  - DeepEP HT path is distributed-runtime usable on this 8x H100 node for a minimal top-1 identity expert smoke.
  - Later AgRs vs DeepEP backend A/B is unblocked, but should wait until dInfer source BSP-G path is stable.
- Next:
  - Start dInfer BSP-G source landing behind feature flags.

### Phase S2-1: dInfer BSP-G source smoke
- **Status:** complete
- Actions taken:
  - Added source BSP-G carrier/attention reduce-scatter/decoder SP lifecycle in `/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py`.
  - Added `--config-set bspg_source` and `GS) C12-BSP-G-SourcePath` to `/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_bsp_moe_dp2.py`.
  - Ran `python3 -m py_compile` on both modified files; both passed.
  - Ran small smoke: `torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --num-runs 1 --mode compare --config-set bspg_source --no-quality`.
  - Archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_source_smoke_20260428.json`.
- Result:
  - A `51.492 ms/fwd`, G `53.451 ms/fwd`, GS source path `55.767 ms/fwd` on small smoke.
  - GS starts, runs, and exits cleanly across all ranks.
  - GS path counts are rank-consistent but match E's known small-batch artifact: `prefill_fallback=19,cold=38,hot_skip=893,hot_update=209`.
- Interpretation:
  - Source-G is live enough for C12 validation.
  - Small smoke is not a correctness blocker, but C12 invariant must be checked before treating source-G as equivalent to measured G.
- Next:
  - Run C12 no-quality A/E/G/GS source retest.

### Phase S3-1: Source-G C12 e2e
- **Status:** complete
- Actions taken:
  - Ran `torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --num-runs 1 --mode compare --config-set bspg_source --no-quality`.
  - Archived `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_source_c12_e2e_20260428.json`.
- Result:
  - A `75.698 ms/fwd`.
  - E `72.186 ms/fwd`, `-4.640%` vs A.
  - G monkey-patch `69.876 ms/fwd`, `-7.692%` vs A.
  - GS source path `71.973 ms/fwd`, `-4.922%` vs A.
  - All C12 paths preserve `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`.
- Interpretation:
  - Source-G correctness/path invariant passes.
  - Source-G is currently `2.097 ms/fwd` slower than monkey-patch G, so it is not yet the measured-best implementation.
- Next:
  - Run C12 component timing for A/E/G/GS to locate the GS vs G gap.

### Phase S5/S6: DeepEP HT C12 and dense/shared timing
- **Status:** complete
- DeepEP HT C12 completed without hang and preserved C12 path counts, but was slower than AgRs: A `129.755` vs `75.522 ms/fwd`, E `77.394` vs `71.942`, G `75.140` vs `69.719`, GS `75.523` vs `69.784`.
- Saved DeepEP C12 result to `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_source_deepep_ht_c12_e2e_20260428.json`.
- Added benchmark-only `dense.mlp` component timing for standalone dense MLP layers; no behavior or EB/s_mask change.
- Dense timing C12 result: `dense.mlp ~= 0.85 ms/fwd` across A/E/G/GS; `moe.shared ~= 3.3-3.4 ms/fwd` under BSP-G; both preserve path counts.
- Saved dense timing result to `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_source_dense_timing_c12_component_20260428.json`.
- Conclusion: keep AgRs backend; do not TP-shard shared/dense MLP in BSP-G source parity. If revisited, require separate flag and targeted A/B.

### Phase S7: Topology TP8 probe
- **Status:** complete as probe; not accepted as default
- Added `--tp-size` override to `/home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_bsp_moe_dp2.py`; default dp2tp4 behavior is unchanged.
- Added QKV KV-head replication parity in `/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py` for `tp_size >= num_key_value_heads`, matching vLLM `QKVParallelLinear` behavior.
- Fixed `_moe_forward_with_context` so FusedMoE gets `ForwardContext` whenever vLLM config exists, not only when DP>1.
- TP8 smoke passed and was saved to `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_source_tp8_smoke_20260428.json`.
- TP8 C12 no-quality result saved to `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_source_tp8_c12_e2e_20260428.json`: GS `58.758 ms/fwd` vs dp2tp4 GS `69.784 ms/fwd`, but path/fwd counts changed.
- TP8 quality smoke saved to `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_source_tp8_quality_smoke_20260428.json`; A baseline snippets were coherent but E/G/GS showed obvious blank/period degeneration on several prompts.
- Conclusion: TP8 is a promising performance probe but fails quick quality smoke; keep dp2tp4 as the valid BSP-G topology for now and split TP8 into a separate quality/root-cause thread.

### 2026-04-28 GS quality confirmation start
- Pre-archived command and judgment boundary in code_building/process_docs/v0.1-init-project/v0.1.15.12s-bsp_g_source_backend_compile_execution.md.
- Running dp2/tp4/ep8 bspg_source quality smoke before making any GS quality claim.

### 2026-04-28 GS quality confirmation result
- Ran dp2/tp4/ep8 `bspg_source` quality smoke without `--no-quality`.
- Saved `/home/wuhang/wuhang/dllm_wh/codex_coding/results/bsp_moe_bspg_source_dp2tp4_quality_smoke_20260428.json`.
- Result: GS `54.513 ms/fwd`, `60 fwd`, path counts `prefill_fallback=19,cold=38,hot_skip=874,hot_update=209`, rank-consistent.
- Manual quality: GS visible snippets for average-speed/math/logic prompts are coherent and match G closely; no TP8-style blank/punctuation collapse.
- Updated `v0.1.15.12s-bsp_g_source_backend_compile_execution.md` with GS quality result, TP8 contrast, and AsyncTP source-check plan.

### 2026-04-28 GS full quality/layout audit start
- Pre-archived full-quality and SP/TP layout correctness audit in `v0.1.15.12s-bsp_g_source_backend_compile_execution.md`.
- Scope: passive diagnostics only, no EB/s_mask/routing/path behavior changes.
- Invariant to check: QKV input must be full-token layout unless true context-parallel attention is implemented.

### 2026-04-28 TP8 MoE forward-check start
- dp2/tp4 full-quality/layout artifact saved: `codex_coding/results/bsp_moe_bspg_source_dp2tp4_full_quality_layout_20260428.json`.
- TP8 full-quality/layout artifact saved: `codex_coding/results/bsp_moe_bspg_source_tp8_full_quality_layout_20260428.json`.
- Key interim conclusion: TP8 E/G/GS have full-token QKV inputs but still collapse, so failure is not illegal SP-through-QKV.
- Running TP8 MoE forward-check to see whether SP-MoE itself diverges strongly from baseline under tp=8.

### 2026-04-28 TP8 MoE root-cause result
- Saved TP8 forward-check artifact: `codex_coding/results/bsp_moe_bspg_source_tp8_forward_check_20260428.json`.
- Ran and saved dp2/tp4 forward-check artifact: `codex_coding/results/bsp_moe_bspg_source_dp2tp4_forward_check_20260428.json`.
- Added benchmark-only `--mode moe-internal-check` to split MoE equivalence into chunk/shared/gate/routed/total stages; no EB/s_mask/routing algorithm change.
- First `moe-internal-check` attempt failed on diagnostic custom-routing signature; fixed keyword-argument compatibility and logged to `.learnings/ERRORS.md`.
- Saved dp2/tp4 MoE internal artifact: `codex_coding/results/bsp_moe_bspg_source_dp2tp4_moe_internal_check_20260428.json`.
- Saved TP8 MoE internal artifact: `codex_coding/results/bsp_moe_bspg_source_tp8_moe_internal_check_20260428.json`.
- Result: dp2/tp4 routed FusedMoE abs_mean max stays `~9.8e-4`; TP8 routed FusedMoE abs_mean max is `~0.87`, while chunk/shared/gate are exact or `~1e-7`.
- Conclusion: TP8 collapse is routed FusedMoE EP/SP communication-contract failure under `dp1,tp8,ep8`, not QKV layout, shared expert, gate logits, or token ordering.

### 2026-04-28 DInferCompileBackend / AsyncTP pattern-hit result
- Added thin dInfer adapter now named `/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/compile_backend.py` with public name `DInferCompileBackend`; it creates fresh vLLM `VllmBackend` instances and does not reimplement AsyncTP.
- Added distributed probe now named `/home/wuhang/wuhang/dllm_wh/codex_coding/src/probe_dinfer_compile_backend_async_tp.py`.
- Historical first probe artifact still uses the typo-era name: `/home/wuhang/wuhang/dllm_wh/codex_coding/results/dinfer_conpile_backend_async_tp_probe_20260428.json`.
- Static check passed after rename: `python3 -m py_compile lib_cite/dInfer/python/dinfer/compile_backend.py codex_coding/src/probe_dinfer_compile_backend_async_tp.py`.
- Final probe artifact: `/home/wuhang/wuhang/dllm_wh/codex_coding/results/dinfer_conpile_backend_async_tp_probe_20260428.json`.
- Result: `compiled_ok=true`, `fused_hit=true`, `semantic_ok=false`, `ok=false`.
- `AllGather -> GEMM` hit `symm_mem.fused_all_gather_matmul` and matched eager exactly (`max_abs=0.0`).
- `GEMM -> ReduceScatter` hit `symm_mem.fused_matmul_reduce_scatter`, but fused output matched `eager/tp`, not eager, because vLLM uses `"avg"` while current eager reduce-scatter is SUM.
- Decision: safe next AsyncTP target is `attention-input all_gather -> QKV GEMM`; do not land `attention O-proj GEMM -> reduce_scatter` fusion until the sum/avg semantic mismatch is explicitly solved.

### 2026-04-28 DInferCompileBackend fused SUM raw-op result
- Pre-archived the fused SUM validation plan in `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12s-bsp_g_source_backend_compile_execution.md` before experiments.
- Added `/home/wuhang/wuhang/dllm_wh/codex_coding/src/probe_dinfer_compile_backend_fused_sum.py`.
- Static check passed: `python3 -m py_compile lib_cite/dInfer/python/dinfer/compile_backend.py codex_coding/src/probe_dinfer_compile_backend_async_tp.py codex_coding/src/probe_dinfer_compile_backend_fused_sum.py`.
- `hidden=256` artifact: `/home/wuhang/wuhang/dllm_wh/codex_coding/results/dinfer_compile_backend_fused_sum_probe_20260428.json`; SUM semantic passed but fused SUM was slower, rankmax `0.3588 ms` vs eager `0.0642 ms`.
- `hidden=2048` artifact: `/home/wuhang/wuhang/dllm_wh/codex_coding/results/dinfer_compile_backend_fused_sum_h2048_probe_20260428.json`; AVG control passed, SUM was slightly outside strict `atol=0.05` (`max_abs=0.0625`) and slower, rankmax `0.3523 ms` vs eager `0.2052 ms`.
- Decision: do not implement the dInfer-local AsyncTP SUM pass now. Keep `GEMM -> ReduceScatter` on eager SUM semantics; pursue `AllGather -> GEMM` / attention-input QKV fusion as the safe AsyncTP candidate.

### 2026-04-28 Exact O-proj rectangular fused SUM follow-up
- User correctly questioned whether square probes represented real AsyncTP/O-proj behavior; pre-archived Phase S4c in `v0.1.15.12s-bsp_g_source_backend_compile_execution.md`.
- Extended `/home/wuhang/wuhang/dllm_wh/codex_coding/src/probe_dinfer_compile_backend_fused_sum.py` with `--in-hidden` / `--out-hidden` while keeping `--hidden` backward-compatible.
- Ran exact LLaDA2 O-proj shape: `tokens=8192,in_hidden=512,out_hidden=2048,tp=4`.
- Artifact: `/home/wuhang/wuhang/dllm_wh/codex_coding/results/dinfer_compile_backend_fused_sum_oproj_rect_probe_20260428.json`.
- Result: SUM semantic passed (`max_abs=0.03125` under `atol=0.08`) and AVG control passed, but fused SUM was still slower: rankmax `0.3602 ms` vs eager `0.1847 ms`, average `0.3473 ms` vs `0.1506 ms`.
- Conclusion: exact O-proj shape confirms current symmetric-memory fused SUM is not a BSP-G O-proj speed path. AsyncTP next target should be `AllGather -> GEMM` around attention-input/QKV, not O-proj SUM fusion.

### 2026-04-28 Exact QKV AllGather+GEMM performance result
- Pre-archived Phase S4d in `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12s-bsp_g_source_backend_compile_execution.md`.
- Added `/home/wuhang/wuhang/dllm_wh/codex_coding/src/probe_dinfer_compile_backend_allgather_qkv.py` and `/home/wuhang/wuhang/dllm_wh/codex_coding/src/probe_raw_fused_allgather_qkv.py`.
- Static check passed: `python3 -m py_compile codex_coding/src/probe_raw_fused_allgather_qkv.py codex_coding/src/probe_dinfer_compile_backend_allgather_qkv.py`.
- First compiled QKV probe reached cache generation but produced no JSON and was killed after abnormal long runtime; cache inspection showed only ranks 0/4 generated `symm_mem.fused_all_gather_matmul`, while other ranks generated eager `vllm.all_gather + aten.mm`, so compiled source integration is currently rank-inconsistent.
- Raw op short artifact: `/home/wuhang/wuhang/dllm_wh/codex_coding/results/raw_fused_allgather_qkv_probe_short_20260428.json`.
- Raw op final artifact: `/home/wuhang/wuhang/dllm_wh/codex_coding/results/raw_fused_allgather_qkv_probe_20260428.json`.
- Exact QKV shape: `x_sp=[2048,2048]`, `weight=[2048,768]`, output `[8192,768]`.
- Result: semantic pass with `max_abs=0.0`, but raw fused `symm_mem.fused_all_gather_matmul` is slower than eager all-gather+GEMM: avg `0.4112 ms` vs `0.1609 ms`, rankmax `0.4636 ms` vs `0.1823 ms`.
- Conclusion: do not integrate AsyncTP `AllGather+QKV GEMM` for BSP-G now; keep attention-input gather eager and return performance focus to BSP-H/F2 collective/layout redesign.

### 2026-04-28 Native AsyncTP threshold and 8192 split result
- Pre-archived Phase S4e in `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12s-bsp_g_source_backend_compile_execution.md`.
- Extended `/home/wuhang/wuhang/dllm_wh/codex_coding/src/probe_raw_fused_allgather_qkv.py` with `--split-chunks`, native-env metadata, post-timing semantic check, and fused repeat check.
- Static check passed: `python3 -m py_compile codex_coding/src/probe_raw_fused_allgather_qkv.py`.
- `M=4096,native env` artifact: `/home/wuhang/wuhang/dllm_wh/codex_coding/results/raw_fused_allgather_qkv_native_m4096_repeatcheck_probe_20260428.json`; repeated semantics failed (`diff_after_timing max_abs=5.46875`) and fused was slower, rankmax `0.2069 ms` vs eager `0.0885 ms`.
- `M=4096,no native env` artifact: `/home/wuhang/wuhang/dllm_wh/codex_coding/results/raw_fused_allgather_qkv_m4096_repeatcheck_probe_20260428.json`; semantics passed but fused was much slower, rankmax `0.4453 ms` vs eager `0.0890 ms`.
- `M=8192,split2,native env` artifact: `/home/wuhang/wuhang/dllm_wh/codex_coding/results/raw_fused_allgather_qkv_native_m8192_split2_probe_20260428.json`; semantics failed (`max_abs=5.5`) and fused was slower, rankmax `0.4412 ms` vs chunked eager `0.1909 ms` and full eager `~0.148-0.150 ms`.
- `M=8192,unsplit,native env` artifact: `/home/wuhang/wuhang/dllm_wh/codex_coding/results/raw_fused_allgather_qkv_native_m8192_probe_20260428.json`; semantics passed but fused remained slower, rankmax `0.4106 ms` vs eager `0.1492 ms`.
- Conclusion: 4096 does not prove a useful fast path for our QKV shape, and adapting 8192 by splitting into 2x4096 is not viable. Keep QKV all-gather eager and move optimization focus back to BSP-H/F2 collective/layout redesign.

### 2026-04-28 EB HetEval512 law probe
- User shifted the paper direction back to EB/dLLM-specific MoE regularities and asked to validate simple laws on historical `HetEval512`.
- Confirmed historical C12/HetEval512 config: `batch=512,gen=256,block=32,dp=2,tp=4,ep=8`.
- Pre-archived plan in `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.13-eb_heteval512_law_probe.md`.
- Added observational script `/home/wuhang/wuhang/dllm_wh/codex_coding/src/collect_eb_heteval512_laws.py`; it does not change EB routing policy and records compressed `S_mask`/top-k/histogram controls.
- Static check passed: `python3 -m py_compile codex_coding/src/collect_eb_heteval512_laws.py`.
- Smoke `batch=32,gen=32` passed and generated `codex_coding/results/eb_heteval512_laws_smoke_20260428_summary.json`.
- Full HetEval512 run completed and generated:
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_heteval512_laws_20260428.log`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_heteval512_laws_20260428_rank0.json.gz`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_heteval512_laws_20260428_rank4.json.gz`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_heteval512_laws_20260428_summary.json`
  - `/home/wuhang/wuhang/dllm_wh/codex_coding/results/eb_heteval512_laws_20260428_extended_controls.json`
- Full run path counts matched C12 history on all ranks: `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`.
- Main result: EB top4 active experts average `197.98` vs no-EB top4 `251.22`, a `21.19%` reduction.
- `S_mask` adjacent Jaccard mean `0.9862`; previous-call coverage `0.9658`, supporting hot-skip / delayed update.
- Static global-popularity same-size coverage `0.9704` slightly exceeded current EB `0.9681`; this suggests a hybrid global-prior + EB-correction research path.
- EB reduced active experts but mean EP load skew increased slightly (`3.118` vs no-EB top4 `2.970`), so active-expert reduction and load balancing must be treated as separate objectives.

### 2026-04-30 EPLB step/rearrange 诊断推进（自动模式）
- 先做参数面扫与长窗口对照，确认仅调 `window_size/step_interval` 不能解释收益：
  - `gen=32` 面扫：`w16/s16=135.383`, `w64/s256=142.767`, `w256/s1024=137.620`, `w1000/s3000=138.801 ms/fwd`；
  - 四组 `overall_skew` 一致（`1.188374`），path counts 一致（`19/38/893/209`）。
  - `gen=256` 极端对照：`w16/s16=86.173` vs `w1000/s3000=89.807 ms/fwd`，但 `skew/path` 仍一致（`1.180346`，`19/171/4199/1007`）。
- 结论：当前主线没有自然推进 `EplbState.step()`，`window/step` 参数对真实重排几乎不生效。

### 2026-04-30 step hook 断言定位与开销拆分
- 在 `collect_eb_heteval512_laws.py` 增加诊断开关：
  - `--eplb-force-step`（生成后强制一步）；
  - `--eplb-step-per-forward`（每次 forward 后推进一步）；
  - 记录 `eplb_step_hook_diag`（calls/errors/traceback）与 `state_step`。
- 初版 `step-per-forward` 触发重排但有断言：
  - `eb_eplb_stepdiag_on_smoke_r1_20260430`: `223.396 ms/fwd`, `calls=27`, `errors=2`。
  - traceback 精确定位到：`rebalance_execute.py:292`，`assert len(expert_weights) == num_moe_layers`。
- 修复诊断桥接（预注册 `expert_weights/moe_layers`，并补 `set_eplb_state`）后：
  - `eb_eplb_stepdiag_on_fixlen_smoke_r1_20260430`: 断言清零（`errors=0`），但 `531.866 ms/fwd`，出现一次 `10.39s` 的重排。
  - 说明“真重排路径”可达，但在线成本极高。
- 进一步拆分“step调用本身 vs 重排本身”：
  - `eb_eplb_stepdiag_off_smoke_r1_20260430`: `163.461 ms/fwd`（对照）
  - `eb_eplb_stepdiag_on_fixview_s256_smoke_r1_20260430`（`step_interval=256`，不触发重排）：`156.454 ms/fwd`，`errors=0`，`calls=29`
  - `eb_eplb_stepdiag_on_fixview_smoke_r1_20260430`（`step_interval=16`，触发重排）：`515.484 ms/fwd`，`errors=0`，`calls=30`
- 阶段结论：
  - 真正的主要开销是 `rearrange_expert_weights_inplace`（通信/搬运），不是 `step()` 调用壳本身；
  - 直接 per-forward 重排不可下沉主线，必须采用保守接入（cold触发 + 最小间隔 + fail-open）。

## Session: 2026-05-01

- 当前自动模式继续推进 EPLB/source-landing 收益回收。
- 新的关键判断是：`runtime-enable but record=off` 已是当前主税，`disable_after_build` 证明静态 placement 本身接近 baseline。
- 下一步准备做 `P1 static-only` 的 bench C12 复测，验证 `redundant=0 + disable_after_build` 是否可以作为默认主线。
- 这轮先只更新归档，不改 EB 语义。


- 已完成 `P1 static-only` C12 bench 复测（`redundant=0 + disable_after_build`）：A/E/G/GS 相对 OFF32 仅 `+0.31% ~ +0.69%`。
- 对照 `ON32-off` 仍为 `+4.13% ~ +5.10%`，确认 runtime 常驻税是主要矛盾。
- path counts 保持 `19/171/3933/931`，语义稳定。

### 2026-05-01 07:27 runtime map identity fastpath 回收结果

- 在 `modeling_llada2_moe.py` 增加 runtime identity passthrough：
  - 条件：`logical_replica_count==1` 且 `logical_to_physical_map[...,0]==logical_id`；
  - 行为：直接透传 `topk_ids`，保留 `indices_type` 与 cold record 语义。
- 同步加入 runtime map 诊断计数并在 `bench_bsp_moe_dp2.py` 打印/写盘：
  - `identity_checks/hits/cache_hit/cache_miss/record_calls`。
- 首轮误判（要求 `slots==1`）导致 0 命中；修正条件后 C12 命中恢复为 100%。
- C12 同版本 A/B（`ON32-off`）：
  - fastpath 关：`A/E/G/GS = 79.179/75.245/73.442/73.114`
  - fastpath 开：`A/E/G/GS = 76.280/72.653/70.481/70.401`
  - 净收益：`-3.66%/-3.44%/-4.03%/-3.71%`
- 相对 OFF（`75.784/72.397/70.149/69.748`）：
  - `ON32-off + fastpath` 仅 `+0.35% ~ +0.94%`
  - `ON32-cold + fastpath` 仅 `+0.62% ~ +1.28%`
- 诊断证据（C12, G）：
  - `identity_hits=5054`, `identity_checks=5054`, `cache_miss=19`, `cache_hit=5035`
  - `cold_only` 时 `record_calls=171`，与 cold 次数对齐。
- path counts 全保持 `19/171/3933/931`，EB 语义未变。

### 2026-05-01 13:44 C12 双轴（性能+负载均衡）接入与三组实验

- 在 `bench_bsp_moe_dp2.py` 落地负载均衡指标采集与结果落盘：
  - 新增 runtime 观测器：`RuntimeLayerLoadCollector`
  - 新增聚合/打印函数：`_collect_load_balance_metrics_from_rankloads`, `_aggregate_load_balance_runs`, `_print_load_balance_summary`
  - 每个 config 结果新增字段：`load_balance_runs`, `load_balance_summary`
- 关键口径调整：
  - 负载采集改为 routing 回调处实时计数，避免依赖 `expert_load_view`；
  - 因此 `OFF` 也能稳定输出非零负载均衡指标。
- C12 三组完整实验（`batch=512,gen=256,tp=4,world=8,num_runs=2,config-set=bspg_source,no-quality`）完成：
  1. OFF:
     - 结果：`bsp_moe_dp2_results_c12_lb2_off_20260501.json`
     - A/E/G/GS: `80.549/78.819/76.619/76.640 ms/fwd`
     - `ep_load_cv` 约 `0.073~0.076`
  2. ON32-off (`runtime on,redundant=32,record=off`):
     - 结果：`bsp_moe_dp2_results_c12_lb2_on32_off_20260501.json`
     - A/E/G/GS: `88.094/84.602/82.673/82.463 ms/fwd`
     - 相对 OFF: `+9.37/+7.34/+7.90/+7.60%`
     - `ep_load_cv` 约 `0.214~0.215`（较 OFF 增加约 `+185%~+193%`）
  3. ON32-cold (`runtime on,redundant=32,record=cold_only`):
     - 结果：`bsp_moe_dp2_results_c12_lb2_on32_cold_20260501.json`
     - A/E/G/GS: `88.043/84.623/82.716/82.435 ms/fwd`
     - 与 ON32-off 几乎重合，仅 `record_calls=171`（off 为 0）
- 不变性校验：
  - 三组 path counts 全一致：`prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`。
- 本轮结论：
  - 双轴能力已可用；
  - 当前 `redundant=32` 运行态同时带来时延回退与负载均衡恶化；
  - 主问题不在 cold-only record，而在 runtime-on 的 mapping/placement 组合路径。

### 2026-05-01 14:40 E=36 config 补齐后双轴复测（OFF vs ON32-off）

- 已补齐配置文件：
  - `/home/wuhang/miniconda3/envs/dllm/lib/python3.10/site-packages/vllm/model_executor/layers/fused_moe/configs/E=36,N=512,device_name=NVIDIA_H100_80GB_HBM3.json`
  - 方式：复制 `E=34,N=512` 模板。
- 运行日志确认 ON32 命中 `E=36` config（不再 fallback default config）。
- C12 同口径复测（`batch=512,gen=256,tp=4,world=8,num_runs=2,bspg_source,no-quality`）：
  1. OFF:
     - 文件：`bsp_moe_dp2_results_c12_lb3_off_e36cfg_20260501.json`
     - A/E/G/GS=`80.506/78.835/76.664/76.764 ms/fwd`
  2. ON32-off:
     - 文件：`bsp_moe_dp2_results_c12_lb3_on32_off_e36cfg_20260501.json`
     - A/E/G/GS=`84.769/81.882/79.907/79.716 ms/fwd`
     - 相对 OFF：`+5.30/+3.86/+4.23/+3.85%`
- 与补齐前 ON32-off（lb2）相比，E36 补齐后的 ON32-off 速度回收：
  - A `-3.77%`, E `-3.21%`, G `-3.35%`, GS `-3.33%`。
- 负载均衡：
  - OFF: `ep_load_cv ~0.073~0.076`
  - ON32-off: `ep_load_cv ~0.214~0.215`（较 OFF `+185%~+192%`）
  - GS rank-load（run0）从 OFF 的相对均匀分布变为 ON 的明显两端低、中间高分布。
- 不变性：
  - path counts 仍一致：`prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`。

### 2026-05-01 17:58 ON32 启动排布改造（weight-balance init）实验完成

- 已实现并接入可开关的 ON32 启动排布策略：
  - 新增 `DINF_EPLB_INIT_PLACEMENT_MODE`：
    - `joint_p1_p5`（默认，历史策略）
    - `weight_balance`（新策略）
  - 新增 `DINF_EPLB_INIT_GLOBAL_EXPERT_LOAD_PATH`：
    - 支持直接加载 `[layers,experts]`/`[sparse_layers,experts]`/`[experts]` tensor
    - 支持 `expert_budgeting_routing_data.pt` 风格 routing payload 自动聚合为 expert-load。
- 本轮仅改初始化排布，不启用 runtime rearrange，不改 EB/s_mask。

- 代码变更：
  - `/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py`
    - 增加 init-placement mode 解析
    - 增加 load-prior 加载/聚合
    - 增加 weight-aware 物理 expert 归属构造
    - 在 `load_state_dict` 的 external-map + redundant 路径接入新模式。

- 先跑 smoke（`batch=32,gen=32`）验证：
  - 命令：`... DINF_EPLB_INIT_PLACEMENT_MODE=weight_balance ...`
  - 结果文件：`bsp_moe_dp2_results_c12_lb4_smoke_on32_wbinit_20260501.json`
  - 路径计数稳定、E=36 config 命中、运行正常。

- 正式 C12 双轴实验（`batch=512,gen=256,tp4,ep8,num_runs=2,bspg_source,no-quality`）：
  1. old init（joint）：
     - `bsp_moe_dp2_results_c12_lb4_on32_oldinit_20260501.json`
     - GS=`79.615 ms/fwd`, `ep_load_cv=0.053773`
  2. wb init（weight_balance）：
     - `bsp_moe_dp2_results_c12_lb4_on32_wbinit_20260501.json`
     - GS=`80.081 ms/fwd`, `ep_load_cv=0.053029`

- 本轮结论：
  - `weight_balance init` 相对 `old init` 仅带来很小的负载指标改善（GS `ep_load_cv -1.38%`），但时延略退（GS `+0.585%`）。
  - 因此“只换启动排布”目前不足以兑现性能收益；下一步需继续针对 runtime map/select 路径或 mapping 分布策略本身优化。

### 2026-05-01 20:02 two-choice 局部性路线（cold 边界触发）验证完成

- 已完成两项关键接线与诊断修复：
  - `two_choice_lb + record=off` 也能收到 cold/hot route path 信号；
  - 修复 two-choice 诊断在 context reset 后被清零不可见的问题。
- 新增 two-choice 局部性控制：
  - 默认 `DINF_EPLB_TWOCHOICE_COLD_ONLY=1`：hot 回退 flat，cold 才做 load-aware 选择与计数更新。
- 结果：
  - smoke：`twochoice-hot` 明显负优化（A `+47%`，G/GS `+16~17%`）。
  - smoke：`twochoice-coldonly` 仍小幅负优化（约 `+1~2%`）。
  - C12：`twochoice-coldonly` 相对 flat：A `+1.31%`, E `+1.68%`, G `-0.73%`, GS `+0.83%`，无稳定净收益。
- 诊断确认:
  - C12 GS: `twochoice_multi=2,275,511`, `twochoice_lb_applied=2,275,511`, `decay/update_calls=171`，说明 cold 边界触发已真实生效。
- 负载：
  - C12 GS `ep_load_cv` 仅 `0.215393 -> 0.214857`，改善很小，未转化为稳定时延收益。
- 结论：
  - 局部性时机思路可行，但当前 two-choice 实现在本路径仍不具性能优势；
  - 主线继续保持 `flat_eager`，two-choice 作为实验分支保留。

### 2026-05-01 21:02 Profiling 专项（单次口径）完成

- 按最新要求，每组仅跑 1 次，完成四组 C12 profiling（含 component timing）：
  - `OFF`
  - `ON32-flat`
  - `ON32-cold`
  - `P1 static-only`
- 结果（以 GS 为主）：
  - OFF: `87.840 ms/fwd`
  - ON32-flat: `93.426`（`+6.36%`）
  - ON32-cold: `92.295`（`+5.07%`）
  - P1 static-only: `91.046`（`+3.65%`）
- path counts 四组一致：`19/171/3933/931`。
- 负载指标（GS）：
  - OFF/P1: `ep_load_cv=0.0305`
  - ON32-flat/cold: `ep_load_cv=0.0538`
- 组件侧（GS）显示的主要上升项（ON32-flat vs OFF）：
  - `moe.native_forward`、`moe.quant_apply`、`moe.combine`、`moe.dispatch`
  - 通信 payload 字节量本身未变（`dispatch/tp_gather/attn_rs` 一致）。
- 阶段结论：
  - 当前回退不是“通信字节量变化”主导，更像是 runtime-on 路径诱发的关键执行段时延上升；
  - `cold_only` 仅小幅缓解，说明 cold record 不是主税；
  - 下一步应做 rank-tail/同步等待方向的深挖 profiling，而不是继续泛化搬运改造。

### 2026-05-01 21:32 Profiling 代码建设启动（rank-tail 扩展）

- 按最新指令开始代码建设。
- 本轮先做 benchmark 计时框架增强，不改 EB/s_mask/映射语义：
  - `component_timing` 增加跨 rank 分布字段（rankmean/rankmin/tail_gap）。
  - 终端摘要输出增加 tail 视角。
- 完成后按“每组跑1次”继续实验并回填结果。

### 2026-05-01 21:49 rank-tail 扩展首轮完成（OFF vs ON32-cold）

- 已完成 `component_timing` 的 rank 分布增强并通过编译。
- 已跑 2 组单次 profiling（C12）：
  - `c12_prof_tail_off_20260501`
  - `c12_prof_tail_on32_cold_20260501`
- GS 结论：
  - `ms_fwd`: `88.079 -> 94.146`（ON32-cold 回退）
  - `attn.tp_reduce_scatter` tail gap: `0.295 -> 0.947`（显著放大）
  - `quant_apply` tail gap: `0.199 -> 0.386`
  - `combine` tail gap: `0.168 -> 0.262`
  - `dispatch` tail gap近似不变。
- path counts 保持完全一致（19/171/3933/931）。

### 2026-05-01 22:07 上下文压缩恢复说明已刷新

- 已按模板覆盖更新 `/home/wuhang/wuhang/dllm_wh/history-chat.txt`。
- 新版内容包含：
  - 必读文档列表与阅读顺序；
  - 当前代码建设点（rank-tail 扩展）；
  - 最新实验结论（OFF vs ON32-cold）；
  - 待办（补齐 ON32-flat / P1-static）和复现命令。
- 本次仅归档，不做新增算法改造。
