# Task Plan: dInfer 与 SGLang diffusion LLM 研究梳理

## Goal
阅读本地 `lib_cite/dInfer` 与 `lib_cite/sglang` 中和 diffusion LLM 相关的实现与文档，并结合官方一手资料，向用户说明 diffusion LLM 的概念、推理流程、与自回归模型的差异，以及当前主流 dLLM 加速技术与后续优化方向。

## Current Phase
Phase 5

## Phases

### Phase 1: Requirements & Discovery
- [x] Understand user intent
- [x] Identify constraints and requirements
- [x] Document findings in findings.md
- **Status:** complete

### Phase 2: Local Codebase Analysis
- [x] Read dInfer diffusion-related architecture and decoding code
- [x] Read SGLang diffusion-LLM-related docs and scheduler integration
- [x] Extract concrete acceleration techniques from local code
- **Status:** complete

### Phase 3: External Verification
- [x] Verify current mainstream framework practices with primary sources
- [x] Cross-check whether claimed techniques are latest/still current
- [x] Distinguish text dLLM from image/video diffusion acceleration
- **Status:** complete

### Phase 4: Synthesis
- [x] Build comparative explanation: dLLM vs AR
- [x] Summarize inference pipeline and acceleration taxonomy
- [x] Infer practical optimization directions for this repo
- **Status:** complete

### Phase 5: Delivery & Archival
- [ ] Deliver concise but technically rigorous answer to user
- [x] Update process/summary files required by repo rules
- [x] Record commands and key files if durable artifacts are created
- **Status:** in_progress

### Phase 6: Formal dInfer Execution
- [x] Install missing runtime dependencies needed by official dInfer import
- [x] Run a real LLaDA2.0-mini experiment through formal dInfer code paths
- [x] Save runnable script and measured results
- **Status:** complete

## Key Questions
1. 在 `dInfer` 中，diffusion LLM 的基本采样与并行解码循环是如何实现的？
2. 在 `sglang` 中，diffusion LLM 支持落在文档、调度器和哪些运行时接口？
3. 当前主流 dLLM 加速主要分成哪些层次：算法、cache、kernel、并行、量化、服务化？
4. 哪些技术是 text diffusion LLM 特有，哪些只是通用推理优化？

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 先以本地代码和文档为主，再用官方资料校对“主流/最新实践” | 用户明确要求先阅读本地引用库，同时“最新实践”具有时间敏感性 |
| 将 SGLang 的 image/video diffusion 与 text diffusion LLM 分开讲 | 两者都叫 diffusion，但优化对象和瓶颈不同，混讲会误导后续方向 |
| 对 `LLaDA2.0-mini` 正式实验优先采用 `LLaDA2MoeModelLM` 的 dInfer benchmark 路线 | 这条路线不依赖 `sglang-kernel` 的额外兼容修复，更适合当前环境快速跑通正式实验 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| `lib_cite/sglang/python/sglang/srt/diffusion` 目录初查不存在 | 1 | 改为全仓搜索 diffusion 相关文档、scheduler 分支和 llada 关键词定位 |
| `import dinfer` 因缺失 `vllm` 失败 | 1 | 安装 `vllm==0.10.2` |
| `modeling_llada2_moe.py` 与当前 `transformers` / `flash_attn` 不兼容 | 1 | 增加局部兼容回退：`is_torch_fx_available`、`flash_attn` 导入降级、默认 RoPE 计算 |

## Notes
- Update phase status as you progress: pending → in_progress → complete
- Re-read this plan before major decisions
- Log ALL errors - they help avoid repetition

---

# Task Plan: BSP-MoE Monkey-Patch Validation

## Goal
验证 Block Sequence Parallel MoE（BSP-MoE）是否能在 C12 真实形态下减少 TP group 内 MoE gate/routing/shared 的冗余，并判断它是否比 Scheme3 routing-logits 节省更值得继续投入。

## Current Phase
Phase B5

## Phases

### Phase B1: Recovery & Constraints
- [x] Read handoff, repo rules, insight ledger, Scheme3 C12 results.
- [x] Check worktree and GPU availability.
- **Status:** complete

### Phase B2: Source-Level Patch Design
- [x] Read C12 baseline script and Scheme3 script.
- [x] Read LLaDA2 MoE forward implementation.
- [x] Check vLLM SP/chunk/gather helpers in local source.
- [x] Decide minimal BSP patch point that preserves EB/S_mask semantics.
- **Status:** complete

### Phase B3: Independent BSP Script
- [x] Create `codex_coding/src/bench_bsp_moe_dp2.py`.
- [x] Add shape-probe mode before changing computation.
- [x] Add optional BSP-MoE path without Scheme3/topk compact/pruning.
- **Status:** complete

### Phase B4: Experiments
- [x] Run shape probe on C12-like config.
- [x] Run smoke quality/performance test.
- [x] Run standard C12 timing if smoke is valid.
- **Status:** complete

### Phase B5: Archival
- [x] Save result files under `codex_coding/results`.
- [x] Add process doc under `code_building/process_docs/v0.1-init-project/`.
- [x] Append progress and key-file/insight updates.
- **Status:** complete

## BSP Guardrails
| Guardrail | Reason |
|---|---|
| First isolate BSP-MoE only | Avoid mixing Scheme3, topk compaction, expert pruning, or scheduler changes. |
| Shape probe before speed claims | Need actual `bsz`, `seq_len`, `N_dp`, `N_sp`, block id, and path type. |
| EB cold/hot_update use global token view | `S_mask` computed from TP-local shards would change semantics and quality. |
| Keep collective order identical across TP ranks | Mismatched gather/reduce order can deadlock. |
| Report e2e, fwd count, ms/fwd, path counts, component timing, and manual quality | Matches current experiment reporting standard. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|

## BSP Result Summary

| Finding | Result |
|---|---|
| Shape correctness | C12 `local_bs=256`, block `seq_len=32` gives `N_dp=8192`; BSP with `tp=4` gives `N_sp=2048`, no padding. |
| EB/S_mask behavior | Path counts identical between baseline and BSP: `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`. |
| No-timing e2e | Baseline `20.1285s / 75.675 ms/fwd`; BSP `19.8475s / 74.615 ms/fwd`; BSP is `-1.40%`. |
| Component timing | Dispatch payload drops `826.877 -> 206.719 MB/fwd`, but combine and TP gather overhead consume the gain. |
| Quality | Five visible verifiable prompts show no BSP-specific semantic degradation; snippets are partially truncated. |

---

# Task Plan: BSP-MoE nsys Collective Profiling

## Goal
用 Nsight Systems sqlite trace 定位 BSP-MoE 为什么没把理论收益稳定兑现到 wall-clock，重点判断 AgRs combine 和 TP all-gather 是否放大，以及 BSP 是否减少了其他开销。

## Current Phase
Phase N5

## Phases

### Phase N1: Trace Recovery
- [x] Confirm A/B short NVTX nsys traces exist.
- [x] Verify logs and sqlite schema.
- **Status:** complete

### Phase N2: Analysis Tool
- [x] Add `codex_coding/src/analyze_nsys_bsp.py`.
- [x] Parse logs, NVTX generate window, component ranges, kernel categories, collectives, memcpy.
- **Status:** complete

### Phase N3: Run And Validate
- [x] Run py_compile.
- [x] Generate JSON and Markdown reports.
- [x] Check key tables manually.
- **Status:** complete

### Phase N4: Interpretation
- [x] Compare nsys short trace with prior full C12 component timing.
- [x] Read vLLM AgRs source to explain group/layout change.
- **Status:** complete

### Phase N5: Archival
- [x] Add process doc.
- [x] Update progress summary, key files, key conclusions, findings, progress.
- **Status:** complete

## nsys Result Summary

| Finding | Result |
|---|---|
| Short trace e2e | A `69.60 ms/fwd`, B `76.72 ms/fwd`; BSP slower under nsys by `+10.23%`. |
| Generate NVTX | rankmax generate window A `3895.928 ms`, B `4406.343 ms`; B slower by `+13.10%`. |
| NCCL total | NCCL rankmax ms/fwd A `7.993`, B `17.114`; `+114.12%`. |
| NCCL AllGather | A `1.837 ms/fwd`, B `5.962 ms/fwd`; `+224.53%`, count `9120 -> 18392`. |
| NCCL Reduce | A `2.381 ms/fwd`, B `7.573 ms/fwd`; `+218.05%`. |
| Work saved | D2D memcpy `41222.3 -> 21307.7 MB`; dense GEMM `0.441 -> 0.213 ms/fwd`. |
| Core conclusion | BSP saves some work, but current monkey-patch collective sequence more than eats the saving. |

---

# Task Plan: M1/M2/M3 BSP-DelayGather Construction

## Goal
按已讨论的两条路线继续建设收益兑现方案：先完成 M1/M2 保守 BSP/SP + delayed gather 主线，再以不改变 EB 算法为前提验证 M3 EB 三路径通信接口。

## Current Phase
Phase M5

## Phases

### Phase M1: Patch Recovery & D-path Safety
- [x] Recover previous D smoke state.
- [x] Detect orphan-rank hang from the initial D all-reduce attempt.
- [x] Switch M3 hot-update pop combine to explicit vLLM EP group.
- [x] Add all-rank controller diagnostics.
- **Status:** complete

### Phase M2: Smoke
- [x] Run A/B/C/D smoke at `batch=32,gen=32`.
- [x] Confirm D no longer hangs.
- [x] Confirm path counts are rank-consistent.
- **Status:** complete

### Phase M3: C12 E2E
- [x] Run A/B/C/D C12 no-timing at `batch=512,gen=256`.
- [x] Archive JSON result.
- [x] Identify current best path.
- **Status:** complete

### Phase M4: C12 Component Timing
- [x] Run A/B/C/D C12 component timing.
- [x] Attribute C vs B and D vs C differences.
- **Status:** complete

### Phase M5: Archival & Decision
- [x] Add process doc.
- [x] Update progress summary, key files, key conclusions, findings, and progress.
- **Status:** complete

## M1/M2/M3 Result Summary

| Finding | Result |
|---|---|
| D-path safety | After switching to explicit `get_ep_group().device_group`, A/B/C/D smoke and C12 runs complete without collective hang. |
| C12 no-timing best | C BSP-DelayGather: `20.227s -> 19.777s`, `76.04 -> 74.35 ms/fwd`, `-2.22%` vs A. |
| M3 interface | D path counts are rank-consistent and reports `ep_reduce_calls=1862`, `ep_reduce_mb=1.907` per rank. |
| M3 wall-clock | D is worse than C in C12 no-timing: `74.98 ms/fwd` vs C `74.35 ms/fwd`. |
| Component cause | C reduces B's `native_forward/quant_apply/combine`; D raises these back near B-level and does not remove token-level AgRs communication. |
| Next engineering target | Downshift C path to source feature flag first; keep M3 as interface proof until EB pop combine can move before token-level AgRs gather. |

---

# Task Plan: BSP C+ / C++ Upper-Bound Experiments

## Goal
在暂不下沉 dInfer/vLLM 源码的前提下，把 BSP 路线的更高收益上限先在独立实验脚本中跑出来：C+ 延长 SP layout 跨层生命周期，C++ 探测用更少同步点的 full EP all-reduce probe 替代当前 combine+TP gather 序列的潜力。

## Current Phase
Phase U5

## Phases

### Phase U1: Patch Recovery & Wiring
- [x] Recover existing partial C+ / C++ edits in `bench_bsp_moe_dp2.py`.
- [x] Fix runtime scope issue in F probe by passing `dp_rank` explicitly.
- [x] Wire E/F paths into compare mode, summary, JSON output, and component timing.
- **Status:** complete

### Phase U2: Smoke
- [x] Run 8GPU A/B/C/D/E/F smoke at `batch=32,gen=32,no_quality`.
- [x] Confirm E/F complete without hang and path counts are rank-consistent.
- [x] Archive smoke JSON.
- **Status:** complete

### Phase U3: C12 E2E
- [x] Run C12 no-timing A/B/C/D/E/F at `batch=512,gen=256`.
- [x] Repeat C12 no-timing once to reduce timing noise.
- [x] Archive both JSON results.
- **Status:** complete

### Phase U4: Component & Quality Checks
- [x] Run C12 component timing for A/B/C/D/E/F.
- [x] Run small quality smoke with snippets for A/B/C/D/E/F.
- [x] Attribute E/F performance behavior.
- **Status:** complete

### Phase U5: Archival & Decision
- [x] Add process doc.
- [x] Update progress summary, key files, key conclusions, findings, and progress.
- **Status:** complete

## C+ / C++ Result Summary

| Finding | Result |
|---|---|
| E path purpose | C+ keeps SP hidden state across sparse decoder layer boundaries and gathers full layout only before attention / final model norm. |
| F path purpose | C++ P0 probe keeps EP dispatch + local expert compute, skips reduce_scatterv combine, and all-reduces a full token buffer as an upper-bound collective experiment. |
| C12 no-timing best | E BSP-CrossLayerSP is best in both runs: `71.90` and `71.69 ms/fwd`. |
| Average no-timing speed | A avg `75.73 ms/fwd`; C avg `74.14`; E avg `71.80`; F avg `72.46`. |
| Average speedup | E is `-5.20%` vs A; F is `-4.32%`; C conservative delayed gather is `-2.10%`. |
| Invariants | A-F C12 path counts are rank-consistent: `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`. |
| Component read | E does not reduce payload further but extends SP lifetime and keeps C-like lower `native_forward/quant_apply/combine`; F reduces wrapper/native time but pays `1311.020 MB/fwd` EP full all-reduce. |
| Quality smoke | Small verifiable snippets for E/F are coherent; formatting differs from A/B/C, so full quality evaluation is still needed before production claims. |
| Next engineering target | E/C+ is the strongest source-downshift candidate; F is a useful collective-design probe, not a production design yet because payload is large. |

---

# Task Plan: BSP-G vLLM SP-Parity Experiments

## Goal
在不进入 BSP-H/F2 跨 group collective 设计、不下沉 dInfer/vLLM 源码的前提下，先把 vLLM SP-parity 能搬的局部优化吃干净：用 attention output reduce-scatter 让 attention 后 residual/norm/MoE 直接进入 SP layout，并验证它是否比 C+/E 的 cross-layer SP 进一步减少同步点或 layout 转换成本。

## Current Phase
Phase G5

## Phases

### Phase G1: Wiring
- [x] Add experimental `G) C12-BSP-G-AttnReduceScatterSP` path in `codex_coding/src/bench_bsp_moe_dp2.py`.
- [x] Patch attention output projection to return SP layout via TP reduce-scatter.
- [x] Preserve EB/S_mask controller path and rank diagnostics.
- **Status:** complete

### Phase G2: Smoke
- [x] Run 8GPU smoke at `batch=32,gen=32,no_quality`.
- [x] Confirm no collective hang.
- [x] Archive smoke JSON.
- **Status:** complete

### Phase G3: C12 E2E
- [x] Run C12 no-quality compare at `batch=512,gen=256`.
- [x] Archive C12 JSON.
- [x] Compare G against A baseline and E/C+.
- **Status:** complete

### Phase G4: Component Timing
- [x] Run C12 component timing for A/B/C/D/E/G/F.
- [x] Attribute G's gain to attention reduce-scatter, MoE buckets, TP gather payload, or wrapper/layout effects.
- [x] Decide whether G is a BSP-G source-downshift candidate or only a monkey-patch upper-bound.
- **Status:** complete

### Phase G5: Archival & Decision
- [x] Add process doc.
- [x] Update findings/progress/key files/key conclusions.
- [x] State next action: finish BSP-G quality/source design or stop and move to BSP-H/F2.
- **Status:** complete

## BSP-G Result Summary

| Finding | Result |
|---|---|
| Smoke liveness | A/B/C/D/E/G/F smoke completed; G did not hang. |
| Smoke timing | G `53.69 ms/fwd` on smoke; smoke is only liveness/invariant, not a speed claim. |
| C12 no-timing best | G `69.55 ms/fwd`, better than A `75.35`, E `71.67`, and F `72.28`. |
| C12 speedup | G is `-7.70%` vs A and `-2.96%` vs E in the first BSP-G C12 run. |
| EB invariants | G path counts match A/E/F: `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`. |
| Repeat stability | G repeat is `69.51 ms/fwd`; two-run average is `69.53 ms/fwd`, `-8.55%` vs A average `76.03`. |
| G vs E | Two-run average G is `-2.93%` vs E/C+ average `71.63 ms/fwd`. |
| Component cause | G reduces `moe.bsp_chunk` from E's `0.872 ms/fwd`, count `5320`, to `0.003 ms/fwd`, count `266`, by making attention output SP directly. |
| Communication cost | G adds `attn.tp_reduce_scatter=5.020 ms/fwd` and `attn_rs_payload=661.502 MB/fwd`; MoE dispatch and TP gather payload stay at `206.719` and `165.375 MB/fwd`. |
| Quality smoke | G snippets are coherent on small verifiable prompts, with local formatting differences; full quality set is still required. |
| Decision | BSP-G should be the next source-downshift candidate before BSP-H/F2; do not change EB/s_mask algorithm. |

## BSP-G Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| First C12 component-timing run OOMed in A baseline prefill because GPUs still reported old benchmark PIDs consuming about `56-58 GB` per card. | 1 | Treat as environment failure, not BSP-G logic failure; verify no live `bench_bsp_moe_dp2`/`torchrun` process and no active compute apps before rerun. |

---

# Task Plan: BSP-G2 vLLM SP-Parity Bundle

## Goal
继续搬运 vLLM SP-parity 中当前能在独立实验脚本落地的收益组件，一次性形成 BSP-G2 bundle，再统一测试，避免后续流程碎片化。G2 必须保留 E 作为独立 reference，不污染 E；不进入 BSP-H/F2 跨 group collective，不改变 EB/s_mask 算法。

## Current Phase
Phase G2-5

## Phases

### Phase G2-1: Code Design & Wiring
- [x] Add G2-specific SP layout carrier / attention-input carrier.
- [x] Move SP-to-full attention input gather into patched attention path for G2.
- [x] Keep E path unchanged.
- [x] Add `G2) C12-BSP-G2-SPParityBundle`.
- [x] Add `--config-set` to run reduced matrices such as `A/E/G2/F`.
- **Status:** complete

### Phase G2-2: Smoke
- [x] Run `A/E/G/G2/F` smoke at `batch=32,gen=32,no_quality`.
- [x] Confirm no hang.
- [x] Archive smoke JSON.
- **Status:** complete

### Phase G2-3: C12 E2E
- [x] Run C12 no-quality repeat set.
- [x] Compare G2 against A, E, and old G.
- [x] Archive JSON.
- **Status:** complete

### Phase G2-4: Component Timing & Quality
- [x] Run C12 component timing for the reduced matrix.
- [x] Attribute G2 changes to attention input gather, attention RS, chunk count, and payload.
- [x] Run quality smoke.
- **Status:** complete

### Phase G2-5: Archival & Decision
- [x] Add process doc.
- [x] Update findings/progress/key files/key conclusions.
- [x] Decide whether G2 is better than G or just code-organization parity.
- **Status:** complete

## G2 Guardrails

| Guardrail | Reason |
|---|---|
| E remains unchanged | E is the reference for cross-layer SP before attention-RS/G2. |
| No BSP-H/F2 collective fusion | G2 is SP-parity bundle, not hierarchical/fused collective design. |
| No EB/s_mask algorithm change | C12 invariants already show compatibility; avoid confounding. |
| Reduced matrix by default for new experiments | User requested only A/E/F/G-style runs after this point. |

## BSP-G2 Result Summary

| Finding | Result |
|---|---|
| Smoke liveness | G2 completed without hang. |
| C12 invariant | A/E/G/G2/F all preserve `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`. |
| C12 e2e | G2 `69.661 ms/fwd`, G `69.676 ms/fwd`, A `75.428 ms/fwd`. |
| G2 vs A | `-7.646%` in C12 e2e. |
| G2 vs G | Effectively tied in e2e; difference is only `0.015 ms/fwd`. |
| Component timing | G2 `79.786 ms/fwd` vs G `78.149 ms/fwd` under instrumentation. |
| Bucket migration | G2 moves old G `moe.tp_all_gather=2.662 ms/fwd,count=5054` to `attn.input_all_gather=2.573 ms/fwd,count=4788`; residual `moe.tp_all_gather=0.141,count=266`. |
| Payload migration | G `tp_gather_payload=165.375 MB/fwd`; G2 `tp_gather_payload=8.704 MB/fwd` plus `attn_input_gather_payload=156.671 MB/fwd`. |
| No new byte reduction | G and G2 both have `attn_rs_payload=661.502 MB/fwd` and `dispatch_payload=206.719 MB/fwd`. |
| Quality smoke | G2 visible snippets match G closely; no obvious new degradation. |
| Decision | G remains the measured best performance path; G2 is source-organization/SP-parity, not a new speed step. |

---

# Task Plan: vLLM SP-Parity Inventory Confirmation

## Goal
维护一张 vLLM SP-parity / SP-MoE / sequence-parallelism 点位确认表，逐项确认哪些已经搬运、哪些是 vLLM 现成但未搬、哪些是我们 dLLM/BSP 自己探索，避免漏掉本可复用的性能收益。

## Current Phase
Phase I1

## Phases

### Phase I1: Inventory Archival
- [x] Create standalone inventory process doc.
- [x] Add vLLM existing SP components table.
- [x] Add our BSP-specific extension table.
- [x] Add confirmation checklist and guardrails.
- **Status:** complete

### Phase I2: Source Confirmation
- [ ] Confirm LLaDA2 shared expert / dense MLP SP behavior.
- [ ] Confirm available all2all backends and SP support in current environment.
- [ ] Confirm compilation-level sequence parallelism portability.
- [ ] Confirm residual-scattered/static-size requirements for source landing.
- **Status:** pending

### Phase I3: Decision
- [ ] Mark each item as portable / not applicable / postponed.
- [ ] Decide next source-downshift target after G.
- [ ] Decide whether BSP-H/F2 should use existing backend or custom collective.
- **Status:** pending

## Inventory Doc

- `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12q-vllm_sp_parity_inventory.md`

---

# Task Plan: BSP-G3 to BSP-G7 vLLM SP Completion

## Goal
按已确认路线连续完成 BSP-G 后续 vLLM SP-parity 补齐：确认 shared expert/dense MLP SP handling、设计 BSP-G source-downshift、验证 AsyncTP/compilation-level SP pattern-hit、盘点 MoE communication backend，并固化 source landing 约束。

## Current Phase
Phase G3

## Phases

### Phase G3: VSP-06 Shared Expert / Dense MLP SP Handling
- [x] Compare vLLM shared expert SP handling with LLaDA2 MoE implementation.
- [x] Confirm current BSP-G shared expert path has or does not have hidden TP redundancy.
- [x] Update inventory and process doc.
- **Status:** complete

### Phase G4: BSP-G Source-Downshift Design
- [x] Identify BSP-G/G2 monkey-patch code boundaries.
- [x] Define source-level feature flags, layout metadata, and fallback behavior.
- [x] Produce source landing design without changing mainline yet.
- **Status:** complete

### Phase G5: VSP-10 AsyncTP / Compilation-Level SP Pattern-Hit
- [x] Read vLLM compilation SP and collective fusion passes.
- [x] Verify whether current LLaDA2/BSP-G graph or source can hit relevant patterns.
- [x] Decide whether to pursue compiler integration or source-level equivalent.
- **Status:** complete

### Phase G6: VSP-08 Backend Inventory
- [x] Inventory AgRs, Naive, DeepEP, PPLX, FlashInfer Cutlass availability.
- [x] Check SP + DP + EP + TP support and local environment status.
- [x] Rank backend candidates by expected payoff and risk.
- **Status:** complete

### Phase G7: VSP-11/VSP-12 Source Landing Constraints
- [x] Inspect residual scattered/static-size/CUDA graph source constraints.
- [x] Define mandatory invariants for source landing.
- [x] Produce final BSP-G3-G7 report.
- **Status:** complete

## Guardrails
| Guardrail | Reason |
|---|---|
| Archive before each experiment/verification step | Preserve context through model compression or interruption. |
| Do not alter EB/s_mask algorithm | BSP-G correctness currently relies on preserved C12 path counts. |
| Do not introduce Scheme3 payload changes | Keep BSP/SP payoff attribution clean. |
| Stop for user decision before installing/compiling new backend dependencies | Backend changes can destabilize the environment. |

## Process Doc

- `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12r-bsp_g3_to_g7_vllm_sp_completion.md`

---

# Task Plan: BSP-G Source / Backend / Compile Execution

## Goal
按已确认顺序把 BSP-G 从实验脚本推进到 dInfer source/backend/compile 闭环：先验证 DeepEP distributed 可用性，再下沉 dInfer BSP-G 源码路径，复测 A/E/G，随后验证 AsyncTP compile pattern-hit 和 backend A/B。

## Current Phase
Phase S4/S5 evaluation complete; exact QKV `AllGather -> GEMM` fusion is not performance-positive for current C12 shape, so next decision should return to BSP-H/F2 collective/layout redesign rather than AsyncTP QKV source integration.

## Phases

### Phase S1: DeepEP Distributed Smoke
- [x] Create standalone DeepEP distributed smoke script.
- [x] Run 8GPU import + Buffer init + dispatch/combine smoke.
- [x] Archive JSON result and update process doc.
- **Status:** complete

### Phase S2: dInfer BSP-G Source Landing
- [x] Add feature-flagged BSP-G source path to LLaDA2 dInfer model code.
- [x] Preserve EB/s_mask semantics and source fallback.
- [x] Avoid forward-time monkey-patch style module mutation.
- **Status:** complete

### Phase S3: A/E/G Source Retest
- [x] Run smoke no-quality.
- [x] Run C12 no-quality latency.
- [x] Run component/backend timing where needed.
- [x] Confirm C12 path-count invariant.
- **Status:** complete

### Phase S4: AsyncTP / Compile Pattern-Hit
- [x] Route standalone dInfer-owned probe through vLLM compile-managed backend.
- [x] Enable sequence parallelism and async TP flags.
- [x] Verify `GEMM+RS` / `AG+GEMM` pattern-hit and archive blockers.
- [x] Validate raw fused SUM feasibility before considering a dInfer-local SUM pass.
- [x] Validate exact QKV `AllGather+GEMM` performance before source integration.
- **Status:** complete

### Phase S5: AgRs vs DeepEP Backend A/B
- [x] Compare AgRs and DeepEP after S1 smoke passed.
- [x] Keep AgRs as reference because DeepEP HT C12 is slower in current shape/config.
- **Status:** complete

## Guardrails
| Guardrail | Reason |
|---|---|
| Archive before every experiment | Preserve context through compression/interruption. |
| Do not change EB/s_mask algorithm | Maintain C12 invariant and quality attribution. |
| Do not introduce Scheme3 payload changes | Keep BSP/SP payoff clean. |
| Do not TP-shard shared/dense MLP until timing justifies it | Current replicated MLP is not the main blocker and naive TP can add communication. |

## Process Doc

- `/home/wuhang/wuhang/dllm_wh/code_building/process_docs/v0.1-init-project/v0.1.15.12s-bsp_g_source_backend_compile_execution.md`
