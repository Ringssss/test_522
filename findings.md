# Findings & Decisions

## Requirements
- 阅读本地 `lib_cite/dInfer` 和 `lib_cite/sglang` 的 diffusion LLM 相关部分。
- 解释什么是 diffusion LLM。
- 解释其主要推理流程。
- 对比 diffusion LLM 与普通自回归模型的差异。
- 梳理当前主流框架中的 dLLM 加速技术、原理与实践状态。
- 给出适合作为后续优化方向的判断。

## Research Findings
- `dInfer` README 将框架拆成 model、diffusion iteration manager、decoder、KV-cache manager 四层，并强调支持 batched inference、threshold parallel decoding、cache 与 block diffusion。
- `dInfer` 目前覆盖 LLaDA、LLaDA-MoE、LLaDA2；其中 README 明确指出 LLaDA2 走 SGLang backend，LLaDA/LLaDA-MoE 可走 vLLM backend。
- `dInfer/python/dinfer/into_sglang/algorithm.py` 说明其核心是把 dInfer 的 parallel decoding strategy 接到 SGLang 的 diffusion framework 上。
- `sglang` 仓库中与 text diffusion LLM 直接相关的线索主要有：
  - `docs/supported_models/text_generation/diffusion_language_models.md`
  - `srt/managers/scheduler.py` 与 `schedule_batch.py` 中的 diffusion LLM 初始化分支
  - README 新闻中对 LLaDA 2.0 diffusion LLM 的支持说明
- `sglang` 仓库中大量 `docs/diffusion/*` 与 `multimodal_gen/*` 主要是 image/video diffusion，不应直接当作 text dLLM 结论来源。
- `dInfer` 的基础推理主线是：生成区初始化为 `mask_id`，按 block 迭代，多轮 forward 后通过 `ThresholdParallelDecoder` 等规则并行 unmask。
- `dInfer` 中可明确识别的加速层包括：
  - threshold / hierarchy / credit-based parallel decoding
  - prefix / dual / vicinity cache
  - block diffusion
  - iteration smoothing（连续 embedding 混合）
  - torch.compile / CUDA graph
  - dynamic batching
- `modeling_llada.py` 与 `modeling_fused_lladamoe.py` 都明确体现出 diffusion LLM 与 AR 的关键区别：默认非因果 / full attention，cache 语义不是简单 append-only。
- `sglang` 的 text dLLM 已经在 `srt.dllm` 中形成原生子系统，包含：
  - `DllmConfig`
  - `LowConfidence`
  - `JointThreshold`
  - 专用 scheduler mixin
- `SGLang` 的 `JointThreshold` 明确支持 `Mask-to-Token` 与 `Token-to-Token` 两阶段，这比单纯 threshold unmask 更接近“先成型再修正”的推理范式。
- 结合本地实现与官方资料，本轮判断当前更成熟的 text dLLM 路线是 `dInfer` 与 `SGLang`；通用 AR serving backend 更多还是被复用为底层承载，而不是原生 text dLLM 系统。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 以 `dInfer` 的解码代码作为 text dLLM 推理流程主线 | 本地仓库中这一部分最直接体现 token-level diffusion 解码 |
| 将 “框架能力” 与 “模型算法” 分开总结 | 避免把 threshold/hierarchy 之类采样策略与 kernel/TP/quantization 混为一谈 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| SGLang diffusion 关键词会同时命中文生图/视频模块 | 单独聚焦 text-generation 文档、scheduler 中 dllm 分支和 LLaDA 关键词 |

## Resources
- Local: `/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/README.md`
- Local: `/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/into_sglang/algorithm.py`
- Local: `/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python/dinfer/decoding/`
- Local: `/home/wuhang/wuhang/dllm_wh/lib_cite/sglang/docs/supported_models/text_generation/diffusion_language_models.md`
- Local: `/home/wuhang/wuhang/dllm_wh/lib_cite/sglang/python/sglang/srt/managers/scheduler.py`
- Web: `https://arxiv.org/abs/2502.09992`
- Web: `https://github.com/NVlabs/Fast-dLLM`
- Web: `https://github.com/inclusionAI/dInfer`
- Web: `https://lmsys.org/blog/2025-12-19-diffusion-llm/`

## Visual/Browser Findings
- 暂无。

*Update this file after every 2 view/browser/search operations*
*This prevents visual information from being lost*

---

## BSP-MoE Findings — 2026-04-27

### Source-Level Constraints
| Finding | Evidence | Implication |
|---|---|---|
| LLaDA2 MoE input is flattened from `[bsz, seq_len, h]` to `[bsz * seq_len, h]`. | `lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py`, `LLaDA2MoeSparseMoeBlock.forward` | BSP must reason about token count `N=bsz*seq_len`, not just batch size. |
| vLLM has native SP helpers: `sequence_parallel_chunk` and `tensor_model_parallel_all_gather`. | runtime package and `lib_cite/vllm/vllm/model_executor/models/utils.py` / Qwen3-MoE | BSP should reuse the vLLM-style MoE-before chunk and MoE-after all-gather pattern. |
| AgRs dispatch/combine changes group when `is_sequence_parallel=True`: DP group becomes EP group. | runtime `vllm/distributed/device_communicators/all2all.py` | BSP can use native FusedMoE dispatch/combine if forward context and local sizes are correct. |
| `set_forward_context(num_tokens=N_dp)` plus `sp_local_sizes(tp_size)` computes per-EP-rank local sizes as `ceil(N_dp/tp_size)`. | runtime `vllm/forward_context.py` | BSP forward must create context with global local-DP token count before calling SP FusedMoE. |
| EB/S_mask can preserve global semantics under native SP FusedMoE. | FusedMoE dispatch gathers hidden/router logits before `quant_method.apply` and custom routing | `ctrl.get_s_mask` still sees gathered global logits, not TP-local logits, if routing stays inside FusedMoE. |
| Direct runtime import of `FusedMoE` needs `sys.modules['deep_ep']=None`. | existing Scheme3 scripts and introspection failure | BSP script must keep the same import guard. |

### Current Patch Decision
Use a minimal monkey-patch at `LLaDA2MoeSparseMoeBlock.forward`:

1. Flatten `[bsz, seq_len, h] -> [N, h]`.
2. Use `sequence_parallel_chunk` to shard tokens across TP ranks.
3. Run shared expert, gate logits, and native `experts.forward_impl` only on the local token shard.
4. Temporarily set each `FusedMoE` instance to `is_sequence_parallel=True` and `sp_size=TP_SIZE`.
5. Use `tensor_model_parallel_all_gather(..., dim=0)` and trim padding to recover `[N, h]`.
6. Keep EB routing inside native FusedMoE so cold/hot_update `S_mask` is computed on global gathered logits.

### BSP-MoE Experiment Results

| Finding | Evidence | Implication |
|---|---|---|
| C12 BSP shape is valid after accounting for block length. | `batch=512, dp=2` gives `local_bs=256`; block forward uses `N_dp=256*32=8192`; `tp=4` gives `N_sp=2048`, no padding. | BSP has enough token granularity; earlier concern about only `64 tokens/rank` was incomplete because it ignored `block_length=32`. |
| BSP preserves EB/S_mask path behavior. | Baseline and BSP both report `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931` in C12 e2e. | The monkey-patch does not change cold/hot/skip timing. |
| BSP forward-check is functional but not bitwise identical. | Layers 0/9/18 have matching path counts; typical `abs_mean≈5e-4~1e-3`, `rel_max≈0.1%~0.7%`. | Output text can differ after many dLLM iterations; quality must be checked manually. |
| BSP dispatch-byte saving is real. | Component timing reports dispatch payload `826.877 -> 206.719 MB/fwd`. | TP-local MoE token redundancy is actually removed. |
| Current monkey-patch only gives small e2e gain. | No-timing C12: baseline `20.1285s`, BSP `19.8475s`, delta `-1.40%`. | Mechanism is promising, but current implementation is not yet a strong system win. |
| Current implementation overhead is the blocker. | Component timing run: BSP combine `8.284 ms/fwd` vs baseline `3.587`; BSP adds TP all-gather `2.618 ms/fwd`. | Next step should target native collective/layout integration, not more Python patch layers. |

## BSP-MoE nsys Collective Findings — 2026-04-27

| Finding | Evidence | Implication |
|---|---|---|
| BSP is slower under short nsys trace. | A `69.60 ms/fwd`; B `76.72 ms/fwd`; NVTX rankmax generate A `3895.928 ms`, B `4406.343 ms`. | Use nsys for relative breakdown, not absolute C12 speed. |
| BSP does reduce non-collective work. | D2D memcpy total `41222.3 -> 21307.7 MB`; dense GEMM rankmax `0.441 -> 0.213 ms/fwd`; full C12 dispatch payload `826.877 -> 206.719 MB/fwd`. | The BSP idea is not empty; there is real work reduction. |
| NCCL overhead dominates the regression. | NCCL rankmax `7.993 -> 17.114 ms/fwd`; AllGather `1.837 -> 5.962`; Reduce `2.381 -> 7.573`. | The blocker is collective layout, especially AllGather/Reduce. |
| AllGather count nearly doubles in BSP. | `NCCL_AllGather` count `9120 -> 18392`. | Matches native AgRs dispatch all-gather plus explicit TP all-gather output collection. |
| vLLM AgRs source explains the path shift. | `AgRsAll2AllManager` uses DP group when `is_sequence_parallel=False`, EP group when `True`. | BSP changes communication group/sequence, not just local token compute. |
| Next direction should be native integration. | Python BSP adds explicit gather and cannot fuse/reorder collectives. | Prioritize native BSP/SP MoE, combine/gather fusion, or preserving SP layout longer. |

## M1/M2/M3 BSP-DelayGather Findings — 2026-04-27

| Finding | Evidence | Implication |
|---|---|---|
| Initial M3 D smoke exposed a collective safety issue. | Rank0 exited while ranks 1-7 stayed alive on GPU, then were killed manually. | D-path collective calls need explicit group and rank diagnostics before any data is trusted. |
| Explicit vLLM EP group fixes the D-path smoke issue. | `dist.all_reduce(pop, group=get_ep_group().device_group)` completed A/B/C/D smoke and C12 runs. | M3 should use EP group in this C12 layout, matching AgRs sequence-parallel group semantics. |
| All A/B/C/D C12 path counts are invariant-consistent. | Every config reports `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931` and rank checks pass. | M1/M2/M3 experiment does not change EB path scheduling. |
| Current no-timing best is C BSP-DelayGather. | A `76.04 ms/fwd`; B `74.80`; C `74.35`; D `74.98`. | M1+M2 gives about `-2.22%` / `1.023x`; M2 is worth source-level downshift first. |
| D validates M3 interface, not performance. | D reports `ep_reduce_calls=1862`, `ep_reduce_mb=1.907` per rank but is slower than C. | Hot-update `pop[E]` all-reduce is tiny and compatible, but current hook position adds sync without removing token-level communication. |
| Component timing explains why C is better than B. | C vs B: `native_forward 38.240 -> 35.227`, `quant_apply 20.161 -> 18.705`, `combine 8.520 -> 6.916`. | Conservative delayed gather partially reduces overhead after MoE/residual layout handling. |
| Component timing explains why D is not better than C. | D vs C: `native_forward 35.227 -> 38.715`, `quant_apply 18.705 -> 20.655`, `combine 6.916 -> 8.801`. | M3 must move deeper, before token-level AgRs gather/routing boundary, to become a real communication saving. |

## BSP C+ / C++ Upper-Bound Findings — 2026-04-28

| Finding | Evidence | Implication |
|---|---|---|
| C+ cross-layer SP is now the best measured BSP path. | C12 no-timing run1: E `71.90 ms/fwd`; repeat: E `71.69 ms/fwd`; A baseline `75.61/75.86`. | Extending SP layout lifetime gives materially larger gain than the earlier conservative C path. |
| Average C12 no-timing gain is about 5.2%. | Two-run average: A `75.73`, C `74.14`, E `71.80`, F `72.46 ms/fwd`; E delta `-5.20%`. | The next source-downshift target should be C+/E, not merely the earlier C path. |
| E preserves EB path invariants. | A-F all report C12 `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`, and rank checks pass. | E does not require changing EB/s_mask algorithm or path schedule. |
| E does not reduce measured payload further. | Component timing E dispatch payload `206.719 MB/fwd`, TP gather payload `165.375 MB/fwd`, same as B/C. | E's win comes from layout lifetime / wrapper overhead reduction, not a new byte-volume reduction. |
| F all-reduce full probe is performance-positive but not production-ready. | C12 no-timing F `72.41/72.51 ms/fwd`; component timing F uses `ep_full_all_reduce=9.467 ms/fwd` and `1311.020 MB/fwd`. | Hierarchical/fused collective design is promising, but full-buffer all-reduce is too payload-heavy as a final design. |
| F confirms combine+gather replacement has upside. | Component timing F removes explicit `moe.combine` and `moe.tp_all_gather`, and wrapper/native drops to `29.195 ms/fwd`. | A production C++ path should target a hierarchical collective or fused reduce_scatterv+gather, not the raw full all-reduce. |
| Quality smoke is not a blocker but not sufficient. | Small `batch=32,gen=32` snippets for E/F remain coherent on verifiable prompts, with some formatting variation. | Before source landing, run the established quality set; snippets only prove no immediate catastrophic semantic break. |

## BSP-G vLLM SP-Parity Findings — 2026-04-28

| Finding | Evidence | Implication |
|---|---|---|
| G is now the best measured BSP path. | C12 no-quality run1 G `69.55 ms/fwd`; repeat G `69.51 ms/fwd`. | Attention output reduce-scatter into SP layout is a stronger local vLLM SP-parity optimization than E/C+ alone. |
| Two-run average speedup is stable. | A average `76.03 ms/fwd`; E average `71.63`; G average `69.53`. | G is `-8.55%` vs A and `-2.93%` vs E; the gain is not a single-run outlier. |
| G preserves EB/s_mask invariants. | C12 A/B/C/D/E/G/F all report `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`. | BSP-G does not require changing EB algorithm or path schedule. |
| G's component mechanism is layout lifecycle, not MoE payload reduction. | Component timing: G `moe.bsp_chunk=0.003 ms/fwd`, count `266`; E `0.872 ms/fwd`, count `5320`. | Attention output directly in SP layout removes repeated full-to-SP chunking after attention. |
| G still pays new attention communication. | Component timing: `attn.tp_reduce_scatter=5.020 ms/fwd`, `attn_rs_payload=661.502 MB/fwd`. | G moves the synchronization boundary; it does not make communication free. |
| G does not solve MoE combine/gather payload. | G dispatch payload `206.719 MB/fwd`, TP gather payload `165.375 MB/fwd`, same as E. | BSP-H/F2 remains needed later for hierarchical/fused collective work. |
| Quality smoke is acceptable but incomplete. | Small `batch=32,gen=32` snippets for G are coherent on verifiable prompts, with formatting variation. | Need full quality set before source landing or production claim. |
| Source-downshift priority changes from E to G. | G beats E by `2.10 ms/fwd` average and uses vLLM-like attention RS idea. | Next engineering step should be BSP-G feature-flag design before BSP-H/F2. |

## BSP-G2 vLLM SP-Parity Bundle Findings — 2026-04-28

| Finding | Evidence | Implication |
|---|---|---|
| G2 is live and C12-correct. | Smoke, C12 e2e, component timing, and quality smoke all complete without hang; C12 path counts stay `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`. | G2 does not change EB/s_mask path scheduling in the real C12 run. |
| G2 matches G in C12 e2e but does not improve it. | C12 e2e: G2 `69.661 ms/fwd`, G `69.676 ms/fwd`, A `75.428 ms/fwd`. | G2 is within measurement noise of G; no new speed step is proven. |
| G2 moves attention-input gather ownership as intended. | Component timing: G `moe.tp_all_gather=2.662 ms/fwd,count=5054`; G2 `attn.input_all_gather=2.573 ms/fwd,count=4788` and residual `moe.tp_all_gather=0.141,count=266`. | The source-design boundary is cleaner: attention owns input gather and output reduce-scatter. |
| G2 does not reduce bytes or sync points. | Payload moves from G `tp_gather_payload=165.375 MB/fwd` to G2 `attn_input_gather_payload=156.671 MB/fwd` plus `tp_gather_payload=8.704 MB/fwd`; both keep `attn_rs_payload=661.502 MB/fwd`. | G2 is accounting/layout-boundary parity, not a hierarchical/fused collective. |
| Component timing favors old G. | Component timing: G `78.149 ms/fwd`, G2 `79.786 ms/fwd`. | Keep G as the measured best performance reference; use G2 only if source landing needs the cleaner ownership model. |
| Quality smoke is acceptable but incomplete. | G2 visible snippets match G closely on the small verifiable prompts. | Full quality evaluation remains required before source landing. |

## BSP-G3 VSP-06 Shared Expert Findings — 2026-04-28

| Finding | Evidence | Implication |
|---|---|---|
| vLLM shared expert SP handling mainly disables TP Linear collectives. | DeepSeek/Llama4 construct shared expert/dense MLP with `disable_tp=is_sequence_parallel`; DeepSeek comment says SP input/output are sharded and weights are replicated, so no collective is needed. | This is important for TP-sharded MLP implementations. |
| Current LLaDA2 shared/dense MLP is not TP Linear. | `LLaDA2MoeMLP` uses plain `nn.Linear` for `gate_proj`, `up_proj`, and `down_proj`. | There is no hidden TP all-reduce to remove via `disable_tp`. |
| LLaDA2-mini does have shared expert, but it is already local-token under BSP-G. | Config: `num_hidden_layers=20`, `first_k_dense_replace=1`, `num_shared_experts=1`; BSP-G calls `shared_mod(hs_sp)`. | `VSP-06` is not a missing performance component for the current route. |
| Source landing still needs a guardrail. | If source-downshift accidentally runs shared expert on full layout, it would reintroduce TP-token duplication. | Preserve shared expert execution on SP-local tokens in BSP-G source design. |

## BSP-G4 Source-Downshift Design Findings — 2026-04-28

| Finding | Evidence | Implication |
|---|---|---|
| BSP-G's true performance mechanism is attention output reduce-scatter. | LLaDA2 attention `dense` is `RowParallelLinear`; G replaces its output all-reduce with token-axis `tensor_model_parallel_reduce_scatter`. | Source landing must implement this path natively; merely moving gather ownership is insufficient. |
| G should be the first source-downshift target. | G is measured-best; G2 ties e2e but is slower under component timing and does not reduce bytes/sync points. | Keep G2 as organization reference, not default source path. |
| Source landing needs explicit layout carrier/metadata. | Monkey-patch uses `SPHiddenState` to carry `hidden_sp`, `bsz`, `seq_len`, and original `N`. | A source-level carrier or equivalent metadata is needed before CUDA graph/static-size work. |
| FusedMoE SP context is mandatory. | Current path wraps each SP MoE call in `set_forward_context(..., num_tokens=N)` and uses `experts.is_sequence_parallel=True`. | Source path must construct/set FusedMoE SP mode and preserve DP/SP local size metadata. |

## BSP-G5 AsyncTP / Compilation-Level SP Findings — 2026-04-28

| Finding | Evidence | Implication |
|---|---|---|
| vLLM SP/AsyncTP passes are present. | `SequenceParallelismPass` and `AsyncTPPass` import in vLLM `0.11.0`. | VSP-10 is not a fake feature; it can be evaluated after source landing. |
| Required ops register under the benchmark DeepEP guard. | Probe JSON shows `vllm.all_reduce/reduce_scatter/all_gather`, `_C.rms_norm`, and `symm_mem.fused_*` exist after relevant imports. | Environment can support the pass stack in principle. |
| Current BSP-G benchmark is not actually using the pass manager. | Probe marks `current_benchmark_is_not_using_vllm_compile_pass_manager=true`; current code calls distributed wrappers directly. | Current timing data does not include AsyncTP fusion benefit. |
| BSP-G/G2 source shapes are conceptually close to AsyncTP patterns. | Source scan finds `dense.quant_method.apply -> reduce_scatter` and `all_gather -> query_key_value` adjacency. | Preserve these shapes during source-downshift to enable future pattern-hit. |
| Direct DeepEP import is unstable in this environment. | Import without `sys.modules["deep_ep"]=None` fails on `deep_ep._C` missing `ncclTeamWorld`. | Backend work must treat DeepEP as installed but currently ABI-broken unless rebuilt/fixed. |

## BSP-G6 Backend Inventory Findings — 2026-04-28

| Finding | Evidence | Implication |
|---|---|---|
| AgRs is current default and reference backend. | `VLLM_ALL2ALL_BACKEND` default is `allgather_reducescatter`; existing BSP-G runs use it. | Keep AgRs as baseline for comparisons. |
| Naive backend is not a performance target. | vLLM source describes it as debug/testing and it uses broadcast/all-reduce style operations. | Do not spend C12 time on it except for diagnosis. |
| PPLX is unavailable. | Probe reports `pplx_kernels` absent and `has_pplx=false`. | Requires separate install/build approval before testing. |
| DeepEP is ABI-broken in the current environment. | Direct import fails: `deep_ep._C ... undefined symbol: ncclTeamWorld`; benchmark guard disables it. | Do not test DeepEP until ABI/NCCL rebuild is fixed. |
| FlashInfer all2all is the most plausible backend smoke candidate. | Probe reports `has_flashinfer_all2all=true`. | Can be tested later with `VLLM_ALL2ALL_BACKEND=flashinfer_all2allv`, but isolate it because workspace/group assumptions may differ. |
| FlashInfer Cutlass MoE is not the next clean drop-in. | Capability exists, but vLLM enables unquantized FP16 path only under specific env/device/DP conditions; current env flag is off. | Not a priority for current bf16 BSP-G communication optimization. |

## BSP-G7 Source Landing Constraint Findings — 2026-04-28

| Finding | Evidence | Implication |
|---|---|---|
| Residual scattered is a static/runtime contract in vLLM. | `is_residual_scattered_for_sp()` requires SP enabled, TP > 1, TP-divisible token count, and token count in compile sizes. | BSP-G source landing needs explicit SP metadata and compile-size discipline. |
| CUDA graph capture sizes are filtered for SP. | vLLM removes capture sizes not divisible by TP when sequence parallelism is enabled. | Any future CUDA graph/source path must capture only TP-multiple token sizes. |
| Original token count must be preserved. | vLLM pads/rounds for SP but slices/gathers using original scheduled token count. | BSP-G carrier must keep original `N` and trim all-gather outputs. |
| Forward-time module mutation conflicts with cudagraph compilation. | vLLM compiler wrapper rejects buffer/module updates during forward under cudagraph mode. | Source landing should avoid per-forward monkey-patching flags; use construction-time flags or value metadata. |
| Current C12 shape is friendly. | `local_bs=256`, `block=32`, so `N=8192`, `tp=4`, `N_sp=2048`. | Shape is not the blocker for source landing; metadata and backend/compiler integration are. |

## BSP-G Source Backend/Shared Findings — 2026-04-28

| Finding | Evidence | Implication |
|---|---|---|
| DeepEP HT is runtime-usable but slower for current C12. | DeepEP HT C12: A `129.755`, E `77.394`, G `75.140`, GS `75.523 ms/fwd`; AgRs reference: A `75.522`, E `71.942`, G `69.719`, GS `69.784`. | Keep AgRs as BSP-G reference backend; do not pursue DeepEP HT unless dispatch shape/backend config changes materially. |
| DeepEP HT preserves EB path invariants. | A/E/G/GS all report `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`. | Backend swap did not change EB/s_mask scheduling, only performance. |
| Standalone dense MLP is not a meaningful current bottleneck. | Dense timing C12: `dense.mlp ~= 0.85 ms/fwd` across A/E/G/GS. | TP-sharding dense MLP has small upside and should not be folded into BSP-G source parity. |
| Sparse shared expert is measurable but not dominant. | BSP-G `moe.shared ~= 3.34 ms/fwd` while `moe.native_forward ~= 34.91`, `moe.quant_apply ~= 18.43`, dispatch/combine/gather/attn-RS remain larger. | Shared expert TP conversion is lower priority than routed MoE collective/fusion work. |

## BSP-G TP8 Topology Findings — 2026-04-28

| Finding | Evidence | Implication |
|---|---|---|
| vLLM does not expose independent EP size in `ParallelConfig`. | EP group is built as `data_parallel_size * tensor_parallel_size`; config has TP and DP but no independent `expert_parallel_size`. | `tp4,dp1,ep8` is not directly expressible in this SPMD benchmark. The legal no-DP 8-GPU comparison is `tp8,dp1,ep8`. |
| dInfer needed KV replication parity for TP8. | LLaDA2-mini has `num_key_value_heads=4`; TP8 requires replicated KV heads like vLLM `QKVParallelLinear`. | Added QKV loader parity before treating TP8 data as valid. |
| TP8 BSP-G is much faster in no-quality C12. | `dp1,tp8,ep8` GS `58.758 ms/fwd` vs `dp2,tp4,ep8` GS `69.784 ms/fwd`; G `58.637` vs `69.719`. | Removing DP duplication plus increasing SP sharding has major speed potential. |
| TP8 baseline A regresses. | A `92.434 ms/fwd` under TP8 vs `75.522` under dp2tp4. | TP8 is not generally better; it specifically helps SP/BSP paths where local token work shrinks. |
| TP8 changes EB/diffusion trajectory in C12. | E/G/GS fwd count changes `266 -> 269`, hot updates `931 -> 950`. | TP8 no-quality speed is not an invariant-preserving A/B; needs quality and root-cause checks. |
| TP8 quick quality smoke fails for BSP paths. | A baseline snippets are coherent; E/G/GS snippets collapse to blank lines/period fragments on several prompts. | Do not adopt TP8 as default. Treat it as a separate investigation thread. |

## GS Quality / AsyncTP Findings — 2026-04-28

| Finding | Evidence | Implication |
|---|---|---|
| dp2/tp4/ep8 GS visible smoke is not quality-broken. | GS snippets solve/start average-speed, quadratic, and logic prompts coherently; no blank/punctuation collapse. | Keep dp2/tp4/ep8 GS as the valid BSP-G source reference, subject to full quality later. |
| TP8 remains invalid despite speed. | TP8 E/G/GS snippets collapse to blanks, periods, or short fragments while A remains coherent. | Do not adopt TP8; root-cause separately. |
| AsyncTP is a compiler-level SP monetization path, not currently consumed. | vLLM `PostGradPassManager` adds `SequenceParallelismPass`/`AsyncTPPass`, but dInfer benchmark uses runtime collectives and ordinary torch compile paths. | Need a dInfer `VllmBackend` pattern-hit probe before expecting AsyncTP speedups. |
| vLLM AsyncTP target patterns are concrete. | Source registers `mm -> vllm.reduce_scatter` and `vllm.all_gather -> mm` replacements using `symm_mem.fused_*` ops. | First dInfer target should be attention output projection + reduce-scatter; all-gather+GEMM can follow via G2-style boundary. |

## BSP-GS TP8 Root-Cause Findings — 2026-04-28

| Finding | Evidence | Implication |
|---|---|---|
| TP8 collapse is not caused by missing attention-input gather. | Layout diagnostics show E/G/GS have `bad_qkv=0` under both dp2/tp4 and TP8; GS TP8 has `1024/1024` full-token QKV records. | Do not try to fix TP8 by changing attention/QKV layout; QKV already sees full sequence. |
| dp2/tp4 BSP/SP MoE equivalence remains small. | Forward-check abs_mean max is `~1.0e-3`; MoE internal routed FusedMoE abs_mean max is `~9.8e-4`. | dp2/tp4 remains the valid BSP-G/GS topology. |
| TP8 SP-MoE diverges before full generation feedback. | TP8 forward-check abs_mean max is `~0.81`; abs_max reaches `114.47`/`97.05` on checked layers. | TP8 quality collapse is rooted in local layer math, not just diffusion trajectory sensitivity. |
| TP8 token order, shared expert, and gate logits are correct. | MoE internal TP8: chunk gather identity `0`, shared gather `0`, gate logits abs_mean max `~4.9e-7`. | Root cause is not SP chunking, shared expert replication, or gate-logit computation. |
| TP8 failure is isolated to routed FusedMoE under `dp1,tp8,ep8`. | MoE internal TP8: routed FusedMoE abs_mean max `~0.87`, abs_max `138`, rel_max `~1.0`; total MoE matches that failure. | vLLM native SP-FusedMoE EP communication contract does not cover this topology. |
| Source reading explains the mechanism. | vLLM EP maps `tp8,dp1` to `FusedMoEParallelConfig(dp_size=1, ep_size=8, tp_size=1)`, and AgRs dispatch/combine in `FusedMoE.forward_impl` only runs when `dp_size > 1`. | A no-DP TP8 route needs a real EP/SP exchange path or different mapping; it is not a safe default. |

## DInferCompileBackend AsyncTP Findings — 2026-04-28

| Finding | Evidence | Implication |
|---|---|---|
| dInfer can reuse vLLM compile backend/pass stack through a thin adapter. | `DInferCompileBackend` creates vLLM `VllmBackend`; distributed probe compiles and runs on `dp2,tp4,ep8`. | The chosen “dInfer owns execution, vLLM owns graph optimization” architecture is viable. |
| AsyncTP pattern-hit works in this environment. | Probe cache contains `symm_mem.fused_all_gather_matmul` and `symm_mem.fused_matmul_reduce_scatter` on all ranks. | We are no longer blocked at import/API level; vLLM fused kernels can be reached from dInfer. |
| `AllGather -> GEMM` is semantically safe in the probe. | `all_gather_gemm` second compiled run vs eager: `max_abs=0.0`, `mean_abs=0.0`. | This is the best first source integration target, likely around attention-input gather + QKV. |
| `GEMM -> ReduceScatter` is not semantically safe as currently used. | `gemm_reduce_scatter` fused output differs from eager SUM by `max_abs ~= 4.6-5.0`, but matches `eager/tp` exactly. | Do not compile BSP-G attention O-proj RS through AsyncTP until sum/avg semantics are handled. |
| vLLM source explains the mismatch. | `AsyncTPPass` replacement uses `symm_mem.fused_matmul_reduce_scatter(..., "avg", ...)`, while current eager `tensor_model_parallel_reduce_scatter` uses NCCL `ReduceOp.SUM`. | Need either a sum variant, a mathematically safe compensation, or a source-level decision that this path expects average semantics. |
| Raw fused SUM is not currently worth landing. | `hidden=256`: SUM semantic pass but fused `0.3588 ms` vs eager `0.0642 ms`; square `hidden=2048`: strict SUM check fails by `max_abs=0.0625` and fused `0.3523 ms` vs eager `0.2052 ms`. | Do not implement a dInfer-local AsyncTP SUM pass now; keep O-proj RS on eager SUM and prioritize `AllGather -> GEMM` fusion exposure. |
| Exact O-proj rectangular shape confirms the no-go. | `x=[8192,512]`, `weight=[512,2048]`: SUM semantic pass with `max_abs=0.03125`, but fused SUM rankmax `0.3602 ms` vs eager `0.1847 ms`, average `0.3473 ms` vs `0.1506 ms`. | The issue was not just square-probe mismatch; current symmetric-memory fused SUM is slower for the real BSP-G O-proj shape too. |
| vLLM AsyncTP benefit is shape/pass-stack dependent. | `SequenceParallelismPass` rewrites all-reduce neighborhoods into RS/local/all-gather patterns, then `AsyncTPPass` fuses `mm + vllm.reduce_scatter`; its replacement uses `"avg"`. | vLLM using AsyncTP does not mean this dInfer O-proj SUM path can use it directly. Our credible next compiler target is `AllGather -> GEMM`, which was semantically exact. |
| Exact QKV `AllGather -> GEMM` raw fusion is correct but slower. | `x_sp=[2048,2048]`, `weight=[2048,768]`: raw fused semantic pass with `max_abs=0.0`, but fused avg/rankmax `0.4112/0.4636 ms` vs eager `0.1609/0.1823 ms`. | Do not integrate `symm_mem.fused_all_gather_matmul` for BSP-G QKV at current C12 shape. Keep eager all-gather+QKV GEMM. |
| Compiled QKV AsyncTP currently has a rank-inconsistent graph issue. | DInferCompileBackend QKV probe cache showed only ranks 0/4 emitted `symm_mem.fused_all_gather_matmul`; other ranks emitted eager `vllm.all_gather + aten.mm`, and the 50-iter compiled run produced no JSON after abnormal long runtime. | Even before performance, source integration needs all ranks to hit the same collective pattern. Current compiled QKV path is not production-safe. |
| Source explains why QKV fusion is weak for C12. | PyTorch symmetric-memory native async all-gather-matmul path is gated to global M `2048 < M <= 4096`; C12 QKV has global M `8192`, so it uses the decomposition/micro-pipeline path. | AsyncTP may help smaller decode-ish shapes, but not this large-token BSP-G prefill-like shape. BSP-H/F2 layout/collective redesign is higher upside. |
| The `M=4096` native threshold is not enough for a win. | With `TORCH_SYMM_MEM_ENABLE_NATIVE_ASYNC_TP=1`, `M=4096` fused rankmax `0.2069 ms` vs eager `0.0885 ms`, and repeated-call semantics failed with `max_abs=5.46875`. | The threshold only enables a native candidate; it is not a correctness/performance guarantee for our QKV shape. Do not use it in dInfer. |
| `8192 -> 2x4096` chunk adaptation is not viable. | Split2 native artifact: semantic failure `max_abs=5.5`; fused rankmax `0.4412 ms` vs chunked eager `0.1909 ms` and full eager `~0.148-0.150 ms`. | Do not adapt current C12 QKV by chunking around the native threshold; it adds collectives and breaks semantics under native path. |

## EB HetEval512 Law Probe — 2026-04-28

| Finding | Evidence | Implication |
|---|---|---|
| HetEval512 C12 configuration was recovered. | `batch=512,gen=256,block=32,dp=2,tp=4,ep=8`; all ranks reported `266` forwards and path counts `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`. | The law probe is aligned with the prior C12 benchmark shape. |
| EB reduces active experts materially at large batch. | Across 10,070 traced layer-forward records, EB top4 unique active experts average `197.98`, no-EB top4 averages `251.22`, and no-EB top8 averages `254.59`. | EB is still a real active-expert / HBM-weight-load control mechanism at HetEval512 scale. |
| The reduction is layer-dependent. | Best layers: L17 `25.17%`, L7 `24.90%`, L9 `24.17%`, L16 `24.02%`; weakest layers: L1 `13.73%`, L0 `14.24%`. | Per-layer budget/adaptive policy is justified; early layers may need conservative budgets. |
| Current EB path/phase does not strongly change active-expert reduction at HetEval512. | cold `22.11%`, hot_skip `21.17%`, hot_update `21.11%` reduction vs no-EB top4. | Large-batch EB behaves more like stable active-set control than phase-specific shrinkage under current parameters. |
| `S_mask` has very strong adjacent stability. | Adjacent Jaccard mean `0.9862`, p50/p90/p95 all `1.0`; update-step dips match `skip_m=5` and improve later in the block. | Previous-call reuse and hot-skip are strongly supported; hot updates can be delayed or made less frequent late in block. |
| EB is much better than random same-size expert selection. | Weighted coverage of current no-EB demand: EB `0.9681` vs random same-size `0.7733`. | The EB selected set is meaningful, not just arbitrary active-set truncation. |
| Large-batch global expert popularity is very strong. | Offline global-popularity same-size set has weighted coverage `0.9704`, slightly above EB `0.9681`. | A paper-worthy next direction is hybrid `global prior + EB correction`, and held-out validation is needed. |
| Previous-call `S_mask` is nearly as predictive as current EB. | Previous-call coverage `0.9658`; previous-block cold coverage `0.9328`. | Intra-block reuse is very safe as a planning signal; cross-block reuse needs correction. |
| EB reduces active experts but slightly worsens linear-placement EP skew. | Mean EP load skew: EB top4 `3.118`, no-EB top4 `2.970`, no-EB top8 `2.882`. | Active-expert reduction and EP load balancing must be separated; EB needs placement/replica/scheduler support for slow-rank tail. |
| Naive request grouping by expert-id centroid is not sufficient. | For group size 64, EB natural group union mean `168.97`, random `193.69`, centroid `176.61`; natural is best likely due prompt-order clustering. | Scheduler research should use better set-similarity/load-aware grouping, not centroid alone. |

## EPLB Step/Remap Debug Findings — 2026-04-30

| Finding | Evidence | Implication |
|---|---|---|
| Mainline does not naturally advance `EplbState.step()`. | Off run keeps `state_step` fixed (`window_step=0,rearrangement_step=12`) while generation finishes. | `window_size/step_interval` tuning alone cannot realize true EPLB remap effects in current path. |
| Per-forward step can trigger real rearrangement. | Logs show `Rearranging experts ...`; `state_step` advances to `window_step~13-14`. | Step path is reachable but not integrated safely/efficiently. |
| Initial step hook failure root cause is structural. | Traceback points to `rebalance_execute.py:292`, `assert len(expert_weights) == num_moe_layers`. | dInfer model object lacks full MixtureOfExperts runtime contract for rearrange path unless bridged. |
| After bridging the MoE contract, assertion can be removed. | `stepdiag_on_fixlen` / `stepdiag_on_fixview` show `errors=0`. | Structural integration problem is solvable by explicit state/weight registration. |
| Rearrangement cost dominates by a large margin. | `step_interval=16` run reaches `~515 ms/fwd` with one `10.39s` rearrange event; `step_interval=256` run (no rearrange) is `156.454 ms/fwd` near/off baseline `163.461`. | `step()` call shell is not the main issue; online expert-weight shuffle is the blocker. |
| Conservative gating is mandatory for production route. | Full per-forward rearrange is catastrophic; long-interval no-rearrange is stable. | Next source step should be `cold + min_gap + fail-open` rather than unconditional step/rearrange. |

## EPLB Runtime Tax Decomposition — 2026-05-01

| Finding | Evidence | Implication |
|---|---|---|
| Runtime-enable but record-off is the dominant tax. | Bench C12 `G` path: `OFF32=69.811 ms/fwd`, `ON32-off=73.368`, `ON34-off=72.591`, `E34-disable-runtime=70.346`. | The per-forward runtime map patch is the main cost, not E34 shape or cold record. |
| Static map-only path is near baseline. | `E34-disable-runtime` is only `+0.535 ms/fwd` vs `OFF32`, with path counts unchanged. | Build-time static placement can remain P1; runtime should not be on the default hot path. |
| `disable_after_build` splits build-time and runtime cost. | `load_state_dict` still loads external expert map and materializes weights, then `set_eplb_runtime_state(False)` turns off runtime mapping. | We can isolate static placement from runtime mapping cleanly. |
| Next experiment should verify P1-only payoff. | `P1 static-only` with `redundant=0 + disable_after_build` is the cleanest follow-up. | If it reaches baseline parity or better, runtime EPLB should be demoted to diagnostic-only. |


## P1 Static-Only Bench Validation — 2026-05-01

| Finding | Evidence | Implication |
|---|---|---|
| P1 static-only is near baseline. | C12 `bspg_source`: `OFF32 -> ON32-disable-runtime` gives A `+0.42%`, E `+0.31%`, G `+0.61%`, GS `+0.69%`. | Static placement can be the default path without meaningful slowdown. |
| Runtime-enable remains the dominant penalty. | Same C12 set: `OFF32 -> ON32-off` is A `+4.13%`, E `+4.59%`, G `+5.10%`, GS `+4.34%`. | Runtime EPLB should stay optional until the common-path tax is solved. |
| Semantics are unchanged in static-only mode. | Path counts remain `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`. | This optimization changes performance path only, not EB schedule logic. |
| Source-path split is usable in practice. | `DINF_EPLB_EXPERT_MAP_PATH` is loaded in `load_state_dict`; `eplb_runtime_disable_after_build` then disables runtime mapping. | The architecture cleanly supports “static map on / runtime off”. |


## EPLB Runtime Identity Map Tax Recovery — 2026-05-01

| Finding | Evidence | Implication |
|---|---|---|
| `runtime-enable` 主税可由 identity passthrough 大幅回收。 | 同版本 C12 `ON32-off`：关闭 fastpath `G=73.442`，开启 fastpath `G=70.481`（`-4.03%`）；`GS=73.114 -> 70.401`（`-3.71%`）。 | 当前主矛盾不是“必须关闭 runtime”，而是“identity 场景不应做 map 计算”。 |
| 关键场景已达到接近 baseline。 | 开启 fastpath 后 `ON32-off vs OFF`：A `+0.65%`，E `+0.35%`，G `+0.47%`，GS `+0.94%`。 | `runtime-enable` 在 `redundant=0` 下可作为可接受主线，而不再固定 +4~5% 税。 |
| `cold_only` 记录路径也明显收敛。 | 开启 fastpath 后 `ON32-cold vs OFF`：A `+1.15%`，E `+1.12%`，G `+0.62%`，GS `+1.28%`。 | cold record 的边际成本相对可控，后续可专注于 `redundant>0` 场景。 |
| 命中率验证为 100%。 | 诊断显示 `identity_hits=identity_checks=5054`（C12），`cache_miss=19`（每层一次），其后全 cache hit。 | 该优化是实打实命中，不是偶然抖动。 |
| EB 不变性保持。 | 各组 path counts 一致 `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`。 | 优化不改 EB/s_mask 语义，仅减少 EPLB runtime 纯开销。 |

## C12 Dual-Axis Bench (Latency + Load Balance) — 2026-05-01

| Finding | Evidence | Implication |
|---|---|---|
| Load-balance metrics are now first-class outputs in bench results. | `bench_bsp_moe_dp2.py` now emits `load_balance_runs` and `load_balance_summary` per config with `ep_load_cv`, `ep_load_max_mean`, `ep_load_p95_p50`, and weighted layer CV. | We can evaluate EPLB/BSP routes on both latency and load skew in the same run artifact. |
| OFF baseline has moderate skew and best latency among the three tested runtime modes. | C12 (`batch=512,gen=256`), OFF: A/E/G/GS = `80.549/78.819/76.619/76.640 ms/fwd`, with `ep_load_cv ≈ 0.073~0.076`, `ep_load_max_mean ≈ 1.138~1.143`. | OFF remains the reference line for both speed and current load-balance behavior. |
| `ON32-off` introduces a clear latency regression. | Relative to OFF, C12 `ON32-off` is A `+9.37%`, E `+7.34%`, G `+7.90%`, GS `+7.60%`. | Runtime-on with `redundant=32` is not yet performance-positive in current source path. |
| `ON32-off` also worsens load-balance skew under this measurement. | `ep_load_cv` rises from OFF `~0.073~0.076` to `~0.214~0.215` (about `+185%~+193%`), while `ep_load_max_mean` rises to `~1.198~1.202`. | Current runtime mapping/placement combination does not improve rank-level routed-load balance for this C12 path. |
| `ON32-cold` is almost identical to `ON32-off` in both latency and skew. | C12 `ON32-cold` vs `ON32-off` differs minimally; only expected diag change is `record_calls=171` vs `0`. | The dominant issue is not cold-record overhead; it is in the runtime-on mapping/placement path itself. |
| EB invariants remain intact across all three groups. | Path counts are identical: `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`. | The observed regressions are optimization-path issues, not EB semantic drift. |

## E=36 Config Backfill Impact (C12) — 2026-05-01

| Finding | Evidence | Implication |
|---|---|---|
| E=36 tuned config backfill is effective and was hit at runtime. | ON32 logs switched from default-config warning to `Using configuration ... E=36,N=512`. | Previous ON32-off numbers included fallback tax; E36 backfill removes that confounder. |
| E36 backfill recovers a meaningful part of ON32 runtime regression. | ON32-off (lb3, E36 config) vs ON32-off (lb2, fallback): A `-3.77%`, E `-3.21%`, G `-3.35%`, GS `-3.33%`. | Kernel-config mismatch was a real contributor to slowdown. |
| Even after E36 backfill, ON32-off remains slower than OFF. | C12 lb3: ON32-off vs OFF is A `+5.30%`, E `+3.86%`, G `+4.23%`, GS `+3.85%`. | Remaining overhead comes from runtime EPLB path and mapping/placement interaction, not just config fallback. |
| Load imbalance gap (OFF vs ON32-off) remains large after E36 backfill. | OFF `ep_load_cv ~0.073~0.076`; ON32-off `~0.214~0.215` (`+185%~+192%`). | E36 config backfill improves compute efficiency but does not solve routed-load skew under current ON32 mapping behavior. |
| The ON32 load shape shows “low tails + overloaded middle ranks”. | GS run0 ON32 rank totals: `[26.13M, 41.63M, 48.83M, 50.99M, 49.30M, 47.99M, 46.01M, 28.42M]` vs OFF more even `[35.89M..48.57M]`. | This is consistent with persistent routing-to-physical concentration, which can amplify straggler effects. |

## ON32 Init Placement (Joint P1/P5 vs Weight-Balance) — 2026-05-01

| Finding | Evidence | Implication |
|---|---|---|
| `weight_balance` init placement path is wired and stable. | Added `DINF_EPLB_INIT_PLACEMENT_MODE` and `DINF_EPLB_INIT_GLOBAL_EXPERT_LOAD_PATH`; smoke run (`batch=32,gen=32`) completed with unchanged path counts and expected E36 config hits. | We can iterate init placement policy without enabling runtime rearrange. |
| Under current implementation, `weight_balance` init does **not** beat old joint init on C12 latency. | C12 ON32-off (`redundant=32,record=off`) GS: old init `79.615 ms/fwd`; wb init `80.081 ms/fwd` (`+0.585%`). Similar pattern on G (`79.814 -> 80.000 ms/fwd`). | Initial load-aware packing alone is insufficient to yield speedup in this path; runtime map/select overhead and other factors still dominate. |
| `weight_balance` init only marginally changes rank-level skew relative to old init. | GS `ep_load_cv`: old `0.053773`, wb `0.053029` (only `-1.38%` relative). | The new packing policy currently provides limited additional balancing over existing joint init for this workload. |
| Both ON32 init variants are already much more balanced than historical OFF(lb3) in this metric, yet still slower. | GS `ep_load_cv`: OFF(lb3) `0.075507`; ON32-old `0.053773`; ON32-wb `0.053029`, while ON32 latency remains `+3.7%~+4.3%` vs OFF(lb3). | In this setup, rank-level routed-load CV is not the primary bottleneck for end-to-end latency; load-balance metric and latency are decoupled here. |
| EB invariants and map diagnostics remain unchanged across old/wb init. | Path counts remain `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`; map diag remains `identity_false=5054`, `record_calls=0`. | The comparison is fair: differences come from init placement policy only, not semantic drift. |

## Two-Choice Locality Route (Cold-Triggered) — 2026-05-01

| Finding | Evidence | Implication |
|---|---|---|
| `two_choice_lb` full-path is prohibitively expensive in current runtime path. | Smoke (`batch=32,gen=32`) vs `flat`: A `+47.12%`, E `+15.44%`, G `+17.15%`, GS `+16.70%`. | Two-choice cannot run on hot path by default. |
| Cold-triggered locality variant is functionally wired and measurable. | With `DINF_EPLB_TWOCHOICE_COLD_ONLY=1`, C12 GS diag reports `twochoice_multi=2,275,511`, `twochoice_lb_applied=2,275,511`, `decay_calls=171`, `update_calls=171`. | The “record/map split by cold boundary” mechanism works as intended. |
| Cold-triggered two-choice still lacks stable latency gain. | C12 (`batch=512,gen=256`) vs `flat`: A `+1.31%`, E `+1.68%`, G `-0.73%`, GS `+0.83%`. | Net effect is near-zero/slightly negative; not sufficient for mainline adoption. |
| Load-balance improvements are too small to offset overhead. | C12 GS `ep_load_cv` only `0.215393 -> 0.214857`; `ep_load_max_mean` `1.202093 -> 1.201047`. | Current local balancing signal is weak for this workload. |
| Benchmark path signal wiring was a hidden blocker and is now fixed. | `setup_routing` now pushes route path when `map_impl==two_choice_lb` even with `record_mode=off`; previously cold-only branch never triggered in this combo. | Future runtime-map experiments can trust cold/hot path-dependent behavior. |
| Runtime diag visibility is now reliable across policy context reset. | `reset_eplb_runtime_map_policy` no longer clears two-choice counters; per-run reset is centralized at bench run start. | Result JSON now reflects actual map-path behavior, reducing false conclusions. |
| EB semantic invariants remain preserved under all tested two-choice variants. | Path counts stay `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931` for C12 A/E/G/GS. | This route is optimization-safe semantically, but currently not performance-positive. |

## C12 Single-Run Profiling (OFF / ON32-flat / ON32-cold / P1-static) — 2026-05-01

| Finding | Evidence | Implication |
|---|---|---|
| Runtime-on remains slower under profiling runs. | GS `ms/fwd`: OFF `87.840`; ON32-flat `93.426` (`+6.36%`); ON32-cold `92.295` (`+5.07%`). | Runtime EPLB path still introduces measurable regression in this setup. |
| cold-only reduces only a small part of runtime penalty. | ON32-cold improves over ON32-flat by about `1.13 ms/fwd` on GS, but remains above OFF. | `record` path is not the dominant overhead; map-on/common path dominates. |
| P1 static-only is better than runtime-on but still above OFF in this run. | GS `91.046 ms/fwd` (`+3.65%` vs OFF), with runtime map diag all-zero. | Static placement alone avoids runtime tax, but there are still non-runtime variances to control in single-run profiling. |
| Load skew degrades in ON32 runtime modes. | GS `ep_load_cv`: OFF/P1 `0.0305`; ON32-flat/cold `0.0538`. | Current ON32 runtime path worsens rank-level routed-load balance in this measurement. |
| Payload volume is not the differentiator across these groups. | GS payloads are identical (`dispatch 206.719 MB/fwd`, `tp_gather 165.375`, `attn_rs 661.502`). | Regression source is not extra bytes; likely execution/synchronization behavior. |
| Critical component time rises on ON32 modes. | GS rank-max `ms/fwd` increases mainly in `moe.native_forward`, `moe.quant_apply`, `moe.combine`, and `moe.dispatch` when ON32 is enabled. | Optimization focus should shift to kernel/sync path interactions, not more mapping heuristics first. |
| Path semantics remain stable across all four groups. | All A/E/G/GS keep `prefill_fallback=19,cold=171,hot_skip=3933,hot_update=931`. | Changes affect performance path only, not EB scheduling correctness. |

## Context Compression Handoff Refresh — 2026-05-01

| Finding | Evidence | Implication |
|---|---|---|
| Recovery handoff has been rewritten and centralized. | `/home/wuhang/wuhang/dllm_wh/history-chat.txt` was fully overwritten with a structured recovery template and current state snapshot. | After context compression, reading `history-chat.txt` first is sufficient to re-enter the current task context quickly. |
| The current execution focus is preserved in the handoff. | New handoff explicitly records rank-tail expansion status, OFF vs ON32-cold tail evidence, and pending ON32-flat/P1-static tail runs. | Prevents post-compression drift into unrelated optimization branches; keeps work on profiling-first critical path. |
