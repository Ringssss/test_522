# Stage Summary

## Current Stage

`v0.1-init-project` is now in the dLLM / MoE / EB direction-selection stage after the C12-AgRs Scheme3 8-GPU validation.

The immediate goal is no longer to expand the current Scheme3 monkey-patch. The next step is to use the maintained insight ledger to choose the next system optimization direction.

## Completed Work

- Established the project rules and archive conventions from `/home/wuhang/wuhang/dllm_wh/docx/next_step.txt`.
- Built the early dInfer / LLaDA2.0-mini baseline and benchmark foundation; this is historical context and should not be repeated unless explicitly requested.
- Completed a long sequence of dLLM cache, IterSmooth, MoE reuse, padding-free MoE, top-p/top-k, and fused MoE kernel investigations.
- Implemented and validated C11:
  - TP attention.
  - LMHead TP.
  - `tp=4, ep=4, dp=1`.
- Implemented and validated C12:
  - `dp=2, tp=4, ep=8`.
  - True DP AllToAll EP behavior.
  - vLLM 0.11.0 `allgather_reducescatter` backend.
- Completed Scheme3 C12-AgRs 8-GPU standard validation:
  - A baseline: `75.62 ms/fwd`.
  - B route-before-dispatch: `79.61 ms/fwd`, `+5.3%`.
  - B2 native-topk: `81.35 ms/fwd`, `+7.6%`.
  - Dispatch payload reduction is real: about `826.9 -> 706 MB/fwd`.
  - Dispatch wall time only improves about `0.45-0.52 ms/fwd`.
  - No Scheme3-specific output quality degradation was observed in the manual five-prompt check.
- Created the canonical insight ledger:
  - `/home/wuhang/wuhang/dllm_wh/docx/context_index/04_insight_ledger.md`.
- Updated the compression recovery handoff:
  - `/home/wuhang/wuhang/dllm_wh/history-chat.txt`.

## Verified Facts

- Current live project root is `/home/wuhang/wuhang/dllm_wh`.
- Current standard multi-GPU application-like config is:
  - `dp=2`
  - `tp=4`
  - `ep=8`
  - `batch_size=512`
  - `gen_length=256`
  - `block_length=32`
  - `VLLM_ALL2ALL_BACKEND=allgather_reducescatter`
- `S_mask` has strong within-block temporal stability, especially in hot skip paths.
- `S_mask` stability supports metadata/schedule reuse, not direct MoE output reuse.
- Scheme3's routing-logits communication saving is theoretically valid and empirically visible in dispatch payload.
- Under the current C12-AgRs backend, Scheme3's standalone wall-clock benefit is too small to overcome added routing/control overhead.
- Fused MoE wall time is often constrained by expert weight HBM loading, so token-expert FLOP savings do not automatically translate to wall-clock speedup.

## Current Hypotheses

- Reducing forward count remains the highest-leverage direction for dLLM acceleration.
- Sequence Parallel / TP / parallelism-structure optimization may be more valuable than local Scheme3 routing-logits communication reduction.
- Native active-expert reduction is only worth pursuing if it reduces unique active experts or expert weight loading.
- Block-stage-aware serving scheduling may be a strong dLLM-specific system contribution, but it needs multi-request evidence.

## Do Not Repeat

- Do not restart from early dInfer/LLaDA2 environment setup.
- Do not repeat early full benchmarks unless the user explicitly asks.
- Do not continue expanding Scheme3 B/B2 monkey-patches as the main path without a new reason.
- Do not evaluate output quality by string matching; use manual semantic judgment as requested by the user.
- Do not report only `ms/fwd`; include end-to-end time, forward count, path counts, and manual quality checks.
- Do not infer output-cache safety from `S_mask` stability.

## Recommended Next Step

Start with discussion, not code. Read `/home/wuhang/wuhang/dllm_wh/docx/context_index/04_insight_ledger.md`, then compare the candidate directions by:

| Direction | Key Question |
|---|---|
| Reduce forward count | Can we remove full model forwards without unacceptable quality loss? |
| Sequence/TP/parallelism structure | Can we remove TP-side redundant work or improve collective structure? |
| Native active-expert reduction | Can we reduce unique active experts / weight loading, not just FLOPs? |
| Block-stage-aware serving scheduler | Can block boundaries and cold/hot/skip phases drive serving-level scheduling? |
| Scheme3-style routing-logits communication | Should it only be combined with larger structural changes? |
