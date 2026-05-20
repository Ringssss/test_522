# Active Threads

## Thread 1

- Name: Insight-led direction selection
- Status: **active**
- Priority: highest
- Current problem definition: Decide the next system optimization direction after Scheme3 8-GPU validation showed real but too-small standalone routing-logits communication savings.
- Confirmed facts:
  - The canonical insight ledger is `/home/wuhang/wuhang/dllm_wh/docx/context_index/04_insight_ledger.md`.
  - Scheme3 A/B/B2 8-GPU result: A `75.62 ms/fwd`, B `79.61 ms/fwd`, B2 `81.35 ms/fwd`.
  - Dispatch payload saving is real, but dispatch wall-time saving is only about `0.45-0.52 ms/fwd`.
  - No Scheme3-specific quality degradation was observed in the manual five-prompt check.
- Current hypothesis:
  - The next high-value direction is likely one of:
    - reducing forward count;
    - Sequence/TP/parallelism structure;
    - native active-expert reduction that actually lowers unique active experts / weight loading;
    - block-stage-aware serving scheduler.
- Missing evidence:
  - No updated scored decision table with `收益上限 / 实现侵入性 / 质量风险 / 验证成本`.
  - No next-direction-specific feasibility plan yet.
- Suggested next step:
  - Discuss the direction table with the user before any code work.

## Thread 2

- Name: Scheme3 route-before-dispatch
- Status: **deprioritized standalone**
- Priority: medium-low
- Current problem definition: Scheme3 is theoretically valid but currently negative in C12-AgRs monkey-patch form.
- Confirmed facts:
  - Routing-logits dispatch payload drops from about `826.9 MB/fwd` to about `706 MB/fwd`.
  - B route-before-dispatch is `+5.3%` slower than A.
  - B2 native-topk is `+7.6%` slower than A.
  - Path counts are identical across A/B/B2.
- Current hypothesis:
  - Scheme3 may still be useful only if fused into a native/Triton/vLLM path with near-zero Python/per-layer overhead, or combined with a larger structural change.
- Missing evidence:
  - No native implementation estimate beyond monkey-patch timing.
- Suggested next step:
  - Do not expand B/B2; only revisit if a larger structural plan needs it as a subcomponent.

## Thread 3

- Name: Sequence/TP/parallelism structure
- Status: **warm**
- Priority: high
- Current problem definition: C12 still has TP-side structural redundancy around gate/routing/shared and collective layout; this may exceed Scheme3 standalone gains.
- Confirmed facts:
  - C11/C12 work showed TP/parallelism structure can dominate local MoE patch gains.
  - Insight ledger marks this as I15 and links it to I14/I17.
- Current hypothesis:
  - Sequence Parallel or similar parallelism restructuring can remove redundant work and improve system-level efficiency.
- Missing evidence:
  - No concrete SP feasibility matrix yet.
  - No implementation-risk breakdown for attention output, MoE input/output, dispatch group, and LMHead.
- Suggested next step:
  - Compare SP against forward-count and scheduler directions before choosing.

## Thread 4

- Name: Native active-expert reduction
- Status: **warm**
- Priority: medium-high
- Current problem definition: Prior top-p/top-k results show theoretical expert-pair savings, but wall-clock benefit requires reducing unique active experts or expert weight loading.
- Confirmed facts:
  - Fused MoE is often expert-weight HBM-load bound.
  - Weight-zero pruning does not save kernel time.
  - Token-expert FLOP savings alone are not enough.
- Current hypothesis:
  - EB/topk/top-p could become useful if paired with active-expert compaction, expert placement, or kernel changes that reduce weight loading.
- Missing evidence:
  - No new native active-expert design after Scheme3 reprioritization.
- Suggested next step:
  - Treat as a candidate direction, not the default next implementation.

## Thread 5

- Name: Block-stage-aware serving scheduler
- Status: **warm**
- Priority: medium-high
- Current problem definition: dLLM block boundaries and cold/hot/skip phases may create scheduling opportunities not present in AR serving.
- Confirmed facts:
  - Block boundary is a natural synchronization and planning point.
  - DP AllToAll collective order alignment is a real dLLM serving constraint.
  - Expert popularity is data-dependent, so static placement assumptions are weak.
- Current hypothesis:
  - A scheduler that groups by block phase, EB state, or expected expert load could be a distinctive systems direction.
- Missing evidence:
  - No multi-request scheduling experiment yet.
  - No serving workload model yet.
- Suggested next step:
  - Include in the direction scoring table and decide whether it is too large for the next immediate experiment.
