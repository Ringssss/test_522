# POC Build Plan on SGLang: What to Build, What Not to Build, and How to Ship the Paper

## 1. Goal of the POC

The POC should not attempt to become a new general-purpose serving system.

The correct goal is:

> Build a subgraph-aware layout optimization prototype inside SGLang that demonstrates clear gains on selected hotspot subgraphs and supports a compelling SC-style paper.

This means the POC should optimize for:
- research clarity,
- implementation focus,
- strong controlled experiments,
- minimal unnecessary system scope.

---

## 2. The right scope

### 2.1 What the POC should do
The POC should:
- identify a small number of hotspot subgraph patterns,
- assign and propagate layouts at the subgraph level,
- select among layout-specialized Triton kernel families,
- eliminate redundant layout conversions,
- optionally enable layout-dependent fusion,
- evaluate on realistic SGLang inference paths.

### 2.2 What the POC should not do
The POC should not try to:
- redesign the serving scheduler,
- reinvent paged KV cache management,
- solve speculative decoding,
- redesign distributed MoE communication,
- replace the entire graph compiler stack,
- support all models or all backends.

This discipline is essential for paper success.

---

## 3. Recommended build strategy

The best strategy is:

- keep SGLang as the serving host,
- keep Triton as the custom-kernel substrate,
- add a new optimization layer on top:
  **subgraph-aware layout planning**

This gives the project:
- a real serving system,
- strong baselines,
- realistic integration,
- and manageable engineering complexity.

---

## 4. Core POC architecture

The POC should contain five main components.

### 4.1 Hotspot subgraph extractor
A pattern matcher over selected inference graph segments.

Initial target patterns:
- quantized FFN / expert FFN chains,
- attention preparation or epilogue chains.

This can be implemented with simple pattern-based recognition first.

### 4.2 Subgraph Layout IR
A compact internal representation for:
- subgraph boundaries,
- edge layout candidates,
- operator legality constraints,
- conversion points,
- fusion eligibility.

This is the conceptual center of the POC.

### 4.3 Layout planner
Given a hotspot subgraph:
- enumerate legal layout candidates,
- evaluate compatibility,
- estimate costs,
- choose a subgraph-wide plan.

This is the main algorithmic contribution.

### 4.4 Layout-specialized Triton kernel families
Implement a small number of specialized kernels for each subgraph type.

Do not try to synthesize arbitrary kernels automatically in the first POC.
Instead, create a controlled menu of high-value kernel families.

### 4.5 SGLang integration hooks
Integrate the planner into the execution path so that:
- subgraph plans are generated,
- kernels are selected accordingly,
- benchmark tools can compare baseline vs optimized paths.

---

## 5. What to build first

### Phase 1: Minimum closed loop
Build only one hotspot class first:
- **quantized FFN / expert FFN**

Why:
- easier to control,
- highly relevant,
- strong layout sensitivity,
- easy to benchmark,
- directly useful for dense and MoE models.

Deliverables:
- one subgraph extractor,
- first version of Layout IR,
- first planner,
- 2-3 layout-specialized Triton kernel families,
- benchmark script integration.

### Phase 2: Second hotspot class
Add:
- **attention preparation / epilogue subgraphs**

This is important because it broadens the paper claim from “FFN-only engineering” to “subgraph-level method across multiple hotspot families.”

### Phase 3: Model-level and serving-level validation
Once two hotspot classes work, evaluate:
- one dense model,
- one MoE model,
- one Qwen3.5-like model.

At this stage, prioritize evaluation quality, not new features.

---

## 6. What is mandatory for the paper

### 6.1 Mandatory: a clear optimization abstraction
The paper must present a strong abstraction:
- layouts are edge-level states,
- operators constrain feasible layout combinations,
- conversions are explicit and avoidable,
- the planner chooses a subgraph-wide layout plan.

Without this, the project will look like heuristic autotuning.

### 6.2 Mandatory: automation
The system must automatically:
- pick among candidate layouts,
- decide when conversions can be removed,
- select layout-specialized kernel families.

Do not rely on manual selection for final results.

### 6.3 Mandatory: at least two hotspot subgraph classes
Only one subgraph type is too narrow for a strong paper.
Two is the minimum healthy target.

### 6.4 Mandatory: strong baselines
At minimum compare against:
- stock SGLang execution path,
- stock Triton path where applicable,
- a strong production backend when available,
- hand-tuned per-op kernels for chosen subgraphs.

### 6.5 Mandatory: layered evaluation
You must report:
- kernel/subgraph metrics,
- offline engine metrics,
- serving metrics.

Without this, the experimental story will be incomplete.

### 6.6 Mandatory: ablation studies
Ablations should isolate:
- planner contribution,
- conversion elimination,
- fusion contribution,
- search strategy,
- cost-model quality.

---

## 7. What is optional but valuable

These are useful, but not required for the first strong paper version:
- hybrid support for more backend choices,
- broader quantization formats,
- more than one GPU generation,
- broader model zoo,
- more aggressive fusion classes,
- planner caching across runs.

If schedule becomes tight, these should not outrank the core method.

---

## 8. What to avoid

### 8.1 Avoid modifying Triton core compiler internals
For the POC, avoid a Triton compiler fork.
The engineering cost is too high, iteration will be too slow, and the paper does not need it.

Instead:
- implement the layout planner externally,
- use Triton as a kernel substrate,
- emulate the relevant layout ideas at the planner/kernel level.

### 8.2 Avoid full runtime redesign
Do not turn this into:
- a scheduler paper,
- a KV cache paper,
- a distributed communication paper,
- a speculative decoding paper.

These are different projects.

### 8.3 Avoid too many hotspot classes
Two or three classes is enough.
More will reduce experimental quality and slow down iteration.

### 8.4 Avoid “benchmark-only” contributions
If the work becomes:
- a set of tuned kernels,
- plus benchmark tables,
- without a clear subgraph optimization framework,

it will look like engineering rather than research.

---

## 9. Suggested technical choices

### 9.1 Serving framework
**SGLang**
- real serving environment,
- strong backends,
- realistic model execution,
- built-in benchmarking tools.

### 9.2 Kernel substrate
**Triton**
- convenient for implementing multiple kernel families,
- aligned with the LinearLayout-inspired design philosophy,
- good balance between productivity and performance.

### 9.3 Planner implementation
**Python first**
- faster iteration,
- easier paper-driven experimentation,
- easier integration with SGLang execution code.

Only optimize planner runtime later if necessary.

### 9.4 Search strategy
For the first POC:
- dynamic programming over small chains, or
- beam search over layout states.

Do not start with heavy ILP or SMT.

### 9.5 Cost model
Use a hybrid design:
- analytic terms for structure,
- microbenchmark-based calibration for realism.

This is strong enough for a systems paper and practical enough to tune.

---

## 10. Suggested implementation roadmap

### Week 1-2: Baseline and instrumentation
Tasks:
- set up SGLang baseline environment,
- select one GPU platform,
- lock software versions,
- run baseline benchmarks,
- identify hotspot subgraphs in selected models,
- collect profiling traces.

Outputs:
- baseline numbers,
- hotspot list,
- measurement scripts.

### Week 3-4: First Subgraph Layout IR and planner
Tasks:
- define layout candidate schema,
- define legality rules for FFN/expert subgraphs,
- implement layout planner,
- implement conversion cost accounting,
- build a minimal search loop.

Outputs:
- IR spec,
- planner prototype,
- toy examples validated.

### Week 5-6: FFN/expert Triton kernel families
Tasks:
- implement 2-3 FFN/expert kernel families,
- integrate planner-driven kernel selection,
- run kernel-level and subgraph-level experiments.

Outputs:
- first performance wins,
- first ablation data.

### Week 7-8: Attention-side support
Tasks:
- add attention-prep or attention-epilogue subgraphs,
- implement corresponding legality rules,
- build attention-side kernel families,
- integrate into SGLang path.

Outputs:
- second hotspot class working,
- broader evaluation coverage.

### Week 9-10: Full evaluation and paper assets
Tasks:
- perform model-level evaluation,
- perform serving-level evaluation,
- run ablations,
- create plots/tables,
- draft method/experiment sections.

Outputs:
- final figures,
- clean experiment tables,
- strong paper narrative.

---

## 11. Code organization suggestion

A clean repository structure could look like this:

```text
project_root/
  planner/
    layout_ir.py
    legality.py
    search.py
    cost_model.py
    patterns.py

  kernels/
    ffn/
      family_blocked.py
      family_swizzled.py
      family_mma_aligned.py
    attention/
      family_blocked.py
      family_swizzled.py

  sglang_integration/
    backend_hooks.py
    plan_cache.py
    dispatch.py

  benchmarks/
    run_kernel_bench.py
    run_offline_bench.py
    run_serving_bench.py

  paper/
    figures/
    tables/
    notes/
```

This structure keeps method, kernels, integration, and evaluation clearly separated.

---

## 12. Experimental plan from a paper-writing perspective

The experiments should answer four questions:

### Q1. Does subgraph-aware layout planning outperform per-op choices?
Need:
- hotspot subgraph latency/speedup tables,
- conversion reduction statistics,
- fusion activation statistics.

### Q2. Is the method robust across models and shapes?
Need:
- dense + MoE + Qwen3.5-like coverage,
- batch and context-length variation,
- precision/quantization variation where feasible.

### Q3. Where do the gains come from?
Need:
- ablations,
- memory/conversion breakdown,
- kernel-family selection analysis.

### Q4. Does this matter in a real serving stack?
Need:
- offline throughput,
- serving metrics (TTFT/TPOT/ITL),
- integration overhead discussion.

If the experiments answer these four questions well, the paper will feel complete.

---

## 13. Success criteria for the POC

A strong POC should achieve:
- clear hotspot subgraph speedups,
- a principled planner story,
- meaningful reduction in layout-conversion overhead,
- consistent model-level gains,
- a reproducible evaluation pipeline.

You do not need perfect end-to-end automation.
You need a compelling and defensible research prototype.

---

## 14. Final execution principle

The most important rule for this POC is:

> Always protect the core research story.

When there is a choice between:
- adding another system feature,
- or strengthening the clarity of the subgraph-aware layout optimization story,

choose the latter.

That is how the POC becomes a strong SC paper rather than a scattered engineering project.
