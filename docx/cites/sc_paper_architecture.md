# SC Paper Architecture: Subgraph-Aware Layout Optimization for LLM Inference

## 1. Positioning

This paper targets a focused but high-impact problem:

**How can we elevate layout optimization from single operators to hotspot subgraphs in LLM inference, so that layout selection, propagation, conversion elimination, and kernel-family choice are optimized jointly rather than per-op?**

The key insight is that modern LLM inference performance is often constrained not only by the efficiency of individual kernels, but by the interaction between neighboring operators:
- mismatched layouts between producer and consumer,
- unnecessary intermediate layout conversions,
- missed fusion opportunities,
- inefficient memory hierarchy usage across subgraph boundaries.

The paper proposes a **subgraph-aware layout optimization framework** built on a unified layout representation inspired by Triton's `LinearLayout` philosophy:
layouts are treated as **first-class optimization objects**, not as incidental implementation details.

---

## 2. Background

### 2.1 Why layout matters in LLM inference

In modern LLM inference stacks, especially for:
- quantized FFN blocks,
- MoE expert FFNs,
- attention preparation and epilogue chains,
- long-context and mixed backend serving,

performance is strongly affected by how tensors are laid out across:
- logical tensor dimensions,
- thread blocks / warps / lanes,
- registers,
- shared memory staging,
- tensor core fragments.

Traditional per-op optimization usually chooses the best implementation for each operator independently. This often misses system-level opportunities:
- a locally optimal output layout for one operator may be a poor input layout for the next;
- conversion kernels may dominate short subgraphs;
- fusion legality often depends on layout compatibility;
- memory traffic can increase due to layout mismatch.

### 2.2 Why existing systems are not enough

Existing high-performance inference systems and kernel libraries are very strong, but they usually focus on:
- operator-level kernel engineering,
- backend-specific specialization,
- serving/runtime optimizations,
- hand-crafted or auto-tuned operator implementations.

What is still underexplored is the following:

> **Layout should be optimized at the subgraph level, not only at the operator level.**

This paper fills that gap.

---

## 3. Motivation

### 3.1 The gap between per-op optimality and subgraph optimality

Suppose a subgraph contains:
- dequant -> matmul -> activation -> matmul
- norm -> qkv projection -> rope/reshape
- dispatch transform -> expert FFN -> combine

A per-op optimizer may choose the best layout for each operator in isolation.
However, the globally best design may instead:
- keep an output layout that is only slightly suboptimal for the producer,
- because it avoids a conversion before the consumer,
- enables epilogue fusion,
- reduces shared-memory bank conflicts in the next stage,
- and improves tensor-core feed efficiency across the whole subgraph.

This is the central motivation of the paper.

### 3.2 Why Triton-style layout abstraction is the right starting point

Triton's `LinearLayout` shows that GPU layouts can be represented as a unified, composable object rather than a large collection of ad hoc special cases.

That idea is extremely valuable, but it is mostly used at the single-op / compiler-internal level.

This work extends that philosophy to the subgraph level:
- layouts become graph-level optimization variables,
- conversions become first-class transform objects,
- layout propagation becomes a subgraph optimization problem,
- kernel-family selection is jointly decided with layout assignment.

---

## 4. Core Technical Contributions

### 4.1 A Subgraph-Level Layout IR

We define a **Subgraph Layout IR** where:
- each edge in a hotspot subgraph carries a candidate layout,
- each operator exposes a legal set of input/output layout combinations,
- layout transforms are explicit and composable,
- fusion opportunities are layout-dependent,
- the optimizer searches for a graph-wide low-cost layout assignment.

This is the conceptual center of the paper.

### 4.2 Layout propagation and conversion elimination

Instead of optimizing each operator independently, the system propagates layout preferences across the subgraph.
It can:
- keep producer output layouts that are directly consumable,
- eliminate redundant layout conversions,
- absorb some transforms into neighboring kernels,
- choose compromise layouts that reduce total subgraph cost.

### 4.3 Joint optimization of layout and kernel family

The optimizer does not only select layouts.
It jointly chooses:
- subgraph edge layouts,
- kernel families,
- tile-related implementation choices,
- whether fusion is enabled.

This transforms layout from a local codegen concern into a graph-level optimization variable.

### 4.4 A practical cost model for hotspot subgraphs

The system uses a lightweight cost model combining:
- estimated memory transaction cost,
- shared-memory conflict penalty,
- tensor-core compatibility,
- register/occupancy approximations,
- layout-conversion overhead,
- fusion benefits.

The model is intentionally practical rather than over-formalized:
it is designed to guide optimization decisions in a real serving system.

---

## 5. Scope of the Paper

This paper is **not** a full end-to-end serving runtime redesign.

It focuses on:
- hotspot subgraph layout optimization,
- kernel-level implementation choices inside those subgraphs,
- integration into a real inference stack (SGLang),
- model-level and serving-level evaluation.

It does **not** primarily target:
- speculative decoding,
- scheduler redesign,
- distributed MoE communication redesign,
- KV paging policy innovation,
- full graph compiler replacement.

This scope is deliberate: it keeps the paper focused, technically deep, and experimentally clean.

---

## 6. Target Hotspot Subgraphs

The paper should focus on **two or three** hotspot classes only.

### 6.1 Quantized FFN / expert FFN subgraphs
Example patterns:
- dequant -> matmul -> bias/residual -> activation -> matmul
- expert-local FFN chains in MoE

Why they matter:
- high arithmetic density,
- strong layout sensitivity,
- important for both dense and MoE models,
- ideal for kernel-family comparison and layout propagation.

### 6.2 Attention preparation or epilogue subgraphs
Example patterns:
- norm -> qkv projection -> rope / reshape / transpose
- attention output -> projection epilogue

Why they matter:
- layout mismatches are common,
- they expose conversion elimination opportunities,
- they connect well to serving backends.

### 6.3 Optional: local MoE transform subgraphs
Example patterns:
- dispatch buffer transform -> expert FFN -> combine

This should remain local and single-node / single-device for the paper POC.
Do not expand into distributed communication optimization.

---

## 7. System Design for the Paper

### 7.1 Overall pipeline
1. Identify hotspot subgraphs in the inference graph.
2. Construct Subgraph Layout IR for each hotspot.
3. Generate legal layout candidates.
4. Evaluate cost of layout assignments and kernel-family choices.
5. Select a low-cost plan.
6. Execute specialized Triton kernels through SGLang integration.

### 7.2 Why SGLang is a good host platform
SGLang already provides:
- mature LLM serving infrastructure,
- multiple attention backends,
- Triton-based kernel paths,
- MoE-related optimization hooks,
- benchmark and profiling tooling.

This allows the paper to evaluate a real system integration without having to build a serving engine from scratch.

---

## 8. Experimental Setup

### 8.1 Platforms
Use one primary GPU platform only:
- H100 preferred, or
- A100 as a fallback.

Do not overextend to multi-vendor support for the paper POC.

### 8.2 Models
Choose a compact but representative set:
- one dense baseline model,
- one MoE baseline model,
- one Qwen3.5-class hybrid/MoE model.

The goal is to show that the method generalizes across model families.

### 8.3 Baselines
At minimum compare against:
- stock SGLang path,
- stock Triton-based backend where applicable,
- a strong production backend when available,
- hand-tuned per-op kernels for selected subgraphs.

### 8.4 Metrics
Report at three levels:

#### Kernel/subgraph level
- latency,
- achieved throughput,
- memory efficiency indicators,
- conversion count reduction.

#### Model/offline engine level
- offline throughput,
- layer/subgraph time breakdown,
- percentage of time covered by optimized hotspots.

#### Serving level
- TTFT,
- TPOT,
- ITL,
- steady-state throughput.

### 8.5 Ablations
Ablations are mandatory:
- no planner, only baseline kernels,
- planner without conversion elimination,
- planner + conversion elimination,
- planner + conversion elimination + fusion,
- layout search variants,
- cost model variants.

---

## 9. Expected Outcomes

Reasonable and credible targets are:

### 9.1 Hotspot subgraph speedup
- average: **1.15x - 1.30x**
- strong cases: potentially higher on layout-sensitive chains

### 9.2 Layer/local pipeline improvement
- **10% - 25%** time reduction on affected layer segments

### 9.3 End-to-end model inference impact
- realistic target: **8% - 18%**
- stronger if hotspot coverage is high and baseline paths are layout-mismatch heavy

The paper should avoid overclaiming extreme end-to-end speedups.
The more persuasive story is:
- consistent hotspot improvements,
- robust gains across models/shapes,
- and measurable serving benefits.

---

## 10. What the Paper Should Emphasize

The paper should repeatedly emphasize the following qualitative message:

> Layout is not merely a kernel-internal implementation detail.
> For LLM inference, layout is a subgraph-level optimization variable whose global coordination can unlock meaningful performance gains.

This is the core intellectual contribution.

---

## 11. What the Paper Should Avoid

To keep the paper strong and focused, avoid these pitfalls:
- turning the work into a general serving-runtime redesign,
- claiming full end-to-end automation for all kernels and all models,
- focusing only on one narrow operator chain,
- presenting only raw autotuning without a unifying optimization formulation,
- relying on a weak baseline.

---

## 12. Suggested Contribution Statement

A strong one-paragraph contribution statement for the paper could be:

> We present a subgraph-aware layout optimization framework for LLM inference that elevates layout from an operator-local implementation choice to a graph-level optimization variable. Inspired by the unified layout abstraction of Triton-style linear layout systems, our method jointly optimizes edge layouts, layout transforms, fusion opportunities, and kernel-family selection over hotspot inference subgraphs. Integrated into SGLang, the framework improves memory hierarchy utilization, eliminates redundant layout conversions, and delivers consistent speedups across quantized FFN, MoE expert, and attention-related subgraphs, translating into measurable model-level and serving-level gains.

---

## 13. Final Qualitative Positioning

If the work is executed well, the result can be characterized as:

- **not** “just another faster kernel,”
- **not** “just a serving trick,”
- **but** a **subgraph-level layout optimization framework** that opens a new space between operator autotuning and full graph compiler design.

That is exactly the positioning that can make the work compelling for an SC-style systems paper.
