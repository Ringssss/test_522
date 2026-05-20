# Core Theory and Technical Ideas: Subgraph-Level Layout IR

## 1. Core Thesis

The key theoretical idea is to treat layout as a **first-class optimization object** not only within a single kernel, but across an entire hotspot subgraph.

The central shift is:

- from **per-op layout optimization**
- to **subgraph-level layout assignment and transformation optimization**

This is not a cosmetic extension.
It changes the optimization unit from a single operator to a graph fragment.

---

## 2. Relationship to Triton LinearLayout

Triton's `LinearLayout` provides the right conceptual foundation:
it treats layout as a structured object that can be composed, inverted, and reasoned about.

At a high level, Triton's layout view can be interpreted as:
- a mapping from GPU execution-space coordinates
  (register, lane, warp, block)
- to logical tensor coordinates.

The important intellectual inheritance for our work is **not** that we must literally reuse the exact Triton implementation.
The important inheritance is this:

> Layout should be represented algebraically, not as an ad hoc tag.

In our work, we extend that philosophy to hotspot subgraphs.

---

## 3. Theoretical Stack

The theory can be organized into three layers:

### Layer 1: layout algebra
A mathematical representation for a single layout object.

### Layer 2: graph-level propagation and compatibility
A theory for how layouts flow across operators in a subgraph.

### Layer 3: optimization over the subgraph
A decision procedure for selecting layouts and transforms that minimize overall cost.

This three-layer view is sufficient for both a strong POC and a paper-quality formulation.

---

## 4. Layer 1: layout algebra

### 4.1 Why GF(2)-style linear layout is a natural starting point

For many GPU-relevant layouts, the interesting structure lives at the bit level:
- swizzles,
- bit permutations,
- fragment mappings,
- tensor-core-friendly ownership patterns,
- lane/register remappings.

These structures are naturally modeled using xor-style linearity over bit vectors.

We can write a layout as:

\[
y = A x \quad \text{over } GF(2)
\]

where:
- `x` is a bit-vector encoding execution-space coordinates,
- `y` is a bit-vector encoding logical or tile-local coordinates,
- `A` is a binary matrix describing the layout.

This is the mathematical spirit behind Triton-style linear layouts.

### 4.2 What this layout algebra provides

A layout algebra should support:

- **composition**
  \[
  L_2 \circ L_1
  \]

- **inverse or pseudo-inverse reasoning**
  when exact reversibility is unavailable but a useful transform still exists

- **equivalence / compatibility testing**
  whether two layouts are effectively identical for a consumer

- **partial ordering**
  whether one layout is “good enough” for another stage without conversion

These operations are what allow layouts to become manipulable optimization objects.

### 4.3 Why ordinary stride-only views are insufficient

A traditional affine/stride-only representation is too weak for many GPU layouts because it does not naturally express:
- xor swizzles,
- tensor-core fragment packing,
- bit-level mixed ownership patterns,
- some bank-conflict avoidance layouts.

A linear-layout-style representation is much better aligned with GPU execution reality.

---

## 5. Layer 2: graph-level layout propagation

### 5.1 Layout becomes an edge state

Consider a hotspot subgraph:

\[
G = (V, E)
\]

Each edge `e \in E` represents an intermediate tensor.
Instead of attaching only shape and dtype, we attach a layout variable:

\[
L_e \in \mathcal{L}
\]

where `\mathcal{L}` is the set of candidate layouts.

This is the critical subgraph-level extension.

### 5.2 Operators define legality constraints

Each operator `v \in V` imposes constraints on its input/output layouts.

For example:

\[
(L_{in_1}, \ldots, L_{in_m}, L_{out_1}, \ldots, L_{out_n}) \in \mathcal{F}_v
\]

where `\mathcal{F}_v` is the feasible layout relation for operator `v`.

This relation can encode:
- exact required layouts,
- acceptable families of layouts,
- layout-dependent fusion legality,
- layout-dependent kernel availability.

### 5.3 Propagation as a compatibility problem

If producer output layout matches consumer input needs, no conversion is required.

We define a compatibility relation:

\[
\mathrm{Compat}(L_u^{out}, L_v^{in})
\]

If true, conversion cost is zero.
If false, a transform is required.

This gives layout propagation a formal meaning:
the optimizer seeks edge layouts that reduce incompatibility across the subgraph.

---

## 6. Layer 3: subgraph optimization problem

### 6.1 Decision variables

The optimizer chooses:
- edge layouts `L_e`,
- operator implementation parameters `\theta_v`,
- optional layout transforms between edges,
- fusion decisions where applicable.

### 6.2 Objective

We define a total cost:

\[
\min_{\{L_e\}, \{\theta_v\}}
\sum_{v \in V} C_v(L_{ins}, L_{outs}, \theta_v)
+ \sum_{e \in E} C_e(L_u^{out} \to L_v^{in})
\]

where:
- `C_v` is the local operator/kernel cost,
- `C_e` is the layout conversion cost between adjacent operators.

This formulation already captures the essence of the problem.

### 6.3 Cost decomposition

A practical cost function can include:

- global memory traffic estimate,
- shared-memory access penalty,
- tensor-core compatibility or underutilization,
- register pressure / occupancy approximation,
- conversion kernel cost,
- writeback elimination benefit,
- fusion gain.

In practice, the cost function can be a weighted sum:

\[
\mathrm{Cost}
=
\alpha M
+ \beta S
+ \gamma R
+ \delta T
+ \eta X
- \lambda F
\]

where:
- `M` = memory transaction cost,
- `S` = shared-memory conflict penalty,
- `R` = register/occupancy penalty,
- `T` = tensor-core inefficiency,
- `X` = explicit conversion overhead,
- `F` = fusion benefit.

---

## 7. What makes this different from per-op autotuning

Per-op autotuning solves:

\[
\min_{\theta_v} C_v
\]

for each operator independently.

Our formulation instead solves a coupled problem over the subgraph:

\[
\min_{\{L_e\}, \{\theta_v\}} \mathrm{TotalSubgraphCost}
\]

This difference is profound because:
- operator choices are no longer independent,
- layout assignment couples neighboring operators,
- locally suboptimal choices may be globally optimal,
- conversion elimination becomes part of the optimization objective.

This is the main theoretical distinction of the work.

---

## 8. Recommended formal abstraction

For the paper and the POC, the cleanest abstraction is:

### 8.1 Layout domain
`\mathcal{L}` = candidate layout space

This may include:
- blocked layouts,
- swizzled layouts,
- MMA-aligned layouts,
- shared-friendly staging layouts,
- quantization-pack-friendly layouts.

### 8.2 Feasibility relation
For each operator:
- legal input/output layout pairs,
- legal fusion combinations,
- legal transform absorption opportunities.

### 8.3 Transform algebra
Layout transforms are objects, not opaque implementation steps.

A transform `T` should support:
- composition,
- simplification,
- cancellation when adjacent transforms compose to identity,
- cost estimation.

### 8.4 Search procedure
The optimizer explores layout assignments over the subgraph using:
- dynamic programming,
- beam search,
- branch-and-bound,
- or a small-state shortest-path formulation.

For the POC, a small-state search method is preferable to heavy ILP.

---

## 9. Practical legality system

A strong theory needs a practical legality model.

For each subgraph type, define:
- a set of legal layouts,
- consumer acceptance conditions,
- fusion preconditions,
- transform absorption rules.

For example:
- a QKV preparation kernel may accept several blocked/swizzled variants,
- a GEMM epilogue may only fuse when the accumulator/output layout satisfies a constraint,
- a quantized FFN path may strongly prefer a layout that matches packed-weight access.

This legality layer acts like a layout type system for the subgraph.

---

## 10. Search strategies that make sense

### 10.1 Dynamic programming over small subgraphs
This is attractive when:
- subgraphs are short chains,
- candidate layout count is small,
- cost is locally decomposable.

### 10.2 Beam search
Useful when:
- candidate space grows,
- exact search is still too expensive,
- approximate but high-quality plans are enough.

### 10.3 Shortest-path interpretation
If states are edge layouts and transitions are operator applications / conversions, some subgraph problems can be written as shortest-path problems in a layered state graph.

This is elegant and paper-friendly.

### 10.4 Why not heavy global solvers first
For a systems paper POC, using heavy ILP or SMT too early is unnecessary.
The method should look principled, but still practical and reproducible.

---

## 11. The minimum mathematics needed for the paper

If the project needs to stay focused, the minimum theory should be:

1. **layout algebra**
   based on a Triton-inspired linear layout view

2. **graph-level layout assignment**
   each edge carries a layout variable

3. **legality constraints**
   each operator constrains valid input/output layouts

4. **costed transform reasoning**
   conversions are explicit, composable, and avoidable

5. **subgraph objective**
   minimize total operator + conversion cost

This is enough to build a convincing paper framework.

---

## 12. The central innovation, stated clearly

The work is not merely about having another layout representation.

The true innovation is:

> We elevate layout from a kernel-local implementation detail to a subgraph-level optimization variable, and solve layout assignment jointly with transform elimination and kernel-family selection.

That should be the conceptual anchor of the entire project.

---

## 13. Theoretical contribution statement

A useful theory-oriented contribution statement is:

> We formulate hotspot subgraph optimization as a constrained layout assignment problem over a graph whose edge states are Triton-inspired linear layout objects. By combining layout algebra, compatibility-aware propagation, and costed transform elimination, the system jointly optimizes subgraph layouts and operator implementations to reduce memory overhead and improve end-to-end inference efficiency.

This statement is compact, precise, and directly reusable in a paper draft.

---

## 14. What not to overcomplicate

For the POC and the paper, avoid overloading the theory with:
- full polyhedral compilation,
- full abstract interpretation machinery,
- general-purpose theorem-proving-style legality reasoning,
- full global graph compiler semantics.

The strongest version of the project remains:
- mathematically principled,
- graph-aware,
- layout-centric,
- and engineering-grounded.
