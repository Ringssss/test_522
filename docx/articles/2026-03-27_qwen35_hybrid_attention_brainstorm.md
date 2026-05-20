# Qwen3.5 Hybrid Attention Brainstorm

## 1. Purpose

This document records the current brainstorming result for optimizing Qwen3.5-style hybrid-attention models, with Qwen3.5-35B-A3B as the initial target.

It focuses on:

- what this model class looks like,
- what makes hybrid attention different from standard full-attention LLMs,
- what the current literature trend is,
- and where the practical optimization opportunities likely are.

This is a brainstorming archive, not a final technical proposal.

## 2. Target Model Snapshot

Target model path:

- `/home/wuhang/models/Qwen3.5-35B-A3B`

Verified from local model files:

- architecture: `Qwen3_5MoeForConditionalGeneration`
- model type: `qwen3_5_moe`
- total parameters: 35B
- activated parameters: about 3B
- layers: 40
- experts: 256
- active experts per token: `8 routed + 1 shared`
- native context length: 262144

The model card describes the language backbone as:

- `10 × (3 × (Gated DeltaNet -> MoE) + 1 × (Gated Attention -> MoE))`

This means the model is neither:

- a standard dense full-attention decoder,
- nor a pure linear-attention model,

but a hybrid architecture that periodically inserts full-attention layers into a mostly linear-attention stack, while every layer also interacts with a sparse MoE block.

## 3. Why Hybrid Attention Matters

### 3.1 Core design intent

The apparent design goal of this family is:

- keep most layers in a linear or finite-state attention regime for high throughput and better scaling to long context,
- but periodically reintroduce full attention to recover associative recall, exact retrieval, and global token interaction quality.

This reflects a broader architecture trend:

- pure full attention is too expensive for very long context,
- pure linear attention often sacrifices too much quality on difficult retrieval or reasoning patterns,
- hybrid attention is a compromise that tries to keep most of the efficiency while restoring some of the lost expressivity.

### 3.2 What changes compared with standard Transformer optimization

A standard full-attention LLM mostly leads to optimization questions around:

- attention kernel efficiency,
- KV cache memory,
- softmax-attention scheduling,
- prefill and decode batching.

Qwen3.5-style hybrid attention adds a second class of bottleneck:

- recurrent or state-space-like state updates for linear-attention layers,
- state memory traffic beyond ordinary KV cache,
- cross-family kernel switching between linear-attention and full-attention layers,
- and interactions between hybrid attention and MoE routing/fusion.

So the problem becomes heterogeneous, not uniform.

## 4. What the Literature Trend Looks Like

Current adjacent literature and model directions suggest the following pattern.

### 4.1 The field is moving away from “all layers full attention”

Recent architectures are increasingly exploring:

- linear attention,
- delta-rule variants,
- recurrent state updates,
- latent or compressed attention,
- hybrid stacking strategies.

The motivation is consistent:

- reduce long-context cost,
- reduce KV memory pressure,
- improve decode efficiency,
- and avoid the quadratic burden of applying full attention everywhere.

### 4.2 Pure linear attention is usually not enough

A recurring lesson in the literature is that pure linear attention often has weaknesses in:

- exact retrieval,
- long-range token association,
- induction-like behavior,
- and some hard reasoning patterns.

This is why hybrid designs are becoming more compelling:

- use cheaper linear-style layers most of the time,
- retain some periodic full-attention layers as capability anchors.

### 4.3 The system literature is lagging behind the model trend

Serving stacks and kernel ecosystems are still much more mature for:

- dense full-attention models,
- standard GQA/MQA models,
- and conventional MoE serving.

By contrast, hybrid linear-attention models still feel relatively under-optimized in systems practice.

That creates an opening:

- the model architecture already exists,
- but the serving and kernel stack is not yet fully standardized or fully tuned.

## 5. Practical Runtime Characteristics Observed Locally

The local online serving baseline was brought up with SGLang on two H100 80GB GPUs.

Runtime characteristics that matter for later optimization:

- full attention backend: `fa3`
- linear attention backend: `triton`
- MoE runner backend: `auto`
- overlap schedule was disabled because the current mamba-style scheduler path is not compatible with overlap scheduling in this configuration

Important memory observations from the running server:

- weight memory per rank is large
- KV cache is significant but not the only large state object
- there is also a large recurrent/state-space-style cache for the linear-attention path

This already tells us something important:

- hybrid attention does not simply “replace KV cache cost with nothing”
- it often shifts part of the system pressure from KV cache into other persistent states

So the optimization target is not just:

- less KV,

but:

- a different state-management problem.

## 6. What Makes This Model Interesting for Research

Qwen3.5-35B-A3B is especially interesting because it combines four dimensions at once:

1. hybrid attention
2. sparse MoE
3. shared expert mechanism
4. long-context support

This means the main bottleneck is unlikely to be explained by a single kernel.

Instead, the dominant cost may come from the interaction among:

- linear-attention state update kernels,
- periodic full-attention layers,
- routed/shared expert execution,
- memory layout or state movement between these stages,
- scheduler policy and graph capture policy.

That is exactly the kind of setting where research-oriented systems work can matter.

## 7. Optimization Space

Below is the current brainstorming list of plausible optimization directions.

### 7.1 Linear-attention kernel path

Potential work:

- optimize Gated DeltaNet kernels directly,
- fuse gate, state update, and output projection more aggressively,
- reduce state read/write traffic,
- separate prefill-path optimization from decode-path optimization.

Why it matters:

- most layers in the model are linear-attention layers,
- so even moderate improvements there can dominate end-to-end gain.

### 7.2 Periodic full-attention layers

Potential work:

- profile full-attention layers separately from linear-attention layers,
- identify whether periodic full-attention layers become latency spikes,
- tune these layers as “sparse expensive checkpoints” in an otherwise cheaper stack.

Why it matters:

- even if only one in four layers uses full attention, those layers may still set the latency floor for some workloads.

### 7.3 Cross-family transition overhead

Potential work:

- study the cost of switching between linear-attention and full-attention execution regimes,
- study whether state/layout changes create hidden barriers,
- study whether graph capture policy and scheduler policy are hybrid-architecture-aware enough.

Why it matters:

- in a heterogeneous stack, transition overhead can be as important as per-kernel speed.

### 7.4 Long-context serving behavior

Potential work:

- isolate prefill behavior under long prompts,
- separate recurrent-state pressure from full-attention residual pressure,
- test chunked prefill and cache policy against hybrid attention specifically.

Why it matters:

- the advertised advantage of the model includes long context,
- so the long-context regime should be treated as a first-class optimization target, not as an afterthought.

### 7.5 Hybrid attention plus MoE coupling

Potential work:

- analyze whether the output layout/state of linear-attention layers is especially friendly or unfriendly to the subsequent MoE block,
- examine whether routed expert and shared expert paths can be fused or overlapped better,
- profile whether MoE dominates certain layers while attention dominates others.

Why it matters:

- the real hotspot may be the pipeline formed by `attention -> MoE`, not either block in isolation.

### 7.6 MoE kernel and configuration tuning

Potential work:

- tune fused MoE kernel configuration for this exact `(E=256, N=256, H100)` shape regime,
- compare backend choices for the MoE path,
- inspect whether TP-only is leaving performance on the table compared with EP-related configurations.

Why it matters:

- the local SGLang runtime explicitly reported missing Triton MoE kernel configs for this model shape on H100,
- which strongly suggests the current baseline is not yet fully tuned.

### 7.7 Scheduling and graph-capture policy

Potential work:

- study whether CUDA graph capture policy should differ between linear-attention-heavy and full-attention-heavy phases,
- study whether overlap scheduling can be recovered or replaced for this model family,
- examine request batching policy under hybrid attention.

Why it matters:

- the local startup spent a long time on CUDA graph capture,
- and the current runtime disabled overlap scheduling for compatibility reasons.

### 7.8 Speculative decoding or MTP-related opportunities

Potential work:

- investigate whether the model's MTP-related training signals can support speculative decoding efficiently,
- evaluate whether hybrid-attention models benefit differently from speculative pipelines compared with standard full-attention models.

Why it matters:

- if decode remains expensive after kernel tuning, speculative approaches may become the next large lever.

## 8. What Looks Most Promising Right Now

At this stage, the strongest short-list is:

1. tune the MoE execution path for this exact model and hardware,
2. separate linear-attention and periodic full-attention profiling,
3. analyze cross-family transition overhead and scheduler constraints,
4. then decide whether the best paper angle is:
   - hybrid-attention kernel/system optimization,
   - hybrid-attention plus MoE pipeline optimization,
   - or long-context hybrid serving optimization.

## 9. Working Hypotheses

These are hypotheses, not yet final conclusions.

### Hypothesis 1

For Qwen3.5-style models, the dominant serving bottleneck is not a single full-attention kernel but a heterogeneous interaction among:

- linear-attention state updates,
- periodic full-attention layers,
- and MoE execution.

### Hypothesis 2

The most practical early win may come from:

- MoE backend/config tuning,
- because the current runtime already signals missing optimized configs.

### Hypothesis 3

A strong research contribution may come from treating the model as a heterogeneous layer system rather than trying to optimize all attention layers with one uniform strategy.

## 10. Caveat About Current Persistence

The earlier online benchmark outputs were observed and summarized during the live run, but they were not yet saved into:

- `/home/wuhang/wuhang/linear_wh/codex_coding/src`
- `/home/wuhang/wuhang/linear_wh/codex_coding/results`

This should be corrected in later runs so that:

- benchmark scripts are versioned under `codex_coding/src`,
- raw outputs, request traces, and result summaries are stored under `codex_coding/results`.

## 11. Suggested Next Experimental Step

The next technically meaningful step should be:

- create a reproducible benchmark harness under `codex_coding/src`,
- persist the online benchmark outputs under `codex_coding/results`,
- and profile the model by separating:
  - hybrid linear-attention layers,
  - periodic full-attention layers,
  - and MoE-heavy sections.

## 12. Reference Pointers

Local references:

- `/home/wuhang/models/Qwen3.5-35B-A3B/config.json`
- `/home/wuhang/models/Qwen3.5-35B-A3B/README.md`
- `/home/wuhang/wuhang/linear_wh/sglang/docs/basic_usage/qwen3_5.md`
- `/home/wuhang/wuhang/linear_wh/sglang/docs/developer_guide/benchmark_and_profiling.md`
- `/home/wuhang/wuhang/linear_wh/sglang/docs/developer_guide/bench_serving.md`

Online references:

- `https://huggingface.co/Qwen/Qwen3.5-35B-A3B`
- `https://arxiv.org/abs/2412.06464`
- `https://arxiv.org/abs/2510.26692`
- `https://arxiv.org/abs/2404.19737`
