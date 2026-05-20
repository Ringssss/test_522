# Errors

Command failures and integration errors.

---

## [ERR-20260428-001] moe_internal_check_routing_signature

**Logged**: 2026-04-28T15:50:20+08:00
**Priority**: medium
**Status**: fixed
**Area**: tests

### Summary
New benchmark-only `moe-internal-check` failed because its diagnostic custom routing function only accepted positional arguments.

### Error
```text
TypeError: apply_routing_for_layer.<locals>.fn() got an unexpected keyword argument 'hidden_states'
```

### Context
- Command: `torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --num-runs 1 --mode moe-internal-check --config-set bspg_source --no-quality`
- vLLM `FusedMoE.select_experts` calls `custom_routing_function` with keyword arguments.
- The production benchmark routing helper used positional compatibility through the existing call path, but the new diagnostic must support keyword calls too.

### Suggested Fix
Define diagnostic routing as `fn(hidden_states=None, gating_output=None, topk=None, renormalize=None, **kwargs)` and use `gating_output`.

### Metadata
- Reproducible: yes
- Related Files: `codex_coding/src/bench_bsp_moe_dp2.py`

---

## [ERR-20260427-001] bsp_moe_d_path_orphan_ranks

**Logged**: 2026-04-27T22:50:00+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Summary
Initial D-path M3 smoke left 7 orphan ranks after rank0 exited, indicating a distributed collective mismatch or process-group issue.

### Error
```
7 Python rank processes remained on GPU1-7 while GPU0/rank0 was gone.
```

### Context
- Command: 8-GPU `bench_bsp_moe_dp2.py --batch-size 32 --gen-length 32 --mode compare --num-runs 1 --no-quality`.
- D path had just introduced hot-update `dist.all_reduce(pop)` on the default group.
- Orphan ranks were killed before rerun.

### Suggested Fix
Use explicit vLLM EP process group for M3 hot-update reductions and add per-rank path-count diagnostics before treating any run as valid.

### Metadata
- Reproducible: unknown
- Related Files: codex_coding/src/bench_bsp_moe_dp2.py

---

## [ERR-20260428-001] bsp_g_component_timing_oom

**Logged**: 2026-04-28T01:57:53+08:00
**Priority**: medium
**Status**: resolved
**Area**: infra

### Summary
First BSP-G C12 component-timing attempt failed in A baseline prefill due to GPU memory occupied by stale 8GPU benchmark processes.

### Error
```
torch.OutOfMemoryError: Tried to allocate 14.39 GiB during modeling_llada2_moe.py logits.float(); GPUs had only about 3-4 GiB free and old benchmark PIDs were reported using about 56-58 GiB per GPU.
```

### Context
- Command: `torchrun --standalone --nproc_per_node=8 codex_coding/src/bench_bsp_moe_dp2.py --batch-size 512 --gen-length 256 --num-runs 1 --mode compare --component-timing --no-quality`.
- Failure happened before BSP-G execution, inside A baseline prefill.
- Subsequent `ps` and `nvidia-smi --query-compute-apps` showed no live benchmark/compute processes after cleanup/release.

### Suggested Fix
Before rerunning full C12 component timing, check `ps -ef | rg 'bench_bsp_moe_dp2|torchrun'` and `nvidia-smi --query-compute-apps=...`; rerun only when GPUs are free.

### Metadata
- Reproducible: no
- Related Files: codex_coding/src/bench_bsp_moe_dp2.py

---

## [ERR-20260428-async-tp-piecewise-static-shape] DInferConpileBackend probe

**Logged**: 2026-04-28T16:58:00+08:00
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
The first `DInferConpileBackend` AsyncTP distributed probe compiled the `aten.mm -> vllm.reduce_scatter` graph but failed on the second compiled call because vLLM `PiecewiseBackend` had no symbolic runtime-shape argument.

### Error
```text
IndexError: list index out of range
vllm/compilation/cuda_piecewise_backend.py: runtime_shape = args[self.sym_shape_indices[0]]
```

### Context
- Command: `VLLM_PATTERN_MATCH_DEBUG=1 torchrun --standalone --nproc_per_node=8 codex_coding/src/probe_dinfer_conpile_backend_async_tp.py --tokens 8192 --hidden 256 ...`
- The dumped FX graph already contained `torch.ops.aten.mm.default` followed by `torch.ops.vllm.reduce_scatter`, so the candidate pattern shape was captured.
- The issue is not NCCL/runtime communication; it is a vLLM piecewise compile contract issue caused by static `dynamic=False/fullgraph=True` inputs.

### Suggested Fix
Use `torch.compile(..., dynamic=True)` and mark input dim 0 dynamic so the subgraph receives a `SymInt` runtime-shape argument; keep `compile_sizes` divisible by TP.

### Metadata
- Reproducible: yes
- Related Files: codex_coding/src/probe_dinfer_conpile_backend_async_tp.py, lib_cite/dInfer/python/dinfer/conpile_backend.py
- Tags: vllm, async-tp, torch-compile, piecewise-backend

---
