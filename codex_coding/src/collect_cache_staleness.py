#!/usr/bin/env python3
"""
Collect MoE output cache staleness data for paper figure.

Measures how decoded token MoE outputs drift from their "first cached" value
as more forwards pass within a block.

For each decoded position p:
  - Record t_decode (iteration when p first becomes decoded)
  - Save MoE output at t_decode as cache baseline
  - At each subsequent step t, compute cos_sim(baseline[p], output_t[p])
  - Record (gap = t - t_decode, cos_sim)

Output: codex_coding/results/plt/cache_staleness_data.json

Launch:
    cd /home/wuhang/wuhang/dllm_wh && \\
    torchrun --nproc_per_node=8 codex_coding/src/collect_cache_staleness.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID = 156895
EOS_ID = 156892
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
BLOCK_LENGTH = 32
BATCH_SIZE = 16
GEN_LENGTH = 256
TARGET_LAYERS = [4, 10, 18]

DATASETS = {
    "GSM8K": REPO_ROOT / "data" / "gsm8k.jsonl",
    "HumanEval": REPO_ROOT / "data" / "humaneval.jsonl",
    "MT-Bench": REPO_ROOT / "data" / "mt_bench.jsonl",
}


def load_prompts(dataset_name: str, path: Path, n: int) -> list[str]:
    lines = path.read_text().strip().split("\n")[:n]
    prompts = []
    for line in lines:
        obj = json.loads(line)
        if dataset_name == "GSM8K":
            prompts.append(obj["question"])
        elif dataset_name == "HumanEval":
            prompts.append(obj["prompt"])
        elif dataset_name == "MT-Bench":
            prompts.append(obj["turns"][0])
    return prompts


# =====================================================================
# StalenessCollector
# =====================================================================

class StalenessCollector:
    """
    Tracks per-position cache baseline and computes staleness vs gap.

    Baseline is captured at t_decode+1 (one step AFTER first decode),
    not at t_decode itself, to skip the initial transition instability.

    For each target layer, maintains:
      - baseline[lid][pos] = MoE output tensor at t_decode+1
      - t_baseline[lid][pos] = step when baseline was captured (-1 = not yet)
      - pending[lid] = bool mask of positions awaiting baseline capture
    """

    def __init__(self, target_layers: list[int]):
        self.target_layers = set(target_layers)
        self._current_block_id = -1
        self._step = 0
        self._prev_decoded: torch.Tensor | None = None
        self._curr_decoded: torch.Tensor | None = None

        self._baseline: dict[int, torch.Tensor | None] = {
            lid: None for lid in target_layers
        }
        self._t_baseline: dict[int, torch.Tensor | None] = {
            lid: None for lid in target_layers
        }
        self._pending: dict[int, torch.Tensor | None] = {
            lid: None for lid in target_layers
        }

        self.results: dict[int, dict[int, list[float]]] = {
            lid: defaultdict(list) for lid in target_layers
        }
        # Step-level results: step -> list of cos_sim (pooled across layers)
        self.step_results: dict[int, list[float]] = defaultdict(list)

    def reset(self):
        self._current_block_id = -1
        self._step = 0
        self._prev_decoded = None
        self._curr_decoded = None
        for lid in self.target_layers:
            self._baseline[lid] = None
            self._t_baseline[lid] = None
            self._pending[lid] = None
            self.results[lid] = defaultdict(list)
        self.step_results = defaultdict(list)

    def on_block_init(self, block_id: int):
        self._current_block_id = int(block_id)
        self._step = 0
        self._prev_decoded = None
        self._curr_decoded = None
        for lid in self.target_layers:
            self._baseline[lid] = None
            self._t_baseline[lid] = None
            self._pending[lid] = None

    def on_forward_start(self, input_ids: torch.Tensor | None):
        if input_ids is None or self._current_block_id < 0:
            return
        is_mask = (input_ids == MASK_ID).view(-1)
        self._curr_decoded = ~is_mask
        self._step += 1

    def on_forward_end(self):
        if self._current_block_id < 0:
            return
        self._prev_decoded = self._curr_decoded

    def record_layer_output(self, layer_id: int, output: torch.Tensor):
        if layer_id not in self.target_layers:
            return
        if self._current_block_id < 0 or self._curr_decoded is None:
            return

        N = output.shape[0]
        curr_dec = self._curr_decoded
        if curr_dec.shape[0] != N:
            return

        step = self._step
        lid = layer_id

        # Initialize on first valid call for this block
        if self._baseline[lid] is None:
            self._baseline[lid] = torch.zeros_like(output)
            self._t_baseline[lid] = torch.full((N,), -1, dtype=torch.long,
                                               device=output.device)
            self._pending[lid] = torch.zeros(N, dtype=torch.bool,
                                             device=output.device)

        baseline = self._baseline[lid]
        t_baseline = self._t_baseline[lid]
        pending = self._pending[lid]

        if baseline.shape[0] != N:
            return

        # Step A: Capture baseline for positions that were pending (decoded last step)
        capture = pending & curr_dec
        if capture.any():
            baseline[capture] = output[capture].detach()
            t_baseline[capture] = step
            pending[capture] = False

        # Step B: Mark newly decoded positions as pending (baseline captured next step)
        if self._prev_decoded is not None and self._prev_decoded.shape[0] == N:
            newly_decoded = curr_dec & ~self._prev_decoded
        else:
            newly_decoded = curr_dec
        no_baseline_yet = t_baseline < 0
        pending |= (newly_decoded & no_baseline_yet)

        # Step C: Compute staleness for positions that have baseline
        has_baseline = t_baseline >= 0
        active = has_baseline & curr_dec
        if not active.any():
            return

        active_baseline = baseline[active]
        active_output = output[active]
        active_t_baseline = t_baseline[active]

        gaps = step - active_t_baseline
        sims = F.cosine_similarity(active_baseline, active_output, dim=-1)

        # Record step-level (all active decoded tokens at this step, pooled across layers)
        self.step_results[step].extend(sims.tolist())

        # Group by gap
        unique_gaps = gaps.unique()
        for g in unique_gaps:
            g_val = int(g.item())
            if g_val < 1:
                continue
            mask_g = gaps == g
            sims_g = sims[mask_g]
            self.results[lid][g_val].extend(sims_g.tolist())


# =====================================================================
# Hook installation (same pattern as collect_moe_output_stability.py)
# =====================================================================

def install_output_hooks(model, collector: StalenessCollector):
    import dinfer.model.modeling_llada2_moe as moe_module

    experts_to_lid = {}
    layer_id = 0
    for _name, mod in model.named_modules():
        if mod.__class__.__name__ != "LLaDA2MoeSparseMoeBlock":
            continue
        experts_to_lid[id(mod.experts)] = layer_id
        layer_id += 1

    orig_moe_fwd_ctx = moe_module._moe_forward_with_context

    def hooked_moe_fwd_ctx(experts, hidden_states_flat, router_logits):
        result = orig_moe_fwd_ctx(experts, hidden_states_flat, router_logits)
        lid = experts_to_lid.get(id(experts))
        if lid is not None:
            output = result[1] if isinstance(result, tuple) else result
            collector.record_layer_output(lid, output)
        return result

    moe_module._moe_forward_with_context = hooked_moe_fwd_ctx
    return {"module": moe_module, "orig_fn": orig_moe_fwd_ctx}


def remove_output_hooks(installed):
    installed["module"]._moe_forward_with_context = installed["orig_fn"]


def install_lifecycle_hooks(dllm, collector: StalenessCollector):
    decoder = dllm.decoder
    model = dllm.model
    orig_block_init = decoder.block_init
    orig_model_forward = model.forward

    def block_init_wrapper(block_x, block_id):
        collector.on_block_init(int(block_id))
        return orig_block_init(block_x, block_id)

    decoder.block_init = block_init_wrapper

    def model_forward_wrapper(input_ids=None, *args, **kwargs):
        collector.on_forward_start(input_ids)
        result = orig_model_forward(input_ids, *args, **kwargs)
        collector.on_forward_end()
        return result

    model.forward = model_forward_wrapper
    return {
        "orig_block_init": orig_block_init,
        "orig_model_forward": orig_model_forward,
        "decoder": decoder,
        "model": model,
    }


def remove_lifecycle_hooks(originals):
    originals["decoder"].block_init = originals["orig_block_init"]
    originals["model"].forward = originals["orig_model_forward"]


# =====================================================================
# Main
# =====================================================================

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if world_size == 8:
        tp_size = 4
    elif world_size == 4:
        tp_size = 2
    else:
        raise AssertionError(f"Requires 8 or 4 GPUs, got {world_size}")

    dp_size = world_size // tp_size
    dp_rank = rank // tp_size
    local_bs = BATCH_SIZE // dp_size
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    pcfg_init = ParallelConfig(
        tensor_parallel_size=1, data_parallel_size=1,
        enable_expert_parallel=True,
    )
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(
            world_size, rank, "env://", local_rank, "nccl"
        )

    pcfg = ParallelConfig(
        tensor_parallel_size=tp_size, data_parallel_size=dp_size,
        data_parallel_rank=dp_rank, enable_expert_parallel=True,
    )
    vllm_cfg = VllmConfig(parallel_config=pcfg)

    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, backend="nccl",
        )

        from vllm.distributed import prepare_communication_buffer_for_model
        from vllm.forward_context import set_forward_context
        from dinfer import (
            BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
            ThresholdParallelDecoder,
        )
        from dinfer.model import LLaDA2MoeModelLM
        from transformers import AutoConfig, AutoTokenizer
        from baseline_optimizations import apply_all_optimizations

        if rank == 0:
            print("=" * 70)
            print("Cache Staleness Decay Collection (K=8 native, no EB)")
            print(f"  batch={BATCH_SIZE} gen={GEN_LENGTH} block={BLOCK_LENGTH}")
            print(f"  tp={tp_size} dp={dp_size} ep={world_size}")
            print(f"  target_layers={TARGET_LAYERS}")
            print(f"  datasets: {list(DATASETS.keys())}")
            print("=" * 70)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True)

        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        with torch.inference_mode():
            warmup_tok = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(
                attn_metadata=None, vllm_config=vllm_cfg,
                num_tokens=warmup_tok.numel(),
            ):
                _ = model(warmup_tok, use_cache=False)

        apply_all_optimizations(model)
        prepare_communication_buffer_for_model(model)

        pad_id = tokenizer.pad_token_id or 0

        decoder = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90,
            mask_id=MASK_ID, eos_id=EOS_ID,
        )

        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend="vllm", lazy_cache_update=True, inplace_cache_update=True,
            )

        # --- warmup ---
        if rank == 0:
            print("  Warmup run...")
        dummy_prompts = ["Hello, world!"] * BATCH_SIZE
        dummy_ids = []
        for text in dummy_prompts:
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=False,
                )
            dummy_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in dummy_ids)
        padded = [
            torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
            if ids.shape[0] < mx else ids
            for ids in dummy_ids
        ]
        warmup_input = torch.stack(padded, dim=0)
        warmup_local = warmup_input[dp_rank * local_bs: (dp_rank + 1) * local_bs].to(device)
        dllm = make_dllm()
        with torch.inference_mode():
            dllm.diff_iteration.iter_no = 0
            _ = dllm.generate(warmup_local.clone(), gen_length=64, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        dist.barrier()

        # --- collect per dataset ---
        all_ds_results = {}

        for ds_name, ds_path in DATASETS.items():
            if rank == 0:
                print(f"\n  Dataset: {ds_name}")

            prompts = load_prompts(ds_name, ds_path, BATCH_SIZE)
            all_ids = []
            for text in prompts:
                if hasattr(tokenizer, "apply_chat_template"):
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        add_generation_prompt=True, tokenize=False,
                    )
                all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
            mx = max(x.shape[0] for x in all_ids)
            padded = [
                torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                if ids.shape[0] < mx else ids
                for ids in all_ids
            ]
            input_ids_full = torch.stack(padded, dim=0)
            my_input = input_ids_full[dp_rank * local_bs: (dp_rank + 1) * local_bs].to(device)

            if rank == 0:
                print(f"    Input shape (local): {tuple(my_input.shape)}")

            collector = StalenessCollector(TARGET_LAYERS)
            dllm = make_dllm()
            hooks = install_output_hooks(model, collector)
            lifecycle = install_lifecycle_hooks(dllm, collector)

            t0 = time.time()
            try:
                with torch.inference_mode():
                    dllm.diff_iteration.iter_no = 0
                    _ = dllm.generate(
                        my_input.clone(), gen_length=GEN_LENGTH,
                        block_length=BLOCK_LENGTH,
                    )
                torch.cuda.synchronize()
                dist.barrier()
                total_fwd = dllm.diff_iteration.num_forwards
            finally:
                remove_output_hooks(hooks)
                remove_lifecycle_hooks(lifecycle)

            elapsed = time.time() - t0
            if rank == 0:
                print(f"    {total_fwd} forwards in {elapsed:.1f}s")
                for lid in TARGET_LAYERS:
                    n_gaps = len(collector.results[lid])
                    total_pts = sum(len(v) for v in collector.results[lid].values())
                    max_gap = max(collector.results[lid].keys()) if n_gaps > 0 else 0
                    print(f"      Layer {lid}: {n_gaps} gaps, {total_pts} pts, max_gap={max_gap}")

            # gather across ranks — gap-level
            local_data = {}
            for lid in TARGET_LAYERS:
                local_data[lid] = {str(g): sims for g, sims in collector.results[lid].items()}
            local_data["_step"] = {str(s): sims for s, sims in collector.step_results.items()}
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_data)

            if rank == 0:
                merged: dict[int, dict[int, list[float]]] = {
                    lid: defaultdict(list) for lid in TARGET_LAYERS
                }
                merged_step: dict[int, list[float]] = defaultdict(list)
                for gd in gathered:
                    for lid in TARGET_LAYERS:
                        for g_str, sims in gd[lid].items():
                            merged[lid][int(g_str)].extend(sims)
                    for s_str, sims in gd["_step"].items():
                        merged_step[int(s_str)].extend(sims)

                # Gap-level stats
                output_results = {}
                for lid in TARGET_LAYERS:
                    gaps_sorted = sorted(merged[lid].keys())
                    gap_stats = {}
                    for g in gaps_sorted:
                        arr = np.array(merged[lid][g])
                        gap_stats[g] = {
                            "n": len(arr),
                            "mean": round(float(arr.mean()), 5),
                            "std": round(float(arr.std()), 5),
                        }
                    output_results[f"layer_{lid}"] = {
                        "gaps": gaps_sorted,
                        "mean": [gap_stats[g]["mean"] for g in gaps_sorted],
                        "std": [gap_stats[g]["std"] for g in gaps_sorted],
                        "n": [gap_stats[g]["n"] for g in gaps_sorted],
                    }

                # Step-level stats
                steps_sorted = sorted(merged_step.keys())
                step_stats = {}
                for s in steps_sorted:
                    arr = np.array(merged_step[s])
                    step_stats[s] = {
                        "n": len(arr),
                        "mean": round(float(arr.mean()), 5),
                        "std": round(float(arr.std()), 5),
                    }
                step_output = {
                    "steps": steps_sorted,
                    "mean": [step_stats[s]["mean"] for s in steps_sorted],
                    "std": [step_stats[s]["std"] for s in steps_sorted],
                    "n": [step_stats[s]["n"] for s in steps_sorted],
                }

                # Print summary
                for lid in TARGET_LAYERS:
                    key = f"layer_{lid}"
                    gaps = output_results[key]["gaps"]
                    means = output_results[key]["mean"]
                    if gaps:
                        print(f"      Layer {lid}: gap=1 mean={means[0]:.4f}, "
                              f"gap=5 mean={means[min(4,len(means)-1)]:.4f}")
                if steps_sorted:
                    print(f"      Step-level: step=1 mean={step_stats[steps_sorted[0]]['mean']:.4f}, "
                          f"step={steps_sorted[-1]} mean={step_stats[steps_sorted[-1]]['mean']:.4f}")

                all_ds_results[ds_name] = {
                    "results": output_results,
                    "step_results": step_output,
                    "total_fwd": total_fwd,
                }

            dist.barrier()

        # --- save ---
        if rank == 0:
            meta = {
                "datasets": list(DATASETS.keys()),
                "batch_size": BATCH_SIZE,
                "gen_length": GEN_LENGTH,
                "block_length": BLOCK_LENGTH,
                "tp_size": tp_size,
                "dp_size": dp_size,
                "world_size": world_size,
                "target_layers": TARGET_LAYERS,
                "topk": 8,
                "eb_enabled": False,
                "baseline": "t_decode+1 (one step after first decode)",
                "measurement": "cosine_similarity(baseline_at_t_decode+1, output_at_t_decode+1+gap)",
            }

            output = {"meta": meta, "datasets": all_ds_results}

            out_path = REPO_ROOT / "codex_coding" / "results" / "plt" / "cache_staleness_data.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\n  Saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
            print("Done.")

        dist.barrier()


if __name__ == "__main__":
    main()
