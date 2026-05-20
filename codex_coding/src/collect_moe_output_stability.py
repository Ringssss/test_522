#!/usr/bin/env python3
"""
Collect MoE output stability data for paper motivation figure.

Measures cosine similarity of routed expert outputs between consecutive
iterations within a block, separated by token type (Decoded vs MASK).

For each dataset (GSM8K, HumanEval, MT-Bench):
  - Hook experts.forward_impl to capture routed MoE output per layer
  - Compare adjacent iterations' outputs per-token (cosine similarity)
  - Separate by decoded/MASK positions
  - Aggregate across all layers and iterations

Output: codex_coding/results/plt/moe_output_stability_data.json

Launch:
    cd /home/wuhang/wuhang/dllm_wh && \\
    torchrun --nproc_per_node=8 codex_coding/src/collect_moe_output_stability.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
N_EXPERTS = 256
BATCH_SIZE = 16
GEN_LENGTH = 256

DATASETS = {
    "GSM8K": REPO_ROOT / "data" / "gsm8k.jsonl",
    "HumanEval": REPO_ROOT / "data" / "humaneval.jsonl",
    "MT-Bench": REPO_ROOT / "data" / "mt_bench.jsonl",
}


# =====================================================================
# Dataset loading
# =====================================================================

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
# OutputStabilityCollector
# =====================================================================

class OutputStabilityCollector:
    """Collects per-token cosine similarity of MoE outputs between consecutive iterations."""

    def __init__(self):
        self._current_block_id = -1
        self._step = 0
        self.prev_outputs: dict[int, torch.Tensor] = {}
        self.prev_decoded_mask: torch.Tensor | None = None
        self.curr_decoded_mask: torch.Tensor | None = None
        self.decoded_sims: list[float] = []
        self.mask_sims: list[float] = []

    def reset(self):
        self.prev_outputs.clear()
        self.prev_decoded_mask = None
        self.curr_decoded_mask = None
        self._current_block_id = -1
        self._step = 0
        self.decoded_sims.clear()
        self.mask_sims.clear()

    def on_block_init(self, block_id: int):
        self.prev_outputs.clear()
        self.prev_decoded_mask = None
        self._current_block_id = int(block_id)
        self._step = 0

    def on_forward_start(self, input_ids: torch.Tensor | None):
        if input_ids is None or self._current_block_id < 0:
            return
        is_mask = (input_ids == MASK_ID).view(-1)
        self.curr_decoded_mask = ~is_mask
        self._step += 1

    def on_forward_end(self):
        if self._current_block_id < 0:
            return
        self.prev_decoded_mask = self.curr_decoded_mask

    def record_layer_output(self, layer_id: int, output: torch.Tensor):
        if self._current_block_id < 0 or self.curr_decoded_mask is None:
            return

        N = output.shape[0]
        curr_dec = self.curr_decoded_mask
        if curr_dec.shape[0] != N:
            self.prev_outputs[layer_id] = output.detach().clone()
            return

        if layer_id in self.prev_outputs and self.prev_decoded_mask is not None:
            prev_out = self.prev_outputs[layer_id]
            prev_dec = self.prev_decoded_mask

            if prev_out.shape[0] != N or prev_dec.shape[0] != N:
                self.prev_outputs[layer_id] = output.detach().clone()
                return

            stable_decoded = prev_dec & curr_dec
            stable_mask = ~prev_dec & ~curr_dec

            if stable_decoded.any():
                d_prev = prev_out[stable_decoded]
                d_curr = output[stable_decoded]
                sims = F.cosine_similarity(d_prev, d_curr, dim=-1)
                self.decoded_sims.extend(sims.tolist())

            if stable_mask.any():
                m_prev = prev_out[stable_mask]
                m_curr = output[stable_mask]
                sims = F.cosine_similarity(m_prev, m_curr, dim=-1)
                self.mask_sims.extend(sims.tolist())

        self.prev_outputs[layer_id] = output.detach().clone()


# =====================================================================
# Hook installation
# =====================================================================

def install_output_hooks(model, collector: OutputStabilityCollector):
    """
    Patch _moe_forward_with_context globally to capture routed expert output.
    This avoids interfering with forward_impl's internal context mechanism.
    """
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


def install_lifecycle_hooks(dllm, collector: OutputStabilityCollector):
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
    tp_rank = rank % tp_size
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
            print("MoE Output Stability Collection (K=8 native, no EB)")
            print(f"  batch={BATCH_SIZE} gen={GEN_LENGTH} block={BLOCK_LENGTH}")
            print(f"  tp={tp_size} dp={dp_size} ep={world_size}")
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
        pad_id = tokenizer.pad_token_id or 0
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
        all_results = {}

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

            collector = OutputStabilityCollector()
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
                print(f"    Local decoded_sims: {len(collector.decoded_sims)}, "
                      f"mask_sims: {len(collector.mask_sims)}")

            # Gather across ranks
            local_data = {
                "decoded_sims": collector.decoded_sims,
                "mask_sims": collector.mask_sims,
            }
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_data)

            if rank == 0:
                merged_decoded = []
                merged_mask = []
                for gd in gathered:
                    merged_decoded.extend(gd["decoded_sims"])
                    merged_mask.extend(gd["mask_sims"])

                dec_arr = np.array(merged_decoded)
                mask_arr = np.array(merged_mask)

                print(f"    Merged decoded: n={len(dec_arr)}, "
                      f"mean={dec_arr.mean():.4f}, std={dec_arr.std():.4f}, "
                      f"min={dec_arr.min():.4f}")
                print(f"    Merged mask:    n={len(mask_arr)}, "
                      f"mean={mask_arr.mean():.4f}, std={mask_arr.std():.4f}, "
                      f"min={mask_arr.min():.4f}")

                all_results[ds_name] = {
                    "decoded_sims": merged_decoded,
                    "mask_sims": merged_mask,
                    "stats": {
                        "decoded": {
                            "n": len(dec_arr),
                            "mean": round(float(dec_arr.mean()), 5),
                            "std": round(float(dec_arr.std()), 5),
                            "q25": round(float(np.percentile(dec_arr, 25)), 5),
                            "median": round(float(np.median(dec_arr)), 5),
                            "q75": round(float(np.percentile(dec_arr, 75)), 5),
                            "min": round(float(dec_arr.min()), 5),
                        },
                        "mask": {
                            "n": len(mask_arr),
                            "mean": round(float(mask_arr.mean()), 5),
                            "std": round(float(mask_arr.std()), 5),
                            "q25": round(float(np.percentile(mask_arr, 25)), 5),
                            "median": round(float(np.median(mask_arr)), 5),
                            "q75": round(float(np.percentile(mask_arr, 75)), 5),
                            "min": round(float(mask_arr.min()), 5),
                        },
                    },
                    "total_fwd": total_fwd,
                }

            dist.barrier()

        # --- save ---
        if rank == 0:
            meta = {
                "batch_size": BATCH_SIZE,
                "gen_length": GEN_LENGTH,
                "block_length": BLOCK_LENGTH,
                "tp_size": tp_size,
                "dp_size": dp_size,
                "world_size": world_size,
                "n_experts": N_EXPERTS,
                "topk": 8,
                "eb_enabled": False,
                "datasets": list(DATASETS.keys()),
                "measurement": "cosine_similarity(routed_expert_output_t, routed_expert_output_{t-1})",
                "aggregation": "all_layers x all_iterations x all_token_positions",
            }

            output = {"meta": meta, "results": all_results}

            out_path = REPO_ROOT / "codex_coding" / "results" / "plt" / "moe_output_stability_data.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(output, f)
            print(f"\n  Saved to {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
            print("Done.")

        dist.barrier()


if __name__ == "__main__":
    main()
