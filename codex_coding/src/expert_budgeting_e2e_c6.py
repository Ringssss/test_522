#!/usr/bin/env python3
"""
v0.1.15.3 Phase 2 supplement — Run only C6 (budgeting + top-p) and quality check.
Reuses the same setup as expert_budgeting_e2e.py.
"""

from __future__ import annotations
import os, sys, socket, json
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/home/wuhang/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
TOP_K_ORIG = 8
SHARED_RATE = 0.419
ROUTING_RATE = 0.581
NUM_EXPERTS = 256

# Import from main script
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))
from expert_budgeting_e2e import (
    PROMPTS, compute_active_set,
    ExpertBudgetingController, BudgetingPlusTopPController,
    install_hooks, remove_hooks
)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoTokenizer, AutoConfig

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print("Phase 2 Supplement — C6 + Quality Check")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)

        BATCH_SIZE = 32
        all_ids = []
        for i in range(BATCH_SIZE):
            text = PROMPTS[i % len(PROMPTS)]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=False)
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = []
        for ids in all_ids:
            if ids.shape[0] < mx:
                ids = torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
            padded.append(ids)
        input_ids = torch.stack(padded, dim=0).to(device)
        prompt_len = input_ids.shape[1]

        # ---- temp=0 DLLM ----
        decoder_t0 = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm = BlockDiffusionLLM(
            model, decoder_t0,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=1, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        # ---- C6: K=150 + top-p=0.75 (5 runs, temp=0) ----
        print("\n=== C6: K=150 + top-p=0.75 ===")
        N_RUNS = 5
        fwds_list = []
        for run in range(N_RUNS):
            ctrl = BudgetingPlusTopPController(K_target=150, quality_floor=0.85, top_p=0.75)
            hooks = install_hooks(model, ctrl)
            try:
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    dllm.diff_iteration.iter_no = 0
                    out = dllm.generate(input_ids.clone(), gen_length=128,
                                        block_length=BLOCK_LENGTH)
                fwds_list.append(dllm.diff_iteration.num_forwards)
                if run == 0:
                    avg_active = sum(ctrl.stats["total_active"]) / max(len(ctrl.stats["total_active"]), 1)
                    avg_exp = sum(ctrl.stats["avg_experts_per_token"]) / max(len(ctrl.stats["avg_experts_per_token"]), 1)
            finally:
                remove_hooks(hooks)

        avg_fwd = sum(fwds_list) / len(fwds_list)
        print(f"  Fwds: {fwds_list}  avg={avg_fwd:.0f}")
        print(f"  Avg active experts: {avg_active:.1f}")
        print(f"  Avg experts/token (after top-p): {avg_exp:.2f}")
        print(f"  ΔFwd vs baseline(149): {avg_fwd - 149:+.0f}")

        # ---- Quality check (temp=0.7) ----
        print(f"\n{'='*80}")
        print(f"OUTPUT QUALITY (temp=0.7)")
        print(f"{'='*80}")

        decoder_t7 = ThresholdParallelDecoder(
            temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm_t7 = BlockDiffusionLLM(
            model, decoder_t7,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=1, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)
        with torch.inference_mode():
            dllm_t7.diff_iteration.num_forwards = 0
            _ = dllm_t7.generate(input_ids.clone(), gen_length=128, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        quality_configs = [
            ("baseline",       None),
            ("K=150",          ExpertBudgetingController(K_target=150, quality_floor=0.85)),
            ("K=150+tp75",     BudgetingPlusTopPController(K_target=150, quality_floor=0.85, top_p=0.75)),
        ]

        quality_outputs = {}
        for cname, ctrl in quality_configs:
            if ctrl is None:
                with torch.inference_mode():
                    dllm_t7.diff_iteration.num_forwards = 0
                    out = dllm_t7.generate(input_ids.clone(), gen_length=128,
                                           block_length=BLOCK_LENGTH)
                quality_outputs[cname] = out[:, prompt_len:].cpu()
            else:
                hooks = install_hooks(model, ctrl)
                try:
                    with torch.inference_mode():
                        dllm_t7.diff_iteration.num_forwards = 0
                        out = dllm_t7.generate(input_ids.clone(), gen_length=128,
                                               block_length=BLOCK_LENGTH)
                    quality_outputs[cname] = out[:, prompt_len:].cpu()
                finally:
                    remove_hooks(hooks)

        for bi in range(3):
            print(f"\n{'─'*80}")
            print(f"  BATCH {bi}: {PROMPTS[bi][:65]}...")
            print(f"{'─'*80}")
            for cname, _ in quality_configs:
                gt = quality_outputs[cname][bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                print(f"\n  [{cname}]:")
                print(f"  {text[:300]}")

        print(f"\n{'='*80}")
        print("COMPLETE SUMMARY (all configs)")
        print(f"{'='*80}")
        prev = {
            "C0:baseline": (149, 220.8),
            "C1:K=200": (146, 208.1),
            "C2:K=180": (147, 194.9),
            "C3:K=150": (149, 176.6),
            "C4:K=120": (151, 165.4),
            "C5:K=150,QF75": (151, 170.2),
        }
        print(f"  {'Config':<18s} {'AvgFwd':>7s} {'dFwd':>5s} {'Active':>7s} {'HBM%':>6s} {'Verdict':>8s}")
        print(f"  {'-'*55}")
        for cn, (fwd, act) in prev.items():
            d = fwd - 149
            h = (1 - act / 220.8) * 100
            v = "—" if cn.startswith("C0") else ("SAFE" if abs(d) <= 2 else "MARGIN")
            print(f"  {cn:<18s} {fwd:>7.0f} {d:>+4.0f} {act:>7.1f} {h:>5.1f}% {v:>8s}")
        # C6
        h6 = (1 - avg_active / 220.8) * 100
        d6 = avg_fwd - 149
        v6 = "SAFE" if abs(d6) <= 2 else ("MARGIN" if abs(d6) <= 5 else "BAD")
        print(f"  {'C6:K=150+tp75':<18s} {avg_fwd:>7.0f} {d6:>+4.0f} {avg_active:>7.1f} {h6:>5.1f}% {v6:>8s}")
        print(f"    (avg {avg_exp:.2f} experts/token after top-p)")

        print("\nDone.")


if __name__ == "__main__":
    main()
