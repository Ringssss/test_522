#!/usr/bin/env python3
"""
v0.1.15.8n — Routing TopK Compression Frontier

S_mask is computed with the original K=8 (unchanged).
Only fused_routing's K and tkg are overridden after S_mask is ready.
This isolates the variable to MoE computation volume only.

Configs tested:
  C10-M5-K4:        K=4, tkg=4  — 4 groups x 1 expert  [reference]
  C10-M5-K2-TKG2:   K=2, tkg=2  — 2 groups x 1 expert
  C10-M5-K2-TKG1:   K=2, tkg=1  — 1 group  x 2 experts
  C10-M5-K1-TKG1:   K=1, tkg=1  — 1 group  x 1 expert  [extreme]
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path
from collections import OrderedDict

import torch

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

from test_fused_eb_triton import (
    fused_routing, FusedEBController,
    _kernel_A_cold, _kernel_B_v3,
    _kernel_C, _kernel_D_v2, _kernel_zero_init,
)
from baseline_optimizations import apply_all_optimizations
from test_heteval128 import PROMPTS, VERIFIABLE

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
GEN_LENGTH = 256
BATCH_SIZE = 128


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
    print("Routing TopK Compression Frontier")
    print(f"  batch={BATCH_SIZE}, gen_length={GEN_LENGTH}, block_length={BLOCK_LENGTH}")
    print(f"  threshold=0.90, M=5, q_major=1.0")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0),
                      use_cache=False)

        print("\nApplying baseline optimizations...")
        n_rms, n_fa = apply_all_optimizations(model)
        print(f"  Fused RMSNorm: {n_rms}, Flash-attn: {n_fa}")

        # Build input
        all_ids = []
        for i in range(BATCH_SIZE):
            text = PROMPTS[i]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=False)
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                  if ids.shape[0] < mx else ids for ids in all_ids]
        input_ids = torch.stack(padded, dim=0).to(device)
        prompt_len = input_ids.shape[1]
        print(f"  Input shape: {input_ids.shape}")

        decoder = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm():
            return BlockDiffusionLLM(
                model, decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True,
                inplace_cache_update=True)

        orig_routings = {}
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                orig_routings[name] = mod.routing

        def restore():
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate" and name in orig_routings:
                    mod.routing = orig_routings[name]

        def patch_eb(ctrl, routing_K, routing_tkg=None):
            """
            S_mask always computed with original K=8 (controller's K).
            routing_K and routing_tkg override fused_routing params only.
            """
            restore()
            idx = 0
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate":
                    b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                         mod.top_k, mod.n_group, mod.topk_group)
                    li = idx
                    rtkg = routing_tkg if routing_tkg is not None else tkg
                    def mk(bb, rr, orig_tk, nn, gg, layer_i, cc, rK, rTKG):
                        def fn(hs, go, topk, renorm):
                            sm = cc.get_s_mask(layer_i, go, bb)
                            w, i = fused_routing(go, bb, rr, s_mask=sm, K=rK, ng=nn, tkg=rTKG)
                            return w.to(go.dtype), i
                        return fn
                    mod.routing = mk(b, r, tk, ng, tkg, li, ctrl, routing_K, rtkg)
                    idx += 1

        results = OrderedDict()

        # Configs: (label, routing_K, routing_tkg, eb_M)
        configs = [
            ("C10_M5_K8",        8, 4, 5),    # baseline (original topk=8)
            ("C10_M5_K4",        4, 4, 5),    # topk=4, M=5
            ("C10_M1_K4",        4, 4, 1),    # topk=4, M=1 (update every fwd)
        ]

        for label, rK, rTKG, ebM in configs:
            print(f"\n{'='*60}")
            print(f"{label}: S_mask(K=8) + routing(K={rK}, tkg={rTKG}), M={ebM}")
            print(f"{'='*60}")
            ctrl = FusedEBController(
                num_layers=19, K=8, M=ebM, K_target=40,
                quality_floor=0.70, q_major=1.0, per_round_cap=8)
            patch_eb(ctrl, routing_K=rK, routing_tkg=rTKG)

            # Warmup
            dllm = make_dllm()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            print(f"  Warmup: {dllm.diff_iteration.num_forwards} fwd, "
                  f"cold={ctrl.cold_count}, hot={ctrl.hot_count}")

            times, fwds = [], []
            for ri in range(2):
                ctrl.prev_N.clear(); ctrl.K_init.clear()
                ctrl.cold_count = 0; ctrl.hot_count = 0; ctrl._bufs.clear()
                dllm = make_dllm(); torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    dllm.diff_iteration.num_forwards = 0
                    _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                      block_length=BLOCK_LENGTH)
                torch.cuda.synchronize(); t1 = time.perf_counter()
                times.append(t1-t0)
                fwds.append(dllm.diff_iteration.num_forwards)
                print(f"    Run {ri+1}: {t1-t0:.3f}s, "
                      f"{dllm.diff_iteration.num_forwards} fwd "
                      f"| cold={ctrl.cold_count} hot={ctrl.hot_count}")

            results[label] = {
                'avg_time': sum(times)/2, 'avg_fwd': sum(fwds)/2,
                'ms_per_fwd': sum(times)/2 / (sum(fwds)/2) * 1000,
            }

        # ---- Quality comparison ----
        print(f"\n{'='*60}")
        print("Quality comparison on verifiable prompts (temp=0.0)")
        print(f"{'='*60}")

        for label, rK, rTKG, ebM in configs:
            ctrl_q = FusedEBController(
                num_layers=19, K=8, M=ebM, K_target=40,
                quality_floor=0.70, q_major=1.0, per_round_cap=8)
            patch_eb(ctrl_q, routing_K=rK, routing_tkg=rTKG)

            # Warmup
            dllm = make_dllm()
            with torch.inference_mode():
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            ctrl_q.prev_N.clear(); ctrl_q.K_init.clear()
            ctrl_q.cold_count = 0; ctrl_q.hot_count = 0; ctrl_q._bufs.clear()

            dllm = make_dllm()
            with torch.inference_mode():
                out = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                    block_length=BLOCK_LENGTH)
            gen_tokens = out[:, prompt_len:]

            print(f"\n  --- {label} (K={rK}, tkg={rTKG}) ---")
            for bi in sorted(VERIFIABLE.keys()):
                gt = gen_tokens[bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                expected = VERIFIABLE[bi]
                print(f"    #{bi} [{expected}]:")
                print(f"      {text[:200]}")

        # ---- Summary ----
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        k8_r = results['C10_M5_K8']
        hdr = f"  {'Config':<20s} {'Time(s)':>8s} {'Fwd':>5s} {'ms/fwd':>8s} {'vs K8':>8s} {'dFwd':>6s}"
        print(hdr)
        print(f"  {'-'*58}")
        for cn, r in results.items():
            d = (r['avg_time'] - k8_r['avg_time']) / k8_r['avg_time'] * 100
            df = r['avg_fwd'] - k8_r['avg_fwd']
            print(f"  {cn:<20s} {r['avg_time']:>8.3f} {r['avg_fwd']:>5.0f} "
                  f"{r['ms_per_fwd']:>8.2f} {d:>+7.1f}% {df:>+5.0f}")

        out_path = REPO_ROOT / "codex_coding" / "results" / "routing_topk_frontier.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
