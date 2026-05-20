#!/usr/bin/env python3
"""
mask_chaos Exp3 quality check at temp=0.7
Quick verification: baseline vs skip<0.3 vs skip<0.5
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path

import torch

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

from test_fused_eb_triton import (
    fused_routing, FusedEBController,
    _kernel_A, _kernel_A_cold, _kernel_B_v3,
    _kernel_C, _kernel_D_v2, _kernel_zero_init,
)
from baseline_optimizations import apply_all_optimizations
from test_heteval128 import PROMPTS

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
GEN_LENGTH = 256
BATCH_SIZE = 128


class MSkipEBController(FusedEBController):
    def __init__(self, *args, skip_m=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_m = skip_m
        self.k_init_history = []; self.s_mask_cache = {}; self.pop_cache = {}
        self._fwd_in_block = {}; self._block_idx = {}

    def cold_path(self, layer_idx, logits, bias):
        N, E = logits.shape; b = self._get_bufs(N, E, logits.device)
        _kernel_zero_init[(1,)](b['pop'], b['sat_flag'], b['sat_count'], b['G'], b['H'], E=E)
        lf = logits.float(); bf = bias.float()
        _kernel_A_cold[(N,)](lf, bf, b['pop'], b['topkm_idx'], b['topkm_w'], b['r'],
                             N, self.rsf, self.quality_floor, lf.stride(0), lf.stride(1),
                             b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                             E=E, KEXT=self.K_ext, KEXT_PAD=16, K=self.K)
        _kernel_B_v3[(1,)](b['pop'], b['s_mask'], self.K_target, E=E)
        q = int(self.q_major * 1000)
        for _ in range(self.MAX_ROUNDS):
            _kernel_C[(N,)](b['topkm_idx'], b['topkm_w'], b['r'],
                           b['s_mask'], b['sat_flag'], b['sat_count'], b['G'], b['H'],
                           N, b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                           E=E, KEXT=self.K_ext, KEXT_PAD=16)
            _kernel_D_v2[(1,)](b['s_mask'], b['sat_flag'], b['sat_count'],
                              b['G'], b['H'], N, q, E=E, CAP=self.cap)
        actual_s = int(b['s_mask'].sum().item())
        self.K_init[layer_idx] = actual_s; self.k_init_history.append(actual_s)
        if layer_idx not in self.s_mask_cache:
            self.s_mask_cache[layer_idx] = torch.zeros(256, device=logits.device, dtype=torch.int32)
            self.pop_cache[layer_idx] = torch.zeros(256, device=logits.device, dtype=torch.float32)
        self.s_mask_cache[layer_idx].copy_(b['s_mask'])
        self._block_idx[layer_idx] = self._block_idx.get(layer_idx, -1) + 1
        self._fwd_in_block[layer_idx] = 0
        self.cold_count += 1; return b['s_mask']

    def hot_path(self, layer_idx, logits, bias):
        fi = self._fwd_in_block.get(layer_idx, 0) + 1; self._fwd_in_block[layer_idx] = fi
        if self.skip_m == float('inf') or fi % self.skip_m != 0:
            self.hot_count += 1; return self.s_mask_cache[layer_idx]
        N, E = logits.shape; K_init = self.K_init.get(layer_idx, 103)
        pop = self.pop_cache[layer_idx]; lf = logits.float()
        _kernel_A[(N,)](lf, bias.float(), pop, N, self.rsf,
                        lf.stride(0), lf.stride(1), E=E, KEXT=self.K_ext, KEXT_PAD=16)
        _kernel_B_v3[(1,)](pop, self.s_mask_cache[layer_idx], K_init, E=E)
        self.hot_count += 1; return self.s_mask_cache[layer_idx]

    def reset(self):
        self.prev_N.clear(); self.K_init.clear(); self.cold_count = 0; self.hot_count = 0
        self._bufs.clear(); self.k_init_history.clear()
        self.s_mask_cache.clear(); self.pop_cache.clear()
        self._fwd_in_block.clear(); self._block_idx.clear()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from dinfer.decoding.generate_uniform import BlockDiffusionIteration
    from transformers import AutoTokenizer, AutoConfig

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print("mask_chaos Exp3 Quality Check (temp=0.7)")
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

        print("\nApplying baseline optimizations...")
        n_rms, n_fa = apply_all_optimizations(model)

        # Build input
        all_ids = []
        for i in range(BATCH_SIZE):
            text = PROMPTS[i]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                  if ids.shape[0] < mx else ids for ids in all_ids]
        input_ids = torch.stack(padded, dim=0).to(device)
        print(f"  Input shape: {input_ids.shape}")

        ctrl = MSkipEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5)

        state = {
            'prev_confidence': None, 'chaos_threshold': 0.0,
            'saved_topk_w': {}, 'saved_hs_flat': {},
        }

        orig_routings = {}
        moe_layer_map = {}
        idx = 0
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer.mlp, 'gate'):
                orig_routings[li] = layer.mlp.gate.routing
                moe_layer_map[li] = idx
                idx += 1

        def patch_all():
            hooks = []
            for li, layer in enumerate(model.model.layers):
                if not hasattr(layer.mlp, 'gate'): continue
                moe = layer.mlp; gate = moe.gate; eb_li = moe_layer_map[li]
                b, r = gate.expert_bias, gate.routed_scaling_factor
                tk, ng, tkg = gate.top_k, gate.n_group, gate.topk_group

                def mk_routing(bb, rr, tt, nn, gg, layer_i, cc):
                    def fn(hs, go, topk, renorm):
                        N = go.shape[0]
                        prev = cc.prev_N.get(layer_i, -1)
                        is_cold = (prev == -1) or (N > prev)
                        sm = cc.get_s_mask(layer_i, go, bb)
                        w, ids = fused_routing(go, bb, rr, s_mask=sm, K=tt, ng=nn, tkg=gg)
                        state['saved_topk_w'][layer_i] = w.detach()
                        return w.to(go.dtype), ids
                    return fn
                gate.routing = mk_routing(b, r, tk, ng, tkg, eb_li, ctrl)

                def mk_pre(layer_i):
                    def pre_hook(mod, inp):
                        hs = inp[0]
                        state['saved_hs_flat'][layer_i] = hs.view(-1, hs.shape[-1]).detach()
                    return pre_hook

                def mk_post(layer_i):
                    def post_hook(mod, inp, out):
                        th = state['chaos_threshold']
                        prev_conf = state['prev_confidence']
                        if th <= 0.0 or prev_conf is None: return out
                        hs_flat = state['saved_hs_flat'].get(layer_i)
                        topk_w = state['saved_topk_w'].get(layer_i)
                        if hs_flat is None or topk_w is None: return out
                        N = hs_flat.shape[0]
                        if prev_conf.shape[0] != N: return out
                        low_conf = prev_conf < th
                        if not low_conf.any(): return out
                        low_conf_gpu = low_conf.to(device)
                        bsz, seq_len, h = out.shape
                        out_flat = out.view(-1, h)
                        shared_out = mod.shared_experts(inp[0])
                        shared_flat = shared_out.view(-1, h)
                        weight_sum = topk_w.sum(dim=1, keepdim=True).to(hs_flat.dtype)
                        identity_routed = hs_flat * weight_sum
                        skip_output = shared_flat + identity_routed
                        new_out = torch.where(low_conf_gpu.unsqueeze(1), skip_output, out_flat)
                        return new_out.view(bsz, seq_len, h)
                    return post_hook

                hooks.append(moe.register_forward_pre_hook(mk_pre(eb_li)))
                hooks.append(moe.register_forward_hook(mk_post(eb_li)))
            return hooks

        lm_output = {}
        def lm_hook(mod, inp, out): lm_output['logits'] = out.detach()
        lm_handle = model.lm_head.register_forward_hook(lm_hook)

        orig_iter_fwd = BlockDiffusionIteration.forward
        def patched_iter_fwd(self_iter, *args, **kwargs):
            result = orig_iter_fwd(self_iter, *args, **kwargs)
            if 'logits' in lm_output:
                logits = lm_output['logits']
                probs = torch.softmax(logits.view(-1, logits.shape[-1]).float(), dim=-1)
                state['prev_confidence'] = probs.max(dim=-1).values.cpu()
            return result
        BlockDiffusionIteration.forward = patched_iter_fwd

        moe_hooks = patch_all()

        # Quality check function
        def check_quality_verbose(output_ids, label):
            checks = {
                0: ("Math avg speed", ["480/7", "68.57", "68.6"]),
                8: ("Quadratic eq", ["x = 2", "x = 3", "x=2", "x=3"]),
                13: ("Logic puzzle", ["B", "C", "D"]),
                19: ("Fibonacci", ["55"]),
                28: ("Planets", ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]),
            }
            print(f"\n  Quality Check [{label}] (temp=0.7):")
            n_pass = 0
            for idx, (desc, keywords) in checks.items():
                if idx >= output_ids.shape[0]: continue
                tokens = output_ids[idx]
                text = tokenizer.decode(tokens[tokens != MASK_ID], skip_special_tokens=True)
                found = sum(1 for kw in keywords if kw.lower() in text.lower())
                ok = found > 0
                n_pass += ok
                status = "PASS" if ok else "FAIL"
                print(f"    #{idx} {desc}: {status}")
                print(f"      Output: {text[:200]}")
            print(f"  Total: {n_pass}/5 {'PASS' if n_pass == 5 else 'PARTIAL'}")
            return n_pass

        # Warmup
        print("\nWarmup...")
        state['chaos_threshold'] = 0.0; state['prev_confidence'] = None; ctrl.reset()
        decoder_w = ThresholdParallelDecoder(temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm = BlockDiffusionLLM(
            model, decoder_w, BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True, maximum_unroll=4, expected_tpf=15,
            backend='vllm', lazy_cache_update=True, inplace_cache_update=True)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd")

        # Run quality checks at temp=0.7
        for th in [0.0, 0.3, 0.5]:
            label = f"skip<{th:.1f}" if th > 0 else "baseline"
            ctrl.reset()
            state['prev_confidence'] = None
            state['chaos_threshold'] = th
            state['saved_topk_w'].clear(); state['saved_hs_flat'].clear()

            decoder_q = ThresholdParallelDecoder(temperature=0.7, threshold=0.90,
                                                  mask_id=MASK_ID, eos_id=EOS_ID)
            dllm = BlockDiffusionLLM(
                model, decoder_q, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                output_ids = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                            block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            fwd = dllm.diff_iteration.num_forwards
            print(f"\n{'='*60}")
            print(f"  {label}: {fwd} fwd")
            check_quality_verbose(output_ids, label)

        # Cleanup
        for h in moe_hooks: h.remove()
        lm_handle.remove()
        BlockDiffusionIteration.forward = orig_iter_fwd
        for li in orig_routings:
            model.model.layers[li].mlp.gate.routing = orig_routings[li]

        print("\nDone.")


if __name__ == "__main__":
    main()
