#!/usr/bin/env python3
"""
Physical Token Separation E2E: C10-M5 + skip (gather/scatter)

For low-confidence tokens:
  - shared_experts: computed normally (all tokens)
  - gate.get_logits: computed normally (all tokens, for EB)
  - fused_routing: only non-skip tokens (physical gather)
  - fused_experts: only non-skip tokens (physical gather)
  - skip tokens: identity pass-through (hs * rsf)

Configs: baseline (no skip), skip<0.5, skip<0.6
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

VERIFY_INDICES = [0, 8, 13, 19, 28]
VERIFY_LABELS = {0: "Math", 8: "QuadEq", 13: "Logic", 19: "Fib", 28: "Planets"}


class M5Controller(FusedEBController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_mask_cache = {}; self.pop_cache = {}
        self._fwd_in_block = {}; self.k_init_history = []

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
        self._fwd_in_block[layer_idx] = 0
        self.cold_count += 1; return b['s_mask']

    def hot_path(self, layer_idx, logits, bias):
        fi = self._fwd_in_block.get(layer_idx, 0) + 1; self._fwd_in_block[layer_idx] = fi
        if fi % 5 != 0:
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
        self.s_mask_cache.clear(); self.pop_cache.clear(); self._fwd_in_block.clear()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
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
    print("Physical Token Separation E2E: C10-M5 + skip")
    print(f"  batch={BATCH_SIZE}, gen={GEN_LENGTH}, M=5, q_major=1.0")
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

        apply_all_optimizations(model)

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
        prompt_len = input_ids.shape[1]

        decoder_t0 = ThresholdParallelDecoder(temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        decoder_t7 = ThresholdParallelDecoder(temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm(dec):
            return BlockDiffusionLLM(
                model, dec, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        # State
        state = {
            'prev_confidence': None,
            'skip_threshold': 0.0,
            'skip_token_count': [],
            'total_token_count': [],
        }

        # Collect weight refs and gate params per MoE layer
        orig_moe_fwds = {}
        moe_layer_map = {}
        moe_info = {}  # eb_li -> {w13, w2, bias, rsf, tk, ng, tkg}
        eb_idx = 0
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer.mlp, 'gate'):
                orig_moe_fwds[li] = layer.mlp.forward
                moe_layer_map[li] = eb_idx
                gate = layer.mlp.gate
                moe_info[eb_idx] = {
                    'w13': layer.mlp.experts.w13_weight,
                    'w2': layer.mlp.experts.w2_weight,
                    'bias': gate.expert_bias,
                    'rsf': gate.routed_scaling_factor,
                    'tk': gate.top_k, 'ng': gate.n_group, 'tkg': gate.topk_group,
                }
                eb_idx += 1

        def restore_all():
            for li in orig_moe_fwds:
                model.model.layers[li].mlp.forward = orig_moe_fwds[li]

        def patch_physical_skip(ctrl):
            restore_all()
            for li, layer in enumerate(model.model.layers):
                if not hasattr(layer.mlp, 'gate'):
                    continue
                moe = layer.mlp
                eb_li = moe_layer_map[li]
                info = moe_info[eb_li]

                def mk_moe_fwd(blk, layer_i, inf):
                    def fwd(hidden_states):
                        # ① shared_experts (all tokens)
                        shared_res = blk.shared_experts(hidden_states)

                        bsz, seq_len, h = hidden_states.shape
                        hs_flat = hidden_states.view(-1, h)
                        N = hs_flat.shape[0]

                        # ② gate.get_logits (all tokens, for EB)
                        logits = blk.gate.get_logits(hs_flat)  # [N, 256]

                        # ③ Determine skip
                        th = state['skip_threshold']
                        prev_conf = state['prev_confidence']

                        if th > 0 and prev_conf is not None and prev_conf.shape[0] == N:
                            # Skip path: call get_s_mask here (only call)
                            s_mask = ctrl.get_s_mask(layer_i, logits, inf['bias'])

                            skip_cpu = prev_conf < th  # [N] bool CPU
                            non_skip_cpu = ~skip_cpu
                            n_skip = int(skip_cpu.sum())
                            N_active = N - n_skip

                            if layer_i == 0:
                                state['skip_token_count'].append(n_skip)
                                state['total_token_count'].append(N)

                            if n_skip > 0 and N_active > 0:
                                non_skip_idx = non_skip_cpu.nonzero(as_tuple=True)[0].to(device)
                                skip_idx = skip_cpu.nonzero(as_tuple=True)[0].to(device)

                                # ⑤ Gather active tokens
                                hs_active = hs_flat[non_skip_idx]          # [N_active, h]
                                logits_active = logits[non_skip_idx]       # [N_active, 256]

                                # ⑥ fused_routing only active tokens
                                topk_w, topk_ids = fused_routing(
                                    logits_active, inf['bias'], inf['rsf'],
                                    s_mask=s_mask, K=inf['tk'], ng=inf['ng'], tkg=inf['tkg'])

                                # ⑦ fused_experts only active tokens
                                routed_active = fused_experts(
                                    hs_active, inf['w13'], inf['w2'],
                                    topk_w, topk_ids)

                                # ⑧ Scatter back
                                routed = torch.empty_like(hs_flat)
                                routed[non_skip_idx] = routed_active
                                routed[skip_idx] = hs_flat[skip_idx] * inf['rsf']

                                routed = routed.view(bsz, seq_len, h)
                                return shared_res + routed

                            elif N_active == 0:
                                # All tokens skipped
                                routed = hs_flat * inf['rsf']
                                return shared_res + routed.view(bsz, seq_len, h)

                        # No skip: normal path via forward_impl
                        routed = blk.experts.forward_impl(
                            hidden_states=hs_flat, router_logits=logits)
                        routed = routed.view(bsz, seq_len, h)
                        return shared_res + routed

                    return fwd

                moe.forward = mk_moe_fwd(moe, eb_li, info)

            # Also patch gate.routing for the no-skip path (forward_impl uses it)
            idx2 = 0
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate":
                    inf2 = moe_info[idx2]
                    li2 = idx2
                    def mk_rt(bb, rr, tt, nn, gg, layer_i, cc):
                        def fn(hs, go, topk, renorm):
                            sm = cc.get_s_mask(layer_i, go, bb)
                            w, ids = fused_routing(go, bb, rr, s_mask=sm, K=tt, ng=nn, tkg=gg)
                            return w.to(go.dtype), ids
                        return fn
                    mod.routing = mk_rt(inf2['bias'], inf2['rsf'], inf2['tk'],
                                        inf2['ng'], inf2['tkg'], li2, ctrl)
                    idx2 += 1

        # LM head hook
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

        def reset_state(ctrl):
            ctrl.reset()
            state['prev_confidence'] = None
            state['skip_token_count'] = []
            state['total_token_count'] = []

        # Global warmup
        print("\nGlobal warmup...")
        ctrl_w = M5Controller(num_layers=19, K=8, M=4, K_target=40,
                               quality_floor=0.70, q_major=1.0, per_round_cap=8)
        state['skip_threshold'] = 0.0
        patch_physical_skip(ctrl_w)
        reset_state(ctrl_w)
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Done: {dllm.diff_iteration.num_forwards} fwd")

        # Run configs
        results = OrderedDict()
        for label, skip_th in [("baseline", 0.0), ("skip<0.5", 0.5), ("skip<0.6", 0.6)]:
            print(f"\n{'='*60}")
            print(f"  {label} (skip_threshold={skip_th})")
            print(f"{'='*60}")

            state['skip_threshold'] = skip_th
            ctrl = M5Controller(num_layers=19, K=8, M=4, K_target=40,
                                 quality_floor=0.70, q_major=1.0, per_round_cap=8)
            patch_physical_skip(ctrl)

            # Timing (temp=0)
            reset_state(ctrl)
            dllm = make_dllm(decoder_t0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fwd_t0 = dllm.diff_iteration.num_forwards
            time_t0 = t1 - t0

            skip_rate = 0.0
            if state['skip_token_count']:
                rates = [s / t for s, t in zip(state['skip_token_count'], state['total_token_count'])]
                skip_rate = sum(rates) / len(rates)

            print(f"  temp=0: {time_t0:.3f}s, {fwd_t0} fwd, {time_t0*1000/fwd_t0:.2f} ms/fwd, skip={skip_rate:.1%}")
            if ctrl.k_init_history:
                import numpy as np
                print(f"  |S| avg={np.mean(ctrl.k_init_history):.1f}")

            # Quality (temp=0.7)
            reset_state(ctrl)
            dllm = make_dllm(decoder_t7)
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                out = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            fwd_t7 = dllm.diff_iteration.num_forwards

            gen_tokens = out[:, prompt_len:]
            print(f"  temp=0.7: {fwd_t7} fwd")
            print(f"  --- Visual Quality ---")
            for bi in VERIFY_INDICES:
                gt = gen_tokens[bi]
                valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
                text = tokenizer.decode(valid, skip_special_tokens=True)
                print(f"    #{bi} [{VERIFY_LABELS[bi]}]: {text[:200]}")

            results[label] = {
                'skip_threshold': skip_th,
                'time_s': time_t0, 'fwd_t0': fwd_t0,
                'ms_per_fwd': time_t0 * 1000 / fwd_t0,
                'skip_rate': skip_rate,
                'fwd_t7': fwd_t7,
            }

        # Cleanup
        lm_handle.remove()
        BlockDiffusionIteration.forward = orig_iter_fwd
        restore_all()

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        base = results.get('baseline', {})
        base_time = base.get('time_s', 1)
        print(f"  {'Config':<12s} {'Time(s)':>8s} {'Fwd(t0)':>8s} {'ms/fwd':>8s} "
              f"{'Skip%':>7s} {'vs base':>8s} {'Fwd(t7)':>8s}")
        print(f"  {'-'*62}")
        for label, r in results.items():
            delta = (r['time_s'] - base_time) / base_time * 100
            print(f"  {label:<12s} {r['time_s']:>8.3f} {r['fwd_t0']:>8d} {r['ms_per_fwd']:>8.2f} "
                  f"{r['skip_rate']:>6.1%} {delta:>+7.1f}% {r['fwd_t7']:>8d}")

        # Reference
        print(f"\n  Reference: C5=12.42s/278fwd, C10-M5(no skip)=12.02s/280fwd (-3.3% vs C5)")

        out_path = REPO_ROOT / "codex_coding" / "results" / "physical_skip_e2e.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
