#!/usr/bin/env python3
"""
v0.1.15.8l — Component-level profiling at batch=128

C5 vs C10-M5, gen_length=64, with EB sub-kernel breakdown.

EB sub-kernel tags:
  Cold: eb_cold_zero_init, eb_cold_K_A, eb_cold_K_B, eb_cold_batchadd
  Hot:  eb_hot_update (K_A + K_B_v3), eb_hot_skip (cache return)

Usage:
  CUDA_VISIBLE_DEVICES=4 conda run -n dllm python nsys_b128_profiling.py
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path
from collections import defaultdict

import torch

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

from test_fused_eb_triton import (
    fused_routing, FusedEBController,
    _kernel_A, _kernel_A_cold, _kernel_B_v3,
    _kernel_C, _kernel_D_v2, _kernel_zero_init,
)
from baseline_optimizations import apply_all_optimizations
from test_heteval128 import PROMPTS  # 128 heterogeneous prompts

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
GEN_LENGTH = 64   # 2 blocks, same as batch=32 profiling
BATCH_SIZE = 128


# ================================================================
# CUDA Event Timer (zero-overhead recording, read after sync)
# ================================================================
class ComponentTimer:
    def __init__(self):
        self._stack = []
        self.data = defaultdict(list)

    def start(self, tag):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._stack.append((tag, ev))

    def end(self):
        tag, start_ev = self._stack.pop()
        end_ev = torch.cuda.Event(enable_timing=True)
        end_ev.record()
        self.data[tag].append((start_ev, end_ev))

    def reset(self):
        self._stack.clear()
        self.data.clear()

    def summarize(self):
        torch.cuda.synchronize()
        result = {}
        for tag in sorted(self.data.keys()):
            pairs = self.data[tag]
            times = [s.elapsed_time(e) for s, e in pairs]
            result[tag] = {
                'count': len(times),
                'total_ms': sum(times),
                'avg_ms': sum(times) / len(times) if times else 0,
            }
        return result


# ================================================================
# Profiling M-skip EB Controller (M=5, sub-kernel timing)
# ================================================================
class ProfilingMSkipEBController(FusedEBController):
    def __init__(self, *args, skip_m=5, timer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_m = skip_m
        self.timer = timer
        self.k_init_history = []
        self.s_mask_cache = {}
        self.pop_cache = {}
        self._fwd_in_block = {}
        self._block_idx = {}
        self.eb_calls = 0
        self.eb_skips = 0

    def cold_path(self, layer_idx, logits, bias):
        N, E = logits.shape
        b = self._get_bufs(N, E, logits.device)
        t = self.timer

        # zero_init
        t.start(f"eb_cold_zero_init_L{layer_idx}")
        _kernel_zero_init[(1,)](b['pop'], b['sat_flag'], b['sat_count'],
                                b['G'], b['H'], E=E)
        t.end()

        lf = logits.float()
        bf = bias.float()

        # K_A_cold
        t.start(f"eb_cold_K_A_L{layer_idx}")
        _kernel_A_cold[(N,)](lf, bf, b['pop'], b['topkm_idx'], b['topkm_w'], b['r'],
                             N, self.rsf, self.quality_floor,
                             lf.stride(0), lf.stride(1),
                             b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                             E=E, KEXT=self.K_ext, KEXT_PAD=16, K=self.K)
        t.end()

        # K_B_v3
        t.start(f"eb_cold_K_B_L{layer_idx}")
        _kernel_B_v3[(1,)](b['pop'], b['s_mask'], self.K_target, E=E)
        t.end()

        # batch-add loop
        t.start(f"eb_cold_batchadd_L{layer_idx}")
        q_major_x1000 = int(self.q_major * 1000)
        for _ in range(self.MAX_ROUNDS):
            _kernel_C[(N,)](b['topkm_idx'], b['topkm_w'], b['r'],
                           b['s_mask'], b['sat_flag'], b['sat_count'], b['G'], b['H'],
                           N, b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                           E=E, KEXT=self.K_ext, KEXT_PAD=16)
            _kernel_D_v2[(1,)](b['s_mask'], b['sat_flag'], b['sat_count'],
                              b['G'], b['H'], N, q_major_x1000, E=E, CAP=self.cap)
        t.end()

        actual_s = int(b['s_mask'].sum().item())
        self.K_init[layer_idx] = actual_s
        self.k_init_history.append(actual_s)

        if layer_idx not in self.s_mask_cache:
            self.s_mask_cache[layer_idx] = torch.zeros(256, device=logits.device, dtype=torch.int32)
            self.pop_cache[layer_idx] = torch.zeros(256, device=logits.device, dtype=torch.float32)
        self.s_mask_cache[layer_idx].copy_(b['s_mask'])

        bi = self._block_idx.get(layer_idx, -1) + 1
        self._block_idx[layer_idx] = bi
        self._fwd_in_block[layer_idx] = 0
        self.cold_count += 1
        return b['s_mask']

    def hot_path(self, layer_idx, logits, bias):
        N, E = logits.shape
        K_init = self.K_init.get(layer_idx, 103)
        t = self.timer

        fi = self._fwd_in_block.get(layer_idx, 0) + 1
        self._fwd_in_block[layer_idx] = fi

        if self.skip_m == float('inf') or fi % self.skip_m != 0:
            # Skip: return cached s_mask
            t.start(f"eb_hot_skip_L{layer_idx}")
            t.end()  # near-zero, just records the event pair
            self.eb_skips += 1
            self.hot_count += 1
            return self.s_mask_cache[layer_idx]

        # Update: K_A + K_B_v3
        t.start(f"eb_hot_update_L{layer_idx}")
        pop = self.pop_cache[layer_idx]
        lf = logits.float()
        _kernel_A[(N,)](lf, bias.float(), pop, N, self.rsf,
                        lf.stride(0), lf.stride(1), E=E, KEXT=self.K_ext, KEXT_PAD=16)
        _kernel_B_v3[(1,)](pop, self.s_mask_cache[layer_idx], K_init, E=E)
        t.end()

        self.eb_calls += 1
        self.hot_count += 1
        return self.s_mask_cache[layer_idx]

    def get_s_mask(self, layer_idx, logits, bias):
        N = logits.shape[0]
        if self.is_new_block(layer_idx, N):
            return self.cold_path(layer_idx, logits, bias)
        else:
            return self.hot_path(layer_idx, logits, bias)


# ================================================================
# Install all instrumentation
# ================================================================
def install_instrumentation(model, timer, config_name, eb_ctrl=None):
    hooks = []

    def hook_mod(mod, tag):
        def pre(m, inp):
            timer.start(tag)
        def post(m, inp, out):
            timer.end()
        hooks.append(mod.register_forward_pre_hook(pre))
        hooks.append(mod.register_forward_hook(post))

    # Embedding
    emb = getattr(model.model, 'word_embeddings',
                  getattr(model.model, 'embed_tokens', None))
    if emb:
        hook_mod(emb, "Embedding")

    for li, layer in enumerate(model.model.layers):
        is_moe = hasattr(layer.mlp, 'gate')

        hook_mod(layer.input_layernorm, f"RMSNorm_pre_L{li}")
        attn = layer.attention if hasattr(layer, 'attention') else layer.self_attn
        hook_mod(attn, f"Attention_L{li}")
        hook_mod(layer.post_attention_layernorm, f"RMSNorm_post_L{li}")

        if not is_moe:
            hook_mod(layer.mlp, f"DenseMLP_L{li}")
            continue

        # --- MoE layer: monkey-patch forward + routing ---
        moe = layer.mlp
        gate = moe.gate
        g_bias = gate.expert_bias
        g_rsf = gate.routed_scaling_factor
        g_tk, g_ng, g_tkg = gate.top_k, gate.n_group, gate.topk_group

        # Patch gate.routing
        # For C10: EB sub-kernel timing is inside the controller,
        # so we just time the overall EB call as eb_total_L{li}
        def make_routing_fn(bias, rsf, tk, ng, tkg, layer_i, ctrl):
            def fn(hidden_states, gating_output, topk, renormalize):
                if ctrl is not None:
                    timer.start(f"eb_total_L{layer_i}")
                    s_mask = ctrl.get_s_mask(layer_i, gating_output, bias)
                    timer.end()
                else:
                    s_mask = None

                timer.start(f"routing_L{layer_i}")
                w, idx = fused_routing(gating_output, bias, rsf,
                                       s_mask=s_mask, K=tk, ng=ng, tkg=tkg)
                timer.end()
                return w.to(gating_output.dtype), idx
            return fn

        gate.routing = make_routing_fn(
            g_bias, g_rsf, g_tk, g_ng, g_tkg, li,
            eb_ctrl if config_name == "C10" else None)

        # Patch MoE block forward
        def make_moe_fwd(blk, layer_i):
            def fwd(hidden_states):
                timer.start(f"shared_L{layer_i}")
                res = blk.shared_experts(hidden_states)
                timer.end()

                bsz, seq_len, h = hidden_states.shape
                hs_flat = hidden_states.view(-1, h)

                timer.start(f"getlogits_L{layer_i}")
                logits = blk.gate.get_logits(hs_flat)
                timer.end()

                timer.start(f"fwd_impl_L{layer_i}")
                y = blk.experts.forward_impl(
                    hidden_states=hs_flat, router_logits=logits)
                timer.end()

                y = y.view(bsz, seq_len, h)

                timer.start(f"residual_L{layer_i}")
                y = y + res
                timer.end()

                return y
            return fwd

        moe.forward = make_moe_fwd(moe, li)

    # Final RMSNorm + LMHead
    hook_mod(model.model.norm, "FinalRMSNorm")
    hook_mod(model.lm_head, "LMHead")

    return hooks


# ================================================================
# Analyze and print results
# ================================================================
def analyze(raw_stats, config_name, total_time_s, total_fwd, eb_ctrl=None):
    total_ms = total_time_s * 1000

    # Aggregate by component type
    agg = defaultdict(lambda: {'count': 0, 'total_ms': 0.0})
    eb_sub = defaultdict(lambda: {'count': 0, 'total_ms': 0.0})

    for tag, s in raw_stats.items():
        # EB sub-kernel tags (inside controller)
        if tag.startswith("eb_cold_zero_init_L"):
            eb_sub['cold_zero_init']['count'] += s['count']
            eb_sub['cold_zero_init']['total_ms'] += s['total_ms']
        elif tag.startswith("eb_cold_K_A_L"):
            eb_sub['cold_K_A']['count'] += s['count']
            eb_sub['cold_K_A']['total_ms'] += s['total_ms']
        elif tag.startswith("eb_cold_K_B_L"):
            eb_sub['cold_K_B']['count'] += s['count']
            eb_sub['cold_K_B']['total_ms'] += s['total_ms']
        elif tag.startswith("eb_cold_batchadd_L"):
            eb_sub['cold_batchadd']['count'] += s['count']
            eb_sub['cold_batchadd']['total_ms'] += s['total_ms']
        elif tag.startswith("eb_hot_update_L"):
            eb_sub['hot_update']['count'] += s['count']
            eb_sub['hot_update']['total_ms'] += s['total_ms']
        elif tag.startswith("eb_hot_skip_L"):
            eb_sub['hot_skip']['count'] += s['count']
            eb_sub['hot_skip']['total_ms'] += s['total_ms']
        elif tag.startswith("eb_total_L"):
            agg['EB_total']['count'] += s['count']
            agg['EB_total']['total_ms'] += s['total_ms']
        # Standard components
        elif tag.startswith("shared_L"):       comp = "shared_experts"
        elif tag.startswith("getlogits_L"):  comp = "gate_getlogits"
        elif tag.startswith("routing_L"):    comp = "routing"
        elif tag.startswith("fwd_impl_L"):   comp = "fwd_impl"
        elif tag.startswith("residual_L"):   comp = "residual"
        elif tag.startswith("Attention_L"):  comp = "Attention"
        elif tag.startswith("RMSNorm_pre_L"):  comp = "RMSNorm_pre"
        elif tag.startswith("RMSNorm_post_L"): comp = "RMSNorm_post"
        elif tag.startswith("DenseMLP_L"):   comp = "DenseMLP"
        elif tag == "Embedding":    comp = "Embedding"
        elif tag == "FinalRMSNorm": comp = "FinalRMSNorm"
        elif tag == "LMHead":       comp = "LMHead"
        else: continue

        if not tag.startswith("eb_"):
            agg[comp]['count'] += s['count']
            agg[comp]['total_ms'] += s['total_ms']

    # Derived: fused_experts = fwd_impl - EB_total - routing
    eb_total_ms = agg['EB_total']['total_ms']
    routing_ms = agg['routing']['total_ms']
    fwd_impl_ms = agg['fwd_impl']['total_ms']
    fused_experts_ms = max(0, fwd_impl_ms - eb_total_ms - routing_ms)

    moe_sub = (agg['shared_experts']['total_ms'] + agg['gate_getlogits']['total_ms']
               + fwd_impl_ms + agg['residual']['total_ms'])
    non_moe = (agg['Attention']['total_ms'] + agg['RMSNorm_pre']['total_ms']
               + agg['RMSNorm_post']['total_ms'] + agg['DenseMLP']['total_ms']
               + agg['Embedding']['total_ms'] + agg['FinalRMSNorm']['total_ms']
               + agg['LMHead']['total_ms'])
    instrumented = moe_sub + non_moe
    gap = total_ms - instrumented

    # Print
    print(f"\n{'='*80}")
    print(f"COMPONENT BREAKDOWN — {config_name} (batch={BATCH_SIZE})")
    print(f"  Wall-clock: {total_time_s:.3f}s ({total_ms:.1f}ms) | "
          f"Fwd: {total_fwd} | ms/fwd: {total_ms/total_fwd:.2f}")
    if eb_ctrl:
        print(f"  EB cold: {eb_ctrl.cold_count} | EB hot: {eb_ctrl.hot_count} "
              f"| updates: {eb_ctrl.eb_calls} | skips: {eb_ctrl.eb_skips}")
    print(f"{'='*80}")

    rows = []
    rows.append(("=== MoE (L1-L19) ===", None))
    rows.append(("  shared_experts", agg['shared_experts']['total_ms']))
    rows.append(("  gate_getlogits", agg['gate_getlogits']['total_ms']))
    if config_name == "C10":
        rows.append(("  EB_total", eb_total_ms))
        rows.append(("    cold_zero_init", eb_sub['cold_zero_init']['total_ms']))
        rows.append(("    cold_K_A", eb_sub['cold_K_A']['total_ms']))
        rows.append(("    cold_K_B (v3)", eb_sub['cold_K_B']['total_ms']))
        rows.append(("    cold_batchadd", eb_sub['cold_batchadd']['total_ms']))
        cold_sum = sum(eb_sub[k]['total_ms'] for k in
                       ['cold_zero_init', 'cold_K_A', 'cold_K_B', 'cold_batchadd'])
        rows.append(("    --- cold subtotal ---", cold_sum))
        rows.append(("    hot_update (K_A+K_B)", eb_sub['hot_update']['total_ms']))
        rows.append(("    hot_skip (cache)", eb_sub['hot_skip']['total_ms']))
        hot_sum = eb_sub['hot_update']['total_ms'] + eb_sub['hot_skip']['total_ms']
        rows.append(("    --- hot subtotal ---", hot_sum))
    rows.append(("  routing (Triton)", routing_ms))
    rows.append(("  fused_experts *", fused_experts_ms))
    rows.append(("  fwd_impl (rout+exp)", fwd_impl_ms))
    rows.append(("  residual_add", agg['residual']['total_ms']))
    rows.append(("  --- MoE subtotal ---", moe_sub))
    rows.append(("", None))
    rows.append(("=== Non-MoE ===", None))
    rows.append(("  Attention", agg['Attention']['total_ms']))
    rows.append(("  RMSNorm_pre", agg['RMSNorm_pre']['total_ms']))
    rows.append(("  RMSNorm_post", agg['RMSNorm_post']['total_ms']))
    rows.append(("  DenseMLP (L0)", agg['DenseMLP']['total_ms']))
    rows.append(("  Embedding", agg['Embedding']['total_ms']))
    rows.append(("  FinalRMSNorm", agg['FinalRMSNorm']['total_ms']))
    rows.append(("  LMHead", agg['LMHead']['total_ms']))
    rows.append(("  --- Non-MoE subtotal ---", non_moe))
    rows.append(("", None))
    rows.append(("Instrumented total", instrumented))
    rows.append(("Gap (Python/decoder)", gap))

    print(f"\n  {'Component':<28s} {'Total(ms)':>10s} {'Avg/fwd':>10s} "
          f"{'%wall':>7s} {'%instr':>7s}")
    print(f"  {'-'*63}")
    for name, ms in rows:
        if ms is None:
            print(f"  {name}")
            continue
        if not name:
            print()
            continue
        pw = ms / total_ms * 100
        pi = ms / instrumented * 100 if instrumented > 0 else 0
        af = ms / total_fwd
        print(f"  {name:<28s} {ms:>10.1f} {af:>10.3f} {pw:>6.1f}% {pi:>6.1f}%")

    # Build per-layer detail
    per_layer = {}
    for li in range(1, 20):
        fi_key = f"fwd_impl_L{li}"
        rt_key = f"routing_L{li}"
        eb_key = f"eb_total_L{li}"
        if fi_key in raw_stats:
            fi = raw_stats[fi_key]['total_ms']
            rt = raw_stats.get(rt_key, {}).get('total_ms', 0)
            eb = raw_stats.get(eb_key, {}).get('total_ms', 0)
            fe = fi - rt - eb
            per_layer[li] = {
                'fwd_impl_ms': fi, 'routing_ms': rt,
                'eb_total_ms': eb, 'fused_experts_ms': fe,
            }

    result = {
        'config': config_name, 'gen_length': GEN_LENGTH, 'batch_size': BATCH_SIZE,
        'total_time_s': total_time_s, 'total_fwd': total_fwd,
        'ms_per_fwd': total_ms / total_fwd,
        'components': {
            'shared_experts': agg['shared_experts']['total_ms'],
            'gate_getlogits': agg['gate_getlogits']['total_ms'],
            'EB_total': eb_total_ms,
            'routing': routing_ms,
            'fused_experts_derived': fused_experts_ms,
            'fwd_impl': fwd_impl_ms,
            'residual': agg['residual']['total_ms'],
            'MoE_subtotal': moe_sub,
            'Attention': agg['Attention']['total_ms'],
            'RMSNorm_pre': agg['RMSNorm_pre']['total_ms'],
            'RMSNorm_post': agg['RMSNorm_post']['total_ms'],
            'DenseMLP': agg['DenseMLP']['total_ms'],
            'Embedding': agg['Embedding']['total_ms'],
            'FinalRMSNorm': agg['FinalRMSNorm']['total_ms'],
            'LMHead': agg['LMHead']['total_ms'],
            'non_MoE_subtotal': non_moe,
            'instrumented_total': instrumented,
            'gap': gap,
        },
        'eb_sub_kernels': {k: v for k, v in eb_sub.items()},
        'per_layer': per_layer,
        'eb_cold_count': eb_ctrl.cold_count if eb_ctrl else 0,
        'eb_hot_count': eb_ctrl.hot_count if eb_ctrl else 0,
        'eb_update_count': eb_ctrl.eb_calls if eb_ctrl else 0,
        'eb_skip_count': eb_ctrl.eb_skips if eb_ctrl else 0,
    }
    return result


# ================================================================
# Main
# ================================================================
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
    print(f"Component Profiling — batch={BATCH_SIZE}, gen_length={GEN_LENGTH}")
    print(f"  C5 vs C10-M5, block={BLOCK_LENGTH}, threshold=0.90")
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

        # Build input (HetEval-128)
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
        print(f"  Input shape: {input_ids.shape} (batch={BATCH_SIZE})")
        mem = torch.cuda.memory_allocated() / 1e9
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU memory: {mem:.1f}GB / {total_mem:.1f}GB")

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

        # Save original routing functions for restore
        orig_routings = {}
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                orig_routings[name] = mod.routing

        orig_moe_fwds = {}
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeSparseMoeBlock":
                orig_moe_fwds[name] = mod.forward

        def restore_all():
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate" and name in orig_routings:
                    mod.routing = orig_routings[name]
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeSparseMoeBlock" and name in orig_moe_fwds:
                    mod.forward = orig_moe_fwds[name]

        results = {}

        for config_name in ["C5", "C10"]:
            print(f"\n{'='*60}")
            print(f"Profiling: {config_name}" +
                  (" (M=5, q_major=1.0)" if config_name == "C10" else " (fused routing only)"))
            print(f"{'='*60}")

            restore_all()

            timer = ComponentTimer()
            eb_ctrl = None
            if config_name == "C10":
                eb_ctrl = ProfilingMSkipEBController(
                    num_layers=19, K=8, M=4, K_target=40,
                    quality_floor=0.70, q_major=1.0, per_round_cap=8,
                    skip_m=5, timer=timer)

            hooks = install_instrumentation(model, timer, config_name, eb_ctrl)
            print(f"  Hooks: {len(hooks)}")

            # Warmup
            print("  Warmup...")
            dllm = make_dllm()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd")

            # Reset
            timer.reset()
            if eb_ctrl:
                eb_ctrl.prev_N.clear(); eb_ctrl.K_init.clear()
                eb_ctrl.cold_count = 0; eb_ctrl.hot_count = 0
                eb_ctrl._bufs.clear(); eb_ctrl.k_init_history.clear()
                eb_ctrl.s_mask_cache.clear(); eb_ctrl.pop_cache.clear()
                eb_ctrl._fwd_in_block.clear(); eb_ctrl._block_idx.clear()
                eb_ctrl.eb_calls = 0; eb_ctrl.eb_skips = 0

            # Profiled generation
            print("  Profiling...")
            dllm = make_dllm()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH,
                                  block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            total_fwd = dllm.diff_iteration.num_forwards
            total_time = t1 - t0
            print(f"  Done: {total_fwd} fwd, {total_time:.3f}s, "
                  f"{total_ms:.1f} ms/fwd" if False else
                  f"  Done: {total_fwd} fwd, {total_time:.3f}s, "
                  f"{total_time/total_fwd*1000:.2f} ms/fwd")

            raw_stats = timer.summarize()
            result = analyze(raw_stats, config_name, total_time, total_fwd, eb_ctrl)
            results[config_name] = result

            # Save individual
            out_path = (REPO_ROOT / "codex_coding" / "results" /
                        f"b128_{config_name.lower()}_profiling.json")
            save_result = {k: v for k, v in result.items()}
            save_result['raw_stats_summary'] = raw_stats
            with open(out_path, "w") as f:
                json.dump(save_result, f, indent=2, default=str)
            print(f"  Saved to {out_path}")

            # Cleanup hooks
            for h in hooks:
                h.remove()

        # ============================================================
        # Cross-comparison
        # ============================================================
        if "C5" in results and "C10" in results:
            c5 = results["C5"]
            c10 = results["C10"]
            print(f"\n{'='*80}")
            print(f"CROSS-COMPARISON: C5 vs C10-M5 (batch={BATCH_SIZE})")
            print(f"{'='*80}")
            print(f"  {'':>25s} {'C5':>12s} {'C10-M5':>12s} {'Delta':>10s}")
            print(f"  {'-'*60}")
            for key in ['fused_experts_derived', 'EB_total', 'routing',
                        'shared_experts', 'gate_getlogits', 'Attention',
                        'fwd_impl', 'MoE_subtotal', 'non_MoE_subtotal',
                        'instrumented_total', 'gap']:
                v5 = c5['components'].get(key, 0)
                v10 = c10['components'].get(key, 0)
                d = v10 - v5
                dp = d / v5 * 100 if v5 > 0 else 0
                print(f"  {key:>25s} {v5:>10.1f}ms {v10:>10.1f}ms "
                      f"{d:>+8.1f}ms ({dp:>+5.1f}%)")

            fe5 = c5['components']['fused_experts_derived']
            fe10 = c10['components']['fused_experts_derived']
            eb10 = c10['components']['EB_total']
            fe_saving = fe5 - fe10
            print(f"\n  fused_experts saving: {fe_saving:.1f}ms ({fe_saving/fe5*100:.1f}%)")
            print(f"  EB overhead:          {eb10:.1f}ms")
            print(f"  Net benefit:          {fe_saving - eb10:.1f}ms")

        print("\nDone.")


if __name__ == "__main__":
    main()
