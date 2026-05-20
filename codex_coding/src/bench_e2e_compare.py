#!/usr/bin/env python3
"""
E2E comparison: baseline dInfer vs optimized dInfer (BSP-G + OPT-2 + SP-LM).

Usage:
  # Baseline (original dInfer, no TP, pure EP)
  EVAL_MODE=baseline torchrun --nproc_per_node=8 bench_e2e_compare.py

  # Optimized (our dInfer, BSP-G + OPT-2 + SP-LM)
  EVAL_MODE=optimized torchrun --nproc_per_node=8 bench_e2e_compare.py
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
MASK_ID = 156895
EOS_ID = 156892
BLOCK_LENGTH = 32
BATCH_SIZE = 512
GEN_LENGTH = 256
NUM_RUNS = 2


def main():
    mode = os.environ.get("EVAL_MODE", "optimized")
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    from test_heteval512 import PROMPTS
    from transformers import AutoConfig, AutoTokenizer

    if mode == "baseline":
        _run_baseline(rank, world_size, local_rank, device, PROMPTS)
    else:
        _run_optimized(rank, world_size, local_rank, device, PROMPTS)


def _run_baseline(rank, world_size, local_rank, device, prompts):
    """Baseline: original baseline_dInfer code, no TP, pure EP=8."""

    # ---- Patch transformers compat BEFORE importing baseline ----
    import transformers.utils.import_utils as _tiu
    if not hasattr(_tiu, 'is_torch_fx_available'):
        _tiu.is_torch_fx_available = lambda: hasattr(torch, "fx")

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if 'default' not in ROPE_INIT_FUNCTIONS:
        def _default_rope_init(config, device=None, **kwargs):
            dim = config.hidden_size // config.num_attention_heads
            rs = getattr(config, 'rope_scaling', {}) or {}
            base = rs.get('rope_theta', getattr(config, 'rope_theta', 10000))
            partial = rs.get('partial_rotary_factor', 1.0)
            rdim = int(dim * partial)
            inv_freq = 1.0 / (base ** (torch.arange(0, rdim, 2, dtype=torch.int64).float() / rdim))
            if device is not None:
                inv_freq = inv_freq.to(device)
            return inv_freq, 1.0
        ROPE_INIT_FUNCTIONS['default'] = _default_rope_init

    # ---- Use baseline dInfer ----
    baseline_path = str(REPO_ROOT / "lib_cite" / "baseline_dInfer" / "python")
    sys.path.insert(0, baseline_path)
    for mod_name in list(sys.modules.keys()):
        if 'dinfer' in mod_name:
            del sys.modules[mod_name]

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    dp_size = world_size
    dp_rank = rank
    pcfg_init = ParallelConfig(
        tensor_parallel_size=1, data_parallel_size=1,
        enable_expert_parallel=True,
    )
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg_init)):
        vllm_dist.init_distributed_environment(
            world_size, rank, "env://", local_rank, "nccl"
        )

    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=dp_size,
        data_parallel_rank=dp_rank,
        enable_expert_parallel=True,
    )
    vllm_cfg = VllmConfig(parallel_config=parallel_config)
    with set_current_vllm_config(vllm_cfg):
        vllm_dist.initialize_model_parallel(1, backend="nccl")

    from dinfer import (
        BlockDiffusionLLM, BlockIteratorFactory,
        KVCacheFactory, ThresholdParallelDecoder,
    )
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoConfig, AutoTokenizer

    if rank == 0:
        print("=" * 70)
        print("BASELINE (original baseline_dInfer) — no TP, no BSP-G, no EB, pure EP=8")
        print(f"  batch={BATCH_SIZE}, gen={GEN_LENGTH}, block={BLOCK_LENGTH}")
        print("=" * 70)

    with set_current_vllm_config(vllm_cfg):
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        )
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        with torch.inference_mode():
            from vllm.forward_context import set_forward_context
            w = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                      num_tokens=w.numel()):
                _ = model(w, use_cache=False)

    if rank == 0:
        print(f"  Model loaded. GPU mem: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    decoder = ThresholdParallelDecoder(
        temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID,
    )

    local_bs = BATCH_SIZE // dp_size
    start_idx = dp_rank * local_bs
    encoded = [tokenizer(prompts[i % len(prompts)], return_tensors="pt")["input_ids"]
               for i in range(start_idx, start_idx + local_bs)]
    max_len = max(e.shape[1] for e in encoded)
    input_ids = torch.full((local_bs, max_len), MASK_ID, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        input_ids[i, :e.shape[1]] = e[0].to(device)

    def make_dllm():
        return BlockDiffusionLLM(
            model, decoder,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True,
            backend="vllm",
        )

    # Baseline needs forward_context for vllm 0.11 EP dispatch compat
    # Wrap model.forward to auto-set context (not an optimization, just compat)
    from vllm.forward_context import set_forward_context
    _orig_model_fwd = model.forward
    def _model_fwd_with_ctx(*args, **kwargs):
        import torch
        input_ids = args[0] if args else kwargs.get('input_ids')
        n = input_ids.numel() if input_ids is not None else 1
        with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg, num_tokens=n):
            return _orig_model_fwd(*args, **kwargs)
    model.forward = _model_fwd_with_ctx

    _run_and_report("baseline", make_dllm, input_ids, tokenizer, rank, device, vllm_cfg)


def _run_optimized(rank, world_size, local_rank, device, prompts):
    """Optimized dInfer: tp=4, dp=2, BSP-G, EB K=4, OPT-2, SP-LM."""
    # Set optimization env vars
    os.environ["DINF_SKIP_LOGITS_FLOAT"] = "1"
    os.environ["DINF_SP_LM_HEAD"] = "1"

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    tp_size = 4
    dp_size = world_size // tp_size
    dp_rank = rank // tp_size

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
        vllm_dist.initialize_model_parallel(tp_size, backend="nccl")

    from vllm.distributed import prepare_communication_buffer_for_model
    from vllm.forward_context import set_forward_context
    from dinfer import (
        BlockDiffusionLLM, BlockIteratorFactory,
        KVCacheFactory, ThresholdParallelDecoder,
    )
    from dinfer.model import LLaDA2MoeModelLM
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeSparseMoeBlock
    from transformers import AutoConfig, AutoTokenizer
    from baseline_optimizations import apply_all_optimizations
    from test_m_skip_sweep import MSkipEBController
    from test_fused_eb_triton import fused_routing, _kernel_A, _kernel_B_v3

    if rank == 0:
        print("=" * 70)
        print("OPTIMIZED dInfer — tp=4, dp=2, BSP-G, EB K=4, OPT-2, SP-LM")
        print(f"  batch={BATCH_SIZE}, gen={GEN_LENGTH}, block={BLOCK_LENGTH}")
        print("=" * 70)

    with set_current_vllm_config(vllm_cfg):
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        )
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        with torch.inference_mode():
            w = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
            with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                      num_tokens=w.numel()):
                _ = model(w, use_cache=False)

        apply_all_optimizations(model)
        prepare_communication_buffer_for_model(model)
        model.set_bsp_sequence_parallel_moe(True)

    if rank == 0:
        print(f"  Model loaded. GPU mem: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    decoder = ThresholdParallelDecoder(
        temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID,
    )

    # EB controller + routing patch
    class SimpleEBCtrl(MSkipEBController):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.current_block_id = -1
            self._last_block_id = {}
        def note_block_start(self, bid):
            self.current_block_id = int(bid)
        def get_s_mask(self, layer_idx, logits, bias):
            prev = self._last_block_id.get(layer_idx, -1)
            if prev != self.current_block_id:
                self._last_block_id[layer_idx] = self.current_block_id
                return self.cold_path(layer_idx, logits, bias)
            return self.hot_path(layer_idx, logits, bias)

    ctrl = SimpleEBCtrl(
        num_layers=19, K=8, M=4, K_target=40,
        quality_floor=0.70, q_major=1.0, per_round_cap=8, skip_m=5,
    )

    # Patch gate routing
    for _n, m in model.named_modules():
        if m.__class__.__name__ == "LLaDA2MoeGate":
            b, r, ng, tkg = m.expert_bias, m.routed_scaling_factor, m.n_group, m.topk_group
            li = sum(1 for _ in []) # need layer index
    # Simpler: use the bench script pattern
    gi = 0
    for _n, m in model.named_modules():
        if m.__class__.__name__ == "LLaDA2MoeGate":
            b = m.expert_bias
            r = m.routed_scaling_factor
            ng = m.n_group
            tkg = m.topk_group
            li = gi
            def mk(bb, rr, nn, gg, layer_i, cc):
                def fn(hs, go, topk, renorm):
                    sm = cc.get_s_mask(layer_i, go, bb)
                    w, idx = fused_routing(go, bb, rr, s_mask=sm, K=4, ng=nn, tkg=gg)
                    return w.to(go.dtype), idx
                return fn
            m.routing = mk(b, r, ng, tkg, li, ctrl)
            gi += 1

    # Hook block_init for EB block clock
    orig_block_init = decoder.block_init
    def block_init_with_clock(block_x, block_id):
        ctrl.note_block_start(int(block_id))
        return orig_block_init(block_x, block_id)
    decoder.block_init = block_init_with_clock

    # Build input (DP: each rank gets local_bs)
    local_bs = BATCH_SIZE // dp_size
    start_idx = dp_rank * local_bs
    encoded = [tokenizer(prompts[i % len(prompts)], return_tensors="pt")["input_ids"]
               for i in range(start_idx, start_idx + local_bs)]
    max_len = max(e.shape[1] for e in encoded)
    input_ids = torch.full((local_bs, max_len), MASK_ID, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        input_ids[i, :e.shape[1]] = e[0].to(device)

    def make_dllm():
        ctrl.current_block_id = -1
        ctrl._last_block_id.clear()
        ctrl.eb_calls = 0
        ctrl.eb_skips = 0
        ctrl._bufs.clear()
        ctrl.k_init_history.clear()
        ctrl.s_mask_cache.clear()
        ctrl.pop_cache.clear()
        ctrl._fwd_in_block.clear()
        ctrl._block_idx.clear()
        return BlockDiffusionLLM(
            model, decoder,
            BlockIteratorFactory(use_block_diffusion=True),
            cache_factory=KVCacheFactory("prefix", is_bd_model=True),
            early_stop=True,
            maximum_unroll=4,
            expected_tpf=15,
            backend="vllm",
            lazy_cache_update=True,
            inplace_cache_update=True,
        )

    _run_and_report("optimized", make_dllm, input_ids, tokenizer, rank, device, vllm_cfg)


def _run_and_report(mode, make_dllm, input_ids, tokenizer, rank, device, vllm_cfg):
    from vllm.config import set_current_vllm_config

    results = {
        "mode": mode,
        "batch_size": input_ids.shape[0],
        "gen_length": GEN_LENGTH,
        "block_length": BLOCK_LENGTH,
    }

    with set_current_vllm_config(vllm_cfg):
        # Warmup
        if rank == 0:
            print("  Warmup...")
        dllm = make_dllm()
        with torch.inference_mode():
            _ = dllm.generate(
                input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH
            )
        torch.cuda.synchronize()
        dist.barrier()

        # Timed runs
        run_times = []
        run_fwds = []
        for ri in range(NUM_RUNS):
            dllm = make_dllm()
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                out = dllm.generate(
                    input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH
                )
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            nfwd = dllm.diff_iteration.num_forwards
            run_times.append(dt)
            run_fwds.append(nfwd)
            if rank == 0:
                print(f"  Run {ri+1}: {dt:.3f}s, {nfwd} fwd, "
                      f"{dt*1000/nfwd:.2f} ms/fwd")

        # Quality snippet
        if out is not None and rank == 0:
            decoded_text = tokenizer.decode(out[0], skip_special_tokens=True)
            results["quality_snippet"] = decoded_text[:500]
            print(f"  Quality: {decoded_text[:200]}...")

    best_time = min(run_times)
    best_fwd = run_fwds[run_times.index(best_time)]
    results["best_time_s"] = best_time
    results["best_ms_per_fwd"] = best_time * 1000 / best_fwd
    results["num_forwards"] = best_fwd
    results["all_times_s"] = run_times

    if rank == 0:
        print(f"\n  RESULT: {best_time:.3f}s, {best_fwd} fwd, "
              f"{best_time*1000/best_fwd:.2f} ms/fwd")

        out_path = str(REPO_ROOT / "codex_coding" / "results" /
                       f"e2e_compare_{mode}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved to {out_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
