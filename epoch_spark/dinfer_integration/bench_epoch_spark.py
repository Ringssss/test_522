#!/usr/bin/env python3
"""Epoch-Spark v3 benchmark. Single model load, baseline first then offload."""
import os, sys, time, json, socket, types, importlib.util
import torch

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
MASK_ID, EOS_ID = 156895, 156892
PROMPTS = [
    "Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h.",
    "Write an essay about the history of artificial intelligence.",
    "Explain Le Chatelier's principle with examples.",
    "Design a REST API for an e-commerce platform.",
    "Analyze the economic impact of climate change.",
    "Explain quantum computing: qubits, superposition.",
    "Design a distributed message queue.",
    "Write a guide to training large language models.",
]

def setup():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if "deep_ep" not in sys.modules or getattr(sys.modules.get("deep_ep"), "__spec__", None) is None:
        sys.modules.pop("deep_ep", None)
        f = types.ModuleType("deep_ep"); f.__spec__ = importlib.util.spec_from_loader("deep_ep", loader=None); f.__path__ = []
        f.Buffer = type("Buffer", (), {"get_dispatch_config": staticmethod(lambda *a, **kw: None), "get_combine_config": staticmethod(lambda *a, **kw: None)})
        f.Config = type("Config", (), {}); f.EventOverlap = type("EventOverlap", (), {}); sys.modules["deep_ep"] = f

def prepare_batch(tokenizer, bs, device):
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(bs)]
    encoded = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in prompts]
    mx = max(e.shape[0] for e in encoded)
    padded = [torch.cat([torch.full((mx-e.shape[0],), MASK_ID, dtype=torch.long), e]) if e.shape[0]<mx else e for e in encoded]
    return torch.stack(padded).to(device)

def run_gen(model, tokenizer, ids, bs, gl, device, ctrl=None):
    from dinfer import BlockIteratorFactory, ThresholdParallelDecoder, BlockDiffusionLLMAttnmask, BlockWiseDiffusionLLM
    dec = ThresholdParallelDecoder(temperature=0, threshold=0.9, mask_id=MASK_ID, eos_id=EOS_ID)
    dllm = (BlockDiffusionLLMAttnmask(model, dec, BlockIteratorFactory(use_block_diffusion=True), early_stop=True) if bs == 1
            else BlockWiseDiffusionLLM(model, dec, BlockIteratorFactory(), early_stop=True))
    if ctrl:
        obi = dllm.decoder.block_init; ofwd = dllm.diff_iteration.forward
        def hbi(b, bid, _c=ctrl, _o=obi):
            if _c.current_block_id >= 0: _c.on_block_end()
            _c.on_block_start(bid, ids[0]); return _o(b, bid)
        def hf(*a, _c=ctrl, _o=ofwd, **kw):
            x = a[2] if len(a) > 2 else None
            if x and hasattr(x, 'data'): _c.on_iter_start(x.data[0] if x.data.dim() > 1 else x.data)
            return _o(*a, **kw)
        dllm.decoder.block_init = hbi; dllm.diff_iteration.forward = hf
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = dllm.generate(ids, gen_length=gl, block_length=32)
    if ctrl: ctrl.on_block_end()
    torch.cuda.synchronize()
    t = time.perf_counter() - t0; n = dllm.num_forwards; pk = torch.cuda.max_memory_allocated(device)/1024**2
    texts = [tokenizer.decode(out[i], skip_special_tokens=True) for i in range(bs)]
    return t, n, pk, texts

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--batch-sizes", type=str, default="1,16,64")
    p.add_argument("--budget", type=int, default=80)
    args = p.parse_args()
    bss = [int(b) for b in args.batch_sizes.split(",")]
    device = torch.device(f"cuda:{args.gpu}"); torch.cuda.set_device(device)
    print("="*100)
    print(f"Epoch-Spark v3 | LLaDA2.0-mini 16B | {torch.cuda.get_device_name(args.gpu)} | budget={args.budget}")
    print("="*100)
    setup()
    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer.epoch_spark_dinfer import EpochSparkController, patch_dinfer_model, patch_dinfer_baseline
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]; sock.close()
    os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT",str(port))
    vcfg = VllmConfig(parallel_config=ParallelConfig(enable_expert_parallel=True))
    results = {}
    with set_current_vllm_config(vcfg):
        if not torch.distributed.is_initialized():
            distributed.init_distributed_environment(1,0,"env://",0,"nccl")
            distributed.initialize_model_parallel(1,backend="nccl")
        from transformers import AutoTokenizer, AutoConfig
        from dinfer.model import LLaDA2MoeModelLM
        tk = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        model = LLaDA2MoeModelLM(config=cfg).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        print(f"[model] GPU: {torch.cuda.memory_allocated(device)/1024**2:.0f} MB")

        # ═══ PHASE 1: All baseline runs (before offload frees weights) ═══
        patch_dinfer_baseline(model)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)
        base_results = {}
        for bs in bss:
            ids = prepare_batch(tk, bs, device)
            print(f"\n  [BASELINE bs={bs}] Running...")
            t, n, pk, texts = run_gen(model, tk, ids, bs, args.gen_length, device)
            avg = t/n*1000; tps = bs*args.gen_length/t
            print(f"  [BASELINE bs={bs}] {avg:.2f} ms/fwd, {tps:.1f} tok/s, peak={pk:.0f} MB")
            print(f"  [BASELINE bs={bs}] Output: {texts[0][:120]}")
            base_results[bs] = {"avg": avg, "tps": tps, "peak": pk, "texts": texts, "n": n}

        # ═══ PHASE 2: Offload runs (frees original expert weights) ═══
        ctrl = EpochSparkController(mask_id=MASK_ID, gpu_budget=args.budget, enable_offload=True)
        patch_dinfer_model(model, ctrl)
        gpu_after = torch.cuda.memory_allocated(device)/1024**2
        print(f"\n[offload] GPU after: {gpu_after:.0f} MB (saved {base_results[bss[0]]['peak']-gpu_after:.0f} MB)")
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)
        off_results = {}
        for bs in bss:
            # Reset controller stats
            ctrl.stats.clear(); ctrl.moe_cache.clear(); ctrl.cache_populated.clear()
            ctrl.current_block_id = -1; ctrl.block_iter_count = 0
            for buf in ctrl.buffers.values(): buf.stats.clear()
            ids = prepare_batch(tk, bs, device)
            print(f"\n  [OFFLOAD bs={bs}] Running...")
            t, n, pk, texts = run_gen(model, tk, ids, bs, args.gen_length, device, ctrl)
            avg = t/n*1000; tps = bs*args.gen_length/t
            s = ctrl.get_summary()
            speedup = base_results[bs]["avg"]/avg if avg > 0 else 0
            match = sum(1 for a, b in zip(base_results[bs]["texts"], texts) if a == b)
            print(f"  [OFFLOAD bs={bs}] {avg:.2f} ms/fwd, {tps:.1f} tok/s, speedup={speedup:.2f}x, peak={pk:.0f} MB")
            print(f"  [OFFLOAD bs={bs}] cached={s['cached_tokens']}, miss={s['miss_tokens']}, swaps={s['buffer_swaps']}")
            print(f"  [OFFLOAD bs={bs}] Output: {texts[0][:120]}")
            print(f"  [QUALITY bs={bs}] Match: {match}/{bs}")
            off_results[bs] = {"avg": avg, "tps": tps, "peak": pk, "speedup": speedup, "match": f"{match}/{bs}", **s}

    # ═══ FINAL TABLE ═══
    print("\n" + "="*110)
    print("FINAL RESULTS — Epoch-Spark v3 (Pingpong Double-Buffer, Zero CPU Compute)")
    print("="*110)
    print(f"{'BS':>4} | {'Mode':<12} | {'ms/fwd':>8} | {'Speedup':>8} | {'tok/s':>10} | {'PeakMB':>8} | {'Cached':>8} | {'Miss':>8} | {'Match':>7}")
    print("-"*110)
    for bs in bss:
        b = base_results[bs]; e = off_results[bs]
        print(f"{bs:>4} | {'Baseline':<12} | {b['avg']:8.2f} | {'1.00x':>8} | {b['tps']:10.1f} | {b['peak']:8.0f} | {'—':>8} | {'—':>8} | {'—':>7}")
        print(f"{bs:>4} | {'Offload':<12} | {e['avg']:8.2f} | {e['speedup']:.2f}x   | {e['tps']:10.1f} | {e['peak']:8.0f} | {e['cached_tokens']:>8} | {e['miss_tokens']:>8} | {e['match']:>7}")
    print("="*110)
    all_r = {f"bs{bs}": {"baseline": {"avg_ms": round(base_results[bs]["avg"],2), "tps": round(base_results[bs]["tps"],1), "peak_mb": round(base_results[bs]["peak"],0)},
                          "offload": {"avg_ms": round(off_results[bs]["avg"],2), "tps": round(off_results[bs]["tps"],1), "peak_mb": round(off_results[bs]["peak"],0), "speedup": round(off_results[bs]["speedup"],2), "match": off_results[bs]["match"]}} for bs in bss}
    with open("bench_epoch_spark_results.json","w") as f: json.dump(all_r, f, indent=2)
    print(f"\nSaved to bench_epoch_spark_results.json")

if __name__ == "__main__":
    main()
