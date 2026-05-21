#!/usr/bin/env python3
"""
Epoch-Spark dInfer Integration Benchmark.

Compares dInfer baseline vs Epoch-Spark patched at bs=1, 16, 64.
Uses dInfer's native generation pipeline.

Usage:
    cd /home/zhujianian/eurosys/dInfer
    /home/zhujianian/miniconda3/envs/crossstage/bin/python tests/bench_epoch_spark.py
"""
import os, sys, time, json, socket
import types, importlib.util
import torch
import numpy as np

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
MASK_ID = 156895
EOS_ID = 156892

PROMPTS = [
    "Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h. The total distance between the two cities is 240 km. What is the average speed for the entire round trip?",
    "Write a detailed essay about the history of artificial intelligence, covering the Dartmouth conference of 1956, the AI winters, the rise of machine learning in the 1990s, and deep learning breakthroughs.",
    "You are a chemistry professor. Explain Le Chatelier's principle with examples.",
    "Design a complete REST API for an e-commerce platform with endpoints for user authentication and product management.",
    "Analyze the global economic impact of climate change across agriculture, energy, real estate sectors.",
    "Explain quantum computing to a classical CS background: qubits, superposition, entanglement.",
    "Design a distributed message queue with partition-based storage and consumer groups.",
    "Write a guide to training large language models covering data collection and architecture decisions.",
]


def setup():
    """One-time env setup: deep_ep stub + distributed init."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # deep_ep stub
    if "deep_ep" not in sys.modules or getattr(sys.modules.get("deep_ep"), "__spec__", None) is None:
        sys.modules.pop("deep_ep", None)
        _fake = types.ModuleType("deep_ep")
        _fake.__spec__ = importlib.util.spec_from_loader("deep_ep", loader=None)
        _fake.__path__ = []
        _fake.Buffer = type("Buffer", (), {
            "get_dispatch_config": staticmethod(lambda *a, **kw: None),
            "get_combine_config": staticmethod(lambda *a, **kw: None),
        })
        _fake.Config = type("Config", (), {})
        _fake.EventOverlap = type("EventOverlap", (), {})
        sys.modules["deep_ep"] = _fake


def run_all(gpu_id, batch_sizes, gen_length):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from transformers import AutoTokenizer, AutoConfig
    from dinfer.model import LLaDA2MoeModelLM
    from dinfer import (BlockIteratorFactory, ThresholdParallelDecoder,
                        BlockDiffusionLLMAttnmask, BlockWiseDiffusionLLM)
    from dinfer.epoch_spark_dinfer import (EpochSparkController, patch_dinfer_model,
                                          unpatch_dinfer_model, patch_dinfer_baseline)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))

    pcfg = ParallelConfig(enable_expert_parallel=True)
    vcfg = VllmConfig(parallel_config=pcfg)

    with set_current_vllm_config(vcfg):
        if not torch.distributed.is_initialized():
            distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
            distributed.initialize_model_parallel(1, backend="nccl")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        print(f"[model] GPU mem: {torch.cuda.memory_allocated(device)/1024**2:.0f} MB")

        # Patch with fused Triton forward (bypasses vLLM forward_context requirement)
        patch_dinfer_baseline(model)

        # Warmup
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)

        all_results = {}

        for bs in batch_sizes:
            print(f"\n{'='*70}")
            print(f"  Batch size = {bs}")
            print(f"{'='*70}")

            prompts = [PROMPTS[i % len(PROMPTS)] for i in range(bs)]
            encoded = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in prompts]
            max_len = max(e.shape[0] for e in encoded)
            padded = []
            for e in encoded:
                if e.shape[0] < max_len:
                    pad = torch.full((max_len - e.shape[0],), MASK_ID, dtype=torch.long)
                    padded.append(torch.cat([pad, e]))
                else:
                    padded.append(e)
            input_ids = torch.stack(padded).to(device)

            decoder = ThresholdParallelDecoder(temperature=0, threshold=0.9, mask_id=MASK_ID, eos_id=EOS_ID)

            def make_dllm():
                if bs == 1:
                    return BlockDiffusionLLMAttnmask(
                        model, decoder, BlockIteratorFactory(use_block_diffusion=True), early_stop=True)
                else:
                    return BlockWiseDiffusionLLM(
                        model, decoder, BlockIteratorFactory(), early_stop=True)

            # --- Baseline ---
            print(f"  [BASELINE] Running...")
            dllm = make_dllm()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                out_base = dllm.generate(input_ids, gen_length=gen_length, block_length=32)
            torch.cuda.synchronize()
            t_base = time.perf_counter() - t0
            n_fwd_base = dllm.num_forwards
            peak_base = torch.cuda.max_memory_allocated(device) / 1024**2
            avg_base = t_base / n_fwd_base * 1000
            tps_base = bs * gen_length / t_base
            base_texts = [tokenizer.decode(out_base[i], skip_special_tokens=True) for i in range(bs)]
            print(f"  [BASELINE] {avg_base:.2f} ms/fwd, {tps_base:.1f} tok/s, "
                  f"{n_fwd_base} fwds, peak={peak_base:.0f} MB")
            print(f"  [BASELINE] Output: {base_texts[0][:150]}")

            # --- Epoch-Spark ---
            print(f"  [EPOCH-SPARK] Running...")
            controller = EpochSparkController(mask_id=MASK_ID, refresh_m=5)
            patch_dinfer_model(model, controller)

            dllm2 = make_dllm()

            # Hook controller into iteration loop
            orig_block_init = dllm2.decoder.block_init
            def hooked_block_init(block, block_id):
                controller.on_block_start(block_id, input_ids[0])
                return orig_block_init(block, block_id)
            dllm2.decoder.block_init = hooked_block_init

            orig_fwd = dllm2.diff_iteration.forward
            def hooked_fwd(*args, **kwargs):
                x = args[2] if len(args) > 2 else None
                if x is not None and hasattr(x, 'data'):
                    controller.on_iter_start(x.data[0] if x.data.dim() > 1 else x.data)
                result = orig_fwd(*args, **kwargs)
                controller.on_iter_end()
                return result
            dllm2.diff_iteration.forward = hooked_fwd

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                out_es = dllm2.generate(input_ids, gen_length=gen_length, block_length=32)
            torch.cuda.synchronize()
            t_es = time.perf_counter() - t0
            n_fwd_es = dllm2.num_forwards
            peak_es = torch.cuda.max_memory_allocated(device) / 1024**2
            avg_es = t_es / n_fwd_es * 1000
            tps_es = bs * gen_length / t_es
            speedup = avg_base / avg_es if avg_es > 0 else 0
            summary = controller.get_summary()
            es_texts = [tokenizer.decode(out_es[i], skip_special_tokens=True) for i in range(bs)]
            print(f"  [EPOCH-SPARK] {avg_es:.2f} ms/fwd, {tps_es:.1f} tok/s, "
                  f"{n_fwd_es} fwds, peak={peak_es:.0f} MB")
            print(f"  [EPOCH-SPARK] speedup={speedup:.2f}x, cached={summary['cached_pct']:.1f}%")
            print(f"  [EPOCH-SPARK] Output: {es_texts[0][:150]}")

            match = sum(1 for a, b in zip(base_texts, es_texts) if a == b)
            print(f"  [QUALITY] Exact text match: {match}/{bs}")

            unpatch_dinfer_model(model)

            all_results[f"bs{bs}"] = {
                "baseline": {"avg_ms": round(avg_base, 2), "tps": round(tps_base, 1),
                             "n_fwd": n_fwd_base, "peak_mb": round(peak_base, 0)},
                "epoch_spark": {"avg_ms": round(avg_es, 2), "tps": round(tps_es, 1),
                                "n_fwd": n_fwd_es, "peak_mb": round(peak_es, 0),
                                "speedup": round(speedup, 2), "cached_pct": round(summary["cached_pct"], 1)},
                "quality": {"exact_match": f"{match}/{bs}"},
                "sample_base": base_texts[0][:300],
                "sample_es": es_texts[0][:300],
            }

    # Final table
    print("\n" + "=" * 100)
    print("FINAL RESULTS (dInfer native pipeline)")
    print("=" * 100)
    print(f"{'BS':>4} | {'Mode':<14} | {'ms/fwd':>8} | {'Speedup':>8} | {'tok/s':>10} | "
          f"{'PeakMB':>8} | {'Cached%':>8} | {'Match':>8}")
    print("-" * 100)
    for bsk in sorted(all_results.keys(), key=lambda k: int(k[2:])):
        r = all_results[bsk]
        bv = bsk[2:]
        b, e, q = r["baseline"], r["epoch_spark"], r["quality"]
        print(f"{bv:>4} | {'Baseline':<14} | {b['avg_ms']:8.2f} | {'1.00x':>8} | "
              f"{b['tps']:10.1f} | {b['peak_mb']:8.0f} | {'—':>8} | {'—':>8}")
        print(f"{bv:>4} | {'Epoch-Spark':<14} | {e['avg_ms']:8.2f} | {e['speedup']:.2f}x"
              f"   | {e['tps']:10.1f} | {e['peak_mb']:8.0f} | "
              f"{e['cached_pct']:7.1f}% | {q['exact_match']:>8}")
    print("=" * 100)

    with open("bench_epoch_spark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to bench_epoch_spark_results.json")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--batch-sizes", type=str, default="1,16,64")
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    print("=" * 90)
    print("Epoch-Spark dInfer Integration Benchmark")
    print(f"  Model: LLaDA2.0-mini (16B, 256 experts)")
    print(f"  GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"  gen_length={args.gen_length}, batch_sizes={batch_sizes}")
    print("=" * 90)

    setup()
    run_all(args.gpu, batch_sizes, args.gen_length)


if __name__ == "__main__":
    main()
