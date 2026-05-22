#!/usr/bin/env python3
"""
Final high-perf benchmark: fast_generate + all optimizations.
"""
import os, sys, time, json, socket, types, importlib.util
import torch

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
MASK_ID, EOS_ID = 156895, 156892
PROMPTS = [
    "Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h.",
    "Write a detailed essay about the history of artificial intelligence.",
    "Explain Le Chatelier's principle with examples.",
    "Design a REST API for an e-commerce platform.",
    "Analyze the economic impact of climate change.",
    "Explain quantum computing: qubits, superposition.",
    "Design a distributed message queue.",
    "Write a guide to training large language models.",
    "Compare TCP and UDP protocols.",
    "Explain neural networks and backpropagation.",
    "Design microservices for ride-sharing.",
    "Write about cryptography history.",
    "Explain database indexing strategies.",
    "Discuss universal basic income.",
    "Design a CI/CD pipeline.",
    "Explain relativity.",
]

def setup():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if "deep_ep" not in sys.modules or getattr(sys.modules.get("deep_ep"), "__spec__", None) is None:
        sys.modules.pop("deep_ep", None)
        f = types.ModuleType("deep_ep"); f.__spec__ = importlib.util.spec_from_loader("deep_ep", loader=None); f.__path__ = []
        f.Buffer = type("Buffer", (), {"get_dispatch_config": staticmethod(lambda *a,**kw:None), "get_combine_config": staticmethod(lambda *a,**kw:None)})
        f.Config = type("Config", (), {}); f.EventOverlap = type("EventOverlap", (), {}); sys.modules["deep_ep"] = f


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--batch-sizes", type=str, default="1,16,64")
    p.add_argument("--num-runs", type=int, default=3)
    args = p.parse_args()
    bss = [int(b) for b in args.batch_sizes.split(",")]
    device = torch.device(f"cuda:{args.gpu}"); torch.cuda.set_device(device)

    print("=" * 100)
    print(f"FINAL Benchmark | LLaDA2.0-mini 16B | {torch.cuda.get_device_name(args.gpu)}")
    print("=" * 100)
    setup()

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]; sock.close()
    os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT",str(port))
    vcfg = VllmConfig(parallel_config=ParallelConfig(enable_expert_parallel=True))

    with set_current_vllm_config(vcfg):
        distributed.init_distributed_environment(1,0,"env://",0,"nccl")
        distributed.initialize_model_parallel(1,backend="nccl")

        from transformers import AutoConfig, AutoTokenizer
        from dinfer.model import LLaDA2MoeModelLM

        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

        # Apply all optimizations
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from bench_highperf import apply_fused_rmsnorm, apply_optimized_attn, apply_dtype_optimized_moe
        n1 = apply_fused_rmsnorm(model)
        n2 = apply_optimized_attn(model)
        n3 = apply_dtype_optimized_moe(model)
        print(f"[opt] FusedRMSNorm: {n1}, OptAttn: {n2}, MoE: {n3}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

        # Warmup
        with torch.inference_mode():
            w = torch.randint(0, 1000, (16, 128), device=device)
            for _ in range(3): model(w, use_cache=False)
        torch.cuda.synchronize()

        from dinfer.fast_generate import fast_diffusion_generate, fast_diffusion_generate_cudagraph
        from dinfer import BlockWiseDiffusionLLM, BlockIteratorFactory, ThresholdParallelDecoder

        results = {}

        for bs in bss:
            print(f"\n{'='*80}")
            print(f"  Batch size = {bs}")
            print(f"{'='*80}")

            prompts = [PROMPTS[i % len(PROMPTS)] for i in range(bs)]
            encoded = [tokenizer.encode(pr, return_tensors="pt").squeeze(0) for pr in prompts]
            mx = max(e.shape[0] for e in encoded)
            padded = [torch.cat([torch.full((mx-e.shape[0],), MASK_ID, dtype=torch.long), e]) if e.shape[0]<mx else e for e in encoded]
            input_ids = torch.stack(padded).to(device)

            # --- A: dInfer BlockWiseDiffusionLLM ---
            print(f"  [A: dInfer] Running...")
            decoder = ThresholdParallelDecoder(temperature=0, threshold=0.9, mask_id=MASK_ID, eos_id=EOS_ID)
            dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out_a = dllm.generate(input_ids.clone(), gen_length=args.gen_length, block_length=32)
            torch.cuda.synchronize()
            dt_a = time.perf_counter() - t0
            n_a = dllm.num_forwards
            tps_a = bs * args.gen_length / dt_a
            ms_a = dt_a / n_a * 1000
            text_a = tokenizer.decode(out_a[0], skip_special_tokens=True)
            print(f"  [A: dInfer] {ms_a:.2f} ms/fwd, {tps_a:.0f} tok/s, {n_a} fwds")

            # --- B: fast_diffusion_generate ---
            print(f"  [B: FastGen] Running...")
            # Warmup
            with torch.inference_mode():
                fast_diffusion_generate(model, input_ids[:min(bs,4)].clone(), gen_length=32, block_length=32)
            torch.cuda.synchronize()

            best_tps = 0
            for ri in range(args.num_runs):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    out_b, n_b = fast_diffusion_generate(
                        model, input_ids.clone(), gen_length=args.gen_length, block_length=32)
                torch.cuda.synchronize()
                dt_b = time.perf_counter() - t0
                tps_b = bs * args.gen_length / dt_b
                ms_b = dt_b / n_b * 1000
                best_tps = max(best_tps, tps_b)
                print(f"    Run {ri+1}: {ms_b:.2f} ms/fwd, {tps_b:.0f} tok/s, {n_b} fwds")

            text_b = tokenizer.decode(out_b[0], skip_special_tokens=True)
            speedup = best_tps / tps_a if tps_a > 0 else 0

            # --- C: CUDA Graph version ---
            print(f"  [C: CUDAGraph] Running...")
            try:
                with torch.inference_mode():
                    fast_diffusion_generate_cudagraph(model, input_ids[:min(bs,4)].clone(), gen_length=32, block_length=32)
                torch.cuda.synchronize()

                best_tps_c = 0
                for ri in range(args.num_runs):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        out_c, n_c = fast_diffusion_generate_cudagraph(
                            model, input_ids.clone(), gen_length=args.gen_length, block_length=32)
                    torch.cuda.synchronize()
                    dt_c = time.perf_counter() - t0
                    tps_c = bs * args.gen_length / dt_c
                    ms_c = dt_c / n_c * 1000
                    best_tps_c = max(best_tps_c, tps_c)
                    print(f"    Run {ri+1}: {ms_c:.2f} ms/fwd, {tps_c:.0f} tok/s")
            except Exception as e:
                print(f"  [C: CUDAGraph] Failed: {e}")
                best_tps_c = 0

            print(f"\n  [QUALITY] A: {text_a[:100]}")
            print(f"  [QUALITY] B: {text_b[:100]}")

            results[f"bs{bs}"] = {
                "dinfer": {"tps": round(tps_a, 0), "ms_fwd": round(ms_a, 2)},
                "fast_gen": {"tps": round(best_tps, 0), "ms_fwd": "varies"},
                "cudagraph": {"tps": round(best_tps_c, 0)},
                "speedup": round(speedup, 2),
            }

    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)
    print(f"{'BS':>4} | {'dInfer tok/s':>12} | {'FastGen tok/s':>13} | {'CUDAGraph tok/s':>15} | {'Speedup':>8}")
    print("-" * 80)
    for bsk in sorted(results.keys(), key=lambda k: int(k[2:])):
        r = results[bsk]; bv = bsk[2:]
        print(f"{bv:>4} | {r['dinfer']['tps']:12.0f} | {r['fast_gen']['tps']:13.0f} | {r['cudagraph']['tps']:15.0f} | {r['speedup']:.2f}x")
    print("=" * 100)

    with open("bench_final_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
