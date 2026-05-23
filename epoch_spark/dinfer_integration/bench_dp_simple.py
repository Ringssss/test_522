#!/usr/bin/env python3
"""
Multi-GPU DP: launch independent workers, each on its own GPU.
No TP communication — pure data parallel throughput.

Usage:
    cd /home/zhujianian/eurosys/dInfer
    python tests/bench_dp_simple.py --n-gpu 2 --batch 16
    python tests/bench_dp_simple.py --n-gpu 4 --batch 64
"""
import os, sys, time, socket, types, importlib.util
import torch
import torch.multiprocessing as mp

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
MASK_ID = 156895
PROMPTS = ["Please solve problems step by step. Problem 1: A train travels from City A to B at 80 km/h.","Write an essay about AI history.","Explain Le Chatelier's principle.","Design a REST API.","Analyze climate change impact.","Explain quantum computing.","Design a message queue.","Write about training LLMs.","Compare TCP and UDP.","Explain neural networks.","Design ride-sharing microservices.","Write about cryptography.","Explain database indexing.","Discuss universal basic income.","Design CI/CD pipeline.","Explain relativity."]


def worker(rank, world_size, batch_total, gen_length, num_runs, result_queue):
    """Single GPU worker."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # deep_ep stub
    if "deep_ep" not in sys.modules:
        f = types.ModuleType("deep_ep"); f.__spec__ = importlib.util.spec_from_loader("deep_ep", loader=None); f.__path__ = []
        f.Buffer = type("Buffer", (), {"get_dispatch_config": staticmethod(lambda *a, **kw: None), "get_combine_config": staticmethod(lambda *a, **kw: None)})
        f.Config = type("Config", (), {}); f.EventOverlap = type("EventOverlap", (), {}); sys.modules["deep_ep"] = f

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

    # Each worker gets its own distributed env on its own GPU
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    vcfg = VllmConfig(parallel_config=ParallelConfig(enable_expert_parallel=True))
    with set_current_vllm_config(vcfg):
        distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
        distributed.initialize_model_parallel(1, backend="nccl")

        from transformers import AutoConfig, AutoTokenizer
        from dinfer.model import LLaDA2MoeModelLM
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from bench_highperf import apply_fused_rmsnorm, apply_optimized_attn, apply_dtype_optimized_moe

        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        apply_fused_rmsnorm(model)
        apply_optimized_attn(model)
        apply_dtype_optimized_moe(model)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

        # Each worker handles batch/world_size
        local_bs = batch_total // world_size
        start_idx = rank * local_bs
        prompts = [PROMPTS[(start_idx + i) % len(PROMPTS)] for i in range(local_bs)]
        encoded = [tokenizer.encode(pr, return_tensors="pt").squeeze(0) for pr in prompts]
        mx = max(e.shape[0] for e in encoded)
        padded = [torch.cat([torch.full((mx-e.shape[0],), MASK_ID, dtype=torch.long), e]) if e.shape[0] < mx else e for e in encoded]
        input_ids = torch.stack(padded).to(device)

        from dinfer.fast_generate import fast_generate_with_kvcache_cudagraph

        # Warmup
        with torch.inference_mode():
            for _ in range(3):
                fast_generate_with_kvcache_cudagraph(model, input_ids, gen_length=32, block_length=32)
            fast_generate_with_kvcache_cudagraph(model, input_ids, gen_length=gen_length, block_length=32)
        torch.cuda.synchronize()

        # Timed
        best_time = float('inf')
        for ri in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out, n = fast_generate_with_kvcache_cudagraph(
                    model, input_ids.clone(), gen_length=gen_length, block_length=32
                )
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            best_time = min(best_time, dt)

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        result_queue.put((rank, best_time, n, text[:100]))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-gpu", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--num-runs", type=int, default=5)
    args = p.parse_args()

    print("=" * 90)
    print(f"DP={args.n_gpu} Simple Benchmark | batch={args.batch} | gen={args.gen_length}")
    print("=" * 90)

    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()

    processes = []
    for rank in range(args.n_gpu):
        p_worker = mp.Process(target=worker, args=(rank, args.n_gpu, args.batch, args.gen_length, args.num_runs, result_queue))
        p_worker.start()
        processes.append(p_worker)

    for p_worker in processes:
        p_worker.join()

    # Collect results
    results = {}
    while not result_queue.empty():
        rank, best_time, n_fwd, text = result_queue.get()
        results[rank] = {"time": best_time, "n_fwd": n_fwd, "text": text}

    if results:
        # All workers ran same duration (approx), throughput = total_tokens / max_time
        max_time = max(r["time"] for r in results.values())
        total_tokens = args.batch * args.gen_length
        tps = total_tokens / max_time
        local_bs = args.batch // args.n_gpu
        per_gpu_tps = (local_bs * args.gen_length) / max_time

        print(f"\n{'='*90}")
        print(f"RESULT: DP={args.n_gpu}, batch={args.batch}")
        print(f"  Total throughput: {tps:.0f} tok/s")
        print(f"  Per-GPU throughput: {per_gpu_tps:.0f} tok/s (local_bs={local_bs})")
        print(f"  Best time: {max_time:.3f}s, {results[0]['n_fwd']} fwds")
        print(f"  Output: {results[0]['text']}")
        print(f"{'='*90}")


if __name__ == "__main__":
    main()
