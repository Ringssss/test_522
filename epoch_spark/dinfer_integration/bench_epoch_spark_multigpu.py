#!/usr/bin/env python3
"""
Epoch-Spark Multi-GPU Benchmark (4×H100, TP=4).

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tests/bench_epoch_spark_multigpu.py
"""
import json, os, sys, time
import torch
import torch.distributed as dist

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
MASK_ID, EOS_ID = 156895, 156892
BLOCK_LENGTH = 32
PROMPTS = [
    "Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h.",
    "Write a detailed essay about the history of artificial intelligence.",
    "Explain Le Chatelier's principle with examples and industrial applications.",
    "Design a complete REST API for an e-commerce platform with authentication.",
    "Analyze the global economic impact of climate change across multiple sectors.",
    "Explain quantum computing: qubits, superposition, entanglement, Shor's algorithm.",
    "Design a distributed message queue with partitions, consumer groups, replication.",
    "Write a comprehensive guide to training large language models from scratch.",
    "Compare TCP and UDP protocols including use cases in modern distributed systems.",
    "Explain the mathematical foundations of neural networks and backpropagation.",
    "Design a microservices architecture for a ride-sharing application.",
    "Write about the history of cryptography from Caesar ciphers through RSA.",
    "Explain database indexing strategies and their trade-offs for OLTP vs OLAP.",
    "Discuss universal basic income with examples from pilot programs worldwide.",
    "Design a CI/CD pipeline for a large monorepo with microservices.",
    "Explain relativity to a physics undergraduate covering special and general.",
]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--num-runs", type=int, default=3)
    args = p.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # deep_ep stub
    import types, importlib.util
    if "deep_ep" not in sys.modules or getattr(sys.modules.get("deep_ep"), "__spec__", None) is None:
        sys.modules.pop("deep_ep", None)
        f = types.ModuleType("deep_ep"); f.__spec__ = importlib.util.spec_from_loader("deep_ep", loader=None); f.__path__ = []
        f.Buffer = type("Buffer", (), {"get_dispatch_config": staticmethod(lambda *a, **kw: None), "get_combine_config": staticmethod(lambda *a, **kw: None)})
        f.Config = type("Config", (), {}); f.EventOverlap = type("EventOverlap", (), {}); sys.modules["deep_ep"] = f

    tp_size = args.tp
    dp_size = world_size // tp_size
    dp_rank = rank // tp_size

    if rank == 0:
        print("=" * 90)
        print(f"Epoch-Spark Multi-GPU | {world_size}× {torch.cuda.get_device_name(0)}")
        print(f"TP={tp_size}, DP={dp_size}, batch={args.batch}, gen={args.gen_length}")
        print("=" * 90)

    from vllm import distributed as vllm_dist
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.config import (CompilationConfig, KernelConfig, DeviceConfig,
                             LoadConfig, SchedulerConfig, AttentionConfig)

    # Bypass VllmConfig.__post_init__ validation (LLaDA2 not in vLLM's model registry)
    pcfg = ParallelConfig(tensor_parallel_size=tp_size, enable_expert_parallel=True)
    vllm_cfg = VllmConfig.__new__(VllmConfig)
    vllm_cfg.parallel_config = pcfg
    vllm_cfg.model_config = None
    vllm_cfg.cache_config = None
    vllm_cfg.compilation_config = CompilationConfig(custom_ops=["none"])
    vllm_cfg.kernel_config = KernelConfig()
    vllm_cfg.device_config = DeviceConfig()
    vllm_cfg.load_config = LoadConfig()
    vllm_cfg.scheduler_config = None
    vllm_cfg.attention_config = None
    vllm_cfg.lora_config = None
    vllm_cfg.speculative_config = None
    vllm_cfg.quant_config = None
    vllm_cfg.observability_config = None
    vllm_cfg.profiler_config = None
    vllm_cfg.kv_transfer_config = None
    vllm_cfg.kv_events_config = None
    vllm_cfg.offload_config = None
    vllm_cfg.structured_outputs_config = None
    vllm_cfg.ec_transfer_config = None
    vllm_cfg.reasoning_config = None
    vllm_cfg.additional_config = None
    vllm_cfg.instance_id = "0"
    vllm_cfg.optimization_level = 0
    vllm_cfg.performance_mode = None
    vllm_cfg.weight_transfer_config = None
    vllm_cfg.shutdown_timeout = 60

    with set_current_vllm_config(vllm_cfg):
        vllm_dist.init_distributed_environment(world_size, rank, "env://", local_rank, "nccl")
        vllm_dist.initialize_model_parallel(tp_size, backend="nccl")

    if rank == 0:
        print(f"  Distributed init OK. TP={tp_size}")
        sys.stdout.flush()

    from transformers import AutoConfig, AutoTokenizer
    from dinfer.model import LLaDA2MoeModelLM
    from dinfer import BlockWiseDiffusionLLM, BlockIteratorFactory, ThresholdParallelDecoder

    # Load model with EP sharding
    with set_current_vllm_config(vllm_cfg):
        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)

    if rank == 0:
        print(f"  Model loaded. GPU mem/rank: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
        sys.stdout.flush()

    # Patch MoE with fused Triton routing
    from dinfer.epoch_spark_dinfer import patch_dinfer_baseline
    with set_current_vllm_config(vllm_cfg):
        patch_dinfer_baseline(model)

    # Warmup single forward
    if rank == 0:
        print("  Warmup forward...")
        sys.stdout.flush()
    with torch.inference_mode(), set_current_vllm_config(vllm_cfg):
        w = torch.arange(64, dtype=torch.long, device=device).unsqueeze(0)
        _ = model(w, use_cache=False)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print("  Warmup forward done.")
        sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    decoder = ThresholdParallelDecoder(temperature=0, threshold=0.9, mask_id=MASK_ID, eos_id=EOS_ID)

    # Prepare batched input
    local_bs = args.batch  # TP=4 DP=1: each rank processes full batch
    encoded = [tokenizer(PROMPTS[i % len(PROMPTS)], return_tensors="pt")["input_ids"] for i in range(local_bs)]
    max_len = max(e.shape[1] for e in encoded)
    input_ids = torch.full((local_bs, max_len), MASK_ID, dtype=torch.long, device=device)
    for i, e in enumerate(encoded):
        input_ids[i, :e.shape[1]] = e[0].to(device)

    if rank == 0:
        print(f"  Input shape: {input_ids.shape}")
        sys.stdout.flush()

    def make_dllm():
        return BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)

    # Warmup generation
    if rank == 0:
        print("  Warmup generation...")
        sys.stdout.flush()
    t_warmup_start = time.perf_counter()
    dllm = make_dllm()
    with torch.inference_mode():
        _ = dllm.generate(input_ids[:1].clone(), gen_length=32, block_length=BLOCK_LENGTH)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"  Warmup done in {time.perf_counter()-t_warmup_start:.1f}s")
        sys.stdout.flush()

    # Timed runs
    results = []
    for ri in range(args.num_runs):
        dllm = make_dllm()
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = dllm.generate(input_ids.clone(), gen_length=args.gen_length, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_fwd = dllm.num_forwards
        total_tokens = args.batch * args.gen_length
        tps = total_tokens / dt
        ms_fwd = dt * 1000 / n_fwd
        results.append({"time_s": dt, "n_fwd": n_fwd, "ms_fwd": ms_fwd, "tps": tps})
        if rank == 0:
            print(f"  Run {ri+1}: {dt:.3f}s, {n_fwd} fwd, {ms_fwd:.2f} ms/fwd, {tps:.0f} tok/s")
            sys.stdout.flush()

    if rank == 0:
        best = min(results, key=lambda r: r["time_s"])
        text = tokenizer.decode(out[0], skip_special_tokens=True) if out is not None else ""
        print(f"\n{'='*90}")
        print(f"RESULT: TP={tp_size}, batch={args.batch}, gen={args.gen_length}")
        print(f"  Best: {best['time_s']:.3f}s, {best['ms_fwd']:.2f} ms/fwd, {best['tps']:.0f} tok/s")
        print(f"  Output: {text[:200]}")
        print(f"{'='*90}")
        with open("bench_multigpu_results.json", "w") as f:
            json.dump({"config": vars(args), "results": results, "best": best, "output": text[:500]}, f, indent=2)


if __name__ == "__main__":
    main()
