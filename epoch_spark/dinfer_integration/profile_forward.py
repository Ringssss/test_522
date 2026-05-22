#!/usr/bin/env python3
"""Profile forward pass component breakdown for LLaDA2.0-mini."""
import sys, os, time, socket, types, importlib.util
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# deep_ep stub
if "deep_ep" not in sys.modules:
    f = types.ModuleType("deep_ep"); f.__spec__ = importlib.util.spec_from_loader("deep_ep", loader=None); f.__path__ = []
    f.Buffer = type("Buffer", (), {"get_dispatch_config": staticmethod(lambda *a,**kw:None), "get_combine_config": staticmethod(lambda *a,**kw:None)})
    f.Config = type("Config", (), {}); f.EventOverlap = type("EventOverlap", (), {}); sys.modules["deep_ep"] = f

import torch
import torch.cuda

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
MASK_ID = 156895

def main():
    device = torch.device("cuda:0"); torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]; sock.close()
    os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT",str(port))
    pcfg = ParallelConfig(enable_expert_parallel=True)
    vcfg = VllmConfig(parallel_config=pcfg)
    with set_current_vllm_config(vcfg):
        distributed.init_distributed_environment(1,0,"env://",0,"nccl")
        distributed.initialize_model_parallel(1,backend="nccl")

        from transformers import AutoConfig, AutoTokenizer
        from dinfer.model import LLaDA2MoeModelLM
        from dinfer.epoch_spark_dinfer import patch_dinfer_baseline

        config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        patch_dinfer_baseline(model)

        # Warmup
        x = torch.randint(0, 1000, (16, 210), device=device)
        with torch.inference_mode():
            for _ in range(3):
                model(x, use_cache=False)
        torch.cuda.synchronize()

        # Profile with torch profiler
        print("=" * 80)
        print("Profiling bs=16, seq=210 forward pass")
        print("=" * 80)

        with torch.inference_mode():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=False,
            ) as prof:
                for _ in range(5):
                    model(x, use_cache=False)
                torch.cuda.synchronize()

        # Print CUDA kernel summary
        print("\n=== Top CUDA Kernels (by total CUDA time) ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        print("\n=== Top CPU Operations ===")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

        # Manual component timing
        print("\n=== Manual Component Timing (10 iters, bs=16, seq=210) ===")
        timings = {"total": [], "embedding": [], "attention": [], "moe": [],
                   "norm": [], "lm_head": [], "python_overhead": []}

        # Hook each component
        attn_time = [0.0]
        moe_time = [0.0]
        norm_time = [0.0]

        def make_timing_hook(name, acc):
            def hook(mod, inp, out):
                torch.cuda.synchronize()
                acc[0] += time.perf_counter()
            return hook

        # Time full forward vs sum of components
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x, use_cache=False)
            torch.cuda.synchronize()
            total = (time.perf_counter() - t0) * 1000
            timings["total"].append(total)

        avg_total = sum(timings["total"]) / len(timings["total"])
        print(f"  Total forward:  {avg_total:.2f} ms")
        print(f"  → At bs=64:     {avg_total * 64/16:.2f} ms (linear est)")
        print(f"  → Throughput:    {16 * 128 / (avg_total/1000 * 128):.0f} tok/s (bs=16, gen=128)")
        print(f"  → Throughput:    {64 * 128 / (avg_total * 64/16 /1000 * 128):.0f} tok/s (bs=64, gen=128)")

        # CUDA graph attempt
        print("\n=== CUDA Graph Test ===")
        static_x = torch.randint(0, 1000, (16, 210), device=device)
        with torch.inference_mode():
            # Warmup
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    model(static_x, use_cache=False)
            torch.cuda.current_stream().wait_stream(s)

            # Try capture
            try:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=s):
                    static_out = model(static_x, use_cache=False)
                torch.cuda.synchronize()

                # Replay
                for _ in range(3):
                    g.replay()
                torch.cuda.synchronize()

                t0 = time.perf_counter()
                for _ in range(20):
                    g.replay()
                torch.cuda.synchronize()
                cg_ms = (time.perf_counter() - t0) / 20 * 1000
                print(f"  CUDA Graph forward: {cg_ms:.2f} ms (vs eager {avg_total:.2f} ms)")
                print(f"  Speedup: {avg_total/cg_ms:.2f}x")
                print(f"  → Throughput with CG: {16 * 128 / (cg_ms/1000 * 128):.0f} tok/s (bs=16)")
            except Exception as e:
                print(f"  CUDA Graph capture failed: {e}")
                print("  (Expected — dynamic shapes from MoE routing)")

if __name__ == "__main__":
    main()
