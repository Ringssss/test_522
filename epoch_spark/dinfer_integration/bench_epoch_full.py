#!/usr/bin/env python3
"""Benchmark: base KVCache vs Epoch-Spark (all mechanisms)."""
import os, sys, time, json, socket, types, importlib.util
import torch
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
MASK_ID = 156895
PROMPTS = ["Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h.","Write a detailed essay about the history of artificial intelligence.","Explain Le Chatelier's principle with examples.","Design a REST API for an e-commerce platform.","Analyze the economic impact of climate change.","Explain quantum computing: qubits, superposition.","Design a distributed message queue.","Write a guide to training large language models.","Compare TCP and UDP protocols.","Explain neural networks and backpropagation.","Design microservices for ride-sharing.","Write about cryptography history.","Explain database indexing strategies.","Discuss universal basic income.","Design a CI/CD pipeline.","Explain relativity."]
def setup():
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    if "deep_ep" not in sys.modules or getattr(sys.modules.get("deep_ep"),"__spec__",None) is None:
        sys.modules.pop("deep_ep",None)
        f=types.ModuleType("deep_ep");f.__spec__=importlib.util.spec_from_loader("deep_ep",loader=None);f.__path__=[]
        f.Buffer=type("Buffer",(),{"get_dispatch_config":staticmethod(lambda *a,**kw:None),"get_combine_config":staticmethod(lambda *a,**kw:None)})
        f.Config=type("Config",(),{});f.EventOverlap=type("EventOverlap",(),{});sys.modules["deep_ep"]=f
def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--gen-length",type=int,default=128)
    p.add_argument("--batch-sizes",type=str,default="16,64")
    p.add_argument("--num-runs",type=int,default=3)
    args=p.parse_args()
    bss=[int(b) for b in args.batch_sizes.split(",")]
    device=torch.device(f"cuda:{args.gpu}");torch.cuda.set_device(device)
    print("="*100)
    print(f"Epoch-Spark Full Stack | LLaDA2.0-mini 16B | {torch.cuda.get_device_name(args.gpu)}")
    print("="*100)
    setup()
    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM);sock.bind(("127.0.0.1",0));port=sock.getsockname()[1];sock.close()
    os.environ.setdefault("MASTER_ADDR","127.0.0.1");os.environ.setdefault("MASTER_PORT",str(port))
    vcfg=VllmConfig(parallel_config=ParallelConfig(enable_expert_parallel=True))
    with set_current_vllm_config(vcfg):
        distributed.init_distributed_environment(1,0,"env://",0,"nccl")
        distributed.initialize_model_parallel(1,backend="nccl")
        from transformers import AutoConfig, AutoTokenizer
        from dinfer.model import LLaDA2MoeModelLM
        sys.path.insert(0,os.path.dirname(__file__))
        from bench_highperf import apply_fused_rmsnorm, apply_optimized_attn, apply_dtype_optimized_moe
        config=AutoConfig.from_pretrained(MODEL_PATH,trust_remote_code=True,local_files_only=True)
        model=LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH,torch_dtype=torch.bfloat16,device=device)
        model=model.to(device)
        apply_fused_rmsnorm(model);apply_optimized_attn(model);apply_dtype_optimized_moe(model)
        tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True,local_files_only=True)
        with torch.inference_mode():
            w=torch.randint(0,1000,(4,64),device=device)
            for _ in range(3): model(w,use_cache=False)
        torch.cuda.synchronize()
        print("[opt] All optimizations applied, warmup done")
        from dinfer.fast_generate import fast_generate_with_kvcache, fast_generate_with_kvcache_cudagraph
        from dinfer.epoch_generate import epoch_spark_generate
        results={}
        for bs in bss:
            print(f"\n{'='*80}\n  Batch size = {bs}\n{'='*80}")
            prompts=[PROMPTS[i%len(PROMPTS)] for i in range(bs)]
            encoded=[tokenizer.encode(pr,return_tensors="pt").squeeze(0) for pr in prompts]
            mx=max(e.shape[0] for e in encoded)
            padded=[torch.cat([torch.full((mx-e.shape[0],),MASK_ID,dtype=torch.long),e]) if e.shape[0]<mx else e for e in encoded]
            input_ids=torch.stack(padded).to(device)
            configs=[
                ("KVCache",fast_generate_with_kvcache,{}),
                ("KV+CG",fast_generate_with_kvcache_cudagraph,{}),
                ("Epoch-Spark",epoch_spark_generate,{"expert_budget_K":100}),
            ]
            bs_results={}
            for label,gen_fn,extra_kw in configs:
                print(f"  [{label}] Running...")
                try:
                    with torch.inference_mode():
                        gen_fn(model,input_ids[:min(bs,4)].clone(),gen_length=32,block_length=32,**extra_kw)
                    torch.cuda.synchronize()
                except Exception as e:
                    print(f"  [{label}] Warmup failed: {e}")
                    import traceback; traceback.print_exc()
                    bs_results[label]={"tps":0}
                    continue
                best_tps=0
                for ri in range(args.num_runs):
                    torch.cuda.synchronize()
                    t0=time.perf_counter()
                    with torch.inference_mode():
                        out,n=gen_fn(model,input_ids.clone(),gen_length=args.gen_length,block_length=32,**extra_kw)
                    torch.cuda.synchronize()
                    dt=time.perf_counter()-t0
                    tps=bs*args.gen_length/dt;ms=dt/n*1000
                    best_tps=max(best_tps,tps)
                    print(f"    Run {ri+1}: {ms:.2f} ms/fwd, {tps:.0f} tok/s, {n} fwds")
                text=tokenizer.decode(out[0],skip_special_tokens=True)
                print(f"    Output: {text[:100]}")
                bs_results[label]={"tps":round(best_tps,0)}
            results[f"bs{bs}"]=bs_results
    print("\n"+"="*80)
    print("FINAL RESULTS")
    print("="*80)
    headers=set()
    for r in results.values():
        headers.update(r.keys())
    headers=sorted(headers)
    hdr=" | ".join(f"{h:>12}" for h in headers)
    print(f"{'BS':>4} | {hdr}")
    print("-"*80)
    for bsk in sorted(results.keys(),key=lambda k:int(k[2:])):
        r=results[bsk];bv=bsk[2:]
        vals=" | ".join(f"{r.get(h,{}).get('tps',0):12.0f}" for h in headers)
        print(f"{bv:>4} | {vals}")
    print("="*80)
    with open("bench_epoch_full_results.json","w") as f:
        json.dump(results,f,indent=2)
if __name__=="__main__":
    main()
