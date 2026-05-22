#!/usr/bin/env python3
"""v5 benchmark: Pre-alloc KV + Triton routing."""
import os, sys, time, socket, types, importlib.util
import torch
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
MASK_ID = 156895
PROMPTS = ["Please solve problems step by step. Problem 1: A train travels from City A to B at 80 km/h.","Write an essay about AI history.","Explain Le Chatelier's principle.","Design a REST API.","Analyze climate change impact.","Explain quantum computing.","Design a message queue.","Write about training LLMs.","Compare TCP and UDP.","Explain neural networks.","Design ride-sharing microservices.","Write about cryptography.","Explain database indexing.","Discuss universal basic income.","Design CI/CD pipeline.","Explain relativity."]
def setup():
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    if "deep_ep" not in sys.modules:
        f=types.ModuleType("deep_ep");f.__spec__=importlib.util.spec_from_loader("deep_ep",loader=None);f.__path__=[]
        f.Buffer=type("Buffer",(),{"get_dispatch_config":staticmethod(lambda *a,**kw:None),"get_combine_config":staticmethod(lambda *a,**kw:None)})
        f.Config=type("Config",(),{});f.EventOverlap=type("EventOverlap",(),{});sys.modules["deep_ep"]=f
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--batch-sizes",type=str,default="16,64");p.add_argument("--gen-length",type=int,default=128);p.add_argument("--num-runs",type=int,default=3)
    args=p.parse_args();bss=[int(b) for b in args.batch_sizes.split(",")]
    device=torch.device("cuda:0");torch.cuda.set_device(device)
    print("="*90);print(f"v5 Benchmark | {torch.cuda.get_device_name(0)}");print("="*90)
    setup()
    from vllm import distributed;from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM);sock.bind(("127.0.0.1",0));port=sock.getsockname()[1];sock.close()
    os.environ.setdefault("MASTER_ADDR","127.0.0.1");os.environ.setdefault("MASTER_PORT",str(port))
    vcfg=VllmConfig(parallel_config=ParallelConfig(enable_expert_parallel=True))
    with set_current_vllm_config(vcfg):
        distributed.init_distributed_environment(1,0,"env://",0,"nccl");distributed.initialize_model_parallel(1,backend="nccl")
        from transformers import AutoConfig, AutoTokenizer;from dinfer.model import LLaDA2MoeModelLM
        from vllm.model_executor.layers.layernorm import rms_norm as vllm_rms_norm
        from dinfer.model.modeling_llada2_moe import LLaDA2MoeRMSNorm

        config=AutoConfig.from_pretrained(MODEL_PATH,trust_remote_code=True,local_files_only=True)
        model=LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH,torch_dtype=torch.bfloat16,device=device)
        model=model.to(device)

        # Apply fused RMSNorm
        for name,m in model.named_modules():
            if isinstance(m,LLaDA2MoeRMSNorm) and "query_layernorm" not in name and "key_layernorm" not in name:
                w,eps=m.weight,m.variance_epsilon
                m.forward=(lambda ww,ee: lambda hs: vllm_rms_norm(hs,ww,ee))(w,eps)

        # Patch attention for pre-alloc KV
        from dinfer.fast_generate_v5 import _patch_attention_for_prealloc_kv, _patch_moe_with_triton_routing, fast_generate_v5
        _patch_attention_for_prealloc_kv(model)
        n_moe = _patch_moe_with_triton_routing(model)
        print(f"[opt] Patched attention + {n_moe} MoE layers")

        tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True,local_files_only=True)

        # Warmup
        with torch.inference_mode():
            w=torch.randint(0,1000,(4,64),device=device)
            for _ in range(3): model(w,use_cache=False)
        torch.cuda.synchronize()
        print("[opt] Warmup done")

        # Also import baseline for comparison
        from dinfer.fast_generate import fast_generate_with_kvcache_cudagraph

        for bs in bss:
            print(f"\n{'='*70}\n  Batch size = {bs}\n{'='*70}")
            prompts=[PROMPTS[i%len(PROMPTS)] for i in range(bs)]
            encoded=[tokenizer.encode(pr,return_tensors="pt").squeeze(0) for pr in prompts]
            mx=max(e.shape[0] for e in encoded)
            padded=[torch.cat([torch.full((mx-e.shape[0],),MASK_ID,dtype=torch.long),e]) if e.shape[0]<mx else e for e in encoded]
            input_ids=torch.stack(padded).to(device)

            for label,fn in [("KV+CG",fast_generate_with_kvcache_cudagraph),("v5:PreKV+TritonRoute",fast_generate_v5)]:
                print(f"  [{label}] Running...")
                try:
                    with torch.inference_mode():
                        fn(model,input_ids[:min(bs,4)].clone(),gen_length=32,block_length=32)
                    torch.cuda.synchronize()
                except Exception as e:
                    print(f"  [{label}] Warmup failed: {e}")
                    import traceback; traceback.print_exc()
                    continue
                best=0
                for ri in range(args.num_runs):
                    torch.cuda.synchronize()
                    t0=time.perf_counter()
                    with torch.inference_mode():
                        out,n=fn(model,input_ids.clone(),gen_length=args.gen_length,block_length=32)
                    torch.cuda.synchronize()
                    dt=time.perf_counter()-t0;tps=bs*args.gen_length/dt;ms=dt/n*1000
                    best=max(best,tps)
                    print(f"    Run {ri+1}: {ms:.2f} ms/fwd, {tps:.0f} tok/s, {n} fwds")
                text=tokenizer.decode(out[0],skip_special_tokens=True)
                print(f"    Output: {text[:100]}")
                print(f"    Best: {best:.0f} tok/s")

if __name__=="__main__":
    main()
