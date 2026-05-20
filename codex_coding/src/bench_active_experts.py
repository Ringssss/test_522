"""Micro-benchmark: MoE kernel time vs number of active experts.
Holds total token-expert pairs constant, varies how many experts receive tokens.
"""
import os, sys, time, json, torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from transformers import AutoConfig
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeModelLM
    from baseline_optimizations import apply_all_optimizations
    from vllm.config import VllmConfig, ModelConfig, CacheConfig
    from vllm.forward_context import set_forward_context
    from vllm.distributed import get_ep_group

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = LLaDA2MoeModelLM(config=config).eval()
    model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
    model = model.to(device)
    model.set_eplb_runtime_state(enable_eplb=False)

    with torch.inference_mode():
        warmup_tok = torch.arange(180, dtype=torch.long, device=device).unsqueeze(0)
        vllm_cfg = VllmConfig(
            model_config=ModelConfig(MODEL_PATH, task="generate",
                                     tokenizer=MODEL_PATH, dtype=torch.bfloat16,
                                     seed=42, revision=None),
            cache_config=CacheConfig(block_size=16,
                                      gpu_memory_utilization=0.9,
                                      cache_dtype="auto"),
        )
        with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                  num_tokens=warmup_tok.numel()):
            _ = model(warmup_tok, use_cache=False)

    apply_all_optimizations(model)

    from dinfer.model.modeling_llada2_moe import LLaDA2MoeSparseMoeBlock
    moe_blocks = [m for _, m in model.named_modules()
                  if isinstance(m, LLaDA2MoeSparseMoeBlock)]
    blk = moe_blocks[0]
    experts = blk.experts
    gate = blk.gate

    ep_group = get_ep_group()
    ep_rank = ep_group.rank_in_group
    local_num_experts = config.num_experts // world_size  # 32

    # Total token-expert pairs to keep constant
    TOTAL_PAIRS = 8192  # = 256 tokens/expert × 32 experts (our batch=512 default)
    N_TOKENS = TOTAL_PAIRS  # each token selects K=1 for simplicity; or K=4 with fewer tokens
    HIDDEN = config.hidden_size
    K = 4
    N_TOKENS = TOTAL_PAIRS // K  # 2048 tokens, each selecting 4 experts

    active_expert_counts = [4, 8, 16, 32]
    n_warmup = 5
    n_measure = 20
    results = []

    for n_active in active_expert_counts:
        # Generate synthetic input
        hs = torch.randn(N_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)

        # Create forced topk_ids: route all tokens to the first n_active LOCAL experts
        local_start = ep_rank * local_num_experts
        active_ids = list(range(local_start, local_start + n_active))

        # Each token gets K=4 experts from the active set (round-robin)
        topk_ids = torch.zeros(N_TOKENS, K, dtype=torch.int32, device=device)
        for k_idx in range(K):
            topk_ids[:, k_idx] = active_ids[k_idx % n_active]
        # Spread more evenly: different tokens get different experts
        for t in range(N_TOKENS):
            for k_idx in range(K):
                topk_ids[t, k_idx] = active_ids[(t * K + k_idx) % n_active]

        topk_weights = torch.ones(N_TOKENS, K, dtype=torch.float32, device=device) / K

        # We need to bypass the normal routing and call the kernel directly
        # Use experts.forward_impl which does dispatch + quant_apply + combine
        # But we need to provide router_logits that produce our desired topk_ids

        # Simpler: call the FusedMoE apply directly with our forced routing
        router_logits = gate.get_logits(hs)

        # Monkey-patch routing to return our forced topk
        orig_routing = gate.routing
        def forced_routing(hidden_states=None, gating_output=None, topk=None, renormalize=None, **kw):
            return topk_weights.to(gating_output.dtype), topk_ids
        gate.routing = forced_routing
        experts.custom_routing_function = forced_routing

        # Warmup
        with torch.inference_mode():
            with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                      num_tokens=N_TOKENS):
                for _ in range(n_warmup):
                    _ = experts.forward_impl(hs, router_logits)
                torch.cuda.synchronize()

        # Measure
        torch.cuda.synchronize()
        dist.barrier()
        times = []
        with torch.inference_mode():
            for _ in range(n_measure):
                with set_forward_context(attn_metadata=None, vllm_config=vllm_cfg,
                                          num_tokens=N_TOKENS):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    _ = experts.forward_impl(hs, router_logits)
                    end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))

        gate.routing = orig_routing
        experts.custom_routing_function = orig_routing

        avg_ms = sum(times[2:]) / len(times[2:])  # skip first 2
        tokens_per_expert = TOTAL_PAIRS // n_active

        local_tensor = torch.tensor([avg_ms], dtype=torch.float64, device=device)
        all_times = [torch.zeros(1, dtype=torch.float64, device=device) for _ in range(world_size)]
        dist.all_gather(all_times, local_tensor)
        rank_times = [t.item() for t in all_times]

        if rank == 0:
            print(f"  n_active={n_active:>3}, tok/expert={tokens_per_expert:>5}, "
                  f"kernel_ms={avg_ms:.3f}, rank_max={max(rank_times):.3f}, "
                  f"rank_min={min(rank_times):.3f}, spread={max(rank_times)-min(rank_times):.3f}")

        results.append({
            "n_active_experts": n_active,
            "total_pairs": TOTAL_PAIRS,
            "tokens_per_expert": tokens_per_expert,
            "kernel_ms_local": avg_ms,
            "rank_times": rank_times,
            "rank_max": max(rank_times),
            "rank_min": min(rank_times),
        })

    if rank == 0:
        print()
        print("=== SCALING ANALYSIS ===")
        base = results[-1]["rank_max"]  # 32 active experts as baseline
        for r in results:
            ratio = r["rank_max"] / base
            expert_ratio = r["n_active_experts"] / 32
            print(f"  n_active={r['n_active_experts']:>3}: {r['rank_max']:.3f} ms "
                  f"({ratio:.2f}x vs 32-active, experts={expert_ratio:.2f}x)")

        out = {"results": results, "total_pairs": TOTAL_PAIRS, "K": K, "N_TOKENS": N_TOKENS}
        out_path = "codex_coding/results/active_expert_sweep_20260503.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved to {out_path}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
