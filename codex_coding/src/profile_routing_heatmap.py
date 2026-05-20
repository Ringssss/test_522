"""Per-layer per-expert routing heatmap: MASK vs decoded token analysis.

Hooks into bench_bsp_moe_dp2's routing path to collect per-layer per-expert
hit counts, separated by MASK/decoded token type.

Usage:
  MASTER_PORT=30150 torchrun --nproc_per_node=8 profile_routing_heatmap.py
"""
import json, os, sys
import torch
import torch.distributed as dist
import numpy as np

MASK_ID = 156895
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    from transformers import AutoConfig, AutoTokenizer
    from dinfer.model.modeling_llada2_moe import LLaDA2MoeModelLM
    from dinfer import (
        BlockDiffusionLLM,
        BlockIteratorFactory,
        KVCacheFactory,
        ThresholdParallelDecoder,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    if rank == 0:
        print("Loading model...")
    model = LLaDA2MoeModelLM(config=config).eval()
    model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
    model = model.to(device)

    first_k = config.first_k_dense_replace
    num_moe_layers = config.num_hidden_layers - first_k
    num_experts = config.num_experts
    moe_blocks = [model.model.layers[i].mlp for i in range(first_k, config.num_hidden_layers)]
    if rank == 0:
        print(f"MoE layers: {num_moe_layers}, experts: {num_experts}, first_k_dense={first_k}")

    # --- MASK position tracker ---
    _mask_state = {"flat_mask": None}

    orig_embed = model.model.embed_tokens.forward
    def hooked_embed(input_ids):
        _mask_state["flat_mask"] = (input_ids == MASK_ID).reshape(-1)
        return orig_embed(input_ids)
    model.model.embed_tokens.forward = hooked_embed

    # --- Per-layer routing stats ---
    mask_hits = torch.zeros((num_moe_layers, num_experts), dtype=torch.int64)
    dec_hits = torch.zeros((num_moe_layers, num_experts), dtype=torch.int64)
    call_count = [0]

    for li, blk in enumerate(moe_blocks):
        orig_gate_fwd = blk.gate.forward

        def make_gate_hook(layer_idx, orig_fn):
            def hooked_gate(hidden_states):
                logits = orig_fn(hidden_states)
                # logits: [N, num_experts] — capture routing decisions
                with torch.no_grad():
                    topk_ids = logits.topk(8, dim=-1).indices  # [N, 8]
                    flat_mask = _mask_state.get("flat_mask")
                    N = hidden_states.shape[0]
                    if flat_mask is not None:
                        fm = flat_mask
                        if fm.shape[0] != N:
                            tp_size = 4
                            tp_rank = rank % tp_size
                            chunk = fm.shape[0] // tp_size
                            fm = fm[tp_rank * chunk:(tp_rank + 1) * chunk]
                        if fm.shape[0] == N:
                            ids_cpu = topk_ids.cpu()
                            m_cpu = fm[:N].cpu().bool()
                            # Vectorized counting
                            for k in range(ids_cpu.shape[1]):
                                col = ids_cpu[:, k]
                                m_ids = col[m_cpu]
                                d_ids = col[~m_cpu]
                                mask_hits[layer_idx].scatter_add_(
                                    0, m_ids.long(), torch.ones_like(m_ids, dtype=torch.int64))
                                dec_hits[layer_idx].scatter_add_(
                                    0, d_ids.long(), torch.ones_like(d_ids, dtype=torch.int64))
                    call_count[0] += 1
                return logits
            return hooked_gate

        blk.gate.forward = make_gate_hook(li, orig_gate_fwd)

    # --- Build input ---
    sys.path.insert(0, os.path.dirname(__file__))
    from test_heteval512 import PROMPTS
    batch_size = 512
    all_ids = [tokenizer(t, return_tensors="pt")["input_ids"][0] for t in PROMPTS[:batch_size]]
    max_len = max(t.shape[0] for t in all_ids)
    padded = [torch.cat([t, torch.full((max_len - t.shape[0],), MASK_ID, dtype=t.dtype)])
              if t.shape[0] < max_len else t[:max_len] for t in all_ids]
    input_ids_full = torch.stack(padded, dim=0)

    tp_size = 4
    dp_size = world // tp_size
    dp_rank = rank // tp_size
    local_bs = batch_size // dp_size
    my_input = input_ids_full[dp_rank * local_bs:(dp_rank + 1) * local_bs].to(device)

    if rank == 0:
        print(f"Input: {tuple(my_input.shape)}, generating 96 tokens (3 blocks)...")

    decoder = ThresholdParallelDecoder(temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=156896)
    dllm = BlockDiffusionLLM(
        model, decoder, BlockIteratorFactory(use_block_diffusion=True),
        cache_factory=KVCacheFactory("prefix", is_bd_model=True),
        early_stop=True, maximum_unroll=4, expected_tpf=15,
        backend="vllm", lazy_cache_update=True, inplace_cache_update=True,
    )

    with torch.no_grad():
        _ = dllm.generate(my_input, generate_length=96)

    if rank == 0:
        print(f"Gate hook calls: {call_count[0]}")

    # --- Aggregate across ranks ---
    m_dev = mask_hits.to(device)
    d_dev = dec_hits.to(device)
    dist.all_reduce(m_dev)
    dist.all_reduce(d_dev)
    mask_all = m_dev.cpu().numpy()
    dec_all = d_dev.cpu().numpy()
    total_all = mask_all + dec_all

    if rank == 0:
        result = {"num_moe_layers": num_moe_layers, "num_experts": num_experts,
                  "gate_hook_calls": call_count[0], "per_layer": []}

        print("\n" + "=" * 72)
        print("Per-Layer Expert Load Distribution: MASK vs Decoded")
        print("=" * 72)

        for li in range(num_moe_layers):
            m = mask_all[li]
            d = dec_all[li]
            t = total_all[li]
            sorted_idx = t.argsort()[::-1]

            total_load = t.sum()
            cv = t.std() / t.mean() if t.mean() > 0 else 0
            top25 = sorted_idx[:25]
            concentration = t[top25].sum() / total_load if total_load > 0 else 0

            mask_frac_overall = m.sum() / total_load if total_load > 0 else 0

            # Top-5 analysis
            top5 = sorted_idx[:5]
            top5_info = []
            for ei in top5:
                tok_total = int(t[ei])
                mask_f = m[ei] / t[ei] if t[ei] > 0 else 0
                top5_info.append({"expert": int(ei), "load": tok_total,
                                  "mask_frac": float(mask_f)})

            # Hot expert mask affinity
            hot_mask = m[top25].sum()
            hot_total = t[top25].sum()
            hot_mask_frac = hot_mask / hot_total if hot_total > 0 else 0
            affinity = hot_mask_frac / mask_frac_overall if mask_frac_overall > 0 else 1

            # Cold expert (bottom 25%) analysis
            bot25 = sorted_idx[-25:]
            cold_mask_frac = m[bot25].sum() / t[bot25].sum() if t[bot25].sum() > 0 else 0
            cold_affinity = cold_mask_frac / mask_frac_overall if mask_frac_overall > 0 else 1

            info = {
                "layer": li, "global_layer": li + first_k,
                "cv": float(cv), "concentration_top10pct": float(concentration),
                "mask_frac_overall": float(mask_frac_overall),
                "hot_expert_mask_affinity": float(affinity),
                "cold_expert_mask_affinity": float(cold_affinity),
                "top5": top5_info,
                "max_load": int(t[sorted_idx[0]]),
                "min_load": int(t[sorted_idx[-1]]),
                "max_min_ratio": float(t[sorted_idx[0]] / t[sorted_idx[-1]]) if t[sorted_idx[-1]] > 0 else float('inf'),
            }
            result["per_layer"].append(info)

            print(f"\n  Layer {li} (global {li + first_k}):")
            print(f"    CV={cv:.4f}  top10% concentration={concentration:.3f}  max/min={info['max_min_ratio']:.1f}x")
            print(f"    Overall MASK fraction: {mask_frac_overall:.1%}")
            print(f"    Hot(top10%) mask affinity:  {affinity:.3f}"
                  f" ({'MASK→hot' if affinity > 1.05 else 'neutral' if affinity > 0.95 else 'decoded→hot'})")
            print(f"    Cold(bot10%) mask affinity: {cold_affinity:.3f}"
                  f" ({'MASK→cold' if cold_affinity > 1.05 else 'neutral' if cold_affinity > 0.95 else 'decoded→cold'})")
            top5_str = [(x['expert'], x['load'], f"{x['mask_frac']:.0%}") for x in top5_info]
            print(f"    Top-5: {top5_str}")

        # Cross-layer consistency
        print(f"\n{'=' * 72}")
        print("Cross-Layer Hot Expert Overlap (Jaccard of top-20)")
        print("=" * 72)
        top20s = [set(total_all[li].argsort()[-20:].tolist()) for li in range(num_moe_layers)]
        for li in range(num_moe_layers - 1):
            j = len(top20s[li] & top20s[li + 1]) / len(top20s[li] | top20s[li + 1])
            shared = len(top20s[li] & top20s[li + 1])
            print(f"  L{li}↔L{li+1}: Jaccard={j:.3f} (shared={shared}/20)")

        global_hot = top20s[0]
        for s in top20s[1:]:
            global_hot &= s
        print(f"\n  Global hot (in ALL layers' top-20): {sorted(global_hot)} ({len(global_hot)} experts)")
        result["global_hot_all_layers"] = sorted(global_hot)

        out_path = "/home/wuhang/wuhang/dllm_wh/codex_coding/results/routing_heatmap_c12_20260502.json"
        result["mask_hits"] = mask_all.tolist()
        result["decoded_hits"] = dec_all.tolist()
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {out_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
