#!/usr/bin/env python3
"""
v0.1.15.7d — Plan A Optimized: Cold/Hot Dual Path

Cold path (block 1st forward): full batch-add with coverage check + sync
Hot path (block 2nd+ forward): popularity top-K_init, NO coverage check, NO sync
                                → pure GPU pipeline, 12 dispatch, 0 sync

Key question: does hot path dispatch hide behind GPU execution (pipeline overlap)?
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

MASK_ID, EOS_ID, BLOCK_LENGTH = 156895, 156892, 32
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
GEN_LENGTH = 256

PROMPTS = [
    "Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h. The total distance between the two cities is 240 km. What is the average speed for the entire round trip?\n\nProblem 2: A rectangular garden has a perimeter of 56 meters.",
    "Write a detailed essay about the history of artificial intelligence, covering the Dartmouth conference of 1956, the AI winters, the rise of machine learning in the 1990s, and deep learning breakthroughs.",
    "You are a chemistry professor. Explain Le Chatelier's principle with examples and how it applies to industrial ammonia production via the Haber process.",
    "Design a complete REST API for an e-commerce platform with endpoints for user authentication, product management, shopping cart operations, and order processing.",
    "Analyze the global economic impact of climate change across agriculture, energy, real estate, and healthcare sectors with specific examples.",
    "Explain quantum computing to a classical CS background: qubits, superposition, entanglement, Shor's algorithm, and current hardware approaches.",
    "You are a systems architect. Design a distributed message queue with partition-based storage, consumer groups, replication, and exactly-once semantics.",
    "Write a comprehensive guide to training large language models covering data collection, tokenizer training, architecture decisions, and distributed training strategies.",
    "Solve the quadratic equation x^2 - 5x + 6 = 0 step by step. Show the factoring method, then verify both solutions by substituting them back into the original equation.",
    "Explain the mathematical foundations of neural networks: backpropagation, gradient descent, loss functions, and the universal approximation theorem.",
    "Design a microservices architecture for a ride-sharing application with real-time matching, pricing, routing, payments, and driver management.",
    "Write about the history of cryptography from Caesar ciphers through RSA, elliptic curve cryptography, and post-quantum cryptographic algorithms.",
    "Explain database indexing strategies: B-trees, hash indexes, bitmap indexes, and their trade-offs for OLTP vs OLAP workloads.",
    "Solve this logic puzzle step by step: If A is true, then B is true. If B is true, then C is true. A is true. What can we conclude about B and C? Then, if D is true only when both B and C are true, what can we conclude about D?",
    "Design a CI/CD pipeline for a large monorepo with microservices, including build caching, parallel testing, canary deployments, and rollback strategies.",
    "Explain the theory of relativity to a physics undergraduate, covering special relativity, time dilation, length contraction, and general relativity basics.",
    "Write a comprehensive comparison of Python, Rust, and Go for systems programming, covering memory safety, concurrency models, and ecosystem maturity.",
    "Design a real-time recommendation engine for a video streaming platform that handles cold start, user preferences, and content diversity.",
    "Explain the CAP theorem and its practical implications for distributed database design, with examples from Cassandra, MongoDB, and CockroachDB.",
    "Write a Python function to compute the nth Fibonacci number. Show the function, then compute fib(1) through fib(10) step by step and list all 10 values.",
    "Design a fraud detection system for a payment processing company using machine learning, rule engines, and real-time streaming analytics.",
    "Explain compiler optimization techniques including SSA form, loop unrolling, vectorization, and register allocation strategies.",
    "Write about the history and future of space exploration, from Apollo missions through SpaceX reusability to planned Mars colonization.",
    "Design an observability platform with distributed tracing, log aggregation, metrics collection, and intelligent alerting for microservices.",
    "Explain the mathematics behind public key cryptography, including modular arithmetic, Euler's theorem, and the RSA algorithm step by step.",
    "Write a guide to modern CSS layout techniques including Flexbox, Grid, Container Queries, and responsive design best practices.",
    "Design a multi-tenant SaaS platform architecture with data isolation, custom domains, billing integration, and horizontal scaling.",
    "Explain how garbage collectors work in JVM, Go, and Python, comparing mark-sweep, generational, and reference counting approaches.",
    "List all 8 planets in our solar system in order from closest to farthest from the Sun. For each planet, state whether it is a terrestrial or gas/ice giant planet, and give its approximate orbital period in Earth years.",
    "Design a real-time collaborative document editor like Google Docs with conflict resolution, offline support, and version history.",
    "Explain operating system memory management: virtual memory, page tables, TLB, demand paging, and memory-mapped files.",
    "Write a comprehensive guide to Kubernetes architecture including pods, services, ingress, operators, and cluster autoscaling.",
]
VERIFIABLE = {
    0: "average speed = 480/7 ~ 68.57 km/h", 8: "x = 2 and x = 3",
    13: "B is true, C is true, D is true", 19: "fib(10) = 55",
    28: "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune",
}

# ================================================================
# Fused Routing Kernel with S_mask (same as v0.1.15.7b)
# ================================================================
@triton.jit
def _fused_routing_k(
    logits_ptr, expert_bias_ptr, s_mask_ptr,
    topk_ids_ptr, topk_weights_ptr,
    N, rsf, sl_n, sl_e, si_n, si_k, sw_n, sw_k,
    HAS_S_MASK: tl.constexpr,
    E: tl.constexpr, K: tl.constexpr,
    NG: tl.constexpr, TKG: tl.constexpr, GS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return
    oe = tl.arange(0, E)
    logits = tl.load(logits_ptr + pid * sl_n + oe * sl_e)
    bias = tl.load(expert_bias_ptr + oe)
    scores = tl.sigmoid(logits)
    sb = scores + bias
    gs = tl.zeros([NG], dtype=tl.float32)
    for g in tl.static_range(NG):
        go = tl.arange(0, GS)
        gv = tl.load(logits_ptr + pid * sl_n + (g * GS + go) * sl_e)
        gsc = tl.sigmoid(gv) + tl.load(expert_bias_ptr + g * GS + go)
        m1 = tl.max(gsc, axis=0)
        gsc2 = tl.where(gsc == m1, float('-inf'), gsc)
        m2 = tl.max(gsc2, axis=0)
        m2 = tl.where(m2 == float('-inf'), 0.0, m2)
        gs = tl.where(tl.arange(0, NG) == g, m1 + m2, gs)
    gm = tl.zeros([NG], dtype=tl.int32)
    gt = gs
    for _ in tl.static_range(TKG):
        gi = tl.argmax(gt, axis=0)
        gm = tl.where(tl.arange(0, NG) == gi, 1, gm)
        gt = tl.where(tl.arange(0, NG) == gi, float('-inf'), gt)
    eg = oe // GS
    ea = tl.zeros([E], dtype=tl.int32)
    for g in tl.static_range(NG):
        ig = (eg == g)
        gsel = tl.sum(tl.where(tl.arange(0, NG) == g, gm, tl.zeros([NG], dtype=tl.int32)))
        ea = tl.where(ig, gsel, ea)
    if HAS_S_MASK:
        sm = tl.load(s_mask_ptr + oe)
        ea = ea & sm
    ms = tl.where(ea == 1, sb, float('-inf'))
    ti = tl.zeros([K], dtype=tl.int32)
    mt = ms
    for _k in tl.static_range(K):
        bi = tl.argmax(mt, axis=0)
        ti = tl.where(tl.arange(0, K) == _k, bi, ti)
        mt = tl.where(oe == bi, float('-inf'), mt)
    ts = tl.zeros([K], dtype=tl.float32)
    for _k in tl.static_range(K):
        idx = tl.sum(tl.where(tl.arange(0, K) == _k, ti, tl.zeros([K], dtype=tl.int32)))
        val = tl.sum(tl.where(oe == idx, scores, tl.zeros([E], dtype=tl.float32)))
        ts = tl.where(tl.arange(0, K) == _k, val, ts)
    ss = tl.sum(ts, axis=0) + 1e-20
    tw = ts / ss * rsf
    ok = tl.arange(0, K)
    tl.store(topk_ids_ptr + pid * si_n + ok * si_k, ti)
    tl.store(topk_weights_ptr + pid * sw_n + ok * sw_k, tw)


def fused_routing(logits, bias, rsf, s_mask=None, K=8, ng=8, tkg=4):
    N, E = logits.shape
    ids = torch.empty((N, K), dtype=torch.int32, device=logits.device)
    wts = torch.empty((N, K), dtype=torch.float32, device=logits.device)
    lf = logits.float() if logits.dtype != torch.float32 else logits
    bf = bias.float() if bias.dtype != torch.float32 else bias
    hm = s_mask is not None
    sp = s_mask if hm else torch.empty(0, dtype=torch.int32, device=logits.device)
    _fused_routing_k[(N,)](
        lf, bf, sp, ids, wts, N, rsf,
        lf.stride(0), lf.stride(1), ids.stride(0), ids.stride(1),
        wts.stride(0), wts.stride(1),
        HAS_S_MASK=hm, E=E, K=K, NG=ng, TKG=tkg, GS=E // ng)
    return wts, ids


# ================================================================
# Plan A Optimized Controller
# ================================================================
class PlanAOptController:
    def __init__(self, K=8, M=4, K_target=40, quality_floor=0.70,
                 q_major=0.95, per_round_cap=8, max_rounds=999,
                 rsf_default=2.5):
        self.K = K
        self.M = M
        self.K_ext = K + M
        self.K_target = K_target
        self.quality_floor = quality_floor
        self.q_major = q_major
        self.per_round_cap = per_round_cap
        self.max_rounds = max_rounds

        self.K_init = {}        # layer_idx → int (from cold start)
        self.prev_N = {}        # layer_idx → int (for block boundary detection)
        self.cold_count = 0
        self.hot_count = 0
        self.cold_rounds = []
        self.cold_active = []

    def is_new_block(self, layer_idx, N):
        prev = self.prev_N.get(layer_idx, -1)
        self.prev_N[layer_idx] = N
        if prev == -1:
            return True       # first ever call → cold start
        return N > prev       # N increased = new block (prefix grew)

    def cold_start(self, layer_idx, logits, expert_bias, rsf):
        """Full batch-add with coverage check. Has GPU→CPU sync."""
        N, E = logits.shape
        K, K_ext = self.K, self.K_ext

        scores_full = torch.sigmoid(logits.float())
        topkm_score, topkm_idx = torch.topk(scores_full, k=K_ext, dim=1)
        topkm_weight = topkm_score / (topkm_score.sum(dim=1, keepdim=True) + 1e-20) * rsf

        sorted_w, _ = topkm_weight.sort(dim=1, descending=True)
        r = self.quality_floor * sorted_w[:, :K].sum(dim=1)

        popularity = torch.zeros(E, device=logits.device, dtype=torch.float32)
        popularity.scatter_add_(0, topkm_idx.reshape(-1), topkm_weight.reshape(-1))

        _, pop_order = popularity.sort(descending=True)
        S_mask = torch.zeros(E, dtype=torch.bool, device=logits.device)
        S_mask[pop_order[:self.K_target]] = True

        token_ids = torch.arange(N, device=logits.device).unsqueeze(1).expand(N, K_ext)
        n_rounds = 0

        for _ in range(self.max_rounds):
            in_S = S_mask[topkm_idx]
            c = (topkm_weight * in_S.float()).sum(dim=1)
            d = (r - c).clamp_min(0.0)
            satisfied = d <= 0
            sat_ratio = satisfied.float().mean().item()  # SYNC

            if sat_ratio >= self.q_major:
                break

            unsat = ~satisfied
            edge_mask = unsat.unsqueeze(1).expand_as(topkm_idx)
            ee = topkm_idx[edge_mask]
            ew = topkm_weight[edge_mask]
            et = token_ids[edge_mask]
            keep = ~S_mask[ee]
            ee, ew, et = ee[keep], ew[keep], et[keep]
            if ee.numel() == 0:
                break

            G = torch.zeros(E, device=logits.device, dtype=torch.float32)
            H = torch.zeros(E, device=logits.device, dtype=torch.float32)
            G.scatter_add_(0, ee, torch.minimum(ew, d[et]))
            H.scatter_add_(0, ee, (c[et] + ew >= r[et]).float())
            Score = H + 0.5 * G
            Score[S_mask] = -1e30
            _, add_order = Score.sort(descending=True)
            new_e = add_order[:self.per_round_cap]
            new_e = new_e[Score[new_e] > 0]
            if new_e.numel() == 0:
                break
            S_mask[new_e] = True
            n_rounds += 1

        active = int(S_mask.sum().item())  # SYNC
        self.K_init[layer_idx] = active
        self.cold_count += 1
        self.cold_rounds.append(n_rounds)
        self.cold_active.append(active)

        return S_mask.int()

    def hot_path(self, layer_idx, logits, expert_bias, rsf):
        """Popularity top-K_init. Pure GPU, zero sync."""
        N, E = logits.shape
        K_ext = self.K_ext
        K_init = self.K_init.get(layer_idx, 142)  # fallback if no cold start yet

        scores_full = torch.sigmoid(logits.float())                       # 1 dispatch
        topkm_score, topkm_idx = torch.topk(scores_full, k=K_ext, dim=1) # 1 dispatch
        topkm_weight = topkm_score / (                                    # 3 dispatch
            topkm_score.sum(dim=1, keepdim=True) + 1e-20) * rsf

        popularity = torch.zeros(E, device=logits.device, dtype=torch.float32)  # 1 alloc
        popularity.scatter_add_(0, topkm_idx.reshape(-1),                 # 1 dispatch
                                topkm_weight.reshape(-1))

        _, pop_order = popularity.sort(descending=True)                   # 1 dispatch
        S_mask = torch.zeros(E, dtype=torch.int32, device=logits.device)  # 1 alloc
        S_mask[pop_order[:K_init]] = 1                                    # 1 dispatch

        self.hot_count += 1
        return S_mask                                                     # NO sync

    def get_s_mask(self, layer_idx, logits, expert_bias, rsf):
        N = logits.shape[0]
        if self.is_new_block(layer_idx, N):
            return self.cold_start(layer_idx, logits, expert_bias, rsf)
        else:
            return self.hot_path(layer_idx, logits, expert_bias, rsf)

    def summary(self):
        total = self.cold_count + self.hot_count
        return {
            "cold_count": self.cold_count,
            "hot_count": self.hot_count,
            "hot_pct": self.hot_count / max(total, 1) * 100,
            "cold_avg_rounds": sum(self.cold_rounds) / max(len(self.cold_rounds), 1),
            "cold_avg_active": sum(self.cold_active) / max(len(self.cold_active), 1),
        }


# ================================================================
# Main
# ================================================================
def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoTokenizer, AutoConfig
    from baseline_optimizations import apply_all_optimizations

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print("v0.1.15.7d — Plan A Optimized: Cold/Hot Dual Path")
    print("  Cold: full batch-add (per-block 1st fwd)")
    print("  Hot: popularity top-K_init, 0 sync (per-block 2nd+ fwd)")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0), use_cache=False)

        print("\nApplying baseline optimizations...")
        n_rms, n_fa = apply_all_optimizations(model)
        print(f"  Fused RMSNorm: {n_rms}, Flash-attn: {n_fa}")

        BATCH_SIZE = 32
        all_ids = []
        for i in range(BATCH_SIZE):
            text = PROMPTS[i]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                  if ids.shape[0] < mx else ids for ids in all_ids]
        input_ids = torch.stack(padded, dim=0).to(device)
        prompt_len = input_ids.shape[1]
        print(f"  Input shape: {input_ids.shape}")

        decoder_t0 = ThresholdParallelDecoder(temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm(decoder):
            return BlockDiffusionLLM(
                model, decoder, BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True, inplace_cache_update=True)

        orig_routings = {}
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                orig_routings[name] = mod.routing

        def restore():
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate" and name in orig_routings:
                    mod.routing = orig_routings[name]

        results = OrderedDict()

        # ============================================================
        # C5: Fused routing only (baseline for this experiment)
        # ============================================================
        print(f"\n{'='*60}")
        print("C5: Fused routing only (no EB)")
        print(f"{'='*60}")

        idx = 0
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                b, r, tk, ng, tkg = mod.expert_bias, mod.routed_scaling_factor, mod.top_k, mod.n_group, mod.topk_group
                def mk(bb, rr, tt, nn, gg):
                    def fn(hs, go, topk, renorm):
                        w, i = fused_routing(go, bb, rr, s_mask=None, K=tt, ng=nn, tkg=gg)
                        return w.to(go.dtype), i
                    return fn
                mod.routing = mk(b, r, tk, ng, tkg)
                idx += 1

        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()

        c5_times, c5_fwds = [], []
        for ri in range(2):
            dllm = make_dllm(decoder_t0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            c5_times.append(t1 - t0)
            c5_fwds.append(dllm.diff_iteration.num_forwards)
            print(f"  Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd")

        c5_avg = sum(c5_times) / len(c5_times)
        c5_fwd = sum(c5_fwds) / len(c5_fwds)
        results["C5"] = {"avg_time": c5_avg, "avg_fwd": c5_fwd,
                         "fwd_per_s": c5_fwd / c5_avg, "ms_per_fwd": c5_avg / c5_fwd * 1000}
        print(f"  Avg: {c5_avg:.3f}s, {c5_fwd:.0f} fwd, {c5_fwd/c5_avg:.1f} fwd/s, "
              f"{c5_avg/c5_fwd*1000:.2f} ms/fwd")
        restore()

        # ============================================================
        # C9: Plan A Optimized (cold/hot dual path)
        # ============================================================
        print(f"\n{'='*60}")
        print("C9: Plan A Optimized — cold/hot dual path")
        print(f"{'='*60}")

        ctrl = PlanAOptController(K=8, M=4, K_target=40, quality_floor=0.70,
                                   q_major=0.95, per_round_cap=8)
        idx = 0
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                b, r, tk, ng, tkg = mod.expert_bias, mod.routed_scaling_factor, mod.top_k, mod.n_group, mod.topk_group
                li = idx
                def mk(bb, rr, tt, nn, gg, layer_i, cc):
                    def fn(hs, go, topk, renorm):
                        sm = cc.get_s_mask(layer_i, go, bb, rr)
                        w, i = fused_routing(go, bb, rr, s_mask=sm, K=tt, ng=nn, tkg=gg)
                        return w.to(go.dtype), i
                    return fn
                mod.routing = mk(b, r, tk, ng, tkg, li, ctrl)
                idx += 1
        print(f"  Patched {idx} gates")

        # Warmup (triggers cold starts)
        print("  Warmup...")
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        wu = ctrl.summary()
        print(f"  Warmup: cold={wu['cold_count']}, hot={wu['hot_count']}, "
              f"cold_avg_rounds={wu['cold_avg_rounds']:.1f}, cold_avg_|S|={wu['cold_avg_active']:.1f}")

        # Reset stats for measurement
        ctrl.cold_count = 0
        ctrl.hot_count = 0
        ctrl.cold_rounds = []
        ctrl.cold_active = []

        c9_times, c9_fwds = [], []
        for ri in range(2):
            ctrl.cold_count = 0
            ctrl.hot_count = 0
            ctrl.cold_rounds = []
            ctrl.cold_active = []

            dllm = make_dllm(decoder_t0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            c9_times.append(t1 - t0)
            c9_fwds.append(dllm.diff_iteration.num_forwards)
            s = ctrl.summary()
            print(f"  Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd | "
                  f"cold={s['cold_count']} hot={s['hot_count']} ({s['hot_pct']:.1f}%) "
                  f"cold_avg_rounds={s['cold_avg_rounds']:.1f} cold_avg_|S|={s['cold_avg_active']:.1f}")

        c9_avg = sum(c9_times) / len(c9_times)
        c9_fwd = sum(c9_fwds) / len(c9_fwds)
        c0_ref = 8.792
        delta_c5 = (c9_avg - c5_avg) / c5_avg * 100
        delta_c0 = (c9_avg - c0_ref) / c0_ref * 100
        final_s = ctrl.summary()

        print(f"\n  C9 avg: {c9_avg:.3f}s, {c9_fwd:.0f} fwd, {c9_fwd/c9_avg:.1f} fwd/s, "
              f"{c9_avg/c9_fwd*1000:.2f} ms/fwd")
        print(f"  vs C5: {delta_c5:+.1f}%  vs C0: {delta_c0:+.1f}%")
        print(f"  Fwd: C5={c5_fwd:.0f} → C9={c9_fwd:.0f} (delta={c9_fwd-c5_fwd:+.0f})")
        print(f"  Path split: cold={final_s['cold_count']}, hot={final_s['hot_count']} ({final_s['hot_pct']:.1f}% hot)")
        print(f"  Cold starts: avg_rounds={final_s['cold_avg_rounds']:.1f}, avg_|S|={final_s['cold_avg_active']:.1f}")

        results["C9"] = {
            "avg_time": c9_avg, "avg_fwd": c9_fwd,
            "fwd_per_s": c9_fwd / c9_avg, "ms_per_fwd": c9_avg / c9_fwd * 1000,
            "delta_vs_c5_pct": delta_c5, "delta_vs_c0_pct": delta_c0,
            "path_stats": final_s,
        }

        # Quality check
        print(f"\n  Quality check (temp=0.7)...")
        decoder_t7 = ThresholdParallelDecoder(temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm_q = make_dllm(decoder_t7)
        with torch.inference_mode():
            _ = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        with torch.inference_mode():
            out = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        gen_tokens = out[:, prompt_len:]
        quality = {}
        for bi in sorted(VERIFIABLE.keys()):
            gt = gen_tokens[bi]
            valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
            text = tokenizer.decode(valid, skip_special_tokens=True)
            quality[bi] = text[:300]
            print(f"    #{bi}: {text[:150]}")
        results["C9"]["quality"] = quality
        restore()

        # ============================================================
        # Summary
        # ============================================================
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"  C0 reference: {c0_ref:.3f}s, 262 fwd")
        print(f"  {'Config':<25s} {'Time':>7s} {'Fwd':>5s} {'Fwd/s':>6s} {'ms/fwd':>7s} {'vs C5':>7s} {'vs C0':>7s}")
        print(f"  {'-'*70}")
        for cn, r in results.items():
            d5 = f"{r.get('delta_vs_c5_pct', 0):+.1f}%" if 'delta_vs_c5_pct' in r else "—"
            d0 = f"{r.get('delta_vs_c0_pct', (r['avg_time']-c0_ref)/c0_ref*100):+.1f}%"
            print(f"  {cn:<25s} {r['avg_time']:>7.3f} {r['avg_fwd']:>5.0f} {r['fwd_per_s']:>6.1f} "
                  f"{r['ms_per_fwd']:>7.2f} {d5:>7s} {d0:>7s}")

        out_path = REPO_ROOT / "codex_coding" / "results" / "plan_a_optimized_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
