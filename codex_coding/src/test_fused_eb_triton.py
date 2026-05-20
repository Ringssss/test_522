#!/usr/bin/env python3
"""
v0.1.15.7e — Fused EB Triton Kernels (Stage 1+2)

4 Triton kernels for Expert Budgeting:
  K_A: per-token sigmoid + topk(12) + normalize + atomic popularity (hot+cold)
  K_B: global sort popularity → S_mask (hot+cold)
  K_C: per-token coverage check + batch-add scoring (cold only)
  K_D: global sat check + expert selection + S update (cold only)

Hot path: K_A + K_B (3 dispatch, 0 sync)
Cold path: K_A_cold + K_B + MAX_ROUNDS×(K_C + K_D) (≤34 dispatch, 0 sync)
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
# Fused Routing Kernel with S_mask (from v0.1.15.7)
# ================================================================
@triton.jit
def _fused_routing_k(
    logits_ptr, bias_ptr, s_mask_ptr, ids_ptr, wts_ptr,
    N, rsf, sl_n, sl_e, si_n, si_k, sw_n, sw_k,
    HAS_S: tl.constexpr, E: tl.constexpr, K: tl.constexpr,
    NG: tl.constexpr, TKG: tl.constexpr, GS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N: return
    oe = tl.arange(0, E)
    lg = tl.load(logits_ptr + pid * sl_n + oe * sl_e)
    bi = tl.load(bias_ptr + oe)
    sc = tl.sigmoid(lg); sb = sc + bi
    gs = tl.zeros([NG], dtype=tl.float32)
    for g in tl.static_range(NG):
        go = tl.arange(0, GS)
        gv = tl.load(logits_ptr + pid * sl_n + (g*GS+go)*sl_e)
        gsc = tl.sigmoid(gv) + tl.load(bias_ptr + g*GS+go)
        m1 = tl.max(gsc,0); gsc2 = tl.where(gsc==m1,float('-inf'),gsc)
        m2 = tl.max(gsc2,0); m2 = tl.where(m2==float('-inf'),0.0,m2)
        gs = tl.where(tl.arange(0,NG)==g, m1+m2, gs)
    gm = tl.zeros([NG],dtype=tl.int32); gt=gs
    for _ in tl.static_range(TKG):
        gi=tl.argmax(gt,0); gm=tl.where(tl.arange(0,NG)==gi,1,gm); gt=tl.where(tl.arange(0,NG)==gi,float('-inf'),gt)
    eg=oe//GS; ea=tl.zeros([E],dtype=tl.int32)
    for g in tl.static_range(NG):
        ig=(eg==g); gsel=tl.sum(tl.where(tl.arange(0,NG)==g,gm,tl.zeros([NG],dtype=tl.int32)))
        ea=tl.where(ig,gsel,ea)
    if HAS_S:
        sm=tl.load(s_mask_ptr+oe); ea=ea&sm
    ms=tl.where(ea==1,sb,float('-inf'))
    ti=tl.zeros([K],dtype=tl.int32); mt=ms
    for _k in tl.static_range(K):
        bx=tl.argmax(mt,0); ti=tl.where(tl.arange(0,K)==_k,bx,ti); mt=tl.where(oe==bx,float('-inf'),mt)
    ts=tl.zeros([K],dtype=tl.float32)
    for _k in tl.static_range(K):
        ix=tl.sum(tl.where(tl.arange(0,K)==_k,ti,tl.zeros([K],dtype=tl.int32)))
        vl=tl.sum(tl.where(oe==ix,sc,tl.zeros([E],dtype=tl.float32)))
        ts=tl.where(tl.arange(0,K)==_k,vl,ts)
    ss=tl.sum(ts,0)+1e-20; tw=ts/ss*rsf
    ok=tl.arange(0,K)
    tl.store(ids_ptr+pid*si_n+ok*si_k,ti); tl.store(wts_ptr+pid*sw_n+ok*sw_k,tw)

def fused_routing(logits, bias, rsf, s_mask=None, K=8, ng=8, tkg=4):
    N,E=logits.shape; ids=torch.empty((N,K),dtype=torch.int32,device=logits.device)
    wts=torch.empty((N,K),dtype=torch.float32,device=logits.device)
    lf=logits.float(); bf=bias.float()
    hm=s_mask is not None; sp=s_mask if hm else torch.empty(0,dtype=torch.int32,device=logits.device)
    _fused_routing_k[(N,)](lf,bf,sp,ids,wts,N,rsf,lf.stride(0),lf.stride(1),ids.stride(0),ids.stride(1),wts.stride(0),wts.stride(1),HAS_S=hm,E=E,K=K,NG=ng,TKG=tkg,GS=E//ng)
    return wts,ids

# ================================================================
# K_A: Per-token expand + atomic popularity (hot path version)
# ================================================================
@triton.jit
def _kernel_A(
    logits_ptr, bias_ptr, pop_ptr,
    N, rsf,
    sl_n, sl_e,
    E: tl.constexpr, KEXT: tl.constexpr, KEXT_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N: return
    oe = tl.arange(0, E)
    lg = tl.load(logits_ptr + pid * sl_n + oe * sl_e)
    bi = tl.load(bias_ptr + oe)
    scores = tl.sigmoid(lg) + bi  # [E]

    # topk(KEXT=12) via iterative argmax, padded to KEXT_PAD=16
    topkm_idx = tl.zeros([KEXT_PAD], dtype=tl.int32)
    topkm_score = tl.zeros([KEXT_PAD], dtype=tl.float32)
    st = scores
    for _k in tl.static_range(KEXT):
        bx = tl.argmax(st, 0)
        bv = tl.max(st, 0)
        topkm_idx = tl.where(tl.arange(0, KEXT_PAD) == _k, bx, topkm_idx)
        topkm_score = tl.where(tl.arange(0, KEXT_PAD) == _k, bv, topkm_score)
        st = tl.where(oe == bx, float('-inf'), st)

    # normalize (only first KEXT elements are valid)
    valid = tl.arange(0, KEXT_PAD) < KEXT
    topkm_score = tl.where(valid, topkm_score, tl.zeros([KEXT_PAD], dtype=tl.float32))
    s_sum = tl.sum(topkm_score, 0) + 1e-20
    topkm_w = topkm_score / s_sum * rsf

    # atomic add to popularity
    for _k in tl.static_range(KEXT):
        idx = tl.sum(tl.where(tl.arange(0, KEXT_PAD) == _k, topkm_idx, tl.zeros([KEXT_PAD], dtype=tl.int32)))
        w = tl.sum(tl.where(tl.arange(0, KEXT_PAD) == _k, topkm_w, tl.zeros([KEXT_PAD], dtype=tl.float32)))
        tl.atomic_add(pop_ptr + idx, w)


# K_A_cold: same as K_A but also stores topkm data + quality threshold r
@triton.jit
def _kernel_A_cold(
    logits_ptr, bias_ptr, pop_ptr,
    topkm_idx_ptr, topkm_w_ptr, r_ptr,
    N, rsf, quality_floor,
    sl_n, sl_e, st_n, st_k,
    E: tl.constexpr, KEXT: tl.constexpr, KEXT_PAD: tl.constexpr, K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N: return
    oe = tl.arange(0, E)
    lg = tl.load(logits_ptr + pid * sl_n + oe * sl_e)
    bi = tl.load(bias_ptr + oe)
    scores = tl.sigmoid(lg) + bi

    topkm_idx = tl.zeros([KEXT_PAD], dtype=tl.int32)
    topkm_score = tl.zeros([KEXT_PAD], dtype=tl.float32)
    st = scores
    for _k in tl.static_range(KEXT):
        bx = tl.argmax(st, 0)
        bv = tl.max(st, 0)
        topkm_idx = tl.where(tl.arange(0, KEXT_PAD) == _k, bx, topkm_idx)
        topkm_score = tl.where(tl.arange(0, KEXT_PAD) == _k, bv, topkm_score)
        st = tl.where(oe == bx, float('-inf'), st)

    valid = tl.arange(0, KEXT_PAD) < KEXT
    topkm_score = tl.where(valid, topkm_score, tl.zeros([KEXT_PAD], dtype=tl.float32))
    s_sum = tl.sum(topkm_score, 0) + 1e-20
    topkm_w = topkm_score / s_sum * rsf

    for _k in tl.static_range(KEXT):
        idx = tl.sum(tl.where(tl.arange(0, KEXT_PAD) == _k, topkm_idx, tl.zeros([KEXT_PAD], dtype=tl.int32)))
        w = tl.sum(tl.where(tl.arange(0, KEXT_PAD) == _k, topkm_w, tl.zeros([KEXT_PAD], dtype=tl.float32)))
        tl.atomic_add(pop_ptr + idx, w)

    # Store topkm data (only first KEXT elements, padded storage)
    ok = tl.arange(0, KEXT_PAD)
    store_mask = ok < KEXT
    tl.store(topkm_idx_ptr + pid * st_n + ok * st_k, topkm_idx, mask=store_mask)
    tl.store(topkm_w_ptr + pid * st_n + ok * st_k, topkm_w, mask=store_mask)

    # Quality threshold: r = QF * sum(top-K expanded weights)
    top_k_sum = tl.sum(tl.where(tl.arange(0, KEXT_PAD) < K, topkm_w, tl.zeros([KEXT_PAD], dtype=tl.float32)), 0)
    r = quality_floor * top_k_sum
    tl.store(r_ptr + pid, r)


# ================================================================
# K_B v2: Global sort popularity → S_mask + zero pop (all in 1 kernel)
# ================================================================
@triton.jit
def _kernel_B_v2(
    pop_ptr, s_mask_ptr, K_init,
    E: tl.constexpr, MAX_K: tl.constexpr,
):
    """Iterative argmax on popularity[E] → S_mask. Also zeros pop for next K_A."""
    oe = tl.arange(0, E)
    pop = tl.load(pop_ptr + oe)

    s_mask = tl.zeros([E], dtype=tl.int32)
    pt = pop
    for _k in tl.static_range(MAX_K):
        bx = tl.argmax(pt, 0)
        # Only mark if _k < K_init (runtime comparison)
        s_mask = tl.where((oe == bx) & (_k < K_init), 1, s_mask)
        pt = tl.where(oe == bx, float('-inf'), pt)

    tl.store(s_mask_ptr + oe, s_mask)
    # Zero popularity for next K_A call
    tl.store(pop_ptr + oe, tl.zeros([E], dtype=tl.float32))


# ================================================================
# K_B v3: tl.sort based (replaces iterative argmax, 1.65x faster)
# ================================================================
@triton.jit
def _kernel_B_v3(
    pop_ptr, s_mask_ptr, K_init,
    E: tl.constexpr,
):
    """Sort-based popularity → S_mask. Also zeros pop for next K_A."""
    offs = tl.arange(0, E)
    pop = tl.load(pop_ptr + offs).to(tl.float32)

    pop_sorted = tl.sort(pop, descending=True)

    kth_idx = K_init - 1
    threshold = tl.max(
        tl.where(offs == kth_idx, pop_sorted, -float("inf")),
        axis=0,
    )

    gt_mask = pop > threshold
    gt_count = tl.sum(gt_mask.to(tl.int32), axis=0)

    eq_mask = pop == threshold
    need_eq = K_init - gt_count
    eq_rank = tl.cumsum(eq_mask.to(tl.int32), axis=0)
    take_eq = eq_mask & (eq_rank <= need_eq)

    sel = gt_mask | take_eq
    tl.store(s_mask_ptr + offs, sel.to(tl.int32))
    tl.store(pop_ptr + offs, tl.zeros([E], dtype=tl.float32))


# ================================================================
# K_C: Per-token coverage check + batch-add scoring
# ================================================================
@triton.jit
def _kernel_C(
    topkm_idx_ptr, topkm_w_ptr, r_ptr,
    s_mask_ptr, sat_flag_ptr,
    sat_count_ptr, G_ptr, H_ptr,
    N, st_n, st_k,
    E: tl.constexpr, KEXT: tl.constexpr, KEXT_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N: return

    # Early exit if already satisfied
    if tl.load(sat_flag_ptr) == 1: return

    ok = tl.arange(0, KEXT_PAD)
    load_mask = ok < KEXT
    idx = tl.load(topkm_idx_ptr + pid * st_n + ok * st_k, mask=load_mask, other=0)
    w = tl.load(topkm_w_ptr + pid * st_n + ok * st_k, mask=load_mask, other=0.0)
    r = tl.load(r_ptr + pid)

    oe = tl.arange(0, E)
    s_mask = tl.load(s_mask_ptr + oe)  # [E]

    # Coverage: c = sum(w[j] for j where s_mask[idx[j]]==1)
    c = tl.zeros([1], dtype=tl.float32)
    for _k in tl.static_range(KEXT):
        eid = tl.sum(tl.where(tl.arange(0, KEXT_PAD) == _k, idx, tl.zeros([KEXT_PAD], dtype=tl.int32)))
        wk = tl.sum(tl.where(tl.arange(0, KEXT_PAD) == _k, w, tl.zeros([KEXT_PAD], dtype=tl.float32)))
        in_s = tl.sum(tl.where(oe == eid, s_mask, tl.zeros([E], dtype=tl.int32)))
        c += wk * in_s.to(tl.float32)

    c_val = tl.sum(c, 0)
    if c_val >= r:
        tl.atomic_add(sat_count_ptr, 1)
        return

    # Unsatisfied: score candidate experts not in S
    d = r - c_val
    for _k in tl.static_range(KEXT):
        eid = tl.sum(tl.where(tl.arange(0, KEXT_PAD) == _k, idx, tl.zeros([KEXT_PAD], dtype=tl.int32)))
        wk = tl.sum(tl.where(tl.arange(0, KEXT_PAD) == _k, w, tl.zeros([KEXT_PAD], dtype=tl.float32)))
        in_s = tl.sum(tl.where(oe == eid, s_mask, tl.zeros([E], dtype=tl.int32)))
        if in_s == 0:  # not in S
            gap_gain = tl.minimum(wk, d)
            hit_gain: tl.float32 = 1.0 if (c_val + wk >= r) else 0.0
            tl.atomic_add(G_ptr + eid, gap_gain)
            tl.atomic_add(H_ptr + eid, hit_gain)


# ================================================================
# K_D: Global sat check + expert selection + S update
# ================================================================
@triton.jit
def _kernel_D_v2(
    s_mask_ptr, sat_flag_ptr, sat_count_ptr,
    G_ptr, H_ptr,
    N, q_major_x1000,
    E: tl.constexpr, CAP: tl.constexpr,
):
    # Early exit
    if tl.load(sat_flag_ptr) == 1: return

    sat_count = tl.load(sat_count_ptr)
    if sat_count * 1000 >= q_major_x1000 * N:
        tl.store(sat_flag_ptr, 1)
        # Zero buffers for next round (even though we won't use them, keeps state clean)
        tl.store(sat_count_ptr, 0)
        oe = tl.arange(0, E)
        tl.store(G_ptr + oe, tl.zeros([E], dtype=tl.float32))
        tl.store(H_ptr + oe, tl.zeros([E], dtype=tl.float32))
        return

    oe = tl.arange(0, E)
    s_mask = tl.load(s_mask_ptr + oe)
    G = tl.load(G_ptr + oe)
    H = tl.load(H_ptr + oe)

    score = H + 0.5 * G
    score = tl.where(s_mask == 1, float('-inf'), score)

    for _c in tl.static_range(CAP):
        bx = tl.argmax(score, 0)
        bv = tl.max(score, 0)
        should_add = bv > 0.0
        s_mask = tl.where((oe == bx) & should_add, 1, s_mask)
        score = tl.where(oe == bx, float('-inf'), score)

    tl.store(s_mask_ptr + oe, s_mask)

    # Zero sat_count + G + H for next round (saves 3 .zero_() dispatches)
    tl.store(sat_count_ptr, 0)
    tl.store(G_ptr + oe, tl.zeros([E], dtype=tl.float32))
    tl.store(H_ptr + oe, tl.zeros([E], dtype=tl.float32))


# ================================================================
# Zero-init kernel (cold start: zero pop + sat_flag in 1 dispatch)
# ================================================================
@triton.jit
def _kernel_zero_init(pop_ptr, sat_flag_ptr, sat_count_ptr, G_ptr, H_ptr, E: tl.constexpr):
    oe = tl.arange(0, E)
    tl.store(pop_ptr + oe, tl.zeros([E], dtype=tl.float32))
    tl.store(G_ptr + oe, tl.zeros([E], dtype=tl.float32))
    tl.store(H_ptr + oe, tl.zeros([E], dtype=tl.float32))
    tl.store(sat_flag_ptr, 0)
    tl.store(sat_count_ptr, 0)


# ================================================================
# Python wrappers
# ================================================================

class FusedEBController:
    MAX_ROUNDS = 27  # guarantees K_target + 27*cap = 40+216 = 256 full coverage

    def __init__(self, num_layers=19, K=8, M=4, K_target=40,
                 quality_floor=0.70, q_major=0.95, per_round_cap=8, rsf=2.5):
        self.K = K
        self.M = M
        self.K_ext = K + M
        self.K_target = K_target
        self.quality_floor = quality_floor
        self.q_major = q_major
        self.cap = per_round_cap
        self.rsf = rsf
        self.num_layers = num_layers

        self.K_init = {}     # layer_idx → int
        self.prev_N = {}     # block boundary detection
        self.cold_count = 0
        self.hot_count = 0

        self._bufs = {}

    def _get_bufs(self, N, E, device):
        key = (N, E)
        if key not in self._bufs:
            KE_PAD = 16
            self._bufs[key] = {
                'pop': torch.zeros(E, device=device, dtype=torch.float32),
                's_mask': torch.zeros(E, device=device, dtype=torch.int32),
                'topkm_idx': torch.empty(N, KE_PAD, device=device, dtype=torch.int32),
                'topkm_w': torch.empty(N, KE_PAD, device=device, dtype=torch.float32),
                'r': torch.empty(N, device=device, dtype=torch.float32),
                'sat_flag': torch.zeros(1, device=device, dtype=torch.int32),
                'sat_count': torch.zeros(1, device=device, dtype=torch.int32),
                'G': torch.zeros(E, device=device, dtype=torch.float32),
                'H': torch.zeros(E, device=device, dtype=torch.float32),
            }
        return self._bufs[key]

    def is_new_block(self, layer_idx, N):
        prev = self.prev_N.get(layer_idx, -1)
        self.prev_N[layer_idx] = N
        if prev == -1: return True
        return N > prev

    def hot_path(self, layer_idx, logits, bias):
        """Hot path: K_A + K_B_v3 = 2 Triton dispatches, 0 sync, 0 .zero_()."""
        N, E = logits.shape
        K_init = self.K_init.get(layer_idx, 142)
        b = self._get_bufs(N, E, logits.device)
        # pop is already zeroed by previous K_B_v3 call

        lf = logits.float()
        _kernel_A[(N,)](lf, bias.float(), b['pop'],
                        N, self.rsf, lf.stride(0), lf.stride(1),
                        E=E, KEXT=self.K_ext, KEXT_PAD=16)

        _kernel_B_v3[(1,)](b['pop'], b['s_mask'], K_init, E=E)

        self.hot_count += 1
        return b['s_mask']

    def cold_path(self, layer_idx, logits, bias):
        """Cold path: zero_init + K_A_cold + K_B_v3 + 27×(K_C + K_D_v2)
        GPU early-exits via sat_flag. Max 3 + 27×2 = 57 dispatches, 1 sync (actual |S|)."""
        N, E = logits.shape
        b = self._get_bufs(N, E, logits.device)

        # 1 dispatch: zero all buffers
        _kernel_zero_init[(1,)](b['pop'], b['sat_flag'], b['sat_count'],
                                b['G'], b['H'], E=E)

        lf = logits.float()
        bf = bias.float()

        # K_A_cold
        _kernel_A_cold[(N,)](
            lf, bf, b['pop'],
            b['topkm_idx'], b['topkm_w'], b['r'],
            N, self.rsf, self.quality_floor,
            lf.stride(0), lf.stride(1),
            b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
            E=E, KEXT=self.K_ext, KEXT_PAD=16, K=self.K)

        # K_B_v3: init S from popularity + zero pop
        _kernel_B_v3[(1,)](b['pop'], b['s_mask'], self.K_target, E=E)

        # Batch-add: 16×(K_C + K_D_v2), GPU early-exits via sat_flag
        q_major_x1000 = int(self.q_major * 1000)
        for _ in range(self.MAX_ROUNDS):
            _kernel_C[(N,)](
                b['topkm_idx'], b['topkm_w'], b['r'],
                b['s_mask'], b['sat_flag'],
                b['sat_count'], b['G'], b['H'],
                N, b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                E=E, KEXT=self.K_ext, KEXT_PAD=16)
            _kernel_D_v2[(1,)](
                b['s_mask'], b['sat_flag'], b['sat_count'],
                b['G'], b['H'],
                N, q_major_x1000,
                E=E, CAP=self.cap)

        # Use actual |S| as K_init for hot path (not theoretical max)
        actual_s = int(b['s_mask'].sum().item())
        self.K_init[layer_idx] = actual_s
        self.cold_count += 1
        return b['s_mask']

    def get_s_mask(self, layer_idx, logits, bias):
        N = logits.shape[0]
        if self.is_new_block(layer_idx, N):
            return self.cold_path(layer_idx, logits, bias)
        else:
            return self.hot_path(layer_idx, logits, bias)


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
    print("v0.1.15.7e — Fused EB Triton Kernels (Stage 1+2)")
    print("  Hot: K_A + K_B (3 dispatch, 0 sync)")
    print("  Cold: K_A_cold + K_B + 16×(K_C+K_D) (≤34 dispatch, 0 sync)")
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

        # ---- C5: Fused routing only ----
        print(f"\n{'='*60}")
        print("C5: Fused routing only")
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
                mod.routing = mk(b, r, tk, ng, tkg); idx += 1

        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        c5_t, c5_f = [], []
        for ri in range(2):
            dllm = make_dllm(decoder_t0); torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            c5_t.append(t1-t0); c5_f.append(dllm.diff_iteration.num_forwards)
            print(f"  Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd")
        c5_avg = sum(c5_t)/2; c5_fwd = sum(c5_f)/2
        results["C5"] = {"avg_time": c5_avg, "avg_fwd": c5_fwd, "fwd_per_s": c5_fwd/c5_avg, "ms_per_fwd": c5_avg/c5_fwd*1000}
        print(f"  Avg: {c5_avg:.3f}s, {c5_fwd:.0f} fwd, {c5_avg/c5_fwd*1000:.2f} ms/fwd")
        restore()

        # ---- C10: Fused EB Triton ----
        print(f"\n{'='*60}")
        print("C10: Fused EB Triton (cold/hot, 0 sync)")
        print(f"{'='*60}")

        ctrl = FusedEBController(num_layers=19, K=8, M=4, K_target=40,
                                  quality_floor=0.70, q_major=0.95, per_round_cap=8)
        idx = 0
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                b, r, tk, ng, tkg = mod.expert_bias, mod.routed_scaling_factor, mod.top_k, mod.n_group, mod.topk_group
                li = idx
                def mk(bb, rr, tt, nn, gg, layer_i, cc):
                    def fn(hs, go, topk, renorm):
                        sm = cc.get_s_mask(layer_i, go, bb)
                        w, i = fused_routing(go, bb, rr, s_mask=sm, K=tt, ng=nn, tkg=gg)
                        return w.to(go.dtype), i
                    return fn
                mod.routing = mk(b, r, tk, ng, tkg, li, ctrl); idx += 1
        print(f"  Patched {idx} gates")

        # Warmup
        print("  Warmup...")
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Warmup: cold={ctrl.cold_count}, hot={ctrl.hot_count}")

        ctrl.cold_count = 0; ctrl.hot_count = 0
        c10_t, c10_f = [], []
        for ri in range(2):
            ctrl.cold_count = 0; ctrl.hot_count = 0
            dllm = make_dllm(decoder_t0); torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            c10_t.append(t1-t0); c10_f.append(dllm.diff_iteration.num_forwards)
            print(f"  Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd | "
                  f"cold={ctrl.cold_count} hot={ctrl.hot_count}")

        c10_avg = sum(c10_t)/2; c10_fwd = sum(c10_f)/2
        c0_ref = 8.792
        d5 = (c10_avg - c5_avg) / c5_avg * 100
        d0 = (c10_avg - c0_ref) / c0_ref * 100
        print(f"\n  C10 avg: {c10_avg:.3f}s, {c10_fwd:.0f} fwd, {c10_avg/c10_fwd*1000:.2f} ms/fwd")
        print(f"  vs C5: {d5:+.1f}%  vs C0: {d0:+.1f}%")
        print(f"  Fwd: C5={c5_fwd:.0f} → C10={c10_fwd:.0f}")

        results["C10"] = {"avg_time": c10_avg, "avg_fwd": c10_fwd, "fwd_per_s": c10_fwd/c10_avg,
                          "ms_per_fwd": c10_avg/c10_fwd*1000, "delta_c5": d5, "delta_c0": d0,
                          "cold": ctrl.cold_count, "hot": ctrl.hot_count}

        # Quality
        print(f"\n  Quality check (temp=0.7)...")
        decoder_t7 = ThresholdParallelDecoder(temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        dllm_q = make_dllm(decoder_t7)
        with torch.inference_mode():
            _ = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        with torch.inference_mode():
            out = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        gen_tokens = out[:, prompt_len:]
        for bi in sorted(VERIFIABLE.keys()):
            gt = gen_tokens[bi]; valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
            text = tokenizer.decode(valid, skip_special_tokens=True)
            print(f"    #{bi}: {text[:150]}")
        restore()

        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"  C0 ref: {c0_ref:.3f}s, 262 fwd")
        for cn, r in results.items():
            d5s = f"{r.get('delta_c5',0):+.1f}%" if 'delta_c5' in r else "—"
            d0s = f"{r.get('delta_c0',(r['avg_time']-c0_ref)/c0_ref*100):+.1f}%"
            print(f"  {cn:<10s} {r['avg_time']:.3f}s {r['avg_fwd']:.0f}fwd {r['ms_per_fwd']:.2f}ms/fwd {d5s} {d0s}")

        out_path = REPO_ROOT / "codex_coding" / "results" / "fused_eb_triton_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
