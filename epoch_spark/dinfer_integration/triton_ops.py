"""
Pre-allocated KV Cache + Triton Fused Routing.

Optimizations:
1. Pre-alloc KV buffer: eliminates torch.cat in DynamicCache.update()
2. Triton fused routing: single kernel for sigmoid+group_topk+expert_topk
"""
import torch
import triton
import triton.language as tl


# ════════════════════════════════════════════════════════════════
# Pre-allocated KV Cache (no torch.cat, just copy_)
# ════════════════════════════════════════════════════════════════

class PreallocKVCache:
    """Pre-allocated KV cache that avoids torch.cat overhead.

    Instead of DynamicCache which does cat() on every update,
    this pre-allocates the full buffer and uses slice assignment.
    """

    def __init__(self, n_layers, max_seq_len, n_kv_heads, head_dim, batch_size, device, dtype=torch.bfloat16):
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.batch_size = batch_size

        # Pre-allocate: [n_layers, 2, batch, n_kv_heads, max_seq, head_dim]
        self.k_cache = torch.zeros(n_layers, batch_size, n_kv_heads, max_seq_len, head_dim,
                                    device=device, dtype=dtype)
        self.v_cache = torch.zeros(n_layers, batch_size, n_kv_heads, max_seq_len, head_dim,
                                    device=device, dtype=dtype)
        self.seq_len = 0  # current valid length

    def update(self, key_states, value_states, layer_idx):
        """Write new K/V into cache at current position. Does NOT advance seq_len.

        key_states: [batch, n_kv_heads, new_seq_len, head_dim]
        Returns full K/V up to current_pos + new_len.
        """
        new_len = key_states.shape[2]
        end = self.seq_len + new_len
        self.k_cache[layer_idx, :, :, self.seq_len:end, :] = key_states
        self.v_cache[layer_idx, :, :, self.seq_len:end, :] = value_states
        return self.k_cache[layer_idx, :, :, :end, :], self.v_cache[layer_idx, :, :, :end, :]

    def advance(self, n_tokens):
        """Advance seq_len after all layers have been updated."""
        self.seq_len += n_tokens

    def crop(self, new_len):
        """Trim back to new_len (for block iteration reuse)."""
        self.seq_len = new_len

    def get_seq_length(self):
        return self.seq_len

    def get_kv(self, layer_idx):
        """Get current K/V for a layer."""
        return (self.k_cache[layer_idx, :, :, :self.seq_len, :],
                self.v_cache[layer_idx, :, :, :self.seq_len, :])


# ════════════════════════════════════════════════════════════════
# Triton Fused Routing (replaces Python topk+sort)
# ════════════════════════════════════════════════════════════════

@triton.jit
def _fused_routing_v2(
    logits_ptr, bias_ptr, ids_ptr, wts_ptr,
    N, rsf,
    stride_ln, stride_le,
    stride_in, stride_ik,
    stride_wn, stride_wk,
    E: tl.constexpr, K: tl.constexpr,
    NG: tl.constexpr, TKG: tl.constexpr, GS: tl.constexpr,
):
    """Fused: sigmoid → group top-2 sum → top-TKG groups → top-K experts → normalize."""
    pid = tl.program_id(0)
    if pid >= N:
        return

    oe = tl.arange(0, E)
    lg = tl.load(logits_ptr + pid * stride_ln + oe * stride_le)
    bi = tl.load(bias_ptr + oe)
    sc = tl.sigmoid(lg)
    sb = sc + bi

    # Group scoring
    gs = tl.zeros([NG], dtype=tl.float32)
    for g in tl.static_range(NG):
        go = tl.arange(0, GS)
        gv = tl.load(logits_ptr + pid * stride_ln + (g * GS + go) * stride_le)
        gsc = tl.sigmoid(gv) + tl.load(bias_ptr + g * GS + go)
        m1 = tl.max(gsc, 0)
        gsc2 = tl.where(gsc == m1, float('-inf'), gsc)
        m2 = tl.max(gsc2, 0)
        m2 = tl.where(m2 == float('-inf'), 0.0, m2)
        gs = tl.where(tl.arange(0, NG) == g, m1 + m2, gs)

    # Top-TKG groups
    gm = tl.zeros([NG], dtype=tl.int32)
    gt = gs
    for _ in tl.static_range(TKG):
        gi = tl.argmax(gt, 0)
        gm = tl.where(tl.arange(0, NG) == gi, 1, gm)
        gt = tl.where(tl.arange(0, NG) == gi, float('-inf'), gt)

    # Expert group mask
    eg = oe // GS
    ea = tl.zeros([E], dtype=tl.int32)
    for g in tl.static_range(NG):
        ig = (eg == g)
        gsel = tl.sum(tl.where(tl.arange(0, NG) == g, gm, tl.zeros([NG], dtype=tl.int32)))
        ea = tl.where(ig, gsel, ea)

    # Top-K within allowed
    ms = tl.where(ea == 1, sb, float('-inf'))
    ti = tl.zeros([K], dtype=tl.int32)
    mt = ms
    for _k in tl.static_range(K):
        bx = tl.argmax(mt, 0)
        ti = tl.where(tl.arange(0, K) == _k, bx, ti)
        mt = tl.where(oe == bx, float('-inf'), mt)

    # Weights
    ts = tl.zeros([K], dtype=tl.float32)
    for _k in tl.static_range(K):
        ix = tl.sum(tl.where(tl.arange(0, K) == _k, ti, tl.zeros([K], dtype=tl.int32)))
        vl = tl.sum(tl.where(oe == ix, sc, tl.zeros([E], dtype=tl.float32)))
        ts = tl.where(tl.arange(0, K) == _k, vl, ts)
    ss = tl.sum(ts, 0) + 1e-20
    tw = ts / ss * rsf

    ok = tl.arange(0, K)
    tl.store(ids_ptr + pid * stride_in + ok * stride_ik, ti)
    tl.store(wts_ptr + pid * stride_wn + ok * stride_wk, tw)


def triton_routing(logits, bias, rsf, K=8, ng=8, tkg=4):
    """Call Triton fused routing. Returns (weights[N,K], ids[N,K])."""
    N, E = logits.shape
    ids = torch.empty((N, K), dtype=torch.int32, device=logits.device)
    wts = torch.empty((N, K), dtype=torch.float32, device=logits.device)
    lf = logits.float()
    bf = bias.float()
    _fused_routing_v2[(N,)](
        lf, bf, ids, wts, N, rsf,
        lf.stride(0), lf.stride(1),
        ids.stride(0), ids.stride(1),
        wts.stride(0), wts.stride(1),
        E=E, K=K, NG=ng, TKG=tkg, GS=E // ng,
    )
    return wts, ids
