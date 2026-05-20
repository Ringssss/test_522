# WarmStart `compute_s_mask` Triton 优化方案（最终整理版）

这份文档整理了当前建议用于替代原始 Python 实现的 **Triton 热路径方案**，目标是：

- 保留算法上的核心要求：
  - `KEXT = 12`（硬性要求，不做 8/6 的近似）
  - 最终输出 `s_mask[256]`
  - `s_mask` 中 `1` 的个数严格等于 `K_init`
- 优化当前 Triton 实现中的明显低效点：
  - 避免原先 `K_B` 的 **单 program + 256 轮 iterative argmax**
  - 避免 `torch.topk(popularity) + zeros + index_put` 的 **3 次 Python dispatch**
- 形成一套适合在本地 GPU 环境中进一步 benchmark 的实现骨架

---

# 1. 背景与关键结论

## 1.1 原始 Python 语义

原始 Python 路径（来自 `WarmStartEBController.compute_s_mask()`）的核心语义是：

```python
scores_full = torch.sigmoid(logits.float())                    # [N, 256]
topkm_score, topkm_idx = torch.topk(scores_full, k=12, dim=1) # [N, 12]
topkm_weight = topkm_score / (topkm_score.sum(dim=1, keepdim=True) + 1e-20) * rsf

popularity = torch.zeros(256, device=logits.device, dtype=torch.float32)
popularity.scatter_add_(0, topkm_idx.reshape(-1), topkm_weight.reshape(-1))

_, pop_order = popularity.sort(descending=True)
S_mask = torch.zeros(256, dtype=torch.bool, device=logits.device)
S_mask[pop_order[:K_init]] = True
```

也就是说，**原始 Python 版并没有显式加 `expert_bias`**。  
如果你们实际线上逻辑确实要加 bias，那么必须确认：

- bias 是否已在更前面的 logits 生成路径中被融合
- 或者 Triton 版是否在这里引入了新的语义

> **重要提醒：如果 Python baseline 没有 bias，而 Triton 版加了 bias，那么这不是“优化”，而是“语义改变”。**

---

## 1.2 当前建议

当前建议保留两阶段结构，但优化实现：

- **K_A（Triton）**：从 `logits[N, 256]` 计算 `popularity[256]`
- **K_B（Triton）**：单次 dispatch，直接从 `popularity[256]` 生成 `s_mask[256]`，并顺手清零 `popularity`

这样做的好处：

- 避免 `torch.topk + zeros + index_put` 的 3 次 dispatch
- 避免旧版 `K_B` 的低效串行 iterative argmax
- 仍然保留和原始 Python 较接近的语义

---

# 2. 关键信息与已知注意点

## 2.1 `KEXT=12` 保持不变
这里明确按你的要求保留：

- `KEXT = 12`

不做 `8` 或 `6` 的近似。

---

## 2.2 关于 bias
下面给出的 `K_A` 代码会把 bias 设计成**可选输入**，原因是：

- 你当前描述的 Triton 版里有 `scores = sigmoid(logits) + bias`
- 但原始 Python 版代码没有这一项

因此 wrapper 中提供两种模式：

- `bias is None`：严格对齐 Python 原版语义
- `bias is not None`：按你当前 Triton 语义执行

建议你们本地 benchmark 时分别对比这两种情况。

---

## 2.3 关于 `K_B` 的 tie-break
`K_B` 本质上是在 `popularity[256]` 上做 top-`K_init` selection。  
当存在相等值（ties）时，不同实现的 tie-break 规则可能不完全一致。

本文给出的 K_B 使用如下 tie-break：

- 所有 `pop > threshold` 的一定入选
- `pop == threshold` 的按**原始 expert index 升序**补齐到 `K_init`

这能保证：

- 输出 mask 中 `1` 的数量严格等于 `K_init`
- 行为是确定性的

但在极少数 tie case 下，**与 PyTorch sort/topk 的具体 tie-break 可能不同**。

---

## 2.4 Triton 版本要求
本文中的 K_B 依赖：

- `tl.sort`
- `tl.cumsum`

你本地需要确认 Triton 版本支持它们，例如：

```python
import triton.language as tl
print(hasattr(tl, "sort"))
print(hasattr(tl, "cumsum"))
```

如果缺少其中之一，需要写 fallback 版本。

---

# 3. Triton K_A：计算 popularity（保留 `KEXT=12`）

## 3.1 设计说明

这个版本的 K_A 采用“一个 program 处理 `BLOCK_N` 个 token”的方式，思路是：

- 对每个 token：
  - 读 `logits[token, :]`
  - `sigmoid`
  - 做 top-12
  - normalize
- 在 program 内维护一个局部 `local_pop[256]`
- 最后再对 global `popularity[256]` 做一次 `atomic_add`

相比“每个 token 直接对全局 popularity 做 12 次 atomic_add”，它更有机会减少全局原子写的开销。

---

## 3.2 代码

```python
import torch
import triton
import triton.language as tl


@triton.jit
def warmstart_popularity_kernel_v2(
    logits_ptr,          # [N, E]
    bias_ptr,            # [E] or dummy
    pop_ptr,             # [E] fp32
    N,
    stride_n,
    stride_e,
    rsf,
    use_bias,            # runtime int: 0 / 1
    E: tl.constexpr,     # 256
    BLOCK_N: tl.constexpr,
    KEXT: tl.constexpr,  # fixed to 12
):
    pid = tl.program_id(0)

    tok_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    expert_offsets = tl.arange(0, E)

    bias = tl.load(bias_ptr + expert_offsets).to(tl.float32)
    local_pop = tl.zeros([E], dtype=tl.float32)

    for i in tl.static_range(BLOCK_N):
        tok = tok_offsets[i]
        valid_tok = tok < N

        lg = tl.load(
            logits_ptr + tok * stride_n + expert_offsets * stride_e,
            mask=valid_tok,
            other=-float("inf"),
        ).to(tl.float32)

        scores = tl.sigmoid(lg)
        if use_bias:
            scores = scores + bias

        st = scores
        top_idx = tl.zeros([KEXT], dtype=tl.int32)
        top_val = tl.zeros([KEXT], dtype=tl.float32)

        # exact top-12 by iterative argmax inside one token
        for k in tl.static_range(KEXT):
            bx = tl.argmax(st, axis=0)
            bv = tl.max(st, axis=0)
            top_idx[k] = bx
            top_val[k] = bv
            st = tl.where(expert_offsets == bx, -float("inf"), st)

        s_sum = tl.sum(top_val, axis=0) + 1e-20
        top_w = top_val / s_sum * rsf

        # accumulate into local_pop
        for k in tl.static_range(KEXT):
            local_pop = tl.where(
                expert_offsets == top_idx[k],
                local_pop + top_w[k],
                local_pop,
            )

    tl.atomic_add(pop_ptr + expert_offsets, local_pop)
```

---

# 4. Triton K_B：1 次 dispatch，直接写 `s_mask`

## 4.1 设计说明

这个版本的 K_B 做的是：

- 一次加载 `popularity[256]`
- 在 program 内排序
- 找到第 `K_init-1` 大的阈值
- 直接生成 `s_mask`
- 清零 `popularity`

相比旧版 K_B：

- 不再做 `256` 轮 iterative argmax
- 不再单纯为了 mask 做一长串串行依赖循环
- 目标是替代：
  - `torch.topk(popularity, k=k_init)`
  - `torch.zeros(...)`
  - `s_mask[top_idx] = True`

---

## 4.2 代码

```python
@triton.jit
def warmstart_select_mask_kernel_v3(
    pop_ptr,          # [256] float32
    s_mask_ptr,       # [256] int32
    K_init,           # runtime int
    E: tl.constexpr,  # must be 256
):
    offs = tl.arange(0, E)

    pop = tl.load(pop_ptr + offs).to(tl.float32)

    # sort descending inside one program
    pop_sorted = tl.sort(pop, descending=True)

    # threshold = pop_sorted[K_init - 1]
    kth_idx = K_init - 1
    neg_inf = -float("inf")
    threshold = tl.max(
        tl.where(offs == kth_idx, pop_sorted, neg_inf),
        axis=0,
    )

    gt_mask = pop > threshold
    gt_count = tl.sum(gt_mask.to(tl.int32), axis=0)

    eq_mask = pop == threshold
    need_eq = K_init - gt_count

    # tie-break among equal values by original expert index order
    eq_rank = tl.cumsum(eq_mask.to(tl.int32), axis=0)
    take_eq = eq_mask & (eq_rank <= need_eq)

    sel = gt_mask | take_eq

    tl.store(s_mask_ptr + offs, sel.to(tl.int32))

    # clear popularity
    tl.store(pop_ptr + offs, tl.zeros([E], dtype=tl.float32))
```

---

# 5. Python wrapper（最终版）

下面给出一版完整 wrapper，包含：

- popularity 计算
- s_mask 生成
- bias 可选
- 输出 `int32` mask，方便后续 Triton kernel 使用

```python
import torch
import triton
import triton.language as tl


def compute_s_mask_triton_final(
    logits: torch.Tensor,              # [N, 256], CUDA
    k_init: int,
    rsf: float = 2.5,
    bias: torch.Tensor | None = None,  # [256], optional
    block_n: int = 16,
    return_popularity: bool = False,
):
    assert logits.is_cuda, "logits must be CUDA tensor"
    assert logits.ndim == 2
    N, E = logits.shape
    assert E == 256, f"expected E=256, got {E}"

    logits_f32 = logits.float().contiguous()

    if bias is None:
        bias_buf = torch.zeros(E, device=logits.device, dtype=torch.float32)
        use_bias = 0
    else:
        assert bias.is_cuda
        assert bias.numel() == E
        bias_buf = bias.float().contiguous()
        use_bias = 1

    popularity = torch.zeros(E, device=logits.device, dtype=torch.float32)
    s_mask = torch.empty(E, device=logits.device, dtype=torch.int32)

    grid_a = (triton.cdiv(N, block_n),)
    warmstart_popularity_kernel_v2[grid_a](
        logits_f32,
        bias_buf,
        popularity,
        N,
        logits_f32.stride(0),
        logits_f32.stride(1),
        rsf,
        use_bias,
        E=256,
        BLOCK_N=block_n,
        KEXT=12,
        num_warps=4,
        num_stages=1,
    )

    if k_init <= 0:
        s_mask.zero_()
        popularity.zero_()
    elif k_init >= 256:
        s_mask.fill_(1)
        popularity.zero_()
    else:
        warmstart_select_mask_kernel_v3[(1,)](
            popularity,
            s_mask,
            k_init,
            E=256,
            num_warps=4,
            num_stages=1,
        )

    if return_popularity:
        return s_mask, popularity
    return s_mask
```

---

# 6. 推荐的 benchmark 对照项

建议你在本地 GPU 环境至少对比下面几组：

## 6.1 原始 Python 基线
```python
def compute_s_mask_python_ref(logits, k_init, rsf=2.5):
    scores_full = torch.sigmoid(logits.float())
    topkm_score, topkm_idx = torch.topk(scores_full, k=12, dim=1)
    topkm_weight = topkm_score / (topkm_score.sum(dim=1, keepdim=True) + 1e-20) * rsf

    popularity = torch.zeros(256, device=logits.device, dtype=torch.float32)
    popularity.scatter_add_(0, topkm_idx.reshape(-1), topkm_weight.reshape(-1))

    _, pop_order = popularity.sort(descending=True)
    s_mask = torch.zeros(256, dtype=torch.bool, device=logits.device)
    s_mask[pop_order[:k_init]] = True
    return s_mask
```

## 6.2 当前旧版 Triton
- 你现在的 K_A + 旧 K_B iterative argmax

## 6.3 本文推荐版本
- `compute_s_mask_triton_final(..., bias=None)`
- `compute_s_mask_triton_final(..., bias=expert_bias)`（如果你们真实逻辑要求 bias）

---

# 7. 建议重点检查的正确性项

## 7.1 `s_mask.sum() == K_init`
必须保证：

```python
assert int(s_mask.sum().item()) == k_init
```

---

## 7.2 与 Python 基线的集合差异
可比较：

```python
ref = compute_s_mask_python_ref(logits, k_init)
out = compute_s_mask_triton_final(logits, k_init).bool()

overlap = (ref & out).sum().item() / k_init
jaccard = (ref & out).sum().item() / (ref | out).sum().item()
```

---

## 7.3 bias 模式的正确性
如果你们最终决定要保留 bias，请一定确认 baseline 语义。  
否则会出现：

- “Triton 版更快但结果不一样”
- 实际上不是优化误差，而是语义根本不一致

---

# 8. 目前我最关心的两个潜在风险

## 8.1 `tl.sort` / `tl.cumsum` 的版本兼容
不同 Triton 版本支持情况可能不同。  
如果编译不过，优先确认是不是这两个 API 的版本问题。

---

## 8.2 K_A 的 `local_pop[256]` 资源占用
K_A 在每个 program 内维护一个 `local_pop[256]`，这会占用一定寄存器/本地资源。  
在你本地测试时建议 sweep：

- `BLOCK_N = 8`
- `BLOCK_N = 16`
- `BLOCK_N = 32`

常见情况是：

- `BLOCK_N` 太小：launch 数多
- `BLOCK_N` 太大：寄存器压力和 occupancy 可能变差

---

# 9. 我当前建议的测试顺序

## 第一阶段：先验证正确性
1. `bias=None`
2. 与 Python 基线比较
3. 检查 `s_mask.sum() == K_init`

## 第二阶段：测 K_B 是否真的收益明显
比较：

- `torch.topk + zeros + index_put`
- `warmstart_select_mask_kernel_v3`

## 第三阶段：再测端到端
比较：

- Python 原版
- 旧 Triton 版
- 本文 Triton 版

---

# 10. 最后总结

当前最终建议版本是：

- **K_A**：Triton，保留 `KEXT=12`
- **K_B**：Triton，1 次 dispatch 直接写 `s_mask`
- **bias**：明确做成可选，避免默默引入与 Python 原版不一致的语义

这版最核心的目标不是激进近似，而是：

- 去掉你当前 K_B 的明显低效实现
- 减少 Python dispatch
- 保持和原始算法尽量一致

如果本地测试通过，你们下一步再决定是否继续做更激进的 K_A 优化或算法级重构。
