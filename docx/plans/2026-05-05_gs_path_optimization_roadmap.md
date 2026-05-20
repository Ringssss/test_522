# GS Path 优化路线图

## 1. 当前基线

### 1.1 配置

- 路径：GS (BSP-G SourcePath)
- 硬件：8×H100 80GB，NVLink
- 并行：tp=4, dp=2, ep=8, SP enabled
- 模型：LLaDA2.0-mini (hidden=2048, 20 layers, 256 experts, K=4 via EB)
- 推理配置：batch=512, gen=256, block=32, threshold=0.90
- **Baseline: 76.4 ms/fwd (无 timing overhead)**

### 1.2 完整 Component Timing 分解

| 组件 | ms/fwd | 占比 | 备注 |
|------|--------|------|------|
| MoE kernel (quant_apply) | 20.6 | 26.9% | fused_moe + silu + moe_sum |
| MoE combine (EP RS) | 11.6 | 15.2% | NCCL Reduce bf16 + straggler |
| MoE dispatch (EP AG) | 8.7 | 11.4% | NCCL AllGather |
| Attention (QKV+flash+KV) | 8.7 | 11.3% | QKV proj 4.1 + flash 3.4 + KV 1.1 |
| LM head | 7.4 | 9.7% | [16384,2048]×[2048,157184] GEMM |
| Attention TP RS | 5.4 | 7.1% | Attention output reduce-scatter |
| logits.float() | 4.1 | 5.3% | bf16→f32 cast, 9.6GB 搬运 |
| Shared expert | 3.4 | 4.4% | Shared expert MLP |
| Gate + EB routing | 2.7 | 3.6% | Gate logits + fused_routing K=4 |
| TP AllGather | 2.7 | 3.5% | SP→full gather |
| Norms | 1.4 | 1.9% | input + post_attn RMSNorm |
| Dense MLP + embedding | 0.9 | 1.2% | Layer 0 + word embedding |
| **总计** | **77.7** | **~101%** | 略超 baseline 是 timing 残余 |

### 1.3 nsys 关键发现

| 发现 | 数据 | 含义 |
|------|------|------|
| GPU 利用率 | 71% | 29% 时间 GPU 空闲 |
| GPU idle | 23.5 ms/fwd | kernel launch gap 累积 |
| Stream 数量 | 1 | 所有操作完全串行 |
| NCCL-Compute overlap | 0% | 通信和计算无任何重叠 |
| Cross-rank compute CV | <1% | 计算完全均衡 |
| Cross-rank NCCL 差异 | 16% (2.8ms) | dp_rank=1 GPU NCCL 更慢 |
| DPMetadata AllReduce | 1.7 ms/fwd | 纯同步开销（batch 固定时不必要）|
| Kernel launch gap 分布 | median 1.7us, P99 187us | 大量小 gap + 少量大 gap |

### 1.4 关键时间分布图

```
76.4 ms/fwd baseline:
┌─────────────────────────────────────────────────────────────────────────────┐
│ MoE kernel 20.6 │ combine 11.6 │ dispatch 8.7 │ attn 8.7 │lm 7.4│...     │
│     26.9%       │    15.2%     │    11.4%     │   11.3%  │ 9.7% │ 25.5%  │
└─────────────────────────────────────────────────────────────────────────────┘

GPU 利用率:
┌─────────────────────────────────────────────────────────────────────────────┐
│████████████████████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░│
│          Kernel busy 57.8 ms (71%)               │  Idle 23.5 ms (29%)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 已排除方向

| 方向 | 预期收益 | 实测结果 | 排除原因 |
|------|----------|----------|----------|
| EPLB 负载均衡 | ~8.7 ms | ~2 ms (2.6%) | Memory-bound dampening 0.06x；compute 已均衡 CV<1% |
| Tiling config auto-tune | — | 无效 | 实际 M=16384 已最优 |
| CUDA Graph (full forward) | ~8.7 ms (11%) | ~2 ms (2%) | batch=512 时 GPU pipeline 隐藏 Python 开销 |
| torch.compile | — | inductor 不兼容 | EP dispatch 的 .cpu() sync 不被 inductor 支持 |
| Pre-filter block waste | — | -1.8% | Early-exit 在 H100 上零成本 |
| DeepEP AlltoAll | — | +66% | NVLink AllGather 更优 |

## 3. 优化项清单

---

### OPT-1: DPMetadata AllReduce 消除

**问题描述**

每次 `set_forward_context` 调用 `DPMetadata.make()` → NCCL AllReduce_u32 + `.cpu()` 同步。在 BSP-G 路径中，每层每次 forward 调用一次（19 layers × 266 fwd = 5054 次/generate）。

nsys 数据：每次 5us NCCL kernel + 82us CPU sync gap = 87us/layer × 19 = **1.65 ms/fwd**。

在 batch 固定的 naive_batching 下，所有 rank 的 token 数永远相同，这个 all_reduce 是完全不必要的。

**方案设计**

在 dInfer `_forward_bsp_g` 中，缓存首次创建的 DPMetadata，后续 forward 直接复用：

```python
# modeling_llada2_moe.py _forward_bsp_g 中:
_CACHED_DP_METADATA = {}

def _forward_bsp_g(self, ...):
    ...
    n_tokens = bsz * seq_len
    cache_key = (n_tokens, dp_size)
    if cache_key not in _CACHED_DP_METADATA:
        # 第一次：正常创建（含 AllReduce）
        with set_forward_context(attn_metadata=None, vllm_config=cfg, num_tokens=n_tokens):
            _CACHED_DP_METADATA[cache_key] = get_forward_context()
            ...  # 正常执行
    else:
        # 后续：直接复用
        with override_forward_context(_CACHED_DP_METADATA[cache_key]):
            ...  # 正常执行
```

**预期收益**：1.7 ms/fwd (2.2%)

**难度**：极低（~10 行代码改动）

**风险**：极低。只要 batch 不变（naive_batching 保证），结果完全等价。

**与 EB 关系**：无直接关系。

**验证方法**：
- 性能：bench script GS path --no-quality，对比 ms/fwd
- 质量：bench script GS path with quality，肉眼检查 verifiable prompts
- Component：--component-timing 确认 DPMetadata 相关 gap 消失

**依赖关系**：无。可独立实施。

---

### OPT-2: 消除 logits.float() 转换

**问题描述**

model forward 末尾执行 `logits = logits.float()`，将 `[512, 32, 157184]` 从 bf16 转为 float32。

- 数据量：512 × 32 × 157184 × 4 bytes = 9.6 GB
- nsys 数据：Copy/Cast 类 kernel = 4.58 ms/fwd
- component timing：`global.logits_float` = 4.1 ms/fwd

这个转换是为了 softmax 精度，但 dLLM 的 ThresholdParallelDecoder 使用 temperature=0 + threshold=0.9，实际只需要比较大小关系（argmax + threshold），对精度要求极低。

**方案设计**

方式 A（最简单）：在 dInfer 源码中删除 `.float()` 调用：
```python
# modeling_llada2_moe.py line 4593-4594
hidden_states = outputs[0]
logits = self.lm_head(hidden_states)
# logits = logits.float()  ← 删除此行
```

方式 B（保守）：通过环境变量控制：
```python
logits = self.lm_head(hidden_states)
if not os.environ.get("DINF_SKIP_LOGITS_FLOAT"):
    logits = logits.float()
```

**预期收益**：4.1 ms/fwd (5.3%)

**难度**：极低（删除一行代码）

**风险**：低。需要验证：
1. ThresholdParallelDecoder 的 `torch.sigmoid(logits)` 在 bf16 下精度是否够
2. argmax/threshold 比较在 bf16 下是否与 float32 一致
3. 验证方式：跑质量对比

**与 EB 关系**：无直接关系。

**验证方法**：
- 性能：bench script GS path --no-quality，对比 ms/fwd
- 质量：bench script GS path with quality，对比 heteval512 输出
- 精度验证：同时跑 bf16 和 float32 路径，对比 decoded tokens 是否一致

**依赖关系**：无。可独立实施。

---

### OPT-3: EB-Aware Selective LM Head

**问题描述**

LM head (`[16384, 2048] × [2048, 157184]`) 对所有 16384 tokens 做 vocab projection，占 7.4 ms/fwd。但在 dLLM block diffusion 中：

- 每次 forward 处理 batch=512 × block_length=32 = 16384 tokens
- 其中大部分是已 decoded 的 token（平均只有 12.9% 是 MASK token）
- Decoder 只需要 MASK token 的 logits 来做 threshold 决策
- 已 decoded token 的 logits 对生成结果无贡献

如果只对 MASK token 计算 lm_head：
- MASK tokens: 16384 × 12.9% ≈ 2113 tokens
- LM head GEMM: [2113, 2048] × [2048, 157184] → 缩小 7.7x
- logits tensor: 2113 × 157184 × 2 bytes ≈ 0.63 GB（vs 当前 4.8 GB）

**方案设计**

```python
# 在 model forward 的 lm_head 之前：
def forward(self, input_ids, ...):
    ...
    hidden_states = outputs[0]  # [batch, seq, hidden]
    
    # Selective LM head: only compute for MASK tokens
    if hasattr(self, '_selective_lm_head') and self._selective_lm_head:
        mask_positions = (input_ids == MASK_ID)  # [batch, seq]
        num_mask = mask_positions.sum().item()
        
        if num_mask > 0 and num_mask < input_ids.numel():
            # Selective path
            hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
            mask_flat = mask_positions.view(-1)
            mask_indices = mask_flat.nonzero(as_tuple=True)[0]
            
            hidden_mask = hidden_flat[mask_indices]  # [num_mask, hidden]
            logits_mask = self.lm_head(hidden_mask)  # [num_mask, vocab]
            
            # Build full logits with zeros for non-MASK positions
            logits_full = torch.zeros(
                hidden_flat.shape[0], logits_mask.shape[-1],
                dtype=logits_mask.dtype, device=logits_mask.device)
            logits_full[mask_indices] = logits_mask
            logits = logits_full.view(batch, seq, -1)
        else:
            # Fallback: all MASK or no MASK
            logits = self.lm_head(hidden_states)
    else:
        logits = self.lm_head(hidden_states)
    ...
```

**注意**：ThresholdParallelDecoder 的 decode 逻辑需要确认：
- decoded token 位置的 logits = 0 是否影响其行为
- 通常 decoder 只看 MASK 位置的 argmax 和 threshold，decoded 位置不参与决策

**预期收益**：
- LM head: 7.4 × (1 - 0.129) ≈ 6.4 ms 节省
- logits.float()（如保留）: 4.1 × (1 - 0.129) ≈ 3.6 ms 节省
- 如与 OPT-2 结合：~6.4 ms
- 如独立（保留 .float()）：~10 ms
- **保守估计：~8 ms (10.5%)**

**难度**：中等
- 需要修改 model forward
- 需要确认 decoder 兼容性
- `num_mask` 随迭代变化：第一次迭代全 MASK → 逐渐减少

**风险**：低-中。
- hidden_states 质量不受影响（attention/MoE 完整执行）
- 只影响 logits 输出
- decoded token 位置的 logits=0 可能影响某些 decoder 逻辑（需验证）

**与 EB 关系**：**直接利用 EB 的信息**。
- EB controller 已知每次 forward 的 MASK 位置分布
- hot_skip 路径中 MASK 位置稳定，可以缓存 mask_indices
- EB 的 s_mask 计算路径不受影响（它用的是 gate logits，不是 lm_head logits）

**验证方法**：
- 性能：bench script GS path --no-quality
- 质量：bench script with quality，检查 heteval512
- 正确性：对比 selective vs full lm_head 的 decoded tokens 是否一致
- 边界：验证全 MASK（第一次迭代）和全 decoded（理论上不会发生）的 fallback

**依赖关系**：无。与 OPT-2 可叠加。

---

### OPT-4: fp8 通信压缩

**问题描述**

EP dispatch (AllGather) 和 combine (ReduceScatter) 发送 bf16 数据：
- dispatch payload: 206.7 MB/fwd（hidden_states + router_logits）
- combine: MoE output reduce-scatter

nsys 数据：NCCL AllGather = 8.0 ms/fwd，NCCL Reduce bf16 = 6.5 ms/fwd。

之前 overhead isolation 实验 G7 直接验证：fp8 通信从 8.4ms 降到 3.3ms（-60%，省 5.1 ms）。

**方案设计**

Monkey-patch `AgRsAll2AllManager` 的 dispatch/combine：

```python
# dispatch:
def patched_dispatch(self, hidden_states, router_logits, is_sp):
    # Cast to fp8 before AllGather
    hs_fp8 = hidden_states.to(torch.float8_e4m3fn)
    lg_fp8 = router_logits.to(torch.float8_e4m3fn)
    
    # AllGather fp8 (half payload)
    hs_gathered_fp8, lg_gathered_fp8 = orig_dispatch(hs_fp8, lg_fp8, is_sp)
    
    # Cast back to bf16
    return hs_gathered_fp8.to(torch.bfloat16), lg_gathered_fp8.to(torch.bfloat16)

# combine:
def patched_combine(self, hidden_states, is_sp):
    hs_fp8 = hidden_states.to(torch.float8_e4m3fn)
    result_fp8 = orig_combine(hs_fp8, is_sp)
    return result_fp8.to(torch.bfloat16)
```

**预期收益**：~5 ms/fwd (6.5%)

**难度**：中等。
- 需要处理 fp8 的 scale factor（e4m3 动态范围有限）
- router_logits 的量化需要特别注意（影响 expert 选择）

**风险**：中等。
- fp8 e4m3 动态范围 ±448，hidden_states 一般在 [-10, 10] 范围内适用
- router_logits 用于 topK 选择，微小量化误差可能改变排序
- 需要质量验证

**与 EB 关系**：间接相关。
- EB 的 hot_skip 路径中 routing 稳定，fp8 量化误差不太可能改变 top-K 选择
- 可以只在 hot_skip 时使用 fp8 通信，cold/hot_update 时保持 bf16
- 这种"EB-path-aware precision"策略可进一步降低风险

**验证方法**：
- 性能：bench script GS path --no-quality
- 质量：bench script with quality
- 精度验证：对比 fp8 vs bf16 通信下的 per-layer routing 选择是否一致
- 安全模式：先只对 hidden_states 做 fp8（保留 router_logits bf16）

**依赖关系**：无。可独立实施。

---

### OPT-5: Shared Expert ∥ EP Dispatch Overlap

**问题描述**

当前 BSP-G 的 MoE block forward 中，shared_expert 和 EP dispatch 是串行执行的：

```
shared_expert(hs_sp)  →  gate(hs_sp)  →  EP dispatch  →  MoE kernel  →  EP combine
    3.4 ms                  2.7 ms         8.7 ms           20.6 ms        11.6 ms
```

但 shared_expert 和 dispatch 只依赖相同的输入 `hs_sp`（post_attention_layernorm 输出），二者互不依赖。Dispatch 是 NCCL AllGather，GPU 在此期间主要等待数据传输——正好可以利用空闲 SM 做 shared_expert 计算。

nsys 确认：所有操作在单一 stream 上，没有任何 overlap。

**方案设计**

在 `LLaDA2MoeSparseMoeBlock.forward_sp` 中引入双 stream：

```python
def forward_sp(self, hidden_states_sp):
    stream_shared = torch.cuda.Stream()
    
    # Stream A (main): gate + dispatch + MoE kernel + combine
    router_logits = self.gate.get_logits(hidden_states_sp)
    
    # Stream B (parallel): shared expert
    with torch.cuda.stream(stream_shared):
        shared_res = self.shared_experts(hidden_states_sp)
    
    # Main stream continues with dispatch
    y_sp = self.experts.forward_impl(hidden_states_sp, router_logits)
    
    # Sync shared expert before residual add
    torch.cuda.current_stream().wait_stream(stream_shared)
    
    if shared_res is not None:
        y_sp = y_sp + shared_res
    return y_sp
```

**预期收益**：~3.4 ms/fwd (4.4%)
- shared_expert (3.4ms) 与 dispatch (8.7ms) 重叠
- 由于 shared < dispatch，完全隐藏

**难度**：中等。
- 需要管理 CUDA stream 生命周期
- 需要确保 stream 同步正确
- 多 stream 可能影响 NCCL 行为

**风险**：低。
- 两个操作完全独立（只读相同输入）
- 无数据竞争
- 但需验证 NCCL 在多 stream 环境下行为正常

**与 EB 关系**：间接。
- gate_logits 计算（2.7ms）也可以和 shared_expert 并行
- EB 的 get_s_mask 依赖 gate_logits，所以 gate 必须在 dispatch 前完成
- 优化后的执行顺序：`[gate ∥ shared] → dispatch → MoE → combine`

**验证方法**：
- 性能：bench script GS path --no-quality
- 质量：bench script with quality
- Component timing：确认 shared 时间被 dispatch 隐藏

**依赖关系**：无。可独立实施。

---

### OPT-6: EB Routing Kernel 融合

**问题描述**

EB routing 路径包含 ~15 个小 kernel（每个 2-20us），之间有大量 launch gap（30-130us）：

```
elementwise_neg  2us  gap=8us
elementwise_mul  5us  gap=32us
CatBatchedCopy  10us  gap=17us
elementwise_add  8us  gap=23us
rms_norm        17us  gap=36us
...
```

19 layers × ~15 kernels × ~50us avg gap ≈ **14 ms/fwd** 的 launch gap 在 EB 相关操作中。

实际 EB 开销（nsys）：
- EB/routing triton: 3.82 ms（kernel 执行时间）
- EB kernels (_kernel_A/C): 2.71 ms
- 加上 launch gap: 估计额外 ~3-4 ms

**方案设计**

将当前分散的 EB 操作（`_kernel_A` + 多个 elementwise + `_fused_routing_k`）融合为 1-2 个 Triton kernel：

```python
@triton.jit
def fused_eb_routing_kernel(
    gate_logits_ptr,  # [N, E_global] 
    bias_ptr,         # [E_global]
    s_mask_ptr,       # [E_global] bool
    rsf,              # scalar
    output_weights_ptr,  # [N, K]
    output_indices_ptr,  # [N, K]
    ...
):
    # 融合: bias add + s_mask apply + grouped topk + weight normalize
    # 替代: _kernel_A + neg + mul + add + cat + _fused_routing_k
    ...
```

**预期收益**：~2-3 ms/fwd (3-4%)
- 消除 ~10 个中间 kernel launch
- 减少中间 tensor 分配

**难度**：高。
- 需要完整重写 EB routing 的 Triton 实现
- grouped topk 在 Triton 中实现复杂
- 需要正确处理 s_mask 的 hot_skip/cold/hot_update 逻辑

**风险**：中等。
- Triton kernel 正确性需要大量测试
- 性能不一定优于当前分散实现（如果 single kernel 占用太多 registers）

**与 EB 关系**：直接优化 EB 本身的开销。

**验证方法**：
- 正确性：单独测试新 kernel vs 原始 EB routing 的输出一致性
- 性能：bench script GS path --no-quality
- 质量：bench script with quality

**依赖关系**：无。但实施前建议先完成 OPT-1~5（ROI 更高）。

---

## 4. 实施优先级和分期计划

### Phase 1: 快速收割（预期总收益 ~5.8 ms, 7.6%）

| 序号 | 优化项 | 预期收益 | 实施时间 |
|------|--------|----------|----------|
| OPT-1 | DPMetadata AllReduce 消除 | 1.7 ms | 0.5h |
| OPT-2 | logits.float() 消除 | 4.1 ms | 0.5h |

特点：代码改动极小，零质量风险，可快速验证。

### Phase 2: 核心优化（预期总收益 ~13-16 ms, 17-21%）

| 序号 | 优化项 | 预期收益 | 实施时间 |
|------|--------|----------|----------|
| OPT-3 | EB-Aware Selective LM Head | ~8 ms | 2-3h |
| OPT-4 | fp8 通信压缩 | ~5 ms | 2-3h |

特点：收益大，需要质量验证，与 EB 机制有结合点。

### Phase 3: 架构优化（预期总收益 ~5-7 ms, 7-9%）

| 序号 | 优化项 | 预期收益 | 实施时间 |
|------|--------|----------|----------|
| OPT-5 | Shared Expert ∥ Dispatch | ~3.4 ms | 2h |
| OPT-6 | EB Kernel 融合 | ~2-3 ms | 4-6h |

特点：需要更深入的系统级改动。

### 累积收益预估（2026-05-05 实测更新）

| 完成到 | 实测 G ms/fwd | 实测增量 | 加速比 |
|--------|--------------|----------|--------|
| Baseline (no opts) | 70.2 | — | 1.00x |
| + OPT-2 (logits.float skip) | 65.9 | -4.3 ms | 1.07x |
| + OPT-2 + SP-LM Head (clean) | **58.2** | **-12.0 ms** | **1.21x** |

已排除的优化项（实测无效）：
- OPT-4 (fp8 dispatch): -1.5 ms 退化（cast 开销 > NVLink 通信节省）
- OPT-5 (shared expert ∥ dispatch): ~0 ms（GPU pipeline 已隐藏）
- CUDA Graph: ~2 ms at batch=512（GPU pipeline 隐藏 Python 开销）

注意：累积收益可能小于各项之和（组件间有串行依赖，某些优化可能互相影响）。

## 5. 验证框架

### 5.1 统一验证流程

每个优化项的验证严格遵循 4 步：

**Step A: 实现**（monkey-patch 或源码修改）

**Step B: 性能 A/B**
```bash
# OFF (baseline)
torchrun --nproc_per_node=8 bench_bsp_moe_dp2.py \
  --config-set bspg_source --profile-target bsp \
  --batch-size 512 --gen-length 256 --num-runs 2 --no-quality

# ON (optimized)
torchrun --nproc_per_node=8 bench_bsp_moe_dp2.py \
  --config-set bspg_source --profile-target bsp \
  --batch-size 512 --gen-length 256 --num-runs 2 --no-quality \
  [--opt-flag]
```

判定标准：
- GS ms/fwd 下降 ≥ 预期收益的 50%
- path counts 完全一致 (19/171/3933/931)
- 无 crash / hang

**Step C: 质量验证**
```bash
torchrun --nproc_per_node=8 bench_bsp_moe_dp2.py \
  --config-set bspg_source --profile-target bsp \
  --batch-size 512 --gen-length 256 --num-runs 1 \
  [--opt-flag]
```

判定标准：
- heteval512 verifiable prompts 输出正确
- 肉眼无明显质量退化

**Step D: Component timing 验证**
```bash
torchrun --nproc_per_node=8 bench_bsp_moe_dp2.py \
  --config-set bspg_source --profile-target bsp \
  --batch-size 512 --gen-length 256 --num-runs 1 --no-quality \
  --component-timing [--opt-flag]
```

判定标准：
- 目标组件时间下降
- 其他组件无退化

### 5.2 回归保护

每当累积多个优化后，跑一次完整的 C12 对照：
- A path (baseline) vs GS path (all opts on)
- 确认总收益 ≈ 各项收益之和（无负面交互）
- path counts 不变
- 质量无退化

### 5.3 关键文件索引

| 文件 | 用途 |
|------|------|
| `codex_coding/src/bench_bsp_moe_dp2.py` | 主 benchmark 脚本 |
| `lib_cite/dInfer/python/dinfer/model/modeling_llada2_moe.py` | dInfer 模型源码 |
| `lib_cite/dInfer/python/dinfer/decoding/utils.py` | KV cache 管理 |
| `codex_coding/src/test_heteval512.py` | 质量验证 prompts |
| `codex_coding/src/baseline_optimizations.py` | 基础优化（flash_attn 等） |
| `codex_coding/src/test_fused_eb_triton.py` | EB fused_routing kernel |
| `codex_coding/src/test_m_skip_sweep.py` | EB controller |

### 5.4 Profiling 数据位置

| 数据 | 路径 |
|------|------|
| nsys trace | `/home/wuhang/tmp_nsys/profile_gs_full.nsys-rep` |
| nsys sqlite | `/home/wuhang/tmp_nsys/profile_gs_full.sqlite` |
| Component timing log | `/tmp/gs_full_v2.log` |
| CUDA Graph POC results | `codex_coding/results/poc_cudagraph_*.json` |
