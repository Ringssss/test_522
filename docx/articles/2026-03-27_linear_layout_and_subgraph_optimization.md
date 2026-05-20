# LinearLayout 的优化思路，以及它对“子图级模型优化”的启发

## 1. 这份归档回答什么问题

这份文档整理两个问题：

1. Triton `LinearLayout` 的核心原理和优化思路是什么。
2. 这套思路能否提升为“子图级别的模型优化”方法论。

本文档是技术归档，不是最终方案设计稿。

## 2. `LinearLayout` 本质上是什么

在 Triton 中，`LinearLayout` 不是一个 layout 名字集合，而是一个统一的代数对象。

它把 layout 建模为：

- 从硬件位置到逻辑张量坐标的映射
- 典型输入维度包括 `register`、`lane`、`warp`、`block`
- 输出维度是逻辑张量维度，如 `dim0`、`dim1`

关键点不是“记录整张映射表”，而是：

- 只记录若干 basis vectors
- 其余映射值由 GF(2) 风格的线性组合得到
- 很多 GPU 常见 layout 变换都可以统一落进这套表示
  - swizzle
  - transpose
  - broadcast
  - fragment packing

因此，`LinearLayout` 的核心价值是把 layout 从“特例化标签”提升成“可计算、可组合、可比较、可求逆/伪逆”的对象。

## 3. `LinearLayout` 的优化思路

### 3.1 先统一表示，再谈优化

Triton 先把不同 encoding 统一转换为 `LinearLayout`，然后在统一表示上做推理，而不是为每一种 encoding 写一套独立的转换规则。

这一步的收益是：

- layout 等价性可以直接判断
- reshape/transpose/swizzle 可以统一处理
- layout conversion 可以形式化，而不是靠大量 case-by-case 逻辑

### 3.2 把 layout conversion 变成代数问题

如果要从 `src layout` 转到 `dst layout`，核心是构造：

- `dst^{-1} ∘ src`

在 Triton 里对应 `invertAndCompose`。

如果 `dst` 可逆，这就是严格意义上的逆组合。
如果 `dst` 不可逆但可覆盖需要的输出空间，则用 pseudoinverse / least-squares 风格的方法构造一个可行转换。

这一步的重要思想是：

- conversion 不是黑盒搬运
- conversion 是一个可分析的映射
- 编译器可以知道“到底哪些维度真的变了”

### 3.3 只保留“真正非平凡”的那部分转换

Triton 不满足于“有一个转换”。
它还会继续约简：

- 如果某些维度本来就是 identity，就 quotient 掉
- 如果转换只影响 `register`，那就不该升级成 shared-memory 级别动作
- 如果 `warp` 或 `block` 维度是 trivial 的，就不需要更重的同步和搬运路径

`minimalCvtLayout` 的本质就是：

- 从完整转换中提取最小非平凡子变换

这是 `LinearLayout` 真正具有“优化性”的关键。

### 3.4 根据“变化发生在哪个子空间”来选择实现手段

Triton 后续 lowering 的决策逻辑不是拍脑袋，而是看最小转换到底动了哪些维度：

- 只动 `register`：线程内重排
- 动到 `lane`：优先 warp shuffle
- 动到 `warp` 或 `block`：shared memory / distributed shared memory
- 完全不动：直接删掉 conversion

因此，`LinearLayout` 不只是在“表示 layout”，它还在驱动“该用什么代价去实现这次转换”。

### 3.5 从 layout 本身推导连续性、广播与可向量化性

`LinearLayout` 还能从映射结构里推导：

- innermost contiguous region
- 最长连续 copy 宽度
- 哪些输入位是 free variables
- 是否存在 broadcast / replication

这些推导会影响：

- async copy
- vectorized load/store
- warp-level transfer
- shared-memory staging

也就是说，它不是“静态描述对象”，而是“优化决策的分析基础”。

## 4. 一句话总结 `LinearLayout` 的优化哲学

`LinearLayout` 的优化哲学可以概括为：

> 先把 layout canonicalize 成统一的代数对象，再把 compatibility、conversion、broadcast、contiguity 都变成可推理问题，最后只为真正非平凡的数据重排付成本。

这和“为每种 layout 单独写一个转换 kernel”是完全不同的思路。

## 5. 这套思想为什么对“子图级优化”有启发

如果把视角从单个 kernel 提升到热点子图，这套思想仍然有很强的指导性。

最应该继承的不是 Triton 的具体维度名，而是它的方法：

- 把 layout 作为一等优化变量
- 把 conversion 显式化
- 判断哪些部分是 trivial，哪些部分必须付代价
- 只优化真正影响全局成本的非平凡部分

## 6. 如何把 `LinearLayout` 思想提升到子图级

### 6.1 从“kernel 内 layout”变成“子图边上的 layout state”

在子图级 IR 中，可以让每条边 `e` 带一个 layout 状态 `L_e`，而不是只带：

- shape
- dtype

这样，layout 就不再是局部实现细节，而是图级状态。

### 6.2 每个算子提供 layout legality，而不是只提供功能语义

每个 operator 不只定义“算什么”，还定义：

- 哪些输入 layout 合法
- 哪些输出 layout 可以产生
- 哪些 layout 组合允许 fusion
- 哪些转换可以被算子吸收

这相当于把 Triton 中的 layout compatibility，提升成子图上的 legality relation。

### 6.3 子图优化目标从“单算子最优”改成“总成本最优”

如果沿用 `LinearLayout` 的思路，子图优化的目标应是：

- operator cost
- layout conversion cost
- memory traffic cost
- fusion gain
- kernel-family suitability

的联合最优，而不是对每个 op 单独选最快 kernel。

这正是“per-op optimality”与“subgraph optimality”的区别。

### 6.4 子图里也应该保留“最小非平凡转换”的思想

在 kernel lowering 里，Triton 会找 `minimalCvtLayout`。

类比到子图里，也不应该把所有 layout 差异一视同仁。
更合理的做法是：

- 识别 producer/consumer 之间真正不兼容的那一部分 layout 差异
- 对 trivial 部分直接传递
- 只对非平凡差异引入 conversion、specialized kernel 或 fusion 约束

这会直接减少：

- 冗余 layout transform
- 不必要的中间写回
- 局部最优但全局更差的 kernel 选择

### 6.5 “实现机制选择”可以提升为“子图执行计划选择”

在 Triton 单点 lowering 里，最小转换决定：

- register reorder
- warp shuffle
- shared memory

在子图级里，类似地可以决定：

- 直接复用 producer 输出 layout
- 换一个 consumer kernel family
- 插入显式 conversion
- 把 conversion 吸收到相邻 kernel
- 启用或禁用某类 fusion

也就是说，`LinearLayout` 的“根据非平凡变化的作用域选择机制”这个思想，完全可以升格为“根据非平凡子图差异选择执行计划”。

## 7. 能直接复用什么，不能直接复用什么

### 7.1 可以直接复用的思想

- 统一 layout 表示
- composition / invert / pseudoinvert 思维
- compatibility 判断
- trivial-vs-nontrivial 约简
- minimal conversion 思维
- 让 layout 分析直接驱动优化决策

### 7.2 不能机械照搬的部分

子图优化不能直接照抄 Triton `LinearLayout`，原因是：

- 子图跨算子后，index space 可能变化
- 有些 layout 不是纯 GF(2) 线性结构
- 量化 pack、tile blocking、epilogue 约束往往带有离散 legality
- 子图问题还要联合 kernel family、fusion、memory hierarchy 和 cost model

因此，更合适的方向是：

- 构建一个 **LinearLayout-inspired Subgraph Layout IR**
- 对可线性化部分保留代数表示
- 对不可线性化部分叠加离散约束与规则系统

## 8. 当前可操作的研究结论

基于当前阅读，可以先得到以下结论：

1. `LinearLayout` 的真正贡献不是“新的 layout 名词”，而是“统一的可计算 layout 代数”。
2. 它最有价值的优化思想是“只保留最小非平凡转换，并让这个结果驱动实现路径选择”。
3. 这套思想非常适合提升到子图级，但需要从“硬件维度变换”扩展为“边级 layout state + operator legality + total subgraph cost”。
4. 如果项目要做子图级 layout 优化，最合理的继承方式不是复刻 Triton 内部实现，而是抽取出它的方法论，构建自己的 Subgraph Layout IR 与 planner。

## 9. 后续讨论建议

后续如果继续讨论，建议按下面顺序推进：

1. 先定义子图级 layout state 的最小表示
2. 再定义 operator legality relation
3. 再定义 conversion / absorption / fusion 的规则
4. 最后定义 cost model 和 search procedure

先不要过早陷入：

- Triton core compiler fork
- 过宽的全图优化
- 过重的 ILP/SMT 求解
- 过多热点类别同时推进

## 10. 关键来源

本轮直接参考了以下源码与文档：

- `/home/wuhang/wuhang/linear_wh/triton/include/triton/Tools/LinearLayout.h`
- `/home/wuhang/wuhang/linear_wh/triton/lib/Tools/LinearLayout.cpp`
- `/home/wuhang/wuhang/linear_wh/triton/lib/Analysis/Utility.cpp`
- `/home/wuhang/wuhang/linear_wh/triton/lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp`
- `/home/wuhang/wuhang/linear_wh/triton/lib/Dialect/TritonGPU/IR/Dialect.cpp`
- `/home/wuhang/wuhang/linear_wh/triton/lib/Conversion/TritonGPUToLLVM/MemoryOpToLLVM.cpp`
- `/home/wuhang/wuhang/linear_wh/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp`
- `/home/wuhang/wuhang/linear_wh/triton/lib/Dialect/TritonGPU/Transforms/CoalesceAsyncCopy.cpp`
- `https://research.ibm.com/publications/triton-an-intermediate-language-and-compiler-for-tiled-neural-network-computations`
- `https://github.com/triton-lang/triton/pull/5309`
