# docx 文档地图与存放查找指南

## 1. 目的

这份文件用于告诉后续 AI：

- `docx/` 各个子目录分别承担什么职责；
- 新文档应该放到哪里；
- 遇到一个问题时，应该先去哪里找；
- 哪些内容不应该放进 `docx/`。

这是一份稳定入口文档，优先级高于零散归档。

## 2. 总体分层

`docx/` 当前应理解为一个分层文档库，而不是单一笔记目录。

| 位置 | 主要作用 | 应存放的内容 | 不应存放的内容 |
| --- | --- | --- | --- |
| `docx/` 根目录 | 当前工作入口、恢复材料、稳定指南 | `next_step.txt`、恢复说明、全局指南 | 大量专题分析、测试脚本、运行结果 |
| `docx/cites/` | 项目立项依据、论文 framing、理论基础 | 架构定位、理论抽象、POC 范围界定 | 运行日志、阶段进展、临时 brainstorm |
| `docx/articles/` | 专题技术分析与阶段性技术归档 | 针对某个主题的深入分析、brainstorm、技术归档 | 过程流水账、仅面向恢复的索引 |
| `docx/context_index/` | 恢复导航与文档索引 | 文档注册表、阶段总结、活跃线程、最小必读集 | 长篇技术正文、实验脚本 |
| `docx/plans/` | 正式计划稿和可执行设计稿预留区 | 未来的阶段计划、设计计划、执行计划 | 技术随想、原始测试输出 |

## 3. 各目录的具体用法

### 3.1 `docx/` 根目录

这里放的是“先读文件”和“跨会话恢复文件”。

当前已知代表文件：

- `next_step.txt`：当前阶段代号、开发规范、归档约束、目录要求。
- `2026-03-27_context_compression_recovery_note.md`：上下文压缩或重启后的恢复说明。
- `00_docx_storage_and_lookup_guide.md`：本文件，用于指导文档存放和查找。

适合放在根目录的文件类型：

- 当前阶段总入口；
- 全局约束说明；
- 会话恢复说明；
- 整个 `docx/` 的稳定导航文档。

不适合放在根目录的内容：

- 面向单一技术主题的长篇分析；
- 原始 benchmark 输出；
- 过程归档和测试记录。

### 3.2 `docx/cites/`

这里放“为什么做这件事”的上层依据。

当前文件表明，这一层主要用于：

- 论文问题定义；
- 项目范围边界；
- 理论抽象；
- POC 的构建原则。

适合放入 `cites/` 的文件：

- 新的 paper architecture note；
- 新的理论框架说明；
- 新的 POC scope / build plan；
- 来自论文、系统设计、方法论层面的总结。

如果一个文件主要回答的是下面这些问题，优先放在 `cites/`：

- 我们的项目到底想解决什么问题？
- 论文应该如何定位？
- 为什么当前范围应该收敛到这个 POC？
- 理论层面的核心抽象是什么？

### 3.3 `docx/articles/`

这里放“已经展开过的专题技术分析”。

当前文件表明，这一层主要用于：

- 针对某个方向的技术归档；
- 代码和运行现象结合后的技术分析；
- 阶段性 brainstorm；
- 未来可以继续深化为设计稿或实验计划的技术笔记。

适合放入 `articles/` 的文件：

- 某个模型结构的分析；
- 某个缓存机制的专题讨论；
- 某个编译器 / runtime 技术的启发式总结；
- 某个研究方向的阶段性技术归档。

建议命名方式：

- `YYYY-MM-DD_topic_name.md`

如果一个文件主要回答的是下面这些问题，优先放在 `articles/`：

- 这个技术方向目前我们已经想清楚了什么？
- 代码和运行现象说明了什么？
- 某个优化点值得如何继续推进？

### 3.4 `docx/context_index/`

这里放“恢复导航层”，不是技术正文层。

当前四个文件已经形成了固定分工：

- `00_document_registry.md`：重要文档总表，说明每个关键文件的角色、主题、优先级和关键结论。
- `01_stage_summary.md`：当前阶段总结，包含已完成工作、已验证事实、未验证假设和建议下一步。
- `02_active_threads.md`：当前活跃线程，说明每条线的状态、缺口和下一步建议。
- `03_current_required_process_docs.md`：最小必读过程文档集合，用于下一次恢复时快速聚焦。

适合放入 `context_index/` 的文件：

- 新的索引；
- 新的阶段快照；
- 新的恢复阅读顺序；
- 面向“快速接手现场”的轻量导航文档。

不适合放入 `context_index/` 的文件：

- 大段技术原理推导；
- 运行实验正文；
- 详细过程日志。

### 3.5 `docx/plans/`

该目录当前为空，应视为“正式计划稿预留区”。

后续如果开始写更正式的计划文档，建议优先放这里，例如：

- 里程碑计划；
- 研究执行计划；
- 设计落地计划；
- 实验计划与评估计划。

只有当一份文档已经从 brainstorm 进入“准备执行”的状态时，才建议从 `articles/` 升级到 `plans/`。

## 4. 遇到问题时该怎么查

### 4.1 想知道当前开发规范

先读：

- `docx/next_step.txt`

### 4.2 想知道当前有哪些重要文档

先读：

- `docx/context_index/00_document_registry.md`

### 4.3 想知道项目现在进行到哪里

先读：

- `docx/context_index/01_stage_summary.md`

### 4.4 想知道当前最热的研究线程是什么

先读：

- `docx/context_index/02_active_threads.md`

### 4.5 想知道最少需要补哪些过程文档

先读：

- `docx/context_index/03_current_required_process_docs.md`

### 4.6 想找某个技术主题的深入分析

按主题去：

- `docx/articles/`

### 4.7 想找论文 framing、理论依据、POC 范围定义

按主题去：

- `docx/cites/`

### 4.8 想在上下文压缩或模型重启后快速恢复

先读：

- `docx/2026-03-27_context_compression_recovery_note.md`
- 然后回到 `context_index/`

## 5. 新文档应如何落位

可以用下面的判断规则：

| 新文件类型 | 建议位置 |
| --- | --- |
| 当前阶段入口、全局说明、恢复说明 | `docx/` 根目录 |
| 论文定位、理论框架、POC 边界 | `docx/cites/` |
| 技术专题分析、brainstorm、机制归档 | `docx/articles/` |
| 恢复导航、阶段摘要、文档注册表 | `docx/context_index/` |
| 正式计划稿、执行计划、实验计划 | `docx/plans/` |

## 6. 哪些内容不要放进 `docx/`

下面这些内容应该放到别处：

| 内容类型 | 正确位置 |
| --- | --- |
| 过程文档、建设记录、归档流水 | `code_building/process_docs/` |
| 进度增量总结 | `code_building/progress_diff_summary.md` |
| 关键文件索引 | `code_building/key_files_index.md` |
| 关键结论 | `code_building/key_conclusion.md` |
| 测试脚本 | `codex_coding/src/` |
| 测试结果、benchmark 输出、运行产物 | `codex_coding/results/` |

`docx/` 存“重要文档”和“面向理解的文档”，不存执行产物。

## 7. 命名建议

建议统一采用以下风格：

- 稳定入口文件：`00_*.md`
- 技术归档文件：`YYYY-MM-DD_topic_name.md`
- 恢复索引文件：`00_*`、`01_*`、`02_*`、`03_*`
- 计划文件：`YYYY-MM-DD_plan_topic.md` 或阶段代号风格文件名

## 8. 当前项目的一个重要注意点

当前 `docx/` 中不少历史文件仍然引用旧路径：

- `/home/wuhang/wuhang/linear_wh/...`

而当前实际项目根目录是：

- `/home/wuhang/wuhang/dllm_wh/...`

因此：

- 这些旧路径可以作为历史上下文参考；
- 但在真正执行命令、打开代码、修改文件之前，必须先核实当前实际路径；
- 不能把历史路径直接当成当前可执行路径使用。

## 9. 后续维护规则

每次新增一个重要文档时，至少同步下面几处：

- 如果它属于 `docx/` 中的重要长期文档，更新 `docx/context_index/00_document_registry.md`
- 更新 `code_building/key_files_index.md`
- 更新 `code_building/progress_diff_summary.md`

每次形成一个新的关键结论时，还需要同步：

- `code_building/key_conclusion.md`

每次形成一次有意义的建设动作时，还需要补：

- `code_building/process_docs/` 下的过程文档

并在过程文档中保留：

- `本轮命令`

## 10. 推荐阅读顺序

新 AI 进入现场时，推荐按以下顺序阅读：

1. `docx/next_step.txt`
2. `docx/00_docx_storage_and_lookup_guide.md`
3. `docx/context_index/00_document_registry.md`
4. `docx/context_index/01_stage_summary.md`
5. `docx/context_index/02_active_threads.md`
6. `docx/context_index/03_current_required_process_docs.md`
7. 按需进入 `docx/cites/` 或 `docx/articles/`

这样可以先建立规则感，再建立现场感，最后进入专题内容。
