# AGENTS.md — 项目协作与代码生成规范

本文件是项目根目录唯一的全局 Agent 规则入口。规则适用于整个仓库；更具体的目录规则可以补充本文件，但不得削弱本文件要求。

## 规则加载与代码范围

- 任何目标路径匹配 `code/**` 的读取、生成、修改、删除、移动、重命名、测试或脚本执行，均视为 code 范围操作。
- 开始 code 范围操作前，必须先读取 `code/AGENTS.md`，再读取 `code/CODE_CONVENTIONS.md`。
- 若任一规则文件不存在、无法读取或内容不完整，禁止继续写入 `code/**`，必须先报告原因。
- `code/CODE_CONVENTIONS.md` 是代码文件分类规范的唯一正文来源；`code/AGENTS.md` 只负责范围说明与执行要求。
- 所有方法均置于 `code/method-<name>/` 下，本工作区的方法目录为 `code/method-a2/`。不得在仓库根目录或 `code/` 根目录直接放置方法的代码、数据和输出。

## 文档规范

- 所有项目文档统一放在 `docs/`。
- `docs/project/plans/` 存放实验执行前形成的计划与设计类文档，包括实验流程与决策规程、实验与实现设计、符号与公式定义、数据契约与生成流程以及指标规范。
- `docs/project/reports/` 存放实验执行后形成的结果与结论文档，包括正式结果摘要、实验执行结论以及审计与复核分析报告。
- 新增文档按“实验执行前归 `plans/`、实验执行后归 `reports/`”判断归档位置；`docs/project/` 根目录保留研究目标与问题动机、技术路线与方案路线、项目入口、决策流程与状态记录等其他项目文档。
- 文档在 `docs/project/`、`docs/project/plans/` 与 `docs/project/reports/` 之间迁移时，必须同步更新 `docs/INDEX.md` 中的路径以及文档正文与其他文档中的交叉引用。
- `docs/superpowers/plans/` 继续存放 Superpowers 实施计划，不存放实验计划；单次运行的原始产物仍按 `code/method-*/output/` 归档，不写入 `docs/project/reports/`。
- `docs/INDEX.md` 是统一文档索引。涉及项目技术细节、任务记录或规范时，先读取索引，再按索引指示读取目标文档。
- 新增或修改 Markdown 文档时，必须同步维护文件头部摘要，并在文档结构或触发关键词变化时更新 `docs/INDEX.md`。
- 项目文档使用正式、严谨的学术书面表达，禁止使用口语化、营销化或网络化表述。
- 项目文档禁止使用表格记录规则或任务信息，改用列表或段落。
- `docs/TASKS.md` 用于登记已完成 Rebase、准备 Push 的任务；每项记录必须填写规定的七个字段。

## 代码生成规范

- `code/**/*.py` 中的模块级 docstring、函数/类 docstring、行内注释以及 TODO/FIXME/NOTE 必须使用中文。
- 修改已有代码时，涉及的英文注释或 docstring 应同步规范为中文。
- 最终交付结果必须来自对应方法的 `output/`，不得将 `checkpoint/` 作为最终交付目录。

## 通用工程行为

- 对非平凡编码先说明会影响实现的关键假设，并优先通过仓库上下文消解歧义。
- 仅在调查后仍存在重要歧义时向用户提问，不为显得自主而猜测。
- 选择能够完整满足当前需求的最小清晰实现，不添加臆测的功能、抽象、配置或强化。
- 设计新结构前先查找并复用仓库已有的辅助函数、抽象、测试和错误处理模式。
- 保持修改具有可追溯的任务因果关系，避免无关重构、格式统一和历史问题清理。
- 仅清理本次修改产生的无用代码，不处理与任务无关的既有 dead code。
- 非平凡任务先定义可验证的成功条件，完成后按影响范围执行足够但不过量的验证。

## Git 操作规范

在本项目中统一使用 `docs/TASKS.md` 路径：

### 分支

- 禁止直接 Push 到 `main` 或 `master`。
- 分支命名格式为 `agent/<task-id>-<描述>`。
- 每项新任务从最新 `main` 创建分支，合并由人工触发，Agent 不得自行 Merge。

### Commit Trailer

每条非 Checkpoint commit 的末尾必须附加以下三个 trailer：

```text
Agent-Task: <任务ID或需求>
Agent-Decision: <关键取舍及理由>
Agent-Limitation: <已知局限，无则填"无">
```

格式要求：键名仅限以上三个；分隔符为英文 `: `；每条独占一行；值内禁止换行；每条值不超过 80 字；trailer 块与正文之间空一行。

### Commit Pipeline

- 当修改超过 5 个文件或任务持续超过 15 分钟时，在接口/数据模型、核心逻辑、测试、文档四个节点分别建立 `[WIP]` Checkpoint；Checkpoint 的 body 仅写一行当前进度，可不带 trailer。
- 任务完成且尚未 Push 时，执行 `git rebase -i main`，将全部 `[WIP]` commit squash 至对应语义 commit；已共享分支禁止 Rebase。
- 最终每个 commit 必须可独立理解、可独立编译、可独立回滚；一个 commit 只包含一个逻辑变更，不得将重构与功能修改混入同一 commit。
- Rebase 完成、Push 前，在 `docs/TASKS.md` 登记本次任务的七个字段：任务 ID、需求摘要、分支、时间、关键决策、已知局限、状态。

### PR 与并行

- PR 描述中的 `Task Description`、`Key Design Decisions`、`Known Limitations` 直接引用 commit trailer 对应字段。
- 多 Agent 并行时，每个 Agent 必须使用独立 Git worktree。

## 规则优先级

- 本文件规定项目级约束。
- `code/AGENTS.md` 仅补充 code 范围执行要求。
- `code/CODE_CONVENTIONS.md` 规定 code 范围的文件组织与输出分类。
- 发生冲突时，项目级规则优先；任何无法判断的情况先暂停写入并请求确认。
