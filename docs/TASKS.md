<!--
本文档：任务追踪记录
触发关键词：任务、task、需求、版本、变更记录、改动记录
检索顺序：1
-->

# 任务追踪记录

每个任务记录以下七个字段：

- 任务 ID
- 需求摘要
- 分支
- 时间：任务开始 / 分支创建 / 合并
- 关键决策
- 已知局限
- 状态：进行中 / 待审查 / 已合并 / 废弃

## 任务列表

### A2-MIG-002：迁入 S0 数据生成模块

- 任务 ID：A2-MIG-002
- 需求摘要：从 `method_a` 仓库 `code/method-a1/src/data_generation/` 迁入仿真后端、S0 数据集生成模块与支撑工具，归档至 `code/method-a2/src/data_generation/`，并更新代码生成状态记录。
- 分支：`main`（沿用仓库初始化分支，尚未设置远端，未推送）
- 时间：任务开始 2026-09-23 / 分支创建沿用既有分支 / 合并待定
- 关键决策：迁移范围限定为两个仿真后端（`opendss_sim.py` 与 `mock_sim.py`）、S0 数据集生成器（`dataset_builder.py`）与两个支撑工具（`waveform.py`、`topology.py`），共 5 个模块；`e0_builder.py`、`e0_cov_builder.py`、`e4_a0_builder.py` 与 `paired_response_builder.py` 四个数据集生成器经确认不迁入；模块内容逐字节保留，仅把包初始化文件的模块级 docstring 由 Method-A1 改为 Method-A2；不迁入依赖这些模块的训练器、损失、模型、识别性与脚本，故本工作区当前不具备实验执行能力；`src/` 已由子包承载，其 `.gitkeep` 占位随之删除。
- 已知局限：唯一迁入的生成器 `build_dataset` 默认后端为需要 COM 注册的 OpenDSS `FaultSimulator`，本工作区未验证任何真实数据生成运行，仅完成 5 个模块的导入验证；其驱动脚本 `scripts/generate_dataset.py` 未迁入，目前只能通过模块接口调用；`src/` 缺少包初始化文件，`src.data_generation` 依赖命名空间包机制导入；迁移后模块内部保留 `code/method-a1` 时代的私有成员耦合，未作重构。
- 状态：待审查

### A2-MIG-001：迁移 Method-A2 项目架构至本工作区

- 任务 ID：A2-MIG-001
- 需求摘要：参考同父目录下的 `method_a` 仓库，把项目治理架构迁移至 `method_a2` 工作区：建立根 `AGENTS.md`、`code/AGENTS.md`、`code/CODE_CONVENTIONS.md`、`code/CODEGEN_STATUS.md`、`docs/git-rules.md`、`docs/INDEX.md`、`docs/TASKS.md`、`.gitignore`、`.githooks/commit-msg` 与 `code/method-a2/` 同构目录骨架，并迁入 Method-A2 的六篇计划与报告文档。
- 分支：`main`（新仓库初始化引导提交，尚未设置远端，未推送）
- 时间：任务开始 2026-09-23 / 分支创建 2026-09-23（`git init -b main`）/ 合并待定
- 关键决策：迁移范围限定为治理骨架与 Method-A2 文档，业务代码、数据集、模型权重与运行产物一律不迁，使本工作区成为纯架构与文档基线；方法目录采用 `code/method-a2/`，符合 `code/CODE_CONVENTIONS.md` 的 `code/method-<name>/` 约定；文档正文保留 method_a 仓库运行时的原始路径口径，仅在文件头摘要追加路径口径说明，不为迁入而改写正文；`.gitignore` 把数据与产物忽略规则由整目录排除改为目录内容排除并保留各目录 `.gitkeep`，使九项目录骨架可被跟踪；编译产物缓存与 `.pytest_cache` 不迁。
- 已知局限：本工作区尚无业务代码、测试与脚本，`code/method-a2/` 下 `src/`、`tests/`、`scripts/` 为空目录，无任何可执行验证能力；迁移后文档内的 `code/method-a1/...` 路径在本工作区不解析，需待代码与产物迁入后改写；未设置远端，未 Push，未合并。
- 状态：待审查
