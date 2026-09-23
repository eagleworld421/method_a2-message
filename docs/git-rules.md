<!--
本文档：AI Agent Git 操作规范
触发关键词：git 规范、git rules、commit 规则、agent git、agent Git 最佳实践、commit message、trailer、分支管理
检索顺序：2
-->

# AI Agent Git 操作规范

本规范以 Desktop/idea2 的 Git 规则为基线，适配本项目统一的 `docs/` 文档路径。

## 规则 01：Commit Trailer 与 PR 描述联动

非 Checkpoint commit 的 message 末尾必须包含：

```text
<type>(<scope>): <摘要>

<正文>

Agent-Task: <任务ID或需求>
Agent-Decision: <关键取舍及理由>
Agent-Limitation: <已知局限，无则填"无">
```

硬性格式要求：

- 键名仅限 `Agent-Task`、`Agent-Decision`、`Agent-Limitation`。
- 键值分隔符必须是英文 `: `。
- 每条 trailer 独占一行，值内禁止换行，每条值不超过 80 字。
- trailer 块与正文之间必须空一行。

发起 PR 时，`Task Description`、`Key Design Decisions`、`Known Limitations` 直接引用对应 trailer 值，不重复填写。

`commit-msg` hook 使用正则 `^(Agent-Task|Agent-Decision|Agent-Limitation): .+` 校验；不满足要求时拒绝提交。以 `[WIP]` 开头的 Checkpoint commit 可跳过 trailer 检查。

## 规则 02：提交粒度管道

### Checkpoint

当修改超过 5 个文件或任务持续超过 15 分钟时，在以下节点分别建立 Checkpoint：

1. 接口或数据模型定义完成后。
2. 核心逻辑实现完成后。
3. 测试编写完成后。
4. 文档更新完成后。

Checkpoint commit 的 subject 必须以 `[WIP]` 开头，body 仅写一行当前进度，可不带 trailer。

### Rebase

任务完成且尚未 Push 时执行：

```bash
git rebase -i main
git log --oneline main..HEAD
```

必须将全部 `[WIP]` commit squash 至对应语义 commit，并确认不存在 `[WIP]` 残留。已共享分支禁止 Rebase。

### Atomic

- 每个最终 commit 必须可独立理解、可独立编译、可独立回滚。
- 一个 commit 只包含一个逻辑变更。
- 禁止将重构与功能修改混入同一条 commit。

### TASKS 登记

Rebase 完成、Push 前，必须在 `docs/TASKS.md` 登记本次任务。七个字段必须全部填写：

1. 任务 ID，与 `Agent-Task` trailer 一致。
2. 需求摘要，本次实现功能的一句话描述。
3. 分支，本次使用的分支名。
4. 时间，任务开始、分支创建、合并三个时间点。
5. 关键决策，引用 `Agent-Decision` trailer 值。
6. 已知局限，引用 `Agent-Limitation` trailer 值。
7. 状态，Push 后为“待审查”，PR 合并后为“已合并”。

禁止未登记任务就 Push。

## 规则 03：Feature Branch 强制

- 禁止直接 Push 到 `main` 或 `master`。
- 分支命名格式：`agent/<task-id>-<描述>`。
- 每项新任务从最新 `main` 创建新分支。
- 合并由人工触发，Agent 不得自行 Merge。

## 规则 04：Worktree 隔离并发 Agent

- 多个 Agent 并行时，每个 Agent 必须使用独立 Git worktree。
- 创建：`git worktree add <路径> -b <分支名>`。
- 任务完成后清理：`git worktree remove <路径>`。

## 规则 05：规范入口

- 以上规则的执行摘要写入项目根目录 `AGENTS.md`。
- Agent 每次任务启动时必须读取根 `AGENTS.md`。
- 根 `AGENTS.md` 负责指向 `code/AGENTS.md` 与 `code/CODE_CONVENTIONS.md`，不在多个文件中复制代码分类正文。
