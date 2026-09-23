# code/AGENTS.md — code 范围执行规则

本文件只适用于项目根目录下的 `code/**` 路径，是根 `AGENTS.md` 的局部补充，不得削弱根规则。

## 强制读取

- 任何目标路径位于 `code/**` 的代码生成、修改、测试、脚本执行或运行产物写入，都必须遵守本文件。
- 首次写入前必须读取项目根目录的 `code/CODE_CONVENTIONS.md`，并严格执行其全部要求。
- 若无法读取 `code/CODE_CONVENTIONS.md`，禁止继续写入 `code/**`，必须先报告原因。
- `code/CODEGEN_STATUS.md` 是状态记录文件；仅在模块状态、设计决策或代码生成进度发生变化时读取并更新。

## 方法目录边界

每个方法使用独立目录，并重复相同的内部结构：

```text
code/method-a2/
├── main.py
├── README.md
├── requirements.txt
├── src/
├── tests/
├── scripts/
├── data/
├── checkpoint/
├── output/
└── logs/
```

本工作区当前只有 `code/method-a2/` 一个方法目录；新增其他方法时使用 `code/method-<name>/` 等同级目录，不得把不同方法的源代码、数据、模型权重或输出混放。

## 执行要求

- 源代码只放在对应方法的 `src/`；测试只放在对应方法的 `tests/`；辅助工具只放在对应方法的 `scripts/`。
- 原始或生成数据写入对应方法的 `data/`；模型权重和中间状态写入 `checkpoint/`；最终结果和报告写入 `output/`；日志写入 `logs/`。
- 不得将代码、数据、模型权重或最终报告写入方法目录之外，也不得把最终报告写入 `checkpoint/`。
- `code/**` 下 Python 文件的注释和 docstring 使用中文。
- 若任务涉及文档、README、计划或测试命令，相关路径必须使用 `tests/` 和 `docs/`。
