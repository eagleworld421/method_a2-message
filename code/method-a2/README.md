# Method-A2 方法目录说明

本文件说明 `code/method-a2/` 的运行方式、目录职责与已知限制。

## 运行方式

统一入口为 `main.py`，参数全部写在文件顶部的配置块中，不需要在命令行手工拼接参数：

1. 在 PyCharm 中右键 `code/method-a2/main.py` 选择运行，或在项目根目录执行
   `python code/method-a2/main.py`；
2. 用顶部的 `MODE` 选择阶段：`"data"` 只生成数据、`"train"` 只训练、
   `"all"` 先生成数据再训练；
3. 用 `CommonConfig`、`DataConfig`、`TrainConfig` 三个配置对象修改参数；配置字段名与
   两个阶段脚本的命令行参数名逐字一致；
4. 运行后控制台打印本次实际参数与实时进度（阶段、完成度、已用时长与 ETA）；完整控制台
   输出写入 `logs/main-<mode>.log`，结构化进度写入 `logs/<run-id>/progress.jsonl`；
5. 需要临时改参而不改文件时，入口参数之外的命令行参数会透传给当前阶段，例如
   `python code/method-a2/main.py --epochs 5 --pooling trimmed_mean`；加 `--dry-run`
   只打印本次参数而不执行。

运行模式为 `all` 时，两个阶段必须使用不同的运行标识与产物目录，因此数据生成使用
`<run-id>-data`、训练使用 `<run-id>-train` 作为运行标识，共用的 `--output-dir` 也同样
加上 `-data` 与 `-train` 后缀；单独运行某一阶段时不做任何改写。

路径口径：配置中的相对路径一律相对本方法目录解析；命令行给出的相对路径按标准语义相对
当前工作目录解析；`CommonConfig.case_name` 是 OpenDSS 算例名（如 `ieee13`），不是路径。

## 数据生成为什么在子进程中执行

以 OpenDSS COM 求解数据的进程无法正常退出：引擎被 `Dispatch` 加载后，解释器收尾会以
Windows 访问冲突（退出码 `-1073741819`，即 `0xC0000005`）结束，且可重复地表现为进程
无限挂起。实测已排除下列方案：显式 `ClearAll` 后释放引用、`pythoncom.CoUninitialize()`、
`os._exit(0)`、把引擎引用保留到进程结束、以及 OpenDSS 引擎自身的 `CloseDSS`（本版本
COM 接口不提供该方法）。仅调用 `CoInitialize`/`CoUninitialize` 而不创建引擎时退出码正常。

因此 `main.py` 在 `MODE` 为 `"data"` 或 `"all"` 时把数据生成交给子进程执行，父进程
（即 PyCharm 中运行的进程）从不加载 OpenDSS 引擎：

- 父进程用与配置文件一致的参数启动 `main.py --child-data`，并每 0.5 秒读取
  `logs/<run-id>/progress.latest.json`，把进度逐行显示在控制台；
- 子进程的标准输出与标准错误统一写入 `logs/<run-id>/child.log`，不继承到控制台。
  OpenDSS 引擎在进程收尾时写到标准错误的消息（形如 `Runtime error 216 at
  00000070E27E79`，Windows 应用日志记为 `OpenDSSEngine.dll` 访问冲突）因此只留在该
  文件里，不会被误读成本次运行的失败原因；
- 子进程写完 `progress.jsonl` 的 `end` 记录即视为数据生成成功，父进程随即强制终止仍然
  挂起的子进程，并自行正常退出（退出码 0），因此 PyCharm 不再把一次成功的数据生成判为
  崩溃；
- 若子进程写入 `error` 记录，或写完结束记录前退出，父进程判定本次尝试失败并重新启动
  子进程重试一次；两次尝试都不成功时才报“数据生成失败”或“数据生成未完成”，并附上
  `progress.jsonl` 记录的失败原因与 `child.log` 末尾若干行；
- 超过两小时仍未写入结束记录则判定超时并终止子进程，超时不重试。

训练阶段不涉及 OpenDSS，因此 `MODE` 为 `"train"` 或 `"all"` 时训练仍在父进程内直接执行，
其进度与产物行为不变。

## 目录职责

- `main.py`：数据生成与训练的统一运行入口，只负责参数、进度显示、子进程调度与调用。
- `src/`：业务源代码。`src/data_generation/` 生成 S0 数据集，`src/model/` 为模型模块，
  `src/training/` 为训练步骤与评价指标，`src/progress.py` 为进度落盘工具。
- `scripts/`：一次性诊断、数据处理与实验辅助脚本。`generate_dataset.py` 与
  `train_marker_mvp.py` 分别是数据生成与训练的命令行入口，`main.py` 调用其执行函数。
- `tests/`：单元测试与契约测试。`tests/test_run_entry.py` 覆盖入口参数映射、路径解析与
  进度显示；`tests/test_data_isolation.py` 覆盖子进程隔离、完成与失败判定。
- `data/`：数据集；`checkpoint/`：模型权重与中间状态；`output/`：最终实验产物；
  `logs/`：运行日志与进度记录。
- `requirements.txt`：尚未创建。

## 已知限制

- 数据生成的子进程会以被强制终止、挂起或访问冲突的方式结束，其退出码（1 或
  `-1073741819`）只作为原始记录打印，不参与成功判定；成功与否以 `progress.jsonl` 的
  结束记录与产物文件为准。直接运行 `scripts/generate_dataset.py` 时仍会遇到同样的退出
  异常，且该异常消息会直接出现在控制台。
- OpenDSS 引擎是原生 Delphi 组件，偶发在首次求解时即抛出异常（形如
  `Runtime error 216 at ...`），与数据内容和参数无关。入口会重试一次，但重试从头重新
  求解，已生成但尚未落盘的样本会被丢弃；每次尝试的输出都追加在
  `logs/<run-id>/child.log`，上一次尝试的进度记录改名为
  `progress.attempt<N>.jsonl` 保留。
- 候选轴口径：`FaultSimulator` 在编译电路后剔除电压源连接的电源母线（IEEE 算例中为
  `sourcebus`），其剔除结果写入 `meta.json` 的 `excluded_source_buses`。因此 IEEE37
  为 38 条母线、39 个候选。2026-11-02 之前生成的 IEEE13 数据集仍包含电源母线
  （16 条母线、17 个候选），跨算例比较时须注意这一口径差异。
- 数据划分：数据生成默认使用两段（训练与测试）划分，训练默认沿用数据目录内置划分；如需
  验证集，用 `DataConfig.val_ratio` 生成三段划分，或用 `TrainConfig.resplit` 让训练脚本按
  分组重新划分。
- 配置与命令行的一致性：`main.py` 的参数定义与两个脚本的命令行程式由
  `tests/test_run_entry.py` 逐项比对；脚本侧参数被改名、改变默认值或改变取值集合时，该测试
  会失败，需同步更新入口。
- 运行入口不新增产物：数据、模型与报告仍由两个阶段脚本写入 `data/`、`output/` 与 `logs/`，
  最终交付结果从 `output/` 获取。
- `MODE = "all"` 只保证“先生成数据再训练”的顺序，不会自动把训练的数据目录指向刚生成的数据；
  `TrainConfig.data_dir` 与 `DataConfig.output_dir` 需要在配置中显式一致。
