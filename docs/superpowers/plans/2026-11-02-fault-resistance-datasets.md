<!-- 摘要：规定在保持 IEEE13 拓扑、故障类型、窗口参数、随机种子与数据划分不变的条件下，用对数均匀采样生成 0.01–20 Ω 与 10–100 Ω 两个 S0 数据集。正式规模为 40 组、1152 样本；统一由 main.py 在子进程中执行；先做 5 组试生成并在确认成功后删除试生成数据；候选间隔审计只使用既有的 run_oracle_reachability.py，不新增脚本，不产出正式结果报告。 -->

# 故障电阻双范围数据集生成实施计划

> **执行约定：** 以下任务按顺序执行，每个任务的判据未满足时不得进入下一任务。本计划按 2026-11-02 的确认结论编写：正式规模 40 组、统一入口 `main.py`、审计只用既有脚本、试生成数据不保留、不出正式报告。

**目标：** 在保持当前数据生成流程、拓扑、故障类型、采样频率、窗口长度、划分方式和随机种子不变的前提下，生成两个仅改变故障电阻范围的数据集。

**方案：** 复用 `main.py` 与 `src/data_generation/dataset_builder.py` 的既有实现，以 `main.py --mode data` 在子进程中分别生成 `0.01–20 Ω` 与 `10–100 Ω` 两个 S0 数据集。先用 `samples_per_bus=5` 做试生成，确认流程与产物可用后删除试生成数据；再用 `samples_per_bus=40` 做正式生成。两个阶段均使用对数均匀采样，分别写入独立的 `data/` 子目录；生成完成后用既有 `scripts/run_oracle_reachability.py` 输出候选间隔审计。

**技术栈：** Python、NumPy、OpenDSS COM、pytest。

**依据：** 用户在 2026-11-02 的要求与同日确认结论；项目级规则见 `AGENTS.md`、`code/AGENTS.md` 和 `code/CODE_CONVENTIONS.md`。

## 全局约束

- 只改变 `res_min`、`res_max` 和 `impedance_sampling` 三个数据生成参数。
- 两个数据集均使用 `case_name=ieee13`、`fault_classes=(0,)`、`normal_ratio=0.8`、`seed=342`；试生成使用 `samples_per_bus=5`，正式生成使用 `samples_per_bus=40`。
- 保持 `fs=200`、`pre_cycles=1`、`post_cycles=2`、`val_ratio=None`、`test_ratio=0.15`。
- 不启用 `hard_pair_file`，避免主动困难候选采样分支影响本轮数据生成。
- 试生成写入 `code/method-a2/data/trial-s0-r0p01-20-loguniform/` 与 `code/method-a2/data/trial-s0-r10-100-loguniform/`；正式生成写入 `code/method-a2/data/s0-r0p01-20-loguniform/` 与 `code/method-a2/data/s0-r10-100-loguniform/`。
- 试生成数据在正式生成成功并通过验收后删除，不保留；删除范围包括对应的 `data/` 与 `logs/` 目录。
- 不新增任何脚本；审计只使用仓库既有脚本。
- 两个数据集各自独立，不拼接、不合并，也不在本轮进入训练。
- 最终验收只读取对应数据目录中的 `meta.json`、数组文件和 `feature_scaler.npz`，不使用 `checkpoint/`。

## 数据生成入口

- 统一使用 `python main.py --mode data`：数据生成在子进程中执行，父进程只依据 `progress.jsonl` 的结束记录判定成功，OpenDSS 引擎在进程收尾时的原生崩溃不影响判定（详见 `code/method-a2/README.md`）。
- `main.py` 会忽略 `--progress`，进度固定写入 `logs/<run-id>/progress.jsonl` 与同目录的 `progress.latest.json`。
- `main.py` 的 `--res-min` 与 `--res-max` 默认值取自配置文件的 100 与 1000，必须显式覆盖为本计划的区间。
- 以下命令均在方法目录 `code/method-a2/` 下执行；输出目录用相对当前目录的路径给出，避免像历史上那样因工作目录不同而写到仓库根目录。
- 两次数据生成都会把控制台日志写入 `logs/main-data.log`（按运行模式命名），后一次覆盖前一次；每次运行的进度记录按 `run-id` 分开保留。

## 规模与划分（40 组口径）

- 故障样本：40 个事件族 × 16 条母线 = 640。
- 正常样本：`round(640 × 0.8)` = 512。
- 样本总数：1152。
- 划分为训练、测试两段，不设验证集：32 个训练族加全部正常样本共 1024 条，8 个测试族共 128 条。
- 试生成（5 组）为 80 个故障样本与 64 个正常样本，共 144 条，仅用于流程验证。

## 验收标准

- 数据集 A 的 `meta.json` 记录 `res_min=0.01`、`res_max=20.0`、`impedance_sampling="log_uniform"`。
- 数据集 B 的 `meta.json` 记录 `res_min=10.0`、`res_max=100.0`、`impedance_sampling="log_uniform"`。
- 两个数据集的 `n_nodes=16`、`n_candidates=17`、`n_samples=1152`、`n_fault=640`、`n_normal=512`、`n_train=1024`、`n_test=128`。
- 两个数据集的 `progress.jsonl` 最后一条记录为 `end`，输出目录包含全部数组文件、`feature_scaler.npz` 与 `meta.json`。
- 两份数据的随机种子、负荷倍率与分组结构一致，唯一差异为故障电阻范围；该性质由同一 `seed` 与相同的抽样顺序保证，并在此后的复核中只作读数核对，不再另写脚本验证。
- 候选间隔审计产物存在：`output/data-audit/<数据目录名>/<划分>/oracle_metrics.json` 与同目录 `oracle_per_event.npz`。

## 操作步骤

### 任务 0：试生成（5 组，两个区间）

**文件：**

- 输出：`code/method-a2/data/trial-s0-r0p01-20-loguniform/`、`code/method-a2/data/trial-s0-r10-100-loguniform/`
- 日志：`code/method-a2/logs/trial-s0-r0p01-20-loguniform/`、`code/method-a2/logs/trial-s0-r10-100-loguniform/`

- [ ] 执行（在 `code/method-a2/` 下）：

```powershell
python main.py --mode data `
  --output-dir data/trial-s0-r0p01-20-loguniform `
  --samples-per-bus 5 --normal-ratio 0.8 --seed 342 `
  --res-min 0.01 --res-max 20 --impedance-sampling log_uniform `
  --fault-classes 0 --run-id trial-s0-r0p01-20-loguniform

python main.py --mode data `
  --output-dir data/trial-s0-r10-100-loguniform `
  --samples-per-bus 5 --normal-ratio 0.8 --seed 342 `
  --res-min 10 --res-max 100 --impedance-sampling log_uniform `
  --fault-classes 0 --run-id trial-s0-r10-100-loguniform
```

- [ ] 判据：两次运行的控制台均输出“数据生成完成”，`progress.jsonl` 最后一条为 `end`，输出目录的 `n_samples=144`、`n_fault=80`、`n_normal=64`。
- [ ] 判据满足后再删除试生成数据（删除动作放在任务 1、任务 2 均成功之后执行，保留到那时作为对照）。

### 任务 1：生成 `0.01–20 Ω` 数据集（40 组）

**文件：**

- 输出：`code/method-a2/data/s0-r0p01-20-loguniform/`
- 日志：`code/method-a2/logs/data-s0-r0p01-20-loguniform/`

- [ ] 执行（在 `code/method-a2/` 下）：

```powershell
python main.py --mode data `
  --output-dir data/s0-r0p01-20-loguniform `
  --samples-per-bus 40 --normal-ratio 0.8 --seed 342 `
  --res-min 0.01 --res-max 20 --impedance-sampling log_uniform `
  --fault-classes 0 --run-id data-s0-r0p01-20-loguniform
```

- [ ] 检查 `progress.jsonl` 最后一条记录为 `end`，且输出目录包含全部数组、`feature_scaler.npz` 和 `meta.json`。

### 任务 2：生成 `10–100 Ω` 数据集（40 组）

**文件：**

- 输出：`code/method-a2/data/s0-r10-100-loguniform/`
- 日志：`code/method-a2/logs/data-s0-r10-100-loguniform/`

- [ ] 执行（在 `code/method-a2/` 下）：

```powershell
python main.py --mode data `
  --output-dir data/s0-r10-100-loguniform `
  --samples-per-bus 40 --normal-ratio 0.8 --seed 342 `
  --res-min 10 --res-max 100 --impedance-sampling log_uniform `
  --fault-classes 0 --run-id data-s0-r10-100-loguniform
```

- [ ] 检查 `progress.jsonl` 最后一条记录为 `end`，且输出目录包含全部数组、`feature_scaler.npz` 和 `meta.json`。
- [ ] 两个数据集的验收标准全部满足后，删除试生成数据与对应日志。

### 任务 3：候选间隔审计（既有脚本）

**文件：**

- 读取：两个数据集目录中的 `meta.json`、`.npy` 与 `feature_scaler.npz`。
- 审计产物：`code/method-a2/output/data-audit/<数据目录名>/<划分>/oracle_metrics.json` 与同目录 `oracle_per_event.npz`。

- [ ] 对每个数据集分别执行测试集与训练集两次（在 `code/method-a2/` 下）：

```powershell
python scripts/run_oracle_reachability.py `
  --data-dir data/s0-r0p01-20-loguniform `
  --output-dir output/data-audit/s0-r0p01-20-loguniform/test `
  --split test --run-id s0-r0p01-20-loguniform `
  --reference-metrics output/node-marker-mvp/train-s0-marker-run1/metrics.json

python scripts/run_oracle_reachability.py `
  --data-dir data/s0-r0p01-20-loguniform `
  --output-dir output/data-audit/s0-r0p01-20-loguniform/train `
  --split train --run-id s0-r0p01-20-loguniform `
  --reference-metrics output/node-marker-mvp/train-s0-marker-run1/metrics.json
```

- [ ] 对 `s0-r10-100-loguniform` 重复上述两条命令，仅替换数据目录与产物目录。
- [ ] 记录每个数据集的 `normalized_margin_median`、`normalized_margin_p05`、`normalized_margin_min`、`normalized_margin_below_budget_ratio` 与 Oracle Top-1、Top-3。
- [ ] 误差预算固定取 `output/node-marker-mvp/train-s0-marker-run1/metrics.json` 的 `test_family_loss`（约 0.0043），该预算来自既有模型而非本轮数据，读数时须注明。

### 任务 4：运行回归检查

- [ ] 运行数据生成相关测试（在 `code/method-a2/` 下）：

```powershell
python -m pytest -q tests/test_impedance_sampling.py tests/test_accuracy_upgrade.py tests/test_marker_mvp.py
```

- [ ] 生成结束后核对交付口径：任何 OpenDSS 生成失败、缺失结束记录、元数据不匹配或数组形状异常均视为失败，不进入训练。

## 恢复与重跑

- 若输出目录已存在且不是本次运行产物，先停止并改用新的运行目录，不覆盖旧数据。
- 若进度文件有 `error` 记录，先读 `logs/<run-id>/child.log` 确认原因；入口会先自动重试一次，两次都失败才报错。
- 若进程写入 `end` 后仍未退出或直接崩溃，以进度文件和数组完整性判断是否成功，不要只依据进程退出码。
- 若只需要重新生成其中一个范围，只重跑对应任务，另一数据集保持不变。

## 复核重点

- `log_uniform` 是否真正应用于 `z_fault`，而不是只写入元数据；由 `y_resist` 的最小值、最大值与中位数间接核对。
- 两个区间的分组结构、负荷倍率与窗口随机数是否一致；同一 `seed` 下除故障电阻外应逐族对应。
- 高阻区间是否产生大量近零候选间隔；该结果作为数据性质记录，不通过随机扰动掩盖。
- 生成结束后训练入口是否显式指向对应新数据目录，避免 `MODE=all` 继续读取旧的 `s0-lg-run3`。

## 本轮不做的事

- 不合并两个电阻区间，不生成第三份数据集。
- 不进入训练，不做模型对比。
- 不新增脚本；几何中点比例、逐元素配对一致性等没有既有脚本支撑的检查不做脚本化审计，只在读数层面核对。
- 不产出 `docs/project/reports/` 下的正式结果报告。
