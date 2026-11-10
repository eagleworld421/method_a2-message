<!-- 摘要：规定把故障电阻双范围数据集从 IEEE13 换到 IEEE37 的完整生成方案：候选轴剔除电源母线后为 38 个母线、39 个候选；两套数据均用 40 个事件族、对数均匀采样，分别覆盖 0.01–20 Ω 与 10–100 Ω；固定输出目录、统一入口命令、验收数字、候选间隔审计与失败恢复方式。 -->

# IEEE37 故障电阻双范围数据集生成实施计划

> **执行约定：** 任务按顺序执行，每个任务的判据未满足时不得进入下一任务。本计划依据 2026-11-02 的确认结论编写：候选轴不包含电源母线、事件族数为 40、电阻范围沿用 0.01–20 Ω 与 10–100 Ω 两组、生成到两个新数据集。

**目标：** 在 IEEE37 算例上生成两套仅故障电阻范围不同的 S0 数据集，作为继 IEEE13 之后的第二个测试系统。

**方案：** 复用 `main.py`、`src/data_generation/` 与既有的候选间隔审计脚本。先由 `FaultSimulator` 在编译电路后剔除电源母线（`sourcebus`），再按 40 个事件族、40 组对数均匀阻抗分别生成两个数据集，最后用 `scripts/run_oracle_reachability.py` 输出测试集与训练集的候选间隔审计。

**技术栈：** Python、NumPy、OpenDSS COM、pytest。

**依据：** 用户在 2026-11-02 的要求与确认结论；项目级规则见 `AGENTS.md`、`code/AGENTS.md` 和 `code/CODE_CONVENTIONS.md`。

## 与 IEEE13 计划的差异

- 算例由 `ieee13` 改为 `ieee37`。
- 母线与候选轴剔除电源母线：`sourcebus` 由理想电压源钳位，其电压不随故障位置变化，保留会使该候选与无故障候选几乎重合。剔除在 `FaultSimulator._compile_and_solve_base` 中完成，剔除结果写入 `meta.json` 的 `excluded_source_buses` 字段。
- 由于母线数由 16 变为 38，同样的 40 个事件族会产生约 2.4 倍的样本量，验收数字相应改变。

## 全局约束

- 只改变 `res_min`、`res_max`、`impedance_sampling` 三个数据生成参数；算例与母线筛选规则由本轮新固定。
- 两个数据集均使用 `case_name=ieee37`、`fault_classes=(0,)`、`normal_ratio=0.8`、`seed=342`、`samples_per_bus=40`。
- 保持 `fs=200`、`pre_cycles=1`、`post_cycles=2`、`val_ratio=None`、`test_ratio=0.15`。
- 不启用 `hard_pair_file`。
- 输出目录：`code/method-a2/data/s0-ieee37-r0p01-20-loguniform/` 与 `code/method-a2/data/s0-ieee37-r10-100-loguniform/`。
- 最终验收只读取对应数据目录中的 `meta.json`、数组文件与 `feature_scaler.npz`，不使用 `checkpoint/`。
- 不新增生产脚本；除 `tests/` 下针对母线剔除规则的单元测试外，不新增代码文件。

## 规模与划分（40 个事件族）

- 母线 38 条、候选 39 个、有向边 74 条（无向 37 条）。
- 故障样本：40 × 38 = 1520。
- 正常样本：`round(1520 × 0.8)` = 1216。
- 样本总数：2736。
- 划分：32 个训练族与全部正常样本共 2432 条，8 个测试族共 304 条，不设验证集。

## 验收标准

- 两个数据集的 `meta.json` 记录 `case="ieee37"`、`n_nodes=38`、`n_candidates=39`、`n_edges=74`。
- `excluded_source_buses` 记录为 `["sourcebus"]`。
- 数据集 A 记录 `res_min=0.01`、`res_max=20.0`、`impedance_sampling="log_uniform"`；数据集 B 记录 `res_min=10.0`、`res_max=100.0`、`impedance_sampling="log_uniform"`。
- `n_samples=2736`、`n_fault=1520`、`n_normal=1216`、`n_train=2432`、`n_test=304`。
- `progress.jsonl` 最后一条记录为 `end`；输出目录包含全部数组、`feature_scaler.npz` 与 `meta.json`。
- 候选间隔审计产物存在：`output/data-audit/<数据目录名>/<划分>/oracle_metrics.json` 与同目录 `oracle_per_event.npz`。

## 操作步骤

### 任务 0：母线剔除规则验证

- [ ] 运行单元测试，确认剔除规则对大小写与相别后缀的处理：

```powershell
python -m pytest -q tests/test_source_bus_exclusion.py
```

- [ ] 加载 IEEE37 后确认母线数为 38、母线名不含 `sourcebus`、`excluded_bus_names` 为 `["sourcebus"]`，且基态求解可行、电压窗口形状为 `(38, 6)`。

### 任务 1：生成 `0.01–20 Ω` 数据集

**文件：**

- 输出：`code/method-a2/data/s0-ieee37-r0p01-20-loguniform/`
- 日志：`code/method-a2/logs/data-s0-ieee37-r0p01-20-loguniform/`

- [ ] 执行（在 `code/method-a2/` 下）：

```powershell
python main.py --mode data `
  --output-dir data/s0-ieee37-r0p01-20-loguniform `
  --case-name ieee37 --samples-per-bus 40 --normal-ratio 0.8 --seed 342 `
  --res-min 0.01 --res-max 20 --impedance-sampling log_uniform `
  --fault-classes 0 --run-id data-s0-ieee37-r0p01-20-loguniform
```

- [ ] 检查 `progress.jsonl` 最后一条记录为 `end`，且输出目录包含全部数组、`feature_scaler.npz` 和 `meta.json`。

### 任务 2：生成 `10–100 Ω` 数据集

**文件：**

- 输出：`code/method-a2/data/s0-ieee37-r10-100-loguniform/`
- 日志：`code/method-a2/logs/data-s0-ieee37-r10-100-loguniform/`

- [ ] 执行（在 `code/method-a2/` 下）：

```powershell
python main.py --mode data `
  --output-dir data/s0-ieee37-r10-100-loguniform `
  --case-name ieee37 --samples-per-bus 40 --normal-ratio 0.8 --seed 342 `
  --res-min 10 --res-max 100 --impedance-sampling log_uniform `
  --fault-classes 0 --run-id data-s0-ieee37-r10-100-loguniform
```

- [ ] 检查 `progress.jsonl` 最后一条记录为 `end`，且输出目录包含全部数组、`feature_scaler.npz` 和 `meta.json`。

### 任务 3：候选间隔审计（既有脚本）

**文件：**

- 审计产物：`code/method-a2/output/data-audit/<数据目录名>/<划分>/`。

- [ ] 对每个数据集分别执行测试集与训练集两次，命令模板：

```powershell
python scripts/run_oracle_reachability.py `
  --data-dir data/s0-ieee37-r0p01-20-loguniform `
  --output-dir output/data-audit/s0-ieee37-r0p01-20-loguniform/test `
  --split test --run-id s0-ieee37-r0p01-20-loguniform `
  --reference-metrics output/node-marker-mvp/train-s0-marker-run1/metrics.json
```

- [ ] 记录 `normalized_margin_median`、`normalized_margin_p05`、`normalized_margin_min`、`normalized_margin_below_budget_ratio` 与 Oracle Top-1、Top-3。
- [ ] 误差预算沿用 `output/node-marker-mvp/train-s0-marker-run1/metrics.json` 的 `test_family_loss`；该预算来自 IEEE13 的 16 母线模型，跨算例只作参考，读数时必须注明。

### 任务 4：回归检查

```powershell
python -m pytest -q tests/
```

- [ ] 全量测试通过（仓库既有的 `test_run_entry` 参数同步失败不算本轮引入，需在报告中注明）。

## 恢复与重跑

- 若输出目录已存在且不是本次运行产物，先停止并改用新的运行目录，不覆盖旧数据。
- 若进度文件有 `error` 记录，先读 `logs/<run-id>/child.log` 确认原因；入口会先自动重试一次，两次都失败才报错。
- 若进程写入 `end` 后仍未退出或直接崩溃，以进度文件和数组完整性判断是否成功，不要只依据进程退出码。
- 若只需重新生成其中一个范围，只重跑对应任务。

## 复核重点

- 母线剔除是否真正生效：`n_nodes=38`、母线名不含 `sourcebus`、边数为 74。
- `log_uniform` 是否真正应用于 `z_fault`：由 `y_resist` 的最小值、最大值与中位数间接核对，注意独立阻抗抽样数等于事件族数。
- 两个数据集的负荷倍率、故障类型与窗口随机数是否逐族一致：同一 `seed` 下除故障电阻外应完全相同。
- 与 IEEE13 数据的口径差异：IEEE13 的两份既有数据集仍包含电源母线（16 母线、17 候选），本轮 IEEE37 数据为 38 母线、39 候选；两者的候选轴口径不同，跨算例比较时须注明。

## 本轮不做的事

- 不重新生成 IEEE13 数据，不改变既有 IEEE13 数据集。
- 不进入训练，不做模型对比。
- 不产出 `docs/project/reports/` 下的正式结果报告。
