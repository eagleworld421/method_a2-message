<!-- 摘要：规定把 Method-A2 的数据生成与审计流程切换到另一个 OpenDSS 算例时执行的初步测试流程，包括算例可用性探测、小规模试生成、数据契约读数核对、候选间隔审计、训练入口烟测与规模资源评估六个步骤，并给出每一步的命令、判据、产物位置与不通过时的处置方式；本文为流程文档，不含具体算例的实测读数与结论。 -->

# Method-A2 算例切换初步测试流程

> 日期：2026-11-02
> 状态：Working Note
> 适用范围：需要把数据生成流程从当前算例（`ieee13`）换到另一个 OpenDSS 算例（如 `ieee37`、`ieee123`）时的可用性验证，不用于正式数据集生成。
> 优先级：本文与代码、正式计划冲突时，以代码与实际运行结果为准。

## 1. 目的与边界

本流程解决一个问题：在不改动 `src/` 与 `scripts/` 的前提下，用最小代价判断一个新算例能否接入现有生成、审计与训练链路，并给出正式生成前的规模与资源预估。

本文只覆盖初步测试，不产出正式数据集、不产出实验结论、不新增脚本。正式生成应另立计划，并在计划中固定算例、样本规模、电阻范围、验收数字与产物目录。

## 2. 前置条件

- 算例主文件存在：`src/data_generation/opendss_sim.py` 中的 `_DSS_CASES` 记录三个算例的绝对路径，先用文件系统确认目标路径存在。
- OpenDSS COM 可用：Python 能 `import win32com.client`，且 `Dispatch("OpenDSSEngine.DSS")` 成功。当前环境为 64 位 Python 3.9.12 与 `OpenDSSEngine.dll` 11.0.0.1。
- 入口可用：`code/method-a2/main.py` 的 `--mode data` 走子进程隔离，`--case-name` 接受算例名而不是路径。
- 工作目录：命令在 `code/method-a2/` 下执行，输出目录用相对当前目录的路径给出，避免因工作目录不同而写到仓库根目录。

## 3. 步骤一：算例可用性探测

目的：在生成任何数据之前确认拓扑能被正确解析，且目标故障类型在每个母线上物理可行。

做法：写一个临时探测脚本放在系统临时目录（不写入仓库），导入 `FaultSimulator` 与 `enumerate_fault_specs`，构造 `FaultSimulator(<case>)`，再调用一次 `generate_scenario(FaultConfig(fault_class=0, fault_bus=0, z_fault=0.1))`。

需要记录的量：

- `_n_nodes`：母线数，等于 `ActiveCircuit.AllBusNames` 的长度。
- `_base_loads` 的长度：负荷数量。
- `line_params` 的条目数：有向边数量的两倍。
- 邻接矩阵的无向边数量。
- `_bus_phase_nodes`：每个母线实际存在的相别节点。
- 支持目标故障类型的（母线，相别）组合数，以及是否存在无有效相别的母线。
- 首个场景的 `fault_feasible`、`fault_phases`、`pre_v` 形状与 6 个通道的非零情况。

判据（全部满足才进入步骤二）：

- 母线数大于零，负荷数与边数不为零。
- 不存在无有效相别的母线。
- 首个场景 `fault_feasible` 为真且 `fault_phases` 非空。
- `pre_v` 形状为（母线数，6），且 6 个通道均有非零值。
- 邻接矩阵的无向边数与线路、变压器参数一致。

不满足时停止，先排查算例文件与 COM 环境，不进入生成。

## 4. 步骤二：小规模试生成

目的：用最小规模跑通「采样事件、逐候选求解、组装窗口、标准化、落盘、写进度」的完整链路。

命令（在 `code/method-a2/` 下）：

```powershell
python main.py --mode data `
  --output-dir data/trial-<case>-r0p01-20-loguniform `
  --case-name <case> --samples-per-bus 5 --normal-ratio 0.8 --seed 342 `
  --res-min 0.01 --res-max 20 --impedance-sampling log_uniform `
  --fault-classes 0 --run-id trial-<case>-r0p01-20-loguniform
```

规模选择依据：`samples_per_bus` 是事件族数，每族覆盖全部母线，因此组数同时决定独立阻抗抽样数、签名库数量与求解次数。5 组在 16 母线算例上约十几秒，在 39 母线算例上约三分钟，足以暴露链路问题又不必等待。

判据：

- 控制台输出「数据生成完成」，父进程退出码为 0。
- `logs/<run-id>/progress.jsonl` 最后一条记录 `kind` 为 `end`，且 `done` 等于 `total`。
- `meta.json` 中 `n_fault` 等于母线数乘组数，`n_normal` 等于 `round(n_fault × 0.8)`，`n_samples` 等于两者之和。
- `n_train`、`n_test` 与 `split_groups` 的分组数一致：训练族数为 `round(组数 × 0.8)`，其余为测试族，正常样本全部进入训练集。

## 5. 步骤三：数据契约读数核对

目的：确认落盘数组与元数据符合既有 S0 契约，避免形状或字段错位在训练阶段才暴露。

做法：只读取，不修改；读数可以用一次性命令完成，不为它新增仓库脚本。

需要核对的字段与形状：

- `meta.json` 的 `case`、`n_nodes`、`n_candidates`、`n_samples`、`n_fault`、`n_normal`、`n_train`、`n_test`、`n_edges`、`res_min`、`res_max`、`impedance_sampling`、`seed`、`window_len`、`fs`。
- `signature_bank` 形状为（样本数，候选数，母线数，窗口长度，6）。
- `X_obs` 与 `X_full` 形状为（样本数，母线数，窗口长度，6）。
- `candidate_mask` 形状为（样本数，候选数），故障行上的均值应接近 1。
- `edge_index` 形状为（有向边数，2），`edge_attr` 形状为（有向边数，5）。
- `y_resist` 在 `y_detect` 为 1 的行上，最小值与最大值应落在 `[res_min, res_max)` 内。

读数时须注意：独立阻抗抽样数等于事件族数，不等于故障样本数。因此判断「对数均匀是否生效」时要按组数估计容差；样本层面的比例只反映同一族内重复计数，不能直接当作分布证据。

## 6. 步骤四：候选间隔审计

目的：确认该算例的候选响应在数值上可区分，并量化「真实候选与最难错误候选」的间隔；这是数据性质，不是模型能力。

命令（对测试集与训练集各执行一次，在 `code/method-a2/` 下）：

```powershell
python scripts/run_oracle_reachability.py `
  --data-dir data/trial-<case>-r0p01-20-loguniform `
  --output-dir output/data-audit/trial-<case>-r0p01-20-loguniform/test `
  --split test --run-id trial-<case> `
  --reference-metrics output/node-marker-mvp/train-s0-marker-run1/metrics.json
```

说明：该脚本用 `--reference-metrics` 指向的 `metrics.json` 读取 `test_family_loss` 作为预测误差预算；不提供预算时脚本在预算比较处会因空值报错，因此必须显式给出其一。

需要记录的读数：

- `n_events`：该划分内参与计算的事件数。
- `normalized.top1`、`normalized.top3`：Oracle 口径的定位命中率。
- `normalized_margin_median`、`normalized_margin_p05`、`normalized_margin_min`。
- `normalized_margin_below_budget_ratio`：间隔低于误差预算的比例，即「近零间隔」比例。
- 零间隔事件数量与涉及的母线：读 `oracle_per_event.npz` 的 `normalized_margin` 与 `event_index`，与 `y_loc` 对照。

判据：

- 脚本自带门槛为 Top-3 不低于 0.85、Top-1 不低于 0.60，未通过说明该算例的候选在该数据规模下不可区分，需要先处理再正式生成。
- 若零间隔事件集中在个别母线，必须在数据说明中标注；这类并列会让 Oracle Top-1 的并列判定偏向索引较小者，读数偏乐观。

## 7. 步骤五：训练入口烟测

目的：确认模型能按新算例的母线数与窗口长度构建，且前向、反向、评估与落盘链路可用。

命令（在 `code/method-a2/` 下）：

```powershell
python scripts/train_marker_mvp.py `
  --data-dir data/trial-<case>-r0p01-20-loguniform `
  --output-dir output/node-marker-mvp/<run>-smoke `
  --epochs 1 --max-batches 2 --run-id <run>-smoke
```

说明：数据目录不含验证集时，训练脚本会按 `group_id` 重新划分训练、验证与测试集，因此少量组数也能跑通。

判据：退出码为 0，且输出目录存在 `metrics.json`、`predictions.npz`、`model.pt`、`report.md`。烟测只验证兼容性与产物完整性，其结果不用于任何模型结论。

## 8. 步骤六：规模与资源评估

目的：在正式生成前给出时间与磁盘的量化预估，避免因数组体量过大中断。

计算方式：

- 每样本签名库字节数等于候选数乘母线数乘窗口长度乘 6 乘 4。
- 求解次数等于（事件族数加正常样本数）乘候选数。
- 单次求解耗时用步骤二的实测值：试生成总耗时除以求解次数。
- 磁盘占用约等于样本数乘每样本签名库字节数；生成期峰值内存约为其两倍，因为签名字典先全量累积再堆叠成数组。

判据：磁盘与内存占用、以及预计耗时都在可接受范围内再进入正式生成；否则先降低事件族数，再重新评估。

## 9. 判定与收尾

通过条件：步骤一、二、三的判据全部满足，步骤四的门槛通过，步骤五（若执行）退出码为 0。

通过后的处置由使用者决定，二者择一：保留试数据作为对照，或删除试数据与对应日志。若要转为正式生成，必须先更新方法入口配置与正式计划文档，至少在计划中固定算例名、样本规模、电阻范围、输出目录与验收数字，并按更新后的计划执行。

不通过时的处置：保留 `logs/<run-id>/progress.jsonl` 与 `logs/<run-id>/child.log`，按生成失败流程排查，不进入正式生成，也不得用试数据替代正式交付。

## 10. 产物位置

- 进度与原生输出：`logs/<run-id>/progress.jsonl`、`logs/<run-id>/child.log`。
- 控制台日志：`logs/main-data.log`（按运行模式命名，多次运行互相覆盖）。
- 试数据：`data/<run-id>/`。
- 候选间隔审计：`output/data-audit/<run-id>/<划分>/oracle_metrics.json` 与同目录 `oracle_per_event.npz`。
- 训练烟测：`output/node-marker-mvp/<run>-smoke/`。

## 11. 已知偏差与风险

- 收尾崩溃：OpenDSS 引擎在进程退出时会以访问冲突结束（控制台表现为 `Runtime error 216 at ...`，Windows 应用日志记为 `OpenDSSEngine.dll` 访问冲突），与本次生成是否成功无关。成功判定只看 `progress.jsonl` 的 `end` 记录与产物完整性，不看子进程退出码。
- 有效阻抗抽样数等于事件族数：5 组只有 5 个独立 `z_fault`，阻抗分布的统计判断必须按组数而不是样本数做。
- 候选轴包含电源母线：`AllBusNames` 的第 0 项通常是电源母线（如 `sourcebus`），当前实现对全部母线一视同仁，因此该母线也参与故障与候选评分；是否排除属于方法层面的决定，不在本流程内改动。
- 调压器或变压器会引入近似重复母线（例如副边母线与其上游母线电气距离很近），可能压缩候选间隔；若审计发现间隔集中在个别母线上，应按数据性质记录，不通过加扰动掩盖。
- 试生成规模过小：5 组时 Top-1 与间隔统计的方差很大，只用于可用性判断，不得据此得出算例之间优劣的结论。

## 12. 本流程不做的事

- 不新增仓库脚本；探测与读数用临时脚本或一次性命令完成，产物不写入仓库。
- 不修改 `src/` 与 `scripts/` 的既有逻辑，也不为新算例单独分支实现。
- 不在初步测试阶段产出正式数据集、正式报告或训练结论。
- 不把试生成数据当作正式交付，也不把它写入 `docs/project/reports/`。
