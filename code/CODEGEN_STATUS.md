<!-- 摘要：记录本工作区各方法代码实现阶段、目录职责和关键可执行能力；Method-A2 已迁入 S0 数据生成模块与特权响应 predictor 模型模块，并新增判据一共享状态可恢复性 Probe 脚本与测试且已完成 1600 事件全量运行；数据集、模型权重与其余实验脚本尚未迁入。 -->

# 代码生成状态

## 总体状态

- 项目状态：本工作区由 `method_a` 仓库迁移得到，已完成治理架构、Method-A2 文档、S0 数据生成模块与模型模块迁移，并新增判据一共享状态可恢复性 Probe 脚本与测试，已完成 1600 事件全量运行；数据集与模型权重仍从 `method_a` 仓库读取。
- 代码根目录：`code/`。
- 方法目录规范：每个方法使用独立的 `code/method-<name>/` 目录，并包含同构的 `src/`、`tests/`、`scripts/`、`data/`、`checkpoint/`、`output/` 和 `logs/` 目录。
- 代码组织规范：以 `code/CODE_CONVENTIONS.md` 为唯一正文来源。

## 方法状态

### method-a2

- 状态：可运行判据一实验；其余实验脚本、数据集与权重未迁入。
- 代码目录：`code/method-a2/`。
- 已迁入的数据生成模块：`src/data_generation/` 含包初始化文件与 5 个模块。
  - 仿真后端两个：`opendss_sim.py` 提供基于 OpenDSS COM 接口的 `FaultSimulator`、故障规格枚举与故障命令构建；`mock_sim.py` 提供不依赖 OpenDSS 的确定性 `MockFaultSimulator`，与前者保持兼容的最小接口，供 smoke 与契约测试使用。
  - 数据集生成器一个：`dataset_builder.py` 生成 S0 全候选签名数据集，含 `build_dataset`、`generate_signature_bank` 与 `load_dataset`。
  - 支撑工具两个：`waveform.py` 由 OpenDSS 相量锚点合成动态电压相量窗口；`topology.py` 构建候选边、拓扑掩码与节点观测掩码。
- 已迁入的模型模块：`src/model/` 含包初始化文件与 3 个模块。`privileged_response_predictor.py` 提供 `CandidateConditionedEncoder`、`PairResponseHead` 与 `PrivilegedResponseSystem`；`gnn.py` 提供 `TopologyGNN`；`temporal.py` 提供 `TemporalEncoder`。实测 `teacher.pt` 的全部 87 项状态字典条目可无缺失、无冗余载入，可训练参数 194372。
- 新增的实验脚本：`scripts/run_error_source_criterion1_probe.py` 实现判据一共享状态可恢复性 Probe，冻结既有编码器后在其表示与观测池化特征上拟合岭回归与多类逻辑回归，含四类负对照、块内一致性、位置可恢复性核验、逐维判定与 9 项产物写出，并对长任务做逐次拟合增量落盘。
- 新增的测试：`tests/test_error_source_criterion1_probe.py`，28 项，覆盖度量定义、按 block 聚合与子采样、训练侧标准化、正则选择口径、判定分支、报告与 CSV 产物渲染、以及环境差异下的求解器选择。
- 已完成的运行：`output/error-source-v2/c1run-20260923T083606Z-seed342/criterion-1/`（1600 事件全量）与 `output/error-source-v2/c1smoke-20260923-seed342/criterion-1/`（每划分 8 个 block 的冒烟）。
- 运行环境注意事项（已在产物 `config.json` 中固化）：本机 `sklearn 1.0.2` 与 `scipy 1.13.1` 不兼容，`Ridge` 默认 `auto`/`cholesky` 求解器会抛 `TypeError`，改用 `lsqr`；多类逻辑回归改用 `newton-cg`，同一 16 类问题上较 `lbfgs` 快约 119 倍。两项选择均不改变 multinomial 与正则网格口径。
- 目录状态：`data/`、`checkpoint/` 为空目录并保留 `.gitkeep` 占位；`main.py`、`README.md`、`requirements.txt` 尚未创建。
- 待迁入内容（来源为 `method_a` 仓库 `code/method-a1/`）：
  - 未迁入的数据集生成器 4 个：`e0_builder.py`、`e0_cov_builder.py`、`e4_a0_builder.py`、`paired_response_builder.py`。四者均依赖已迁入的仿真后端与支撑工具，迁入后即可运行。
  - 未迁入的其他源代码：`src/identifiability.py`、`src/pi_response_dataset.py`、`src/pi_response_losses.py`、`src/pi_response_trainer.py`、`src/pi_response_heartbeat.py`、`src/losses.py`。
  - 实验脚本 10 个：`probe_a2_structure.py`、`probe_a2_structure_final.py`、`probe_a2_error_source.py`、`probe_a2_predictor_error.py`、`probe_a2_margin_floor.py`、`run_error_source_experiments.py`、`audit_error_source_results.py`、`run_identifiability_audit.py`、`run_identifiability_nn.py`、`run_identifiability_baseline.py`。
  - 测试 2 个：`test_error_source_experiments.py`、`test_identifiability.py`。
  - 数据集与权重：`data/pi-response/paired-v1-ieee13-1600-seed342`、`data/pi-response/paired-v1-ieee13-640-seed342`、`data/pi-response-smoke/`、`data/pi-response-probe/` 与 `checkpoint/pi-response/pi-response-full-20260920-1600-seed342-v2/`。判据一实验目前在运行期从 `method_a` 仓库读取并在 `config.json` 中固化来源，尚未迁入本工作区。

## 更新要求

- 新增或删除模块、改变目录职责、完成关键代码阶段或确认新的设计决策时，更新对应方法状态。
- 状态记录只反映代码生成进度，不替代 `docs/TASKS.md` 的 Git 任务登记。
