<!-- 摘要：记录本工作区各方法代码实现阶段、目录职责和关键可执行能力；Method-A2 已完成项目架构迁移并迁入 S0 数据生成模块，包含两个仿真后端、一个数据集生成器与两个支撑工具；实验脚本、测试、数据集、模型权重与运行产物尚未迁入。 -->

# 代码生成状态

## 总体状态

- 项目状态：本工作区由 `method_a` 仓库迁移得到，已完成治理架构、Method-A2 文档与数据生成模块迁移；尚未迁入实验脚本、测试、数据集、模型权重与运行产物。
- 代码根目录：`code/`。
- 方法目录规范：每个方法使用独立的 `code/method-<name>/` 目录，并包含同构的 `src/`、`tests/`、`scripts/`、`data/`、`checkpoint/`、`output/` 和 `logs/` 目录。
- 代码组织规范：以 `code/CODE_CONVENTIONS.md` 为唯一正文来源。

## 方法状态

### method-a2

- 状态：架构骨架与数据生成模块已迁入，实验脚本、测试与其余源代码未迁入。
- 代码目录：`code/method-a2/`。
- 当前内容：`src/data_generation/` 已迁入包初始化文件与 5 个模块，按职责分为三类。
  - 仿真后端两个：`opendss_sim.py` 提供基于 OpenDSS COM 接口的 `FaultSimulator`、故障规格枚举与故障命令构建；`mock_sim.py` 提供不依赖 OpenDSS 的确定性 `MockFaultSimulator`，与前者保持兼容的最小接口，供 smoke 与契约测试使用。
  - 数据集生成器一个：`dataset_builder.py` 生成 S0 全候选签名数据集，含 `build_dataset`、`generate_signature_bank` 与 `load_dataset`，调用仿真后端完成逐候选独立求解。
  - 支撑工具两个：`waveform.py` 由 OpenDSS 相量锚点合成动态电压相量窗口；`topology.py` 构建候选边、拓扑掩码与节点观测掩码。
  - 5 个模块全部通过导入验证，导入阶段不触发 OpenDSS 调用；模块间依赖闭合，无指向未迁入模块的引用。
- 目录状态：`tests/`、`scripts/`、`data/`、`checkpoint/`、`output/`、`logs/` 为空目录并保留 `.gitkeep` 占位；`main.py`、`README.md`、`requirements.txt` 尚未创建。
- 待迁入内容（来源为 `method_a` 仓库 `code/method-a1/`）：
  - 未迁入的数据集生成器 4 个：`e0_builder.py` 生成 E0 信息充分性数据；`e0_cov_builder.py` 生成 E0-COV 模板覆盖库；`e4_a0_builder.py` 生成 E4-A0 校准阻抗数据；`paired_response_builder.py` 生成 paired-v1 特权物理条件响应数据集并附数据契约校验。四者均依赖已迁入的仿真后端与支撑工具，迁入后即可运行。
  - 数据生成模块之外的其他源代码：`src/identifiability.py`、`src/pi_response_dataset.py`、`src/pi_response_losses.py`、`src/pi_response_trainer.py`、`src/pi_response_heartbeat.py`、`src/losses.py`、`src/model/privileged_response_predictor.py`、`src/model/gnn.py`、`src/model/temporal.py`。
  - 实验脚本 10 个：`probe_a2_structure.py`、`probe_a2_structure_final.py`、`probe_a2_error_source.py`、`probe_a2_predictor_error.py`、`probe_a2_margin_floor.py`、`run_error_source_experiments.py`、`audit_error_source_results.py`、`run_identifiability_audit.py`、`run_identifiability_nn.py`、`run_identifiability_baseline.py`。
  - S0 数据集生成驱动脚本 `scripts/generate_dataset.py` 尚未迁入，故 `build_dataset` 目前只能通过模块接口调用。
  - 测试 2 个：`test_error_source_experiments.py`、`test_identifiability.py`。
  - 数据集与权重：`data/pi-response/paired-v1-ieee13-1600-seed342`、`data/pi-response/paired-v1-ieee13-640-seed342`、`data/pi-response-smoke/`、`data/pi-response-probe/` 与 `checkpoint/pi-response/pi-response-full-20260920-1600-seed342-v2/`。
  - 运行产物：`output/error-source/esrun-20260922-error-source/`、`output/identifiability/` 与 `logs/error-source/`。

## 更新要求

- 新增或删除模块、改变目录职责、完成关键代码阶段或确认新的设计决策时，更新对应方法状态。
- 状态记录只反映代码生成进度，不替代 `docs/TASKS.md` 的 Git 任务登记。
