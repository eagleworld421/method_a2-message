<!-- 摘要：记录本工作区各方法代码实现阶段、目录职责和关键可执行能力；Method-A2 已完成项目架构迁移，`code/method-a2/` 同构目录骨架已建立，业务源代码、测试、脚本、数据集与运行产物尚未迁入；既有实现仍位于 method_a 仓库的 `code/method-a1/` 下。 -->

# 代码生成状态

## 总体状态

- 项目状态：本工作区由 `method_a` 仓库迁移得到，当前只完成治理架构与 Method-A2 文档迁移，尚未迁入任何业务代码、数据集、模型权重或运行产物。
- 代码根目录：`code/`。
- 方法目录规范：每个方法使用独立的 `code/method-<name>/` 目录，并包含同构的 `src/`、`tests/`、`scripts/`、`data/`、`checkpoint/`、`output/` 和 `logs/` 目录。
- 代码组织规范：以 `code/CODE_CONVENTIONS.md` 为唯一正文来源。

## 方法状态

### method-a2

- 状态：架构骨架已建立，业务代码未迁入。
- 代码目录：`code/method-a2/`。
- 当前内容：`src/`、`tests/`、`scripts/`、`data/`、`checkpoint/`、`output/`、`logs/` 七个同构目录已创建并保留 `.gitkeep` 占位；`main.py`、`README.md`、`requirements.txt` 尚未创建。
- 待迁入内容（来源为 `method_a` 仓库 `code/method-a1/`，本次迁移未执行）：
  - 源代码依赖闭包共 14 个模块，包含 `src/identifiability.py`、`src/pi_response_dataset.py`、`src/pi_response_losses.py`、`src/pi_response_trainer.py`、`src/pi_response_heartbeat.py`、`src/losses.py`、`src/model/privileged_response_predictor.py`、`src/model/gnn.py`、`src/model/temporal.py` 与 `src/data_generation/` 下的 `waveform.py`、`opendss_sim.py`、`paired_response_builder.py`、`dataset_builder.py`、`topology.py`。
  - 实验脚本 10 个：`probe_a2_structure.py`、`probe_a2_structure_final.py`、`probe_a2_error_source.py`、`probe_a2_predictor_error.py`、`probe_a2_margin_floor.py`、`run_error_source_experiments.py`、`audit_error_source_results.py`、`run_identifiability_audit.py`、`run_identifiability_nn.py`、`run_identifiability_baseline.py`。
  - 测试 2 个：`test_error_source_experiments.py`、`test_identifiability.py`。
  - 数据集与权重：`data/pi-response/paired-v1-ieee13-1600-seed342`、`data/pi-response/paired-v1-ieee13-640-seed342`、`data/pi-response-smoke/`、`data/pi-response-probe/` 与 `checkpoint/pi-response/pi-response-full-20260920-1600-seed342-v2/`。
  - 运行产物：`output/error-source/esrun-20260922-error-source/`、`output/identifiability/` 与 `logs/error-source/`。

## 更新要求

- 新增或删除模块、改变目录职责、完成关键代码阶段或确认新的设计决策时，更新对应方法状态。
- 状态记录只反映代码生成进度，不替代 `docs/TASKS.md` 的 Git 任务登记。
