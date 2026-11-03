<!-- 摘要：记录本工作区各方法代码实现阶段、目录职责和关键可执行能力；新增 GReF 门控响应能量定位独立路径并通过契约及合成链路验证，补充无模型依赖的真实候选响应距离检验与预测响应案例对照、完整训练批次梯度复核和候选拓扑单调性图件工具及统一电气尺度的距离分析工具；Method-A2 已迁入 S0 数据生成模块与特权响应 predictor 模型模块，并新增判据一共享状态可恢复性 Probe 脚本与测试，支持线性与浅层非线性两种 probe 家族，且已完成两种口径的 1600 事件全量运行；数据集、模型权重与其余实验脚本尚未迁入。 -->

# 代码生成状态

## 总体状态

- 项目状态：本工作区由 `method_a` 仓库迁移得到，已完成治理架构、Method-A2 文档、S0 数据生成模块与模型模块迁移，并新增判据一共享状态可恢复性 Probe 脚本与测试，已完成 1600 事件全量运行；数据集与模型权重仍从 `method_a` 仓库读取。
- 代码根目录：`code/`。
- 方法目录规范：每个方法使用独立的 `code/method-<name>/` 目录，并包含同构的 `src/`、`tests/`、`scripts/`、`data/`、`checkpoint/`、`output/` 和 `logs/` 目录。
- 代码组织规范：以 `code/CODE_CONVENTIONS.md` 为唯一正文来源。

## 方法状态

### method-a2

- 同事件预测响应图件：`scripts/plot_predicted_response_cases.py` 匹配 Oracle 下降九宫格的三个案例，保留真实响应参照，冻结推理后恢复物理标幺尺度；保存 114 个候选配对、预测响应、完整及主体放大九宫格，并通过 Oracle 复算、测试预测一致性及解析尺度核验。产物位于 `output/gref/paired-oracle-prediction-20261103/`。

- 真实响应距离检验：`scripts/analyze_oracle_distance.py` 审计同族真实响应库与观测一致性，恢复电压标幺尺度，分析两组全部 3040 个故障事件及历史测试子集的距离结构；`scripts/plot_oracle_detail.py` 提供主体候选放大图。四项新增解析测试与既有十三项相关测试通过，最终统计、差异矩阵、全部候选数值与图件位于 `output/oracle-distance/ieee37-20261103/final/`。

- GReF 电气距离工具：`scripts/export_ieee37_electrical.py` 从独立算例副本导出实际相矩阵、绕组及端口导纳；`scripts/analyze_gref_electrical_distance.py` 核验 IEEE 官方长度并统一 1 MVA 标幺尺度，比较固定能量与重构误差的跳数、长度及阻抗距离关系；六项新增解析测试与七项拓扑测试通过，正式图件、参数及全部候选数值位于 `output/gref/electrical-distance-20261103/final/`。

- GReF 拓扑检查工具：`scripts/analyze_gref_topology.py` 检查完整测试集的候选能量与重构误差相对故障点跳数的单调性、平台、下降和分支差异，按实际类别选择至多三个案例作图；七项解析测试通过，最终图表与全部候选数值位于 `output/gref/topology-analysis-20261103-final/`。

- GReF 梯度复核工具：`scripts/analyze_gref_gradients.py` 复用既有梯度统计函数，对两组最佳模型的完整训练批次分别关闭与启用 Dropout 进行加权梯度及响应误差分解，验证模型状态不变；产物位于 `output/gref/gradient-analysis-20261103/`，分析见 `docs/project/reports/Method-A2-GReF-IEEE37损失梯度与重构退化分析.md`。

- GReF 新增路径：`src/gref/` 提供纯响应距离评分、真实反事实间隔门控、Top-1 交叉熵与全候选 Smooth L1 重构；复用 Node Marker 响应骨干，移除旧可训练评分头，不修改历史训练入口。
- GReF 训练入口：`scripts/train_gref.py`，按最终分组划分拟合逐节点通道归一化，显式要求正能量容差、来源与温度，保存可复算预测和实际执行的置换、响应屏蔽审计，拒绝覆盖已有输出。
- GReF 验证：新增 `tests/test_gref.py` 共 19 项，与相关回归测试合计 49 项通过；命令行两轮合成冒烟产物位于 `output/gref/implementation-smoke-20261103-6bc8cd58/`。
- GReF 当前边界：已完成 IEEE37 数值容差标定、验证集温度预选及种子 342 首轮训练测试；尚未完成多种子验证，当前配置存在明显重构退化。设计见 `docs/project/plans/Method-A2-GReF门控响应能量定位设计与实现.md`，结果见 `docs/project/reports/Method-A2-GReF-IEEE37首轮训练测试报告.md`。

- 状态：可运行判据一实验；其余实验脚本、数据集与权重未迁入。
- 代码目录：`code/method-a2/`。
- 已迁入的数据生成模块：`src/data_generation/` 含包初始化文件与 5 个模块。
  - 仿真后端两个：`opendss_sim.py` 提供基于 OpenDSS COM 接口的 `FaultSimulator`、故障规格枚举与故障命令构建；`mock_sim.py` 提供不依赖 OpenDSS 的确定性 `MockFaultSimulator`，与前者保持兼容的最小接口，供 smoke 与契约测试使用。
  - 数据集生成器一个：`dataset_builder.py` 生成 S0 全候选签名数据集，含 `build_dataset`、`generate_signature_bank` 与 `load_dataset`。
  - 支撑工具两个：`waveform.py` 由 OpenDSS 相量锚点合成动态电压相量窗口；`topology.py` 构建候选边、拓扑掩码与节点观测掩码。
- 已迁入的模型模块：`src/model/` 含包初始化文件与 3 个模块。`privileged_response_predictor.py` 提供 `CandidateConditionedEncoder`、`PairResponseHead` 与 `PrivilegedResponseSystem`；`gnn.py` 提供 `TopologyGNN`；`temporal.py` 提供 `TemporalEncoder`。实测 `teacher.pt` 的全部 87 项状态字典条目可无缺失、无冗余载入，可训练参数 194372。
- 新增的实验脚本：`scripts/run_error_source_criterion1_probe.py` 实现判据一共享状态可恢复性 Probe，冻结既有编码器后在其表示与观测池化特征上拟合 probe，含四类负对照、块内一致性、位置可恢复性核验、逐维判定与 9 项产物写出，并对长任务做逐次拟合增量落盘。以 `--probe-kinds linear|mlp` 选择 probe 家族：`linear` 为岭回归与多类逻辑回归，`mlp` 为单隐藏层浅层非线性（宽度 × 权重衰减在验证划分上选择，选定配置用 3 个随机种子重训并聚合）。另提供 `--block-limit` 小样本冒烟开关，按 block 截断以保持划分不重叠与块内共享工况两条性质。
- 新增的测试：`tests/test_error_source_criterion1_probe.py`，34 项，覆盖度量定义、按 block 聚合与子采样、训练侧标准化、正则与宽度选参口径、判定分支、报告与 CSV 产物渲染、两种 probe 家族的分派与聚合规则、以及环境差异下的求解器选择。
- 已完成的运行：
  - `output/error-source-v2/c1run-20260923T083606Z-seed342/criterion-1/`：线性口径，1600 事件全量，判定 S1 成立（0.8033）、S2 未定（0.1211）。
  - `output/error-source-v2/c1nl-20260923-seed342/criterion-1/`：浅层非线性口径，1600 事件全量，判定 S1 成立（0.6665）、S2 未定（0.1657）。
  - `output/error-source-v2/c1smoke-20260923-seed342/` 与 `c1nl-smoke-20260923-seed342/`：每划分 8 个 block 的冒烟，产物中已标记 `smoke_run`。
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
