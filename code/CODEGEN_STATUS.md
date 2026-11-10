<!-- 摘要：记录本工作区各方法代码实现阶段、目录职责和关键可执行能力；新增 GReF 门控响应能量定位独立路径并通过契约及合成链路验证，补充无模型依赖的真实候选响应距离检验与预测响应案例对照、完整训练批次梯度复核和候选拓扑单调性图件工具及统一电气尺度的距离分析工具；新增 GReF 三项损失（能量／排序／图平滑）训练模式与三项梯度分析入口；新增 GReF 可选训练期无向边删除消融与 S1-EN 推理期边删除评估入口；Method-A2 已迁入 S0 数据生成模块与特权响应 predictor 模型模块，新增判据一共享状态可恢复性 Probe 脚本与测试（支持线性与浅层非线性两种 probe 家族并完成两种口径的 1600 事件全量运行），并新增 `main.py` 统一运行入口与 `tests/test_run_entry.py`，使数据生成与训练可由文件顶部配置直接运行；数据集、模型权重与其余实验脚本尚未迁入。 -->

# 代码生成状态

<!-- 摘要补充：新增 RaReF 独立服务器包构建与训练、验证、冻结测试入口及契约验证，支持两套 IEEE37 电阻数据的独立选择。 -->

## RaReF 服务器交付工具（2026-11-10）

- `method-a2/scripts/build_raref_server_bundle.py` 从指定来源工作区复制 RaReF 最小依赖闭包、已登记 IEEE37 A（0.01–20 Ω）与 B（10–100 Ω）两套完整数据及配置，生成含文件摘要的独立目录、压缩归档与归档摘要，拒绝任一数据元信息不一致和覆盖已有交付。
- `method-a2/scripts/run_raref_server.py` 提供环境检查、合成冒烟、训练、验证、冻结测试与完整流程入口；复用既有模型和评价实现，训练后仅用验证集选模及标定概率温度，测试保留独占启动标记。
- 运行入口通过 `--dataset` 选择数据，验证与测试固定使用训练记录的数据目录。契约验证入口为 `method-a2/tests/test_raref_server_bundle.py`；双数据集交付位于来源方法的 `output/raref-server-20261110-ab/`，原单数据集包保留。安装和验证边界见 `docs/project/plans/Method-A2-RaReF服务器运行包说明.md`。

## 总体状态

- 项目状态：本工作区由 `method_a` 仓库迁移得到，已完成治理架构、Method-A2 文档、S0 数据生成模块与模型模块迁移，并新增判据一共享状态可恢复性 Probe 脚本与测试，已完成 1600 事件全量运行；数据集与模型权重仍从 `method_a` 仓库读取。
- 代码根目录：`code/`。
- 方法目录规范：每个方法使用独立的 `code/method-<name>/` 目录，并包含同构的 `src/`、`tests/`、`scripts/`、`data/`、`checkpoint/`、`output/` 和 `logs/` 目录。
- 代码组织规范：以 `code/CODE_CONVENTIONS.md` 为唯一正文来源。

## 方法状态

### method-a2

- 同事件预测响应图件：`scripts/plot_predicted_response_cases.py` 匹配 Oracle 下降九宫格的三个案例，保留真实响应参照，冻结推理后恢复物理标幺尺度；保存 114 个候选配对、预测响应、完整及主体放大九宫格，并通过 Oracle 复算、测试预测一致性及解析尺度核验。产物位于 `output/gref/paired-oracle-prediction-20261103/`。
- 三项损失模型九宫格：`scripts/plot_predicted_response_cases_three_term.py` 是上述脚本的可参数化副本，不修改原脚本；通过 `--a-run-dir`／`--b-run-dir` 指定 A/B 三项损失模型，复用原案例、尺度与分类函数，输出 `output/gref/paired-oracle-prediction-three-term-20261104/` 的完整、放大九宫格、114 个候选数值与校验信息。结果见 `docs/project/reports/Method-A2-三项损失模型同事件预测响应九宫格对照.md`。
- IEEE37 候选索引拓扑图：`scripts/plot_ieee37_topology_index.py` 读取官方母线坐标、真实边表与候选顺序，重绘拓扑并把母线编号替换为候选索引 0–37，输出 PNG/SVG、`index_bus_mapping.csv` 与说明至 `output/visualizations/ieee37_topology_index/`；顶部 799/799r 与 775 在作图时略微拉开，原参考位图不被修改。
- 同案例 Oracle 响应图：`scripts/plot_oracle_response_cases.py` 读取三项损失九宫格 `analysis.json` 中保存的三个案例与 `oracle_values`，按同一 detail/full 格式重绘真实响应差异，红★位于 (0,0)，输出 `oracle-response-{detail,full}.png/.svg`、逐候选 CSV 与分析 JSON 至 `output/gref/paired-oracle-prediction-three-term-20261104/`，并追加到该目录 `gallery.html`。
- 三项损失曲线：`scripts/plot_three_term_loss_curves.py` 读取 A/B 三项损失运行的 `train_log.csv`，按 λ=(1,0.1,0.1) 折算能量、排序、平滑与总损失，绘制训练实线与验证虚线的双面板对数曲线，标注最佳轮次；产物位于 `output/gref/three-term-loss-curves-20261104/`。

- 真实响应距离检验：`scripts/analyze_oracle_distance.py` 审计同族真实响应库与观测一致性，恢复电压标幺尺度，分析两组全部 3040 个故障事件及历史测试子集的距离结构；`scripts/plot_oracle_detail.py` 提供主体候选放大图。四项新增解析测试与既有十三项相关测试通过，最终统计、差异矩阵、全部候选数值与图件位于 `output/oracle-distance/ieee37-20261103/final/`。

- GReF 电气距离工具：`scripts/export_ieee37_electrical.py` 从独立算例副本导出实际相矩阵、绕组及端口导纳；`scripts/analyze_gref_electrical_distance.py` 核验 IEEE 官方长度并统一 1 MVA 标幺尺度，比较固定能量与重构误差的跳数、长度及阻抗距离关系；六项新增解析测试与七项拓扑测试通过，正式图件、参数及全部候选数值位于 `output/gref/electrical-distance-20261103/final/`。

- GReF 拓扑检查工具：`scripts/analyze_gref_topology.py` 检查完整测试集的候选能量与重构误差相对故障点跳数的单调性、平台、下降和分支差异，按实际类别选择至多三个案例作图；七项解析测试通过，最终图表与全部候选数值位于 `output/gref/topology-analysis-20261103-final/`。

- GReF 梯度复核工具：`scripts/analyze_gref_gradients.py` 复用既有梯度统计函数，对两组最佳模型的完整训练批次分别关闭与启用 Dropout 进行加权梯度及响应误差分解，验证模型状态不变；产物位于 `output/gref/gradient-analysis-20261103/`，分析见 `docs/project/reports/Method-A2-GReF-IEEE37损失梯度与重构退化分析.md`。

- GReF 新增路径：`src/gref/` 提供纯响应距离评分、真实反事实间隔门控、Top-1 交叉熵与全候选 Smooth L1 重构；复用 Node Marker 响应骨干，移除旧可训练评分头，不修改历史训练入口。
- GReF 三项损失路径：`src/gref/losses.py` 新增 `reconstruction_energy`、`list_energy_loss`、`unique_undirected_edges`、`graph_smoothness_terms` 与 `three_term_losses`，实现“逐元素均方重构能量＋部署能量软列表 KL＋误差场图拉普拉斯平滑”；`gref_losses` 保持原有门控交叉熵路径不变。训练入口新增 `--loss-mode three-term`、三项权重、排序温度（缺省由训练集真实能量间隔标定）、分段检查点与最佳轮优化器状态保存。梯度分析入口为 `scripts/analyze_gref_three_term_gradients.py`，输出三项加权梯度的范数、份额、两两余弦、合成投影与误差图频分解。
- GReF 三项损失验证：新增 `tests/test_gref_three_term.py` 共 13 项，覆盖手算重构能量、列表 KL 的零值与反转、掩码与正常候选隔离、平滑零空间与节点置换、三项组合与梯度路径、入口参数与端到端冒烟；与既有 GReF 相关测试合计通过。设计见 `docs/project/plans/Method-A2-GReF三项损失改造计划.md`。
- GReF 三项损失正式运行：IEEE37 A/B 种子 342、200 轮已完成，产物位于 `output/gref/three-term-ieee37-a-seed342/` 与 `.../three-term-ieee37-b-seed342/`，36 批次三项梯度分析位于 `output/gref/three-term-gradient-a-20261104/` 与 `.../three-term-gradient-b-20261104/`；结果与机制解释见 `docs/project/reports/Method-A2-GReF三项损失实验报告.md`。
- GReF 训练入口：`scripts/train_gref.py`，按最终分组划分拟合逐节点通道归一化，显式要求正能量容差、来源与温度，保存可复算预测和实际执行的置换、响应屏蔽审计，拒绝覆盖已有输出。
- GReF 验证：新增 `tests/test_gref.py` 共 19 项，与相关回归测试合计 49 项通过；命令行两轮合成冒烟产物位于 `output/gref/implementation-smoke-20261103-6bc8cd58/`。
- GReF 当前边界：已完成 IEEE37 数值容差标定、验证集温度预选及种子 342 首轮训练测试；尚未完成多种子验证，当前配置存在明显重构退化。设计见 `docs/project/plans/Method-A2-GReF门控响应能量定位设计与实现.md`，结果见 `docs/project/reports/Method-A2-GReF-IEEE37首轮训练测试报告.md`。
- GReF 边删除评估与可选消融：`src/gref/data.py` 新增 `build_undirected_edge_groups` 与 `apply_edge_drop`，按无向边成对清除掩码；`src/gref/training.py` 新增 `--edge-drop-prob` 与 `--edge-drop-max`，默认值为 0，正常训练不启用。正式 S1-EN 协议只使用正常 S0 数据训练，边删除仅在推理阶段施加；训练期边删除仅作为可选消融，不进入正式主结果。新增 `scripts/eval_gref_edge_drop.py` 在 IEEE37 测试集上评估干净、逐条单边删除与随机 1/2/3 边删除；`tests/test_gref_edge_drop.py` 共 5 项通过。正式 S1-EN 结果使用干净基线 `output/gref/ieee37-a-seed342-20261103/` 的 `s1en-eval/`：干净 Top-1 为 97.81%，单边删除均值 96.60%，随机 1/2/3 边删除均值 96.16%/95.57%/94.74%。显式开启训练期删除的额外消融产物位于 `output/gref/s1en-ieee37-seed342-20261105/`，其对应值为 97.37%、97.38%、97.39%/97.37%/97.37%，仅作对照。
- S2 节点观测评估入口：`scripts/eval_s2_nodes.py` 在冻结模型上评估部分节点观测。推理前先核对数据形状与元数据、标准化、模型配置与状态字典、测试划分、候选轴与节点掩码，检查失败即停止且不创建输出目录；再计算完整观测基线并与运行目录 `metrics.json` 核对复现性；随后按 `seed+1009*i` 生成或通过 `--node-masks` 读取同一份 `node_masks.npz`，逐组掩码推理。输出 `s2_node_metrics.json`（含逐掩码指标、逐事件名次与 signed margin、分层汇总、真实节点观测／未观测子集的完整指标分解与数据指纹）、`s2_node_metrics.md`、`s2_input_checks.json` 与 `node_masks.npz`。正式运行与结果见 `docs/project/reports/Method-A2-S2-部分节点观测GReF与三项损失冻结推理报告.md`。
- 区域概率损失：`src/gref/region.py` 提供候选拓扑跳数距离、列随机区域核、区域概率、最小区域、区域监督目标与区域指标；`src/gref/losses.py` 新增 `deployment_probabilities` 与 `region_term_losses`，复用原有重构能量与排序 KL 并保持其定义不变；`src/gref/data.py` 新增 `sample_node_masks` 覆盖完整观测与真实节点被观测/未观测两类缺测；`src/gref/training.py` 新增 `--loss-mode region`、区域核与节点掩码参数，训练日志记录区域项与四类掩码计数。新增 `scripts/eval_region_term.py` 在冻结检查点上标定单节点概率温度，输出完整观测与五级共享掩码的排名、概率与区域指标并写出 `region_metrics.json`。新增 `tests/test_gref_region.py` 12 项通过。第一阶段结果见 `docs/project/reports/Method-A2-区域概率损失第一阶段实验报告.md`。
- 区域概率损失第二阶段脚手架：`src/gref/training.py` 新增 `--skip-test-evaluation`，扫描运行只写验证集产物、不评价测试集；`scripts/evaluate_gref.py` 在冻结检查点上执行一次标准测试评价并写出 `metrics.json`、`predictions.npz`、`shortcut_audit.json` 与 `dataset_audit.json`；`scripts/analyze_gref_region_gradients.py` 测量三项加权梯度、余弦与合成投影并反推权重目标；`scripts/summarize_region_sweep.py` 汇总验证集区域指标、约束判定与选模；`scripts/eval_region_term.py` 新增 `--split validation` 与独立验证掩码种子。设计见 `docs/project/plans/Method-A2-区域概率损失第二阶段权重扫描补遗.md`。
- 区域概率损失第二阶段结果：10 组权重扫描完成，8 组通过约束，选定 (λ_E,λ_R,λ_region)=(1.0,0.141,0.194)，实测投影份额 4%/49%/47%；测试完整 Top-1/3/5 为 92.54%/99.12%/100%，部分观测 Top-1 87.68%/81.23%/74.25%/66.62%/53.86%，未观测子集覆盖率 97.78%–94.13%、单节点 Top-1 70.00%–27.17%。结果见 `docs/project/reports/Method-A2-区域概率损失第二阶段权重扫描实验报告.md`，汇总见 `output/gref/region-sweep-summary-20261110/weight_scan_summary.json`。
- 区域概率损失多种子复核：`scripts/summarize_region_multiseed.py` 汇总种子 342/343/344 的三配置配对验证结果。区域配置相对原始 A tau1x 的未观测覆盖率 +19.8±2.6 个百分点、未观测 Top-1 +51.6±4.4 个百分点；第二阶段相对第一阶段完整/观测/总体 Top-1 分别 +2.63/+7.36/+4.99 个百分点，未观测 Top-1 与覆盖率差异落在种子波动内。结果见 `docs/project/reports/Method-A2-区域概率损失多种子配对复核报告.md`，汇总见 `output/gref/region-multiseed-summary-20261110/multi_seed_summary.json`。
- 区域概率损失未观测定向与校准：`scripts/summarize_region_exp2.py` 按覆盖率、区域大小与观测子集约束汇总 β_miss×σ 六组候选并选模，`scripts/analyze_region_calibration.py` 在已保存验证区域概率上计算 α∈{0.05,0.1,0.2} 的经验覆盖率与区域大小。仅 (β_obs,β_miss,σ)=(0.05,0.4,1.0) 通过约束，未观测 Top-1 相对对照 +2.12 个百分点；σ=0.5 区域约 3.5 个节点但覆盖率 0.84–0.88。结果见 `docs/project/reports/Method-A2-区域概率损失未观测定向与置信度校准实验报告.md`，汇总见 `output/gref/region-exp2-summary-20261110/`。

- 状态：可运行数据生成、训练与判据一实验；其余实验脚本、数据集与权重未迁入。
- 代码目录：`code/method-a2/`。
- 已迁入的数据生成模块：`src/data_generation/` 含包初始化文件与 5 个模块。
  - 仿真后端两个：`opendss_sim.py` 提供基于 OpenDSS COM 接口的 `FaultSimulator`、故障规格枚举与故障命令构建；`mock_sim.py` 提供不依赖 OpenDSS 的确定性 `MockFaultSimulator`，与前者保持兼容的最小接口，供 smoke 与契约测试使用。
  - 数据集生成器一个：`dataset_builder.py` 生成 S0 全候选签名数据集，含 `build_dataset`、`generate_signature_bank` 与 `load_dataset`。
  - 支撑工具两个：`waveform.py` 由 OpenDSS 相量锚点合成动态电压相量窗口；`topology.py` 构建候选边、拓扑掩码与节点观测掩码。
- 已迁入的模型模块：`src/model/` 含包初始化文件与 3 个模块。`privileged_response_predictor.py` 提供 `CandidateConditionedEncoder`、`PairResponseHead` 与 `PrivilegedResponseSystem`；`gnn.py` 提供 `TopologyGNN`；`temporal.py` 提供 `TemporalEncoder`。实测 `teacher.pt` 的全部 87 项状态字典条目可无缺失、无冗余载入，可训练参数 194372。
- 新增的实验脚本：`scripts/run_error_source_criterion1_probe.py` 实现判据一共享状态可恢复性 Probe，冻结既有编码器后在其表示与观测池化特征上拟合 probe，含四类负对照、块内一致性、位置可恢复性核验、逐维判定与 9 项产物写出，并对长任务做逐次拟合增量落盘。以 `--probe-kinds linear|mlp` 选择 probe 家族：`linear` 为岭回归与多类逻辑回归，`mlp` 为单隐藏层浅层非线性（宽度 × 权重衰减在验证划分上选择，选定配置用 3 个随机种子重训并聚合）。另提供 `--block-limit` 小样本冒烟开关，按 block 截断以保持划分不重叠与块内共享工况两条性质。
- 新增的测试：`tests/test_error_source_criterion1_probe.py`，34 项，覆盖度量定义、按 block 聚合与子采样、训练侧标准化、正则与宽度选参口径、判定分支、报告与 CSV 产物渲染、两种 probe 家族的分派与聚合规则、以及环境差异下的求解器选择。
- 新增的运行入口：`main.py` 把数据生成与训练的参数集中到文件顶部的 `CommonConfig`、`DataConfig` 与 `TrainConfig`，由 `MODE` 选择 `data`、`train` 或 `all`，先打印本次实际使用的参数，再调用 `scripts/generate_dataset.py::main` 与 `scripts/train_marker_mvp.py::run`。入口自带参数解析器（默认值取配置文件，字段名与脚本命令行参数逐字一致），透传的命令行参数可临时覆盖配置，`all` 模式下按阶段过滤透传参数并强制两阶段使用不同的运行标识与产物目录。为支持自定义进度显示，两个脚本的执行函数各新增一个可选 `progress` 参数，默认行为不变。
- 数据生成的子进程隔离：以 OpenDSS COM 求解的进程无法正常退出（实测为访问冲突 0xC0000005 或无限挂起，且显式 `ClearAll` 后释放引用、`pythoncom.CoUninitialize()`、`os._exit(0)`、保留引用到进程结束均无效；本机 OpenDSS COM 接口不提供 `CloseDSS`）。因此在 `data` 与 `all` 模式下由父进程启动 `main.py --child-data` 子进程执行数据生成，父进程只轮询 `logs/<run-id>/progress.latest.json` 显示进度，并以子进程写入的 `end` 记录判定成功，随后强制终止仍挂起的子进程并正常退出；`error` 记录判为失败、缺结束记录判为未完成、超过两小时判为超时。`train` 模式仍在父进程内直接执行。
- 新增的测试：`tests/test_run_entry.py`，13 项，覆盖入口解析器与两个阶段脚本参数定义的逐项比对（参数名、取值集合与多值形式）、默认值来源、配置取值与命令行覆盖的优先级、相对路径解析、非法参数与未声明配置字段的失败行为、路径与运行标识生成，以及进度显示格式与进度写入器的落盘行为。
- 新增的测试：`tests/test_data_isolation.py`，11 项，覆盖子进程命令行拼装与可选参数省略、相对路径按当前工作目录解析、完成与失败记录的判定、残缺行容错、进度快照容错，以及子进程挂起、写入失败记录、超时、无结束记录退出四类父进程判定分支。
- 已完成的运行：
  - `data/s0-lg-run3`：单相接地故障、`samples_per_bus=10` 的 S0 数据集，训练侧已在其上完成阶段零至阶段三的验证运行。
  - `output/error-source-v2/c1run-20260923T083606Z-seed342/criterion-1/`：线性口径，1600 事件全量，判定 S1 成立（0.8033）、S2 未定（0.1211）。
  - `output/error-source-v2/c1nl-20260923-seed342/criterion-1/`：浅层非线性口径，1600 事件全量，判定 S1 成立（0.6665）、S2 未定（0.1657）。
  - `output/error-source-v2/c1smoke-20260923-seed342/` 与 `c1nl-smoke-20260923-seed342/`：每划分 8 个 block 的冒烟，产物中已标记 `smoke_run`。
- 运行环境注意事项（已在产物 `config.json` 中固化）：本机 `sklearn 1.0.2` 与 `scipy 1.13.1` 不兼容，`Ridge` 默认 `auto`/`cholesky` 求解器会抛 `TypeError`，改用 `lsqr`；多类逻辑回归改用 `newton-cg`，同一 16 类问题上较 `lbfgs` 快约 119 倍。两项选择均不改变 multinomial 与正则网格口径。
- 目录状态：`data/` 与 `output/` 下已有运行产物，`checkpoint/` 仍为空目录并保留 `.gitkeep` 占位；`main.py` 与 `README.md` 已创建，`requirements.txt` 尚未创建。
- 运行环境注意事项（数据生成）：以 OpenDSS COM 生成数据时，进程在数据与进度记录写完并打印“运行完成”之后无法正常退出，实测表现为 Windows 访问冲突（退出码 `-1073741819`）或无限挂起；直接运行 `scripts/generate_dataset.py` 时同样出现，属既有引擎退出行为。入口已按上文以子进程隔离处理，成功判定以产物文件与进度记录的 `end` 为准。
- 待迁入内容（来源为 `method_a` 仓库 `code/method-a1/`）：
  - 未迁入的数据集生成器 4 个：`e0_builder.py`、`e0_cov_builder.py`、`e4_a0_builder.py`、`paired_response_builder.py`。四者均依赖已迁入的仿真后端与支撑工具，迁入后即可运行。
  - 未迁入的其他源代码：`src/identifiability.py`、`src/pi_response_dataset.py`、`src/pi_response_losses.py`、`src/pi_response_trainer.py`、`src/pi_response_heartbeat.py`、`src/losses.py`。
  - 实验脚本 10 个：`probe_a2_structure.py`、`probe_a2_structure_final.py`、`probe_a2_error_source.py`、`probe_a2_predictor_error.py`、`probe_a2_margin_floor.py`、`run_error_source_experiments.py`、`audit_error_source_results.py`、`run_identifiability_audit.py`、`run_identifiability_nn.py`、`run_identifiability_baseline.py`。
  - 测试 2 个：`test_error_source_experiments.py`、`test_identifiability.py`。
  - 数据集与权重：`data/pi-response/paired-v1-ieee13-1600-seed342`、`data/pi-response/paired-v1-ieee13-640-seed342`、`data/pi-response-smoke/`、`data/pi-response-probe/` 与 `checkpoint/pi-response/pi-response-full-20260920-1600-seed342-v2/`。判据一实验目前在运行期从 `method_a` 仓库读取并在 `config.json` 中固化来源，尚未迁入本工作区。

## 更新要求

- 新增或删除模块、改变目录职责、完成关键代码阶段或确认新的设计决策时，更新对应方法状态。
- 状态记录只反映代码生成进度，不替代 `docs/TASKS.md` 的 Git 任务登记。
