<!-- 摘要：归档候选评分运行 candidate-score-full-lg-run3-seed342-ckpt 的 S0 训练与测试、S1 推理拓扑边屏蔽和 S2 部分节点观测测试。记录 2026-10-08 数据生成、2026-10-09 训练及测试产物时间、s0-lg-run3 数据集、模型与代码入口、实际候选评分口径、参数、运行流程、结果和复核边界；文档于 2026-10-13 整理。S0 测试 Top-1/3/5 为 66.67%/92.71%/98.96%；S1 枚举屏蔽 1 至 3 条无向边，S2 每个观测规模使用 10 组掩码。S1/S2 原始执行脚本与独立运行时间日志未归档，明确区分已有结果、代码接口与重建流程。 -->

# Method-A2 候选评分 S0、S1、S2 实验记录

## 1. 实验定位与结论

本记录以 `code/method-a2/output/candidate-score-full-lg-run3-seed342-ckpt/` 中的配置、指标、模型和扰动测试结果为依据，归档本次成功运行。运行标识为 `candidate-score-full-lg-run3-seed342-ckpt`，数据集为 `code/method-a2/data/s0-lg-run3/`。

S0 测试集 96 个故障事件中，Top-1 命中 64 个、Top-3 命中 89 个、Top-5 命中 95 个，分别为 **66.67%、92.71%、98.96%**。该结果记录了直接候选评分分支在本数据划分上的定位能力，不构成多种子稳定性或跨拓扑有效性结论。S1 在输入图中屏蔽 1 至 3 条无向边后仍保留较高的 Top-3；S2 的性能随观测节点减少明显下降。

本次采用的定位值是模型的 **`candidate_score` 直接评分头输出**，分数越低越优；预测反事实响应用于辅助监督和响应诊断。不得把本次定位结果解释为“事实观测与预测反事实响应的距离排序”。本次训练亦不同于早期 Node Marker MVP 的族一致性与列表级损失方案。

相关历史背景见 [Node Marker MVP 执行报告](Method-A2-Node-Marker-MVP执行报告.md) 与 [准确率提升验证执行报告](Method-A2-排除既有方法后准确率提升验证执行报告.md)。本记录是后续候选评分运行的独立归档，不替换历史报告的否定结论。

## 2. 实验日期与证据来源

以下时间统一使用 Asia/Singapore（UTC+08:00），原始进度文件保存 UTC 时间。

- 数据生成成功批次：2026-10-08 20:29:51 至 20:50:43；日志记录总耗时 `1252.442` 秒，元数据记录仿真累计耗时 `1249.8499` 秒。
- 同名数据日志先前存在 20:10:11 开始、20:29:48 报错的批次，进程号为 `24040`；最终成功批次进程号为 `28556`。日期和耗时均以最终批次为准，不将重试前后记录拼为一次成功运行。
- 候选评分训练：2026-10-09 12:19:28 至 12:24:15；训练产物写出完成时记录总耗时 `290.386` 秒。阶段开始时间与总耗时计时起点略有不同，不以时间戳相减替代原始耗时记录。
- S1 完整 1/2/3 条边结果文件最后修改时间：2026-10-09 12:45:53。
- S2 12/8 个节点结果文件最后修改时间：2026-10-09 12:59:15。
- S2 15/13 个节点结果文件最后修改时间：2026-10-09 13:07:29。
- 文档整理日期：2026-10-13。

数据与训练时间分别来自 `logs/data-s0-lg-run3/progress.jsonl` 和 `logs/candidate-score-full-lg-run3-seed342-ckpt/progress.jsonl`。S1/S2 的 JSON 未保存独立起止时间，以上时间仅为本地文件最后修改时间，不能视为精确的测试开始、结束或耗时证据。

本次整理时仓库 HEAD 为 `652abc286b32aa144541bf0ebadf97c8f93edfce`，它仅表示文档核查时的代码状态。运行配置未固化实验时 commit 或源码哈希，且整理时存在运行入口等未提交变更，因此不将该 HEAD 宣称为实验时的确定代码版本。

## 3. 数据集及划分

唯一基础数据集是 `code/method-a2/data/s0-lg-run3/`。S1/S2 均复用此数据，不另行生成物理网络变化后的数据集。

- 算例：`ieee13`；实际图包含 16 个母线节点、15 条无向边、30 条有向边。
- 候选轴：16 个故障位置加 1 个正常候选，共 17 个；定位排名仅使用前 16 个故障候选。
- 样本规模：1280 条，其中 640 条故障、640 条正常。
- 故障族：40 个，每族含 16 个事实故障位置；训练使用 28 个族，验证和测试各使用 6 个族。
- 训练集 1088 条，其中 448 条故障及 640 条正常；验证集和测试集分别为 96 条故障事件。
- 单相接地故障，故障阻抗范围 `0.1～100 Ω`，负荷倍率采样范围 `0.8～1.2`，随机种子 `342`。
- 窗口采样率 `200 Hz`，故障前 1 周波、故障后 2 周波，共 12 个时间步；通道为 `Re_A, Im_A, Re_B, Im_B, Re_C, Im_C`。
- 窗口由 OpenDSS 故障前后相量锚点构造，包含确定性的过渡过程；并非 OpenDSS 逐时间步暂态波形求解。
- 候选相关随机扰动关闭；同族共享事件参数，事实观测取该族真实故障候选签名。
- 标准化统计量仅由训练样本的有效候选签名计算，统一保存为逐通道 `feature_scaler.npz`。

实际故障族划分为：

- 训练：`0,2,4,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,29,30,31,32,35,36,37,39`。
- 验证：`5,14,25,27,33,34`。
- 测试：`1,3,24,26,28,38`。

数据文件包括 `X_obs.npy`、`X_full.npy`、`signature_bank.npy`、`group_id.npy`、`mask.npy`、`candidate_mask.npy`、`edge_index.npy`、`edge_attr.npy`、`edge_mask.npy`、`y_detect.npy`、`y_loc.npy`、`y_class.npy`、`y_resist.npy`、三个划分索引、`feature_scaler.npz` 和 `meta.json`。

`X_obs` 形状为 `[1280,16,12,6]`，`signature_bank` 形状为 `[1280,17,16,12,6]`。`dataset_audit.json` 记录三个划分两两故障族交集均为 0，候选覆盖最小值为 17，候选掩码平均覆盖比例为 1.0。

## 4. 代码与训练配置

以下代码路径均相对项目根目录。

- `code/method-a2/scripts/generate_dataset.py`：数据生成参数入口。
- `code/method-a2/src/data_generation/dataset_builder.py`：事件参数采样、反事实签名族生成、分组划分和训练统计量标准化。
- `code/method-a2/src/data_generation/opendss_sim.py`：IEEE 算例加载、实际相别识别和故障前后相量计算。
- `code/method-a2/src/data_generation/waveform.py`：从相量锚点构造六通道窗口。
- `code/method-a2/scripts/train_marker_mvp.py`：训练、验证选模、最终测试、探针与模型保存。
- `code/method-a2/src/model/marker_response_predictor.py`：Node Marker、共享表示、反事实响应头和直接候选评分头。
- `code/method-a2/src/model/temporal.py` 与 `gnn.py`：时序编码与图消息传递。
- `code/method-a2/src/model/losses.py` 与 `src/training/marker_training.py`：候选交叉熵、成对排序和响应辅助损失。
- `code/method-a2/src/training/evaluation.py`：分组划分、稳定排序 Top-K、Oracle 和探针工具。
- `code/method-a2/src/progress.py` 与 `scripts/report_progress.py`：运行进度落盘及读取。

训练配置来自运行目录的 `config.json`：

- CPU，随机种子 `342`，Adam，学习率 `0.001`。
- 200 轮，批量大小 64，`max_batches=0`（不截断批次）。
- `pooling=mean`，`temporal_mode=single`，时序残差块 3 层、共享图编码 2 层。
- 派生通道和不确定性分支均关闭，可训练参数 39209 个。
- `lambda_rank=0.1`、`lambda_response=0.1`、`rank_margin=0.1`、`rank_temperature=1.0`。
- 沿用数据内置训练/验证/测试划分，不重新划分。
- 按验证 Top-1 最大选模；相同 Top-1 时选验证响应损失更小者。产物记录 `best_epoch=186`，保留代码中的原始轮次编号。

总损失为 `L_loc + 0.1 L_rank + 0.1 L_family`。本次未启用不确定性，`L_family` 实际调用 `masked_response_loss`，为掩码加权 Smooth L1 响应损失，而非均方差族损失。位置损失使用负候选分数作为交叉熵 logits，排序项使用真实候选与错误候选的 softplus 成对损失。正常样本参与响应监督，但不参与故障位置分类和最终故障排名。

最终权重保存为运行目录中的 `model.pt`，含 `model_state_dict`、模型配置和 `best_epoch`。SHA-256 为 `64928933a089515c7e3bee92bba329c909d467a8c7c5c060c32f8725e8fd504c`。

当前另有 `code/method-a2/main.py` 和 `README.md` 提供统一配置入口；它们是整理时存在的运行设施，不能据此推定 2026-10-09 的原始启动方式。历史训练参数以 `config.json` 为准。

## 5. S0 完整观测测试

S0 使用完整节点观测、完整输入拓扑，模型以 `candidate_score` 升序排列 16 个故障候选；相同分数按 `np.argsort(..., kind="stable")` 排序。

- 验证集：Top-1 `76.04%`，Top-3 `94.79%`，Top-5 `98.96%`，平均名次 `1.45833`。
- 测试集：Top-1 `66.67%`，Top-3 `92.71%`，Top-5 `98.96%`，平均名次 `1.65625`，名次中位数 `1`。
- 测试 signed margin 中位数 `0.2917614`，负间隔比例 `33.33%`；这是直接评分头的间隔，不是物理响应距离间隔。
- 测试损失：响应辅助项 `0.0033592123`，定位项 `0.82199788`，排序项 `0.093877889`，总损失 `0.83172160`。
- 测试 Oracle：Top-1 `96.875%`，Top-3/5 均为 `100%`；Oracle 响应距离间隔中位数 `2.17270875e-5`，精确零间隔事件 6 个。此处引用本运行目录的 `metrics.json`，不混入其他 Oracle 脚本的距离口径。

依据文件为 `metrics.json`、`validation_metrics.json`、`oracle_metrics.json` 和 `predictions.npz`。

## 6. S1 输入拓扑边屏蔽测试

S1 使用同一 S0 模型，保持原始电压观测与物理签名不变，仅在推理时将所选无向边对应的两个有向边屏蔽，不重新训练，不执行新的断线物理仿真。

对 15 条无向边枚举屏蔽 1、2、3 条的全部组合，组合数分别为 15、105、455，总计 575 组。每组在训练、验证、测试的故障子集上计算 Top-K。以下数值为各规模全部组合的等权平均，不是单个最佳组合：

- 屏蔽 1 条边：测试 Top-1/3/5 为 `64.44% / 90.76% / 97.92%`；训练为 `68.32% / 92.25% / 97.99%`，验证为 `70.35% / 93.06% / 98.26%`。
- 屏蔽 2 条边：测试 Top-1/3/5 为 `61.44% / 88.90% / 96.73%`；训练为 `64.89% / 90.66% / 96.84%`，验证为 `65.71% / 90.93% / 97.27%`。
- 屏蔽 3 条边：测试 Top-1/3/5 为 `58.78% / 86.90% / 95.52%`；训练为 `61.68% / 88.88% / 95.77%`，验证为 `61.57% / 88.84% / 96.36%`。

相对 S0，三组测试 Top-1 平均下降约 2.22、5.23、7.88 个百分点。该结果支持有限输入拓扑缺失下的性能描述，不等价于实际电网断线、错误新增边或跨拓扑泛化验证。

完整组合、边编号与母线名称映射见 `s1_topk_report_1_2_3_edges.json`，阅读版见同名 `.md`。其他补充产物为 `s1_edge_mask_sweep.json`（15 条单边测试）和 `s1_perturbation_range_sweep.json`（扰动规模扫描）。

## 7. S2 部分节点观测测试

S2 同样复用 S0 权重与数据，保持拓扑和 16 个故障候选完整，对节点观测施加掩码，不重新训练。每个观测规模采用 10 组掩码，每组固定用于训练、验证与测试，原始数据文件不修改。

- 观测 15/16 个节点：测试 Top-1/3/5 均值 `50.31% / 81.56% / 94.38%`；训练为 `52.01% / 83.66% / 94.89%`，验证为 `53.33% / 81.88% / 95.00%`。
- 观测 13/16 个节点：测试均值 `35.94% / 50.00% / 76.15%`；训练为 `37.77% / 51.05% / 76.70%`，验证为 `36.67% / 51.25% / 75.31%`。
- 观测 12/16 个节点：测试均值 `31.77% / 51.35% / 68.85%`；训练为 `33.26% / 51.05% / 68.01%`，验证为 `32.60% / 53.96% / 69.58%`。
- 观测 8/16 个节点：测试均值 `21.88% / 36.56% / 47.81%`；训练为 `21.34% / 35.96% / 47.05%`，验证为 `23.12% / 37.81% / 48.02%`。

15 节点的 10 组掩码中存在重复屏蔽母线（如 `634` 和 `SourceBus`），因此这里是保存的 10 次测试记录平均，不是 10 个互异掩码的平均，也不是 10 个训练随机种子的平均。13 与 12 节点结果来自不同掩码组，不能把个别 Top-K 差异解释为严格单调规律。

依据文件为 `s2_node_mask_15_13_10groups.json/.md` 和 `s2_node_mask_10groups.json/.md`，其中保存观测母线、屏蔽母线与分集指标。S2 只记录完整观测训练模型遭遇缺测时的性能；其下降不代表缺测条件下重新训练模型的性能上限。

## 8. 运行流程与复核方法

### 8.1 数据生成与训练

流程为：加载 IEEE13 算例 → 采样共享负荷和故障参数 → 生成 17 候选签名族 → 从真实位置签名取得事实观测 → 按族划分 → 训练统计量标准化 → 模型训练 → 验证选模 → 一次最终测试 → 保存权重、指标和预测。

以下命令按已保存参数重建，运行位置为项目根目录。它们不是原始控制台命令的逐字归档；采用新目录，避免覆盖本次证据。

```powershell
python code/method-a2/scripts/generate_dataset.py --output-dir code/method-a2/data/s0-lg-repro --case-name ieee13 --samples-per-bus 40 --normal-ratio 1 --seed 342 --val-ratio 0.15 --test-ratio 0.15 --fault-classes 0 --run-id data-s0-lg-repro
python code/method-a2/scripts/train_marker_mvp.py --data-dir code/method-a2/data/s0-lg-repro --output-dir code/method-a2/output/candidate-score-lg-repro --epochs 200 --batch-size 64 --learning-rate 0.001 --lambda-rank 0.1 --lambda-response 0.1 --rank-margin 0.1 --rank-temperature 1 --temporal-layers 3 --encoder-layers 2 --pooling mean --temporal-mode single --seed 342 --device cpu --run-id candidate-score-lg-repro
```

当前 README 记录 OpenDSS COM 进程退出异常及子进程隔离方案。需要受控退出时使用 `main.py` 的数据模式，并按以上参数配置 `DataConfig`；训练模式显式指向生成的数据目录。`MODE="all"` 不自动连接数据输出和训练输入，必须核对两处路径。复现仍需已安装且已注册的 OpenDSS COM、对应 IEEE 算例和可用 Python 依赖；历史产物没有完整环境版本快照。

### 8.2 S1/S2 已有结果复核与重建接口

当前仓库没有发现独立归档的 S1/S2 执行脚本，因此无法给出原始脚本路径或一条已验证的重跑命令。已有结果可以直接读取第 6、7 节 JSON，逐组指标取平均，并与各文件的 `means` 核对；这不需要重新训练。

按当前代码接口重建测试时，应遵循：

1. 用 `load_dataset()` 加载 `s0-lg-run3`；读取 `model.pt` 的模型配置并构造 `MarkerResponsePredictor`，其中 `derived_channels` 对应构造参数 `use_derived_channels`；载入 `model_state_dict`，切换 `eval()`，关闭梯度。
2. 对训练、验证、测试索引分别仅选 `y_loc >= 0` 的故障样本，数量分别为 448、96、96。
3. S0：使用完整边与完整观测前向，读取 `candidate_score[:, :16]`，调用 `rank_metrics()`。
4. S1：从 `edge_index` 配对反向边，在基准 `edge_mask` 上同时屏蔽两个方向；枚举 1/2/3 条无向边，逐组前向和统计。
5. S2：将保存的观测母线映射为长度 16 的节点掩码，通过 `node_mask` 接口传入模型；不将被屏蔽节点从候选集合删除。当前 `encode_shared()` 会把被屏蔽节点的标准化输入置零并将掩码传给共享 GNN。
6. 对相同规模的掩码结果取等权平均，保留具体边和母线清单，写入新的 `output/` 子目录。

上述步骤是依据当前模型接口整理的重建规程。由于缺少原始 S1/S2 脚本，历史执行是否还对节点表示或输入进行额外处理、随机掩码抽样种子等细节不能从保存结果完全确认；原始 JSON 中的具体掩码清单应作为复核固定条件。

### 8.3 最终产物与测试记录

本次最终交付目录包含 `model.pt`、`config.json`、`dataset_audit.json`、`train_log.csv`、`validation_history.csv`、`validation_metrics.json`、`metrics.json`、`predictions.npz`、`oracle_metrics.json`、`hard_negative_pairs.json`、`probes.json`、`shortcut_audit.json`、`report.md` 和前述 S1/S2 文件。

`unit_tests.txt` 仅保存如下测试命令，没有保存退出码或通过数量，故不能将其作为全部测试通过的证据：

```powershell
python -m pytest -q code/method-a2/tests/test_progress.py code/method-a2/tests/test_marker_mvp.py code/method-a2/tests/test_accuracy_upgrade.py
```

本次文档整理只复核已有结果、日期、配置与路径，不重新运行数据生成、训练或扰动测试。判据一共享状态 Probe 使用另一套 paired-v1 数据和历史 teacher 权重，不能混作本次 S0/S1/S2 的运行步骤；其独立结论见 [判据一执行结论](Method-A2-判据一-共享状态可恢复性Probe执行结论.md)。

## 9. 适用边界

本次成功限定为单种子、单一 IEEE13 拓扑、单相接地、保存的数据划分和直接评分头协议。S1/S2 为同一模型的推理扰动测试，并非独立物理数据集上的重新训练实验。反事实响应误差、Oracle 物理间隔和评分头间隔属于不同量，不能直接比较或据此证明反事实距离定位的有效性。未来扩大故障类型、跨拓扑、引入真实测量噪声或训练缺测模型时，应另行登记数据契约、执行参数和独立测试结果。
