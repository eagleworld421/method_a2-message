<!-- 摘要：记录 Method-A2 Node Marker MVP 在 s0-marker-run1 上的数据生成与训练执行结果。第 1 节记录执行环境与代码基线；第 2 节记录数据生成参数、产物、实测耗时与数据契约核验结论（反事实族合法、无跨集分组、无候选相关随机扰动、事实观测与签名库对应位置逐元素一致）；第 3 节记录训练配置、逐轮损失曲线、可训练参数量与测试集指标；第 4 节记录故障定位结果并按 scheme P2 判定为未显示增益；第 5 节记录契约与进度测试结果；第 6 节记录产物清单与可复核入口；第 7 节如实记录 ETA 估计偏差、埋点未触发条件与响应距离口径差异；第 8 节记录未覆盖内容与不可外推边界。含结论表述，不含方法有效性主张。 -->

# Method-A2 Node Marker MVP 执行报告（s0-marker-run1）

> 运行日期：2026-10-08
> 代码基线：Git commit `c68a78b`（分支 `agent/DOC-PUSH-001-doc-sync`，工作树含未提交改动）
> 执行范围：S0 单一拓扑、完整观测、16 个故障候选加 1 个正常候选
> 执行依据：`docs/project/plans/Method-A2-Node-Marker-MVP执行方案与上下文同步.md`、`docs/project/plans/Method-A2-节点级共享表示-MVP实施方案.md`、`docs/project/Method-A2-核心公式.md`

## 1. 执行环境与代码基线

本次执行在 Windows 环境下使用 Python 3.9.12、PyTorch 2.5.1+cu121（仅使用 CPU）、NumPy 1.26.4。OpenDSS 通过 COM 接口驱动，`pywin32` 与 `pythoncom` 可正常加载，IEEE 13 节点算例文件位于 `C:\Program Files\OpenDSS\IEEETestCases\13Bus\IEEE13Nodeckt.dss`。

执行前的可用性核验结果为：`ieee13` 返回 16 个母线、15 个负荷、30 条有向边参数，单次场景生成实测 0.22 秒。该核验确认 OpenDSS 与必要数据在本环境可用，未触发执行方案第 7 节的阻塞条件。

代码基线未做任何修改。本次执行只新增一个可复用的进度读取辅助脚本 `code/method-a2/scripts/inspect_progress_rates.py`，未修改核心公式文档、模型源码、数据生成源码与训练源码。工作树中既有的未提交改动在执行前已被确认，本次未提交、未回滚、未覆盖。

## 2. 数据生成

### 2.1 参数与产物

运行标识为 `data-s0-marker-run1`，状态 `completed`。命令为：

```powershell
python code/method-a2/scripts/generate_dataset.py `
  --output-dir code/method-a2/data/s0-marker-run1 `
  --case-name ieee13 `
  --samples-per-bus 10 `
  --normal-ratio 1.0 `
  --seed 342 `
  --run-id data-s0-marker-run1 `
  --no-terminal-progress
```

产物写入 `code/method-a2/data/s0-marker-run1/`，包含 16 个数组文件与 `feature_scaler.npz`、`meta.json`。`meta.json` 记录的关键契约量为：`n_nodes = 16`、`n_candidates = 17`、`n_edges = 30`、`n_samples = 320`、`n_fault = 160`、`n_normal = 160`、`n_train = 288`、`n_test = 32`、`window_len = 12`、`feature_dim = 6`、`feature_channels = [Re_A, Im_A, Re_B, Im_B, Re_C, Im_C]`、`res_min = 0.1`、`res_max = 100.0`、`seed = 342`、`counterfactual_perturbation = disabled`。数组形状为 `signature_bank [320,17,16,12,6]`、`X_obs [320,16,12,6]`、`edge_index [30,2]`，与实施方案第 3.3 节规定的 `N=16`、`C=17`、`T=12`、`F=6` 一致。

实测总耗时为 147.247 秒，其中仿真累计 146.48 秒，共调用 OpenDSS 求解 5441 次。稳定阶段的单场景实测速率约为 0.458 秒，热身阶段为 0.188 秒。

### 2.2 数据契约核验

核验在产物文件上独立复算，不读取生成脚本的自报值。结论如下。

- 反事实族合法：10 个非负 `group_id`，每个分组恰好包含 16 条样本，其故障位置覆盖 0 至 15 且互不重复，说明同一分组内固定事件参数、只改变事实故障位置，满足核心公式第 5.2 节对正配对集合 $\mathcal P_+$ 的构造要求。
- 分组不跨集合：训练集故障分组为 `{0,1,2,4,5,6,7,9}`，测试集故障分组为 `{3,8}`，交集为空；`dataset_audit.json` 中 `train_test_group_overlap` 为 0。
- 候选相关随机扰动已移除：同一 `group_id` 内正常候选响应在各样本之间逐元素完全相同，最大逐元素差为 0.0。该结果说明波形生成中随候选索引变化的随机相位、幅值噪声与阻尼扰动在本次数据中不产生影响，严格配对语义成立。
- 事实观测与目标一致：对全部 160 个故障样本，`X_obs[i]` 与 `signature_bank[i, y_loc[i]]` 逐元素精确相等，说明事实观测确实等于签名库中真实位置那一项，符合核心公式第 5.5 节记录的 $X_m=S^{CF}_{m,K_m^{true}}$。
- 正常候选语义：正常候选位于候选轴索引 16，其响应在同一分组内恒定、在不同分组之间随负荷倍率变化（事件间标准差均值 0.0156，最大逐元素差 0.162），符合"无故障基态窗口由共享事件参数决定"的数据契约；160 个正常样本的 `y_loc` 全为 -1，不进入故障母线定位评价。
- 训练集与测试集样本数分别为 288 与 32，其中训练集含 160 个正常样本，测试集不含正常样本，测试集全部为 32 个故障事件。

## 3. 训练

### 3.1 配置

运行标识为 `train-s0-marker-run1`，状态 `completed`。命令为：

```powershell
python code/method-a2/scripts/train_marker_mvp.py `
  --data-dir code/method-a2/data/s0-marker-run1 `
  --output-dir code/method-a2/output/node-marker-mvp/train-s0-marker-run1 `
  --epochs 20 `
  --batch-size 16 `
  --seed 342 `
  --run-id train-s0-marker-run1 `
  --no-terminal-progress
```

训练配置为：`epochs = 20`、`batch_size = 16`、`learning_rate = 0.001`、`lambda_pair = 0.1`、`lambda_list = 0.1`、`rank_temperature = 1.0`、`seed = 342`、`device = cpu`。可训练参数量为 36040，其中共享时序编码器与固定边均值图编码器输出节点级共享表示 `Z_m`，候选条件化解码器由两层候选条件化消息传递层与节点输出头构成。

模型路径不包含 teacher 分支、特权阻抗输入、QKV、候选—节点注意力、可学习边打分、`global_repr` 或 `g_m`。同一事件的全部候选复用同一个 `Z_m`，候选差异只经 `(b_k, t_k)` 进入解码器。`K_m^{true}`、真实反事实响应、真实阻抗与 `group_id` 均不进入部署前向；`group_id` 与真实位置仅用于 `L_pair` 的配对构造与离线评价。

### 3.2 训练耗时

实测总耗时为 29.904 秒，其中训练阶段 29.014 秒、评估阶段 0.190 秒、写出阶段 0.055 秒。训练阶段共执行 360 个批次，平均单批次实测 0.0786 秒。整次运行写入 367 条进度记录，包含 3 条 `start`、362 条 `unit`、3 条 `stage_done` 与 1 条 `end`，不存在 `error` 记录。

### 3.3 逐轮损失

三项损失按方案要求同时报告，不增设新的监督项。按轮次平均的结果如下（`family` 为 $L_{\mathrm{family}}$ 即未加权的均方响应重构误差，`pair` 为 $L_{\mathrm{pair}}$，`list` 为 $L_{\mathrm{list}}$，`total` 为 $L_{\mathrm{train}}$）。

- 第 1 轮：`family 0.912563`、`pair 0.077293`、`list 0.000619`、`total 0.920354`
- 第 5 轮：`family 0.141591`、`pair 0.008326`、`list 0.000367`、`total 0.142461`
- 第 10 轮：`family 0.011692`、`pair 0.001861`、`list 0.000364`、`total 0.011914`
- 第 15 轮：`family 0.006211`、`pair 0.001635`、`list 0.000364`、`total 0.006411`
- 第 20 轮：`family 0.006104`、`pair 0.001073`、`list 0.000364`、`total 0.006247`

末批次（第 20 轮最后一个批次）的三项值为 `family 0.002493`、`pair 0.000736`、`list 0.0000058`、`total 0.002567`。响应重构项在 20 轮内下降约两个数量级并进入平台；配对一致性项同步下降，说明同组内不同事实位置的共享表示趋于一致；列表排序项从 0.000619 降至 0.000364 后基本不再变化，其绝对值在加权后（权重 0.1）对总损失的贡献约为 0.01%，即该项在本次配置下几乎未产生有效梯度信号。

### 3.4 测试集指标

测试集包含 32 个故障事件。指标结果为 `test_family_loss 0.004295`、`test_pair_loss 0.002459`、`test_list_loss 0.0000141`、`test_total_loss 0.004542`。测试族损失高于训练末批次损失（0.002493）约 1.7 倍，处于同一量级。

指标记录的可信性已独立核验：由 `predictions.npz` 中的预测响应与数据文件中的目标响应直接复算未加权均方族损失，得 0.00429458，与 `metrics.json` 记录的 0.00429458 一致。

预测响应不为常数：32 个测试事件中，同一事件内不同候选的响应最大差全部大于 0（最小值 0.157），说明候选条件确实进入了解码输出。

## 4. 故障定位结果

定位评价只在 16 个故障候选上进行，正常候选不并入定位排名。部署评分口径为事实观测与预测响应之间的未加权均方差，按节点、时间与通道求均值。

- Top-1 为 0.09375，即在 32 个测试事件中 3 个事件将真实候选排在第一；16 个故障候选中随机猜测的期望值为 0.0625。
- Top-3 为 0.34375。
- 真实候选平均名次为 6.34，中位名次为 5.5，随机猜测的期望名次为 8.5。
- 预测 signed margin（最优候选评分减真实候选评分）的中位数为 $-1.581\times10^{-4}$，负值比例为 90.625%，即绝大多数事件的最优候选不是真实候选。
- 评分量级特征：真实候选评分中位数为 $1.580\times10^{-3}$，全候选最小评分中位数为 $1.143\times10^{-3}$，候选间评分极差中位数仅为 $8.69\times10^{-4}$。三重事实说明定位排序发生在很低的信噪比下：模型对真实候选的响应吻合程度已经接近其对任意候选的吻合程度，而候选之间的评分差异又远小于这一吻合量级。

按实施方案第 8.2 节的分层验收判定：P0（接口与信息边界）通过，P1（真实训练可运行）通过，P2（方法比较）在本轮配置下未显示增益。该判定不支持据此把核心公式中的十维候选描述正式替换为 `(b_k, t_k)`，也不支持宣称定位能力已经建立。

## 5. 契约与进度测试

按执行要求运行：

```powershell
python -m pytest -q code/method-a2/tests/test_progress.py code/method-a2/tests/test_marker_mvp.py
```

结果为 `9 passed in 2.04s`。覆盖内容包含 Marker 与正常候选语义、前向输出形状与共享表示唯一性、不返回全局表示、候选敏感性与梯度回传、候选置换等变、损失掩码、无序随机扰动的动态窗口，以及进度写入与读取（含残缺行容错与完成态不报虚假剩余时间）。

## 6. 产物清单与可复核入口

数据侧产物位于 `code/method-a2/data/s0-marker-run1/`，包含 `X_obs.npy`、`X_full.npy`、`signature_bank.npy`、`mask.npy`、`edge_index.npy`、`edge_attr.npy`、`edge_mask.npy`、`y_detect.npy`、`y_loc.npy`、`y_class.npy`、`y_resist.npy`、`group_id.npy`、`train_idx.npy`、`test_idx.npy`、`feature_scaler.npz` 与 `meta.json`。

日志侧产物位于 `code/method-a2/logs/data-s0-marker-run1/` 与 `code/method-a2/logs/train-s0-marker-run1/`，各含 `progress.jsonl` 与 `progress.latest.json`。两次运行均未生成 `error.log`，`run.log` 因使用 `--no-terminal-progress` 而为空文件，不是异常。

最终交付产物位于 `code/method-a2/output/node-marker-mvp/train-s0-marker-run1/`，按实施方案第 9 节要求包含 8 项：`config.json`（参数、数据契约副本、运行标识、进度文件路径、可训练参数量）、`dataset_audit.json`（分组存在性、跨集重叠、扰动状态）、`train_log.csv`（360 行逐批次三项损失）、`predictions.npz`（测试集预测响应、事实观测、真实位置、测试索引）、`metrics.json`（测试集四项损失与定位指标）、`shortcut_audit.json`（teacher、全局表示、QKV 与边打分均为未使用的声明）、`unit_tests.txt`（测试命令）、`report.md`（最小执行记录）。

进度查询入口为 `python code/method-a2/scripts/report_progress.py --run-id <run-id>`；速率与剩余时间趋势查询入口为 `python code/method-a2/scripts/inspect_progress_rates.py <progress.jsonl 路径>`，后者为本次执行新增的辅助脚本。

## 7. 执行偏差与已知限制

- 剩余时间估计偏差：数据生成开始时报告的估计剩余时间约为 77 秒，实际用时 147 秒，实测为估计的约 1.9 倍。偏差原因为 OpenDSS COM 的单场景耗时在热身阶段为 0.188 秒、在稳定阶段上升至 0.458 秒，早期记录按当时偏低的速率外推。训练阶段的估计约为 15 秒、实际 30 秒，偏差方向相同。按《Method-A2-运行进度与时长汇报规范》第 6.5 节要求，此处如实记录偏差而不修正历史估计。
- 埋点未触发条件：本次训练共 360 个批次，单个批次最长实测 0.079 秒，均短于写盘间隔，因此进度文件中不存在 `tick` 记录。该情形不触发规范第 4.3 节规定的"单元耗时超过写盘间隔却无中间记录"的不合格条件。
- 响应距离口径差异：当前 `L_family` 与 `L_list` 内部使用的距离为节点、时间与通道的未加权均值，核心公式第 4 节定义的 $d_S$ 则要求按训练集保存的节点尺度 $s_{n,f}$ 作归一化。两者在量纲与数值上不等价，本次执行未修改该实现。该差异同时影响损失量级与第 4 节所述的定位信噪比，属于需要单独决策是否修正的既有实现问题，本报告不对其作进一步结论。
- 列表排序项未生效：`L_list` 在加权后对总损失的贡献约为 0.01%，且在 20 轮内基本不变，因此本轮结果不能用于判断列表级排序损失是否有效，也不能用于判断排序温度取值是否合适。
- 无验证集：本次数据生成只产出 `train_idx` 与 `test_idx`，没有独立的验证集，因此排序温度、辅助损失权重与 checkpoint 选择规则均未依据验证集确定，测试集仅用于最终评估。
- 无 checkpoint 交付：本轮训练入口不保存模型权重，`code/method-a2/checkpoint/` 未产生本次运行的文件，最终交付全部来自 `output/`。这满足"不得把 checkpoint 作为最终交付目录"的规则，但也意味着本轮结果不可从权重层面复现，只能通过相同种子重跑复现。
- 单一种子：本次只运行种子 342 的单次实验，未报告初始化离散度，因此不能区分模型表现与训练波动。

## 8. 未覆盖内容与不可外推边界

- 本次执行未运行 A 基线（十维候选描述）与 B 路径的对照实验，因此第 4 节的定位结果不能解释为两种候选条件化实现的系统比较，也不能归因于候选描述的信息内容差异。
- 本次执行未实现或运行实施方案第 7.2 节要求的完整反捷径审计，`shortcut_audit.json` 只记录了三项结构性声明，候选置换、节点置换等变、跨事件 `Z` 置换与常数 `Z` 替换、远端节点候选敏感性等评价操作均未在真实数据上执行。
- 本次执行未涉及 S1 错误拓扑、S2 部分观测、S3 跨拓扑与 S4 高阻场景，任何关于容错性或泛化性的推断均超出本轮证据范围。
- 本次执行未修改核心公式中的 $e_k$ 定义，Node Marker 仍属待批准的接口替换方案；是否正式替换必须依据后续完整的 A/B 对照与审计结果决定。
