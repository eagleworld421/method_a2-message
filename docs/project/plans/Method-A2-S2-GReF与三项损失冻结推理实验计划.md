<!-- 摘要：本文规定 Method-A2 在 IEEE37 S0 数据上对标准 GReF 与高 Top-1 的GRReF A tau1x 模型进行 S2 部分节点观测冻结推理对比的执行参数、掩码生成、评价指标、产物契约和验收条件。实验复用 `s0-ieee37-r0p01-20-loguniform` 的物理响应、划分和标准化，保持真实拓扑不变，在推理期按 35、30、25、20、15 个观测节点生成各 10 组可复现掩码，并将同一组掩码用于两个冻结模型。本文为实验执行前计划，待用户确认参数后执行，不包含实验结果。 -->

# Method-A2 S2 GReF 与三项损失冻结推理实验计划

日期：2026-11-09

文档定位：实验执行前计划。本文只规定 S2 部分节点观测条件下的冻结模型推理对比，不进行重新训练，不修改拓扑，不进行训练期拓扑增强。

## 1. 实验目标

在相同 IEEE37 S0 数据、相同测试划分和相同节点观测掩码下，比较标准 GReF 与GRReF 模型面对部分节点观测时的候选定位性能变化，并记录观测证据减少对候选排序、响应能量和置信间隔的影响。

本实验只改变部署输入中的 `node_mask`。真实拓扑、候选集合、物理响应、模型参数、输入标准化和测试事件顺序均保持固定。

## 2. 待确认的执行参数

### 2.1 数据集

- 数据集名称：`s0-ieee37-r0p01-20-loguniform`。
- 实际路径：`code/method-a2/data/s0-ieee37-r0p01-20-loguniform/`。
- 算例：IEEE37，38 个非电源母线节点，74 条有向真实拓扑边，39 个候选，其中包含正常候选。
- 电阻范围：0.01–20 Ω，对数均匀抽样。
- 测试划分：沿用两个冻结模型各自 `split.npz` 中的测试索引。当前两套运行均对应 228 个测试事件。
- 数据处理：不复制、不改写原始数据，不生成新的 OpenDSS 物理响应。S2 掩码仅在批次构造时传入模型。

### 2.2 冻结模型

- 标准 GReF：`code/method-a2/output/gref/ieee37-a-seed342-20261103/`。
  - 损失口径：标准 GReF 的门控定位项与全候选响应重构项。
  - 配置：`temperature=0.012815295718610287`，`lambda_family=0.1`。
- GRReF A tau1x：`code/method-a2/output/gref/three-term-ieee37-a-tau1x-seed342/`。
  - 既有完整观测测试 Top-1：82.89%。
  - 权重：`lambda_energy=1.0`、`lambda_rank=0.3`、`lambda_smooth=0.1`。
  - 排序温度：`rank_temperature=0.0040640985`。

本轮三项损失只使用 A tau1x 检查点。该检查点对应既有完整观测测试 Top-1 超过 80% 的三项损失配置。

### 2.3 节点观测掩码

- 观测节点数：35、30、25、20、15。
- 每个观测规模：10 组随机掩码。
- 掩码随机种子：342。
- 生成规则：第 `i` 个观测规模使用 `seed = 342 + 1009*i`，从 38 个节点中无放回抽取指定数量，掩码按节点索引升序记录。
- 公平性约束：两个模型必须读取完全相同的 `node_masks.npz`，不得分别重新抽样。
- 完整观测基线：38 个节点全部观测，作为两个模型各自的 S0 对照。

35、30、25、20、15 个观测节点用于覆盖从轻度到重度的连续缺测程度。由于当前 IEEE37 有 38 个节点，35 个节点保留轻度缺测，15 个节点保留较强缺测。

## 3. 推理流程

执行入口为 `code/method-a2/scripts/eval_s2_nodes.py`，每个模型单独运行一次。

每次运行依次完成以下步骤：

1. 读取运行目录中的 `config.json`、`model.pt`、`normalization.npz` 和 `split.npz`。
2. 加载登记的数据集并核对节点数、候选数、测试索引和模型配置。
3. 生成完整观测基线和四个部分观测规模的十组节点掩码。
4. 对每组掩码在冻结模型上完成测试集推理。
5. 以同一候选有效性掩码计算定位指标，并保存逐掩码节点索引和指标。
6. 汇总每个观测规模的均值、最小值和最大值，并计算相对完整观测基线的变化。

推荐执行命令如下，用户确认参数后再运行：

```powershell
$env:PYTHONPATH=(Resolve-Path 'code/method-a2').Path
python code/method-a2/scripts/eval_s2_nodes.py `
  --run-dir code/method-a2/output/gref/ieee37-a-seed342-20261103 `
  --data-dir code/method-a2/data/s0-ieee37-r0p01-20-loguniform `
  --output-dir code/method-a2/output/gref/ieee37-a-seed342-20261103/s2-node-eval `
  --observed-nodes 35 30 25 20 15 --masks-per-level 10 --seed 342 --device cuda

python code/method-a2/scripts/eval_s2_nodes.py `
  --run-dir code/method-a2/output/gref/three-term-ieee37-a-tau1x-seed342 `
  --data-dir code/method-a2/data/s0-ieee37-r0p01-20-loguniform `
  --output-dir code/method-a2/output/gref/three-term-ieee37-a-tau1x-seed342/s2-node-eval `
  --observed-nodes 35 30 25 20 15 --masks-per-level 10 --seed 342 --device cuda
```

## 4. 评价指标

每个观测规模、每个掩码和每个模型均记录：

- Top-1、Top-3、Top-5；
- 平均名次和名次中位数；
- 正确候选相对最佳错误候选的 signed margin 中位数；
- signed margin 小于零的事件比例；
- 相对完整观测基线的 Top-1、Top-3、Top-5 和平均名次变化。

主报告分别给出两个模型在四个观测规模上的掩码均值、最小值和最大值，并保留逐掩码结果用于复核。GReF 与GRReF 的比较使用相同掩码逐项对齐，不将不同随机掩码混合平均。

## 5. 产物契约

每个模型的输出目录必须包含：

- `s2_node_metrics.json`：运行配置、损失模式、测试规模、完整观测基线、逐掩码结果和分层汇总；
- `s2_node_metrics.md`：面向复核的指标摘要；
- `node_masks.npz`：四个观测规模使用的节点掩码；
- 原始运行配置和模型目录保持不变，不覆盖已有 S0、S1 产物。

实验完成后新增结果报告归档到 `docs/project/reports/`，报告至少包含数据核对、两个模型的损失配置、掩码生成规则、分层指标、模型间差异和单种子边界。

## 6. 验收条件

执行前检查必须通过：

- 两个运行目录均能加载对应模型和标准化文件；
- 两个模型的 `n_nodes`、候选轴和测试索引与数据一致；
- `node_masks.npz` 中每个掩码为二值、观测节点数准确且无重复；
- 两个模型使用相同的掩码数组；
- 完整观测基线可复现；
- 所有输出指标为有限数值，模型前向不产生 NaN；
- 原始 S0 数据文件的修改时间和内容不发生变化。

若任一模型无法加载、测试索引不一致或掩码数量不足，停止正式评估并先修正配置。正式结果只在四个观测规模和每级十组掩码均完成后归档。

## 7. 解释边界

本实验比较的是两个已训练模型在 S2 部分观测输入下的冻结推理表现。结果可用于描述观测证据减少对定位指标和候选排序的影响，不能直接解释为缺测条件下重新训练模型的性能上限，也不能把 GReF 与三项损失差异归因于单一损失项之外的因素。

本计划不包含训练期节点缺测增强、重新训练、传感器噪声建模、S1 与 S2 混合错误或选择性拒识阈值校准。

