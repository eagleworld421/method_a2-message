<!-- 摘要：记录 Method-A2 RaReF 论文主线、已核验证据、实验结果、主张边界、章节结构、段落级写作计划与证据索引。本文以 RaReF 为唯一正文方法主线，GReF/GRReF 仅作为方法演化和对照背景；主要证据来自 IEEE 37 节点、单相接地故障、部分节点观测与区域概率校准实验。 -->

# Method-A2 RaReF 论文结构与证据台账

记录日期：2026-11-11

## 1. 论文主线

一句话主线：RaReF 将候选故障的响应能量评分转化为拓扑感知的区域概率预测，使配电网在部分节点观测和真实故障节点不可观测时能够输出具有经验覆盖率、可校准且可解释的故障区域。

建议标题：

1. `RaReF: Region-aware Response-energy Fault Localization under Partial Observability`
2. `Calibrated Topological Region Prediction for Distribution-network Fault Localization with Missing Node Measurements`
3. `Region-aware Counterfactual Response-energy Fault Localization for Distribution Networks`

正文方法名称统一为 RaReF（Region-aware Response-energy Fault Localization）。历史目录和参数中的 `gref`、`three-term` 与 `region` 保留为代码兼容名称。GReF 和 GRReF 只用于说明方法演化、基线和消融，不作为并列贡献展开。

## 2. 项目已完成内容

### 2.1 已实现

- 拓扑感知区域核：按候选节点间图跳数距离构造列归一的指数核。
- 候选单节点概率：由候选响应能量经温度缩放 softmax 得到。
- 区域概率：将单节点概率通过拓扑核扩散为区域概率。
- 条件区域目标：根据真实故障节点是否被观测使用不同的 `beta_obs` 和 `beta_miss`。
- 联合损失：响应能量、软列表排序和区域概率交叉熵。
- 训练期节点掩码增强：覆盖完整观测、真实节点被观测、真实节点未被观测和正常样本部分观测。
- 验证集温度标定、最小累计概率区域、覆盖率、区域大小、P90、拓扑半径和中心距离评价。

主要实现文件：

- `code/method-a2/src/gref/region.py`
- `code/method-a2/src/gref/losses.py`
- `code/method-a2/src/gref/training.py`
- `code/method-a2/scripts/eval_region_term.py`

### 2.2 已运行

- RaReF 第一阶段训练与测试。
- 区域损失权重扫描。
- 种子 342、343、344 的配对验证复核。
- `beta_miss` 与区域核尺度 `sigma` 定向实验。
- 区域概率经验覆盖率校准。
- 完整观测和 35、30、25、20、15 个观测节点的部分观测实验。

### 2.3 已保存

- 训练、验证、测试指标 JSON。
- 预测响应、区域概率、节点掩码和区域核 NPZ 文件。
- 权重扫描汇总、梯度分析和多种子汇总。
- 训练日志、配置、数据审计、快捷方式审计和报告文件。

## 3. 数据和实验口径

主数据集为 `s0-ieee37-r0p01-20-loguniform`：

- IEEE 37 节点配电网；
- 38 个节点，39 个候选，包括 38 个故障候选和 1 个正常候选；
- 单相接地故障；
- 故障阻抗范围 0.01–20 Ω，对数均匀抽样；
- 2736 条样本，其中 1520 条故障、1216 条正常；
- 40 个故障事件族；
- 训练、验证、测试规模为 2280、228、228；
- OpenDSS 仿真生成响应窗口；
- 主要部分观测协议为 35、30、25、20、15 个观测节点，各 10 组共享掩码。

数据登记入口：`docs/project/Method-A2-主要实验数据集登记.md`。

## 4. RaReF 方法公式

候选能量转换为单节点概率：

$$
p_{m,k}=\operatorname{softmax}(-e_{m,k}/T).
$$

拓扑区域核：

$$
K(r\mid k)=\frac{\exp[-d_G(r,k)/\sigma]}{\sum_j\exp[-d_G(j,k)/\sigma]}.
$$

区域概率：

$$
q_{m,r}=\sum_k p_{m,k}K(r\mid k).
$$

条件区域监督目标：

$$
y_{m,r}=(1-\beta)\mathbf{1}[r=k_m]+\beta K(r\mid k_m),
$$

其中真实故障节点被观测时使用 `beta_obs`，未被观测时使用 `beta_miss`。

联合目标：

$$
L=\lambda_E L_E+\lambda_R L_R+\lambda_{\mathrm{region}}L_{\mathrm{region}}.
$$

选定配置为：

$$
(\lambda_E,\lambda_R,\lambda_{\mathrm{region}})=(1.0,0.141,0.194),
$$

并使用 `sigma=1.0`、`beta_obs=0.1`、`beta_miss=0.4` 作为主测试配置。

## 5. 当前最完整结果

选定运行：`code/method-a2/output/gref/region-sweep-e1-m30-seed342/`。

完整观测测试结果：

- Top-1：92.54%；
- Top-3：99.12%；
- Top-5：100%；
- 平均名次：1.1009；
- `alpha=0.1` 区域覆盖率：100%；
- 平均区域大小：7.14 个节点。

部分观测总体 Top-1：

- 35 个观测节点：87.68%；
- 30 个观测节点：81.23%；
- 25 个观测节点：74.25%；
- 20 个观测节点：66.62%；
- 15 个观测节点：53.86%。

真实故障节点未被观测时：

- 单节点 Top-1：70.00%、59.58%、43.21%、37.78%、27.17%；
- `alpha=0.1` 区域覆盖率：97.78%、97.92%、97.82%、96.48%、94.13%；
- 平均区域大小约为 7.15–8.19 个节点。

真实故障节点被观测时，单节点 Top-1 为 89.19%、87.00%、90.40%、92.58%、94.78%。

## 6. 多种子和校准结论

种子 342、343、344 的验证集配对复核显示：

- RaReF 区域定位配置相对原始三项损失基线，未观测子集单点 Top-1 平均提高约 51.6 个百分点；
- 未观测区域覆盖率平均提高约 19.8 个百分点；
- 第二阶段配置相对第一阶段，完整观测 Top-1 平均提高 2.63 个百分点，观测子集提高 7.36 个百分点，部分观测总体提高 4.99 个百分点；
- 第二阶段相对第一阶段的未观测单点 Top-1 差值为 `+0.85±5.36` 个百分点，处于种子波动范围内，不能宣称单点恢复稳定改善。

校准结果显示，选定 RaReF 配置在未观测条件下的经验覆盖率为：

- `alpha=0.05`：98.2%；
- `alpha=0.10`：96.1%；
- `alpha=0.20`：90.8%。

这些值高于对应名义覆盖率，说明区域概率具有偏保守的覆盖特征。校准目前主要基于验证集，不能替代独立测试集校准。

## 7. 论文目录和章节任务

### 7.1 Introduction

说明配电网故障定位、部分观测困难、单点预测的过度确定性、候选反事实响应和 RaReF 的区域概率思想。贡献最多写四条：拓扑区域概率损失、观测条件建模、区域覆盖率评价、IEEE 37 部分观测验证。

### 7.2 Related Work

分为模型驱动故障定位、配电网 GNN、概率集合和置信度校准三个主题。需要补充外部文献，不能用项目内部报告替代相关工作引用。

### 7.3 Problem Formulation

定义事实观测、候选故障节点、候选响应、真实拓扑、观测拓扑、节点掩码、单节点定位和区域定位目标。

### 7.4 Proposed Method

建议小节：

1. Counterfactual Response Prediction；
2. Candidate Energy Scoring；
3. Topology-aware Region Kernel；
4. Conditional Region Supervision；
5. Joint Training Objective；
6. Missing-node Augmentation；
7. Calibrated Region Inference。

### 7.5 Method Properties

不写未经证明的理论定理。建议写区域核归一性、区域概率归一性、最小累计概率区域构造和经验覆盖率校准性质。

### 7.6 Experimental Setup

说明数据、OpenDSS 仿真、训练验证测试划分、掩码协议、基线、指标、超参数和随机种子。

### 7.7 Results and Discussion

依次写完整观测、部分观测、真实故障节点未观测、区域覆盖率、区域大小、概率熵和失败案例。

### 7.8 Ablation Study

包括图平滑替换、损失权重扫描、`beta_miss`/`sigma` 定向实验和校准消融。

### 7.9 Limitations

明确单一拓扑、合成数据、单相故障、测试种子有限、缺少噪声和现场数据、未观测条件下单点恢复不稳定。

### 7.10 Conclusion

结论应强调区域定位和经验覆盖率，不应写成跨拓扑泛化或理论覆盖保证。

## 8. 段落级写作计划

### Introduction

- I-1：配电网故障定位的应用重要性。
- I-2：部分节点观测使单点定位不可靠。
- I-3：候选反事实响应将定位转化为观测一致性比较。
- I-4：单点概率无法表达拓扑邻近和信息不足。
- I-5：RaReF 使用拓扑核将单点概率扩散为区域概率。
- I-6：列出方法、训练协议和实验验证贡献。
- I-7：限定研究范围为 IEEE 37 合成单相接地故障和部分观测。

### Related Work

- R-1：模型驱动和响应匹配故障定位。
- R-2：配电网图神经网络和拓扑建模。
- R-3：集合预测、区域定位和置信度校准。

### Results

- E-1：数据和评价协议。
- E-2：完整观测定位结果。
- E-3：观测节点逐级减少时的 Top-K。
- E-4：真实故障节点未观测时的单点与区域结果。
- E-5：区域覆盖率和区域大小权衡。
- E-6：多种子验证和校准结果。
- E-7：失败案例与单点恢复边界。

## 9. 主张状态

### 已核验证据

- RaReF 的区域核、区域概率、条件目标和训练流程已实现。
- 完整观测和部分观测实验已运行并保存产物。
- RaReF 相对原始三项损失基线的区域覆盖率优势已在三种子验证集上复核。
- 经验覆盖率和区域大小已保存并可复算。

### 有结果但尚未完整核验

- 选定配置的测试集只有种子 342 一次评价。
- 权重扫描的细小覆盖率差异接近重复运行波动。
- 校准结果主要来自验证集样本内评价。

### 可提出的研究假设

- 拓扑扩散区域比强制单点预测更适合真实故障节点不可观测的场景。
- 区域概率损失可以缓解部分观测下的过度自信。
- 区域输出可能比单点输出更适合作为部署时的选择性诊断结果。

### 当前没有支持的主张

- 跨拓扑泛化；
- 真实噪声鲁棒性；
- 现场数据有效性；
- 多相和多故障泛化；
- 理论覆盖率保证；
- 未观测故障节点的稳定单点恢复；
- 统计显著性和普遍规律。

## 10. 关键证据缺口

1. 跨拓扑测试。
2. 多种子独立测试集。
3. 独立校准集。
4. 真实测量噪声和非随机缺测。
5. 多相、高阻和多故障场景。
6. 公平外部基线。
7. 覆盖率和区域大小的统计显著性。
8. 区域损失的独立因果贡献。
9. 区域覆盖理论或有限样本保证。
10. 真实配电网现场数据。

## 11. 建议写作顺序

1. 研究问题和贡献；
2. RaReF 方法总览；
3. Problem Formulation；
4. Experimental Setup；
5. Results；
6. Ablation and Calibration；
7. Introduction；
8. Related Work；
9. Abstract、Title、Limitations 和 Conclusion。

下一步优先写三段：

1. Problem Formulation 中的部分观测故障定位定义；
2. Method 中的 Region-aware Probability Construction；
3. Results 中的 Unobserved Fault-node Evaluation。

## 12. 主张—证据索引

- C1：区域核实现——`src/gref/region.py`、`region_kernel.npz`。
- C2：区域概率损失——`src/gref/losses.py`、`train_log.csv`。
- C3：节点掩码增强——`src/gref/training.py`、`config.json`。
- C4：IEEE 37 数据契约——`docs/project/Method-A2-主要实验数据集登记.md`、数据目录 `meta.json`。
- C5：完整观测 92.54% Top-1——选定运行目录 `metrics.json`。
- C6：部分观测区域覆盖率——选定运行目录 `region_metrics.json`。
- C7：多种子区域收益——`region-multiseed-summary-20261110/multi_seed_summary.json`。
- C8：权重扫描选模——`region-sweep-summary-20261110/weight_scan_summary.json`。
- C9：定向参数和校准——`region-exp2-summary-20261110/exp2_summary.json` 及校准 JSON。
- C10：单点恢复尚不稳定——`Method-A2-区域概率损失多种子配对复核报告.md`。

## 13. 直接来源文件

- [RaReF 第一阶段报告](Method-A2-区域概率损失第一阶段实验报告.md)
- [RaReF 第二阶段权重扫描](Method-A2-区域概率损失第二阶段权重扫描实验报告.md)
- [RaReF 多种子复核](Method-A2-区域概率损失多种子配对复核报告.md)
- [RaReF 未观测定向与校准](Method-A2-区域概率损失未观测定向与置信度校准实验报告.md)
- [主要实验数据集登记](../Method-A2-主要实验数据集登记.md)
- [核心公式](../Method-A2-核心公式.md)
