<!-- 摘要：规定 GReF 从“门控能量交叉熵＋全候选重构”改为“能量＋排序＋图平滑”GRReF 的预登记实验，包括三项公式、逐元素重构能量的均方定义、部署能量软列表 KL 的单一温度与训练侧标定规则、图拉普拉斯平滑的掩码与归一化、固定权重 (1,0.1,0.1)、训练与评价协议、产物契约，以及训练后三项梯度范数、两两余弦、合成投影与图频分解的分析协议与判定边界。本文为实验执行前的计划，不含实验结果。 -->

# Method-A2 GRReF改造计划

## 1. 背景与目标

[首轮训练测试报告](../reports/Method-A2-GReF-IEEE37首轮训练测试报告.md)与[损失梯度与重构退化分析](../reports/Method-A2-GReF-IEEE37损失梯度与重构退化分析.md)已确认：当前目标 $L=L_{loc}+0.1L_{family}$ 中，门控能量交叉熵在编码器上比加权重构强约 292 倍与 538 倍，在解码器上与其方向余弦中位数为 $-0.418$ 与 $-0.271$，且把预测响应替换为真实响应后总目标反而升高（A 自 0.17992 升至 1.09238，B 自 0.16684 升至 0.67245）。这说明交叉熵本身不是以物理真值为最优的目标项。

本计划把训练目标改为三项：

$$L=\lambda_E L_E+\lambda_R L_R+\lambda_S L_S,\qquad (\lambda_E,\lambda_R,\lambda_S)=(1.0,\,0.1,\,0.1).$$

- $L_E$：逐元素响应重构能量，保持唯一绝对锚定；
- $L_R$：部署能量的软列表排序，替换原交叉熵，只约束候选顺序而不要求人为间隔；
- $L_S$：误差场在图上的拉普拉斯平滑，约束误差的空间结构。

本次范围只改变训练目标与对应的分析与测试，不改变数据、分组划分、标准化口径、模型结构、部署评分定义与评价指标。

## 2. 与既有损失的关系

- 停用门控权重 $w_m$ 与温度 $T$：二者只服务于交叉熵，新的排序项不需要。
- 不采用 `value`（评分值回归真实距离）与 `gap`（评分差回归真实距离差）：既有消融显示二者进一步降低 Top-K 并主导梯度。
- 不采用 `rank-pair`（带可信间隔筛选的 pairwise logistic）：[核心公式](../Method-A2-核心公式.md)第 5.3 节已规定它只作为消融或基线，默认不与列表项叠加。
- 不加入族内表示一致性项 `pair`：本数据族的候选随机扰动关闭，但该项属独立消融，不进入本次三项。
- 正常候选仍参与 $L_E$ 与 $L_S$，不参与 $L_R$ 与故障定位评价。

## 3. 记号与共同口径

- 事件 $m$，候选 $k\in\mathcal K_m$（前 $N$ 个为故障候选，末尾为正常候选），故障候选掩码 $A_{m,k}$，观测掩码 $M_{m,n,t,c}\in\{0,1\}$。
- 在最终训练侧标准化坐标下记观测、真实响应与预测响应为 $\bar X_m$、$\bar S_{m,k}$、$\widehat{\bar S}_{m,k}$；坐标口径与首轮完全一致。
- 部署能量

$$e_{m,k}=\frac{\sum_{n,t,c}M_{m,n,t,c}\left(\bar X_{m,n,t,c}-\widehat{\bar S}_{m,k,n,t,c}\right)^2}{\sum_{n,t,c}M_{m,n,t,c}},\qquad
e^*_{m,k}=\frac{\sum_{n,t,c}M_{m,n,t,c}\left(\bar X_{m,n,t,c}-\bar S_{m,k,n,t,c}\right)^2}{\sum_{n,t,c}M_{m,n,t,c}}.$$

- 部署评分、Top-K 评价、选模之外的测试评价均沿用首轮口径，不做任何改动。

## 4. 三项损失定义

### 4.1 能量项

$$e^{rec}_{m,k}=\frac{\sum_{n,t,c}M_{m,n,t,c}\left(\widehat{\bar S}_{m,k,n,t,c}-\bar S_{m,k,n,t,c}\right)^2}{\sum_{n,t,c}M_{m,n,t,c}},$$

$$L_E=\frac{\sum_{m,k}A_{m,k}\,e^{rec}_{m,k}}{\sum_{m,k}A_{m,k}}.$$

- 采用均方而非 Huber：与部署能量 $e_{m,k}$ 同一泛函，梯度与误差成线性关系，便于逐项余弦解释。
- 报告时另给出 SmoothL1 口径的重构指标，以与首轮结果对照；该指标只用于评价，不参与训练。
- 分母为有效候选数，使每个候选等权；缺失节点或通道由 $M$ 同时排除于分子与分母。

### 4.2 排序项

$$p^*_{m,k}=\frac{A_{m,k}\exp\left(-e^*_{m,k}/\tau\right)}{\sum_{u}A_{m,u}\exp\left(-e^*_{m,u}/\tau\right)},\qquad
\hat p_{m,k}=\frac{A_{m,k}\exp\left(-e_{m,k}/\tau\right)}{\sum_{u}A_{m,u}\exp\left(-e_{m,u}/\tau\right)},$$

$$L_R=\frac{1}{|\mathcal M_f|}\sum_{m\in\mathcal M_f}D_{KL}\left(p^*_m\,\|\,\hat p_m\right),\qquad
D_{KL}=\sum_{k}p^*_{m,k}\left(\log p^*_{m,k}-\log\hat p_{m,k}\right),$$

其中 $\mathcal M_f$ 为批次内真实位置有效的故障事件。

- 排序对象为部署能量 $e_{m,k}$；因 $a_{m,k}=\sqrt{e_{m,k}}$ 单调，按 $a$ 排序与按 $e$ 排序等价，故该式即“预测响应偏离幅度的顺序分布接近真实顺序分布”。
- 只使用故障候选；正常候选的任意响应变化不改变 $L_R$。
- 温度 $\tau>0$ 是唯一新增超参数：$\tau\to0$ 退化为对 $\arg\min e^*$ 的硬交叉熵，$\tau\to\infty$ 退化为无监督。
- $\tau$ 的标定规则：在训练集上计算每个故障事件的真实能量，取该事件全部有效故障候选两两绝对差的中位数，再对全部故障事件取中位数，作为 $\tau$；结果写入 `config.json`，训练开始前冻结。该规则只使用训练集，不访问验证集与测试集。允许通过命令行显式覆盖，但覆盖值必须记录来源。

### 4.3 图平滑项

图 $G=(V,E)$ 取数据中的 74 条有向边去重后的 37 条无向边，$L=D-W$ 为组合拉普拉斯矩阵，首轮边权 $w_{uv}=1$。记逐节点误差向量 $f_{m,k,n}=\widehat{\bar S}_{m,k,n,\cdot}-\bar S_{m,k,n,\cdot}\in\mathbb R^{TF}$，则

$$f_{m,k}^\top Lf_{m,k}=\sum_{(u,v)\in E}w_{uv}\left\|f_{m,k,u}-f_{m,k,v}\right\|_2^2,$$

$$L_S=\frac{\sum_{m,k}A_{m,k}\,f_{m,k}^\top Lf_{m,k}}{2TF\sum_{m,k}A_{m,k}\sum_{(u,v)}w_{uv}}.$$

- 常数误差场位于 $L$ 的零空间，故该项不惩罚共模误差；响应水平必须由 $L_E$ 锚定，报告中须显式说明。
- 节点掩码生效时，只累计两端节点均有效的边，分母按实际有效边权同步调整。
- 电气阻抗加权只作为后续敏感性口径，本轮不启用；现有 `edge_attr` 的线路长度与变压器量纲缺陷尚未修复，直接使用会混淆损失定义。

### 4.4 总目标

$$L=L_E+0.1L_R+0.1L_S.$$

- 三项权重在训练开始前固定，不根据验证集或测试集调整。
- 不使用自适应权重、梯度归一化或不确定性加权。

## 5. 训练协议

- 数据：`s0-ieee37-r0p01-20-loguniform` 与 `s0-ieee37-r10-100-loguniform` 两套已登记数据集；核对名称、路径、电阻范围、算例与规模后使用。
- 划分：沿用首轮 `split_by_group` 的最终划分（28 个训练故障族加全部正常样本、6 个验证族、6 个测试族），种子 342，比例 0.15/0.15。
- 标准化：沿用首轮“仅最终训练有效候选、逐节点通道”的仿射重拟合，不重复除以旧标准差。
- 模型与优化器：`GrefPredictor` 保持不变；Adam，学习率 0.001，批大小 64，200 轮；评价模式与 Dropout 口径与首轮一致。
- 选模：验证集严格 Top-1 最大优先，其次能量项最小；测试集只在最终检查点上评价一次。
- 种子：首轮只运行种子 342；不据单种子结果宣称方法有效性。
- 不使用测试集选择 $\tau$、$\lambda$、边权或训练轮次。

## 6. 产物契约

- 输出目录：`code/method-a2/output/gref/three-term-ieee37-{a,b}-seed342/`；日志目录：`code/method-a2/logs/three-term-ieee37-{a,b}-seed342/`。
- 必须保存：`config.json`（含损失定义文本、三项权重、$\tau$ 及其训练侧标定分位数、边权口径、公式版本）、`normalization.npz`、`split.npz`、`dataset_audit.json`、`gate_audit.json`（只作诊断）、逐轮 `train_log.csv`（含三项分量）、`model.pt`、`checkpoint-epochNNN.pt`（第 1、50、100、200 轮）、`optimizer.pt`（最佳轮）、`metrics.json`、`predictions.npz`、`shortcut_audit.json`、`report.md`。
- `predictions.npz` 至少保存评分、观测、响应目标、预测响应、候选掩码、节点掩码、位置与测试索引，支持离线复算。
- 候选置换与响应屏蔽审计沿用首轮实现；审计结论写入 `shortcut_audit.json`。

## 7. 训练后的梯度与余弦分析协议

- 固定最佳检查点，覆盖全部 2280 条训练样本，种子 342 固定打乱，批大小 64，共 36 个混合批次；分别测量关闭与启用 Dropout 两种模式，不执行优化器更新，并核验参数逐元素不变。
- 模块分组：编码器为 `temporal.*` 与 `encoder_gnn.*`；解码器为 `decoder_input.*`、`decoder_layers.*` 与 `output.*`；旧评分头已为无参数恒等层。
- 逐批统计：三项加权梯度（$\lambda_i g_i$）的范数、RMS、份额；两两余弦 $\cos(g_E,g_R)$、$\cos(g_E,g_S)$、$\cos(g_R,g_S)$；合成梯度 $g_\Sigma$ 上的投影 $P_i=g_i^\top g_\Sigma/\|g_\Sigma\|^2$；总范数。
- 汇总口径：逐批结果的中位数与四分位、最小与最大值、负余弦批次占比；同时报告能量项与排序项、平滑项的范数比。
- 补充分析：（1）用训练时保存的 Adam 一阶与二阶矩，把其中一项梯度单独代入更新式，得到单目标反事实 Adam 更新方向，报告三项更新方向的两两余弦与位移范数，并明确其为线性化近似；（2）误差的图频分解：报告 $\mathrm{tr}(f^\top f)$、$f^\top Lf$、Rayleigh 商 $f^\top Lf/(f^\top f)$ 与图高频子空间能量占比，检验平滑项是否主要作用于误差的空间变化分量。
- 预先登记的模式预期：$\cos(g_E,g_R)<0$ 在解码器上占多数；$\cos(g_E,g_S)\ge0$ 且通常显著小于 1；$\cos(g_R,g_S)$ 无强先验，按实测报告。若实测与之矛盾，按实测修正解释，不修改判据。
- 与首轮 $L_{loc}+0.1L_{family}$ 在同一批次、同一模块定义下的结果并列报告；因检查点不同，只作描述性对照。

## 8. 判定与解释边界

- 本实验只检验“更换损失项后，三项梯度结构、重构质量与定位指标如何变化”，不单独宣称方法有效。
- 主要读数为：测试 Top-1/3/5、能量项与 SmoothL1 重构误差、图平滑值、排序 KL、三项梯度范数与两两余弦。
- 已知风险：既有受控实验显示列表级排序的 Top-1 低于交叉熵（49.31% 对 63.89%），因此本轮 Top-1 可能低于首轮；该结果只作为“排序项替代交叉熵”的代价记录，不回头用测试集调参。
- 单种子、固定拓扑、干净完整观测、单相接地、每组 6 个独立测试事件族，均为结论边界。

## 9. 停止条件

- 训练损失出现非有限值，或某项梯度范数长期为零（编码器与解码器上同时为零），停止并报告实现问题。
- 验证集上三项梯度份额长期由单项占据超过 95%，停止其解释性主张，转入独立的权重受控实验。
- 测试集只在最终检查点评价一次；不因测试结果追加训练或选参。

## 10. 复现命令与验收清单

```powershell
python code/method-a2/scripts/train_gref.py `
  --data-dir code/method-a2/data/s0-ieee37-r0p01-20-loguniform `
  --output-dir code/method-a2/output/gref/three-term-ieee37-a-seed342 `
  --loss-mode three-term --lambda-energy 1 --lambda-rank 0.1 --lambda-smooth 0.1 `
  --energy-tolerance 2.231209336901537e-6 `
  --tolerance-source "首轮 A 标定 calibration-a.json，GRReF 下仅用于门控诊断" `
  --temperature 0.012815295718610287 `
  --epochs 200 --batch-size 64 --learning-rate 0.001 --seed 342 --device cuda `
  --save-epochs 1,50,100,200 --no-terminal-progress
```

注：GRReF 下 `--energy-tolerance` 与 `--temperature` 不进入训练目标，仅保留给门控诊断审计与命令行兼容；其取值沿用首轮标定产物。B 数据对应把容差、温度与输入输出目录替换为 B 组标定值与 `three-term-ieee37-b-seed342`。

```powershell
python code/method-a2/scripts/analyze_gref_three_term_gradients.py `
  --run-dir code/method-a2/output/gref/three-term-ieee37-a-seed342 `
  --data-dir code/method-a2/data/s0-ieee37-r0p01-20-loguniform `
  --output code/method-a2/output/gref/three-term-gradient-a-20261104
```

验收清单：

- 两套数据的 `config.json` 均记录三项权重、$\tau$ 及其标定分位数、边权口径与公式版本。
- `train_log.csv` 含三项分量与总损失，且不缺失任何一轮。
- 最佳检查点、`optimizer.pt`、四个分段检查点、`predictions.npz` 与 `metrics.json` 齐全。
- 梯度分析产物包含 36 批次、两种 Dropout 模式、两个模块的逐批记录与汇总统计。
- 测试集只有一次最终评价；任何后续配置变更均须另立预登记实验。
