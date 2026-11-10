<!-- 摘要：规定 GRReF在 A 组数据上的排序温度扫描预登记实验。固定 λ=(1,0.3,0.1)，只改变列表排序温度 τ，取训练侧最近竞争者间隔中位数 τ0=4.0640985e-3 的 1、3、10、35 倍共四点；规定数据划分、训练与选模口径、重构与间隔的并列读数、测试集只评一次、产物清单与停止条件。本文为实验执行前的计划，不含实验结果。 -->

# Method-A2 GRReF A 组排序温度扫描计划

## 1. 背景与目标

[GRReF 实验报告](../reports/Method-A2-GReF三项损失实验报告.md)记录 A 组测试 Top-1 为 53.95%，而重构 SmoothL1 降至 0.00837、评分—真实能量 Spearman 为 0.9703。分析指出的主要嫌疑是排序温度：当前 $\tau=0.1427038386464119$ 由“全部候选对间隔的中位数”标定，约为训练侧最近竞争者间隔中位数 $4.0640985\times10^{-3}$ 的 35 倍。当 $\tau$ 远大于首位两项的间隔时，软目标分布在最关键的首位对比上接近等权，排序项在该处几乎没有梯度。

本实验只检验一个假设：**把 $\tau$ 调小到首位间隔尺度、同时把排序权重从 0.1 提高到 0.3，能否在验证集上提高严格 Top-1**，并同时测量重构质量与间隔压缩的代价。

## 2. 固定条件

- 数据：只使用 `code/method-a2/data/s0-ieee37-r0p01-20-loguniform`（A 组，0.01–20 Ω）。
- 划分与标准化：沿用首轮 `split_by_group` 的 28/6/6 故障族划分与仅训练侧仿射重拟合；种子 342。
- 模型：`GrefPredictor`，结构不变。
- 优化：Adam，学习率 0.001，批大小 64，200 轮，设备 CUDA。
- 损失权重：$\lambda_E=1$、$\lambda_R=0.3$、$\lambda_S=0.1$，与三项损失报告中的 0.1 相比只提高排序权重。
- 选模：验证集严格 Top-1 最大优先，其次能量项最小；测试集只在最终检查点评一次。
- 门控容差与交叉熵温度不进入三项训练目标，沿用 A 组首轮标定值，仅用于门控诊断审计。

## 3. τ 网格与来源

令 $\tau_0=4.0640985\times10^{-3}$ 为训练侧全部故障事件真实候选与最近竞争候选间隔的中位数，来源为 A 组 `gate_audit.json` 的正间隔中位数分位数（训练集、种子 342，与首轮逐位一致）。

网格取 $\tau\in\{1,3,10,35\}\times\tau_0$：

- `tau1x`：$4.0640985\times10^{-3}$；
- `tau3x`：$1.21922955\times10^{-2}$；
- `tau10x`：$4.0640985\times10^{-2}$；
- `tau35x`：$1.422434475\times10^{-1}$。

`tau35x` 与当前全部候选对标定的温度（0.1427038）近似相等，用于在 $\lambda_R=0.3$ 下隔离权重变化的影响。四个点全部由训练侧统计给出，不使用验证集或测试集。

## 4. 读数与判据

每个配置报告：

- 验证集与测试集 Top-1/3/5、平均名次；
- 能量项、排序项、平滑项损失；
- 全候选重构 MSE 与 SmoothL1、共模误差占总误差比例；
- 预测与真实真实候选间隔的中位数及比值；
- 评分—真实能量 Spearman 中位数；
- 三项梯度在编码器与解码器上的份额与两两余弦（仅在进入下一阶段的配置上执行完整 36 批次分析）。

判据按以下顺序：

1. 验证集 Top-1 是否高于当前 A 组GRReF 的 55.70%；
2. 若提高，重构 SmoothL1 与共模占比是否仍显著优于首轮 CE（0.5009 与 90.09%）；
3. 预测间隔是否向真实间隔恢复（当前比值约 0.08）。

测试集结果只用于最终记录，不参与选择。若四个配置的验证 Top-1 均不高于 55.70%，则判定“温度与权重不足以恢复首位间隔”，停止本方向并转入间隔保持型排序项设计。

## 5. 产物

- 输出目录：`code/method-a2/output/gref/three-term-ieee37-a-tau{1x,3x,10x,35x}-seed342/`。
- 日志目录：`code/method-a2/logs/three-term-ieee37-a-tau{1x,3x,10x,35x}-seed342/`。
- 每个目录保存 `config.json`（含 $\tau$、$\lambda$、公式版本与来源）、`normalization.npz`、`split.npz`、`train_log.csv`、`model.pt`、`optimizer.pt`、第 1/50/100/200 轮检查点、`metrics.json`、`validation_metrics.json`、`predictions.npz`、`shortcut_audit.json`、`gate_audit.json` 与 `report.md`。
- 汇总结果写入 `docs/project/reports/` 的后续报告，不在本计划中预填数字。

## 6. 停止条件

- 任一配置出现非有限损失或非有限梯度，立即停止该配置并记录。
- 四个配置验证 Top-1 全部不高于 55.70%，停止本方向。
- 测试集只评一次；不得根据测试结果追加训练、修改 $\tau$ 或 $\lambda$。

## 7. 复现命令

```powershell
python code/method-a2/scripts/train_gref.py `
  --data-dir code/method-a2/data/s0-ieee37-r0p01-20-loguniform `
  --output-dir code/method-a2/output/gref/three-term-ieee37-a-tau1x-seed342 `
  --loss-mode three-term --lambda-energy 1 --lambda-rank 0.3 --lambda-smooth 0.1 `
  --rank-temperature 0.0040640985 `
  --energy-tolerance 2.231209336901537e-6 `
  --tolerance-source "首轮 A 标定 calibration-a.json，GRReF 下仅用于门控诊断" `
  --temperature 0.012815295718610287 `
  --epochs 200 --batch-size 64 --learning-rate 0.001 --seed 342 --device cuda `
  --save-epochs 1,50,100,200 --no-terminal-progress
```

其余三点只替换 `--rank-temperature` 与输出目录。

## 8. 边界

- 单种子 342、单数据集、固定拓扑、干净完整观测、单相接地，每组 6 个独立测试事件族；228 个测试事件下单事件影响约 0.44 个百分点。
- 本实验只检验温度—权重方向，不检验间隔保持型排序项，也不宣称方法有效性。
- 若验证集显示提高但重构明显退化，只记录权衡，不据此宣称两者同时改善。
