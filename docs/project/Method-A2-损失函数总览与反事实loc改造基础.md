<!-- 摘要：统一记录 Method-A2 当前代码和实验中出现过的训练损失、评价损失与计划损失。文档按旧直接候选评分、Node Marker MVP、反事实候选评分、受控监督对比和梯度消融分组，给出公式、训练目标、实际作用路径、权重、已知缺陷及是否可用于后续反事实 loc 改造；明确区分实际参与反向传播的损失与仅用于离线评价的物理距离指标。文档结论是：旧 loc 具有最强的 Top-1 直接驱动但存在分类捷径，反事实 list 能恢复连续物理距离却不能直接提供足够强的第一名间隔；后续 loc 应把评分绑定到预测反事实响应能量，并使用可分辨性门控，而不能仅把 rank-pair 改名为能量损失。 -->

# Method-A2 损失函数总览与反事实 `loc` 改造基础

## 1. 文档目的与统一记号

本文记录项目中已经实现、训练使用、消融使用或作为评价指标出现过的损失函数。文档同时为后续把 `loc` 改造成反事实监督损失提供共同记号和缺陷清单。

记事实观测为 (X_m)，候选 (k) 的真实反事实响应为 (S^{CF}_{m,k})，模型预测响应为 (widehat S^{CF}_{m,k})，模型直接输出的候选分数为 (s_{m,k})。旧直接评分路径约定分数越小越可能是真实候选；新的反事实评分路径也采用低分优先。

真实反事实距离和预测反事实距离分别记为：

\[
d^{CF}_{m,k}=D(X_m,S^{CF}_{m,k}),
\qquad
\widehat d_{m,k}=D(X_m,\widehat S^{CF}_{m,k}).
\]

两者用途不同：(d^{CF}_{m,k}) 是训练期的物理教师或离线评价量，(widehat d_{m,k}) 是模型部署时可计算的候选能量。除非特别说明，`candidate_mask` 只保留物理上可行候选，正常候选不参与故障母线定位排序。

## 2. 旧直接候选评分路径

代码入口为 `code/method-a2/src/model/losses.py` 与 `code/method-a2/src/training/marker_training.py`。完整训练目标为：

\[
\mathcal L_{\mathrm{old}}
=
\lambda_{\mathrm{loc}}\mathcal L_{\mathrm{loc}}
+\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{rank}}
+\lambda_{\mathrm{response}}\mathcal L_{\mathrm{response}},
\]

默认权重为 (1.0、0.1、0.1)。

### 2.1 `loc`：候选分类损失

真实位置标签为 (y_m)，候选有效掩码为 (M_{m,k})。由于低分表示更可能，代码使用 (-s_{m,k}) 作为分类 logits：

\[
\mathcal L_{\mathrm{loc}}
=-
\frac{1}{|\mathcal B_f|}
\sum_{m\in\mathcal B_f}
\log
\frac{\exp(-s_{m,y_m}/\tau)}
{\sum_{k:M_{m,k}=1}\exp(-s_{m,k}/\tau)}.
\]

训练目标是直接把真实位置的候选分数压到最低。它是旧路径 Top-1 的主要驱动力，能形成较大的真实候选 margin。

主要缺陷是候选评分可以绕过反事实响应，直接学习 (X_m\rightarrow y_m) 的分类映射。只要共享表示或候选标记含有位置信息，分类损失就能下降，即使 (widehat S^{CF}_{m,k}) 不具有物理意义。

### 2.2 `rank`：真实位置对错误候选的成对排序损失

对真实位置 (y_m) 与每个有效错误候选 (k) 使用 softplus margin：

\[
\mathcal L_{\mathrm{rank}}
=
\frac{1}{|\mathcal P|}
\sum_{(m,k)\in\mathcal P}
\operatorname{softplus}
\left(
\frac{s_{m,y_m}-s_{m,k}+\mu}{\tau}
\right).
\]

训练目标是让真实候选分数低于错误候选至少一个人为设定的 margin (mu)。它强化 Top-1，但没有使用真实反事实响应来决定错误候选的物理难度，所有错误候选通常受到相同形式的约束。

它与 `loc` 共同形成强分类监督，可能放大候选编号或拓扑位置捷径；固定 margin 还会强迫物理上近似等价的候选产生人工分离。

### 2.3 `response`：反事实响应重构辅助损失

实际旧路径使用候选掩码下的 Smooth L1：

\[
\mathcal L_{\mathrm{response}}
=
\frac{\sum_{m,k,n,t,f}M_{m,k}\,
\operatorname{SmoothL1}(\widehat S^{CF}_{m,k,n,t,f},S^{CF}_{m,k,n,t,f})}
{\sum_{m,k,n,t,f}M_{m,k}}.
\]

训练目标是预测整个反事实响应族。它对评分头没有直接梯度，主要更新响应解码器，并在训练早期通过共享编码器间接影响候选评分。旧梯度分析显示其后期共享梯度占比很低，因此不能把它视为旧定位性能的主要驱动。

缺陷包括响应元素数量多导致损失尺度与定位项不同，以及对共享表示的间接作用可能与 `loc`、`rank` 产生早期方向冲突。它本身不保证候选第一名正确。

## 3. Node Marker MVP 路径

Node Marker MVP 的训练入口仍调用 `compute_marker_losses`，实际反向传播使用 `family`、`loc`、`rank` 三项：

\[
\mathcal L_{\mathrm{MVP}}
=
\lambda_{\mathrm{loc}}\mathcal L_{\mathrm{loc}}
+\lambda_{\mathrm{rank}}\mathcal L_{\mathrm{rank}}
+\lambda_{\mathrm{response}}\mathcal L_{\mathrm{family}}.
\]

### 3.1 `family` 的响应重构变体

无不确定性分支时，`family` 实际调用 `masked_response_loss`，即上节的候选掩码 Smooth L1。代码同时提供 `masked_family_loss`，其形式为：

\[
\mathcal L_{\mathrm{family,MSE}}
=
\frac{\sum M_{m,k,n,t,f}(\widehat S-S^{CF})^2}
{\sum M_{m,k,n,t,f}}.
\]

该 MSE 变体在测试中用于函数行为核验和部分升级实验，并不是当前旧直接评分主训练的默认实现。MSE 对大响应误差更敏感，容易被少数通道或候选主导。

### 3.2 `L_pair`：同组共享表示一致性

对于同一反事实族内、不同事实故障位置的合法配对 (a,b)，节点级共享表示一致性为：

\[
\mathcal L_{\mathrm{pair}}
=
\frac{1}{N d_z}
\left\|Z_{c,a}-Z_{c,b}\right\|_2^2.
\]

训练目标是让同一物理事件族的共享状态表示接近，同时保留候选位置差异交给候选条件解码器。它只有在 `group_id` 能证明同一仿真参数组时才有意义。

已知缺陷是：若把不同物理事件误配成正样本，会直接压平应保留的状态差异；若先把节点表示平均再计算一致性，又会掩盖节点级差异。因此该项不能在没有合法配对证据时强行启用。

### 3.3 `L_list` 在 MVP 中的状态

MVP 评价阶段计算过基于真实和预测反事实距离的列表结构指标，但 `compute_marker_losses` 的实际训练总损失并未把列表项加入反向传播。执行报告显示该列表项在对应运行中几乎不影响总损失，因此它不能被当作 MVP 已验证有效的训练监督。

## 4. 预测响应与不确定性变体

### 4.1 异方差响应损失

当模型输出 (log\sigma^2) 时，使用：

\[
\mathcal L_{\mathrm{hetero}}
=
\frac{1}{\sum M}
\sum M
\left[
\frac{(S^{CF}-\mu)^2}{2\exp(\log\sigma^2)}
+\frac{1}{2}\log\sigma^2
\right].
\]

训练目标是同时预测响应均值与候选相关不确定性。它可以表达不同候选的响应预测难度，但方差分支容易通过增大方差降低重构项；必须配合对数方差截断、方差边界和误差—方差相关性检查。

已有阶段实验出现异方差损失取负值的情况，说明当前实现和权重下方差项可能主导目标，不能直接解释为性能提升。

## 5. 并行反事实候选评分路径

代码入口为 `code/method-a2/src/counterfactual_scoring/losses.py`。该路径包含四个目标：

\[
\mathcal L_{\mathrm{CF}}
=
0.1\mathcal L_{\mathrm{family}}
+\mathcal L_{\mathrm{list}}
+\lambda_{\mathrm{value}}\mathcal L_{\mathrm{value}}
+\lambda_{\mathrm{gap}}\mathcal L_{\mathrm{gap}}.
\]

### 5.1 `family`：反事实族重构

该路径使用候选掩码 Smooth L1：

\[
\mathcal L_{\mathrm{family}}
=
\frac{\sum M\operatorname{SmoothL1}(\widehat S^{CF}-S^{CF})}
{\sum M}.
\]

它负责响应族的物理重构，对独立评分头没有直接梯度。

### 5.2 `list`：真实反事实距离列表级 KL

先由真实响应构造距离软分布：

\[
p^*_{m,k}
=
\operatorname{softmax}_k
\left(-\frac{d^{CF}_{m,k}}{\tau_d}
\right),
\]

再让候选分数产生预测分布：

\[
\hat p_{m,k}
=
\operatorname{softmax}_k
\left(-\frac{s_{m,k}}{\tau_s}
\right),
\qquad
\mathcal L_{\mathrm{list}}
=D_{KL}(p^*_m\|\hat p_m).
\]

训练目标是恢复连续物理距离排序。它比旧 `loc` 更少依赖离散位置标签，但在候选真实距离很接近时，目标分布趋于高熵，不能自然形成强 Top-1 margin。

### 5.3 `value`：评分值回归真实距离

\[
\mathcal L_{\mathrm{value}}
=
\frac{1}{\sum M}
\sum_{m,k}M_{m,k}
\operatorname{SmoothL1}
\left(s_{m,k},\frac{d^{CF}_{m,k}}{c_d}\right).
\]

训练目标是让自由评分数值接近归一化真实距离。缺陷是直接暴露于距离尺度和候选间微小间隔；当前实验中它后期主导共享编码器和评分头梯度，却使 Top-K 下降。

### 5.4 `gap`：评分差回归真实距离差

对满足间距阈值的候选对：

\[
\mathcal L_{\mathrm{gap}}
=
\operatorname{SmoothL1}
\left(
s_{m,i}-s_{m,j},
\frac{d^{CF}_{m,i}-d^{CF}_{m,j}}{c_d}
\right).
\]

训练目标是复现候选间物理距离差。缺陷是对小间距和仿真噪声最敏感；阈值只能排除部分近等价候选，不能保证剩余间距足够大。当前消融中 `gap` 后期主导共享梯度，Top-3 和 Top-5 均下降。

## 6. 受控监督对比中的损失组合

为判断 Top-K 下降是否由评分监督替换造成，受控实验在同一个反事实评分模型上使用：

- `old_score`：`family + loc + rank`，代表离散位置监督；
- `list`：`family + list`，代表真实反事实距离列表监督；
- `list_value`：`family + list + value`；
- `list_gap`：`family + list + gap`。

三种子实验显示 `old_score` 的 Top-1/3/5 均值为 63.89%/88.19%/95.49%，`list` 为 49.31%/72.22%/89.58%。但是 `list` 的评分—真实距离 Spearman 均值为 0.8901，高于 `old_score` 的 0.8632。这说明反事实列表项更好地恢复了连续物理结构，却没有产生足够强的离散第一名决策。

因此，不能把 `list` 的 Top-K 下降解释为反事实响应没有信息；更准确的解释是连续物理排序目标与离散根因定位目标存在错配。`value` 和 `gap` 进一步放大了该错配。

## 7. 各损失项缺陷的归纳

- `loc`：Top-1 驱动最强，但可能退化为事实观测到位置的分类模型，缺少反事实物理约束。
- `rank`：强化真实候选 margin，但不区分物理可分辨和近等价错误候选，固定 margin 可能制造人工间隔。
- `response`/`family`：保证反事实响应重构，不能单独保证定位；对评分头没有直接梯度，尺度受响应元素数量影响。
- `pair`：可约束同一物理事件族的共享表示，但依赖严格合法配对，错误配对会压平物理状态差异。
- `heteroscedastic`：可表达响应不确定性，但方差通道可能通过增大方差逃避重构，需要额外边界检查。
- `list`：具有物理排序意义，但候选距离接近时目标分布高熵，Top-1 margin 不足。
- `value`：直观但对距离尺度和小间距敏感，容易主导梯度。
- `gap`：试图恢复物理间距，但最容易受小间距、噪声和阈值选择影响。

## 8. 后续反事实 `loc` 改造的约束

后续 `loc` 改造必须满足三个条件。

第一，候选评分应绑定预测反事实响应能量，例如：

\[
e_{m,k}=D(X_m,\widehat S^{CF}_{m,k}),
\]

而不是继续使用可以脱离响应分支自由变化的候选分类 logit。

第二，真实反事实响应只能提供物理教师：用于确定真实物理候选、选择困难候选和估计可分辨性；不应直接回归所有候选间的精确距离差。

第三，对低于响应噪声地板的候选间距不施加强制分离。否则新的损失虽然名称上是能量定位，数学上仍然只是带固定 margin 的 `rank-pair`，并且会与 `response` 损失对错误候选产生相反梯度。

后续候选方案应优先采用“响应能量 + 真实反事实困难候选加权 + 可分辨性门控”的 `loc` 实现。它的必要验证包括：能量是否确实由 (widehat S^{CF}) 产生、`loc` 与 `response` 的共享梯度余弦是否可接受、Top-1 是否提高、以及真实距离相关性是否保持。若候选在真实响应上不可分辨，任何损失都不能凭空消除 Top-1 上限。

## 9. 代码与文档入口

- 旧损失定义：`code/method-a2/src/model/losses.py`；
- 旧训练组合：`code/method-a2/src/training/marker_training.py`；
- 反事实损失定义：`code/method-a2/src/counterfactual_scoring/losses.py`；
- 反事实受控损失组合：`code/method-a2/src/counterfactual_scoring/controlled_losses.py`；
- 旧路径多损失消融报告：`docs/project/reports/Method-A2-s0-lg-run3多损失消融与梯度分析报告.md`；
- 反事实路径消融报告：`docs/project/reports/Method-A2-s0-lg-run3反事实监督候选评分消融与梯度分析报告.md`；
- 旧监督与反事实监督受控对比报告：`docs/project/reports/Method-A2-反事实监督与旧评分监督受控对比实验报告.md`。
