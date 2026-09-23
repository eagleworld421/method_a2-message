<!-- 摘要：规定 Method-A2 最小可辨识性实验的数据资格审计（D0）、最近邻一致性实验（I1）与直接映射 baseline（I2）三步流程，以及符号定义、距离口径、划分规则、判定逻辑与输出目录；本文为实验执行前的计划文档，不含实验结果。 -->

<!-- 路径口径说明：本文正文中的路径为 Method-A2 在 method_a 仓库运行时的原始口径（形如 `code/method-a1/...`），按本次迁移约定保留原文，未作改写。本工作区 method_a2 只迁入 A2 文档，未迁入代码与运行产物，故这些路径在本工作区内不解析；对应代码、数据集与产物仍位于 method_a 仓库的 `code/method-a1/` 下。后续如需在本工作区运行，应迁入 `code/method-a2/` 并在迁入时把路径改写为同名相对路径。 -->

# Method-A2：最小可辨识性实验计划

> 日期：2026-09-22  
> 状态：Working Plan  
> 适用范围：Method-A2 当前 S0 / 单拓扑 / 全观测条件  
> 优先级：若本文与代码/Git、正式规范、已批准设计或最新实验报告冲突，以代码/Git 和正式项目文档为准。  
> 本文只研究可辨识性，不设计 `E_\theta`、decoder、TCN、GNN、Transformer 或新的训练损失。

---

## 1. 实验目标

当前需要回答的问题是：

\[
\boxed{
\text{仅从单个事实响应 }X_m,\text{ 是否能够稳定确定同一事件的完整反事实响应族 }\mathcal S_m^{CF}?
}
\]

物理事实响应定义为：

\[
X_m=\psi\left(G_m^{true},H_m,C_m,K_m^{true}\right).
\]

对同一个事件 \(m\)，把故障位置干预为候选 \(k\) 后得到：

\[
S^{CF}_{m,k}
=
\psi\left(
G_m^{true},
H_m,
C_m,
\operatorname{do}(K_m=k)
\right).
\]

完整反事实响应族为：

\[
\boxed{
\mathcal S_m^{CF}
=
\left\{
S^{CF}_{m,k}
\right\}_{k\in\mathcal K_m}
}
\]

因此当前研究的目标映射是：

\[
\boxed{
F:X_m\rightarrow \mathcal S_m^{CF}.
}
\]

本实验不试图通过有限数据“数学证明”该映射在整个连续物理空间中唯一存在，而只回答两个实际问题：

1. 现有数据中，是否存在明显的“\(X\) 很相近，但完整反事实响应族差异很大”的反例；
2. 在独立测试事件上，仅使用 \(X\) 是否已经能够通过一个简单直接的映射恢复较接近的完整响应族。

为尽量减少实验量，本计划只包含：

\[
\boxed{
D0\ \text{数据资格审计}
\rightarrow
I1\ \text{最近邻一致性实验}
\rightarrow
I2\ \text{直接映射 baseline}
}
\]

其中 D0 不算正式实验，只是防止后续实验建立在错误的数据配对上。

---

# 2. 符号与具体物理意义

本节中的符号必须在代码、日志和报告中保持一致，不得另行复用。

## 2.1 事件级物理变量

### \(m\)

事件索引。

一个事件 \(m\) 表示一组固定的真实物理条件，包括真实拓扑、运行背景、除位置以外的故障条件，以及实际发生的故障位置。

---

### \(G_m^{true}\)

事件 \(m\) 的真实电网拓扑。

当前最小实验只在 S0 条件下进行，因此应固定：

\[
G_m^{obs}=G_m^{true}.
\]

不同拓扑不得混入本轮可辨识性实验。

---

### \(H_m\)

事件 \(m\) 的运行背景状态，例如：

- 负荷状态；
- 工作点；
- 故障发生前的系统运行状态。

\(H_m\) 是物理定义中的变量，但本实验不要求模型显式恢复 \(H_m\)。

---

### \(C_m\)

事件 \(m\) 中“除故障位置之外”的故障事件条件，例如：

- 故障类型；
- 故障相别；
- 故障阻抗；
- 故障发生时刻；
- 其他在同一事件的候选位置遍历中保持不变的物理条件。

**重要：本文中的 \(C_m\) 始终表示故障事件条件，不表示候选数量。**

候选数量统一写成：

\[
|\mathcal K_m|.
\]

---

### \(K_m^{true}\)

事件 \(m\) 的真实故障位置。

它只能用于：

- 仿真生成；
- 数据一致性检查；
- 离线结果分析。

它不得作为 I1 或 I2 的预测输入。

---

### \(k\)

一个候选故障位置。

---

### \(\mathcal K_m\)

事件 \(m\) 的有效候选集合。

本轮实验要求不同事件能够使用同一套可对齐的候选语义。如果 S0 数据中所有事件候选集合一致，则记为：

\[
\mathcal K.
\]

若候选集合不一致，则只能在相同 candidate mask 的事件之间比较，不能直接把不同候选轴的响应族求距离。

---

## 2.2 响应变量

### \(X_m\)

事件 \(m\) 的事实观测响应：

\[
X_m
=
\psi(
G_m^{true},
H_m,
C_m,
K_m^{true}
).
\]

当前数据含义沿用 Method-A2 已确认定义：

\[
X_m\in\mathbb R^{N\times T\times 6}.
\]

其中：

- \(N\)：节点数；
- \(T\)：时间采样点数；
- 6：三相复电压拆分后的 6 个实值通道。

建议通道语义在代码中从现有数据契约读取，不要重新假设顺序。若现有契约确认为常见顺序，则可能对应：

\[
[
\Re(V_a),\Im(V_a),
\Re(V_b),\Im(V_b),
\Re(V_c),\Im(V_c)
].
\]

实验报告必须打印实际通道名称及顺序。

---

### \(S^{CF}_{m,k}\)

事件 \(m\) 在候选故障位置 \(k\) 下的真实反事实响应：

\[
S^{CF}_{m,k}
=
\psi(
G_m^{true},
H_m,
C_m,
\operatorname{do}(K_m=k)
).
\]

其数据含义和单个 \(X_m\) 相同：

\[
S^{CF}_{m,k}
\in
\mathbb R^{N\times T\times6}.
\]

也就是说，它仍然是所有节点、所有时间点的三相电压 Re/Im 响应，只是故障位置从事实位置改为了候选 \(k\)。

---

### \(\mathcal S_m^{CF}\)

事件 \(m\) 的完整反事实响应族：

\[
\mathcal S_m^{CF}
=
\left\{
S^{CF}_{m,k}
\right\}_{k\in\mathcal K_m}.
\]

如果候选轴共有 \(|\mathcal K_m|\) 个候选，则实现中可以组织为：

\[
\mathcal S_m^{CF}
\in
\mathbb R^{
|\mathcal K_m|
\times N\times T\times6
}.
\]

候选轴必须按候选物理身份对齐。

严禁仅按数组位置比较两个事件，而不先验证该位置对应的是同一个候选母线/正常候选。

---

## 2.3 距离

正式 Method-A2 文档中的响应差异是带节点尺度归一化的平方误差。为了在本实验中避免“平方量”和“距离”混用，统一定义：

\[
\rho(A,B)=\sqrt{d_S(A,B)}.
\]

其中 \(d_S\) 必须直接复用项目中正式定义和实际实现，不在本实验中另造新的标准化方法。

因此：

### 事实响应距离

\[
\boxed{
D_X(i,j)
=
\rho(X_i,X_j)
}
\]

表示两个事件的事实响应有多相近。

### 完整反事实响应族距离

候选对齐后定义：

\[
\boxed{
D_{\mathcal S}(i,j)
=
\sqrt{
\frac{1}{|\mathcal K|}
\sum_{k\in\mathcal K}
d_S(
S^{CF}_{i,k},
S^{CF}_{j,k}
)
}
}
\]

它表示两个事件的**完整反事实响应族整体差异**。

注意：

\[
D_X
\]

比较的是一个事实响应；

\[
D_{\mathcal S}
\]

比较的是整个候选响应族。

二者不能混为同一个指标。

---

# 3. D0：现有数据资格审计

## 3.1 目的

在正式实验前确认：

> 现有数据是否真的能为每一个事件 \(m\) 提供一个事实响应 \(X_m\) 和与它严格配对的完整反事实响应族 \(\mathcal S_m^{CF}\)。

此前实验记录中提到 paired-v1 为每个物理事件保存了真实阻抗条件下的全候选配对响应，但本实验不得只依赖文档描述，必须以实际代码、manifest 和数组为准。

---

## 3.2 数据搜索顺序

实现脚本应按以下顺序查找数据，而不是重新生成 OpenDSS 数据：

1. 查找 paired-v1 或当前等价的数据集；
2. 查找其 manifest / metadata；
3. 确认是否能通过事件 ID 找到：
   - 事实响应；
   - 全候选反事实响应；
   - 候选身份；
   - 真实故障位置；
   - 工况；
   - 故障类型；
   - 相别；
   - 故障阻抗；
   - topology id；
   - observation mask；
4. 只有现有数据不满足要求时，才报告缺失项，不得静默从不同实验数据集中拼接响应族。

---

## 3.3 每个事件必须构造成统一记录

推荐内部记录结构：

```text
EventRecord
  event_id
  family_key
  topology_id
  candidate_ids
  candidate_mask
  true_location
  condition_metadata
  fault_metadata
  x
  s_cf
```

其中：

```text
x.shape    == [N, T, 6]
s_cf.shape == [K, N, T, 6]
```

这里的 `K` 是有效候选数，不得使用变量名 `C`，避免和物理条件 \(C_m\) 混淆。

---

## 3.4 family_key 的定义

为了防止同一个完整反事实族同时进入训练集和测试集，应构造：

```text
family_key =
    所有在候选位置干预过程中保持不变的物理条件
```

原则上对应：

\[
(G_m^{true},H_m,C_m).
\]

实际实现时，应由数据中真实存在的字段组成，例如：

```text
topology
load / operating-condition identifiers
fault type
fault phase
fault impedance
fault time
其他固定事件参数
```

**不得包含 \(K_m^{true}\)。**

因为改变真实故障位置并不会改变由同一个 \((H,C)\) 定义的理论完整响应族。

如果现有数据无法可靠构造 `family_key`，脚本必须在审计结果中明确写出这一限制。

---

## 3.5 必须执行的检查

### 检查 D0-1：shape

所有有效事件必须满足：

```text
x.ndim == 3
s_cf.ndim == 4
x.shape == s_cf[k].shape
```

且最后一维必须为 6。

---

### 检查 D0-2：候选轴

检查：

```text
len(candidate_ids) == s_cf.shape[0]
```

并验证所有纳入比较的事件：

```text
candidate_ids_i == candidate_ids_j
```

或者拥有完全相同的有效 candidate mask。

若候选轴不能严格对齐，则这些事件不能直接进入 I1。

---

### 检查 D0-3：事实响应与真实候选响应一致

找到真实故障位置在 candidate axis 中对应的位置 `k_true_index`，计算：

\[
e_m=
\rho(
X_m,
S^{CF}_{m,K_m^{true}}
).
\]

输出：

```text
median(e_m)
p95(e_m)
max(e_m)
```

如果二者来自同一次仿真保存，理论上应接近 0。

如果来自独立 OpenDSS 重复求解，则允许存在数值差异，但必须在报告中说明。

若大量事件出现明显不一致，应停止实验，先修复数据配对问题。

---

### 检查 D0-4：同一响应族内部固定变量

对同一 `event_id/family_key` 的所有候选：

- topology 不变；
- 工况不变；
- fault type 不变；
- fault phase 不变；
- fault impedance 不变；
- fault time 不变；
- 仅 candidate location 改变。

若发现其他物理条件也随 \(k\) 改变，则当前数据不符合本文对 \(\mathcal S_m^{CF}\) 的定义。

---

### 检查 D0-5：排除数值重复和数据重复

建立两个标识：

```text
event_id
family_key
```

并识别：

- 完全重复数组；
- 同一个物理场景的 OpenDSS 独立重复；
- 同一 family 的多个事实锚点。

这些样本可以保留用于审计，但在 I1 搜索最近邻时：

```text
禁止把 same family_key 的样本互相当作最近邻证据
```

否则会人为得到大量“近 X -> 近 family”的成功结果。

---

## 3.6 D0 输出

必须生成：

```text
data_audit.json
eligible_events.csv
invalid_events.csv
```

`data_audit.json` 至少包含：

```json
{
  "num_raw_events": 0,
  "num_eligible_events": 0,
  "num_invalid_events": 0,
  "x_shape": [],
  "s_cf_shape": [],
  "candidate_ids": [],
  "num_unique_family_keys": 0,
  "x_true_cf_error_median": 0,
  "x_true_cf_error_p95": 0,
  "x_true_cf_error_max": 0,
  "channel_names": [],
  "normalization_source": ""
}
```

---

## 3.7 D0 通过条件

只有满足下面三条才继续：

1. 能为大多数目标事件恢复完整的 \((X_m,\mathcal S_m^{CF})\)；
2. 候选轴能严格对齐；
3. \(X_m\) 与真实候选的 \(S^{CF}_{m,K_m^{true}}\) 配对正确。

若失败，实验结束，结论应写成：

> 当前数据不满足可辨识性实验的数据资格要求，尚不能判断 \(X\rightarrow\mathcal S^{CF}\)。

---

# 4. I1：最近邻一致性实验

## 4.1 目的

直接检查：

\[
\boxed{
X_i\approx X_j
\quad\Longrightarrow\quad
\mathcal S_i^{CF}\approx\mathcal S_j^{CF}
}
\]

在现有数据中是否成立。

这是本计划中最重要的实验。

---

## 4.2 输入数据

仅使用 D0 标记为 eligible 的事件。

第一轮严格限制为：

- 同一个真实 topology；
- 全观测；
- 同一个响应定义；
- 同一个标准化方式；
- 相同候选轴；
- S0 条件。

不得混入 S1 错误拓扑或 S2 部分观测。

---

## 4.3 距离实现

对任意两个事件 \(i,j\)：

```python
dx = sqrt(d_S(X[i], X[j]))
```

完整响应族：

```python
ds_sq = 0
for k in candidate_axis:
    ds_sq += d_S(S_cf[i, k], S_cf[j, k])

ds = sqrt(ds_sq / num_candidates)
```

注意：

- `d_S` 必须调用项目现有正式实现，或者严格复刻正式公式；
- 不允许在脚本里临时换成普通未标准化 MSE；
- 不允许把 candidate 轴和 node 轴混在一起后直接算一个未经核对的整体 MSE；
- `candidate_ids` 必须先对齐。

---

## 4.4 最近邻搜索

对每个 anchor 事件 \(i\)，候选邻居集合定义为：

```text
所有 eligible 事件 j
且：
j != i
family_key[j] != family_key[i]
topology[j] == topology[i]
candidate_ids[j] == candidate_ids[i]
```

然后：

\[
\boxed{
j^*(i)
=
\arg\min_j D_X(i,j)
}
\]

保存：

\[
D_X(i,j^*),
\qquad
D_{\mathcal S}(i,j^*).
\]

如果事件数不超过约 5000，可以直接分块计算全 pair 距离矩阵。

不要为了方便只随机抽少量 pair，因为最近邻实验的关键就是尽可能找到真正最接近的 \(X\)。

---

## 4.5 对每个 anchor 必须保存的字段

生成：

```text
nearest_neighbor_pairs.csv
```

每行至少包括：

```text
anchor_event_id
neighbor_event_id
anchor_family_key
neighbor_family_key

dx
ds

anchor_true_location
neighbor_true_location

anchor_fault_impedance
neighbor_fault_impedance

anchor_fault_type
neighbor_fault_type

anchor_fault_phase
neighbor_fault_phase

anchor_condition_id
neighbor_condition_id
```

如果某个字段不存在，不允许猜测，填 `NA` 并在审计中说明。

---

## 4.6 核心统计

### 指标 I1-A：最近邻输入距离

报告：

```text
DX median
DX p10
DX p05
DX p01
DX min
```

---

### 指标 I1-B：对应的完整响应族距离

对同一批最近邻报告：

```text
DS median
DS p90
DS p95
DS max
```

---

### 指标 I1-C：输入越近时，目标是否同步变近

把所有 anchor 按 `dx` 从小到大排序，分别查看：

```text
最近的 1%
最近的 5%
最近的 10%
全部最近邻
```

对每组报告：

```text
median(ds)
p90(ds)
p95(ds)
```

核心问题只有一个：

> 当 `dx` 已经进入最小的 1% 或 5% 时，`ds` 是否也明显进入较低水平？

---

## 4.7 必须生成的图

### 图 1：核心散点图

横轴：

\[
D_X
\]

纵轴：

\[
D_{\mathcal S}.
\]

每个点代表一个 anchor 及其输入最近邻。

文件：

```text
i1_dx_vs_family_distance.png
```

---

### 图 2：按输入距离分桶

把最近邻 pair 按 `dx` 分成 10 个等样本数 bin。

每个 bin 画：

```text
median(ds)
p25(ds)
p75(ds)
```

文件：

```text
i1_binned_family_distance.png
```

它用于直接观察：

\[
D_X\downarrow
\]

时，

\[
D_{\mathcal S}
\]

是否随之下降。

---

## 4.8 反例输出

自动输出最值得人工检查的 20 个 pair。

排序规则：

1. `dx` 越小越优先；
2. 在 `dx` 很小的样本中，`ds` 越大越优先。

推荐做法：

```text
先取 dx 最小的 5% pair
再按 ds 降序取前 20
```

生成：

```text
i1_top20_counterexample_candidates.csv
```

并保存对应：

```text
X_i
X_j
S_cf_i
S_cf_j
```

的引用路径或数组索引，不复制大数组。

---

## 4.9 I1 如何解释

### 支持当前猜想

如果观察到：

\[
D_X\text{ 越小}
\Rightarrow
D_{\mathcal S}\text{ 通常也越小},
\]

并且在最接近的 1%/5% 输入 pair 中没有大量“很近的 \(X\) + 很远的 family”，则：

> 当前数据支持 \(X\rightarrow\mathcal S^{CF}\) 至少具有局部一致性。

这不是数学证明，但说明现有数据中没有明显反例。

---

### 反对当前猜想

如果大量存在：

\[
D_X\text{ 已处于最小范围},
\]

但：

\[
D_{\mathcal S}
\]

仍与普通事件对一样大，甚至非常大，则说明：

> 单个 \(X\) 对完整反事实响应族的约束可能不足。

此时应优先检查 top counterexample 的物理元数据，而不是立即修改模型结构。

---

# 5. I2：直接 \(X\rightarrow\mathcal S^{CF}\) baseline

## 5.1 目的

I1 只检查局部关系。

I2 再回答一个非常实际的问题：

> 在完全不给模型 \(H,C,K^{true}\) 的情况下，只输入 \(X\)，一个普通的直接监督映射能否在未见事件上恢复完整反事实响应族？

I2 不是 Method-A2 最终模型设计，也不用于选择 TCN/GNN/Transformer。

它只是可学习性的 sanity check。

---

## 5.2 数据划分

### 划分单位

必须按：

```text
family_key
```

划分，而不是按 candidate response 划分。

推荐：

```text
train 70%
validation 15%
test 15%
```

固定随机种子，例如：

```text
seed = 342
```

同一个 `family_key` 的所有数据必须进入同一个 split。

---

## 5.3 输入和目标

输入只能是：

\[
X_m.
\]

禁止输入：

- \(H_m\)；
- \(C_m\)；
- 故障阻抗；
- \(K_m^{true}\)；
- candidate ID；
- node ID 旁路特征；
- 真实标签；
- 测试集统计量。

目标是完整：

\[
\mathcal S_m^{CF}.
\]

实现前将候选轴按 canonical `candidate_ids` 排列。

---

## 5.4 数据表示

使用项目正式响应标准化。

然后：

```python
x_vec = X.reshape(-1)
y_vec = S_cf.reshape(-1)
```

即：

\[
x_m\in\mathbb R^{N T 6},
\]

\[
y_m\in\mathbb R^{|\mathcal K| N T 6}.
\]

这里 flatten 只是为了构造一个不带结构假设的诊断 baseline，不改变物理定义。

---

## 5.5 baseline 的固定实现

为了避免实验人员自行挑模型导致结论漂移，默认只实现一个全连接 MLP baseline：

```text
Input: D = N*T*6

Linear(D, 1024)
GELU
Linear(1024, 1024)
GELU
Linear(1024, P)

P = |K|*N*T*6
```

如果输出维度导致显存不足，只允许把 hidden width 从 1024 降为 512；不得修改成 GNN、TCN、Transformer 或 candidate-specific decoder。

这是一个纯粹的：

\[
X\rightarrow\text{完整 family tensor}
\]

函数拟合器。

---

## 5.6 训练规则

优化器：

```text
AdamW
lr = 1e-3
weight_decay = 1e-4
```

batch size：

```text
min(64, 可用训练样本数)
```

最大 epoch：

```text
500
```

early stopping：

```text
监控 validation family RMSE
patience = 40
```

随机种子：

```text
342
```

损失：

直接使用完整响应族上的正式标准化平方误差均值：

\[
\mathcal L
=
\frac1{|\mathcal K|}
\sum_k
d_S(
\widehat S^{CF}_{m,k},
S^{CF}_{m,k}
).
\]

**不得加入任何额外 ranking loss、contrastive loss、identity loss 或其他 Method-A2 结构性损失。**

---

## 5.7 训练前必须先做“小样本过拟合检查”

随机从 train 中固定选 32 个 `family_key`。

只用这 32 个样本训练同一 baseline。

目的不是评价泛化，而是确认：

- 输出 shape 正确；
- loss 实现正确；
- 模型容量基本足够；
- 优化过程能够拟合训练样本。

生成：

```text
i2_overfit32_curve.csv
i2_overfit32_summary.json
```

如果连 32 个样本都无法明显降低训练误差，则：

> I2 baseline 本身无效，不能据此讨论可辨识性。

必须先排查实现。

---

## 5.8 正式评价

正式训练后分别报告：

### family RMSE

\[
E_{\text{family}}
=
D_{\mathcal S}(
\widehat{\mathcal S}_m,
\mathcal S_m
).
\]

报告 train / validation / test：

```text
mean
median
p90
p95
```

---

### candidate-wise RMSE

对每个候选 \(k\)：

\[
E_k
=
\rho(
\widehat S^{CF}_{m,k},
S^{CF}_{m,k}
).
\]

检查是否只有少数候选预测较差。

---

### 不以 Top-1 作为主要判定

可以附带计算定位结果，但本实验主要研究：

\[
X\rightarrow\mathcal S^{CF}.
\]

因此不能出现：

> “Top-1 高，所以可辨识”

这种结论。

主要指标必须是完整 family prediction error。

---

# 6. I1 与 I2 的联合判定

最终不需要复杂统计检验，只使用下面的逻辑。

## 情况 A

I1：

\[
\text{近 }X\rightarrow\text{近 family}
\]

且 I2：

```text
train error 低
test error 也低
```

结论：

\[
\boxed{
\text{当前 S0 数据支持 }
X\rightarrow\mathcal S^{CF}
\text{ 具有稳定的近似可辨识性和可学习性。}
}
\]

这时可以继续讨论 Method-A2 的具体 predictor 设计。

---

## 情况 B

I1：

\[
\text{近 }X\rightarrow\text{近 family}
\]

但 I2：

```text
train 低
test 明显高
```

结论：

> 当前数据没有显示明显的任务级矛盾，但现有样本覆盖或统计泛化能力不足。

不能据此说任务不可辨识。

---

## 情况 C

I1：

\[
\text{近 }X\rightarrow\text{近 family}
\]

但 I2 连 training set 都无法拟合。

结论：

> 首先怀疑 baseline 容量、优化或实现，而不是可辨识性。

尤其要先检查 32-sample overfit test。

---

## 情况 D

I1 中大量出现：

\[
D_X\text{ 很小}
\]

但：

\[
D_{\mathcal S}\text{ 很大},
\]

同时 I2 test family error 也长期较高。

结论：

\[
\boxed{
\text{现有数据强烈质疑单个 }X
\text{ 足以稳定确定完整反事实响应族。}
}
\]

下一步应该分析这些 counterexample 的 \(H,C,K^{true}\) 差异，而不是立刻设计更复杂网络。

---

# 7. 本实验能够和不能够证明什么

## 可以支持的结论

如果 I1、I2 均正向：

> 在当前 S0、当前 IEEE13/现有拓扑、全观测、现有故障条件覆盖和当前数据精度范围内，单个事实响应 \(X_m\) 对完整反事实响应族 \(\mathcal S_m^{CF}\) 具有稳定的经验对应关系，并能够被一个不使用隐藏物理条件的直接监督映射学习。

---

## 不允许写出的结论

即使实验完全成功，也不得写：

> 已数学证明 \(X\) 唯一决定 \(\mathcal S^{CF}\)。

有限数据无法证明整个连续状态空间中的严格唯一性。

也不得外推到：

- S1 错误拓扑；
- S2 部分观测；
- S3 跨拓扑；
- S4 高阻域外条件；
- 真实传感器噪声；
- 当前数据范围之外的故障类型或工况。

---

# 8. 推荐脚本和输出目录

推荐新增独立实验目录，不修改 E0/E1/E4/E5 原始输出：

```text
code/method-a2/
  scripts/
    run_identifiability_data_audit.py
    run_identifiability_nn.py
    run_identifiability_baseline.py

  output/
    identifiability/
      <run-id>/
        data_audit.json
        eligible_events.csv
        invalid_events.csv

        nearest_neighbor_pairs.csv
        i1_top20_counterexample_candidates.csv
        i1_dx_vs_family_distance.png
        i1_binned_family_distance.png
        i1_summary.json

        split_manifest.json

        i2_overfit32_curve.csv
        i2_overfit32_summary.json

        i2_train_curve.csv
        i2_metrics.json
        i2_candidate_metrics.csv

        decision.json
        report.md
```

如果当前项目仍统一放在 `code/method-a1/`，则保持现有项目目录约定，不要为了本实验迁移代码。目录位置以 Git 当前结构为准。

---

# 9. `decision.json` 建议格式

```json
{
  "data_qualified": true,

  "i1": {
    "nearest_input_family_consistency": "support | concern | invalid",
    "num_pairs": 0,
    "num_counterexample_candidates": 0
  },

  "i2": {
    "overfit32_passed": true,
    "train_family_rmse": 0.0,
    "validation_family_rmse": 0.0,
    "test_family_rmse": 0.0,
    "learnability": "support | insufficient | invalid"
  },

  "overall": "support | concern | insufficient | invalid",

  "scope": "S0 only"
}
```

不要设置一个未经数据校准的硬数值阈值来自动宣布“可辨识”。

正式判断必须同时阅读：

1. I1 的最近邻散点与 counterexample；
2. I2 的 train/test family error；
3. D0 数据审计是否可靠。

---

# 10. 最终执行顺序

Codex/agent 实现时严格按照：

```text
Step 1
读取代码、数据契约和 manifest
↓
Step 2
运行 D0 数据资格审计
↓
D0 不通过 -> 停止，不进行 I1/I2
↓
Step 3
冻结 eligible event set 和 candidate axis
↓
Step 4
运行 I1 最近邻一致性
↓
Step 5
保存 top-20 counterexample candidates
↓
Step 6
按 family_key 构造 train/val/test split
↓
Step 7
先运行 I2 32-sample overfit sanity check
↓
未通过 -> I2 无效，先修实现
↓
Step 8
运行完整 I2 baseline
↓
Step 9
联合 I1 + I2 生成 decision.json 和 report.md
```

---

# 11. 最小实验的核心逻辑

本计划最终只依赖两个问题：

### I1

\[
\boxed{
\text{事实响应最相似的两个不同物理事件，
它们的完整反事实响应族是否也相似？}
}
\]

### I2

\[
\boxed{
\text{只给 }X_m\text{，不提供 }H_m,C_m,K_m^{true},
\text{是否能够在未见事件上直接预测完整 }\mathcal S_m^{CF}?
}
\]

如果这两个问题都得到正面结果，才有充分理由继续讨论：

\[
\text{如何设计 Method-A2 predictor。}
\]

如果 I1 已经出现大量稳定反例，那么继续增加 predictor 复杂度之前，应先重新检查：

\[
\boxed{
X_m\text{ 是否真的包含生成完整反事实响应族所需的信息。}
}
\]

---

## 12. 本文依据

本文的符号与问题定义依据当前 Working Context：

- `Method-A2-核心公式.md`
- `Method-A2-chat-handoff.md`

现有实验数据可用性判断还参考：

- `method-a1-decision-process-summary(1).md`
- `method-a1-e0cov-e1-result-summary.md`
- `method-a1-e4a0-result-summary.md`
- `method-a1-e5-result-summary.md`

其中已有实验只作为数据来源和前置证据；本文没有把 E0/E1/E4/E5 的结果重新解释为“已经证明可辨识”。
