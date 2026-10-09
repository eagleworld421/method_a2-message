<!-- 摘要：Method-A2 Node Marker MVP 的执行上下文与实现计划。明确核心公式与 MVP 计划的优先级、候选标记 b_k/t_k 的语义、去除 teacher/QKV/边打分/global_repr 的实现边界、严格反事实配对数据的生成与分割、训练损失、测试验收和阻塞通知规则。本文件用于后续 Agent 执行，不修改正式核心公式。 -->

# Method-A2 Node Marker MVP 执行方案与上下文同步

**文档状态：** 已按方案完成首版代码；真实训练和全量数据验证待执行。  
**适用范围：** Method-A2 的 S0 MVP：单一拓扑、完整观测、故障候选加正常候选。  
**权威定义：** `docs/project/Method-A2-核心公式.md`。  
**实施参考：** `docs/project/plans/Method-A2-节点级共享表示-MVP实施方案.md`。  
**重要限制：** 本轮先试用 Node Marker，不修改核心公式中的十维 `e_k` 定义；实验结果出来后再决定是否正式替换。

## 一、执行前必须保持的上下文

Method-A2 的目标是从事实观测和观测拓扑中得到候选无关的节点级共享表示，再对每个候选位置并行预测反事实响应。共享表示不是显式恢复真实物理变量 `U_m`，也不被解释为可直接观测的物理量。

核心结构保持为：

$$
Z_m=E_\theta(X_m,G_m^{obs},M_m),
$$

$$
\widehat S^{CF}_{m,k}=D_\phi(Z_m,e_k,G_m^{obs},M_m).
$$

同一个事件的全部候选必须复用同一个 `Z_m`；候选变化只能通过 `e_k` 进入解码器。编码器不得读取真实故障位置、真实故障阻抗、真实反事实响应或训练期 `group_id`。

本文件只负责把已确认决策转成可执行接口，不替代核心公式，也不允许悄悄修改核心公式。

## 二、已经确定的设计决策

### 2.1 候选标记 `b_k` 和正常标志 `t_k`

对于 `N` 个真实候选节点，故障候选 `k` 的位置标记为：

$$
b_k\in\{0,1\}^{N},\qquad b_{k,n}=\mathbf 1[n=k].
$$

也就是说，`b_k` 是长度为 `N` 的 one-hot 节点位置向量。它表示“本次反事实假设把故障干预放在哪个节点”，不表示真实故障标签，也不表示候选槽位编号。

正常候选使用：

$$
b_{\varnothing}=\mathbf 0_N,qquad t_{\varnothing}=1.
$$

故障候选使用：

$$
t_k=0.
$$

因此本轮候选条件为：

$$
e_k=(b_k,t_k).
$$

`t_k` 是候选类型位，只区分“故障位置候选”和“正常候选”，不是事件正常与否的观测标签。正常候选的真实响应由仿真数据契约定义为无故障基态窗口；正常候选可以参与响应重构，但不参与故障母线定位排序。

### 2.2 teacher 路径

新 MVP 去除 teacher 路径。新模型不接收真实阻抗、真实故障时刻、真实故障类别或任何其他部署不可得条件，也不计算 teacher response。

旧 `PrivilegedResponseSystem`、旧 teacher checkpoint 和历史 Probe 如需回放，只能作为历史基线资源保留；新模型不得导入旧 teacher 类，不得把旧 teacher 输入迁移到 Node Marker 路径。

### 2.3 图编码与解码

新路径不使用 QKV、不使用候选—节点注意力、不使用可学习边打分，也不计算 `global_repr` 或 `g_m`。

编码器只输出：

$$
Z_m\in\mathbb R^{N\times d_z}.
$$

图消息传递使用固定有效边上的均值聚合，边属性只能通过普通线性投影参与消息，不产生注意力分数。候选解码器使用同一组参数处理所有候选，并在解码开始时把候选查询广播到全部输出节点，随后进行两层候选条件化图消息传递。首版默认 `d_z=32`、候选条件化图层数 `L=2`，不进行额外维度搜索。

### 2.4 反事实配对

当前候选相关随机扰动会破坏严格配对。本轮生成数据时去除 `build_dynamic_window` 中的随机相位、幅值噪声和阻尼扰动，保持相同仿真参数组下不同事实故障位置只改变故障位置 `k`。

同一反事实族共享：负荷倍率、故障类型、故障阻抗和其他已定义的事件参数。不同事实故障位置只能改变候选干预位置。数据生成器必须保存非负 `group_id`，同一 `group_id` 对应同一组共享事件参数。正常样本使用 `group_id=-1`，不参与 `L_pair` 正配对。

训练和测试划分必须按 `group_id` 完成，不能把同一反事实族的不同事实位置拆到不同集合中。

### 2.5 损失

保持核心公式已有的三项损失，不增加新的训练监督：

$$
L_{train}=L_{family}+\lambda_{pair}L_{pair}+\lambda_{list}L_{list}.
$$

- `L_family`：有效候选上的反事实响应重构损失。
- `L_pair`：仅对合法同组事实样本的节点级 `Z_m` 计算一致性；不要求不同候选的预测响应一致。
- `L_list`：只在故障候选集合上，根据真实和预测反事实响应相对事实观测的距离构造列表级排序损失；正常候选不进入故障母线排序。

若数据审计发现不能证明存在合法 `group_id` 配对，则跳过 `L_pair`，执行 `L_family+\lambda_{list}L_{list}`，并在运行配置中记录停用原因。

## 三、不得引入的内容

后续 Agent 不得在新路径中加入以下内容：teacher 分支、特权阻抗输入、训练期真实条件输入、`global_repr`、`g_m`、QKV、候选—节点注意力、可学习边打分、候选编号 embedding lookup、旧十维候选特征、真实故障位置或 `group_id` 作为部署前向输入，以及没有数据依据时人为构造 `L_pair` 正样本。

旧十维候选特征只用于 A 基线回放。Node Marker B 路径只使用 `(b_k,t_k)`。

## 四、实现文件边界与接口

实现前必须再次读取 `code/AGENTS.md` 和 `code/CODE_CONVENTIONS.md`。新代码只放在 `code/method-a2/` 下。

建议文件职责如下：

- `src/model/marker_response_predictor.py`：实现 `MarkerResponsePredictor`，输入 `X_m`、`G_m^{obs}`、`M_m`、候选 Marker，输出 `response` 和 `Z`。
- `src/model/gnn.py`：新增或隔离不含 QKV 和打分的固定均值消息层；不得破坏旧 Probe 所需的历史类。
- `src/model/losses.py`：实现 `L_family`、`L_pair`、`L_list`，不把损失逻辑嵌入模型前向。
- `src/data_generation/dataset_builder.py`：生成共享事件族、禁用随机扰动、保存 `group_id`，并按族划分数据。
- `tests/test_marker_mvp.py`：覆盖接口、梯度、候选和节点置换、正常候选隔离及损失掩码。

模型输出至少应包含：

```text
response: [B,C,N,T,F]
Z:        [B,N,d_z]
```

不得返回或计算 `global_repr`。候选输入为：

```text
candidate_markers: [B,C,N]
candidate_types:   [B,C,1]
candidate_mask:    [B,C]
```

当未显式提供候选张量时，模型可以根据 `N` 自动生成 `N` 个故障候选加一个正常候选，但不能依赖固定 `N=16`。S0 数据的默认形状为 `N=16`、`C=17`、`T=12`、`F=6`。

## 五、执行顺序

1. 审计当前仓库状态、数据文件、OpenDSS 可用性和训练入口；确认未提交的历史文档修改不被覆盖。
2. 先编写 Marker 构造、正常候选语义、候选掩码和 `group_id` 的失败测试。
3. 实现不含 QKV、边打分和 `g_m` 的共享编码器与候选条件化解码器。
4. 实现三项损失及严格的候选、响应和配对掩码。
5. 修改仿真数据生成：去除候选相关随机扰动，按共享事件组生成所有事实故障位置，保存 `group_id`，按族划分 train/test。
6. 使用 Mock 或小批量合成张量完成 forward、backward、有限值、候选置换、节点置换和正常候选隔离测试。
7. 若当前仓库没有可用的真实数据或 OpenDSS 运行依赖，先生成合法仿真数据；若仿真仍无法运行，立即通知用户，不用 Mock 结果冒充物理实验。
8. 在 A 基线和 B Node Marker 模型使用相同数据划分、训练预算和评价口径的前提下运行对照。
9. 运行完成后核验输出形状、参数量、损失记录、`Z` 使用情况、候选排序、配对资格和所有反捷径审计。
10. 只有在代码和实验结果复核后，才讨论是否正式修改核心公式中的 `e_k` 定义；本执行阶段不修改核心公式。

## 六、验收标准

### P0：接口与信息边界

- `b_k` 是节点 one-hot，正常候选为全零 `b` 加 `t=1`。
- 同一事件所有候选共享同一个 `Z_m`。
- 输出形状为 `[B,C,N,T,F]`。
- 新路径不存在 teacher、QKV、边打分和 `global_repr`。
- `K_m^{true}`、真实响应、真实阻抗和 `group_id` 不进入部署前向。
- 候选置换和节点同步置换测试通过。

### P1：数据和训练可运行

- 反事实族的 `group_id` 合法且不跨 train/test。
- 候选相关随机扰动已禁用。
- 有合法配对时启用 `L_pair`；无合法配对时明确停用。
- 响应预测不是全常数，三项损失均能完成有限值 forward/backward。

### P2：方法比较

- B 与 A 使用相同数据划分、候选集合、训练预算和评价指标。
- 报告 `L_family`、Top-1/Top-3、真实候选名次、signed margin、候选间结构保真、`Z` 置换影响和工程开销。
- 是否替换十维候选特征只能依据预先记录的 A/B 结果判断，不能依据单次测试集偶然提升。

## 七、仍需通知用户的阻塞条件

以下情况属于关键阻塞，不能自行猜测：OpenDSS 或必要数据无法运行且无法生成合法仿真数据；无法确认节点 ID 与张量节点轴的映射；无法确定正常候选的无故障基态响应；无法验证同一 `group_id` 的共享参数和无扰动生成；核心公式与实际数据字段发生无法兼容的冲突；或需要修改核心公式才能实现但用户尚未批准。

除此之外的实现细节应直接按本文件、MVP 实施方案和核心公式执行，不再反复请求确认。

## 八、交付物

最终实验产物必须位于 `code/method-a2/output/node-marker-mvp/<run-id>/`，至少包括 `config.json`、`dataset_audit.json`、`unit_tests.txt`、`train_log.csv`、`predictions.npz`、`metrics.json`、`shortcut_audit.json` 和 `report.md`。不能把 checkpoint 目录作为最终交付目录。

本文件是后续 Agent 的执行上下文入口。任何 Agent 开始工作前必须先阅读本文件、[Method-A2-节点级共享表示-MVP实施方案.md](Method-A2-节点级共享表示-MVP实施方案.md) 和 [Method-A2-核心公式.md](../Method-A2-核心公式.md)，再读取代码规范；若发现冲突，停止并报告冲突位置。

## 九、当前执行状态

已完成：

- 新建 Node Marker 模型、无 QKV 均值消息传递、三项损失和最小训练步骤；
- 新增 `scripts/train_marker_mvp.py`，可读取现有 `.npy` 数据并输出最小训练产物；
- 默认关闭波形随机扰动；
- 数据生成器按共享事件族保存 `group_id` 并按族划分；
- 7 项 Node Marker 契约测试通过；
- 使用 `s0-smoke` 完成 1 个 batch 的训练 smoke，并生成 `output/node-marker-mvp/smoke/` 产物；
- 旧 teacher 模型和旧 Probe 路径未被新模型导入。

尚未完成：

- OpenDSS 真实数据重新生成；
- A/B 完整训练对照和反捷径审计；
- 最终 `output/node-marker-mvp/<run-id>/` 产物。

当前已知环境限制：仓库既有 Probe 测试使用 Windows 路径解析，在本执行环境中收集时触发 `WinError 5`；该问题属于既有测试环境路径权限，不是 Node Marker 测试失败。新代码的语法核验、模型 smoke 和 Node Marker 测试均已通过。
