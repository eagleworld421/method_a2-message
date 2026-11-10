<!-- 摘要：IEEE 37 OpenDSS 模型审计总入口，汇总文件完整性、节点/线路/负荷/变压器/调压器/电源参数、OpenDSS 求解验证结果、24 V 电源参考点建议、已验证与未确定事项，并索引全部交付物。 -->

# IEEE 37 OpenDSS 模型审计

> 文档定位：Method-A2 IEEE 37 实体配电网主干设计的模型审计总入口。  
> 数据依据：用户提供的 `37Bus.rar` 解压文件；电气结论以 `IEEE37_Load_Connection_Map.csv1` 与 `IEEELineCodes.DSS` 原文为准。  
> 审计状态：静态解析、图论检查与 OpenDSS COM 实际求解均已完成；未修改任何原始 DSS 文件。  
> 重要边界：本文只做模型审计与参数提取，不计算最终 PCB 元件值，不生成 PCB 布线，不把故障注入电路加入主干。

## 0. 审计结论摘要

- 模型在 OpenDSS 中完整可解：39 个母线名（含 `SourceBus`）、117 个相节点、35 条馈线线路加 1 条 Jumper、30 个负荷、4 个变压器/调压器对象、2 个调压控制器。
- 馈线本体 L1–L35 是严格的树；唯一的多重边是 799–799r 之间的 `reg1a`、`reg1c` 和 B 相 `Jumper`。没有孤立母线、重复线路、未定义节点或相别连接错误。
- 四个被引用 LineCode（721–724）的完整 3×3 R/X/C 矩阵已提取；线路总阻抗 = 单长矩阵 × 长度（mi），并与 `Yprim` 反推结果校验，最大差约 1.144e-16 Ω。
- 30 个负荷全部为 Delta 接线；29 个单相 Delta 支路加 1 个三相 Delta 负荷 `S728`，展开为 32 个相间支路。Model 1/2/4 的数量分别为 15/7/8。
- 原模型基态不是轻载：`SubXF` 视在负载约 121.1%（3029 kVA / 2500 kVA），4.8 kV 侧电流负载率约 135.2%；`reg1a`、`reg1c` 分别约 90.1%、88.5%，且 `creg1a` 已在最大档位 16。
- `Bus 799` 可以作为 24 V 电源的拓扑参考点，但必须把 24 V 定义为 4.8 kV 的 1.0 pu 等值，并决定是否保留缩放后的 SubXF/上游等效阻抗；若要做故障注入，不能使用理想 24 V 源。

## 1. 数据来源与审计边界

### 1.1 文件来源

当前文件系统中未找到 `37Bus.rar` 本体，审计使用此前解压目录：

`D:\ds_harness\codex_project\method_a2\code\method-a2\data\electrical-audit-ieee37-20261103`

该目录中的 `IEEE37_Load_Connection_Map.csv1`、`IEEELineCodes.DSS`、`IEEE37_Load_Connection_Map.csv0` 与 OpenDSS 安装目录 `C:\Program Files\OpenDSS\IEEETestCases\37Bus` 的同名文件 SHA256 完全一致。提取目录缺少 `ieee37_EXP_METERS.CSV`，该文件仅作为历史仪表导出表从同一 canonical 目录补齐，不参与电气计算。

这不是“用另一套网上模型替代用户文件”：三个电气文件与用户提取件逐字节一致，所有统计都能追溯到 DSS 的具体行号。

### 1.2 本次做了什么

- 逐行解析 `IEEE37_Load_Connection_Map.csv1`：`Redirect`、`BusCoords`、`New` 对象、`like=` 继承、`Set`、`Solve`。
- 逐行解析 `IEEELineCodes.DSS`：29 个 LineCode 定义，其中 721–724 被本模型引用。
- 用 OpenDSS COM 实际编译并求解原始模型，导出母线电压、线路电流、负荷功率、变压器/调压器功率、调压器档位和短路阻抗。
- 用图论检查母线级与相节点级的连通性、回路、并联边、孤立节点和相别一致性。
- 用 `Yprim` 的端子互导纳块独立校验线路总阻抗矩阵。

### 1.3 本次没有做什么

- 没有修改 `IEEE37_Load_Connection_Map.csv1` 或 `IEEELineCodes.DSS`。
- 没有把 Model 1/4 改成 Model 2，没有改负荷 P/Q，没有删除调压器/变压器。
- 没有计算 24 V 缩比后的最终 R/L/C 元件值，没有生成 KiCad 原理图或 PCB 布线。
- 没有把故障注入电路加入主干模型。
- 没有做 50 Hz 重新求解；本次求解为 60 Hz 单快照。

## 2. 文件索引

| 文件 | 内容 | 状态 |
|---|---|---|
| IEEE37_Load_Connection_Map.csv2 | 本文件：结论、索引、最终问题回答 | 完成 |
| IEEE37_Integrity_Report.md | 文件清单、引用关系、完整性检查 | 完成 |
| IEEE37_Topology.md | 节点—线路拓扑、主馈线、分支、末端、图论检查 | 完成 |
| IEEE37_Line_Impedance_Matrix.csv2 | 39 个母线名的电压等级、相节点、连接、负荷、电压、短路阻抗 | 完成 |
| IEEE37_Line_Impedance_Matrix.csv0 | 35 条馈线 + Jumper 的端点、相数、LineCode、长度、单长/总 R/X/C、电流、损耗 | 完成 |
| IEEE37_LineCode_Table.csv | 721–724 的完整 R/X/C 矩阵、单位、序分量、使用条数 | 完成 |
| IEEE37_LineCode_All_Definitions.csv | IEEELineCodes.DSS 中全部 29 个 LineCode 的定义，含未引用项 | 完成 |
| IEEE37_Line_Impedance_Matrix.csv | 逐线路逐元素的单长与总 R/X/C 矩阵长表 | 完成 |
| IEEE37_Line_Parameter_Analysis.md | 矩阵单位、单长与总阻抗、序分量、PCB 简化风险 | 完成 |
| IEEE37_Line_Impedance_Matrix.csv1 | 30 个负荷对象的节点、相别、Delta、Model、kV/kW/kvar/PF、CVR、实际 P/Q | 完成 |
| IEEE37_Load_Connection_Map.csv | 32 个相间支路的 PhaseFrom/PhaseTo、名义/求解 P/Q、支路电压 | 完成 |
| IEEE37_Load_Analysis.md | 负荷模型统计、S728 分配、固定 RLC 转换边界 | 完成 |
| IEEE37_Transformer_Table.csv | SubXF、XFM1、reg1a、reg1c 的绕组、容量、阻抗、变比、潮流、负载率 | 完成 |
| IEEE37_Regulator_Table.csv | creg1a、creg1c 的 Vreg、Band、PT/CT、LDC、档位、延时 | 完成 |
| IEEE37_Line_Parameter_Analysis.md2 | Vsource、SubXF、Open-Delta、XFM1、799/799r、24 V 等效阻抗与五个问题 | 完成 |
| IEEE37_Solver_Validation.md | OpenDSS 编译/收敛、电压/电流/功率/档位、静态一致性、复现命令 | 完成 |
| IEEE37_Line_Impedance_Matrix.csv4 | 结构化的求解日志：版本、编译、收敛、关键读数、限制 | 完成 |
| IEEE37_Bus_Voltages.csv | 39 个母线的 Vab/Vbc/Vca、pu、序分量、LN 电压、Zsc | 完成 |
| IEEE37_Line_Currents.csv | 逐线路逐端子逐相电流幅值/相角/标幺/方向 | 完成 |
| IEEE37_Load_Powers.csv | 30 个负荷的定义值与求解值对照 | 完成 |
| IEEE37_Transformer_Flows.csv | 变压器两侧 P/Q/S、损耗、负载率、最大电流 | 完成 |
| IEEE37_Regulator_State.csv | 调压器档位、Tap、799/799r 电压 | 完成 |
| IEEE37_Line_Parameter_Analysis.md9 | 连通分量、回路、并联边、孤立节点、相别错误等检查 | 完成 |
| IEEE37_Line_Parameter_Analysis.md1 | 四个文件的字节数、SHA256、引用状态 | 完成 |
| IEEE37_File_References.csv | Redirect/BusCoords/入口/辅助表的引用关系 | 完成 |
| IEEE37_Solver_Raw.json | 求解器原始读数（含 Yprim、Zsc、Powers），供复核 | 完成 |
| IEEE37_Line_Impedance_Matrix.csv7 | 关键计数与统计的机器可读摘要 | 完成 |
| IEEE37_Line_Impedance_Matrix.csv6 | 完整拓扑图（按电压等级着色，标注线路、变压器、调压器、负荷） | 完成 |
| IEEE37_Topology.svg | 拓扑图矢量副本，便于打印/编辑 | 完成 |
| source/IEEE37_Load_Connection_Map.csv1 | 原始主文件副本，未修改 | 只读副本 |
| source/IEEELineCodes.DSS | 原始线路配置副本，未修改 | 只读副本 |
| source/IEEE37_Load_Connection_Map.csv0 | 原始坐标副本，未修改 | 只读副本 |
| source/ieee37_EXP_METERS.CSV | 历史仪表导出副本，未修改 | 只读副本 |

线路、LineCode、负荷、变压器和调压器表均带有 `SourceLine` 列（适用时），用于回查 `IEEE37_Load_Connection_Map.csv1` 或 `IEEELineCodes.DSS` 的具体行号；`IEEE37_Solver_Raw.json` 保留求解器原始数组，便于逐项复核。

## 3. 关键计数

| 项目 | 数值 | 说明 |
|---|---|---|
| 母线名（含 SourceBus） | 39 | 静态解析与求解器一致 |
| 网络母线（不含 SourceBus） | 38 | 799、799r、701–744、775 |
| 相节点 | 117 | 39 × 3；不含变压器内部中性导体 |
| 线路对象 | 36 | 35 条三相馈线 + 1 条单相 Jumper |
| 馈线总长度 | 18.01 mi | L1–L35 长度之和 |
| 负荷对象 | 30 | 29 单相 Delta + 1 三相 Delta |
| 负荷相间支路 | 32 | 29 + 3（S728） |
| 负荷母线 | 25 | 30 个对象分布在 25 个母线 |
| 变压器/调压器 | 4 | SubXF、XFM1、reg1a、reg1c |
| 调压控制器 | 2 | creg1a、creg1c |
| LineCode 定义/引用 | 29 / 4 | 全部定义 29，本模型引用 721–724 |
| 负荷总 P | 2457.0 kW | DSS 定义值 |
| 负荷总 Q | 1201.0 kvar | DSS 定义值 |
| 负荷总 S | 2734.82 kVA | 由 P/Q 计算 |
| 负荷整体 PF | 0.8984 | 滞后 |

## 4. 最终八个问题的明确结论

### 4.1 当前 DSS 模型是否完整？

**电气上完整，文件层面有一个辅助文件需要说明。**

- `IEEE37_Load_Connection_Map.csv1` 的两条文件引用（`IEEELineCodes.DSS`、`IEEE37_Load_Connection_Map.csv0`）均存在，大小写一致；没有缺失的电气文件。
- 没有重复对象、无效 LineCode 引用、未定义母线或相别错误。
- 但模型依赖若干 OpenDSS 默认值：线路单位 `mi`、NormAmps 400 A / EmergAmps 600 A、负荷 CVR 1/2、调压器 `%r=0.2%`、变压器 `Xht/Xlt`、tap 上下限。这些默认值在 OpenDSS 中可解，但移植到 KiCad/其他工具时必须显式写出。
- `ieee37_EXP_METERS.CSV` 不是 DSS 引用，且其数值与本次 1.0 pu 快照不一致；不能作为模型完整性或参数校验的依据。

结论：**作为 OpenDSS 模型完整；作为自描述、可移植的电气数据库，需要把上述默认值固化到派生模型中。**

### 4.2 是否能确认所有节点和线路连接关系？

**能。** 所有线路的 `Bus1`/`Bus2` 均来自 DSS 原文，未按名称推断。共 39 个母线名、117 个相节点、35 条三相馈线、1 条单相 Jumper、4 个变压器/调压器对象。

- 母线级图：1 个连通分量；40 条边（35 线路 + 1 Jumper + 4 变压器/调压器）。
- 馈线本体 L1–L35：36 个节点、35 条边，是树。
- 回路秩 2，全部来自 799–799r 的三条并联支路（reg1a、reg1c、Jumper）；没有其他意外环路。
- 没有孤立母线；775 通过 XFM1 连接，是“无负荷末端”，不是断开分支。
- 相别检查：三相线路均为 1-1/2-2/3-3；单相 Delta 负荷均接在两个存在的相节点之间；无错误。

### 4.3 是否能确认所有线路的三相 R/X/C 矩阵？

**能。** 721–724 的完整 3×3 矩阵已逐元素提取：

- R/X 单位 Ω/mi，C 单位 nF/mi；长度单位 mi。
- 721–724 的 R/X 矩阵有非零互阻抗；C 矩阵的非对角元为零（对角电容）。
- 单长矩阵 × 长度 = 整条线路总矩阵；与 `Yprim` 的 `-inv(Y12)` 校验，最大差约 1.144e-16 Ω。
- 721–724 不是完全换位对称矩阵，序分量存在小的序间耦合；不能默认“三相对称解耦”。

限制：DSS 没有显式写 `Units`，`mi` 是 OpenDSS 默认值；`NormAmps/EmergAmps` 也是默认值，不是真实载流量。矩阵本身可确认，单位和载流量属性需要显式固化。

### 4.4 是否能确认所有负荷的节点、相别、接线和 P/Q？

**能，但固定 RLC 的“等价性”只在基准点成立。**

- 30 个负荷对象、25 个负荷母线；29 个单相 Delta、1 个三相 Delta（`S728`）。
- 展开为 32 个相间支路；每个支路的 PhaseFrom/PhaseTo、名义 kW/kvar、求解电压和求解 P/Q 均在 `IEEE37_Load_Connection_Map.csv`。
- 总定义负荷 2457.0 kW、1201.0 kvar；求解总 P/Q 与定义值一致（差异来自电压依赖）。
- Model 1/2/4 数量为 15/7/8；CVRwatts=1、CVRvars=2 是 OpenDSS 默认值。
- 所有负荷 `Conn=Delta`，`Enabled=true`，`Status=0`（Fixed）；没有 ZIPV，没有时变曲线。
- `S728` 的三个支路名义各 42 kW / 21 kvar；总功率 126 kW / 63 kvar。

限制：Model 1 恒功率、Model 4 指数/CVR 不能由固定 RLC 在全电压范围精确复现；只能做“基准工作点等效”。`S714a` 等 Model 4 的 P 按 V^1、Q 按 V^2（默认指数）变化，固定 RLC 的 P 按 V^2，二者不同。

### 4.5 是否能确认 SubXF、Open-Delta 调压器和 XFM1 的真实参数？

**能确认 DSS 中写出的参数和 OpenDSS 求解器解析出的参数；但“真实”仅指模型定义，不是设备铭牌实测。**

- `SubXF`：230/4.8 kV Δ–Δ，2500 kVA，Xhl=8%，%r=1%/绕组；4.8 kV 侧 R≈0.184320 Ω、X≈0.737280 Ω、|Z|≈0.759971 Ω；基态视在负载 121.1%，电流负载率 135.2%。
- `reg1a`/`reg1c`：单相 4.8/4.8 kV，2000 kVA/绕组，Xhl=1%，%r 未写、默认 0.2%/绕组；`creg1a` Vreg=122 V、Band=2 V、PT=40、CT=350 A、R=-0.201 V、X=3.348 V、档位 16（tap=1.1）；`creg1c` R=2.799 V、X=1.848 V、档位 14（tap=1.0875）。
- `XFM1`：4.8/0.48 kV Δ–Δ，500 kVA，Xhl=1.81%，%r=0.045%/绕组；变比 10:1；775 侧基态无负荷，潮流接近零。

限制：调压器 `%r`、`Xht`、`Xlt`、tap 上下限、`Delay` 等来自默认值；调压器 `R/X` 是 LDC 设置，不是绕组阻抗。若要把这些参数用于实体设备，需要按缩比基准重新设计，而不是照搬铭牌。

### 4.6 Bus 799 是否适合作为 24 V AC 电源参考点？

**适合作为拓扑参考点，但不是无条件的“直接接 24 V 理想源”。**

- 799 是 SubXF 二次侧、Open-Delta 调压器输入，保留它可以保留 reg1a/reg1c、B 相 Jumper、L35 和 701 以下全部馈线；这是最接近原拓扑的电源边界。
- 24 V 必须明确是 **4.8 kV 的 1.0 pu 等值**（线电压 RMS，三相三线 Delta），而不是“799 基态 0.923 pu 电压”。若把 24 V 钳在 799 的负荷电压上，标幺基准会偏移。
- 若实验包含故障注入，建议在 24 V 源后串联缩放后的 SubXF+上游戴维南阻抗；若只做正常工况演示，可以用理想源，但必须标注“短路特性不代表原模型”。
- 把 24 V 接在 799r 会绕过调压器和 799，属于派生简化模型；若要做调压器实验，不应选 799r。

结论：**799 是推荐基准，但 24 V 的定义、源阻抗、调压器实现和基态重载处理必须一起确定。**

### 4.7 哪些数据已经验证，哪些只是静态提取？

**求解器验证（实际计算）：**

- 编译成功、错误号 0、基态收敛（14 次迭代、4 次控制迭代）。
- 39 个母线、117 个相节点、73 个元件与静态解析一致。
- 所有母线 Vab/Vbc/Vca、线路电流、负荷 P/Q、变压器/调压器功率、调压器档位。
- 线路总阻抗与 `Yprim` 的交叉校验；母线/线路/负荷/变压器名称集合的一致性。
- 基态最低/最高电压（799 0.9232 pu；799r 1.0294 pu）、最大线路电流（L35 369.6 A）、SubXF/调压器负载率。

**静态提取（文本解析，未由求解器逐项证明）：**

- 文件引用、对象行号、`like=` 继承关系、注释掉的历史矩阵。
- 变压器/调压器的铭牌参数、负荷定义值、LineCode 矩阵原文。
- `CVRwatts/CVRvars`、`Units`、`NormAmps`、调压器 `%r` 等默认值的“未写出”事实。

**工程设计建议（不是原始模型数据）：**

- 24 V 作为 1.0 pu 的定义、SubXF 等效阻抗的缩放、平台功率基准、源阻抗实现、调压器实体方案、PCB 模块划分。
- 这些建议需要用户确认后才能进入 KiCad 设计。

**尚未确定：**

- 平台频率的最终冻结：本次按 60 Hz 求解；项目其他设计文档已记录 60 Hz 运行频率，若该决策仍有效则无需换算，若改为 50 Hz 需重算 X/C 和负荷 RLC。
- 平台功率/电流基准、电源容量与短路能力、电源保护/限流。
- 调压器是动态调压、固定档位还是旁路；XFM1 是否保留、775 是否接负荷。
- 固定 RLC 的转换基准、互耦实现方式、PCB 元件值与公差。
- 故障注入模块的接口与保护配合（属于模块 3，不在本次范围）。

### 4.8 当前还缺少哪些信息，才能进入 KiCad 主干原理图设计？

至少需要先确定以下事项；在这些事项冻结前，可以画“连接关系图”，但不能冻结元件值和 PCB：

1. **平台频率的最终冻结**：本次按 60 Hz 求解，项目其他设计文档已记录 60 Hz 运行频率；若最终改为 50 Hz，线路 X/C、变压器漏抗和负荷 RLC 都要重算。
2. **电压/功率基准**：24 V 对应 4.8 kV 的 1.0 pu；功率基准或电流基准（例如 50/100/250/500 VA）决定所有阻抗的缩放系数。
3. **电源实现与等值阻抗**：24 V 源是变压器、功放还是可编程源；其输出阻抗、短路电流、限流/保护行为；是否串联 SubXF 等值阻抗。
4. **基态重载的处理**：是否复现 SubXF 121% / 调压器 90% 的基态；若不复现，负荷缩放系数和电压水平如何调整。
5. **调压器实现**：动态调压、固定档位（16/14）还是旁路；LDC 是否实现；799/799r 的测量与端口。
6. **XFM1/775 的处理**：775 是否接负荷；XFM1 是否作为独立模块保留；0.48 kV 侧如何缩比。
7. **负荷 RLC 转换**：Model 1/4 的基准点、频率、Q 符号、每支路 R/L/C；是否接受“仅基准点等效”的误差。
8. **三相互耦实现**：完整 3×3 R/X 矩阵在 PCB 上如何实现；若简化为独立三相支路，允许的误差是多少。
9. **相序与接地**：A-B-C 相序、三线制 Delta、无中性线；安全地/屏蔽地如何与功率地分离。
10. **测量与保护**：Vab/Vbc/Vca 测量点、端子定义、DAQ 输入保护、故障模块接口、电流/热设计边界。

## 5. 已验证 / 静态提取 / 工程建议 / 未确定（汇总）

| 类别 | 内容 | 状态 |
|---|---|---|
| 已验证 | 编译/收敛、39 母线、117 相节点、73 元件、电压/电流/功率/档位、Yprim 阻抗校验 | 可复现 |
| 静态提取 | DSS 原文、对象行号、LineCode 矩阵、负荷定义、变压器铭牌、like= 继承 | 可追溯到行号 |
| 工程建议 | 24 V 1.0 pu 定义、SubXF 等值阻抗缩放、799/799r 选择、PCB 模块划分 | 待用户确认 |
| 未确定 | 频率、功率基准、电源硬件、调压器实现、固定 RLC 值、互耦实现、保护配合 | 进入 KiCad 前必须冻结 |

## 6. 已知限制

- `.rar` 本体未在文件系统中找到；审计使用此前解压目录，三个电气文件与 canonical 安装逐字节一致。若用户手中的 `37Bus.rar` 与 canonical 版本不同，应以用户文件重新运行本脚本。
- `ieee37_EXP_METERS.CSV` 与本次快照不一致，未用于任何电气结论。
- OpenDSS COM 不返回完整解析器警告列表；只检查了错误号和电路建立。
- `Line.R1`/`LineCode.R1` 标量属性存在活动对象缓存；本审计使用矩阵，未使用这些标量。
- 本次只求解 60 Hz 单快照正常工况，没有故障、时序、50 Hz 或温度场景。
- 本文所有“建议”均是工程建议，不是原模型数据；不得把建议值写回原始 DSS。

## 7. 复现

```powershell
python code/method-a2/scripts/audit_ieee37_model.py `
  --source-dir code/method-a2/data/electrical-audit-ieee37-20261103 `
  --out code/method-a2/output/ieee37_model_audit `
  --publish docs/project/reports/IEEE37_Model_Audit
```

脚本只读取原始 DSS；`IEEE37_Solver_Raw.json` 与 `IEEE37_Line_Impedance_Matrix.csv4` 记录求解器的原始读数，CSV 与 Markdown 由同一份解析结果生成。
