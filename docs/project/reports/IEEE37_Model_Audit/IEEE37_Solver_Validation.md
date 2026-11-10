<!-- 摘要：IEEE 37 OpenDSS 实际求解验证报告，记录求解器版本、编译错误、收敛迭代、节点电压、线路电流、变压器/调压器状态、静态解析一致性与复现命令。 -->

# IEEE 37 OpenDSS 求解验证报告

> 文档定位：任务 6 的实际求解结果与验证边界。  
> 数据来源：OpenDSS COM 引擎（版本见下）、原始 `ieee37.dss`。  
> 状态：已实际编译并求解；本文所有电压、电流、功率和档位均来自求解器，不是静态推测。

## 1. 求解环境与编译

| 项目 | 结果 |
|---|---|
| OpenDSS 版本 | Version 11.0.0.1 (64-bit build) - Charlottesville; License Status: Open  |
| 编译命令 | Compile "D:\ds_harness\codex_project\method_a2\code\method-a2\data\electrical-audit-ieee37-20261103\ieee37.dss" |
| 编译错误号 | 0 |
| 编译错误描述 | （空） |
| 电路名 | ieee37 |
| 母线数 | 39 |
| 相节点数 | 117 |
| 元件数 | 73 |

OpenDSS COM 在编译阶段返回错误号 0，没有错误描述。模型正常建立电路，说明主文件、`Redirect IEEELineCodes.DSS` 和 `BusCoords IEEE37_BusXY.csv` 均被正确加载。

## 2. 收敛与迭代

| 指标 | 结果 |
|---|---|
| 是否收敛 | yes |
| 迭代次数 | 14 |
| 控制迭代次数 | 4 |
| 总迭代次数 | 14 |
| 最大迭代上限 | 100 |
| 收敛容差 | 0.0001 |
| 求解模式 | 0 |
| 控制模式 | 0 |
| 负荷倍率 | 1.0 |
| 频率 | 60.0 Hz |
| 总耗时 | 448.8 ms |
| 求解耗时 | 432.1 ms |

求解在 14 次迭代、4 次控制迭代后收敛；控制迭代次数不为零，说明调压器控制参与了求解，当前档位是控制作用后的结果。

## 3. 节点电压

全网 117 个相节点的线电压范围：

- 最低：`799`，Vmin = 0.9232 pu；
- 最高：`799r`，Vmax = 1.0294 pu；
- 最大负序/正序比：`740`，3.441%。

关键母线：

| 母线 | Vab/kV | Vbc/kV | Vca/kV | Vmin/pu | Vmax/pu |
|---|---|---|---|---|---|
| SourceBus | 230 | 230 | 230 | 1 | 1 |
| 799 | 4.433 | 4.552 | 4.431 | 0.9232 | 0.9483 |
| 799r | 4.821 | 4.941 | 4.773 | 0.9945 | 1.029 |
| 701 | 4.763 | 4.889 | 4.695 | 0.978 | 1.019 |
| 709 | 4.664 | 4.823 | 4.59 | 0.9563 | 1.005 |
| 775 | 0.4664 | 0.4823 | 0.459 | 0.9563 | 1.005 |
| 741 | 4.601 | 4.798 | 4.533 | 0.9445 | 0.9995 |

799 的电压低于 799r，符合“SubXF 二次侧 → 调压器升压 → L35 → 701”的预期。注意 799 的 0.923 pu 不是电源电压，而是 SubXF 在基态负荷下的二次侧电压；电源 `SourceBus` 仍为 1.0 pu。

## 4. 线路电流与损耗

电流最大的线路：

| 线路 | LineCode | Imax/A | 负载率/%（按 400 A） | Loss/kW |
|---|---|---|---|---|
| L35 | 721 | 369.6 | 92.4 | 28.44 |
| L1 | 722 | 268.2 | 67.0 | 10.96 |
| L4 | 722 | 189.9 | 47.5 | 5.93 |
| L6 | 723 | 147.1 | 36.8 | 4.08 |
| L17 | 723 | 131.1 | 32.8 | 1.54 |
| L27 | 723 | 131.1 | 32.8 | 1.11 |
| L14 | 723 | 122.9 | 30.7 | 1.36 |
| L28 | 723 | 106.0 | 26.5 | 1.78 |

- 全网最大电流：`L35`，369.62 A，为默认 `NormAmps=400 A` 的 92.41%。
- 全网线路总有功损耗约 62.327 kW。
- 400 A 是 OpenDSS 默认值，不是实际导体额定值；这里的“负载率”只能作为模型内部相对指标，不能直接用于 PCB 选型。

## 5. 变压器与调压器状态

| 设备 | P1/kW | Q1/kvar | S1/kVA | 视在负载率/% | 二次侧电流负载率/% |
|---|---|---|---|---|---|
| SubXF | 2588.5 | 1572.6 | 3028.7 | 121.1 | 135.2 |
| XFM1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| reg1a | 778.1 | 1625.2 | 1801.9 | 90.1 | 88.7 |
| reg1c | 1734.9 | -354.4 | 1770.7 | 88.5 | 85.8 |

- `SubXF` 基态视在负载约 121.1%，二次侧电流负载率约 135.2%；原模型基态是重载工况。
- `XFM1` 基态负载接近 0%，因为 775 没有负荷。
- `reg1a`、`reg1c` 的视在负载分别约 90.1%、88.5%，均接近 90%。

调压器档位：

| 控制器 | 档位 | Tap 比值 | Vreg/kV | Band/kV | LDC R/V | LDC X/V |
|---|---|---|---|---|---|---|
| creg1a | 16 | 1.1 | 4.88 | 0.08 | -0.201 | 3.35 |
| creg1c | 14 | 1.0875 | 4.88 | 0.08 | 2.8 | 1.85 |

`creg1a` 档位 16（最大档，tap=1.1）；`creg1c` 档位 14（tap=1.0875）。这说明基态 A–B 支路已经需要最大升压，后续若在实体平台上继续增加负荷，调压器没有更多升压裕度。

## 6. 静态解析与求解器的一致性

| 检查项 | 静态解析 | 求解器 | 一致 |
|---|---|---|---|
| 母线名称集合 | 39 | 39 | yes |
| 母线数（含 SourceBus） | 39 | 39 | yes |
| 相节点数 | 117 | 117 | yes |
| 线路对象数（含 Jumper） | 36 | 36 | yes |
| 负荷对象数 | 30 | 30 | yes |
| 变压器/调压器数 | 4 | 4 | yes |
| 调压控制器数 | 2 | 2 | yes |
| 被引用 LineCode 集合 | 721,722,723,724 | 721,722,723,724 | yes |

线路总阻抗的独立校验：由 `Yprim` 端子互导纳块反推的串联阻抗与“单长矩阵 × 长度”的最大绝对差为 1.144e-16 Ω；静态 LineCode 单长矩阵与求解器实际加载矩阵的最大绝对差为 3.553e-15。两个差值都处于双精度舍入量级，说明矩阵、长度和单位口径一致。

## 7. 求解器接口的已知限制

- COM 接口没有返回完整的解析器警告列表；本次只检查了 `Error.Number`/`Error.Description` 和电路是否建立。错误号为 0，不能证明“没有任何非致命警告”，但可以证明没有编译级错误。
- `Bus.ZscRefresh()` 会改变求解状态；本脚本在读取全部电压、电流和功率之后才刷新短路阻抗，避免把电压读数清零。复现时不要在每个母线读电压前调用 `ZscRefresh`。
- `Line.R1`、`LineCode.R1` 等标量属性在该 OpenDSS 版本中表现出活动对象缓存：四个 LineCode 返回了相同的 0.058/0.1206。本审计不使用这些标量，全部由 3×3 矩阵重新计算序分量。
- `Zsc1` 是母线处的驱动点阻抗，包含下游网络的并联效应，不等于单独的 SubXF 漏抗；SubXF 的铭牌漏抗应按 `IEEE37_Source_Analysis.md` 第 2 节的铭牌公式单独计算。
- 求解是 60 Hz 单快照，不是时序或故障求解；故障注入模块属于后续工作，本次未定义 `Fault` 元件。

## 8. 复现方法

### 8.1 OpenDSS GUI

1. 打开 `ieee37.dss`（确保 `IEEELineCodes.DSS` 与 `IEEE37_BusXY.csv` 同目录）。
2. 执行 `Solve`。
3. 执行 `Show Voltages LL Nodes`、`Show Currents`、`Show Powers kVA Elements`、`Show Taps`。
4. 如需导出，执行 `Export Voltages`、`Export Currents`、`Export Powers`。

### 8.2 Python COM

```python
import win32com.client, os
p = os.path.abspath("ieee37.dss")
e = win32com.client.Dispatch("OpenDSSEngine.DSS")
e.AllowForms = False
e.Text.Command = f'Compile "{p}"'
c = e.ActiveCircuit
c.Solution.Solve()
print(c.Solution.Converged, c.Solution.Iterations, c.Solution.ControlIterations)
for name in c.AllBusNames:
    c.SetActiveBus(name)
    print(name, c.ActiveBus.kVBase, c.ActiveBus.VLL)
```

### 8.3 本审计脚本

```powershell
python code/method-a2/scripts/audit_ieee37_model.py `
  --source-dir code/method-a2/data/electrical-audit-ieee37-20261103 `
  --out code/method-a2/output/ieee37_model_audit `
  --publish docs/project/reports/IEEE37_Model_Audit
```

## 9. 结论

原始模型在 OpenDSS 中编译成功、基态潮流收敛，节点电压、线路电流、变压器/调压器功率和档位均已导出。静态解析与求解器在母线、相节点、线路、负荷、变压器和 LineCode 集合上一致；线路总阻抗与 `Yprim` 校验通过。求解结果表明原模型基态存在 SubXF 与调压器重载，这是后续 24 V 缩比设计必须正视的边界。
