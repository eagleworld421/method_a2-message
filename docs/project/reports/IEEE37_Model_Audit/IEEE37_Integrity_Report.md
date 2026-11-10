<!-- 摘要：IEEE 37 OpenDSS 模型完整性检查报告，记录四个源文件、Redirect/BusCoords 引用、对象计数、默认值依赖和缺失/重复/无效引用检查结论。 -->

# IEEE 37 DSS 模型完整性检查报告

> 文档定位：任务 1 的模型完整性审计结果。  
> 数据来源：`ieee37.dss`、`IEEELineCodes.DSS`、`IEEE37_BusXY.csv`、`ieee37_EXP_METERS.CSV`。  
> 状态：静态文本解析与 OpenDSS COM 求解均已执行；求解器版本见 `IEEE37_Solver_Validation.md`。

## 1. 审计对象与来源

本次审计的电气依据是用户提供的 `37Bus.rar` 解压内容。当前文件系统中未找到该 `.rar` 本体，但已有一份此前解压的源目录：

`D:\ds_harness\codex_project\method_a2\code\method-a2\data\electrical-audit-ieee37-20261103`

该目录中的 `ieee37.dss`、`IEEELineCodes.DSS` 和 `IEEE37_BusXY.csv` 与 OpenDSS 安装目录 `C:\Program Files\OpenDSS\IEEETestCases\37Bus` 中的同名文件逐字节一致（SHA256 相同）。提取目录缺少 `ieee37_EXP_METERS.CSV`，该文件仅作为历史仪表导出表从同一 canonical 目录补齐；它不参与电气拓扑和参数计算。

因此，本次审计没有用“网上的另一套 IEEE 37 模型”替代用户文件：三个电气文件与用户提取件逐字节一致，且所有结论均可追溯到具体 DSS 行号。

## 2. 文件清单

| 文件 | 字节数 | SHA256 | DSS 引用 |
|---|---|---|---|
| ieee37.dss | 7427 | 97c2571c1698be56c164d3b4c3a1c7a536df976631d3dba4d5108b8b44498b81 | master |
| IEEELineCodes.DSS | 11852 | 855910b3757efb1084c8310747a3459881cc135bd773665cdd6db16ff9b50ae1 | yes |
| IEEE37_BusXY.csv | 755 | e1af4a2e8667938db07ad8bd5704605d77bfea70678ecf89b255b2405f65afce | yes |
| ieee37_EXP_METERS.CSV | 1793 | e94e0cb7a425ed17314d901fdb3534b473fe0f8026fdc8ca5257139f460ad2a2 | no（辅助表，未被 DSS 引用） |

`ieee37_EXP_METERS.CSV` 只有一行仪表数据（Year 0、Hour 1、Meter MAIN），不是 DSS 输入，也不包含支路或节点定义。其 kWh/kvarh 数值与本次 1.0 pu 负荷快照不对应，不能用于参数提取。

## 3. 文件引用关系

| 类型 | 引用 | 源文件 | 源行 | 存在 |
|---|---|---|---|---|
| redirect | IEEELineCodes.DSS | ieee37.dss | 22 | yes |
| buscoords | IEEE37_BusXY.csv | ieee37.dss | 102 | yes |
| entry | ieee37.dss | ieee37.dss | 1 | yes |
| auxiliary | ieee37_EXP_METERS.CSV | （未引用） | — | yes |

主文件只包含两条文件级引用：

- `Redirect IEEELineCodes.DSS`：第 22 行，大小写与磁盘文件名完全一致。
- `BusCoords IEEE37_BusXY.csv`：第 102 行，大小写一致。

没有 `Compile` 其他主文件，没有 `Loads.dss`、`Transformers.dss` 或 `Regulators.dss` 外部引用。变压器、调压器、负荷和线路全部直接定义在 `ieee37.dss` 中。

## 4. 对象计数

| 对象 | 数量 | 说明 |
|---|---|---|
| 母线名称（含 SourceBus） | 39 | 求解器与静态解析一致 |
| 网络母线（不含 SourceBus） | 38 | 799、799r、701–744、775 |
| 母线相节点（1.2.3） | 117 | 39 个母线 × 3 相；不含变压器内部中性导体 |
| 线路对象（含 Jumper） | 36 | L1–L35 共 35 条三相线路，另 1 条单相 Jumper |
| 馈线线路 L1–L35 | 35 | 全部 Phases=3，Bus1/Bus2 均为 .1.2.3 |
| Jumper | 1 | 799.2–799r.2，B 相 1 mΩ 集中连接 |
| 负荷对象 | 30 | 29 个单相 Delta + 1 个三相 Delta（S728） |
| 变压器/调压器对象 | 4 | SubXF、XFM1、reg1a、reg1c |
| 调压控制器 | 2 | creg1a、creg1c |
| LineCode 定义总数 | 29 | IEEELineCodes.DSS 中全部定义 |
| 被本模型引用的 LineCode | 4 | 721、722、723、724 |

## 5. 缺失文件、重复定义与无效引用

- 缺失文件：无。`Redirect` 与 `BusCoords` 的目标均存在；`ieee37_EXP_METERS.CSV` 不是 DSS 引用，缺失不影响电气模型。
- 重复定义：未发现同名 `New` 对象重复定义。`reg1c like=reg1a` 和 `creg1c like=creg1a` 是显式继承后创建新对象，不是重复定义。
- 无效引用：未发现。所有线路、负荷、变压器和调压器引用的母线都由对象自身或变压器绕组创建；所有 `LineCode` 引用都能在 `IEEELineCodes.DSS` 中找到。
- 路径与大小写：`ieee37.dss` 使用相对路径引用两个文件，大小写与磁盘一致。迁移到 Linux 等大小写敏感环境时仍可解析。
- 注释与有效定义：`IEEELineCodes.DSS` 中 721–724 的若干 `!!!~` 行是注释掉的历史矩阵，当前生效的是其后的 `~` 行；解析器已忽略注释行。`ieee37.dss` 中的 `show` 命令均已注释，不会影响求解。

## 6. 依赖 OpenDSS 默认值的参数

以下参数在 DSS 中没有显式写出，而是由 OpenDSS 默认值补齐。它们是模型“完整可求解”与“可移植到其他工具”之间的主要差异：

- `LineCode` 与 `Line` 的 `Units`：未写，OpenDSS 默认 `mi`。求解器返回 `Units=0`，即英里。
- `LineCode` 的 `NormAmps` 与 `EmergAmps`：未写，求解器返回 400 A / 600 A。这是 OpenDSS 默认值，不是 IEEE 37 导体的真实载流量。
- 负荷 `CVRwatts`、`CVRvars`：未写，求解器返回 1.0 / 2.0。Model 4 负荷的电压指数即由这两个默认值决定。
- 负荷 `Status`：未写，求解器返回 0（Fixed）。`Enabled` 均为 true。
- 调压器 `reg1a`/`reg1c` 的 `%r`：未写，求解器返回 0.2%/绕组。SubXF 与 XFM1 的 `%r` 是显式写出的。
- 变压器 `Xht`、`Xlt`：未写，求解器返回 35%、30%；两绕组正序计算只使用 `Xhl`，但这两个默认值仍属于对象属性。
- 变压器 `MaxTap`、`MinTap`、`NumTaps`：未写，求解器返回 1.1、0.9、32。
- 调压控制器 `Delay`、`MaxTapChange`、`VoltageLimit`：`Delay=15 s`、`MaxTapChange=16` 来自默认值，`VoltageLimit=0` 表示未设置。

这些默认值不影响本次 OpenDSS 求解的可重复性，但在导出到 KiCad、MATLAB 或其他潮流工具时必须显式写出，否则会得到不同的线路长度、载流量或负荷电压特性。

## 7. 发现与严重性

- 高：`SubXF` 在 1.0 pu 负荷下视在功率约 3028.7 kVA，为 2500 kVA 额定值的 121.1%；电流负载率更高（见 `IEEE37_Transformer_Table.csv`）。原模型的基态不是“轻载”工况。
- 高：`reg1a` 与 `reg1c` 的视在负载分别约 1801.9 kVA 与 1770.7 kVA，为各自 2000 kVA 额定值的 90.1% 与 88.5%；两台调压器均接近 90% 视在负载，且 `creg1a` 已在最大档位 16。
- 中：`XFM1` 二次侧 775 没有负荷，基态潮流接近零；该支路在模型中存在但不参与功率传输。
- 中：线路 `NormAmps=400 A` 是默认值，不能直接当作 PCB 铜箔或端子选型依据。
- 中：`Bus 799` 在基态为 0.923 pu（Vab/Vca），低于 799r；24 V 电源若定义为 799 的 1.0 pu，必须同时决定是否保留 SubXF 的电压跌落。
- 低：`ieee37_EXP_METERS.CSV` 与当前快照不一致，不能作为校验数据。

## 8. 结论

就 OpenDSS 求解而言，模型是完整的：文件引用闭合、对象定义齐全、无缺失母线、无无效 LineCode 引用、无重复对象。就“可直接移植的电气数据库”而言，模型仍依赖若干 OpenDSS 默认值，且基态存在变压器/调压器重载。上述两点已在后续表中显式记录，不构成拓扑错误。
