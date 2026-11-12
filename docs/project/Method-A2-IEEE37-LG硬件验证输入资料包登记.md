<!-- 摘要：登记 IEEE37 浮地 Delta 单相接地故障硬件初步验证的两份输入资料包，说明解压后的数据、来源档案、参考文档与历史包分类路径，记录重复文件去重、校验清单及未展开的嵌套 RAR。 -->

# Method-A2 IEEE37-LG 硬件验证输入资料包登记

登记日期：2026-11-12。

本次登记对应根目录新增的两份 ZIP：`Method-A2_IEEE37_LG仿真验证_输入资料包_V1.0.zip` 与 `Method-A2_IEEE37_LG线路实体实现方案与参数_V1.6.zip`。资料用于 24 V AC、50 VA、60 Hz、三相三线制浮地 Delta 条件下的 IEEE37 L-G 硬件实现初步验证，尚不构成已验证的硬件设计或实验数据集。

## 归档路径

统一归档目录：`code/method-a2/data/ieee37-lg-hardware-validation-input-v1.0/`。

- `00_source_packages/`：两份原始 ZIP，保留来源包。
- `01_source_archives/`：`37Bus.rar` 与 `IEEE37_Model_Audit.rar` 原始来源档案；本次仅解压外层 ZIP，嵌套 RAR 留待模型审计时单独展开。
- `02_reference_docs/`：V1.0–V1.6 设计、缩比与审计复核文档。
- `03_calculation_tables/`：9 份 IEEE37 线路、负荷、节点电压及并联电容 CSV 计算表。
- `04_previous_packages/`：V1.4、V1.5、V1.6 历史 ZIP。
- `README.md` 与 `SHA256SUMS.txt`：原资料包说明和校验清单。

第二份 ZIP 中的 5 份 CSV 与 V1.6 文档已与第一份资料包逐项核对为相同 SHA-256，归档时只保留一份规范副本；两份原始 ZIP 均保留在 `00_source_packages/`。

## 使用边界

该目录属于硬件验证输入资料与计算参考，不新增 S0/S1/S2 主要实验数据集，因此不写入《Method-A2-主要实验数据集登记》。后续若由这些资料生成可供实验调用的独立数据集，必须先按数据集登记规则补充名称、路径、电阻范围、算例和特殊处理，再执行实验。
