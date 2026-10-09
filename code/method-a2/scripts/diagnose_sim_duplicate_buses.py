"""诊断：检查 OpenDSS 在不同故障母线上是否返回逐元素相同的母线电压响应。

本脚本只读取仿真，不写入数据文件。它用于判定阶段零发现的候选响应完全重复现象
是物理上真实的电气等价，还是故障命令未生效造成的仿真假象。
"""

import sys
from pathlib import Path

import numpy as np

METHOD_DIR = Path(__file__).absolute().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.data_generation.opendss_sim import FaultConfig, FaultSimulator  # noqa: E402


def main() -> int:
    """对若干故障配置打印关键母线的原始电压并比较差异。"""
    simulator = FaultSimulator("ieee13")
    simulator._compile_and_solve_base({})
    print("母线索引与名称：")
    for index, name in enumerate(simulator._bus_names):
        print(f"  {index:2d} {name:10s} phases={simulator._bus_phase_nodes[index]}")
    print("550-646 方向负荷连接已移除：")
    print("相邻母线对 (645,646) 是否在边表中：", (6, 7) in simulator.line_params or (7, 6) in simulator.line_params)
    print("相邻母线对 (646,611) 是否在边表中：", (7, 10) in simulator.line_params or (10, 7) in simulator.line_params)
    print("相邻母线对 (645,611) 是否在边表中：", (6, 10) in simulator.line_params or (10, 6) in simulator.line_params)

    for fault_class in (0, 4):
        for z_fault in (0.1, 10.0, 100.0):
            for bus in (5, 6, 7, 8, 10):
                result = simulator.generate_scenario(FaultConfig(
                    fault_class=fault_class, fault_bus=bus, z_fault=z_fault,
                ))
                pre = result["pre_v"]
                post = result["post_v"]
                delta = np.abs(post - pre)
                same_6_7 = float(np.abs(post[6] - post[7]).max())
                same_6_10 = float(np.abs(post[6] - post[10]).max())
                print(
                    f"class={fault_class} z={z_fault:6.1f} fault_bus={bus:2d} "
                    f"({simulator._bus_names[bus]:>4s}) |max delta_post-pre|={delta.max():.3e} "
                    f"fault_bus_delta={delta[bus].max():.3e} "
                    f"|V6-V7|max={same_6_7:.3e} |V6-V10|max={same_6_10:.3e}"
                )
    simulator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
