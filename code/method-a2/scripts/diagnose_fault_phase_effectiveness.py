"""诊断：区分「故障未生效」与「重复求解数值噪声」。

只做仿真读取，不写入数据文件。对每个候选母线，把注入故障后的母线电压与
同一仿真实例内紧邻两次无故障求解的电压差作对比：若故障引起的电压变化不高于
重复求解的数值噪声，则判定该故障命令未生效。
"""

import sys
from pathlib import Path

import numpy as np

METHOD_DIR = Path(__file__).absolute().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.data_generation.opendss_sim import FaultConfig, FaultSimulator  # noqa: E402


def main() -> int:
    """打印每个母线在各故障类型下的电压变化与数值噪声对比。"""
    # 终端编码在 Windows 下可能不是 UTF-8，这里显式设置输出编码
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    simulator = FaultSimulator("ieee13")
    simulator._compile_and_solve_base({})
    base_repeat = []
    for _ in range(3):
        simulator._compile_and_solve_base({})
        base_repeat.append(simulator._read_voltages())
    noise = max(
        float(np.abs(base_repeat[index] - base_repeat[0]).max())
        for index in range(1, len(base_repeat))
    )
    print(f"重复无故障求解的数值噪声水平: {noise:.6e}")
    print()
    ineffective: list[tuple[int, int, str, tuple[int, ...], float]] = []
    for fault_class in (0, 1, 2, 3, 4):
        print(f"=== 故障类型 {fault_class} ===")
        for bus in range(simulator._n_nodes):
            values = []
            for z_fault in (0.1, 1.0, 10.0, 100.0):
                result = simulator.generate_scenario(FaultConfig(
                    fault_class=fault_class, fault_bus=bus, z_fault=z_fault,
                ))
                values.append(float(np.abs(result["post_v"] - result["pre_v"]).max()))
            verdict = "未生效" if max(values) <= 1e-3 else "生效"
            if verdict == "未生效":
                ineffective.append(
                    (fault_class, bus, simulator._bus_names[bus],
                     simulator._bus_phase_nodes[bus], max(values))
                )
            print(
                f"  bus {bus:2d} {simulator._bus_names[bus]:>9s} "
                f"phases={simulator._bus_phase_nodes[bus]} "
                f"max|dV|={max(values):.3e} {verdict}"
            )
    print()
    print("未生效的（故障类型, 母线）组合汇总：")
    for fault_class, bus, name, phases, delta in ineffective:
        print(f"  class={fault_class} bus={bus:2d} {name:>9s} phases={phases} max|dV|={delta:.3e}")
    simulator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
