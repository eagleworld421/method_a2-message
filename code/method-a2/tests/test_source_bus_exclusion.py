"""电源母线剔除逻辑的单元测试。

母线与候选轴必须排除由理想电压源钳位的电源母线；本文件用纯函数测试该剔除规则，
不依赖 OpenDSS COM。
"""

from __future__ import annotations

import sys
from pathlib import Path

METHOD_DIR = Path(__file__).absolute().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.data_generation.opendss_sim import FaultSimulator  # noqa: E402


def test_source_bus_is_excluded_from_bus_axis():
    """电源母线必须从母线序列中移除，其余母线保持原有顺序。"""
    kept, dropped = FaultSimulator.exclude_source_buses(
        ["sourcebus", "799", "709"], {"sourcebus"}
    )
    assert kept == ["799", "709"]
    assert dropped == ["sourcebus"]


def test_exclusion_ignores_case_and_phase_suffix():
    """比较时忽略大小写与相别后缀，避免 OpenDSS 返回格式差异导致漏剔除。"""
    kept, dropped = FaultSimulator.exclude_source_buses(
        ["SourceBus", "799"], {"sourcebus.1.2.3"}
    )
    assert kept == ["799"]
    assert dropped == ["SourceBus"]


def test_no_source_bus_keeps_everything():
    """没有电源母线信息时不得误删任何母线。"""
    names = ["a", "b", "c"]
    kept, dropped = FaultSimulator.exclude_source_buses(names, set())
    assert kept == names
    assert dropped == []


def test_multiple_sources_are_all_excluded():
    """多个电压源时应剔除全部电源母线。"""
    kept, dropped = FaultSimulator.exclude_source_buses(
        ["s1", "x", "s2"], {"s1", "s2"}
    )
    assert kept == ["x"]
    assert dropped == ["s1", "s2"]
