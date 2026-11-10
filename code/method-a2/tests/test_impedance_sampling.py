"""验证故障阻抗采样方式的边界与对数均匀语义。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

METHOD_DIR = Path(__file__).absolute().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.data_generation.dataset_builder import _sample_event


class _FakeSimulator:
    """为事件采样测试提供最小的仿真器属性。"""

    _bus_phase_nodes = [(1, 2, 3)] * 4
    _base_loads = {"load": object()}


def test_log_uniform_impedance_stays_in_s4_range_and_is_log_uniform():
    """S4 采样应落在区间内，且 log 阈值两侧数量大致均衡。"""
    rng = np.random.default_rng(342)
    values = np.asarray([
        _sample_event(
            rng,
            _FakeSimulator(),
            100.0,
            1000.0,
            fault_classes=(0,),
            impedance_sampling="log_uniform",
        ).z_fault
        for _ in range(2000)
    ])

    assert values.min() >= 100.0
    assert values.max() <= 1000.0
    fraction_below_geometric_mean = (values < np.sqrt(100.0 * 1000.0)).mean()
    assert 0.45 < fraction_below_geometric_mean < 0.55


def test_log_uniform_rejects_non_positive_bounds():
    """对数均匀采样不得接受非正阻抗边界。"""
    rng = np.random.default_rng(342)
    try:
        _sample_event(
            rng,
            _FakeSimulator(),
            0.0,
            1000.0,
            fault_classes=(0,),
            impedance_sampling="log_uniform",
        )
    except ValueError as exc:
        assert "正数" in str(exc)
    else:
        raise AssertionError("非正阻抗边界未触发校验")
