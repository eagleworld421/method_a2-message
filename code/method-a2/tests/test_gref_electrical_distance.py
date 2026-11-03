"""统一标幺、模态端口、并联处理与距离比较的解析测试。"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_gref_electrical_distance import Q, differential_impedance, port_admittance, relationship


def series_y(z):
    """构造相同电压等级的纯串联两端口导纳。"""
    y = np.linalg.inv(z)
    return np.block([[y, -y], [-y, y]])


def test_series_line_modal_impedance_and_capacity_base():
    """核验相电压与三相容量基准，并验证改变容量仅整体缩放阻抗。"""
    z = np.diag([1+.3j, 2+.4j, 3+.5j])
    actual, residual = differential_impedance(series_y(z), [4800/np.sqrt(3)]*2)
    expected = Q.T @ z @ Q / (4800**2/1e6)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    doubled, _ = differential_impedance(series_y(z), [4800/np.sqrt(3)]*2, 2e6)
    np.testing.assert_allclose(doubled, actual*2, atol=1e-15)
    assert residual < 1e-12


def test_transformer_voltage_bases_and_blocked_zero_sequence():
    """不同电压侧一致标幺化，且不以零序被阻断为有限零序阻抗。"""
    z_pu = .0018+.0362j
    modal_y = Q @ Q.T / z_pu
    y_pu = np.block([[modal_y, -modal_y], [-modal_y, modal_y]])
    bases = np.array([4800/np.sqrt(3), 480/np.sqrt(3)])
    d_inverse = np.diag(1/np.repeat(bases, 3))
    y_si = (1e6/3) * d_inverse @ y_pu @ d_inverse
    actual, residual = differential_impedance(y_si, bases)
    np.testing.assert_allclose(actual, np.eye(2)*z_pu, atol=1e-14)
    assert residual < 1e-12
    assert np.linalg.matrix_rank(y_si) == 2


def test_untransmittable_modal_current_is_rejected():
    """只有一条相通道时，不为缺失的传输模态静默构造距离。"""
    y = np.zeros((6, 6))
    y[0, 0] = y[3, 3] = 1
    y[0, 3] = y[3, 0] = -1
    with pytest.raises(ValueError, match="可传输"):
        differential_impedance(y, [100, 100])


def test_parallel_elements_and_phase_mapping():
    """并联先求导纳和，端子相序反转不改变物理结果。"""
    y = np.array([[2, -2], [-2, 2]], complex)
    flat = y.flatten(order="F")
    raw = np.column_stack([flat.real, flat.imag]).ravel().tolist()
    a = dict(buses=["a.1", "b.2"], nconds=1, node_order=[1, 2], yprim=raw)
    b = dict(buses=["b.2", "a.1"], nconds=1, node_order=[2, 1], yprim=raw)
    total = port_admittance([a, b], ("a", "b"))
    assert total[0, 0] == total[4, 4] == 4
    assert total[0, 4] == total[4, 0] == -4
    assert np.count_nonzero(total) == 4


def test_scale_invariance_and_zero_length_ties():
    """统一正比例缩放保持相关与逆序比例，零长度候选仍被保留。"""
    distance = np.array([0., 0., 1., 2.])
    values = np.array([0., 1., 3., 2.])
    one = relationship(distance, values, 0, .01)
    two = relationship(distance*304.8, values, 0, .01)
    assert one == pytest.approx(two)
    assert one["ordered_pair_count"] == 3
    assert one["pair_violation"] == pytest.approx(1/3)


def test_positive_tree_weights_preserve_outward_orientation():
    """无论故障点位于何处，正权重保持树上的路径方向。"""
    graph = nx.Graph([(0, 1, {"weight": .001}), (1, 2, {"weight": 10}), (1, 3, {"weight": .1})])
    for root in graph:
        hops = nx.single_source_shortest_path_length(graph, root)
        weighted = nx.single_source_dijkstra_path_length(graph, root)
        for u, v in graph.edges():
            assert np.sign(hops[u]-hops[v]) == np.sign(weighted[u]-weighted[v])
