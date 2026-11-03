"""真实响应差异、物理尺度及反事实族配对的解析测试。"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_oracle_distance import audit_group, classify_oracle, response_distances


def test_response_difference_recovers_physical_channel_scale():
    """用独立解析数值检查通道尺度恢复、对称性及零对角。"""
    bank = np.array([[[[0., 0.]]], [[[1., 3.]]]])
    result = response_distances(bank, [2., 4.])
    np.testing.assert_array_equal(result, [[0., 74.], [74., 0.]])
    # 任意共同均值平移不改变响应之间的距离。
    np.testing.assert_array_equal(response_distances(bank+10, [2., 4.]), result)


def test_tiny_physical_differences_are_not_swallowed_by_absolute_floor():
    """相对容差应保留很小的物理差异，且正比例尺度不改变分类。"""
    graph = nx.path_graph(4)
    for scale in (1., 1e-14, 1e8):
        row = classify_oracle(graph, 0, np.array([0., 2., 1., 3.])*scale)
        assert row["decrease_present"]
        assert row["worst_edge"] == [1, 2]
        assert row["tolerance"] == pytest.approx(.03*scale)


def group_fixture():
    """构造具有唯一事实位置和严格共享响应库的最小完整族。"""
    bank = np.arange(39*6).reshape(39, 1, 1, 6).astype(float)
    return dict(signature_bank=np.repeat(bank[None], 38, axis=0), y_loc=np.arange(38),
                y_resist=np.ones(38), y_class=np.zeros(38), X_obs=bank[:38].copy(),
                mask=np.ones((38, 1)), candidate_mask=np.ones((38, 39)))


def test_group_audit_rejects_unpaired_bank_and_noisy_observation():
    """同族库不一致和观测噪声必须被识别，不能混入物理位置差异。"""
    data = group_fixture()
    bank, maximum = audit_group(data, np.arange(38))
    assert maximum == 0 and bank.shape == (38, 1, 1, 6)
    data["signature_bank"][1, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="响应库不一致"):
        audit_group(data, np.arange(38))
    data = group_fixture()
    data["X_obs"][1, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="观测噪声"):
        audit_group(data, np.arange(38))


def test_group_audit_rejects_changed_conditions_and_missing_locations():
    """不能把不同电阻条件或不完整位置覆盖视为完整反事实族。"""
    data = group_fixture()
    data["y_resist"][1] = 2
    with pytest.raises(ValueError, match="条件不一致"):
        audit_group(data, np.arange(38))
    with pytest.raises(ValueError, match="完整覆盖"):
        audit_group(group_fixture(), np.arange(37))
