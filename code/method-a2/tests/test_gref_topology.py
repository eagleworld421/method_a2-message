"""候选拓扑单调性分类与案例选择的解析测试。"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_gref_topology import build_graph, classify_values, select_examples


def test_path_strict_increase_and_fault_origin():
    """路径从真实故障点向外严格增长，不混入数组索引方向。"""
    graph = nx.path_graph(4)
    row = classify_values(graph, 3, [4., 3., 2., 1.])
    assert row["strict_increase"] and not row["decrease_present"]
    assert row["distance"] == [3, 2, 1, 0]
    assert row["wrong_candidate_pair_violation_fraction"] == 0


def test_plateau_without_decrease():
    """平坦段属于非递减含平台，而不是严格增长。"""
    row = classify_values(nx.path_graph(4), 0, [0., 1., 1.001, 2.])
    assert row["plateau_present"] and row["nondecreasing_with_plateau"]
    assert not row["strict_increase"]


def test_plateau_can_overlap_decrease():
    """同一事件可同时含平台和下降，但互斥主类只能取下降。"""
    row = classify_values(nx.path_graph(4), 0, [0., 2., 1., 1.])
    assert row["decrease_present"] and row["plateau_present"]
    assert row["primary"] == "decrease_present"
    assert row["worst_edge"] == [1, 2]


def test_branch_difference_does_not_imply_path_decrease():
    """分支间存在很大差异时，各自路径仍可严格增长。"""
    graph = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 4)])
    row = classify_values(graph, 0, [0., 1., 4., 2., 5.])
    assert row["strict_increase"] and row["branch_difference"]
    assert row["wrong_candidate_pair_violation_fraction"] > 0


def test_constant_values_and_threshold_sensitivity():
    """常量曲线判为平台，数值小变化仅在给定相对阈值下合并。"""
    graph = nx.path_graph(3)
    assert classify_values(graph, 0, [1., 1., 1.])["nondecreasing_with_plateau"]
    assert classify_values(graph, 0, [0., .001, 1.], 0)["strict_increase"]
    assert classify_values(graph, 0, [0., .001, 1.], .01)["plateau_present"]


def test_graph_rejects_disconnected_or_cyclic_topology():
    """不连通和有环数据不能冒充本轮树结构检查。"""
    with pytest.raises(ValueError):
        build_graph(3, np.array([[0, 1]]))
    with pytest.raises(ValueError):
        build_graph(3, np.array([[0, 1], [1, 2], [2, 0]]))


def test_example_selection_handles_absent_and_small_classes():
    """实际案例不足时不凑数，案例足够时保持不重复。"""
    assert select_examples([], "decrease_present") == []
    rows = [{"dataset": label, "test_index": i, "group_id": i, "decrease_present": True,
             "max_relative_drop": float(i)} for i, label in enumerate(["a", "a", "b", "b"])]
    selected = select_examples(rows, "decrease_present")
    assert len(selected) == 3
    assert {r["dataset"] for r in selected} == {"a", "b"}
    assert len({r["test_index"] for r in selected}) == 3
