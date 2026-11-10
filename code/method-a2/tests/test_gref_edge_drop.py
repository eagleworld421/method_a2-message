"""训练期无向边删除掩码的单元测试。"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

METHOD_DIR = Path(__file__).resolve().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.gref.data import apply_edge_drop, build_undirected_edge_groups


def _edge_index() -> np.ndarray:
    """构造两条无向边、四条有向边的固定图。"""
    return np.asarray([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=np.int64)


def test_undirected_groups_pair_both_directions():
    """分组应把每条无向边的两个方向放在同一组。"""
    groups = build_undirected_edge_groups(_edge_index())
    assert len(groups) == 2
    assert all(len(group) == 2 for group in groups)
    assert {tuple(sorted(group.tolist())) for group in groups} == {(0, 1), (2, 3)}


def test_edge_drop_deletes_whole_undirected_edge():
    """启用删除时，一条无向边的两个方向必须同时置零。"""
    groups = build_undirected_edge_groups(_edge_index())
    batch = {"edge_mask": torch.ones(3, 4)}
    apply_edge_drop(batch, groups, drop_prob=1.0, max_drop=1, rng=np.random.default_rng(7))
    mask = batch["edge_mask"].numpy()
    for row in range(mask.shape[0]):
        assert mask[row].sum() == 2
        for group in groups:
            assert mask[row, group].min() == mask[row, group].max()


def test_edge_drop_probability_zero_keeps_mask():
    """删除概率为零时掩码完全不变。"""
    groups = build_undirected_edge_groups(_edge_index())
    batch = {"edge_mask": torch.ones(2, 4)}
    apply_edge_drop(batch, groups, drop_prob=0.0, max_drop=2, rng=np.random.default_rng(3))
    assert torch.equal(batch["edge_mask"], torch.ones(2, 4))


def test_edge_drop_max_can_remove_all_edges():
    """最大删除数覆盖全部无向边时，掩码应全为零。"""
    groups = build_undirected_edge_groups(_edge_index())
    batch = {"edge_mask": torch.ones(2, 4)}
    apply_edge_drop(batch, groups, drop_prob=1.0, max_drop=2, rng=np.random.default_rng(11))
    assert torch.equal(batch["edge_mask"], torch.zeros(2, 4))


def test_edge_drop_rejects_invalid_arguments():
    """非法概率或最大删除数必须直接报错。"""
    groups = build_undirected_edge_groups(_edge_index())
    batch = {"edge_mask": torch.ones(1, 4)}
    with pytest.raises(ValueError):
        apply_edge_drop(batch, groups, drop_prob=1.1, max_drop=1, rng=np.random.default_rng(1))
    with pytest.raises(ValueError):
        apply_edge_drop(batch, groups, drop_prob=0.5, max_drop=0, rng=np.random.default_rng(1))
