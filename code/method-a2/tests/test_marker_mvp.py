"""Node Marker MVP 的接口、信息边界和损失契约测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

METHOD_DIR = Path(__file__).absolute().parents[1]
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from src.data_generation.waveform import build_dynamic_window
from src.model.losses import (
    candidate_classification_loss,
    pairwise_candidate_ranking_loss,
    masked_family_loss,
    pair_consistency_loss,
)
from src.model.marker_response_predictor import (
    MarkerResponsePredictor,
    build_candidate_markers,
)
from src.training.marker_training import compute_marker_losses, train_step


def _batch():
    """构造最小图批次。"""
    batch, nodes, steps, features = 2, 4, 6, 6
    edges = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=torch.long
    )
    edge_attr = torch.ones(edges.shape[0], 5)
    model = MarkerResponsePredictor(
        nodes, steps, features, temporal_hidden=8, shared_dim=8,
        decoder_hidden=8, dropout=0.0,
    )
    x = torch.randn(batch, nodes, steps, features)
    markers, types = build_candidate_markers(nodes)
    return model, x, edges, edge_attr, markers, types


def test_marker_semantics_and_normal_candidate():
    """故障候选为 one-hot，最后一个候选为全零位置标记和正常类型位。"""
    markers, types = build_candidate_markers(4)
    assert torch.equal(markers[:4], torch.eye(4))
    assert torch.equal(markers[4], torch.zeros(4))
    assert torch.equal(types[:4], torch.zeros(4, 1))
    assert torch.equal(types[4], torch.ones(1))


def test_marker_forward_shape_shared_representation_and_no_global_repr():
    """前向输出保持候选族形状，只返回节点级 Z。"""
    model, x, edges, edge_attr, markers, types = _batch()
    result = model(x, edges, edge_attr, candidate_markers=markers, candidate_types=types)
    assert result["response"].shape == (2, 5, 4, 6, 6)
    assert result["candidate_score"].shape == (2, 5)
    assert result["Z"].shape == (2, 4, 8)
    assert "g_m" not in result
    assert "global_repr" not in result


def test_marker_forward_candidate_sensitivity_and_gradient():
    """候选位置改变会影响预测，且梯度可回传共享编码器。"""
    model, x, edges, edge_attr, markers, types = _batch()
    result = model(x, edges, edge_attr, candidate_markers=markers, candidate_types=types)
    assert not torch.allclose(result["response"][:, 0], result["response"][:, 1])
    result["response"].square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_candidate_permutation_permuted_outputs():
    """候选输入置换时，输出仅按同一置换变化。"""
    model, x, edges, edge_attr, markers, types = _batch()
    model.eval()
    with torch.no_grad():
        original_output = model(x, edges, edge_attr, candidate_markers=markers, candidate_types=types)
        original = original_output["response"]
        original_score = original_output["candidate_score"]
        permutation = torch.tensor([2, 4, 0, 3, 1])
        permuted = model(
            x, edges, edge_attr,
            candidate_markers=markers[permutation],
            candidate_types=types[permutation],
        )
    assert torch.allclose(permuted["response"], original[:, permutation], atol=1e-6, rtol=1e-5)
    assert torch.allclose(permuted["candidate_score"], original_score[:, permutation], atol=1e-6, rtol=1e-5)


def test_losses_follow_candidate_and_pair_masks():
    """响应、配对和列表损失可以在有效掩码上计算。"""
    prediction = torch.zeros(2, 5, 4, 6, 6)
    target = torch.ones_like(prediction)
    observation = torch.zeros(2, 4, 6, 6)
    candidate_mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.float32
    )
    fault_mask = candidate_mask.clone()
    fault_mask[:, -1] = 0
    assert masked_family_loss(prediction, target, candidate_mask) > 0
    assert pair_consistency_loss(torch.zeros(2, 4, 8), torch.ones(2, 4, 8)) > 0
    scores = torch.tensor([[0.1, 0.8, 0.9, 1.0, 1.1], [0.2, 0.4, 0.6, 0.8, 1.0]])
    locations = torch.tensor([0, 1])
    assert candidate_classification_loss(scores, locations, fault_mask) >= 0
    assert pairwise_candidate_ranking_loss(scores, locations, fault_mask) >= 0


def test_waveform_default_has_no_random_perturbation():
    """默认动态窗口不再加入候选相关随机扰动。"""
    pre = np.asarray([[1.0, 1.0, 1.0, 0.0, -120.0, 120.0]], dtype=np.float32)
    post = pre.copy()
    first = build_dynamic_window(pre, post, fs=200.0, pre_cycles=1.0, post_cycles=2.0,
                                 rng=np.random.default_rng(1))
    second = build_dynamic_window(pre, post, fs=200.0, pre_cycles=1.0, post_cycles=2.0,
                                  rng=np.random.default_rng(99))
    assert np.array_equal(first, second)


def test_training_step_uses_only_student_inputs_and_batched_edge_mask():
    """最小训练步骤支持批量边掩码，且不需要 teacher 字段。"""
    model, x, edges, edge_attr, markers, types = _batch()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    target = torch.randn(2, 5, 4, 6, 6)
    batch = {
        "observation": x,
        "target": target,
        "edge_index": edges,
        "edge_attr": edge_attr,
        "edge_mask": torch.ones(2, edges.shape[0]),
        "candidate_mask": torch.ones(2, 5),
        "fault_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]], dtype=torch.float32),
        "location": torch.tensor([0, 1], dtype=torch.long),
    }
    losses = train_step(model, optimizer, batch)
    assert {"family", "loc", "rank", "total"} <= set(losses)
    assert all(np.isfinite(value) for value in losses.values())
