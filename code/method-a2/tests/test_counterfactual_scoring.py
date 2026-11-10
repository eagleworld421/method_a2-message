"""并行反事实候选评分路径的最小行为测试。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.counterfactual_scoring.losses import gap_loss, list_loss, true_distances, value_loss


def test_true_distances_use_real_counterfactual_targets():
    observation = torch.zeros(1, 2, 1, 1)
    target = torch.tensor([[[[[0.0]]], [[[1.0]]], [[[2.0]]]]])
    result = true_distances(observation, target)
    assert torch.allclose(result, torch.tensor([[0.0, 1.0, 4.0]]))


def test_value_and_gap_losses_backpropagate_to_scores():
    scores = torch.tensor([[0.2, 0.8, 1.5]], requires_grad=True)
    distances = torch.tensor([[0.0, 1.0, 4.0]])
    mask = torch.ones_like(scores)
    loss = value_loss(scores, distances, mask, 1.0) + gap_loss(scores, distances, mask, 1.0, 0.1)
    loss.backward()
    assert scores.grad is not None
    assert torch.any(scores.grad.abs() > 0)


def test_list_loss_changes_when_physical_targets_are_permuted():
    scores = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    mask = torch.ones_like(scores)
    first = list_loss(scores, torch.tensor([[0.0, 1.0, 4.0]]), mask)
    second = list_loss(scores, torch.tensor([[4.0, 1.0, 0.0]]), mask)
    assert not torch.allclose(first, second)
