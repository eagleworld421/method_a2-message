"""受控评分监督接口的行为测试。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.counterfactual_scoring.controlled_losses import compute_controlled_losses


def _batch():
    return {
        "observation": torch.zeros(1, 2, 1, 1),
        "target": torch.tensor([[[[[0.0]]], [[[1.0]]], [[[2.0]]]]]),
        "candidate_mask": torch.ones(1, 3),
        "location": torch.tensor([0]),
    }


def _output():
    score = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    response = torch.zeros(1, 3, 1, 1, 1, requires_grad=True)
    return {"candidate_score": score, "response": response}


def test_old_and_counterfactual_configs_have_different_active_targets():
    old = compute_controlled_losses(_output(), _batch(), "old_score", 1.0, 0.1)
    physical = compute_controlled_losses(_output(), _batch(), "list", 1.0, 0.1)
    assert old["total"].requires_grad and physical["total"].requires_grad
    assert not torch.allclose(old["total"], physical["total"])


def test_ablation_weights_disable_value_and_gap():
    losses = compute_controlled_losses(_output(), _batch(), "list", 1.0, 0.1)
    losses["total"].backward()
    assert losses["value"].detach().item() >= 0
    assert losses["gap"].detach().item() >= 0
