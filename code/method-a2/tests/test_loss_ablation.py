"""旧直接评分路径损失梯度分析的接口测试。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.loss_ablation.analysis import gradient_report, parameter_groups


def test_gradient_report_has_norm_share_cosine_and_projection():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    groups = {"shared": list(model.parameters())}
    losses = {"a": (model(torch.ones(1, 2)) ** 2).mean(), "b": (model(-torch.ones(1, 2)) ** 2).mean()}
    report = gradient_report(losses, model, {"a": 1.0, "b": 0.1})
    assert "norm_share" in report["groups"]["other"]
    assert "cosine" in report["groups"]["other"]
    assert "projection" in report["groups"]["other"]


def test_parameter_groups_cover_old_model_parameters():
    model = torch.nn.Linear(2, 1)
    groups = parameter_groups(model)
    assert sum(len(values) for values in groups.values()) == len(list(model.parameters()))
