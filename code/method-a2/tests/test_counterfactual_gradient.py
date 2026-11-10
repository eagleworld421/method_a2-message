"""反事实评分梯度统计的行为测试。"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.counterfactual_scoring.gradient import gradient_report, parameter_groups


def test_gradient_report_records_disabled_loss_as_zero():
    model = torch.nn.Linear(2, 1)
    x = torch.ones(1, 2)
    first = (model(x) ** 2).mean()
    second = ((model(x) - 1.0) ** 2).mean()
    report = gradient_report({"first": first, "second": second}, model, {"first": 1.0, "second": 0.0})
    group = report["groups"]["other"]
    assert group["gradient_norm"]["second"] == 0.0
    assert group["norm_share"]["second"] == 0.0


def test_counterfactual_model_groups_include_score_and_response():
    from src.counterfactual_scoring.model import CounterfactualScoringModel

    groups = parameter_groups(CounterfactualScoringModel(2, 2, 1))
    assert groups["shared"]
    assert groups["response"]
    assert groups["score"]
