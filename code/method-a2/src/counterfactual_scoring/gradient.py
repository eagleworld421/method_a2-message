"""反事实评分路径的多目标梯度统计。"""

from __future__ import annotations

import math
from typing import Iterable

import torch


def _flatten(grads: Iterable[torch.Tensor | None], params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """把参数梯度展平为同一向量，并为未使用参数补零。"""
    values = []
    for grad, param in zip(grads, params):
        values.append(torch.zeros_like(param).reshape(-1) if grad is None else grad.reshape(-1))
    return torch.cat(values) if values else torch.zeros(0)


def parameter_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    """按反事实评分模型的功能模块组织参数。"""
    groups = {"shared": [], "response": [], "score": [], "other": []}
    for name, param in model.named_parameters():
        if name.startswith("backbone.temporal.") or name.startswith("backbone.encoder_gnn."):
            groups["shared"].append(param)
        elif (name.startswith("backbone.decoder_input.")
              or name.startswith("backbone.decoder_layers.")
              or name.startswith("backbone.output.")
              or name.startswith("backbone.output_logvar.")):
            groups["response"].append(param)
        elif name.startswith("backbone.score_head.") or name.startswith("score_head."):
            groups["score"].append(param)
        else:
            groups["other"].append(param)
    return groups


def gradient_report(
    losses: dict[str, torch.Tensor],
    model: torch.nn.Module,
    weights: dict[str, float],
) -> dict:
    """报告各目标的加权梯度范数、占比、余弦相似度和总梯度投影。"""
    report = {"groups": {}, "weights": weights}
    for group_name, params in parameter_groups(model).items():
        if not params:
            continue
        vectors = {}
        for loss_name, loss in losses.items():
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            vectors[loss_name] = _flatten(grads, params).detach() * float(weights.get(loss_name, 1.0))
        total = sum(vectors.values(), torch.zeros_like(next(iter(vectors.values()))))
        total_norm = float(total.norm().item())
        norms = {name: float(vector.norm().item()) for name, vector in vectors.items()}
        norm_sum = sum(norms.values())
        parameter_count = sum(param.numel() for param in params)
        result = {
            "parameter_count": int(parameter_count),
            "gradient_norm": norms,
            "gradient_rms": {name: value / math.sqrt(parameter_count) for name, value in norms.items()},
            "norm_share": {name: value / norm_sum if norm_sum > 1e-12 else None for name, value in norms.items()},
            "total_gradient_norm": total_norm,
            "cosine": {},
            "projection": {},
        }
        names = list(vectors)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                left_vector, right_vector = vectors[left], vectors[right]
                left_norm, right_norm = left_vector.norm(), right_vector.norm()
                cosine = torch.dot(left_vector, right_vector) / (left_norm * right_norm).clamp_min(1e-12)
                result["cosine"][f"{left}__{right}"] = (
                    float(cosine.item()) if left_norm > 1e-12 and right_norm > 1e-12 else None
                )
        result["projection"] = {
            name: float(torch.dot(vector, total).item() / (total_norm * total_norm))
            if total_norm > 1e-12 else None
            for name, vector in vectors.items()
        }
        report["groups"][group_name] = result
    return report
