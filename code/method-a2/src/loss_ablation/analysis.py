"""多损失梯度统计，不改动原训练模块。"""

from __future__ import annotations

import math
from typing import Iterable

import torch


def _flat_grads(grads: Iterable[torch.Tensor | None], params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """把一组可能缺失的梯度展平为同一向量。"""
    values = []
    for grad, param in zip(grads, params):
        values.append(torch.zeros_like(param).reshape(-1) if grad is None else grad.reshape(-1))
    return torch.cat(values) if values else torch.zeros(0)


def parameter_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    """按旧模型模块边界返回参数组。"""
    groups = {"shared": [], "score": [], "response": [], "other": []}
    for name, param in model.named_parameters():
        if name.startswith("temporal.") or name.startswith("encoder_gnn."):
            groups["shared"].append(param)
        elif name.startswith("score_head."):
            groups["score"].append(param)
        elif name.startswith("decoder_input.") or name.startswith("decoder_layers.") or name.startswith("output.") or name.startswith("output_logvar."):
            groups["response"].append(param)
        else:
            groups["other"].append(param)
    return groups


def gradient_report(losses: dict[str, torch.Tensor], model: torch.nn.Module,
                    weights: dict[str, float]) -> dict:
    """计算各损失在各参数组上的范数、余弦和合成投影。"""
    groups = parameter_groups(model)
    report = {"groups": {}, "weights": weights}
    for group_name, params in groups.items():
        if not params:
            continue
        vectors = {}
        for loss_name, loss in losses.items():
            raw = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            vector = _flat_grads(raw, params).detach()
            vectors[loss_name] = vector * float(weights.get(loss_name, 1.0))
        total = sum(vectors.values(), torch.zeros_like(next(iter(vectors.values()))))
        total_norm = float(total.norm().item())
        norms = {name: float(vector.norm().item()) for name, vector in vectors.items()}
        denom = sum(norms.values())
        group_result = {
            "parameter_count": int(sum(param.numel() for param in params)),
            "gradient_norm": norms,
            "gradient_rms": {name: value / math.sqrt(sum(param.numel() for param in params)) for name, value in norms.items()},
            "norm_share": {name: value / denom if denom > 1e-12 else None for name, value in norms.items()},
            "total_gradient_norm": total_norm,
            "cosine": {},
            "projection": {},
        }
        names = list(vectors)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                left_vector, right_vector = vectors[left], vectors[right]
                left_norm, right_norm = left_vector.norm(), right_vector.norm()
                value = torch.dot(left_vector, right_vector) / (left_norm * right_norm).clamp_min(1e-12)
                group_result["cosine"][f"{left}__{right}"] = float(value.item()) if left_norm > 1e-12 and right_norm > 1e-12 else None
        if total_norm > 1e-12:
            group_result["projection"] = {
                name: float((torch.dot(vector, total) / (total_norm * total_norm)).item())
                for name, vector in vectors.items()
            }
        else:
            group_result["projection"] = {name: None for name in vectors}
        report["groups"][group_name] = group_result
    return report
