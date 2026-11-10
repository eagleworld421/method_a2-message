"""同一反事实评分模型上的旧评分与反事实评分监督接口。"""

from __future__ import annotations

import torch

from ..model.losses import candidate_classification_loss, pairwise_candidate_ranking_loss
from .losses import family_loss, gap_loss, list_loss, true_distances, value_loss


WEIGHTS = {
    "family": 0.1,
    "loc": 1.0,
    "rank": 0.1,
    "list": 1.0,
    "value": 1.0,
    "gap": 1.0,
}

CONFIGS = {
    "old_score": ("family", "loc", "rank"),
    "list": ("family", "list"),
    "list_value": ("family", "list", "value"),
    "list_gap": ("family", "list", "gap"),
}


def compute_controlled_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: str,
    distance_scale: float,
    gap_threshold: float,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """计算所有诊断损失，并只激活当前配置指定的候选评分目标。"""
    if config not in CONFIGS:
        raise ValueError(f"未知受控配置：{config}")
    n_fault = output["candidate_score"].shape[1] - 1
    distances = true_distances(batch["observation"], batch["target"])
    fault_mask = batch["candidate_mask"].clone()
    fault_mask[:, n_fault:] = 0.0
    losses = {
        "family": family_loss(output["response"], batch["target"], batch["candidate_mask"]),
        "loc": candidate_classification_loss(
            output["candidate_score"], batch["location"], fault_mask, temperature=temperature
        ),
        "rank": pairwise_candidate_ranking_loss(
            output["candidate_score"], batch["location"], fault_mask,
            margin=0.1, temperature=temperature,
        ),
        "list": list_loss(
            output["candidate_score"], distances, batch["candidate_mask"],
            temperature=temperature, distance_scale=distance_scale, n_fault_candidates=n_fault,
        ),
        "value": value_loss(
            output["candidate_score"], distances, batch["candidate_mask"],
            distance_scale=distance_scale, n_fault_candidates=n_fault,
        ),
        "gap": gap_loss(
            output["candidate_score"], distances, batch["candidate_mask"],
            distance_scale=distance_scale, gap_threshold=gap_threshold, n_fault_candidates=n_fault,
        ),
    }
    active = set(CONFIGS[config])
    losses["total"] = sum(
        (losses[name] * WEIGHTS[name] for name in active),
        torch.zeros_like(losses["family"]),
    )
    losses["true_distances"] = distances
    return losses
