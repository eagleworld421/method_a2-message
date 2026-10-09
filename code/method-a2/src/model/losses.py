"""Method-A2 候选评分、响应重构和配对排序损失。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_family_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算有效候选上的掩码均方响应重构损失。"""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction 和 target 必须同形状 [B,C,N,T,F]")
    weight = torch.ones_like(prediction)
    if candidate_mask is not None:
        if candidate_mask.shape != prediction.shape[:2]:
            raise ValueError("candidate_mask 必须为 [B,C]")
        weight = weight * candidate_mask.to(weight.dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    if response_mask is not None:
        if response_mask.ndim == 2:
            response_mask = response_mask.unsqueeze(-1).unsqueeze(-1)
        if response_mask.shape[:2] != prediction.shape[:2] or response_mask.shape[-2:] != prediction.shape[-2:]:
            raise ValueError("response_mask 必须可广播到 [B,C,N,T,F]")
        weight = weight * response_mask.to(weight.dtype)
    error = (prediction - target).pow(2) * weight
    return error.sum() / weight.sum().clamp_min(1.0)


def masked_response_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算有效候选上的掩码 Huber 响应辅助损失。"""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction 和 target 必须同形状 [B,C,N,T,F]")
    weight = torch.ones_like(prediction)
    if candidate_mask is not None:
        if candidate_mask.shape != prediction.shape[:2]:
            raise ValueError("candidate_mask 必须为 [B,C]")
        weight = weight * candidate_mask.to(weight.dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    if response_mask is not None:
        if response_mask.ndim == 2:
            response_mask = response_mask.unsqueeze(-1).unsqueeze(-1)
        if response_mask.shape[:2] != prediction.shape[:2] or response_mask.shape[-2:] != prediction.shape[-2:]:
            raise ValueError("response_mask 必须可广播到 [B,C,N,T,F]")
        weight = weight * response_mask.to(weight.dtype)
    error = F.smooth_l1_loss(prediction, target, reduction="none") * weight
    return error.sum() / weight.sum().clamp_min(1.0)


def candidate_classification_loss(
    scores: torch.Tensor,
    locations: torch.Tensor,
    candidate_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """直接监督候选评分，分数越低表示候选越可能是真实位置。"""
    if scores.ndim != 2:
        raise ValueError("scores 必须为 [B,C]")
    if locations.ndim != 1 or locations.shape[0] != scores.shape[0]:
        raise ValueError("locations 必须为 [B]")
    if candidate_mask.shape != scores.shape:
        raise ValueError("candidate_mask 必须为 [B,C]")
    if temperature <= 0:
        raise ValueError("temperature 必须为正数")
    valid_rows = locations >= 0
    if not valid_rows.any():
        return scores.sum() * 0.0
    selected_scores = scores[valid_rows]
    selected_locations = locations[valid_rows].long()
    selected_mask = candidate_mask[valid_rows].bool()
    if torch.any(selected_locations >= scores.shape[1]):
        raise ValueError("locations 包含越过候选轴的索引")
    if not torch.all(selected_mask.gather(1, selected_locations.unsqueeze(1))):
        raise ValueError("真实故障候选不能被 candidate_mask 屏蔽")
    logits = -selected_scores / float(temperature)
    logits = logits.masked_fill(~selected_mask, torch.finfo(logits.dtype).min)
    return F.cross_entropy(logits, selected_locations)


def pairwise_candidate_ranking_loss(
    scores: torch.Tensor,
    locations: torch.Tensor,
    candidate_mask: torch.Tensor,
    margin: float = 0.1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """要求真实候选评分低于有效错误候选评分。"""
    if scores.ndim != 2 or locations.ndim != 1:
        raise ValueError("scores 必须为 [B,C]，locations 必须为 [B]")
    if candidate_mask.shape != scores.shape:
        raise ValueError("candidate_mask 必须为 [B,C]")
    if margin < 0 or temperature <= 0:
        raise ValueError("margin 不得为负数，temperature 必须为正数")
    losses = []
    for row in range(scores.shape[0]):
        location = int(locations[row].item())
        if location < 0:
            continue
        valid = candidate_mask[row].bool()
        if not bool(valid[location]):
            raise ValueError("真实故障候选不能被 candidate_mask 屏蔽")
        negatives = valid.clone()
        negatives[location] = False
        if negatives.any():
            true_score = scores[row, location]
            negative_scores = scores[row, negatives]
            losses.append(
                F.softplus((true_score - negative_scores + margin) / float(temperature)).mean()
            )
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def pair_consistency_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """在严格同组配对样本上比较节点级共享表示。"""
    if z_a.shape != z_b.shape or z_a.ndim != 3:
        raise ValueError("配对表示必须同形状 [B,N,D]")
    return (z_a - z_b).pow(2).sum() / (z_a.shape[1] * z_a.shape[2])


def heteroscedastic_response_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    logvar: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    logvar_clamp: float = 8.0,
) -> tuple[torch.Tensor, float]:
    """计算异方差高斯响应损失，并返回方差被截断的元素比例。

    损失形式为 `(S-mu)^2 / (2 sigma^2) + 0.5 log sigma^2`，其中 `logvar` 已由
    解码器限制在 `[-logvar_clamp, logvar_clamp]`。返回值第二项为对数方差落在
    截断边界上的元素比例，用于判断方差是否被约束主导。
    """
    if prediction.shape != target.shape or logvar.shape != prediction.shape:
        raise ValueError("prediction、target 与 logvar 必须同形状 [B,C,N,T,F]")
    weight = torch.ones_like(prediction)
    if candidate_mask is not None:
        if candidate_mask.shape != prediction.shape[:2]:
            raise ValueError("candidate_mask 必须为 [B,C]")
        weight = weight * candidate_mask.to(weight.dtype).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    variance = torch.exp(logvar)
    error = (prediction - target).pow(2) / (2.0 * variance) + 0.5 * logvar
    loss = (error * weight).sum() / weight.sum().clamp_min(1.0)
    clamped_ratio = float((logvar.abs() >= float(logvar_clamp) - 1e-6).float().mean().item())
    return loss, clamped_ratio
