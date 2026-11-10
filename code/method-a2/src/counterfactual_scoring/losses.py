"""反事实候选评分的真实距离、列表、数值和间距损失。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def true_distances(observation: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算完整观测下真实反事实响应的候选距离。"""
    return (target - observation.unsqueeze(1)).pow(2).mean(dim=(2, 3, 4))


def family_loss(prediction: torch.Tensor, target: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    """计算候选反事实响应重构的掩码 Smooth L1 损失。"""
    weight = candidate_mask.to(prediction.dtype)[:, :, None, None, None]
    error = F.smooth_l1_loss(prediction, target, reduction="none") * weight
    return error.sum() / weight.expand_as(error).sum().clamp_min(1.0)


def list_loss(scores: torch.Tensor, distances: torch.Tensor, candidate_mask: torch.Tensor,
              temperature: float = 1.0, distance_scale: float = 1.0,
              n_fault_candidates: int | None = None) -> torch.Tensor:
    """用真实反事实距离构造软候选分布并计算列表级 KL。"""
    n_fault = n_fault_candidates or scores.shape[1] - 1
    valid = candidate_mask[:, :n_fault].bool()
    normalized = distances[:, :n_fault] / max(float(distance_scale), 1e-8)
    target_prob = torch.softmax((-normalized / temperature).masked_fill(~valid, -torch.inf), dim=1)
    log_prob = torch.log_softmax(scores[:, :n_fault].masked_fill(~valid, -torch.inf) / -temperature, dim=1)
    return F.kl_div(log_prob, target_prob, reduction="batchmean")


def value_loss(scores: torch.Tensor, distances: torch.Tensor, candidate_mask: torch.Tensor,
               distance_scale: float, n_fault_candidates: int | None = None) -> torch.Tensor:
    """约束评分值接近训练尺度归一化后的真实反事实距离。"""
    n_fault = n_fault_candidates or scores.shape[1] - 1
    valid = candidate_mask[:, :n_fault].to(scores.dtype)
    target = distances[:, :n_fault] / max(float(distance_scale), 1e-8)
    error = F.smooth_l1_loss(scores[:, :n_fault], target, reduction="none") * valid
    return error.sum() / valid.sum().clamp_min(1.0)


def gap_loss(scores: torch.Tensor, distances: torch.Tensor, candidate_mask: torch.Tensor,
             distance_scale: float, gap_threshold: float, n_fault_candidates: int | None = None) -> torch.Tensor:
    """在可靠真实候选间距上约束评分间距，不强迫近等价候选排序。"""
    n_fault = n_fault_candidates or scores.shape[1] - 1
    valid = candidate_mask[:, :n_fault].bool()
    total = scores.sum() * 0.0
    count = 0
    for row in range(scores.shape[0]):
        indices = torch.where(valid[row])[0]
        if indices.numel() < 2:
            continue
        true_gap = (distances[row, indices][:, None] - distances[row, indices][None, :]) / max(float(distance_scale), 1e-8)
        score_gap = scores[row, indices][:, None] - scores[row, indices][None, :]
        mask = torch.triu(true_gap.abs() >= float(gap_threshold), diagonal=1)
        if mask.any():
            total = total + F.smooth_l1_loss(score_gap[mask], true_gap[mask], reduction="sum")
            count += int(mask.sum().item())
    return total / max(count, 1)


def compute_losses(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                   ablation: str, distance_scale: float, gap_threshold: float,
                   temperature: float = 1.0, lambda_family: float = 0.1,
                   lambda_list: float = 1.0, lambda_value: float = 1.0,
                   lambda_gap: float = 1.0) -> dict[str, torch.Tensor]:
    """按消融配置计算总损失和各项原始损失。"""
    distances = true_distances(batch["observation"], batch["target"])
    family = family_loss(output["response"], batch["target"], batch["candidate_mask"])
    listing = list_loss(output["candidate_score"], distances, batch["candidate_mask"], temperature, distance_scale)
    value = value_loss(output["candidate_score"], distances, batch["candidate_mask"], distance_scale)
    gap = gap_loss(output["candidate_score"], distances, batch["candidate_mask"], distance_scale, gap_threshold)
    total = lambda_family * family + lambda_list * listing
    if ablation in ("value", "both"):
        total = total + lambda_value * value
    if ablation in ("gap", "both"):
        total = total + lambda_gap * gap
    return {"family": family, "list": listing, "value": value, "gap": gap,
            "total": total, "true_distances": distances}
