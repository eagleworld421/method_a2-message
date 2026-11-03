"""GReF 的统一响应能量、物理教师门控与双目标损失。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..model.losses import masked_response_loss


def observation_mask(observation: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """把节点或完整观测掩码扩展为 `[B,N,T,F]`，拒绝无有效观测的事件。"""
    if observation.ndim != 4:
        raise ValueError("观测必须为 [B,N,T,F]")
    if mask is None:
        mask = torch.ones_like(observation)
    elif mask.ndim == 2 and mask.shape == observation.shape[:2]:
        mask = mask[:, :, None, None].expand_as(observation)
    elif mask.shape != observation.shape:
        raise ValueError("观测掩码必须为 [B,N] 或 [B,N,T,F]")
    mask = mask.to(device=observation.device, dtype=observation.dtype)
    if not torch.isfinite(mask).all() or ((mask != 0) & (mask != 1)).any():
        raise ValueError("观测掩码必须由有限的零和一组成")
    if (mask.sum(dim=(1, 2, 3)) == 0).any():
        raise ValueError("每个事件必须存在有效观测")
    return mask


def response_energy(
    observation: torch.Tensor, response: torch.Tensor, mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """在已经统一标准化的响应坐标中计算能量，不再次除以旧标准差。"""
    if response.ndim != 5 or response.shape[0] != observation.shape[0] or response.shape[2:] != observation.shape[1:]:
        raise ValueError("响应必须为与观测对应的 [B,C,N,T,F]")
    weight = observation_mask(observation, mask)
    difference = torch.where(weight[:, None].bool(), response - observation[:, None], 0.0)
    return difference.square().sum(dim=(2, 3, 4)) / weight.sum(dim=(1, 2, 3))[:, None]


@torch.no_grad()
def teacher_gate(
    energies: torch.Tensor, locations: torch.Tensor, fault_mask: torch.Tensor,
    tolerance: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """只根据真实响应计算停止梯度的权重和间隔，正常事件不提供定位监督。"""
    if energies.ndim != 2 or fault_mask.shape != energies.shape or locations.shape != energies.shape[:1]:
        raise ValueError("教师能量、故障候选掩码或位置形状不匹配")
    threshold = torch.as_tensor(tolerance, dtype=energies.dtype, device=energies.device)
    if threshold.ndim == 0:
        threshold = threshold.expand(energies.shape[0])
    if threshold.shape != locations.shape or not torch.isfinite(threshold).all() or (threshold <= 0).any():
        raise ValueError("能量容差必须为正有限标量或逐事件向量")
    if ((locations < -1) | (locations >= energies.shape[1])).any():
        raise ValueError("真实位置越界，正常样本只能使用 -1")
    weights = torch.zeros_like(threshold)
    margins = torch.zeros_like(threshold)
    rows = torch.where(locations >= 0)[0]
    if rows.numel() == 0:
        return weights, margins
    labels = locations[rows].long()
    valid = fault_mask[rows].bool()
    if not valid[torch.arange(rows.numel(), device=rows.device), labels].all():
        raise ValueError("真实位置必须是有效故障候选")
    selected = energies[rows]
    if not torch.isfinite(selected[valid]).all():
        raise ValueError("有效教师能量必须为有限数值")
    negatives = valid.clone()
    negatives[torch.arange(rows.numel(), device=rows.device), labels] = False
    gap = selected.masked_fill(~negatives, torch.inf).min(dim=1).values - selected.gather(1, labels[:, None]).squeeze(1)
    reliable = negatives.any(dim=1) & (gap > threshold[rows])
    # 先筛选再相除，避免无负候选的无穷间隔产生无穷除无穷。
    active_rows = rows[reliable]
    active_gap = gap[reliable]
    weights[active_rows] = (active_gap - threshold[active_rows]) / (active_gap + 1e-12)
    margins[rows] = torch.where(negatives.any(dim=1), gap, 0.0)
    return weights, margins


def gref_losses(
    output: dict[str, torch.Tensor], observation: torch.Tensor, target: torch.Tensor,
    candidate_mask: torch.Tensor, locations: torch.Tensor, *, tolerance: float | torch.Tensor,
    temperature: float, lambda_family: float = 0.1,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """计算门控交叉熵与全候选重构，定位项直接依赖预测响应。"""
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(lambda_family) or lambda_family <= 0:
        raise ValueError("温度和重构权重必须为正有限数")
    response = output["response"]
    weight = observation_mask(observation, mask)
    energies = response_energy(observation, response, weight)
    if candidate_mask.shape != energies.shape or output["candidate_types"].shape != (*energies.shape, 1):
        raise ValueError("候选掩码或候选类型形状不匹配")
    fault_mask = candidate_mask.bool() & (output["candidate_types"].squeeze(-1) == 0)
    teacher = response_energy(observation.detach(), target.detach(), weight)
    weights, margins = teacher_gate(teacher, locations, fault_mask, tolerance)
    active = weights > 0
    if active.any():
        logits = (-energies[active] / temperature).masked_fill(~fault_mask[active], -torch.inf)
        ce = F.cross_entropy(logits, locations[active].long(), reduction="none")
        numerator = (ce * weights[active]).sum()
    else:
        numerator = energies.sum() * 0.0
    loc = numerator / (weights.sum() + 1e-12)
    family_mask = weight[:, None].expand_as(response)
    available = family_mask.bool() & candidate_mask[:, :, None, None, None].bool()
    # 在损失计算前去除缺失值，避免零掩码与非有限目标相乘仍产生非有限结果。
    family = masked_response_loss(torch.where(available, response, 0.0),
                                  torch.where(available, target, 0.0), candidate_mask, family_mask)
    return {
        "total": loc + lambda_family * family, "loc": loc, "family": family,
        "loc_numerator": numerator, "weight_sum": weights.sum(),
        "active_events": active.sum(), "fault_events": (locations >= 0).sum(),
        "gate_weights": weights, "teacher_margins": margins, "teacher_energy": teacher,
    }
