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
    weight = observation_mask(observation, mask)
    return _masked_response_energy(observation, response, weight)


def _masked_response_energy(observation, response, weight):
    """用已经验证的完整掩码计算逐候选响应能量。"""
    if response.ndim != 5 or response.shape[0] != observation.shape[0] or response.shape[2:] != observation.shape[1:]:
        raise ValueError("响应必须为与观测对应的 [B,C,N,T,F]")
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


def deployment_probabilities(
    energies: torch.Tensor, fault_mask: torch.Tensor, temperature: float,
) -> torch.Tensor:
    """在有效故障候选上按部署能量构造温度缩放概率，返回 `[B,C]`。

    正常候选与被掩码候选的概率恒为零；没有有效故障候选的事件整行为零，
    避免全 `-inf` 的 softmax 产生 NaN。
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("概率温度必须为正有限数")
    if energies.ndim != 2 or fault_mask.shape != energies.shape:
        raise ValueError("能量与故障候选掩码必须为同形状 [B,C]")
    fault = fault_mask.bool()
    logits = torch.where(
        fault, -energies / float(temperature),
        torch.full_like(energies, -torch.inf),
    )
    probabilities = torch.zeros_like(energies)
    rows = fault.any(dim=1)
    if rows.any():
        probabilities[rows] = torch.softmax(logits[rows], dim=1)
    return probabilities


def region_term_losses(
    output: dict[str, torch.Tensor], observation: torch.Tensor, target: torch.Tensor,
    candidate_mask: torch.Tensor, locations: torch.Tensor, kernel: torch.Tensor, *,
    rank_temperature: float, lambda_energy: float = 1.0, lambda_rank: float = 0.3,
    lambda_region: float = 0.1, beta_obs: float = 0.1, beta_miss: float = 0.4,
    mask: torch.Tensor | None = None, predicted_energy: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | float | int]:
    """RaReF 区域概率损失：保留能量与排序项，以区域交叉熵替换图平滑项。

    单节点概率由部署能量在故障候选上做温度缩放 softmax 得到；区域概率为
    `q = p K`；区域目标按真实节点是否被观测在两个 beta 之间切换。返回各分项、
    分子分母与带权总损失，便于训练日志与评价复用同一口径。
    """
    for name, value in (
        ("lambda_energy", lambda_energy), ("lambda_rank", lambda_rank),
        ("lambda_region", lambda_region),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须为正有限数")
    for name, value in (("beta_obs", beta_obs), ("beta_miss", beta_miss)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} 必须位于 [0,1]")
    if not math.isfinite(rank_temperature) or rank_temperature <= 0:
        raise ValueError("排序温度必须为正有限数")
    response = output["response"]
    if candidate_mask.shape != response.shape[:2]:
        raise ValueError("候选掩码形状不匹配")
    if output["candidate_types"].shape != (*candidate_mask.shape, 1):
        raise ValueError("候选类型形状不匹配")
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("区域核必须为方阵 [N,N]")
    n_fault = candidate_mask.shape[1] - 1
    if n_fault < 1 or kernel.shape[0] != n_fault:
        raise ValueError("区域核维度必须等于故障候选数（末尾为正常候选）")
    if response.shape[2] != n_fault:
        raise ValueError("区域核的节点数与故障候选数不一致")
    if not bool((output["candidate_types"].squeeze(-1)[:, -1] == 1).all()):
        raise ValueError("候选轴必须为故障候选加末尾正常候选")
    if locations.numel() and int(locations.max()) >= n_fault:
        raise ValueError("真实位置必须落在故障候选轴内")
    weight = observation_mask(observation, mask)
    fault_mask = candidate_mask.bool() & (output["candidate_types"].squeeze(-1) == 0)
    candidate = candidate_mask.to(response.dtype)

    reconstruction = reconstruction_energy(response, target, weight)
    energy_numerator = (reconstruction * candidate).sum()
    energy_denominator = candidate.sum()
    energies = predicted_energy if predicted_energy is not None else response_energy(observation, response, weight)
    true_energies = response_energy(observation, target.detach(), weight)
    rank_numerator, rank_events = list_energy_loss(
        energies, true_energies, fault_mask, locations, rank_temperature
    )

    probabilities = deployment_probabilities(energies, fault_mask, rank_temperature)
    # 核按 K[r,k] 存储，q_r = sum_k p_k K[r,k] = p @ K^T
    region = probabilities[:, :n_fault] @ kernel.to(
        device=probabilities.device, dtype=probabilities.dtype).t()
    rows = torch.where((locations >= 0) & fault_mask.any(dim=1))[0]
    if rows.numel():
        labels = locations[rows].long()
        observed = (
            mask[rows, labels] > 0 if mask is not None
            else torch.ones(rows.numel(), dtype=torch.bool, device=rows.device)
        )
        beta = torch.where(
            observed,
            torch.full_like(observed, float(beta_obs), dtype=region.dtype),
            torch.full_like(observed, float(beta_miss), dtype=region.dtype),
        ).unsqueeze(1)
        one_hot = F.one_hot(labels, num_classes=region.shape[1]).to(region.dtype)
        target_region = (1.0 - beta) * one_hot + beta * kernel.to(
            device=region.device, dtype=region.dtype
        )[:, labels].t()
        negative_log_likelihood = -(
            target_region * torch.log(region[rows].clamp_min(1e-12))
        ).sum(dim=1)
        region_numerator = negative_log_likelihood.sum()
        region_events = int(rows.numel())
        region_loss = negative_log_likelihood.mean()
    else:
        region_numerator = region.sum() * 0.0
        region_events = 0
        region_loss = region_numerator

    energy = energy_numerator / energy_denominator.clamp_min(1.0)
    rank = rank_numerator / rank_events if rank_events > 0 else rank_numerator * 0.0
    total = lambda_energy * energy + lambda_rank * rank + lambda_region * region_loss
    return {
        "total": total, "energy": energy, "rank": rank, "region": region_loss,
        "energy_numerator": energy_numerator, "energy_denominator": energy_denominator,
        "rank_numerator": rank_numerator, "rank_denominator": float(rank_events),
        "region_numerator": region_numerator, "region_denominator": float(region_events),
        "fault_events": int((locations >= 0).sum().item()), "rank_events": int(rank_events),
        "region_events": int(region_events), "energies": energies,
        "teacher_energy": true_energies,
    }


def reconstruction_energy(
    response: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """逐候选的掩码均方响应重构能量，返回 `[B,C]`。

    能量对时间与通道求和、对有效观测数归一；缺失目标先以零掩码替换，避免
    零掩码与非有限值相乘产生非有限结果。
    """
    if response.shape != target.shape or response.ndim != 5:
        raise ValueError("response 与 target 必须同形状 [B,C,N,T,F]")
    weight = observation_mask(response[:, 0], mask)
    difference = torch.where(weight[:, None].bool(), response - target, 0.0)
    return difference.square().sum(dim=(2, 3, 4)) / weight.sum(dim=(1, 2, 3))[:, None]


def list_energy_loss(
    energies: torch.Tensor, true_energies: torch.Tensor, fault_mask: torch.Tensor,
    locations: torch.Tensor, temperature: float,
) -> tuple[torch.Tensor, int]:
    """在故障候选上计算部署能量的软列表 KL，返回分子与有效事件数。

    真实能量与预测能量分别构造温度缩放的软候选分布；正常样本与没有有效故障
    候选的事件不参与。分子为逐事件 KL 之和，调用方以有效事件数归一。
    """
    if energies.ndim != 2 or true_energies.shape != energies.shape:
        raise ValueError("预测能量与真实能量必须为同形状 [B,C]")
    if fault_mask.shape != energies.shape or locations.shape != energies.shape[:1]:
        raise ValueError("故障候选掩码或真实位置形状不匹配")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("排序温度必须为正有限数")
    fault = fault_mask.bool()
    rows = (locations >= 0) & fault.any(dim=1)
    if not rows.any():
        return energies.sum() * 0.0, 0
    logits = (-energies[rows] / temperature).masked_fill(~fault[rows], -torch.inf)
    true_logits = (-true_energies[rows] / temperature).masked_fill(~fault[rows], -torch.inf)
    log_probability = torch.log_softmax(logits, dim=1)
    true_log_probability = torch.log_softmax(true_logits, dim=1)
    target_probability = true_log_probability.exp()
    # 目标概率为零处 `0 * log 0` 必须显式置零，否则产生 NaN。
    terms = torch.where(
        target_probability > 0,
        target_probability * (true_log_probability - log_probability),
        torch.zeros_like(target_probability),
    )
    return terms.sum(), int(rows.sum())


def unique_undirected_edges(
    edge_index: torch.Tensor, edge_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 `[E,2]` 的有向边去重为无向边，并按重复次数平均边权。"""
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError("edge_index 必须为 [E,2]")
    if edge_index.numel() == 0:
        empty = edge_index.new_zeros((0, 2), dtype=torch.long)
        return empty, edge_index.new_zeros((0,))
    pairs = torch.sort(edge_index.long(), dim=1).values
    unique, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    if edge_weights is None:
        weights = torch.ones(unique.shape[0], dtype=torch.float32, device=edge_index.device)
    else:
        if edge_weights.shape != (pairs.shape[0],):
            raise ValueError("边权形状必须为 [E]")
        if not torch.isfinite(edge_weights).all() or (edge_weights <= 0).any():
            raise ValueError("边权必须为正有限数")
        values = edge_weights.to(dtype=torch.float32)
        total = torch.zeros(unique.shape[0], dtype=torch.float32, device=values.device)
        total = total.scatter_add(0, inverse, values)
        count = torch.zeros_like(total).scatter_add(0, inverse, torch.ones_like(values))
        weights = total / count.clamp_min(1.0)
    return unique, weights


def graph_smoothness_terms(
    response: torch.Tensor, target: torch.Tensor, candidate_mask: torch.Tensor,
    edge_index: torch.Tensor, edge_weights: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """计算误差场 `f^T L f` 的分子与归一化分母。

    分母取 `2 * T * F * 有效候选数 * 有效边权和`，使该项与逐元素重构能量同尺度；
    节点观测掩码生效时只保留两端均有效的边。
    """
    if response.shape != target.shape or response.ndim != 5:
        raise ValueError("response 与 target 必须同形状 [B,C,N,T,F]")
    candidate_mask = candidate_mask.bool()
    if candidate_mask.shape != response.shape[:2]:
        raise ValueError("候选掩码必须为 [B,C]")
    edges, weights = unique_undirected_edges(edge_index, edge_weights)
    batch, _, n_nodes, steps, features = response.shape
    if edges.numel() and int(edges.max()) >= n_nodes:
        raise ValueError("边端点越界")
    weight = observation_mask(response[:, 0], mask)
    valid_node = weight.sum(dim=(2, 3)) > 0
    # 误差场只在有效观测处定义；缺失处置零不影响其他节点。
    field = torch.where(weight[:, None].bool(), response - target, 0.0)
    if edges.numel() == 0 or not candidate_mask.any():
        return field.sum() * 0.0, 0.0
    difference = field[:, :, edges[:, 0]] - field[:, :, edges[:, 1]]
    square = difference.square().sum(dim=(3, 4))
    edge_valid = (valid_node[:, edges[:, 0]] & valid_node[:, edges[:, 1]]).to(square.dtype)
    edge_valid = edge_valid * weights.unsqueeze(0)
    candidate = candidate_mask.to(square.dtype)
    numerator = (square * edge_valid.unsqueeze(1) * candidate.unsqueeze(2)).sum()
    denominator = (
        2.0 * steps * features * (edge_valid.sum(dim=1).unsqueeze(1) * candidate).sum()
    )
    return numerator, float(denominator)


def three_term_losses(
    output: dict[str, torch.Tensor], observation: torch.Tensor, target: torch.Tensor,
    candidate_mask: torch.Tensor, locations: torch.Tensor, edge_index: torch.Tensor, *,
    rank_temperature: float, lambda_energy: float = 1.0, lambda_rank: float = 0.1,
    lambda_smooth: float = 0.1, mask: torch.Tensor | None = None,
    edge_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | float | int]:
    """计算能量、排序与图平滑三项，返回分量、分子分母与带权总损失。

    能量项为逐候选掩码均方响应重构；排序项为部署能量的软列表 KL；平滑项为
    误差场的组合拉普拉斯二次型。三项分母分别按候选数、有效事件数与元素数归一。
    """
    for name, value in (
        ("lambda_energy", lambda_energy), ("lambda_rank", lambda_rank),
        ("lambda_smooth", lambda_smooth),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须为正有限数")
    if not math.isfinite(rank_temperature) or rank_temperature <= 0:
        raise ValueError("排序温度必须为正有限数")
    response = output["response"]
    if candidate_mask.shape != response.shape[:2]:
        raise ValueError("候选掩码形状不匹配")
    if output["candidate_types"].shape != (*candidate_mask.shape, 1):
        raise ValueError("候选类型形状不匹配")
    weight = observation_mask(observation, mask)
    fault_mask = candidate_mask.bool() & (output["candidate_types"].squeeze(-1) == 0)
    candidate = candidate_mask.to(response.dtype)

    reconstruction = reconstruction_energy(response, target, weight)
    energy_numerator = (reconstruction * candidate).sum()
    energy_denominator = float(candidate.sum().item())
    energies = response_energy(observation, response, weight)
    true_energies = response_energy(observation, target.detach(), weight)
    rank_numerator, rank_events = list_energy_loss(
        energies, true_energies, fault_mask, locations, rank_temperature
    )
    smooth_numerator, smooth_denominator = graph_smoothness_terms(
        response, target, candidate_mask, edge_index, edge_weights, weight
    )

    energy = energy_numerator / max(energy_denominator, 1.0)
    rank = rank_numerator / rank_events if rank_events > 0 else rank_numerator * 0.0
    smooth = smooth_numerator / smooth_denominator if smooth_denominator > 0 else smooth_numerator * 0.0
    total = lambda_energy * energy + lambda_rank * rank + lambda_smooth * smooth
    return {
        "total": total, "energy": energy, "rank": rank, "smooth": smooth,
        "energy_numerator": energy_numerator, "energy_denominator": energy_denominator,
        "rank_numerator": rank_numerator, "rank_denominator": float(rank_events),
        "smooth_numerator": smooth_numerator, "smooth_denominator": float(smooth_denominator),
        "fault_events": int((locations >= 0).sum().item()), "rank_events": int(rank_events),
        "energies": energies, "teacher_energy": true_energies,
    }
