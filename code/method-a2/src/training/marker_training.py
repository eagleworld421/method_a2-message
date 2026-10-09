"""Node Marker 候选评分训练步骤，不读取特权 teacher 条件。"""

from __future__ import annotations

import torch

from ..model.losses import (
    candidate_classification_loss,
    heteroscedastic_response_loss,
    masked_family_loss,
    masked_response_loss,
    pairwise_candidate_ranking_loss,
)


def compute_marker_losses(
    model_output: dict[str, torch.Tensor],
    observation: torch.Tensor,
    target: torch.Tensor,
    candidate_mask: torch.Tensor,
    fault_mask: torch.Tensor,
    location: torch.Tensor | None = None,
    lambda_rank: float = 0.1,
    lambda_response: float = 0.1,
    rank_margin: float = 0.1,
    rank_temperature: float = 1.0,
    uncertainty_weight: float = 1.0,
    logvar_clamp: float = 8.0,
) -> dict[str, torch.Tensor]:
    """计算候选定位、候选对排序和响应重构三项损失。"""
    if location is None:
        raise ValueError("候选评分训练必须提供 location")
    clamped_ratio = torch.zeros((), dtype=observation.dtype, device=observation.device)
    if "response_logvar" in model_output:
        family, clamped = heteroscedastic_response_loss(
            model_output["response"], target, model_output["response_logvar"],
            candidate_mask, logvar_clamp=logvar_clamp,
        )
        clamped_ratio = torch.as_tensor(clamped, dtype=family.dtype, device=family.device)
    else:
        family = masked_response_loss(model_output["response"], target, candidate_mask)
    location_loss = candidate_classification_loss(
        model_output["candidate_score"], location, fault_mask,
        temperature=rank_temperature,
    )
    ranking = pairwise_candidate_ranking_loss(
        model_output["candidate_score"], location, fault_mask,
        margin=rank_margin, temperature=rank_temperature,
    )
    total = location_loss + lambda_rank * ranking + lambda_response * family
    if "response_logvar" in model_output:
        total = total * float(uncertainty_weight)
    return {
        "family": family,
        "loc": location_loss,
        "rank": ranking,
        "total": total,
        "logvar_clamped_ratio": clamped_ratio,
    }


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    lambda_rank: float = 0.1,
    lambda_response: float = 0.1,
    rank_margin: float = 0.1,
    rank_temperature: float = 1.0,
    uncertainty_weight: float = 1.0,
    logvar_clamp: float = 8.0,
) -> dict[str, float]:
    """执行一个无 teacher 的前向、反向和参数更新步骤。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        batch["observation"], batch["edge_index"], batch["edge_attr"],
        edge_mask=batch.get("edge_mask"),
        candidate_mask=batch["candidate_mask"],
        node_mask=batch.get("node_mask"),
    )
    losses = compute_marker_losses(
        output, batch["observation"], batch["target"], batch["candidate_mask"],
        batch["fault_mask"], batch.get("location"),
        lambda_rank=lambda_rank, lambda_response=lambda_response,
        rank_margin=rank_margin,
        rank_temperature=rank_temperature,
        uncertainty_weight=uncertainty_weight,
        logvar_clamp=logvar_clamp,
    )
    losses["total"].backward()
    optimizer.step()
    return {name: float(value.detach().cpu()) for name, value in losses.items()}
