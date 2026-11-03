"""复用 Node Marker 响应骨干，以固定响应能量替代自由评分头。"""

from __future__ import annotations

import torch
from torch import nn

from ..model.marker_response_predictor import MarkerResponsePredictor
from .losses import observation_mask, response_energy


class GrefPredictor(MarkerResponsePredictor):
    """候选评分完全由预测响应与事实观测的掩码平方距离决定。"""

    def __init__(self, n_nodes: int, time_steps: int, feature_dim: int = 6, **kwargs):
        """复用响应参数，移除旧可训练评分头并固定基础响应路径。"""
        if kwargs.get("uncertainty", False) or kwargs.get("use_derived_channels", False):
            raise ValueError("GReF 首版不启用异方差或派生通道")
        super().__init__(n_nodes, time_steps, feature_dim, **kwargs)
        self.score_head = nn.Identity()

    def forward(
        self, x_obs: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        candidate_markers: torch.Tensor | None = None,
        candidate_types: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        response_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """仅接收部署可见输入，真实响应、标签与分组不得进入前向。"""
        mask = observation_mask(x_obs, response_mask)
        if node_mask is not None:
            mask = mask * observation_mask(x_obs, node_mask)
            mask = observation_mask(x_obs, mask)
        visible = torch.where(mask.bool(), x_obs, 0.0)
        output = super().forward(
            visible, edge_index, edge_attr, edge_mask=edge_mask,
            candidate_markers=candidate_markers, candidate_types=candidate_types,
            candidate_mask=candidate_mask, node_mask=node_mask,
        )
        output["candidate_score"] = response_energy(x_obs, output["response"], mask)
        return output
