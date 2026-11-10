"""不改动旧模型的反事实监督候选评分模型。"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..model.marker_response_predictor import MarkerResponsePredictor


class CounterfactualScoringModel(nn.Module):
    """复用旧响应骨干，并使用独立评分头预测物理候选评分。"""

    def __init__(self, n_nodes: int, time_steps: int, feature_dim: int = 6):
        super().__init__()
        self.backbone = MarkerResponsePredictor(
            n_nodes=n_nodes,
            time_steps=time_steps,
            feature_dim=feature_dim,
            pooling="mean",
            temporal_mode="single",
            use_derived_channels=False,
            uncertainty=False,
            temporal_layers=3,
            encoder_layers=2,
        )
        score_dim = self.backbone.shared_dim + self.backbone.pool_dim + 1
        self.score_head = nn.Sequential(
            nn.Linear(score_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """输出候选响应、独立评分以及共享表示。"""
        output = self.backbone(
            batch["observation"],
            batch["edge_index"],
            batch["edge_attr"],
            edge_mask=batch.get("edge_mask"),
            node_mask=batch.get("node_mask"),
            candidate_mask=batch["candidate_mask"],
        )
        query = torch.bmm(output["candidate_markers"], output["Z"])
        score_input = torch.cat(
            [query, output["z_shared"].unsqueeze(1).expand(-1, query.shape[1], -1),
             output["candidate_types"]], dim=-1
        )
        output["candidate_score"] = self.score_head(score_input).squeeze(-1)
        return output
