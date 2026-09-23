"""特权物理条件教师与无阻抗部署响应 predictor。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gnn import TopologyGNN
from .temporal import TemporalEncoder


class CandidateConditionedEncoder(nn.Module):
    """由观测、拓扑和候选物理描述生成候选条件表示。

    候选到节点的对应关系由固定物理特征匹配得到，不使用候选编号
    embedding。候选顺序置换时，关注矩阵和输出随候选特征同步置换。
    """

    def __init__(
        self,
        n_nodes: int,
        time_steps: int,
        feature_dim: int = 6,
        candidate_feature_dim: int = 10,
        temporal_hidden: int = 32,
        temporal_out: int = 32,
        gnn_hidden: int = 32,
        candidate_hidden: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.time_steps = int(time_steps)
        self.feature_dim = int(feature_dim)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.temporal = TemporalEncoder(
            in_dim=feature_dim,
            hidden_dim=temporal_hidden,
            out_dim=temporal_out,
        )
        self.gnn = TopologyGNN(
            in_dim=temporal_out,
            hidden_dim=gnn_hidden,
            edge_dim=5,
        )
        self.candidate_project = nn.Sequential(
            nn.Linear(candidate_feature_dim, candidate_hidden),
            nn.ReLU(),
            nn.Linear(candidate_hidden, candidate_hidden),
        )
        self.match_temperature = nn.Parameter(torch.tensor(0.20))

    def forward(
        self,
        x_obs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
        candidate_features: torch.Tensor,
        node_features: torch.Tensor,
        candidate_mask: torch.Tensor = None,
    ) -> dict:
        """返回候选条件节点表示、候选嵌入和物理匹配关注矩阵。"""
        if x_obs.ndim != 4:
            raise ValueError("x_obs 必须为 [B,N,T,F]")
        if candidate_features.ndim == 2:
            candidate_features = candidate_features.unsqueeze(0).expand(
                x_obs.shape[0], -1, -1
            )
        if candidate_features.ndim != 3 or candidate_features.shape[0] != x_obs.shape[0]:
            raise ValueError("candidate_features 必须为 [C,D] 或 [B,C,D]")
        if node_features.ndim == 2:
            node_features = node_features.unsqueeze(0).expand(
                x_obs.shape[0], -1, -1
            )
        if (
            node_features.ndim != 3
            or node_features.shape[0] != x_obs.shape[0]
            or node_features.shape[1] != x_obs.shape[1]
        ):
            raise ValueError("node_features 必须为 [N,D] 或 [B,N,D]")
        node_temporal = self.temporal(x_obs)
        node_repr, global_repr = self.gnn(
            node_temporal, edge_index, edge_attr, edge_mask=edge_mask
        )
        candidate_embed = self.candidate_project(candidate_features)
        node_mean = node_features.mean(dim=1, keepdim=True)
        node_std = node_features.std(dim=1, keepdim=True).clamp_min(1e-6)
        candidate_mean = candidate_features.mean(dim=1, keepdim=True)
        candidate_std = candidate_features.std(dim=1, keepdim=True).clamp_min(1e-6)
        node_norm = (node_features - node_mean) / node_std
        candidate_norm = (candidate_features - candidate_mean) / candidate_std
        distance = (
            candidate_norm.unsqueeze(2) - node_norm.unsqueeze(1)
        ).pow(2).sum(dim=-1)
        temperature = torch.nn.functional.softplus(self.match_temperature) + 0.01
        logits = -distance / temperature
        no_fault = candidate_features[:, :, -1] > 0.5
        logits = logits.masked_fill(no_fault.unsqueeze(-1), 0.0)
        if candidate_mask is not None:
            valid = candidate_mask.to(dtype=logits.dtype).unsqueeze(-1)
            logits = logits.masked_fill(valid <= 0, float("-inf"))
        attention = torch.softmax(logits, dim=-1)
        attention = attention.masked_fill(no_fault.unsqueeze(-1), 0.0)
        mean_attention = torch.full_like(attention, 1.0 / attention.shape[-1])
        attention = torch.where(
            no_fault.unsqueeze(-1), mean_attention, attention
        )
        return {
            "node_repr": node_repr,
            "global_repr": global_repr,
            "candidate_embed": candidate_embed,
            "candidate_features": candidate_features,
            "node_features": node_features,
            "attention": attention,
        }


class PairResponseHead(nn.Module):
    """对每个候选和每个输出节点解码响应的共享头。

    头输入包含输出节点表示、候选物理描述、候选嵌入、输出节点物理描述和
    候选—节点匹配权重；该结构让每个输出节点显式知道候选是谁。
    """

    def __init__(
        self,
        n_nodes: int,
        time_steps: int,
        feature_dim: int,
        candidate_feature_dim: int,
        node_feature_dim: int,
        candidate_hidden: int = 32,
        hidden_dim: int = 64,
        node_repr_dim: int = None,
        condition_dim: int = 0,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.time_steps = int(time_steps)
        self.feature_dim = int(feature_dim)
        pair_dim = (
            int(node_repr_dim if node_repr_dim is not None else hidden_dim)
            + candidate_hidden
            + candidate_feature_dim
            + node_feature_dim
            + 1
            + int(condition_dim)
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(hidden_dim, time_steps * feature_dim)

    def forward(
        self,
        node_repr: torch.Tensor,
        candidate_embed: torch.Tensor,
        candidate_features: torch.Tensor,
        node_features: torch.Tensor,
        attention: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """输出 `[B,C,N,T,F]` 候选响应。"""
        batch_size, n_candidates, _ = candidate_embed.shape
        n_nodes = node_repr.shape[1]
        node_expand = node_repr.unsqueeze(1).expand(
            batch_size, n_candidates, n_nodes, -1
        )
        candidate_embed_expand = candidate_embed.unsqueeze(2).expand(
            batch_size, n_candidates, n_nodes, -1
        )
        candidate_features_expand = candidate_features.unsqueeze(2).expand(
            batch_size, n_candidates, n_nodes, -1
        )
        node_features_expand = node_features.unsqueeze(1).expand(
            batch_size, n_candidates, n_nodes, -1
        )
        parts = [
            node_expand,
            candidate_embed_expand,
            candidate_features_expand,
            node_features_expand,
            attention.unsqueeze(-1),
        ]
        if condition is not None:
            if condition.ndim != 2 or condition.shape[0] != batch_size:
                raise ValueError("condition 必须为 [B,K]")
            parts.append(
                condition.unsqueeze(1).unsqueeze(2).expand(
                    batch_size, n_candidates, n_nodes, -1
                )
            )
        pair = torch.cat(parts, dim=-1)
        hidden = self.pair_mlp(pair)
        decoded = self.decoder(hidden)
        return decoded.reshape(
            batch_size,
            n_candidates,
            n_nodes,
            self.time_steps,
            self.feature_dim,
        )


class PrivilegedResponseSystem(nn.Module):
    """共享编码器、特权条件教师和无阻抗部署 predictor 的组合模块。"""

    def __init__(
        self,
        n_nodes: int,
        time_steps: int,
        feature_dim: int = 6,
        candidate_feature_dim: int = 10,
        node_feature_dim: int = None,
        temporal_hidden: int = 32,
        temporal_out: int = 32,
        gnn_hidden: int = 32,
        candidate_hidden: int = 32,
        hidden_dim: int = 64,
        impedance_hidden: int = 16,
    ):
        super().__init__()
        if node_feature_dim is None:
            node_feature_dim = candidate_feature_dim
        self.n_nodes = int(n_nodes)
        self.n_candidates = None
        self.no_fault_idx = int(n_nodes)
        self.node_feature_dim = int(node_feature_dim)
        self.candidate_feature_dim = int(candidate_feature_dim)
        self.encoder = CandidateConditionedEncoder(
            n_nodes=n_nodes,
            time_steps=time_steps,
            feature_dim=feature_dim,
            candidate_feature_dim=candidate_feature_dim,
            temporal_hidden=temporal_hidden,
            temporal_out=temporal_out,
            gnn_hidden=gnn_hidden,
            candidate_hidden=candidate_hidden,
            hidden_dim=hidden_dim,
        )
        self.student_head = PairResponseHead(
            n_nodes=n_nodes,
            time_steps=time_steps,
            feature_dim=feature_dim,
            candidate_feature_dim=candidate_feature_dim,
            node_feature_dim=self.node_feature_dim,
            candidate_hidden=candidate_hidden,
            hidden_dim=hidden_dim,
            node_repr_dim=gnn_hidden,
            condition_dim=0,
        )
        self.teacher_head = PairResponseHead(
            n_nodes=n_nodes,
            time_steps=time_steps,
            feature_dim=feature_dim,
            candidate_feature_dim=candidate_feature_dim,
            node_feature_dim=self.node_feature_dim,
            candidate_hidden=candidate_hidden,
            hidden_dim=hidden_dim,
            node_repr_dim=gnn_hidden,
            condition_dim=impedance_hidden,
        )
        self.impedance_encoder = nn.Sequential(
            nn.Linear(1, impedance_hidden),
            nn.ReLU(),
            nn.Linear(impedance_hidden, impedance_hidden),
        )

    def forward_student(
        self,
        x_obs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
        candidate_features: torch.Tensor,
        node_features: torch.Tensor,
        candidate_mask: torch.Tensor = None,
        node_mask: torch.Tensor = None,
    ) -> dict:
        """无阻抗部署前向；签名中不存在任何特权条件参数。

        `node_mask` 属于部署可见观测掩码 M；当前 paired-v1 为全节点观测，
        掩码在损失和残差计算中生效，编码器不额外读取未观测节点。
        """
        if node_mask is not None and (
            node_mask.ndim != 2 or node_mask.shape[1] != self.n_nodes
        ):
            raise ValueError("node_mask 必须为 [B,N]")
        encoded = self.encoder(
            x_obs,
            edge_index,
            edge_attr,
            edge_mask,
            candidate_features,
            node_features,
            candidate_mask=candidate_mask,
        )
        response = self.student_head(
            encoded["node_repr"],
            encoded["candidate_embed"],
            encoded["candidate_features"],
            encoded["node_features"],
            encoded["attention"],
        )
        return {
            "response": response,
            "attention": encoded["attention"],
        }

    def forward_teacher(
        self,
        x_obs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
        candidate_features: torch.Tensor,
        node_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        r_star: torch.Tensor,
        node_mask: torch.Tensor = None,
        detach_shared: bool = True,
    ) -> dict:
        """训练期特权条件前向；只有该方法允许读取真实阻抗。

        `detach_shared=True` 时停止教师响应损失对共享编码器的梯度，
        对应计划规定的默认梯度归属；设为 False 表示教师损失也更新共享
        编码器，必须作为单独变体报告。
        """
        encoded = self.encoder(
            x_obs,
            edge_index,
            edge_attr,
            edge_mask,
            candidate_features,
            node_features,
            candidate_mask=candidate_mask,
        )
        condition = self.impedance_encoder(
            torch.log1p(torch.clamp(r_star, min=0.0))
        )
        node_repr = encoded["node_repr"]
        candidate_embed = encoded["candidate_embed"]
        attention = encoded["attention"]
        if detach_shared:
            node_repr = node_repr.detach()
            candidate_embed = candidate_embed.detach()
            attention = attention.detach()
        response = self.teacher_head(
            node_repr,
            candidate_embed,
            encoded["candidate_features"],
            encoded["node_features"],
            attention,
            condition=condition,
        )
        return {
            "response": response,
            "attention": attention,
        }

    def trainable_parameter_count(self) -> int:
        """返回可训练参数数量。"""
        return int(
            sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
        )
