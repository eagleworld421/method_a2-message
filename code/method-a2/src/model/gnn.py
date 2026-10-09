"""不含边可信度任务的拓扑消息传递 GNN。"""

import math

import torch
import torch.nn as nn


class _MessageLayer(nn.Module):
    """使用节点投影和边属性的单层消息传递。"""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_score = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """聚合源节点消息并更新目标节点。"""
        b, n, d = x.shape
        if edge_index.numel() == 0:
            return self.norm(x + self.update(torch.cat([x, torch.zeros_like(x)], dim=-1)))
        src = edge_index[:, 0].long()
        dst = edge_index[:, 1].long()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        logits = (q[:, dst, :] * k[:, src, :]).sum(dim=-1) / math.sqrt(d)
        logits = logits + self.edge_score(edge_attr).squeeze(-1).unsqueeze(0)
        if edge_mask.ndim == 1:
            active = edge_mask.to(dtype=x.dtype).unsqueeze(0).expand(b, -1)
        elif edge_mask.ndim == 2:
            active = edge_mask.to(dtype=x.dtype)
        else:
            raise ValueError("edge_mask 必须为 [E] 或 [B,E]")
        weights = torch.sigmoid(logits) * active
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        messages = v[:, src, :] * weights.unsqueeze(-1)
        aggregate = torch.zeros((b, n, d), dtype=x.dtype, device=x.device)
        aggregate.scatter_add_(1, dst.view(1, -1, 1).expand(b, -1, d), messages)
        return self.norm(x + self.update(torch.cat([x, aggregate], dim=-1)))


class TopologyGNN(nn.Module):
    """将 TCN 节点表示与观测拓扑融合为节点和全局表示。"""

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 64,
        edge_dim: int = 5,
        n_layers: int = 3,
        n_heads: int = 2,
        dropout: float = 0.1,
    ):
        del n_heads
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            _MessageLayer(hidden_dim, edge_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor = None,
    ) -> tuple:
        """返回节点表示 `[B,N,D]` 和全局表示 `[B,D]`。"""
        h = self.input_proj(x)
        if edge_mask is None:
            edge_mask = torch.ones(edge_index.shape[0], device=x.device, dtype=x.dtype)
        edge_index = edge_index.to(device=x.device)
        edge_attr = edge_attr.to(device=x.device, dtype=x.dtype)
        edge_mask = edge_mask.to(device=x.device)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, edge_mask)
        return h, h.mean(dim=1)


class _MeanMessageLayer(nn.Module):
    """使用固定边权平均聚合的消息传递层，不引入 QKV 或边打分。"""

    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """按有效边做无参数评分的均值消息聚合。"""
        batch_size, n_nodes, hidden_dim = x.shape
        aggregate = torch.zeros_like(x)
        if edge_index.numel() > 0:
            src = edge_index[:, 0].long()
            dst = edge_index[:, 1].long()
            if edge_mask.ndim == 1:
                active = edge_mask.to(dtype=x.dtype).unsqueeze(0).expand(batch_size, -1)
            elif edge_mask.ndim == 2:
                active = edge_mask.to(dtype=x.dtype)
            else:
                raise ValueError("edge_mask 必须为 [E] 或 [B,E]")
            messages = x[:, src, :] + self.edge_proj(edge_attr).unsqueeze(0)
            messages = messages * active.unsqueeze(-1)
            aggregate.scatter_add_(
                1,
                dst.view(1, -1, 1).expand(batch_size, -1, hidden_dim),
                messages,
            )
            degree = torch.zeros(
                (batch_size, n_nodes), dtype=x.dtype, device=x.device
            )
            degree.scatter_add_(1, dst.view(1, -1).expand(batch_size, -1), active)
            aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(-1)
        return self.norm(x + self.update(torch.cat([x, aggregate], dim=-1)))


class PlainTopologyGNN(nn.Module):
    """仅使用固定边均值聚合的拓扑编码器，不计算全局表示。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        edge_dim: int = 5,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            _MeanMessageLayer(hidden_dim, edge_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor = None,
        node_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """返回节点级共享表示 `[B,N,D]`。"""
        if x.ndim != 3:
            raise ValueError("拓扑编码器输入必须为 [B,N,D]")
        h = self.input_proj(x)
        edge_index = edge_index.to(device=x.device)
        edge_attr = edge_attr.to(device=x.device, dtype=x.dtype)
        if edge_mask is None:
            edge_mask = torch.ones(edge_index.shape[0], device=x.device, dtype=x.dtype)
        edge_mask = edge_mask.to(device=x.device, dtype=x.dtype)
        if node_mask is not None:
            if node_mask.ndim != 2 or node_mask.shape[:2] != x.shape[:2]:
                raise ValueError("node_mask 必须为 [B,N]")
            node_weight = node_mask.to(dtype=h.dtype).unsqueeze(-1)
            h = h * node_weight
        else:
            node_weight = None
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, edge_mask)
            if node_weight is not None:
                h = h * node_weight
        return h
