"""Method-A2 Node Marker 候选条件化响应预测器。

本模块实现 Node Marker B 路径：候选无关的共享编码器输出节点级表示，候选条件
通过位置标记 `b_k` 与候选类型位 `t_k` 进入候选条件化解码器。模块提供三类可开关
的能力，用于排除既有方法后的准确率提升验证：

- 鲁棒共享池化：`mean`、`trimmed_mean`、`mean_median_concat`，用于降低事实故障
  位置对事件级共享表示的污染；
- 多尺度时间编码与派生观测通道：在原始六通道之外构造幅值、相角与时间差分；
- 候选响应不确定性：在每个候选上同时输出响应均值与对数方差。

共享编码器始终只接收部署可见输入，不读取真实故障位置、`group_id`、真实阻抗或
真实反事实响应，也不计算 `global_repr`、QKV 或可学习边打分。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gnn import PlainTopologyGNN
from .temporal import DualScaleTemporalEncoder, TemporalEncoder

POOLING_MODES = ("mean", "trimmed_mean", "mean_median_concat")


def build_candidate_markers(
    n_nodes: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造故障候选 one-hot 标记和正常候选类型位。"""
    if n_nodes < 1:
        raise ValueError("n_nodes 必须为正整数")
    markers = torch.zeros((n_nodes + 1, n_nodes), device=device, dtype=dtype)
    markers[:n_nodes] = torch.eye(n_nodes, device=device, dtype=dtype)
    types = torch.zeros((n_nodes + 1, 1), device=device, dtype=dtype)
    types[n_nodes, 0] = 1.0
    return markers, types


def build_derived_channels(x_obs: torch.Tensor) -> torch.Tensor:
    """由实部／虚部观测构造派生通道，返回 `[B,N,T,6*F/2]`。

    通道顺序为 `[Re, Im, |V|, angle, d|V|/dt, d angle/dt]`，按每个复数对的三个相别
    依次排列。相角按时间轴做连续相位展开后再差分，避免在正负 π 附近出现跳变；
    两个差分通道的首个时间步取零。
    """
    if x_obs.ndim != 4 or x_obs.shape[-1] % 2 != 0:
        raise ValueError("派生通道要求输入为 [B,N,T,F] 且最后一维为偶数（实部／虚部成对）")
    real = x_obs[..., 0::2]
    imag = x_obs[..., 1::2]
    magnitude = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-12)
    angle = torch.atan2(imag, real)
    # 沿时间轴做连续相位展开
    difference = angle[..., 1:, :] - angle[..., :-1, :]
    difference = (difference + torch.pi) % (2.0 * torch.pi) - torch.pi
    unwrapped = torch.cat([
        angle[..., :1, :],
        angle[..., :1, :] + torch.cumsum(difference, dim=-2),
    ], dim=-2)
    magnitude_delta = torch.zeros_like(magnitude)
    magnitude_delta[..., 1:, :] = magnitude[..., 1:, :] - magnitude[..., :-1, :]
    angle_delta = torch.zeros_like(unwrapped)
    angle_delta[..., 1:, :] = unwrapped[..., 1:, :] - unwrapped[..., :-1, :]
    return torch.cat([x_obs, magnitude, unwrapped, magnitude_delta, angle_delta], dim=-1)


def _trimmed_mean(x: torch.Tensor, fraction: float = 0.1) -> torch.Tensor:
    """在节点维去除最高与最低各 `fraction` 比例后的均值，返回 `[B,D]`。"""
    n_nodes = x.shape[1]
    cut = int(n_nodes * float(fraction))
    if cut <= 0 or 2 * cut >= n_nodes:
        return x.mean(dim=1)
    ordered, _ = torch.sort(x, dim=1)
    return ordered[:, cut:n_nodes - cut, :].mean(dim=1)


class _CandidateMessageLayer(nn.Module):
    """候选维展平后的固定边均值消息层。"""

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
        """按有效边进行固定均值消息聚合。"""
        batch_candidates, n_nodes, hidden_dim = x.shape
        aggregate = torch.zeros_like(x)
        if edge_index.numel() > 0:
            src = edge_index[:, 0].long()
            dst = edge_index[:, 1].long()
            active = edge_mask.to(dtype=x.dtype)
            if active.ndim == 1:
                active = active.unsqueeze(0).expand(batch_candidates, -1)
            elif active.ndim != 2 or active.shape[0] != batch_candidates:
                raise ValueError("候选边掩码必须为 [E] 或 [B*C,E]")
            messages = x[:, src, :] + self.edge_proj(edge_attr).unsqueeze(0)
            messages = messages * active.unsqueeze(-1)
            aggregate.scatter_add_(
                1,
                dst.view(1, -1, 1).expand(batch_candidates, -1, hidden_dim),
                messages,
            )
            degree = torch.zeros(
                (batch_candidates, n_nodes), dtype=x.dtype, device=x.device
            )
            degree.scatter_add_(1, dst.view(1, -1).expand(batch_candidates, -1), active)
            aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(-1)
        return self.norm(x + self.update(torch.cat([x, aggregate], dim=-1)))


class MarkerResponsePredictor(nn.Module):
    """共享节点表示和候选 Marker 条件化解码器。"""

    def __init__(
        self,
        n_nodes: int,
        time_steps: int,
        feature_dim: int = 6,
        edge_dim: int = 5,
        temporal_hidden: int = 32,
        shared_dim: int = 32,
        decoder_hidden: int = 32,
        decoder_layers: int = 2,
        dropout: float = 0.1,
        temporal_layers: int = 3,
        encoder_layers: int = 2,
        pooling: str = "mean",
        temporal_mode: str = "single",
        use_derived_channels: bool = False,
        uncertainty: bool = False,
        logvar_clamp: float = 8.0,
    ):
        super().__init__()
        if pooling not in POOLING_MODES:
            raise ValueError(f"未知池化方式：{pooling}，可选值为 {list(POOLING_MODES)}")
        if temporal_mode not in ("single", "dual"):
            raise ValueError(f"未知时间编码方式：{temporal_mode}")
        self.n_nodes = int(n_nodes)
        self.time_steps = int(time_steps)
        self.feature_dim = int(feature_dim)
        self.shared_dim = int(shared_dim)
        self.pooling = pooling
        self.temporal_mode = temporal_mode
        self.use_derived_channels = bool(use_derived_channels)
        self.uncertainty = bool(uncertainty)
        self.logvar_clamp = float(logvar_clamp)
        # 启用派生通道时输入维度为原始通道数的三倍：原始实部／虚部、幅值与展开相角、
        # 幅值差分与相角差分
        input_dim = feature_dim * 3 if self.use_derived_channels else feature_dim
        encoder_class = DualScaleTemporalEncoder if temporal_mode == "dual" else TemporalEncoder
        self.temporal = encoder_class(
            in_dim=input_dim, hidden_dim=temporal_hidden, out_dim=shared_dim,
            n_layers=temporal_layers,
        )
        self.encoder_gnn = PlainTopologyGNN(
            in_dim=shared_dim, hidden_dim=shared_dim, edge_dim=edge_dim,
            n_layers=encoder_layers, dropout=dropout
        )
        self.pool_dim = shared_dim * (2 if pooling == "mean_median_concat" else 1)
        # 解码器输入为节点表示、候选查询、事件级共享表示、节点标记位与候选类型位
        self.decoder_input = nn.Linear(shared_dim * 2 + self.pool_dim + 2, decoder_hidden)
        self.decoder_layers = nn.ModuleList([
            _CandidateMessageLayer(decoder_hidden, edge_dim, dropout)
            for _ in range(decoder_layers)
        ])
        self.output = nn.Linear(decoder_hidden, time_steps * feature_dim)
        # 候选评分只读取候选节点表示、事件级共享表示和正常候选类型位。
        self.score_head = nn.Sequential(
            nn.Linear(shared_dim + self.pool_dim + 1, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, 1),
        )
        if self.uncertainty:
            self.output_logvar = nn.Linear(decoder_hidden, time_steps * feature_dim)

    def encode_shared(
        self,
        x_obs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """只计算候选无关的节点级共享表示 `Z_m`。"""
        if x_obs.ndim != 4:
            raise ValueError("x_obs 必须为 [B,N,T,F]")
        if x_obs.shape[1] != self.n_nodes or x_obs.shape[3] != self.feature_dim:
            raise ValueError("x_obs 的节点数或特征数与模型配置不一致")
        if self.use_derived_channels:
            # 派生通道由模型内部构造，避免把额外特征写入数据契约；
            # 此时输入必须是原始六通道的实部／虚部观测
            if x_obs.shape[-1] != self.feature_dim:
                raise ValueError(
                    "启用派生通道时输入通道数必须等于 feature_dim（原始实部／虚部观测）"
                )
            x_obs = build_derived_channels(x_obs)
        if node_mask is not None:
            if node_mask.ndim != 2 or node_mask.shape[:2] != x_obs.shape[:2]:
                raise ValueError("node_mask 必须为 [B,N]")
            x_obs = x_obs * node_mask.to(dtype=x_obs.dtype).unsqueeze(-1).unsqueeze(-1)
        temporal = self.temporal(x_obs)
        return self.encoder_gnn(
            temporal, edge_index, edge_attr, edge_mask=edge_mask, node_mask=node_mask
        )

    def pool_shared(self, z: torch.Tensor) -> torch.Tensor:
        """按配置的池化方式把节点级表示汇聚为事件级共享表示。"""
        mean = z.mean(dim=1)
        if self.pooling == "mean":
            return mean
        if self.pooling == "trimmed_mean":
            return _trimmed_mean(z)
        median = z.median(dim=1).values
        return torch.cat([mean, median], dim=-1)

    def forward(
        self,
        x_obs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        candidate_markers: torch.Tensor | None = None,
        candidate_types: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """并行预测全部候选，返回响应、`Z` 与事件级共享表示。"""
        z = self.encode_shared(x_obs, edge_index, edge_attr, edge_mask, node_mask)
        batch_size = x_obs.shape[0]
        if candidate_markers is None and candidate_types is None:
            candidate_markers, candidate_types = build_candidate_markers(
                self.n_nodes, device=x_obs.device, dtype=x_obs.dtype
            )
        elif candidate_markers is None or candidate_types is None:
            raise ValueError("candidate_markers 和 candidate_types 必须同时提供")
        if candidate_markers.ndim == 2:
            candidate_markers = candidate_markers.unsqueeze(0).expand(batch_size, -1, -1)
        if candidate_types.ndim == 2:
            candidate_types = candidate_types.unsqueeze(0).expand(batch_size, -1, -1)
        if candidate_markers.ndim != 3 or candidate_markers.shape[0] != batch_size:
            raise ValueError("candidate_markers 必须为 [C,N] 或 [B,C,N]")
        if candidate_types.shape[:2] != candidate_markers.shape[:2] or candidate_types.shape[-1] != 1:
            raise ValueError("candidate_types 必须为 [C,1] 或 [B,C,1]")
        if candidate_markers.shape[-1] != self.n_nodes:
            raise ValueError("candidate_markers 的节点维度与模型不一致")
        candidate_markers = candidate_markers.to(device=z.device, dtype=z.dtype)
        candidate_types = candidate_types.to(device=z.device, dtype=z.dtype)

        # 候选查询 q_{m,k} 取自候选无关的节点级表示：故障候选读取标记所在节点的
        # 表示，正常候选因 b 全零而得到零向量。事件级共享表示 z_shared 作为独立
        # 输入广播到全部候选与全部节点，用于承载与候选位置无关的事件上下文。
        pooled = self.pool_shared(z)
        query = torch.bmm(candidate_markers, z)

        n_candidates = candidate_markers.shape[1]
        pooled_expand = pooled.unsqueeze(1).unsqueeze(1).expand(
            batch_size, n_candidates, self.n_nodes, -1
        )
        z_expand = z.unsqueeze(1).expand(batch_size, n_candidates, self.n_nodes, -1)
        query_expand = query.unsqueeze(2).expand(-1, -1, self.n_nodes, -1)
        marker_expand = candidate_markers.unsqueeze(-1)
        type_expand = candidate_types.unsqueeze(2).expand(-1, -1, self.n_nodes, -1)
        score_input = torch.cat([
            query,
            pooled.unsqueeze(1).expand(-1, n_candidates, -1),
            candidate_types,
        ], dim=-1)
        candidate_score = self.score_head(score_input).squeeze(-1)
        hidden = self.decoder_input(
            torch.cat([z_expand, query_expand, pooled_expand, marker_expand, type_expand], dim=-1)
        )
        hidden = hidden.reshape(batch_size * n_candidates, self.n_nodes, -1)
        if edge_mask is None:
            decoder_edge_mask = torch.ones(
                edge_index.shape[0], device=x_obs.device, dtype=x_obs.dtype
            )
        elif edge_mask.ndim == 2:
            if edge_mask.shape[0] != batch_size:
                raise ValueError("批量 edge_mask 的第一维必须等于 batch_size")
            decoder_edge_mask = edge_mask.unsqueeze(1).expand(
                batch_size, n_candidates, -1
            ).reshape(batch_size * n_candidates, -1)
        else:
            decoder_edge_mask = edge_mask
        for layer in self.decoder_layers:
            hidden = layer(hidden, edge_index, edge_attr, decoder_edge_mask)
        response = self.output(hidden).reshape(
            batch_size, n_candidates, self.n_nodes, self.time_steps, self.feature_dim
        )
        if candidate_mask is not None and candidate_mask.shape != (batch_size, n_candidates):
            raise ValueError("candidate_mask 必须为 [B,C]")
        result = {
            "response": response,
            "candidate_score": candidate_score,
            "Z": z,
            "z_shared": pooled,
            "candidate_markers": candidate_markers,
            "candidate_types": candidate_types,
        }
        if self.uncertainty:
            logvar = self.output_logvar(hidden).reshape(
                batch_size, n_candidates, self.n_nodes, self.time_steps, self.feature_dim
            )
            result["response_logvar"] = logvar.clamp(-self.logvar_clamp, self.logvar_clamp)
        return result
