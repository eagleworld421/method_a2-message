"""TCN 时序编码器：将每个母线的动态相量窗口压缩为节点向量。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    """带左侧因果填充的残差卷积块。"""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.dilation = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对时间维进行因果残差变换。"""
        identity = x
        out = F.pad(x, (self.pad, 0))
        out = self.dropout(torch.relu(self.conv1(out)))
        out = F.pad(out, (self.pad, 0))
        out = self.dropout(torch.relu(self.conv2(out)))
        out = self.norm((out + identity).transpose(1, 2)).transpose(1, 2)
        return out


class TemporalEncoder(nn.Module):
    """将 `[B,N,T,F]` 窗口编码为 `[B,N,out_dim]`。"""

    def __init__(
        self,
        in_dim: int = 6,
        hidden_dim: int = 64,
        out_dim: int = 128,
        n_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            _ResidualBlock(hidden_dim, kernel_size, 2 ** i, dropout)
            for i in range(n_layers)
        ])
        self.out_proj = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码每个母线的时间窗口。"""
        if x.ndim != 4:
            raise ValueError("TCN 输入必须为 [B,N,T,F]")
        b, n, t, f = x.shape
        h = self.input_proj(x.reshape(b * n, t, f).transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        pooled = torch.cat([h.mean(dim=-1), h.max(dim=-1).values], dim=-1)
        return self.out_proj(pooled).reshape(b, n, -1)
