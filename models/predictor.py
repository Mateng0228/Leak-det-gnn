"""
Normal-state predictor (denoising + 1-step ahead forecast) for sensor pressures.

Inputs (per batch):
  x:      (B, L_in, S)   - noisy pressures at sensors
  x_time: (B, L_in, 9)   - time features (hour_sin, hour_cos, day-of-week onehot)

Output:
  y_hat:  (B, S)         - predicted next-step clean pressure
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D causal convolution: output at time t depends only on <= t inputs."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.pad > 0:
            y = y[..., :-self.pad]
        return y


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        y = self.conv1(x).transpose(1, 2) # (B, L, C)
        y = self.norm1(y)
        y = F.relu(y)
        y = self.dropout(y).transpose(1, 2) # (B, C, L)

        y = self.conv2(y).transpose(1, 2)
        y = self.norm2(y)
        y = F.relu(y)
        y = self.dropout(y).transpose(1, 2)

        return x + y


class NormalPredictorTCN(nn.Module):
    """Simple TCN over time with channels = (S + time_dim)."""
    def __init__(
        self,
        num_sensors: int,
        time_dim: int = 9,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.time_dim = int(time_dim)
        in_ch = self.num_sensors + self.time_dim

        self.input_proj = nn.Conv1d(in_ch, hidden_channels, kernel_size=1)
        self.tcn = nn.Sequential(*[TCNBlock(hidden_channels, kernel_size, 2**i, dropout) for i in range(num_blocks)])
        self.head = nn.Linear(hidden_channels, self.num_sensors)

    def forward(self, x: torch.Tensor, x_time: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([x, x_time], dim=-1)  # (B, L, S + time_dim)
        feat = feat.transpose(1, 2)            # (B, C, L)
        h = self.input_proj(feat)              # (B, hidden, L)
        h = self.tcn(h)                        # (B, hidden, L)
        h_last = h[:, :, -1]                   # (B, hidden)
        return self.head(h_last)               # (B, S)


class NormalPredictorGRU(nn.Module):
    """GRU baseline predictor."""
    def __init__(
        self,
        num_sensors: int,
        time_dim: int = 9,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.time_dim = int(time_dim)
        in_dim = self.num_sensors + self.time_dim

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, self.num_sensors)

    def forward(self, x: torch.Tensor, x_time: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([x, x_time], dim=-1)  # (B, L, in_dim)
        out, _ = self.gru(feat)
        return self.head(out[:, -1, :])
