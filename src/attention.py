"""可复用 SE / CBAM 注意力模块，用于自定义网络实验。"""
import torch
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                                  nn.SiLU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                                     nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1))
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_gate = torch.sigmoid(self.channel(x) + self.channel(torch.amax(x, 1, keepdim=True)))
        x = x * channel_gate
        spatial = torch.cat((torch.mean(x, 1, keepdim=True), torch.amax(x, 1, keepdim=True)), 1)
        return x * torch.sigmoid(self.spatial(spatial))
