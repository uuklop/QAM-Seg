"""ConvLiteUNet: capacity-matched CNN stand-in for MambaLiteUNet.

Same ``c_list`` depths and U-Net topology (6-stage encoder / decoder with
skip fusion). Replaces Adaptive Multi-branch Mamba blocks with residual
conv blocks so MQE+QGC SSL experiments can run without ``mamba_ssm``.

This is NOT MambaLiteUNet and must not be reported as Mamba metrics.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )
        self.skip = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class AttentionGate(nn.Module):
    """Lightweight additive attention gate (no Mamba)."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.GroupNorm(min(8, F_int), F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.GroupNorm(min(8, F_int), F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=True)
        a = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(a)


class ConvLiteUNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        input_channels: int = 3,
        c_list=None,
    ):
        super().__init__()
        if c_list is None:
            c_list = [16, 32, 48, 64, 96, 128]
        assert len(c_list) == 6
        self.c_list = list(c_list)

        self.encoder1 = ConvBlock(input_channels, c_list[0])
        self.encoder2 = ConvBlock(c_list[0], c_list[1])
        self.encoder3 = ConvBlock(c_list[1], c_list[2])
        self.encoder4 = ConvBlock(c_list[2], c_list[3])
        self.encoder5 = ConvBlock(c_list[3], c_list[4])
        self.encoder6 = ConvBlock(c_list[4], c_list[5])

        self.decoder1 = ConvBlock(c_list[5], c_list[4])
        self.decoder2 = ConvBlock(c_list[4], c_list[3])
        self.decoder3 = ConvBlock(c_list[3], c_list[2])
        self.decoder4 = ConvBlock(c_list[2], c_list[1])
        self.decoder5 = ConvBlock(c_list[1], c_list[0])

        self.attention5 = AttentionGate(c_list[4], c_list[4], max(c_list[4] // 2, 8))
        self.attention4 = AttentionGate(c_list[3], c_list[3], max(c_list[3] // 2, 8))
        self.attention3 = AttentionGate(c_list[2], c_list[2], max(c_list[2] // 2, 8))
        self.attention2 = AttentionGate(c_list[1], c_list[1], max(c_list[1] // 2, 8))

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with pooling (mirrors MambaLiteUNet spatial schedule)
        e1 = F.gelu(F.max_pool2d(self.encoder1(x), 2, 2))
        e2 = F.gelu(F.max_pool2d(self.encoder2(e1), 2, 2))
        e3 = F.gelu(F.max_pool2d(self.encoder3(e2), 2, 2))
        e4 = F.gelu(F.max_pool2d(self.encoder4(e3), 2, 2))
        e5 = F.gelu(F.max_pool2d(self.encoder5(e4), 2, 2))
        bottleneck = F.gelu(self.encoder6(e5))

        d5 = F.gelu(self.decoder1(bottleneck))
        d5 = d5 + self.attention5(g=d5, x=e5)

        d4 = F.gelu(self.decoder2(d5))
        d4 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True)
        d4 = d4 + self.attention4(g=d4, x=e4)

        d3 = F.gelu(self.decoder3(d4))
        d3 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)
        d3 = d3 + self.attention3(g=d3, x=e3)

        d2 = F.gelu(self.decoder4(d3))
        d2 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)
        d2 = d2 + self.attention2(g=d2, x=e2)

        d1 = F.gelu(self.decoder5(d2))
        d1 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=True)
        d1 = d1 + e1

        out = F.interpolate(self.final(d1), scale_factor=2, mode="bilinear", align_corners=True)
        return torch.sigmoid(out)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = ConvLiteUNet()
    x = torch.randn(1, 3, 256, 256)
    y = m(x)
    print(y.shape, count_parameters(m))
