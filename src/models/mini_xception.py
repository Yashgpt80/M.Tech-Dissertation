"""Mini-Xception (Arriaga et al. 2017) - classic compact FER2013 baseline.

Architecture: depthwise-separable residual blocks operating on 48x48 grayscale.
~60k parameters. The last conv block before global pooling is exposed as
`last_conv` for Grad-CAM hooks.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_bn_relu(in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class SeparableConv(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.pw(self.dw(x)))


class XceptionBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.sep1 = SeparableConv(in_c, out_c)
        self.sep2 = SeparableConv(out_c, out_c)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.sep1(x)
        out = self.relu(out)
        out = self.sep2(out)
        out = self.pool(out)
        return out + res


class MiniXception(nn.Module):
    def __init__(self, num_classes: int = 7, in_channels: int = 1, width: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            _conv_bn_relu(in_channels, width, k=3, s=1, p=1),
            _conv_bn_relu(width, width, k=3, s=1, p=1),
        )
        c1, c2, c3, c4 = width * 2, width * 4, width * 8, width * 16
        self.block1 = XceptionBlock(width, c1)
        self.block2 = XceptionBlock(c1, c2)
        self.block3 = XceptionBlock(c2, c3)
        self.block4 = XceptionBlock(c3, c4)
        self.last_conv = nn.Conv2d(c4, num_classes, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x  # feature map for Grad-CAM hooks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)
        logits_map = self.last_conv(f)        # (B, C, H', W')
        logits = self.gap(logits_map).flatten(1)
        return logits

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Target layer for Grad-CAM (feature map before the class conv)."""
        return self.block4
