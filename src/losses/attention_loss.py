"""Attention-alignment losses between a differentiable CAM and a target mask.

All inputs are expected to be in [0, 1] and have shape (B, H, W). The CAM is
resized (bilinear) to match the mask spatial size.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _resize_cam(cam: torch.Tensor, size: int) -> torch.Tensor:
    cam = cam.unsqueeze(1)  # (B,1,H,W)
    cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
    return cam.squeeze(1)


def attention_loss(cam: torch.Tensor, mask: torch.Tensor, kind: str = "cosine") -> torch.Tensor:
    """Compute attention-mask alignment loss.

    cam:  (B, H_c, W_c) in [0,1], differentiable
    mask: (B, H_m, W_m) in [0,1], target
    kind: cosine | bce | mse
    """
    B = cam.shape[0]
    target_size = mask.shape[-1]
    if cam.shape[-1] != target_size:
        cam = _resize_cam(cam, target_size)

    if kind == "cosine":
        c = cam.view(B, -1)
        m = mask.view(B, -1)
        c = c / (c.norm(dim=1, keepdim=True) + 1e-8)
        m = m / (m.norm(dim=1, keepdim=True) + 1e-8)
        sim = (c * m).sum(dim=1)            # (B,)
        return (1.0 - sim).mean()

    if kind == "mse":
        return F.mse_loss(cam, mask)

    if kind == "bce":
        return F.binary_cross_entropy(cam.clamp(0, 1), mask.clamp(0, 1))

    raise ValueError(f"Unknown attention loss kind: {kind}")
