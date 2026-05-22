"""Differentiable Grad-CAM and Grad-CAM++ for use as a *training-time* loss.

The standard `pytorch-grad-cam` library returns numpy arrays and breaks the
graph, which is fine for evaluation but useless if we want to backprop a loss
through the CAM. Here we implement the CAM computation with `torch.autograd.grad`
keeping `create_graph=True` so we can take a second backward pass.

Usage:
    cam_extractor = DifferentiableGradCAM(model, target_layer)
    logits = model(x)                       # forward
    cam = cam_extractor.compute(logits, y)  # (B, H', W') in [0,1], differentiable

The forward hook captures the feature map activations on each `model(x)` call.
The target class for each sample is `y` (use ground truth during training).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, plus_plus: bool = False,
                 channels_last_2d: bool = False):
        """If `channels_last_2d` is True, the captured activation is assumed to be
        in `(B, H, W, C)` layout (e.g. timm Swin/SwinV2 blocks) and is permuted to
        `(B, C, H, W)` on the fly so the rest of the CAM math is layout-agnostic.
        """
        self.model = model
        self.target_layer = target_layer
        self.plus_plus = plus_plus
        self.channels_last_2d = channels_last_2d
        self._activations: Optional[torch.Tensor] = None
        self._handle = target_layer.register_forward_hook(self._save_act)

    def _save_act(self, module, inp, out):  # type: ignore[no-untyped-def]
        # Keep the *exact* tensor the model uses downstream so it stays on the
        # autograd graph from `logits`. Layout (BHWC vs BCHW) is normalised in
        # `compute()` via a permute that *is* on the graph from `score` because
        # `feat` is computed from `self._activations` directly.
        self._activations = out

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def compute(self, logits: torch.Tensor, target: torch.Tensor,
                normalize: bool = True) -> torch.Tensor:
        """Return CAM of shape (B, H', W') with gradients enabled."""
        if self._activations is None:
            raise RuntimeError("No activations captured — call model(x) before compute().")
        feat = self._activations
        # Select logit for the target class for each sample
        idx = target.view(-1, 1)
        score = logits.gather(1, idx).sum()
        # Differentiate w.r.t. the *actual* feature node on the graph (keeps create_graph=True valid).
        grads = torch.autograd.grad(score, feat, create_graph=True, retain_graph=True)[0]
        # If the layer emits BHWC (Swin etc.), permute both tensors to BCHW *now*
        # — the graph is already established, so this is a no-op for autograd.
        if self.channels_last_2d and feat.ndim == 4:
            feat = feat.permute(0, 3, 1, 2)
            grads = grads.permute(0, 3, 1, 2)
        # (B, C, H', W')
        if self.plus_plus:
            # Grad-CAM++ weights
            grads2 = grads ** 2
            grads3 = grads ** 3
            denom = 2 * grads2 + (feat * grads3).sum(dim=(2, 3), keepdim=True)
            denom = torch.where(denom != 0, denom, torch.ones_like(denom))
            alpha = grads2 / denom
            weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        else:
            weights = grads.mean(dim=(2, 3), keepdim=True)        # (B, C, 1, 1)
        cam = (weights * feat).sum(dim=1)                          # (B, H', W')
        cam = F.relu(cam)
        if normalize:
            B = cam.shape[0]
            flat = cam.view(B, -1)
            mx = flat.max(dim=1, keepdim=True).values
            mx = torch.clamp(mx, min=1e-8)
            cam = cam / mx.view(B, 1, 1)
        return cam
