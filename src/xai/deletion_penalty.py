"""Deletion-penalty auxiliary loss.

Motivation: on Swin-Tiny the cosine attention loss successfully drives
pointing-game to 100% but *increases* deletion AUC — i.e. the model becomes
robust to masking out the very pixels its CAM claims are important. The CAM
no longer reflects model reliance.

The deletion penalty fixes this by adding a term that *requires* the
prediction confidence on the true class to drop when the top-`top_frac`
CAM-highlighted pixels are removed from the input. Formally:

    loss_del = -CE(model(x_masked), y)     # NEGATIVE CE: we MINIMISE this,
                                            # so we MAXIMISE the deletion CE,
                                            # i.e. drive confidence DOWN.

A single extra forward pass per step (~30 % step-time overhead).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def deletion_penalty(model: nn.Module,
                     x: torch.Tensor,             # (B, C, H, W)
                     y: torch.Tensor,             # (B,) int64
                     cam: torch.Tensor,           # (B, h, w), already on graph
                     criterion: nn.Module | None = None,    # unused, kept for compat
                     top_frac: float = 0.10,
                     ) -> torch.Tensor:
    """Return a scalar penalty: lower = model is more sensitive to deleting
    its own top-saliency region. The returned tensor is detached from `cam`'s
    autograd graph w.r.t. the mask choice (the mask is computed without
    gradient) but *is* on the graph through the model weights.
    """
    B, C, H, W = x.shape
    # Upsample CAM to input resolution; do not track grads through the mask itself.
    with torch.no_grad():
        cam_up = F.interpolate(cam.unsqueeze(1).float(), size=(H, W),
                               mode="bilinear", align_corners=False).squeeze(1)
        flat = cam_up.view(B, -1)
        k = max(1, int(flat.shape[1] * top_frac))
        # Pixels strictly above the (1-top_frac)-quantile are "salient".
        thr = flat.kthvalue(flat.shape[1] - k + 1, dim=1).values    # (B,)
        mask = (cam_up >= thr.view(B, 1, 1)).float()                # (B, H, W)
        mask = mask.unsqueeze(1)                                    # (B, 1, H, W)
        fill = x.mean(dim=(2, 3), keepdim=True)                     # neutral filler

    x_del = x * (1.0 - mask) + fill * mask                          # blank top-k%
    logits_del = model(x_del)
    # Hinge penalty: only fires while the model is still *confident* on the
    # true class after deletion (logp_true > margin). Once logp_true falls
    # below the margin, the penalty is 0 with no gradient — preventing the
    # optimiser from spiralling into pathologically wrong logits on OOD
    # deleted inputs (which can have unbounded log-softmax magnitudes).
    margin = -2.0           # exp(-2) ≈ 0.135 → "uncertain enough"
    logp = torch.log_softmax(logits_del, dim=1).gather(1, y.view(-1, 1)).squeeze(1)
    return (logp - margin).clamp(min=0.0).mean()
