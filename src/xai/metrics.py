"""Additional XAI evaluation metrics beyond deletion/insertion AUC.

- `faithfulness_correlation`: Spearman rank-correlation between per-patch CAM
  intensity and the drop in target-class logit when that patch is masked out.
  Higher = more faithful (the CAM ordering predicts which regions the model
  actually relies on). Bhatt et al., 2020.

- `gini_sparsity`: Gini coefficient of the flattened CAM. 0 = uniform, ~1 =
  single peak. Useful for diagnosing over-concentrated explanations (e.g.,
  the Swin-Tiny XAI variant whose deletion-AUC went up because attention is
  forced onto a tight mask region).

Both functions are designed to be cheap enough to add to `src.xai_eval`'s
main loop (~10% overhead).
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Faithfulness correlation (patch-based)
# ---------------------------------------------------------------------------

def _spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    """Pure-numpy Spearman correlation; returns NaN if input is constant."""
    if a.size < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


@torch.no_grad()
def faithfulness_correlation(model: torch.nn.Module,
                             x: torch.Tensor,             # (B, C, H, W) on device
                             y: torch.Tensor,             # (B,) int64 on device
                             cam: torch.Tensor,           # (B, h, w) saliency
                             grid: int = 7,
                             ) -> np.ndarray:
    """Split each image into a `grid x grid` partition, mask each cell with the
    channel-wise mean, measure the drop in the target-class softmax probability,
    and return per-sample Spearman correlation between the cell-mean CAM value
    and the prediction drop. Higher is better.
    """
    B, C, H, W = x.shape
    device = x.device

    # Cell sizes (may be uneven on the right/bottom; we use np.array_split semantics)
    h_bounds = np.linspace(0, H, grid + 1, dtype=int)
    w_bounds = np.linspace(0, W, grid + 1, dtype=int)

    # Per-channel mean over each image (the "neutral" filler we paint in)
    fill = x.mean(dim=(2, 3), keepdim=True)               # (B, C, 1, 1)

    # Baseline confidence on the true class
    p0 = F.softmax(model(x), dim=1).gather(1, y.view(-1, 1)).squeeze(1)  # (B,)
    p0_np = p0.cpu().numpy()

    # Upsample CAM to image resolution once
    cam_up = F.interpolate(cam.unsqueeze(1).float(), size=(H, W),
                           mode="bilinear", align_corners=False).squeeze(1)
    cam_np = cam_up.cpu().numpy()                          # (B, H, W)

    cell_cams = np.zeros((B, grid * grid), dtype=np.float32)
    cell_drops = np.zeros((B, grid * grid), dtype=np.float32)

    k = 0
    for ih in range(grid):
        for iw in range(grid):
            y0, y1 = h_bounds[ih], h_bounds[ih + 1]
            x0, x1 = w_bounds[iw], w_bounds[iw + 1]
            xp = x.clone()
            xp[:, :, y0:y1, x0:x1] = fill                  # blank the cell
            p = F.softmax(model(xp), dim=1).gather(1, y.view(-1, 1)).squeeze(1)
            cell_drops[:, k] = (p0 - p).cpu().numpy()      # positive when model lost confidence
            cell_cams[:, k] = cam_np[:, y0:y1, x0:x1].mean(axis=(1, 2))
            k += 1

    corrs = np.array([_spearmanr(cell_cams[i], cell_drops[i]) for i in range(B)],
                     dtype=np.float32)
    return corrs                                           # (B,) with possible NaNs


# ---------------------------------------------------------------------------
# Sparsity (Gini coefficient)
# ---------------------------------------------------------------------------

def gini_sparsity(cam: torch.Tensor | np.ndarray) -> np.ndarray:
    """Gini coefficient of flattened CAM values per sample.

    0 = perfectly uniform, ~1 = single-pixel peak. We follow the standard
    formula on non-negative values (CAM is ReLU'd upstream).
    """
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()
    cam = np.asarray(cam, dtype=np.float64)
    if cam.ndim == 2:
        cam = cam[None]
    B = cam.shape[0]
    out = np.zeros(B, dtype=np.float32)
    for i in range(B):
        v = cam[i].reshape(-1)
        v = np.clip(v, 0.0, None)
        s = v.sum()
        if s <= 1e-12:
            out[i] = 0.0
            continue
        v = np.sort(v)
        n = v.size
        # Gini = (2 * sum_{i} i * v_i) / (n * sum(v))  -  (n+1)/n
        idx = np.arange(1, n + 1)
        out[i] = float((2.0 * (idx * v).sum()) / (n * s) - (n + 1.0) / n)
    return out
