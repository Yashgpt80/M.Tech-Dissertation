"""Multi-XAI qualitative comparison: Grad-CAM, Grad-CAM++, SHAP, LIME.

Picks N samples per class from the test split and produces a single
publication-grade figure with one row per (sample, method).

SHAP uses GradientExplainer (works on any torch model). LIME uses the standard
image_explanation with quickshift segmentation. Both are slow per-sample, so
we keep N small (default 1 per class) and run on CPU if needed.

Usage:
    python -m src.xai_visualize --config configs/xai_guided_resnet50.yaml \
        --ckpt runs/resnet50_xai_guided/best.pt --per_class 1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .data.fer_dataset import build_dataloaders, EMOTIONS
from .models import build_model
from .utils.config import load_config
from .utils.viz import overlay_cam
from .xai.gradcam import DifferentiableGradCAM


def _denorm(x: torch.Tensor) -> np.ndarray:
    img = (x * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img if img.shape[2] > 1 else img[..., 0]


def _to_numpy_image(x: torch.Tensor) -> np.ndarray:
    img = _denorm(x)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


def _shap_attribution(model: torch.nn.Module, x: torch.Tensor, target: int,
                      background: torch.Tensor) -> np.ndarray:
    """Run SHAP GradientExplainer and return a (H,W) saliency map."""
    import shap
    explainer = shap.GradientExplainer(model, background)
    sv = explainer.shap_values(x, nsamples=10)
    # sv shape varies by shap version: list-of-tensors or numpy [B,C,H,W,classes]
    if isinstance(sv, list):
        a = sv[target][0]                       # (C,H,W) for sample 0
    else:
        a = sv[0, ..., target]                  # (C,H,W)
    if a.ndim == 3:
        a = np.abs(a).sum(axis=0)
    return a


def _lime_attribution(model: torch.nn.Module, img_norm: np.ndarray, target: int,
                      device: torch.device, in_channels: int) -> np.ndarray:
    """Run LIME image explanation and return a (H,W) saliency map."""
    from lime import lime_image
    from skimage.segmentation import quickshift

    H, W = img_norm.shape[:2]
    img_lime = img_norm.copy()
    if img_lime.ndim == 2:
        img_lime = np.stack([img_lime] * 3, axis=-1)

    def predict(arr: np.ndarray) -> np.ndarray:
        # arr: (B,H,W,3) in [0,1]
        if in_channels == 1:
            x = arr.mean(axis=-1, keepdims=True)
        else:
            x = arr
        x = (x - 0.5) / 0.5
        t = torch.from_numpy(x.transpose(0, 3, 1, 2)).float().to(device)
        with torch.no_grad():
            p = F.softmax(model(t), dim=1).cpu().numpy()
        return p

    explainer = lime_image.LimeImageExplainer()
    expl = explainer.explain_instance(
        img_lime.astype(np.double), predict, labels=(target,),
        hide_color=0, num_samples=200,
        segmentation_fn=lambda im: quickshift(im, kernel_size=2, max_dist=10, ratio=0.2),
    )
    _, mask = expl.get_image_and_mask(target, positive_only=True, num_features=5, hide_rest=False)
    return mask.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--per_class", type=int, default=1)
    ap.add_argument("--shap_bg", type=int, default=16, help="background batch size for SHAP")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent / "xai_visualize"
    out_dir.mkdir(parents=True, exist_ok=True)

    m_cfg = cfg["model"]; d_cfg = cfg["data"]
    _, _, test_loader, _ = build_dataloaders(
        csv_path=cfg["csv_path"], cache_dir=cfg["cache_dir"],
        image_size=m_cfg["image_size"], in_channels=m_cfg["in_channels"],
        batch_size=32, num_workers=0, use_sampler=False,
    )

    model, target_layer, model_meta = build_model(m_cfg["name"], num_classes=7,
                                                  in_channels=m_cfg["in_channels"], pretrained=False)
    state = torch.load(args.ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)

    # Build SHAP background tensor
    bg_imgs = []
    for batch in test_loader:
        bg_imgs.append(batch[0])
        if sum(b.shape[0] for b in bg_imgs) >= args.shap_bg:
            break
    bg = torch.cat(bg_imgs, dim=0)[:args.shap_bg].to(device)

    # Pick samples per class
    picks: List[Tuple[torch.Tensor, int]] = []
    classes_done = {c: 0 for c in range(len(EMOTIONS))}
    for batch in test_loader:
        x, y = batch[0], batch[1]
        for i in range(len(y)):
            c = int(y[i])
            if classes_done[c] < args.per_class:
                picks.append((x[i].clone(), c))
                classes_done[c] += 1
            if all(v >= args.per_class for v in classes_done.values()):
                break
        if all(v >= args.per_class for v in classes_done.values()):
            break

    cl2d = model_meta["channels_last_2d"]
    cam_ext    = DifferentiableGradCAM(model, target_layer, plus_plus=False, channels_last_2d=cl2d)
    cam_ext_pp = DifferentiableGradCAM(model, target_layer, plus_plus=True,  channels_last_2d=cl2d)

    n = len(picks)
    methods = ["image", "GradCAM", "GradCAM++", "SHAP", "LIME"]
    fig, axes = plt.subplots(len(methods), n, figsize=(2.0 * n, 2.0 * len(methods)))
    if n == 1:
        axes = axes[:, None]
    for j, (img_t, cls) in enumerate(picks):
        x = img_t.unsqueeze(0).to(device)
        y = torch.tensor([cls], device=device)
        img_disp = _to_numpy_image(img_t)
        gray = img_disp.mean(-1)

        # CAMs
        logits = model(x); cam = cam_ext.compute(logits, y, normalize=True)[0].detach().cpu().numpy()
        logits2 = model(x); cam_pp = cam_ext_pp.compute(logits2, y, normalize=True)[0].detach().cpu().numpy()

        # SHAP
        try:
            shap_map = _shap_attribution(model, x, cls, bg)
            shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)
        except Exception as e:
            print(f"[shap fail] {e}")
            shap_map = np.zeros(img_disp.shape[:2])

        # LIME
        try:
            lime_map = _lime_attribution(model, img_disp, cls, device, m_cfg["in_channels"])
        except Exception as e:
            print(f"[lime fail] {e}")
            lime_map = np.zeros(img_disp.shape[:2])

        axes[0, j].imshow(img_disp, cmap="gray" if img_disp.ndim == 2 else None)
        axes[0, j].set_title(EMOTIONS[cls], fontsize=9)
        axes[1, j].imshow(overlay_cam(img_disp, cam))
        axes[2, j].imshow(overlay_cam(img_disp, cam_pp))
        axes[3, j].imshow(overlay_cam(img_disp, shap_map))
        axes[4, j].imshow(overlay_cam(img_disp, lime_map))
        for r in range(len(methods)):
            axes[r, j].axis("off")

    for r, m in enumerate(methods):
        axes[r, 0].set_ylabel(m, fontsize=10, rotation=0, ha="right", va="center")
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

    plt.tight_layout()
    out_path = out_dir / "xai_grid.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    cam_ext.remove(); cam_ext_pp.remove()
    print(f"Saved multi-XAI grid to {out_path}")


if __name__ == "__main__":
    main()
