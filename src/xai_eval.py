"""Post-hoc XAI evaluation.

Loads a trained model and computes:
- Quantitative faithfulness on the test split:
    * Deletion AUC (lower is better)
    * Insertion AUC (higher is better)
    * AOPC (Area Over the Perturbation Curve)
    * Pointing-game accuracy vs landmark masks (if masks.npz is provided)
- Qualitative visualizations for a few samples per class:
    * Grad-CAM, Grad-CAM++ overlays
    * SHAP image masker (DeepExplainer or GradientExplainer)
    * LIME superpixel explanation
    Saved as a single grid image.

Usage:
    python -m src.xai_eval --config configs/xai_guided_resnet50.yaml \
        --ckpt runs/resnet50_xai_guided/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data.fer_dataset import build_dataloaders, EMOTIONS
from .models import build_model
from .utils.config import load_config
from .utils.viz import overlay_cam
from .xai.gradcam import DifferentiableGradCAM
from .xai.metrics import faithfulness_correlation, gini_sparsity


# ---------------- Faithfulness ----------------

def _cam_for_batch(model, cam_ext, x, y) -> torch.Tensor:
    logits = model(x)
    cam = cam_ext.compute(logits, y, normalize=True)  # (B,H',W')
    return cam.detach()


def deletion_insertion_curves(model, cam_ext, x, y, steps: int = 20,
                              mode: str = "deletion") -> np.ndarray:
    """Return per-sample probability curve over `steps+1` perturbation levels.

    For 'deletion' we start from the original image and progressively zero-out
    the top-k% most salient pixels; for 'insertion' we start from a blurred
    baseline and progressively reveal salient pixels.
    """
    device = x.device
    B, C, H, W = x.shape
    cam = _cam_for_batch(model, cam_ext, x, y)                 # (B,H',W')
    cam_up = F.interpolate(cam.unsqueeze(1), size=(H, W),
                           mode="bilinear", align_corners=False).squeeze(1)   # (B,H,W)
    # Rank pixels by saliency
    flat = cam_up.view(B, -1)
    order = flat.argsort(dim=1, descending=True)                # most salient first
    n_pix = H * W

    if mode == "insertion":
        base = F.avg_pool2d(x, kernel_size=15, stride=1, padding=7)  # blurred
    else:
        base = torch.zeros_like(x)

    probs = []
    with torch.no_grad():
        for s in range(steps + 1):
            k = int(n_pix * s / steps)
            mask = torch.zeros(B, n_pix, device=device)
            if k > 0:
                mask.scatter_(1, order[:, :k], 1.0)
            mask = mask.view(B, 1, H, W).expand_as(x)
            if mode == "deletion":
                cur = x * (1 - mask) + base * mask
            else:  # insertion: start blurred, reveal pixels
                cur = base * (1 - mask) + x * mask
            logit = model(cur)
            p = F.softmax(logit, dim=1).gather(1, y.view(-1, 1)).squeeze(1)  # (B,)
            probs.append(p.cpu().numpy())
    return np.stack(probs, axis=1)  # (B, steps+1)


def auc_curve(curve: np.ndarray) -> np.ndarray:
    """Trapezoidal AUC normalized by curve length."""
    trapezoid = getattr(np, "trapezoid", None) or np.trapz   # numpy 2 renamed trapz
    return trapezoid(curve, axis=1) / (curve.shape[1] - 1)


def pointing_game(cam: np.ndarray, mask: np.ndarray) -> float:
    """Pointing-game: argmax of CAM lies inside mask>threshold."""
    B = cam.shape[0]
    hits = 0
    thr = mask.reshape(B, -1).max(axis=1) * 0.5
    for i in range(B):
        idx = np.unravel_index(np.argmax(cam[i]), cam[i].shape)
        # Upscale mask to CAM resolution if needed
        m = mask[i]
        if m.shape != cam[i].shape:
            m = cv2.resize(m, (cam[i].shape[1], cam[i].shape[0]),
                           interpolation=cv2.INTER_LINEAR)
        if m[idx] >= thr[i]:
            hits += 1
    return hits / B


# ---------------- Qualitative ----------------

def _denorm(x: torch.Tensor) -> np.ndarray:
    """Reverse Normalize(mean=0.5,std=0.5) to [0,1] HxWxC numpy."""
    img = (x * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    if img.shape[2] == 1:
        img = img[..., 0]
    return img


def qualitative_grid(model, cam_ext, cam_ext_pp, loader, device, out_path: Path,
                     per_class: int = 2) -> None:
    """Make a grid: rows = (image, GradCAM, GradCAM++); cols = samples per class."""
    classes_done = {c: 0 for c in range(len(EMOTIONS))}
    picks: List[Tuple[torch.Tensor, int]] = []
    for batch in loader:
        x, y = batch[0], batch[1]
        for i in range(len(y)):
            c = int(y[i])
            if classes_done[c] < per_class:
                picks.append((x[i].clone(), c))
                classes_done[c] += 1
            if all(v >= per_class for v in classes_done.values()):
                break
        if all(v >= per_class for v in classes_done.values()):
            break

    n = len(picks)
    fig, axes = plt.subplots(3, n, figsize=(2.0 * n, 6.5))
    if n == 1:
        axes = axes[:, None]
    for j, (img_t, cls) in enumerate(picks):
        x = img_t.unsqueeze(0).to(device)
        y = torch.tensor([cls], device=device)
        cam = _cam_for_batch(model, cam_ext, x, y)[0].cpu().numpy()
        cam_pp = _cam_for_batch(model, cam_ext_pp, x, y)[0].cpu().numpy()
        img = _denorm(img_t)
        axes[0, j].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axes[0, j].set_title(EMOTIONS[cls], fontsize=9)
        axes[0, j].axis("off")
        axes[1, j].imshow(overlay_cam(np.atleast_3d(img).mean(-1) if img.ndim == 2 else img, cam))
        axes[1, j].axis("off")
        axes[2, j].imshow(overlay_cam(np.atleast_3d(img).mean(-1) if img.ndim == 2 else img, cam_pp))
        axes[2, j].axis("off")
    axes[0, 0].set_ylabel("image"); axes[1, 0].set_ylabel("Grad-CAM"); axes[2, 0].set_ylabel("Grad-CAM++")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_batches", type=int, default=20,
                    help="Faithfulness uses first N test batches to keep cost bounded.")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent / "xai_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    m_cfg = cfg["model"]; d_cfg = cfg["data"]
    _, _, test_loader, _ = build_dataloaders(
        csv_path=cfg["csv_path"], cache_dir=cfg["cache_dir"],
        image_size=m_cfg["image_size"], in_channels=m_cfg["in_channels"],
        batch_size=min(32, d_cfg["batch_size"]), num_workers=2, use_sampler=False,
    )

    model, target_layer, model_meta = build_model(m_cfg["name"], num_classes=7,
                                                  in_channels=m_cfg["in_channels"], pretrained=False)
    state = torch.load(args.ckpt, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)  # needed for Grad-CAM backprop

    cl2d = model_meta["channels_last_2d"]
    cam_ext    = DifferentiableGradCAM(model, target_layer, plus_plus=False, channels_last_2d=cl2d)
    cam_ext_pp = DifferentiableGradCAM(model, target_layer, plus_plus=True,  channels_last_2d=cl2d)

    # Faithfulness
    del_curves, ins_curves = [], []
    pg_scores = []
    faith_scores: list[np.ndarray] = []
    gini_scores: list[np.ndarray] = []
    masks_npz = cfg.get("masks", {}).get("npz_path")
    test_masks = None
    if masks_npz and Path(masks_npz).exists():
        test_masks = np.load(masks_npz)["test"].astype(np.float32)

    for bi, (x, y, idx) in enumerate(tqdm(test_loader, desc="faithfulness", total=args.max_batches)):
        if bi >= args.max_batches:
            break
        x = x.to(device); y = y.to(device)
        del_curves.append(deletion_insertion_curves(model, cam_ext, x, y, args.steps, "deletion"))
        ins_curves.append(deletion_insertion_curves(model, cam_ext, x, y, args.steps, "insertion"))
        # New metrics: faithfulness correlation + Gini sparsity (both use a single CAM)
        with torch.enable_grad():
            x_g = x.detach().requires_grad_(True)
            logits = model(x_g)
            cam_b_t = cam_ext.compute(logits, y, normalize=True).detach()
        faith_scores.append(faithfulness_correlation(model, x, y, cam_b_t, grid=7))
        gini_scores.append(gini_sparsity(cam_b_t))
        if test_masks is not None:
            cam_b = cam_b_t.cpu().numpy()
            m_b = test_masks[idx.numpy()]
            pg_scores.append(pointing_game(cam_b, m_b))

    del_curves = np.concatenate(del_curves, axis=0)
    ins_curves = np.concatenate(ins_curves, axis=0)
    del_auc = float(auc_curve(del_curves).mean())
    ins_auc = float(auc_curve(ins_curves).mean())
    aopc = float((ins_curves - del_curves).mean())
    pg = float(np.mean(pg_scores)) if pg_scores else None
    faith_arr = np.concatenate(faith_scores)
    gini_arr = np.concatenate(gini_scores)
    faith_mean = float(np.nanmean(faith_arr))
    gini_mean = float(np.mean(gini_arr))

    # Save curves + summary
    np.savez(out_dir / "faithfulness.npz", deletion=del_curves, insertion=ins_curves,
             faithfulness=faith_arr, gini=gini_arr)
    import json
    summary = {"deletion_auc": del_auc, "insertion_auc": ins_auc, "aopc": aopc,
               "pointing_game": pg,
               "faithfulness_corr": faith_mean,
               "gini_sparsity": gini_mean,
               "n_samples": int(del_curves.shape[0])}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(summary)

    # Plot curves
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    xs = np.linspace(0, 1, del_curves.shape[1])
    axs[0].plot(xs, del_curves.mean(0)); axs[0].set_title(f"Deletion (AUC={del_auc:.3f})")
    axs[0].set_xlabel("fraction pixels removed"); axs[0].set_ylabel("p(true class)")
    axs[1].plot(xs, ins_curves.mean(0)); axs[1].set_title(f"Insertion (AUC={ins_auc:.3f})")
    axs[1].set_xlabel("fraction pixels inserted"); axs[1].set_ylabel("p(true class)")
    plt.tight_layout(); fig.savefig(out_dir / "curves.png", dpi=150); plt.close(fig)

    # Qualitative grid
    qualitative_grid(model, cam_ext, cam_ext_pp, test_loader, device,
                     out_dir / "qualitative.png", per_class=2)

    cam_ext.remove(); cam_ext_pp.remove()
    print(f"Saved XAI eval to {out_dir}")


if __name__ == "__main__":
    main()
