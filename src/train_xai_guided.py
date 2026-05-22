"""XAI-guided training entrypoint.

Adds a Grad-CAM attention alignment loss to the standard CE loss:

    L = CE(logits, y) + lambda_attn * L_attn(CAM(x; y), mask(x; y))

Usage:
    python -m src.train_xai_guided --config configs/xai_guided_resnet50.yaml

Requires precomputed masks via `python -m src.data.landmark_masks`.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score

from .data.fer_dataset import build_dataloaders
from .losses import attention_loss
from .models import build_model
from .utils.config import load_config, save_config
from .utils.logging import JsonlLogger, get_logger
from .utils.seed import set_seed
from .xai.gradcam import DifferentiableGradCAM
from .xai.deletion_penalty import deletion_penalty


# ---------- helpers ----------

def cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return LambdaLR(optimizer, lr_lambda)


def load_masks(npz_path: str, target_size: int | None = None,
               blur_sigma: float = 0.0) -> dict:
    """Load masks. If `target_size` is given and differs from cached cam_size,
    masks are bilinearly resized to (target_size, target_size). If
    `blur_sigma > 0`, a Gaussian filter is applied per-sample after resizing
    (softens the mask, helps when CAM resolution is coarse). Renormalised to
    [0,1] per sample in both cases.
    """
    import torch.nn.functional as F
    data = np.load(npz_path)
    out = {"cam_size": int(data["cam_size"])}
    for split in ("train", "val", "test"):
        m = data[split].astype(np.float32)         # (N, S, S)
        if target_size is not None and m.shape[-1] != target_size:
            t = torch.from_numpy(m).unsqueeze(1)   # (N,1,S,S)
            t = F.interpolate(t, size=(target_size, target_size),
                              mode="bilinear", align_corners=False)
            m = t.squeeze(1).numpy()
        if blur_sigma and blur_sigma > 0:
            from scipy.ndimage import gaussian_filter
            for i in range(m.shape[0]):
                m[i] = gaussian_filter(m[i], sigma=float(blur_sigma))
        # renormalize to [0,1] per-sample
        mx = m.reshape(m.shape[0], -1).max(axis=1).reshape(-1, 1, 1)
        m = m / np.clip(mx, 1e-8, None)
        out[split] = m
    if target_size is not None:
        out["cam_size"] = target_size
    return out


def make_mask_source(masks_for_split: np.ndarray, y_split: np.ndarray, kind: str
                     ) -> np.ndarray:
    """Return effective per-sample masks given the chosen mask_source kind."""
    if kind == "landmark":
        return masks_for_split.astype(np.float32)
    if kind == "uniform":
        M = masks_for_split.shape[-1]
        return np.ones((masks_for_split.shape[0], M, M), dtype=np.float32) / (M * M)
    if kind == "class_avg":
        # Replace every mask with the class-average mask
        out = np.zeros_like(masks_for_split, dtype=np.float32)
        for c in np.unique(y_split):
            idx = np.where(y_split == c)[0]
            avg = masks_for_split[idx].astype(np.float32).mean(axis=0)
            mx = avg.max() if avg.max() > 0 else 1.0
            out[idx] = avg / mx
        return out
    if kind == "random":
        # Per-class random mask (same per class) — sanity check
        rng = np.random.default_rng(0)
        M = masks_for_split.shape[-1]
        per_class = {c: rng.random((M, M)).astype(np.float32) for c in range(int(y_split.max()) + 1)}
        for m in per_class.values():
            m /= m.max()
        return np.stack([per_class[int(c)] for c in y_split], axis=0)
    raise ValueError(f"Unknown mask_source: {kind}")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += ce(logits, y).item()
        n += y.numel()
        ys.append(y.cpu().numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    return {"loss": loss_sum / max(1, n),
            "acc": float(accuracy_score(ys, ps)),
            "macro_f1": float(f1_score(ys, ps, average="macro"))}


# ---------- main training loop ----------

def train(cfg: dict) -> Path:
    set_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    log = get_logger("xai_guided", out_dir / "train.log")
    jsonl = JsonlLogger(out_dir / "metrics.jsonl")
    log.info(f"Device: {device}; out_dir: {out_dir}")

    m_cfg, d_cfg, t_cfg = cfg["model"], cfg["data"], cfg["train"]

    train_loader, val_loader, test_loader, cw = build_dataloaders(
        csv_path=cfg["csv_path"], cache_dir=cfg["cache_dir"],
        image_size=m_cfg["image_size"], in_channels=m_cfg["in_channels"],
        batch_size=d_cfg["batch_size"], num_workers=d_cfg["num_workers"],
        use_sampler=d_cfg.get("use_sampler", True),
    )

    # Load masks
    masks_npz = cfg["masks"]["npz_path"]
    if not Path(masks_npz).exists():
        raise FileNotFoundError(
            f"Mask cache not found at {masks_npz}. "
            f"Run: python -m src.data.landmark_masks --csv {cfg['csv_path']} --out {masks_npz}"
        )
    target_cam_size = cfg["masks"].get("cam_size")
    blur_sigma = float(cfg["masks"].get("blur_sigma", 0.0))
    masks = load_masks(masks_npz, target_size=target_cam_size, blur_sigma=blur_sigma)
    log.info(f"Masks: cam_size={masks['cam_size']}; train shape={masks['train'].shape}; "
             f"blur_sigma={blur_sigma}")
    # Re-load raw labels to construct effective masks
    train_y = np.load(Path(cfg["cache_dir"]) / "train_y.npy")
    mask_source = t_cfg.get("mask_source", "landmark")
    train_masks = make_mask_source(masks["train"], train_y, mask_source)
    train_masks_t = torch.from_numpy(train_masks).float()    # on CPU; indexed per-batch
    log.info(f"Mask source: {mask_source} | nonzero ratio: {(train_masks > 0).mean():.3f}")

    # Model
    target_override = m_cfg.get("gradcam_layer", "auto")
    model, target_layer, model_meta = build_model(m_cfg["name"], num_classes=7,
                                                  in_channels=m_cfg["in_channels"],
                                                  pretrained=m_cfg.get("pretrained", True),
                                                  target_override=target_override)
    if t_cfg.get("init_from"):
        log.info(f"Loading init weights from {t_cfg['init_from']}")
        state = torch.load(t_cfg["init_from"], map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)

    # Optional: freeze early stages (e.g. Swin layers[0,1]). The config field
    # `train.freeze_stages: [int,...]` is interpreted as indices into
    # `model.layers` for Swin-family backbones; for non-Swin backbones the
    # equivalent attribute is `model.layer{i+1}` (ResNet) or `model.blocks[i]`
    # (EfficientNet) — we just look up the first matching attribute.
    freeze_stages = t_cfg.get("freeze_stages") or []
    if freeze_stages:
        for i in freeze_stages:
            stage = None
            if hasattr(model, "layers"):                       # Swin / SwinV2
                stage = model.layers[i]
            elif hasattr(model, f"layer{i+1}"):                # ResNet
                stage = getattr(model, f"layer{i+1}")
            elif hasattr(model, "blocks"):                     # EfficientNet
                stage = model.blocks[i]
            if stage is None:
                log.warning(f"freeze_stages: could not locate stage {i}; skipping")
                continue
            for p in stage.parameters():
                p.requires_grad_(False)
            stage.eval()
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        log.info(f"Frozen stages {freeze_stages}: {n_frozen/1e6:.2f}M params no-grad")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {m_cfg['name']} | params: {n_params/1e6:.2f}M total / "
             f"{n_trainable/1e6:.2f}M trainable | "
             f"target_layer: {type(target_layer).__name__} (override={target_override}) | "
             f"channels_last_2d: {model_meta['channels_last_2d']}")
    model.to(device)

    cam_extractor = DifferentiableGradCAM(
        model, target_layer,
        plus_plus=(t_cfg.get("cam_method", "gradcam") == "gradcam_pp"),
        channels_last_2d=model_meta["channels_last_2d"],
    )

    weight = torch.tensor(cw, device=device) if t_cfg.get("use_class_weights", True) else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=t_cfg.get("label_smoothing", 0.0))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(trainable, lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])
    epochs = t_cfg["epochs"]
    steps_per_epoch = len(train_loader)
    sched = cosine_with_warmup(optim, epochs * steps_per_epoch,
                                t_cfg.get("warmup_epochs", 0) * steps_per_epoch)

    lambda_attn = float(t_cfg.get("lambda_attn", 0.5))
    warmup_xai = int(t_cfg.get("warmup_xai_epochs", 0))
    attn_kind = t_cfg.get("attn_loss", "cosine")
    grad_clip = t_cfg.get("grad_clip", 0.0)
    lambda_del = float(t_cfg.get("lambda_deletion", 0.0))
    del_top_frac = float(t_cfg.get("deletion_top_frac", 0.10))

    best_f1, best_epoch, patience = -1.0, -1, 0
    for epoch in range(epochs):
        model.train()
        # Keep frozen stages in eval mode (BN/dropout stay deterministic on frozen feats).
        if freeze_stages:
            for i in freeze_stages:
                stage = None
                if hasattr(model, "layers"):
                    stage = model.layers[i]
                elif hasattr(model, f"layer{i+1}"):
                    stage = getattr(model, f"layer{i+1}")
                elif hasattr(model, "blocks"):
                    stage = model.blocks[i]
                if stage is not None:
                    stage.eval()
        t0 = time.time()
        sums = {"ce": 0.0, "attn": 0.0, "del": 0.0, "total": 0.0, "n_batches": 0}
        use_attn = epoch >= warmup_xai

        for x, y, idx in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)

            logits = model(x)
            loss_ce = criterion(logits, y)
            loss = loss_ce

            if use_attn:
                cam = cam_extractor.compute(logits, y, normalize=True)  # (B, H', W')
                # Gather per-sample target mask
                m_batch = train_masks_t[idx].to(device, non_blocking=True)  # (B, M, M)
                loss_attn = attention_loss(cam, m_batch, kind=attn_kind)
                loss = loss + lambda_attn * loss_attn
                sums["attn"] += float(loss_attn.detach())

                if lambda_del > 0:
                    loss_del = deletion_penalty(
                        model, x, y, cam, criterion=criterion, top_frac=del_top_frac,
                    )
                    loss = loss + lambda_del * loss_del
                    sums["del"] += float(loss_del.detach())

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optim.step()
            sched.step()

            sums["ce"] += float(loss_ce.detach())
            sums["total"] += float(loss.detach())
            sums["n_batches"] += 1

        n_b = max(1, sums["n_batches"])
        val = evaluate(model, val_loader, device)
        dt = time.time() - t0
        log.info(
            f"epoch {epoch+1:03d}/{epochs} | "
            f"ce {sums['ce']/n_b:.4f} | attn {sums['attn']/n_b:.4f} (on={use_attn}) | "
            f"del {sums['del']/n_b:.4f} | "
            f"val_acc {val['acc']*100:.2f}% | val_f1 {val['macro_f1']*100:.2f}% | "
            f"lr {sched.get_last_lr()[0]:.2e} | {dt:.1f}s"
        )
        jsonl.log({
            "epoch": epoch + 1,
            "train_ce": sums["ce"] / n_b,
            "train_attn": sums["attn"] / n_b,
            "train_del": sums["del"] / n_b,
            "use_attn": use_attn,
            **{f"val_{k}": v for k, v in val.items()},
            "lr": sched.get_last_lr()[0], "elapsed_s": dt,
        })

        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, patience = val["macro_f1"], epoch + 1, 0
            torch.save({"model": model.state_dict(), "epoch": best_epoch, "val": val},
                       out_dir / "best.pt")
            log.info(f"  >> new best macro-F1 {best_f1*100:.2f}%")
        else:
            patience += 1
            if patience >= t_cfg.get("early_stop_patience", 10):
                log.info(f"Early stop at epoch {epoch+1}.")
                break

    cam_extractor.remove()

    # Final test evaluation with best weights
    ckpt = torch.load(out_dir / "best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    test = evaluate(model, test_loader, device)
    log.info(f"BEST epoch {best_epoch} | TEST acc {test['acc']*100:.2f}% | macro-F1 {test['macro_f1']*100:.2f}%")
    jsonl.log({"final_test": test, "best_epoch": best_epoch})
    return out_dir / "best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
