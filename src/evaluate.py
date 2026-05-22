"""Evaluation: load checkpoint, run on test split, report metrics, save artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score

from .data.fer_dataset import build_dataloaders, EMOTIONS
from .models import build_model
from .utils.config import load_config
from .utils.viz import plot_confusion_matrix


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader, device: torch.device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
    return np.concatenate(ys), np.concatenate(ps)


def evaluate(cfg: dict, ckpt_path: str | Path, out_dir: str | Path | None = None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_cfg = cfg["model"]
    d_cfg = cfg["data"]

    _, _, test_loader, _ = build_dataloaders(
        csv_path=cfg["csv_path"],
        cache_dir=cfg["cache_dir"],
        image_size=m_cfg["image_size"],
        in_channels=m_cfg["in_channels"],
        batch_size=d_cfg["batch_size"],
        num_workers=d_cfg["num_workers"],
        use_sampler=False,
    )

    model, _, _ = build_model(m_cfg["name"], num_classes=7,
                              in_channels=m_cfg["in_channels"],
                              pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)

    y_true, y_pred = run_inference(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4,
                                   output_dict=True, zero_division=0)
    metrics = {"accuracy": acc, "macro_f1": f1m, "per_class": report}

    if out_dir is None:
        out_dir = Path(ckpt_path).parent / "eval"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    plot_confusion_matrix(y_true, y_pred, out_path=out_dir / "confusion_matrix.png",
                          title=f"{cfg.get('run_name','model')} - test")
    np.savez(out_dir / "preds.npz", y_true=y_true, y_pred=y_pred)

    print(f"Accuracy: {acc*100:.2f}%  Macro-F1: {f1m*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4, zero_division=0))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg, args.ckpt, args.out)


if __name__ == "__main__":
    main()
