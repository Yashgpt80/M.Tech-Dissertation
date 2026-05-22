"""Baseline training entrypoint.

Usage:
    python -m src.train_baseline --config configs/baseline_mini_xception.yaml
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
from .models import build_model
from .utils.config import load_config, save_config
from .utils.logging import JsonlLogger, get_logger
from .utils.seed import set_seed


def cosine_with_warmup(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return LambdaLR(optimizer, lr_lambda)


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
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    return {
        "loss": loss_sum / max(1, n),
        "acc": float(accuracy_score(ys, ps)),
        "macro_f1": float(f1_score(ys, ps, average="macro")),
    }


def train(cfg: dict) -> Path:
    set_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    log = get_logger("baseline", out_dir / "train.log")
    jsonl = JsonlLogger(out_dir / "metrics.jsonl")
    log.info(f"Device: {device}; out_dir: {out_dir}")

    m_cfg, d_cfg, t_cfg = cfg["model"], cfg["data"], cfg["train"]

    train_loader, val_loader, test_loader, cw = build_dataloaders(
        csv_path=cfg["csv_path"], cache_dir=cfg["cache_dir"],
        image_size=m_cfg["image_size"], in_channels=m_cfg["in_channels"],
        batch_size=d_cfg["batch_size"], num_workers=d_cfg["num_workers"],
        use_sampler=d_cfg.get("use_sampler", True),
    )
    log.info(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    model, _, _ = build_model(m_cfg["name"], num_classes=7,
                              in_channels=m_cfg["in_channels"],
                              pretrained=m_cfg.get("pretrained", True))
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model: {m_cfg['name']} | params: {n_params/1e6:.2f}M")
    model.to(device)

    weight = torch.tensor(cw, device=device) if t_cfg.get("use_class_weights", True) else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=t_cfg.get("label_smoothing", 0.0))

    optim = AdamW(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])

    epochs = t_cfg["epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = t_cfg.get("warmup_epochs", 0) * steps_per_epoch
    sched = cosine_with_warmup(optim, total_steps, warmup_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=t_cfg.get("amp", True) and device.type == "cuda")
    best_f1, best_epoch, patience = -1.0, -1, 0
    grad_clip = t_cfg.get("grad_clip", 0.0)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y, _idx in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += loss.item()

        train_loss = running / max(1, len(train_loader))
        val = evaluate(model, val_loader, device)
        dt = time.time() - t0
        log.info(
            f"epoch {epoch+1:03d}/{epochs} | tr_loss {train_loss:.4f} | "
            f"val_loss {val['loss']:.4f} | val_acc {val['acc']*100:.2f}% | "
            f"val_f1 {val['macro_f1']*100:.2f}% | lr {sched.get_last_lr()[0]:.2e} | {dt:.1f}s"
        )
        jsonl.log({"epoch": epoch + 1, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()},
                   "lr": sched.get_last_lr()[0], "elapsed_s": dt})

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_epoch = epoch + 1
            patience = 0
            torch.save({"model": model.state_dict(), "epoch": best_epoch, "val": val},
                       out_dir / "best.pt")
            log.info(f"  >> new best macro-F1 {best_f1*100:.2f}% (saved best.pt)")
        else:
            patience += 1
            if patience >= t_cfg.get("early_stop_patience", 10):
                log.info(f"Early stop at epoch {epoch+1} (no improvement in {patience}).")
                break

    # Final test eval with best weights
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
