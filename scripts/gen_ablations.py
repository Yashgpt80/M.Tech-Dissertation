"""Generate ablation configs for the dissertation sweep.

Three coordinated sub-sweeps to keep total compute bounded but cover all
dissertation claims:

  S1 (lambda)      : ResNet50, mask=landmark, cam=gradcam, lambda in {0.1,0.3,0.5,1.0}
  S2 (mask source) : ResNet50, lambda=BEST_FROM_S1, cam=gradcam, mask in
                     {landmark, class_avg, uniform, random}
  S3 (cam method)  : ResNet50, lambda=BEST_FROM_S1, mask=landmark, cam in
                     {gradcam, gradcam_pp}
  S4 (backbone)    : models {mini_xception, resnet50, efficientnet_b0} with the
                     best (lambda, mask, cam) from S1-S3
                     (mini_xception runs on 48x48 grayscale; the others on 224x224 RGB)

This script just *writes* the configs to `configs/abl/`. The sweep runner
(`scripts/run_sweep.py`) executes them and the notebook aggregates results.
"""
from __future__ import annotations

import copy
from pathlib import Path
import yaml


CONFIG_DIR = Path("configs/abl")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def base_resnet50() -> dict:
    return {
        "run_name": "PLACEHOLDER",
        "seed": 0,
        "csv_path": "fer2013.csv",
        "cache_dir": "runs/_cache",
        "out_dir": "runs",
        "model": {"name": "resnet50", "in_channels": 3, "image_size": 224, "pretrained": True},
        "data": {"batch_size": 48, "num_workers": 4, "use_sampler": True},
        "train": {
            "epochs": 18, "optimizer": "adamw", "lr": 3.0e-4, "weight_decay": 1.0e-4,
            "scheduler": "cosine", "warmup_epochs": 2, "label_smoothing": 0.1,
            "use_class_weights": True, "grad_clip": 1.0, "amp": True,
            "early_stop_patience": 6, "eval_every": 1,
            "init_from": None, "warmup_xai_epochs": 2,
            "lambda_attn": 0.5, "attn_loss": "cosine",
            "cam_method": "gradcam", "mask_source": "landmark",
        },
        "masks": {"npz_path": "runs/masks.npz", "cam_size": 14},
    }


def base_for_model(name: str) -> dict:
    cfg = base_resnet50()
    if name == "mini_xception":
        cfg["model"] = {"name": "mini_xception", "in_channels": 1, "image_size": 48, "pretrained": False}
        cfg["data"]["batch_size"] = 128
        cfg["train"]["epochs"] = 35
        cfg["train"]["lr"] = 1.0e-3
        cfg["train"]["amp"] = True
        cfg["masks"]["cam_size"] = 3   # mini-xception block4 spatial = 3
    elif name == "efficientnet_b0":
        cfg["model"] = {"name": "efficientnet_b0", "in_channels": 3, "image_size": 224, "pretrained": True}
    elif name == "swin_tiny":
        cfg["model"] = {"name": "swin_tiny_patch4_window7_224", "in_channels": 3,
                        "image_size": 224, "pretrained": True}
        cfg["data"]["batch_size"] = 24
        cfg["train"]["lr"] = 5.0e-5
        cfg["train"]["weight_decay"] = 0.05
        cfg["train"]["warmup_epochs"] = 1
        cfg["masks"]["cam_size"] = 7   # Swin-Tiny last stage at 224 input
    return cfg


def write(name: str, cfg: dict) -> Path:
    cfg = copy.deepcopy(cfg)
    cfg["run_name"] = name
    out = CONFIG_DIR / f"{name}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def main():
    paths = []

    # S1: lambda sweep
    for lam in [0.1, 0.3, 0.5, 1.0]:
        cfg = base_resnet50()
        cfg["train"]["lambda_attn"] = lam
        paths.append(write(f"S1_resnet50_lam{lam:g}", cfg))

    # S2: mask source sweep (assume best lambda = 0.5; user can edit later)
    for ms in ["landmark", "class_avg", "uniform", "random"]:
        cfg = base_resnet50()
        cfg["train"]["lambda_attn"] = 0.5
        cfg["train"]["mask_source"] = ms
        paths.append(write(f"S2_resnet50_mask-{ms}", cfg))

    # S3: cam method
    for cam in ["gradcam", "gradcam_pp"]:
        cfg = base_resnet50()
        cfg["train"]["lambda_attn"] = 0.5
        cfg["train"]["cam_method"] = cam
        paths.append(write(f"S3_resnet50_cam-{cam}", cfg))

    # S4: backbone comparison with default best (lam=0.5, landmark, gradcam)
    for m in ["mini_xception", "resnet50", "efficientnet_b0", "swin_tiny"]:
        cfg = base_for_model(m)
        cfg["train"]["lambda_attn"] = 0.5
        # Warm-start from the matching plain baseline so XAI fine-tunes from a
        # well-converged CE checkpoint (matches the default protocol used in S4
        # for ResNet50). If the baseline checkpoint isn't present at run time,
        # `train_xai_guided.py` falls back to fresh init.
        cfg["train"]["init_from"] = f"runs/{m}_baseline/best.pt"
        cfg["train"]["warmup_xai_epochs"] = 1
        paths.append(write(f"S4_{m}_xai", cfg))

    # Pure baselines for the same backbones (lambda=0 == CE only)
    for m in ["mini_xception", "resnet50", "efficientnet_b0", "swin_tiny"]:
        cfg = base_for_model(m)
        cfg["train"]["lambda_attn"] = 0.0
        cfg["train"]["warmup_xai_epochs"] = 9999  # effectively never enable
        paths.append(write(f"S4_{m}_baseline", cfg))

    print(f"Wrote {len(paths)} ablation configs to {CONFIG_DIR}/")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
