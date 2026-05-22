"""Canonical anatomical region masks for FER2013.

FER2013 faces are roughly aligned and centered in 48x48 frames, so we don't
need expensive landmark detection. We instead define a *canonical* layout:

    y / 48  : 0 .... 1
    [0.00, 0.20]  forehead
    [0.18, 0.34]  brows
    [0.30, 0.50]  eyes
    [0.46, 0.66]  nose / cheeks
    [0.60, 0.86]  mouth
    [0.84, 1.00]  chin

x ranges are symmetric. Each region is rendered as a soft Gaussian-blurred
elliptical patch and combined per-emotion.

This module is a drop-in replacement for `landmark_masks` and is what the
training pipeline now uses. It produces the same `runs/masks.npz` schema:

    train, val, test : (N, M, M) float16   # M == cam_size
    train_detected, val_detected, test_detected : (N,) bool   # always True here
    cam_size : int
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .fer_dataset import parse_and_cache, EMOTIONS

# Region: list of (cx, cy, rx, ry) ellipses in *normalized* [0,1] face coords.
# Regions are designed to match standard FER2013 facial alignment.
REGIONS: Dict[str, List[Tuple[float, float, float, float]]] = {
    "left_eye":   [(0.32, 0.42, 0.10, 0.06)],
    "right_eye":  [(0.68, 0.42, 0.10, 0.06)],
    "left_brow":  [(0.32, 0.30, 0.12, 0.04)],
    "right_brow": [(0.68, 0.30, 0.12, 0.04)],
    "nose":       [(0.50, 0.55, 0.07, 0.10)],
    "mouth":      [(0.50, 0.74, 0.18, 0.07)],
    "left_cheek": [(0.28, 0.62, 0.09, 0.07)],
    "right_cheek":[(0.72, 0.62, 0.09, 0.07)],
}

REGION_GROUPS: Dict[str, List[str]] = {
    "eyes":   ["left_eye", "right_eye"],
    "brows":  ["left_brow", "right_brow"],
    "mouth":  ["mouth"],
    "nose":   ["nose"],
    "cheeks": ["left_cheek", "right_cheek"],
}

# Per-emotion target groups (same as in plan)
EMOTION_REGIONS: Dict[int, List[str]] = {
    0: ["eyes", "brows"],                 # angry
    1: ["mouth", "nose"],                 # disgust
    2: ["eyes", "brows", "mouth"],        # fear
    3: ["mouth", "cheeks"],               # happy
    4: ["eyes", "mouth"],                 # sad
    5: ["eyes", "brows", "mouth"],        # surprise
    6: ["face"],                          # neutral -> whole face
}


def _draw_region(canvas: np.ndarray, region_name: str) -> None:
    h, w = canvas.shape
    for (cx, cy, rx, ry) in REGIONS[region_name]:
        cv2.ellipse(canvas, center=(int(cx * w), int(cy * h)),
                    axes=(int(rx * w), int(ry * h)),
                    angle=0, startAngle=0, endAngle=360,
                    color=1.0, thickness=-1)


def _whole_face(canvas: np.ndarray) -> None:
    h, w = canvas.shape
    cv2.ellipse(canvas, center=(w // 2, int(h * 0.55)),
                axes=(int(w * 0.40), int(h * 0.50)),
                angle=0, startAngle=0, endAngle=360, color=1.0, thickness=-1)


def build_emotion_mask(emotion: int, size: int) -> np.ndarray:
    """Soft mask in [0,1] of shape (size,size) for the given emotion class."""
    canvas = np.zeros((size, size), dtype=np.float32)
    groups = EMOTION_REGIONS[emotion]
    if "face" in groups:
        _whole_face(canvas)
    else:
        for g in groups:
            for region_name in REGION_GROUPS[g]:
                _draw_region(canvas, region_name)
    sigma = max(1.0, size / 16.0)
    canvas = cv2.GaussianBlur(canvas, ksize=(0, 0), sigmaX=sigma)
    if canvas.max() > 0:
        canvas /= canvas.max()
    return canvas


def build_masks(csv_path: str, cache_dir: str, out_path: str,
                cam_size: int = 14) -> None:
    paths = parse_and_cache(csv_path, cache_dir)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)

    # Class-canonical masks (one per emotion, at cam_size resolution)
    proto = np.stack([build_emotion_mask(c, cam_size) for c in range(len(EMOTIONS))],
                     axis=0)  # (7, M, M)

    out: Dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        y = np.load(paths[f"{split}_y"])
        masks = proto[y].astype(np.float16)        # (N, M, M)
        out[split] = masks
        out[f"{split}_detected"] = np.ones(len(y), dtype=bool)
        print(f"[{split}] N={len(y)}  mask shape per sample: {masks.shape[1:]}")

    np.savez_compressed(out_path, cam_size=cam_size, **out)
    print(f"Saved canonical masks to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="fer2013.csv")
    ap.add_argument("--cache_dir", default="runs/_cache")
    ap.add_argument("--out", default="runs/masks.npz")
    ap.add_argument("--cam_size", type=int, default=14)
    args = ap.parse_args()
    build_masks(args.csv, args.cache_dir, args.out, args.cam_size)


if __name__ == "__main__":
    main()
