"""FER2013 dataset utilities.

The dataset CSV `fer2013.csv` has three columns:
    emotion   : int in [0, 6]
    pixels    : space-separated 48*48 ints
    Usage     : Training / PublicTest / PrivateTest

This module:
- parses the CSV once and caches images as a single .npy memmap for fast reload.
- exposes a `FER2013Dataset` that returns (image_tensor, label, index) with
  optional Albumentations augmentation.
- builds train/val/test splits matching the standard FER2013 protocol.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(EMOTIONS)
IMG_SIZE_RAW = 48


def parse_and_cache(csv_path: str | Path, cache_dir: str | Path) -> dict:
    """Parse the FER2013 csv once and cache numpy arrays for each split.

    Returns dict of paths to cached .npy files.
    """
    csv_path = Path(csv_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    flag = cache_dir / "DONE"
    paths = {
        "train_x": cache_dir / "train_x.npy",
        "train_y": cache_dir / "train_y.npy",
        "val_x": cache_dir / "val_x.npy",
        "val_y": cache_dir / "val_y.npy",
        "test_x": cache_dir / "test_x.npy",
        "test_y": cache_dir / "test_y.npy",
    }
    if flag.exists() and all(p.exists() for p in paths.values()):
        return {k: str(v) for k, v in paths.items()}

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    def split_arrays(sub: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n = len(sub)
        x = np.zeros((n, IMG_SIZE_RAW, IMG_SIZE_RAW), dtype=np.uint8)
        y = sub["emotion"].to_numpy().astype(np.int64)
        for i, row in enumerate(sub["pixels"].tolist()):
            arr = np.fromstring(row, sep=" ", dtype=np.uint8) \
                if hasattr(np, "fromstring") else np.array(row.split(), dtype=np.uint8)
            x[i] = arr.reshape(IMG_SIZE_RAW, IMG_SIZE_RAW)
        return x, y

    splits = {
        "train": df[df["Usage"] == "Training"],
        "val":   df[df["Usage"] == "PublicTest"],
        "test":  df[df["Usage"] == "PrivateTest"],
    }
    for name, sub in splits.items():
        x, y = split_arrays(sub.reset_index(drop=True))
        np.save(paths[f"{name}_x"], x)
        np.save(paths[f"{name}_y"], y)

    flag.write_text("ok")
    return {k: str(v) for k, v in paths.items()}


def build_transforms(image_size: int, train: bool, in_channels: int = 1) -> A.Compose:
    """Build Albumentations pipeline. Output is float tensor in [0,1], CHW."""
    ops = []
    if image_size != IMG_SIZE_RAW:
        ops.append(A.Resize(image_size, image_size, interpolation=1))
    if train:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.06, scale=(0.92, 1.08), rotate=(-12, 12), p=0.7),
            A.CoarseDropout(num_holes_range=(1, 1),
                            hole_height_range=(image_size // 8, image_size // 5),
                            hole_width_range=(image_size // 8, image_size // 5),
                            fill=0, p=0.3),
        ]
    ops += [
        A.Normalize(mean=[0.5] * in_channels, std=[0.5] * in_channels, max_pixel_value=255.0),
        ToTensorV2(),
    ]
    return A.Compose(ops)


class FER2013Dataset(Dataset):
    """In-memory FER2013 split.

    Parameters
    ----------
    x_path, y_path : cached .npy files (uint8 images, int64 labels)
    image_size     : output H==W after resize (48 keeps native, 224 for ImageNet backbones)
    in_channels    : 1 for grayscale, 3 to repeat channel for pretrained backbones
    transform      : albumentations transform; if None, default eval transform used
    return_index   : whether to return the global sample index
    """

    def __init__(self,
                 x_path: str | Path,
                 y_path: str | Path,
                 image_size: int = 48,
                 in_channels: int = 1,
                 transform: Optional[A.Compose] = None,
                 return_index: bool = True):
        self.x = np.load(x_path)  # (N, 48, 48) uint8
        self.y = np.load(y_path)  # (N,) int64
        assert self.x.shape[0] == self.y.shape[0]
        self.image_size = image_size
        self.in_channels = in_channels
        self.transform = transform if transform is not None else build_transforms(image_size, train=False, in_channels=in_channels)
        self.return_index = return_index

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        img = self.x[idx]  # (48,48) uint8
        if self.in_channels == 3:
            img = np.stack([img, img, img], axis=-1)  # (48,48,3)
        else:
            img = img[..., None]  # (48,48,1)
        out = self.transform(image=img)
        x = out["image"]  # (C,H,W) float
        y = int(self.y[idx])
        if self.return_index:
            return x, y, idx
        return x, y


def class_weights(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    w = counts.sum() / (num_classes * np.clip(counts, 1, None))
    return w.astype(np.float32)


def make_sampler(y: np.ndarray, num_classes: int = NUM_CLASSES) -> WeightedRandomSampler:
    w_per_class = class_weights(y, num_classes)
    sample_weights = w_per_class[y]
    return WeightedRandomSampler(sample_weights.tolist(), num_samples=len(y), replacement=True)


def build_dataloaders(csv_path: str, cache_dir: str, image_size: int, in_channels: int,
                       batch_size: int, num_workers: int = 4, use_sampler: bool = True
                       ) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    paths = parse_and_cache(csv_path, cache_dir)
    train_ds = FER2013Dataset(paths["train_x"], paths["train_y"], image_size, in_channels,
                              transform=build_transforms(image_size, True, in_channels))
    val_ds   = FER2013Dataset(paths["val_x"],   paths["val_y"],   image_size, in_channels,
                              transform=build_transforms(image_size, False, in_channels))
    test_ds  = FER2013Dataset(paths["test_x"],  paths["test_y"],  image_size, in_channels,
                              transform=build_transforms(image_size, False, in_channels))

    train_y = np.load(paths["train_y"])
    cw = class_weights(train_y)

    sampler = make_sampler(train_y) if use_sampler else None
    shuffle = sampler is None
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             persistent_workers=num_workers > 0)
    return train_loader, val_loader, test_loader, cw
