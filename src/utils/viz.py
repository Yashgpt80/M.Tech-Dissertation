from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          classes: Sequence[str] = EMOTIONS,
                          normalize: bool = True,
                          out_path: str | Path | None = None,
                          title: str = "Confusion Matrix") -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    cm_norm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm if normalize else cm, annot=True, fmt=".2f" if normalize else "d",
                xticklabels=classes, yticklabels=classes, cmap="Blues", ax=ax,
                cbar=True, square=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def overlay_cam(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """img: HxW or HxWx3 in [0,1]; cam: HxW in [0,1]. Returns HxWx3 in [0,1]."""
    import cv2
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = (1 - alpha) * img + alpha * heat
    return np.clip(out, 0, 1)
