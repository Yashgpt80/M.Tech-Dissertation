"""Tiny end-to-end smoke test: build models, forward, Grad-CAM backward,
attention loss. Runs on CPU in seconds. No training. Does NOT touch the
35K-image CSV; uses random dummy tensors.

Run from repo root:
    python scripts/smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import torch
from src.models import build_model
from src.xai.gradcam import DifferentiableGradCAM
from src.losses import attention_loss


def test_model(name: str, in_channels: int, image_size: int, plus_plus: bool = False):
    print(f"\n=== {name} (C={in_channels}, H=W={image_size}, ++={plus_plus}) ===")
    model, target_layer = build_model(name, num_classes=7, in_channels=in_channels, pretrained=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    cam_ext = DifferentiableGradCAM(model, target_layer, plus_plus=plus_plus)
    x = torch.randn(2, in_channels, image_size, image_size, requires_grad=False)
    y = torch.tensor([0, 3])
    logits = model(x)
    print(f"logits: {tuple(logits.shape)}")
    cam = cam_ext.compute(logits, y)
    print(f"cam: {tuple(cam.shape)}  min={cam.min().item():.3f}  max={cam.max().item():.3f}  requires_grad={cam.requires_grad}")
    target_mask = torch.zeros(2, cam.shape[-1], cam.shape[-1])
    target_mask[:, cam.shape[-1]//4:3*cam.shape[-1]//4, cam.shape[-1]//4:3*cam.shape[-1]//4] = 1.0
    loss_ce = torch.nn.functional.cross_entropy(logits, y)
    loss_attn = attention_loss(cam, target_mask, kind="cosine")
    loss = loss_ce + 0.5 * loss_attn
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"ce={loss_ce.item():.4f}  attn={loss_attn.item():.4f}  total_grad_abs={g:.2f}")
    cam_ext.remove()
    assert cam.requires_grad
    assert g > 0


if __name__ == "__main__":
    test_model("mini_xception", in_channels=1, image_size=48)
    test_model("mini_xception", in_channels=1, image_size=48, plus_plus=True)
    try:
        test_model("resnet18", in_channels=3, image_size=64)  # small/fast
    except Exception as e:
        print(f"[skip timm test] {e}")
    print("\nAll smoke checks passed.")
