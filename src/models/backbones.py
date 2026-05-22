"""Backbone models via `timm` plus a uniform builder for baselines.

`build_model(name, ...)` returns a tuple `(model, target_layer, meta)` where
`target_layer` is the layer to hook for Grad-CAM and `meta` is a small dict
describing layout-specific quirks (currently `channels_last_2d` for Swin-style
hierarchical transformers whose blocks emit `(B, H, W, C)` instead of
`(B, C, H, W)`).
"""
from __future__ import annotations

from typing import Tuple, Dict, Any

import torch.nn as nn

from .mini_xception import MiniXception


def build_timm_model(name: str, num_classes: int, in_channels: int = 3,
                     pretrained: bool = True, drop_rate: float = 0.2,
                     target_override: str = "auto",
                     ) -> Tuple[nn.Module, nn.Module, Dict[str, Any]]:
    import timm
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes,
                              in_chans=in_channels, drop_rate=drop_rate)
    target_layer, meta = _find_gradcam_target(name, model, override=target_override)
    return model, target_layer, meta


def _find_gradcam_target(name: str, model: nn.Module,
                         override: str = "auto") -> Tuple[nn.Module, Dict[str, Any]]:
    """Locate the Grad-CAM target layer.

    `override` controls non-default hook points:
      - "auto"        : the default (final feature stage).
      - "penultimate" : second-to-last stage; useful for Swin to get a finer 14x14
                        CAM instead of the coarse 7x7 final stage.
    """
    n = name.lower()
    meta: Dict[str, Any] = {"channels_last_2d": False}
    if "resnet" in n:
        if override == "penultimate":
            return model.layer3[-1], meta
        return model.layer4[-1], meta
    if "efficientnet" in n:
        # timm efficientnet exposes `conv_head` / blocks[-1]
        if override == "penultimate":
            return model.blocks[-2], meta
        return model.blocks[-1], meta
    if "convnext" in n:
        if override == "penultimate":
            return model.stages[-2].blocks[-1], meta
        return model.stages[-1].blocks[-1], meta
    if "swin" in n:
        # timm hierarchical Swin / SwinV2: model.layers[i].blocks[j]; output is BHWC
        meta["channels_last_2d"] = True
        if override == "penultimate":
            # Stage -2: 14x14 spatial, 384 channels at swin_tiny@224
            return model.layers[-2].blocks[-1], meta
        return model.layers[-1].blocks[-1], meta
    if "vit" in n or "deit" in n:
        # ViT/DeiT: hook last block's norm1; outputs (B, N, C) tokens — caller must reshape
        meta["channels_last_2d"] = False  # token layout; not currently supported by gradcam
        return model.blocks[-1].norm1, meta
    # Fallback: last Conv2d
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError(f"Could not auto-locate Grad-CAM target layer for {name}")
    return last_conv, meta


def build_model(name: str, num_classes: int, in_channels: int,
                pretrained: bool = True,
                target_override: str = "auto",
                ) -> Tuple[nn.Module, nn.Module, Dict[str, Any]]:
    """Top-level factory used by training scripts.

    Returns ``(model, target_layer, meta)``. ``meta`` carries layout flags such
    as ``channels_last_2d`` (True for Swin-family backbones).

    ``target_override`` (str) lets configs request a non-default Grad-CAM hook
    point. Currently supported: ``"auto"`` (default) and ``"penultimate"``.
    """
    if name == "mini_xception":
        # mini_xception only has the canonical target layer; override is a no-op.
        m = MiniXception(num_classes=num_classes, in_channels=in_channels)
        return m, m.gradcam_target_layer, {"channels_last_2d": False}
    return build_timm_model(name, num_classes, in_channels=in_channels,
                            pretrained=pretrained, target_override=target_override)
