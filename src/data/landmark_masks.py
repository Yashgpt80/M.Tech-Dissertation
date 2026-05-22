"""Compatibility shim: previously used MediaPipe; now delegates to canonical
region masks (`region_masks.py`) which work better on 48x48 FER2013 images.

Keeping the same module name so old `python -m src.data.landmark_masks` calls
in the README still work.
"""
from .region_masks import build_masks, main, EMOTION_REGIONS, build_emotion_mask  # noqa: F401

if __name__ == "__main__":
    main()
