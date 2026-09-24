"""Image preprocessing for ImageNet classifiers, using only Pillow and NumPy.

This matches torchvision's standard eval transform (resize the short side,
center-crop, scale to [0, 1], normalize with the ImageNet mean/std), so the
serving image does not need PyTorch.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def load_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def preprocess(image: Image.Image, resize: int = 256, crop: int = 224) -> np.ndarray:
    """Return a float32 array of shape (3, crop, crop)."""
    w, h = image.size
    scale = resize / min(w, h)
    image = image.resize((max(crop, round(w * scale)), max(crop, round(h * scale))),
                         Image.Resampling.BILINEAR)
    w, h = image.size
    left, top = (w - crop) // 2, (h - crop) // 2
    image = image.crop((left, top, left + crop, top + crop))
    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (x - MEAN) / STD


def preprocess_bytes(data: bytes, resize: int = 256, crop: int = 224) -> np.ndarray:
    """Decode and preprocess one image into a batch of one: (1, 3, crop, crop)."""
    return preprocess(load_image(data), resize, crop)[None]
