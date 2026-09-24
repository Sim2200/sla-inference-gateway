"""ImageNetV2 (matched-frequency) split into a calibration set and an evaluation set.

ImageNetV2 is a fresh test set collected after ImageNet, so pretrained ImageNet
models have never seen it. Folder names are ImageNet class indices (0-999).
A fixed-seed shuffle holds out 500 images for int8 calibration; the other
9,500 are used only for evaluation, so calibration never sees test images.
"""

from __future__ import annotations

import random
from pathlib import Path

DEFAULT_ROOT = Path("data/imagenetv2-matched-frequency-format-val")
CALIBRATION_SIZE = 500
SEED = 1234


def all_images(root: Path = DEFAULT_ROOT) -> list[tuple[Path, int]]:
    items = [(p, int(p.parent.name)) for p in root.glob("*/*.jpeg")]
    items.sort(key=lambda item: str(item[0]))
    if len(items) != 10_000:
        raise SystemExit(f"expected 10,000 ImageNetV2 images under {root}, found {len(items)}")
    return items


def split(root: Path = DEFAULT_ROOT) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Return (calibration, evaluation) lists of (path, label)."""
    items = all_images(root)
    random.Random(SEED).shuffle(items)
    return items[:CALIBRATION_SIZE], items[CALIBRATION_SIZE:]
