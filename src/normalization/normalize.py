"""Config-driven dispatcher: select Macenko or Vahadane at runtime (Module 2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.normalization.macenko import MacenkoNormalizer
from src.normalization.vahadane import VahadaneNormalizer


def build_normalizer(method: str, alpha: float = 1.0, beta: float = 0.15,
                      luminosity_threshold: float = 0.8):
    if method == "macenko":
        return MacenkoNormalizer(alpha=alpha, beta=beta, luminosity_threshold=luminosity_threshold)
    if method == "vahadane":
        return VahadaneNormalizer(luminosity_threshold=luminosity_threshold)
    raise ValueError(f"Unknown normalization method: {method}")


def normalize_patch_file(
    patch_path: str | Path,
    out_path: str | Path,
    normalizer,
) -> None:
    image = np.array(Image.open(patch_path).convert("RGB"))
    normalized = normalizer.transform(image)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalized).save(out_path)
