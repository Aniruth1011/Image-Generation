"""Tissue-vs-background detection (Module 1: background removal / tissue extraction)."""
from __future__ import annotations

import cv2
import numpy as np


def detect_tissue_mask(
    image_rgb: np.ndarray,
    method: str = "otsu",
) -> np.ndarray:
    """Return a boolean mask (H, W) that is True where tissue is present.

    method:
        "otsu"           -> Otsu threshold on grayscale, tissue = dark regions
        "hsv_saturation"  -> threshold on HSV saturation channel (robust to
                              scanner background which is usually low-saturation white)
    """
    if method == "hsv_saturation":
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask.astype(bool)

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # remove tiny speckles
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


def tissue_percentage(mask: np.ndarray) -> float:
    return float(mask.mean())
