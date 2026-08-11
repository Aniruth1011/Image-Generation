"""Artifact detection / removal (Module 3): blur, folds, pen markings, dust,
scanner artifacts, empty patches. Produces a per-patch quality_score,
artifact_type and discard_flag.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PatchQualityReport:
    quality_score: float          # 0 (bad) - 1 (good)
    artifact_type: str            # "none" | "blur" | "fold" | "pen_mark" | "dust" | "empty" | "scanner"
    discard_flag: bool


def _blur_score(gray: np.ndarray) -> float:
    """Variance of Laplacian; low variance => blurry."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _is_empty(image_rgb: np.ndarray, std_threshold: float = 6.0) -> bool:
    return float(image_rgb.std()) < std_threshold


def _pen_mark_score(image_rgb: np.ndarray) -> float:
    """Rough heuristic: strong saturated blue/green/black regions are often
    pen annotations rather than tissue. Returns fraction of pixels flagged."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    pen_mask = (
        ((hue > 90) & (hue < 140) & (sat > 80))  # blue-ish pen
        | ((hue > 40) & (hue < 90) & (sat > 100))  # green-ish pen
    )
    return float(pen_mask.mean())


def _fold_score(gray: np.ndarray) -> float:
    """Tissue folds often appear as very dark, high-contrast streaks;
    approximate via fraction of near-black pixels with high local contrast."""
    dark_mask = gray < 40
    return float(dark_mask.mean())


def _dust_score(gray: np.ndarray) -> float:
    """Small isolated bright specks -> dust/debris."""
    bright_mask = (gray > 245).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(bright_mask)
    return float(num_labels) / (gray.size + 1e-6)


def assess_patch_quality(image_rgb: np.ndarray) -> PatchQualityReport:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    if _is_empty(image_rgb):
        return PatchQualityReport(quality_score=0.0, artifact_type="empty", discard_flag=True)

    blur = _blur_score(gray)
    pen = _pen_mark_score(image_rgb)
    fold = _fold_score(gray)
    dust = _dust_score(gray)

    # normalize sub-scores into [0, 1] "badness" and combine
    blur_bad = 1.0 if blur < 50 else 0.0          # low variance => blurry
    pen_bad = 1.0 if pen > 0.05 else 0.0
    fold_bad = 1.0 if fold > 0.15 else 0.0
    dust_bad = 1.0 if dust > 0.02 else 0.0

    badness = max(blur_bad, pen_bad, fold_bad, dust_bad)
    quality_score = 1.0 - badness

    if pen_bad:
        artifact_type = "pen_mark"
    elif fold_bad:
        artifact_type = "fold"
    elif blur_bad:
        artifact_type = "blur"
    elif dust_bad:
        artifact_type = "dust"
    else:
        artifact_type = "none"

    discard_flag = quality_score < 0.5
    return PatchQualityReport(
        quality_score=quality_score, artifact_type=artifact_type, discard_flag=discard_flag
    )
