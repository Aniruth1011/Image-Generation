"""Post-synthesis domain randomization (Module 9): rotation, flip, blur,
JPEG compression, noise, stain shift, brightness, contrast, elastic
transform, all with per-transform configurable probabilities.
"""
from __future__ import annotations

import random

import albumentations as A
import numpy as np


def build_augmentation_pipeline(cfg) -> A.Compose:
    """cfg is the `augmentation` sub-config (see configs/augmentation.yaml)."""
    t = cfg.transforms
    transforms = []

    if t.rotate.enabled:
        transforms.append(A.Rotate(limit=t.rotate.limit_degrees, p=t.rotate.p))
    if t.flip.enabled:
        if t.flip.horizontal:
            transforms.append(A.HorizontalFlip(p=t.flip.p))
        if t.flip.vertical:
            transforms.append(A.VerticalFlip(p=t.flip.p))
    if t.gaussian_blur.enabled:
        transforms.append(
            A.GaussianBlur(
                sigma_limit=tuple(t.gaussian_blur.sigma_range), p=t.gaussian_blur.p
            )
        )
    if t.jpeg_compression.enabled:
        transforms.append(
            A.ImageCompression(
                quality_range=tuple(t.jpeg_compression.quality_range),
                p=t.jpeg_compression.p,
            )
        )
    if t.gaussian_noise.enabled:
        lo, hi = t.gaussian_noise.std_range
        transforms.append(
            A.GaussNoise(std_range=(lo, hi), p=t.gaussian_noise.p)
        )
    if t.brightness.enabled or t.contrast.enabled:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=t.brightness.limit if t.brightness.enabled else 0,
                contrast_limit=t.contrast.limit if t.contrast.enabled else 0,
                p=max(t.brightness.p if t.brightness.enabled else 0,
                      t.contrast.p if t.contrast.enabled else 0),
            )
        )
    if t.elastic_transform.enabled:
        transforms.append(
            A.ElasticTransform(
                alpha=t.elastic_transform.alpha,
                sigma=t.elastic_transform.sigma,
                p=t.elastic_transform.p,
            )
        )

    return A.Compose(transforms)


def stain_shift(image: np.ndarray, alpha_range: tuple[float, float], beta_range: tuple[float, float]) -> np.ndarray:
    """Simple linear stain-intensity jitter in optical-density space, applied
    with its own probability outside the albumentations pipeline since it's
    a domain-specific (H&E) transform rather than a generic image op."""
    alpha = random.uniform(*alpha_range)
    beta = random.uniform(*beta_range)
    od = -np.log((image.astype(np.float64) + 1) / 256.0)
    od = od * alpha + beta
    shifted = np.clip(255.0 * np.exp(-od), 0, 255).astype(np.uint8)
    return shifted


def apply_domain_randomization(image: np.ndarray, cfg) -> np.ndarray:
    if not cfg.enabled or random.random() > cfg.apply_probability:
        return image

    pipeline = build_augmentation_pipeline(cfg)
    augmented = pipeline(image=image)["image"]

    t = cfg.transforms
    if t.stain_shift.enabled and random.random() < t.stain_shift.p:
        augmented = stain_shift(
            augmented, tuple(t.stain_shift.alpha_range), tuple(t.stain_shift.beta_range)
        )

    return augmented
