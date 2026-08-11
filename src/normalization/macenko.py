"""Macenko stain normalization (Module 2).

Reference: Macenko et al., "A method for normalizing histology slides for
quantitative analysis", ISBI 2009.
"""
from __future__ import annotations

import numpy as np


class MacenkoNormalizer:
    def __init__(self, alpha: float = 1.0, beta: float = 0.15, luminosity_threshold: float = 0.8):
        self.alpha = alpha
        self.beta = beta
        self.luminosity_threshold = luminosity_threshold
        self.stain_matrix_target: np.ndarray | None = None
        self.max_conc_target: np.ndarray | None = None

    def fit(self, target_image: np.ndarray) -> "MacenkoNormalizer":
        self.stain_matrix_target, self.max_conc_target = self._estimate_stain_matrix(
            target_image
        )
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.stain_matrix_target is None:
            # use a canonical reference H&E stain matrix if no target was fit
            self.stain_matrix_target = np.array(
                [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]]
            )
            self.max_conc_target = np.array([1.9705, 1.0308])

        stain_matrix_src, _ = self._estimate_stain_matrix(image)
        concentrations = self._get_concentrations(image, stain_matrix_src)

        max_conc_src = np.percentile(concentrations, 99, axis=0)
        max_conc_src[max_conc_src == 0] = 1e-6
        concentrations *= (self.max_conc_target / max_conc_src)

        od = concentrations @ self.stain_matrix_target.T
        normalized = 255.0 * np.exp(-od)
        normalized = np.clip(normalized, 0, 255).reshape(image.shape).astype(np.uint8)
        return normalized

    def _estimate_stain_matrix(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        img = image.reshape(-1, 3).astype(np.float64)
        od = -np.log((img + 1) / 256.0)

        mask = (od > self.beta).any(axis=1)
        od_thresh = od[mask]
        if od_thresh.shape[0] < 10:
            od_thresh = od

        # eigendecomposition of the covariance to find the stain plane
        cov = np.cov(od_thresh.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        top2 = eigvecs[:, -2:]

        proj = od_thresh @ top2
        angles = np.arctan2(proj[:, 1], proj[:, 0])
        min_angle = np.percentile(angles, self.alpha)
        max_angle = np.percentile(angles, 100 - self.alpha)

        v1 = top2 @ np.array([np.cos(min_angle), np.sin(min_angle)])
        v2 = top2 @ np.array([np.cos(max_angle), np.sin(max_angle)])

        if v1[0] > v2[0]:
            stain_matrix = np.stack([v1, v2], axis=1)
        else:
            stain_matrix = np.stack([v2, v1], axis=1)
        stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=0, keepdims=True)

        concentrations = self._get_concentrations(image, stain_matrix)
        max_conc = np.percentile(concentrations, 99, axis=0)
        return stain_matrix, max_conc

    @staticmethod
    def _get_concentrations(image: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        img = image.reshape(-1, 3).astype(np.float64)
        od = -np.log((img + 1) / 256.0)
        concentrations, _, _, _ = np.linalg.lstsq(stain_matrix, od.T, rcond=None)
        return concentrations.T
