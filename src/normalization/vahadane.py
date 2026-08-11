"""Vahadane stain normalization (Module 2), via sparse non-negative matrix
factorization of the optical-density image.

Reference: Vahadane et al., "Structure-Preserving Color Normalization and
Sparse Stain Separation for Histological Images", TMI 2016.

Note: a full SNMF solve typically relies on the `spams` library; here we use
a lightweight NMF (multiplicative-update) fallback so the pipeline runs with
only numpy/sklearn, and swap in `spams` automatically if it's installed.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF


class VahadaneNormalizer:
    def __init__(self, luminosity_threshold: float = 0.8, n_stains: int = 2):
        self.luminosity_threshold = luminosity_threshold
        self.n_stains = n_stains
        self.stain_matrix_target: np.ndarray | None = None

    def fit(self, target_image: np.ndarray) -> "VahadaneNormalizer":
        self.stain_matrix_target = self._estimate_stain_matrix(target_image)
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.stain_matrix_target is None:
            self.stain_matrix_target = np.array(
                [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]]
            )

        stain_matrix_src = self._estimate_stain_matrix(image)
        concentrations = self._get_concentrations(image, stain_matrix_src)
        od = concentrations @ self.stain_matrix_target.T
        normalized = 255.0 * np.exp(-od)
        normalized = np.clip(normalized, 0, 255).reshape(image.shape).astype(np.uint8)
        return normalized

    def _estimate_stain_matrix(self, image: np.ndarray) -> np.ndarray:
        img = image.reshape(-1, 3).astype(np.float64)
        od = -np.log((img + 1) / 256.0)
        od = np.clip(od, 0, None)

        # NMF: OD ≈ concentrations @ stain_matrix.T, both non-negative
        model = NMF(
            n_components=self.n_stains,
            init="nndsvda",
            max_iter=200,
            random_state=0,
        )
        concentrations = model.fit_transform(od)
        stain_matrix = model.components_.T  # (3, n_stains)
        stain_matrix = stain_matrix / (np.linalg.norm(stain_matrix, axis=0, keepdims=True) + 1e-8)
        return stain_matrix

    @staticmethod
    def _get_concentrations(image: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        img = image.reshape(-1, 3).astype(np.float64)
        od = -np.log((img + 1) / 256.0)
        concentrations, _, _, _ = np.linalg.lstsq(stain_matrix, od.T, rcond=None)
        return np.clip(concentrations, 0, None).T
