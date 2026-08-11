"""Image-quality metrics (Module 11): SSIM, PSNR, LPIPS, FID, KID,
Precision/Recall, Density/Coverage.
"""
from __future__ import annotations

import lpips
import numpy as np
import torch
from cleanfid import fid as cleanfid
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
from sklearn.neighbors import NearestNeighbors


def compute_ssim(real: np.ndarray, fake: np.ndarray) -> float:
    return float(skimage_ssim(real, fake, channel_axis=-1, data_range=255))


def compute_psnr(real: np.ndarray, fake: np.ndarray) -> float:
    return float(skimage_psnr(real, fake, data_range=255))


class LPIPSScorer:
    def __init__(self, net: str = "alex", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, real: torch.Tensor, fake: torch.Tensor) -> float:
        """Expects tensors in [-1, 1], shape (B, 3, H, W)."""
        return float(self.model(real.to(self.device), fake.to(self.device)).mean())


def compute_fid(real_dir: str, fake_dir: str) -> float:
    """Wraps clean-fid; both dirs should contain only images of one class."""
    return float(cleanfid.compute_fid(real_dir, fake_dir))


def compute_kid(real_dir: str, fake_dir: str) -> float:
    return float(cleanfid.compute_kid(real_dir, fake_dir))


def precision_recall(
    real_embeddings: np.ndarray, fake_embeddings: np.ndarray, k: int = 5
) -> dict:
    """Improved precision & recall for generative models (Kynkäänniemi et al., 2019)."""
    nn_real = NearestNeighbors(n_neighbors=k).fit(real_embeddings)
    real_radii = nn_real.kneighbors(real_embeddings)[0][:, -1]

    nn_fake = NearestNeighbors(n_neighbors=k).fit(fake_embeddings)
    fake_radii = nn_fake.kneighbors(fake_embeddings)[0][:, -1]

    d_fake_to_real = nn_real.kneighbors(fake_embeddings)[0][:, 0]
    d_real_to_fake = nn_fake.kneighbors(real_embeddings)[0][:, 0]

    precision = float(np.mean(d_fake_to_real <= real_radii[nn_real.kneighbors(fake_embeddings)[1][:, 0]]))
    recall = float(np.mean(d_real_to_fake <= fake_radii[nn_fake.kneighbors(real_embeddings)[1][:, 0]]))
    return {"precision": precision, "recall": recall}


def density_coverage(
    real_embeddings: np.ndarray, fake_embeddings: np.ndarray, k: int = 5
) -> dict:
    """Density & Coverage (Naeem et al., 2020) — more robust than P&R to outliers."""
    nn_real = NearestNeighbors(n_neighbors=k).fit(real_embeddings)
    real_radii = nn_real.kneighbors(real_embeddings)[0][:, -1]

    dists_fake_to_real, idx = nn_real.kneighbors(fake_embeddings)
    density = float(
        np.mean(
            [
                np.sum(dists_fake_to_real[i] <= real_radii[idx[i]]) / k
                for i in range(len(fake_embeddings))
            ]
        )
    )
    coverage = float(np.mean(dists_fake_to_real[:, 0] <= real_radii[idx[:, 0]]))
    return {"density": density, "coverage": coverage}
