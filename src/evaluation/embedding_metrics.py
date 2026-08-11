"""Embedding-space metrics using a pathology encoder (Module 11): cosine
similarity, feature distance, cluster overlap, MMD, Fréchet distance.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity


def mean_cosine_similarity(real_embeddings: np.ndarray, fake_embeddings: np.ndarray) -> float:
    sims = cosine_similarity(real_embeddings, fake_embeddings)
    return float(sims.mean())


def mean_feature_distance(real_embeddings: np.ndarray, fake_embeddings: np.ndarray) -> float:
    real_c = real_embeddings.mean(axis=0)
    fake_c = fake_embeddings.mean(axis=0)
    return float(np.linalg.norm(real_c - fake_c))


def cluster_overlap(
    real_embeddings: np.ndarray, fake_embeddings: np.ndarray, n_clusters: int = 4
) -> float:
    """Adjusted Rand Index between cluster assignments learned jointly and
    the true real/fake origin, as a proxy for distributional overlap."""
    combined = np.concatenate([real_embeddings, fake_embeddings], axis=0)
    origin = np.array([0] * len(real_embeddings) + [1] * len(fake_embeddings))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(combined)
    return float(adjusted_rand_score(origin, kmeans.labels_))


def maximum_mean_discrepancy(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """RBF-kernel MMD^2 (unbiased estimator)."""
    def rbf(a, b):
        dist = np.sum(a**2, 1)[:, None] + np.sum(b**2, 1)[None, :] - 2 * a @ b.T
        return np.exp(-gamma * dist)

    k_xx = rbf(x, x)
    k_yy = rbf(y, y)
    k_xy = rbf(x, y)
    n, m = len(x), len(y)
    mmd2 = (
        (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
        + (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
        - 2 * k_xy.mean()
    )
    return float(mmd2)


def frechet_distance(real_embeddings: np.ndarray, fake_embeddings: np.ndarray) -> float:
    """Fréchet distance between two Gaussians fit to the embedding sets
    (the same formula underlying FID, but applied to arbitrary embeddings)."""
    mu1, mu2 = real_embeddings.mean(0), fake_embeddings.mean(0)
    sigma1 = np.cov(real_embeddings, rowvar=False)
    sigma2 = np.cov(fake_embeddings, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
