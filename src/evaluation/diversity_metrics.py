"""Diversity metrics (Module 11): pairwise LPIPS/cosine distance, mode
collapse score, unique-image ratio, class distribution.
"""
from __future__ import annotations

import imagehash
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_distances


def pairwise_cosine_distance(embeddings: np.ndarray, num_pairs: int = 5000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    idx_a = rng.integers(0, n, num_pairs)
    idx_b = rng.integers(0, n, num_pairs)
    dists = 1 - np.sum(
        embeddings[idx_a] * embeddings[idx_b], axis=1
    ) / (
        np.linalg.norm(embeddings[idx_a], axis=1) * np.linalg.norm(embeddings[idx_b], axis=1) + 1e-8
    )
    return float(dists.mean())


def mode_collapse_score(embeddings: np.ndarray) -> float:
    """Ratio of average pairwise distance to the embedding-space diameter;
    values near 0 indicate collapse onto a small number of modes."""
    dists = cosine_distances(embeddings)
    avg_dist = dists[np.triu_indices_from(dists, k=1)].mean()
    return float(avg_dist)


def unique_image_ratio(image_paths: list[str], hash_threshold: int = 3) -> float:
    hashes = [imagehash.phash(Image.open(p)) for p in image_paths]
    unique = []
    for h in hashes:
        if not any(h - u <= hash_threshold for u in unique):
            unique.append(h)
    return len(unique) / len(hashes)


def class_distribution(labels: list[str]) -> dict:
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    return {str(u): float(c / total) for u, c in zip(unique, counts)}
