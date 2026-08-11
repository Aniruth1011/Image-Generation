"""Privacy-evaluation metrics (Module 11) for the pipeline's OWN synthetic
output: nearest-neighbor distance to real training data, duplicate
detection, and embedding similarity, used to flag possible memorization of
real patient images before a synthetic dataset is released.
"""
from __future__ import annotations

import imagehash
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors


def nearest_neighbor_distance(
    synthetic_embeddings: np.ndarray,
    real_embeddings: np.ndarray,
    k: int = 5,
    metric: str = "cosine",
) -> np.ndarray:
    """For each synthetic sample, distance to its k-nearest REAL neighbor.
    A very small distance is the signal used to flag possible memorization."""
    nn_model = NearestNeighbors(n_neighbors=k, metric=metric).fit(real_embeddings)
    distances, _ = nn_model.kneighbors(synthetic_embeddings)
    return distances[:, 0]  # distance to closest real neighbor


def duplicate_detection(
    synthetic_paths: list[str], real_paths: list[str], hash_method: str = "phash", threshold: int = 3
) -> list[dict]:
    """Perceptual-hash comparison between synthetic and real images to catch
    near-exact duplicates (a strong memorization signal)."""
    hash_fn = getattr(imagehash, hash_method)
    real_hashes = {p: hash_fn(Image.open(p)) for p in real_paths}

    flagged = []
    for s_path in synthetic_paths:
        s_hash = hash_fn(Image.open(s_path))
        for r_path, r_hash in real_hashes.items():
            if s_hash - r_hash <= threshold:
                flagged.append(
                    {"synthetic": s_path, "closest_real": r_path, "hash_distance": int(s_hash - r_hash)}
                )
                break
    return flagged


def embedding_similarity_flags(
    synthetic_embeddings: np.ndarray,
    real_embeddings: np.ndarray,
    synthetic_ids: list[str],
    real_ids: list[str],
    threshold: float = 0.95,
) -> list[dict]:
    """Cosine-similarity based memorization flag: any synthetic sample whose
    nearest real neighbor exceeds `threshold` similarity is reported."""
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(synthetic_embeddings, real_embeddings)
    flags = []
    for i, row in enumerate(sims):
        best_j = int(row.argmax())
        if row[best_j] >= threshold:
            flags.append(
                {
                    "synthetic_id": synthetic_ids[i],
                    "closest_real_id": real_ids[best_j],
                    "cosine_similarity": float(row[best_j]),
                }
            )
    return flags
