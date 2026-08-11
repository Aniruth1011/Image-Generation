"""Dataset / embedding-space analysis (Module 5): class balance, PCA/t-SNE/
UMAP projections, cluster statistics, nearest neighbors, inter/intra-class
distances.
"""
from __future__ import annotations

from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

try:
    import umap

    _HAS_UMAP = True
except ImportError:  # pragma: no cover
    _HAS_UMAP = False


def class_balance(labels: list[str]) -> pd.Series:
    return pd.Series(labels).value_counts()


def plot_class_balance(labels: list[str], out_path: str | Path) -> None:
    counts = class_balance(labels)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values)
    plt.ylabel("count")
    plt.title("Class balance")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def project_embeddings(
    embeddings: np.ndarray, method: str = "pca", n_components: int = 2, seed: int = 42
) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=n_components, random_state=seed).fit_transform(embeddings)
    if method == "tsne":
        return TSNE(n_components=n_components, random_state=seed, init="pca").fit_transform(
            embeddings
        )
    if method == "umap":
        if not _HAS_UMAP:
            raise ImportError("umap-learn is not installed")
        return umap.UMAP(n_components=n_components, random_state=seed).fit_transform(embeddings)
    raise ValueError(f"Unknown projection method: {method}")


def plot_embedding_projection(
    embeddings: np.ndarray, labels: list[str], method: str, out_path: str | Path
) -> None:
    proj = project_embeddings(embeddings, method=method)
    df = pd.DataFrame({"x": proj[:, 0], "y": proj[:, 1], "class": labels})
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="class", s=12, alpha=0.7)
    plt.title(f"{method.upper()} projection of embeddings")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def cluster_statistics(embeddings: np.ndarray, labels: list[str]) -> dict:
    label_ids = pd.factorize(labels)[0]
    score = silhouette_score(embeddings, label_ids) if len(set(labels)) > 1 else float("nan")
    return {"silhouette_score": score, "num_classes": len(set(labels))}


def nearest_neighbors(embeddings: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Returns (distances, indices) of the k nearest neighbors for every row,
    using FAISS for efficient search."""
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    distances, indices = index.search(embeddings, k + 1)  # +1 to skip self
    return distances[:, 1:], indices[:, 1:]


def inter_intra_class_distances(embeddings: np.ndarray, labels: list[str]) -> dict:
    labels_arr = np.array(labels)
    classes = sorted(set(labels))
    intra, inter = {}, {}

    for c in classes:
        mask = labels_arr == c
        if mask.sum() > 1:
            class_embs = embeddings[mask]
            centroid = class_embs.mean(axis=0)
            intra[c] = float(np.mean(np.linalg.norm(class_embs - centroid, axis=1)))

    centroids = {c: embeddings[labels_arr == c].mean(axis=0) for c in classes}
    for i, c1 in enumerate(classes):
        for c2 in classes[i + 1 :]:
            inter[f"{c1}_vs_{c2}"] = float(np.linalg.norm(centroids[c1] - centroids[c2]))

    return {"intra_class": intra, "inter_class": inter}
