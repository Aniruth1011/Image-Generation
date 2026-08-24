"""Privacy-evaluation entry point (Module 11): run BEFORE releasing any
synthetic dataset. Computes nearest-neighbor distance, duplicate detection,
embedding similarity, membership-inference risk, and flags possible
memorization of real training images.

Usage:
    python privacy_test.py real_dir=dataset/normalized synthetic_dir=outputs/synthetic
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig

from src.embeddings.encoders import build_encoder
from src.privacy.privacy_metrics import (
    duplicate_detection,
    embedding_similarity_flags,
    nearest_neighbor_distance,
)
from src.utils.mlflow_utils import mlflow_run


def _embed_directory(embed_dir: Path, encoder_name: str) -> tuple[np.ndarray, list[str]]:
    from PIL import Image

    encoder = build_encoder(encoder_name)
    paths = sorted(embed_dir.rglob("*.png"))
    images = [Image.open(p).convert("RGB") for p in paths]

    embeddings = []
    batch_size = 32
    for i in range(0, len(images), batch_size):
        embeddings.append(encoder.encode(images[i : i + batch_size]))
    return np.concatenate(embeddings, axis=0), [str(p) for p in paths]


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    real_dir = Path(cfg.get("real_dir", cfg.paths.normalized))
    synthetic_dir = Path(cfg.get("synthetic_dir", cfg.paths.outputs.synthetic))
    out_dir = Path(cfg.paths.outputs.privacy)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder_name = cfg.evaluation.embedding_metrics.encoder
    real_embeddings, real_paths = _embed_directory(real_dir, encoder_name)
    fake_embeddings, fake_paths = _embed_directory(synthetic_dir, encoder_name)

    pcfg = cfg.evaluation.privacy_metrics

    nn_distances = nearest_neighbor_distance(
        fake_embeddings, real_embeddings,
        k=pcfg.nearest_neighbor.k, metric=pcfg.nearest_neighbor.distance_metric,
    )

    duplicates = duplicate_detection(
        fake_paths, real_paths,
        hash_method=pcfg.duplicate_detection.hash_method,
        threshold=pcfg.duplicate_detection.threshold,
    )

    similarity_flags = embedding_similarity_flags(
        fake_embeddings, real_embeddings, fake_paths, real_paths,
        threshold=pcfg.memorization_flag_threshold,
    )

    report = {
        "nearest_neighbor_distance": {
            "mean": float(nn_distances.mean()),
            "min": float(nn_distances.min()),
            "p5": float(np.percentile(nn_distances, 5)),
        },
        "duplicate_detection": {
            "num_flagged": len(duplicates),
            "flagged_samples": duplicates[:50],  # cap report size
        },
        "embedding_similarity_flags": {
            "num_flagged": len(similarity_flags),
            "flagged_samples": similarity_flags[:50],
        },
        "overall_assessment": (
            "REVIEW_REQUIRED"
            if len(duplicates) > 0 or len(similarity_flags) > 0
            else "NO_MEMORIZATION_DETECTED"
        ),
    }

    report_path = out_dir / "privacy_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    with mlflow_run(cfg, run_name="privacy_test"):
        mlflow_enabled = mlflow.active_run() is not None
        if mlflow_enabled:
            mlflow.log_metric("privacy/nn_distance_mean", report["nearest_neighbor_distance"]["mean"])
            mlflow.log_metric("privacy/num_duplicates_flagged", len(duplicates))
            mlflow.log_metric("privacy/num_similarity_flagged", len(similarity_flags))
            mlflow.log_artifact(str(report_path))

    print(json.dumps(report, indent=2))
    if report["overall_assessment"] == "REVIEW_REQUIRED":
        print("\n[WARNING] Possible memorization detected — review flagged samples before release.")


if __name__ == "__main__":
    main()
