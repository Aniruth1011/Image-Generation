"""Full evaluation suite (Module 11): image quality, embedding metrics,
diversity metrics, and (optionally) the internal downstream-utility check.
Privacy metrics are run separately via privacy_test.py.

Usage:
    python evaluate.py real_dir=dataset/normalized synthetic_dir=outputs/synthetic
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig

from src.embeddings.encoders import build_encoder
from src.evaluation.diversity_metrics import (
    class_distribution,
    mode_collapse_score,
    pairwise_cosine_distance,
    unique_image_ratio,
)
from src.evaluation.embedding_metrics import (
    cluster_overlap,
    frechet_distance,
    maximum_mean_discrepancy,
    mean_cosine_similarity,
    mean_feature_distance,
)
from src.evaluation.image_quality import compute_fid, compute_kid, density_coverage, precision_recall
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
    out_dir = Path(cfg.evaluation.report.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder_name = cfg.evaluation.embedding_metrics.encoder
    real_embeddings, real_paths = _embed_directory(real_dir, encoder_name)
    fake_embeddings, fake_paths = _embed_directory(synthetic_dir, encoder_name)

    report: dict = {}

    # --- image quality (embedding-space proxies for FID/KID/P&R/D&C) ---
    report["image_quality"] = {
        "fid": compute_fid(str(real_dir), str(synthetic_dir)),
        "kid": compute_kid(str(real_dir), str(synthetic_dir)),
        **precision_recall(real_embeddings, fake_embeddings, k=cfg.evaluation.image_quality.precision_recall.k),
        **density_coverage(real_embeddings, fake_embeddings),
    }

    # --- embedding metrics ---
    report["embedding_metrics"] = {
        "cosine_similarity": mean_cosine_similarity(real_embeddings, fake_embeddings),
        "feature_distance": mean_feature_distance(real_embeddings, fake_embeddings),
        "cluster_overlap": cluster_overlap(real_embeddings, fake_embeddings),
        "mmd": maximum_mean_discrepancy(real_embeddings, fake_embeddings),
        "frechet_distance": frechet_distance(real_embeddings, fake_embeddings),
    }

    # --- diversity metrics ---
    fake_labels = [Path(p).parent.name for p in fake_paths]
    report["diversity_metrics"] = {
        "pairwise_cosine_distance": pairwise_cosine_distance(
            fake_embeddings, num_pairs=cfg.evaluation.diversity_metrics.num_pairs
        ),
        "mode_collapse_score": mode_collapse_score(fake_embeddings),
        "unique_image_ratio": unique_image_ratio(
            fake_paths, hash_threshold=cfg.evaluation.diversity_metrics.dedup_hash_threshold
        ),
        "class_distribution": class_distribution(fake_labels),
    }

    report_path = out_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    with mlflow_run(cfg, run_name="evaluation"):
        mlflow_enabled = mlflow.active_run() is not None
        for section, metrics in report.items():
            for k, v in metrics.items():
                if mlflow_enabled and isinstance(v, (int, float)):
                    mlflow.log_metric(f"{section}/{k}", v)
        if mlflow_enabled:
            mlflow.log_artifact(str(report_path))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
