"""Batch-extract embeddings for all patches (Module 4)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.embeddings.encoders import build_encoder


def extract_embeddings_for_patches(
    patch_paths: list[str | Path],
    encoder_name: str,
    out_dir: str | Path,
    batch_size: int = 32,
) -> None:
    encoder = build_encoder(encoder_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = []

    for i in tqdm(range(0, len(patch_paths), batch_size), desc=f"Embedding ({encoder_name})"):
        batch_paths = patch_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        embeddings = encoder.encode(images)

        for path, emb in zip(batch_paths, embeddings):
            stem = Path(path).stem
            embedding_path = out_dir / f"{stem}_embedding.npy"
            np.save(embedding_path, emb)
            manifest_records.append(
                {
                "patch_file": str(path),
                "embedding_file": str(embedding_path),
                "embedding_model": encoder_name,
                "embedding_dim": int(emb.shape[-1]),
                }
            )

    with open(out_dir / "embeddings_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)
