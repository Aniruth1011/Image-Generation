"""Tiling / patch extraction (Module 1)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.dataset.wsi_loader import SlideReader
from src.preprocessing.tissue_detection import detect_tissue_mask, tissue_percentage


def extract_patches(
    slide_path: str | Path,
    out_dir: str | Path,
    patch_size: int = 256,
    stride: int = 224,
    minimum_tissue_percentage: float = 0.4,
    tissue_threshold_method: str = "otsu",
    class_label: str | None = None,
) -> list[dict]:
    """Tile a slide/image into patches, discarding background-only tiles.

    Returns a list of metadata dicts (one per kept patch), which the caller
    can accumulate into dataset/metadata/patches.csv.
    """
    slide_path = Path(slide_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = SlideReader(slide_path)
    width, height = reader.dimensions

    records = []
    idx = 0
    for y in range(0, max(height - patch_size, 1), stride):
        for x in range(0, max(width - patch_size, 1), stride):
            patch = reader.read_region((x, y), level=0, size=(patch_size, patch_size))
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue

            mask = detect_tissue_mask(patch, method=tissue_threshold_method)
            pct = tissue_percentage(mask)
            if pct < minimum_tissue_percentage:
                continue

            patch_name = f"{slide_path.stem}_x{x}_y{y}.png"
            patch_path = out_dir / patch_name
            Image.fromarray(patch).save(patch_path)

            record = {
                "patch_file": str(patch_path),
                "source_slide": str(slide_path),
                "x": x,
                "y": y,
                "patch_size": patch_size,
                "tissue_percentage": pct,
                "class": class_label,
            }
            records.append(record)
            idx += 1

    reader.close()
    return records


def save_patch_metadata(records: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
