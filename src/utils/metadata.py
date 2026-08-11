"""Per-image metadata generation and export (Module 10)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class ImageMetadata:
    filename: str
    class_label: str
    generator_version: str
    seed: int
    generation_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    augmentation: str = "none"
    embedding_model: str | None = None
    latent_model: str = "vqvae"
    image_size: int = 256
    guidance_scale: float | None = None
    sampler: str | None = None
    num_inference_steps: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class MetadataStore:
    """Accumulates ImageMetadata records and exports metadata.csv / .json."""

    def __init__(self) -> None:
        self._records: list[ImageMetadata] = []

    def add(self, record: ImageMetadata) -> None:
        self._records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self._records])

    def export_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)

    def export_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in self._records], f, indent=2)
