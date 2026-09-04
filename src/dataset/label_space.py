"""Utilities for persisting and reloading the active class label space."""
from __future__ import annotations

import json
from pathlib import Path


def save_label_space(
    metadata_dir: str | Path,
    classes: list[str],
    dataset_kind: str,
    label_mode: str,
) -> Path:
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    out_path = metadata_dir / "label_space.json"
    payload = {
        "classes": classes,
        "num_classes": len(classes),
        "dataset_kind": dataset_kind,
        "label_mode": label_mode,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_label_space(metadata_dir: str | Path) -> dict | None:
    metadata_dir = Path(metadata_dir)
    in_path = metadata_dir / "label_space.json"
    if not in_path.exists():
        return None
    return json.loads(in_path.read_text(encoding="utf-8"))


def resolve_class_names(normalized_dir: str | Path, metadata_dir: str | Path) -> list[str]:
    label_space = load_label_space(metadata_dir)
    if label_space and label_space.get("classes"):
        return list(label_space["classes"])

    normalized_dir = Path(normalized_dir)
    if not normalized_dir.exists():
        raise FileNotFoundError(
            "Could not resolve class names: "
            f"metadata label space missing at {Path(metadata_dir) / 'label_space.json'} "
            f"and normalized directory not found at {normalized_dir}"
        )
    return sorted(
        path.name
        for path in normalized_dir.iterdir()
        if path.is_dir()
    )
