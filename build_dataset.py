"""End-to-end data pipeline entry point (Modules 1-4): tissue detection ->
patch extraction -> stain normalization -> artifact scoring -> embedding
extraction. Run this before train_vqvae.py.

Usage:
    python build_dataset.py
    python build_dataset.py normalization.method=vahadane
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.dataset.esd_dataset import (
    ESDPatchLabeler,
    esd_class_names,
    esd_slide_paths,
    is_esd_dataset_root,
    summarize_esd_slide_labels,
)
from src.dataset.label_space import save_label_space
from src.dataset.patch_extractor import extract_patches, save_patch_metadata
from src.dataset.wsi_loader import discover_input_files
from src.embeddings.extract_embeddings import extract_embeddings_for_patches
from src.normalization.normalize import build_normalizer, normalize_patch_file
from src.preprocessing.artifact_removal import assess_patch_quality


def _resolve_standard_input_groups(cfg: DictConfig, raw_dir: Path) -> list[tuple[str, list[Path]]]:
    ingestion = cfg.dataset.get("ingestion", {})
    excluded_dirs = list(ingestion.get("excluded_dir_names", []))

    grouped_inputs: list[tuple[str, list[Path]]] = []
    for class_name in cfg.dataset.classes:
        class_raw_dir = raw_dir / class_name
        if not class_raw_dir.exists():
            continue
        slide_paths = discover_input_files(
            class_raw_dir,
            cfg.dataset.wsi.input_format,
            recursive=True,
            excluded_dir_names=excluded_dirs,
        )
        grouped_inputs.append((class_name, slide_paths))
    return grouped_inputs


def _resolve_dataset_kind(cfg: DictConfig, raw_dir: Path) -> str:
    ingestion = cfg.dataset.get("ingestion", {})
    layout = ingestion.get("layout", "auto")
    if layout == "auto":
        if is_esd_dataset_root(raw_dir):
            return "esd"
        return "standard"
    if layout == "esd":
        return "esd"
    return "standard"


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    raw_dir = Path(cfg.paths.data.raw)
    patches_dir = Path(cfg.paths.data.patches)
    normalized_dir = Path(cfg.paths.normalized)
    metadata_dir = Path(cfg.paths.data.metadata)
    for d in (patches_dir, normalized_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Module 1: patch extraction ---
    all_records = []
    dataset_kind = _resolve_dataset_kind(cfg, raw_dir)
    if dataset_kind == "esd":
        label_mode = cfg.dataset.ingestion.get("esd_label_mode", "fine")
        classes = esd_class_names(label_mode)
        save_label_space(metadata_dir, classes, dataset_kind="esd", label_mode=label_mode)

        slide_paths = esd_slide_paths(raw_dir)
        for slide_path in slide_paths:
            labeler = ESDPatchLabeler(
                raw_dir,
                slide_path,
                downsample_factor=cfg.dataset.ingestion.get("esd_annotation_downsample_factor", 64),
                label_mode=label_mode,
                min_label_fraction=cfg.dataset.ingestion.get("esd_min_label_fraction", 0.05),
            )
            records = extract_patches(
                slide_path,
                patches_dir,
                patch_size=cfg.dataset.patching.patch_size,
                stride=cfg.dataset.patching.stride,
                minimum_tissue_percentage=cfg.dataset.patching.minimum_tissue_percentage,
                tissue_threshold_method=cfg.dataset.patching.tissue_threshold_method,
                patch_labeler=labeler.label_patch,
            )
            all_records.extend(records)

        slide_label_summary = summarize_esd_slide_labels(raw_dir, slide_paths, label_mode)
        summary_path = metadata_dir / "dataset_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "dataset_kind": "esd",
                    "num_slides": len(slide_paths),
                    "slide_label_presence": slide_label_summary,
                    "label_mode": label_mode,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        classes = list(cfg.dataset.classes)
        save_label_space(metadata_dir, classes, dataset_kind="standard", label_mode="folder")
        input_groups = _resolve_standard_input_groups(cfg, raw_dir)
        for class_name, slide_paths in input_groups:
            class_patch_dir = patches_dir / class_name
            for slide_path in slide_paths:
                records = extract_patches(
                    slide_path,
                    class_patch_dir,
                    patch_size=cfg.dataset.patching.patch_size,
                    stride=cfg.dataset.patching.stride,
                    minimum_tissue_percentage=cfg.dataset.patching.minimum_tissue_percentage,
                    tissue_threshold_method=cfg.dataset.patching.tissue_threshold_method,
                    class_label=class_name,
                )
                all_records.extend(records)
    save_patch_metadata(all_records, metadata_dir / "patches.json")
    print(f"Extracted {len(all_records)} patches")

    # --- Module 3: artifact scoring (before normalization, on raw patches) ---
    import numpy as np
    from PIL import Image

    quality_records = []
    for record in tqdm(all_records, desc="Scoring patch quality"):
        image = np.array(Image.open(record["patch_file"]).convert("RGB"))
        report = assess_patch_quality(image)
        quality_records.append({**record, **report.__dict__})
    save_patch_metadata(quality_records, metadata_dir / "patches_quality.json")

    kept_records = [r for r in quality_records if not r["discard_flag"]]
    print(f"Kept {len(kept_records)}/{len(quality_records)} patches after artifact filtering")

    # --- Module 2: stain normalization ---
    normalizer = build_normalizer(
        method=cfg.normalization.method,
        alpha=cfg.normalization.alpha,
        beta=cfg.normalization.beta,
        luminosity_threshold=cfg.normalization.luminosity_threshold,
    )
    if cfg.normalization.target_image:
        import numpy as np
        target = np.array(Image.open(cfg.normalization.target_image).convert("RGB"))
        normalizer.fit(target)

    for record in tqdm(kept_records, desc=f"Normalizing ({cfg.normalization.method})"):
        class_name = record["class"]
        out_path = normalized_dir / class_name / Path(record["patch_file"]).name
        normalize_patch_file(record["patch_file"], out_path, normalizer)
        record["normalized_file"] = str(out_path)
    save_patch_metadata(kept_records, metadata_dir / "patches_final.json")

    # --- Module 4: embedding extraction ---
    embed_dir = Path(cfg.paths.embeddings)
    encoder_name = cfg.evaluation.embedding_metrics.encoder
    normalized_paths = [r["normalized_file"] for r in kept_records]
    if normalized_paths:
        extract_embeddings_for_patches(normalized_paths, encoder_name, embed_dir)

    print("Dataset build complete.")
    print(f"  Patches:    {patches_dir}")
    print(f"  Normalized: {normalized_dir}")
    print(f"  Embeddings: {embed_dir}")
    print(f"  Metadata:   {metadata_dir}")


if __name__ == "__main__":
    main()
