"""Dataset adapter for the ESD104 Kaggle slide layout."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ESD_COLOR_TO_FINE_LABEL = {
    "#8B0000": "tub1",
    "#FF00FF": "tub2",
    "#800080": "pap",
    "#4B0082": "others",
    "#8A2BE2": "normal_gland",
    "#0000FF": "chronic_gastritis",
    "#4682B4": "chronic_atrophic_gastritis",
    "#00FF00": "lymphoid_follicles",
    "#008000": "complete_intestinal_metaplasia",
    "#FFFF00": "incomplete_intestinal_metaplasia",
}

ESD_FINE_LABEL_ORDER = [
    "normal_gland",
    "chronic_gastritis",
    "chronic_atrophic_gastritis",
    "complete_intestinal_metaplasia",
    "incomplete_intestinal_metaplasia",
    "lymphoid_follicles",
    "tub1",
    "tub2",
    "pap",
    "others",
]

ESD_TO_COHORT4_LABEL = {
    "normal_gland": "Normal",
    "chronic_gastritis": "HP",
    "chronic_atrophic_gastritis": "HP",
    "complete_intestinal_metaplasia": "IM",
    "incomplete_intestinal_metaplasia": "IM",
    "lymphoid_follicles": "Mixed",
    "tub1": "Mixed",
    "tub2": "Mixed",
    "pap": "Mixed",
    "others": "Mixed",
}

ESD_COHORT4_ORDER = ["Normal", "HP", "IM", "Mixed"]


def is_esd_dataset_root(raw_dir: str | Path) -> bool:
    raw_dir = Path(raw_dir)
    xml_dir = raw_dir / "ESD_40X_annotation_downsample64_xml" / "ESD_40X_annotation_downsample64_xml"
    return raw_dir.exists() and xml_dir.exists() and any(raw_dir.glob("*.svs"))


def esd_slide_paths(raw_dir: str | Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    return sorted(raw_dir.glob("*.svs"))


def esd_class_names(label_mode: str) -> list[str]:
    if label_mode == "cohort4":
        return list(ESD_COHORT4_ORDER)
    return list(ESD_FINE_LABEL_ORDER)


@dataclass
class PatchLabel:
    class_name: str
    label_source: str
    label_fraction: float
    label_counts: dict[str, int]

    def to_record_fields(self) -> dict:
        return {
            "class": self.class_name,
            "label_source": self.label_source,
            "label_fraction": self.label_fraction,
            "label_counts": self.label_counts,
        }


class ESDPatchLabeler:
    def __init__(
        self,
        raw_dir: str | Path,
        slide_path: str | Path,
        downsample_factor: int = 64,
        label_mode: str = "fine",
        min_label_fraction: float = 0.05,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.slide_path = Path(slide_path)
        self.downsample_factor = downsample_factor
        self.label_mode = label_mode
        self.min_label_fraction = min_label_fraction
        self.xml_path = (
            self.raw_dir
            / "ESD_40X_annotation_downsample64_xml"
            / "ESD_40X_annotation_downsample64_xml"
            / f"{self.slide_path.stem}.xml"
        )
        self.mask = self._build_mask()

    def _project_label(self, fine_label: str) -> str:
        if self.label_mode == "cohort4":
            return ESD_TO_COHORT4_LABEL[fine_label]
        return fine_label

    def _build_mask(self) -> np.ndarray:
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Missing ESD XML annotation for slide: {self.slide_path}")

        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        polygons: list[tuple[str, list[tuple[float, float]]]] = []
        max_x = 0.0
        max_y = 0.0

        for ann in root.iter("Annotation"):
            color = ann.attrib.get("Color", "").upper()
            fine_label = ESD_COLOR_TO_FINE_LABEL.get(color)
            if fine_label is None:
                continue

            coords_node = ann.find("Coordinates")
            if coords_node is None:
                continue

            coords: list[tuple[float, float]] = []
            for coord in coords_node.findall("Coordinate"):
                x = float(coord.attrib["X"])
                y = float(coord.attrib["Y"])
                coords.append((x, y))
                max_x = max(max_x, x)
                max_y = max(max_y, y)

            if coords:
                polygons.append((self._project_label(fine_label), coords))

        width = max(1, math.ceil(max_x) + 2)
        height = max(1, math.ceil(max_y) + 2)
        label_names = esd_class_names(self.label_mode)
        label_to_idx = {label: idx for idx, label in enumerate(label_names)}

        mask = Image.new("I", (width, height), 0)
        drawer = ImageDraw.Draw(mask)
        for label_name, coords in polygons:
            drawer.polygon(coords, fill=label_to_idx[label_name] + 1)

        return np.array(mask, dtype=np.int32) - 1

    def label_patch(self, x: int, y: int, patch_size: int) -> PatchLabel | None:
        x0 = max(0, x // self.downsample_factor)
        y0 = max(0, y // self.downsample_factor)
        x1 = max(x0 + 1, math.ceil((x + patch_size) / self.downsample_factor))
        y1 = max(y0 + 1, math.ceil((y + patch_size) / self.downsample_factor))

        region = self.mask[y0:y1, x0:x1]
        if region.size == 0:
            return None

        foreground = region[region >= 0]
        if foreground.size == 0:
            return None

        values, counts = np.unique(foreground, return_counts=True)
        order = np.argsort(counts)[::-1]
        values = values[order]
        counts = counts[order]
        total = int(counts.sum())
        label_names = esd_class_names(self.label_mode)
        class_name = label_names[int(values[0])]
        label_fraction = float(counts[0] / max(total, 1))
        if label_fraction < self.min_label_fraction:
            return None

        label_counts = {
            label_names[int(idx)]: int(count)
            for idx, count in zip(values.tolist(), counts.tolist())
        }
        return PatchLabel(
            class_name=class_name,
            label_source="esd_xml_downsample64",
            label_fraction=label_fraction,
            label_counts=label_counts,
        )


def summarize_esd_slide_labels(
    raw_dir: str | Path,
    slide_paths: list[Path],
    label_mode: str,
) -> dict[str, int]:
    class_names = esd_class_names(label_mode)
    counts = Counter({name: 0 for name in class_names})

    for slide_path in slide_paths:
        labeler = ESDPatchLabeler(raw_dir, slide_path, label_mode=label_mode)
        present = np.unique(labeler.mask[labeler.mask >= 0])
        for idx in present.tolist():
            counts[class_names[int(idx)]] += 1

    return dict(counts)
