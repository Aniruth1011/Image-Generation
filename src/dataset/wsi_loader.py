"""Whole-slide-image (or plain image) loading (Module 1).

Uses openslide when available (proper WSI formats: .svs/.ndpi/.tiff);
falls back to PIL for already-tiled image inputs (.png/.jpg).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

try:
    import openslide

    _HAS_OPENSLIDE = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_OPENSLIDE = False


class SlideReader:
    """Uniform interface over WSI files and plain images."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.is_wsi = _HAS_OPENSLIDE and self.path.suffix.lower() in {
            ".svs",
            ".ndpi",
            ".tiff",
            ".tif",
        }
        if self.is_wsi:
            self._slide = openslide.OpenSlide(str(self.path))
        else:
            self._image = Image.open(self.path).convert("RGB")

    @property
    def dimensions(self) -> tuple[int, int]:
        if self.is_wsi:
            return self._slide.dimensions
        return self._image.size

    def base_magnification(self) -> float | None:
        if not self.is_wsi:
            return None
        props = self._slide.properties
        mag = props.get("openslide.objective-power")
        return float(mag) if mag is not None else None

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> np.ndarray:
        if self.is_wsi:
            region = self._slide.read_region(location, level, size).convert("RGB")
            return np.array(region)
        # plain image: location/level are ignored, we crop directly
        x, y = location
        w, h = size
        crop = self._image.crop((x, y, x + w, y + h))
        return np.array(crop)

    def close(self) -> None:
        if self.is_wsi:
            self._slide.close()


def iter_input_files(raw_dir: str | Path, extensions: list[str]) -> Iterator[Path]:
    raw_dir = Path(raw_dir)
    for ext in extensions:
        yield from sorted(raw_dir.rglob(f"*{ext}"))


def discover_input_files(
    raw_dir: str | Path,
    extensions: list[str],
    recursive: bool = True,
    excluded_dir_names: list[str] | None = None,
) -> list[Path]:
    raw_dir = Path(raw_dir)
    excluded = {name.lower() for name in (excluded_dir_names or [])}
    globber = raw_dir.rglob if recursive else raw_dir.glob
    discovered: list[Path] = []
    seen: set[Path] = set()

    for ext in extensions:
        for path in sorted(globber(f"*{ext}")):
            if any(part.lower() in excluded for part in path.parts):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(path)

    return discovered
