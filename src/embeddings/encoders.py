"""Pathology foundation-model encoders (Module 4).

Supports Virchow2, UNI, CONCH (all via Hugging Face `timm`/`transformers`,
gated checkpoints requiring `huggingface-cli login`) and DINOv3 (torch hub /
timm). All encoders expose a uniform `.encode(images) -> np.ndarray` API so
the rest of the pipeline is agnostic to which backbone is selected.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image


class BaseEncoder(ABC):
    name: str

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = None
        self.transform = None

    @abstractmethod
    def load(self) -> None:
        ...

    @torch.no_grad()
    def encode(self, images: list[Image.Image] | list[np.ndarray]) -> np.ndarray:
        if self.model is None:
            self.load()
        batch = torch.stack([self.transform(self._to_pil(im)) for im in images]).to(self.device)
        features = self.model(batch)
        if isinstance(features, dict):  # some HF models return a dict
            features = features.get("pooler_output", features.get("last_hidden_state"))
            if features.dim() == 3:
                features = features[:, 0]  # CLS token
        return features.cpu().numpy()

    @staticmethod
    def _to_pil(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        return Image.fromarray(image)


class Virchow2Encoder(BaseEncoder):
    name = "virchow2"

    def load(self) -> None:
        import timm

        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow2", pretrained=True, num_classes=0
        ).eval().to(self.device)
        cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**cfg)


class UNIEncoder(BaseEncoder):
    name = "uni"

    def load(self) -> None:
        import timm

        self.model = timm.create_model(
            "hf-hub:MahmoodLab/uni", pretrained=True, num_classes=0
        ).eval().to(self.device)
        cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**cfg)


class CONCHEncoder(BaseEncoder):
    name = "conch"

    def load(self) -> None:
        # CONCH ships its own loader (open_clip-style); see MahmoodLab/CONCH.
        from conch.open_clip_custom import create_model_from_pretrained

        self.model, self.transform = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/conch"
        )
        self.model = self.model.eval().to(self.device)

    @torch.no_grad()
    def encode(self, images) -> np.ndarray:
        if self.model is None:
            self.load()
        batch = torch.stack([self.transform(self._to_pil(im)) for im in images]).to(self.device)
        features = self.model.encode_image(batch, proj_contrast=False, normalize=False)
        return features.cpu().numpy()


class DINOv3Encoder(BaseEncoder):
    name = "dinov3"

    def load(self) -> None:
        import timm

        self.model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0
        ).eval().to(self.device)
        cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**cfg)


ENCODER_REGISTRY: dict[str, type[BaseEncoder]] = {
    "virchow2": Virchow2Encoder,
    "uni": UNIEncoder,
    "conch": CONCHEncoder,
    "dinov3": DINOv3Encoder,
}


def build_encoder(name: str, device: str | None = None) -> BaseEncoder:
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Options: {list(ENCODER_REGISTRY)}")
    kwargs = {"device": device} if device else {}
    return ENCODER_REGISTRY[name](**kwargs)
