"""Class / attribute conditioning for the latent diffusion model (Module 7).

Currently supports disease-class conditioning (Normal / HP / IM / Mixed);
the `extra` dict is a placeholder for future scalar conditions (disease
burden, stain intensity, artifact level, magnification), each of which can
be embedded and concatenated/summed with the class embedding.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassConditioning(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int):
        super().__init__()
        # +1 slot reserved for the "unconditional" (CFG null) token
        self.embedding = nn.Embedding(num_classes + 1, embedding_dim)
        self.null_token_id = num_classes

    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(class_ids)

    def null_conditioning(self, batch_size: int, device: torch.device) -> torch.Tensor:
        ids = torch.full((batch_size,), self.null_token_id, dtype=torch.long, device=device)
        return self.embedding(ids)


class ScalarConditioning(nn.Module):
    """Embeds a continuous scalar attribute (e.g. disease_burden in [0, 1])
    via sinusoidal features + an MLP, for future extra-conditioning support."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        half = self.embedding_dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=values.device) * (9.2103 / half)  # log(1e4)
        )
        args = values[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)
