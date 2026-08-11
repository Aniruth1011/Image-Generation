"""VQ-VAE composite loss: Reconstruction + Perceptual (LPIPS) + Codebook +
Commitment (Module 6)."""
from __future__ import annotations

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVAELoss(nn.Module):
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        codebook_weight: float = 1.0,
        commitment_weight: float = 0.25,
        perceptual_net: str = "vgg",
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.codebook_weight = codebook_weight
        self.commitment_weight = commitment_weight
        self.lpips = lpips.LPIPS(net=perceptual_net)
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def forward(
        self, recon: torch.Tensor, target: torch.Tensor, vq_info: dict
    ) -> tuple[torch.Tensor, dict]:
        recon_loss = F.mse_loss(recon, target)
        perceptual_loss = self.lpips(recon, target).mean()
        codebook_loss = vq_info["codebook_loss"]
        commitment_loss = vq_info["commitment_loss"]

        total = (
            self.reconstruction_weight * recon_loss
            + self.perceptual_weight * perceptual_loss
            + self.codebook_weight * codebook_loss
            + self.commitment_weight * commitment_loss
        )
        return total, {
            "loss/total": total.item(),
            "loss/reconstruction": recon_loss.item(),
            "loss/perceptual": perceptual_loss.item(),
            "loss/codebook": codebook_loss.item(),
            "loss/commitment": commitment_loss.item(),
            "loss/perplexity": vq_info["perplexity"].item(),
        }
