"""VQ-VAE latent encoder (Module 6): Encoder -> vector-quantized latent z ->
Decoder -> Reconstruction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        channel_multipliers: list[int],
        num_res_blocks: int,
        latent_channels: int,
    ):
        super().__init__()
        layers = [nn.Conv2d(in_channels, base_channels, 3, padding=1)]
        ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                layers.append(ResBlock(ch))
                if ch != out_ch:
                    layers.append(nn.Conv2d(ch, out_ch, 1))
                    ch = out_ch
            if i < len(channel_multipliers) - 1:
                layers.append(Downsample(ch))
        layers += [nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv2d(ch, latent_channels, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        base_channels: int,
        channel_multipliers: list[int],
        num_res_blocks: int,
        latent_channels: int,
    ):
        super().__init__()
        ch = base_channels * channel_multipliers[-1]
        layers = [nn.Conv2d(latent_channels, ch, 1)]
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                layers.append(ResBlock(ch))
                if ch != out_ch:
                    layers.append(nn.Conv2d(ch, out_ch, 1))
                    ch = out_ch
            if i < len(channel_multipliers) - 1:
                layers.append(Upsample(ch))
        layers += [
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VectorQuantizerEMA(nn.Module):
    """VQ layer with an exponential-moving-average codebook update
    (van den Oord et al., 2017 / Razavi et al., 2019)."""

    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        use_ema: bool = True,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.use_ema = use_ema

        embed = torch.randn(codebook_size, codebook_dim)
        self.register_buffer("embedding", embed)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        # z: (B, C, H, W) -> (B, H, W, C)
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat = z_perm.view(-1, self.codebook_dim)

        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(1)
        )
        indices = distances.argmin(1)
        one_hot = F.one_hot(indices, self.codebook_size).type(flat.dtype)
        quantized = one_hot @ self.embedding
        quantized = quantized.view(z_perm.shape)

        if self.training and self.use_ema:
            self.cluster_size.data.mul_(self.ema_decay).add_(
                one_hot.sum(0), alpha=1 - self.ema_decay
            )
            embed_sum = one_hot.t() @ flat
            self.embed_avg.data.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n
            )
            self.embedding.data.copy_(self.embed_avg / cluster_size.unsqueeze(1))

        commitment_loss = F.mse_loss(quantized.detach(), z_perm)
        codebook_loss = F.mse_loss(quantized, z_perm.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized = z_perm + (quantized - z_perm).detach()  # straight-through
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        perplexity = torch.exp(
            -torch.sum(
                (one_hot.mean(0) + 1e-10) * torch.log(one_hot.mean(0) + 1e-10)
            )
        )
        return quantized, loss, {
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "perplexity": perplexity,
            "indices": indices.view(z.shape[0], z.shape[2], z.shape[3]),
        }


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: list[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        latent_channels: int = 4,
        codebook_size: int = 8192,
        codebook_dim: int = 4,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        use_ema_codebook: bool = True,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels, base_channels, list(channel_multipliers), num_res_blocks, latent_channels
        )
        self.quantizer = VectorQuantizerEMA(
            codebook_size, codebook_dim, commitment_cost, ema_decay, use_ema_codebook
        )
        self.decoder = Decoder(
            in_channels, base_channels, list(channel_multipliers), num_res_blocks, latent_channels
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        z = self.encoder(x)
        z_q, vq_loss, vq_info = self.quantizer(z)
        recon = self.decoder(z_q)
        return recon, vq_loss, vq_info
