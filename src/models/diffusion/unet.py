"""Conditional U-Net noise predictor for latent diffusion (Module 7)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv.unbind(1)
        attn = torch.softmax((q.transpose(-1, -2) @ k) / (c // self.num_heads) ** 0.5, dim=-1)
        out = (v @ attn.transpose(-1, -2)).reshape(b, c, h, w)
        return x + self.proj(out)


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 192,
        channel_multipliers: list[int] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attention_resolutions: list[int] = (16, 8),
        num_heads: int = 8,
        dropout: float = 0.1,
        cond_dim: int = 512,
    ):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.cond_proj = nn.Linear(cond_dim, time_dim)
        self.base_channels = base_channels

        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        ch = base_channels
        input_block_chans = [ch]
        resolution = 64  # latent resolution assumption; relative logic only
        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, out_ch, time_dim, dropout)])
                if resolution in attention_resolutions:
                    block.append(AttentionBlock(out_ch, num_heads))
                self.down_blocks.append(block)
                ch = out_ch
                input_block_chans.append(ch)
            if level < len(channel_multipliers) - 1:
                self.down_blocks.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                input_block_chans.append(ch)
                resolution //= 2

        self.mid_block1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid_attn = AttentionBlock(ch, num_heads)
        self.mid_block2 = ResBlock(ch, ch, time_dim, dropout)

        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = input_block_chans.pop()
                block = nn.ModuleList([ResBlock(ch + skip_ch, out_ch, time_dim, dropout)])
                if resolution in attention_resolutions:
                    block.append(AttentionBlock(out_ch, num_heads))
                self.up_blocks.append(block)
                ch = out_ch
            if level > 0:
                self.up_blocks.append(
                    nn.ModuleList([nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)])
                )
                resolution *= 2

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, cond_emb: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(timesteps, self.base_channels))
        emb = t_emb + self.cond_proj(cond_emb)

        h = self.input_conv(x)
        hs = [h]
        for block in self.down_blocks:
            if isinstance(block[0], nn.Conv2d):
                h = block[0](h)
            else:
                h = block[0](h, emb)
                if len(block) > 1:
                    h = block[1](h)
            hs.append(h)

        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        for block in self.up_blocks:
            if isinstance(block[0], nn.ConvTranspose2d):
                h = block[0](h)
            else:
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = block[0](h, emb)
                if len(block) > 1:
                    h = block[1](h)

        return self.out_conv(F.silu(self.out_norm(h)))
