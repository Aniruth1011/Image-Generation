"""Noise schedule + training/sampling logic for conditional latent diffusion
(Module 7): classifier-free guidance, EMA, mixed precision, gradient
accumulation, checkpointing are orchestrated from train_diffusion.py; this
module holds the schedule math and the DDPM/DDIM step logic.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(schedule_type: str, timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, timesteps)
    if schedule_type == "cosine":
        s = 0.008
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 1e-4, 0.999)
    raise ValueError(f"Unknown schedule: {schedule_type}")


class GaussianDiffusion(nn.Module):
    """Wraps a noise-prediction UNet with the forward/reverse diffusion math."""

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        schedule_type: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = make_beta_schedule(schedule_type, timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1mac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_1mac * noise

    def training_loss(
        self, x0: torch.Tensor, cond_emb: torch.Tensor
    ) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = self.model(x_t, t, cond_emb)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: tuple[int, ...],
        cond_emb: torch.Tensor,
        null_emb: torch.Tensor | None,
        guidance_scale: float = 4.0,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        device: str = "cuda",
    ) -> torch.Tensor:
        step_indices = torch.linspace(
            0, self.timesteps - 1, num_inference_steps, dtype=torch.long
        ).flip(0)
        x = torch.randn(shape, device=device)

        for i, t in enumerate(step_indices):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            if guidance_scale > 1.0 and null_emb is not None:
                eps_cond = self.model(x, t_batch, cond_emb)
                eps_uncond = self.model(x, t_batch, null_emb)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = self.model(x, t_batch, cond_emb)

            alpha_t = self.alphas_cumprod[t]
            alpha_prev = (
                self.alphas_cumprod[step_indices[i + 1]]
                if i + 1 < len(step_indices)
                else torch.tensor(1.0, device=device)
            )
            x0_pred = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)

            sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = (
                alpha_prev.sqrt() * x0_pred
                + (1 - alpha_prev - sigma**2).sqrt() * eps
                + sigma * noise
            )
        return x


class EMA:
    """Exponential moving average of model weights (Module 7)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()
