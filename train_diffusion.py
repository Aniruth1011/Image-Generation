"""Train the conditional latent diffusion model (Module 7).

Requires a trained VQ-VAE checkpoint (see train_vqvae.py) to encode images
into the latent space the diffusion model operates on.

Usage:
    python train_diffusion.py vqvae_checkpoint=checkpoints/vqvae/vqvae_final.pt
"""
from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.models.diffusion.conditioning import ClassConditioning
from src.models.diffusion.ddpm import EMA, GaussianDiffusion
from src.models.diffusion.unet import UNetModel
from src.models.vqvae.model import VQVAE
from src.utils.mlflow_utils import log_config_artifact, mlflow_run
from src.utils.seed import set_seed


class LabeledPatchDataset(Dataset):
    """Expects dataset/normalized/<class_name>/*.png, one subfolder per class
    in the same order as cfg.dataset.classes."""

    def __init__(self, root_dir: str, classes: list[str], image_size: int = 256):
        self.classes = classes
        self.samples: list[tuple[Path, int]] = []
        for idx, cls in enumerate(classes):
            for p in sorted((Path(root_dir) / cls).glob("*.png")):
                self.samples.append((p, idx))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        from PIL import Image

        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.diffusion.training.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- frozen VQ-VAE for latent encoding ---
    vqvae = VQVAE(**cfg.vqvae.model).to(device).eval()
    vqvae_ckpt = cfg.get("vqvae_checkpoint", str(Path(cfg.paths.checkpoints.vqvae) / "vqvae_final.pt"))
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
    for p in vqvae.parameters():
        p.requires_grad_(False)

    dataset = LabeledPatchDataset(cfg.paths.normalized, cfg.dataset.classes)
    loader = DataLoader(
        dataset,
        batch_size=cfg.diffusion.training.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.dataloader.num_workers,
        pin_memory=True,
    )

    unet = UNetModel(
        **cfg.diffusion.model.unet, cond_dim=cfg.diffusion.model.conditioning.embedding_dim
    ).to(device)
    diffusion = GaussianDiffusion(unet, **cfg.diffusion.model.noise_schedule).to(device)
    conditioner = ClassConditioning(
        cfg.diffusion.model.conditioning.num_classes, cfg.diffusion.model.conditioning.embedding_dim
    ).to(device)

    params = list(unet.parameters()) + list(conditioner.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=cfg.diffusion.training.lr, weight_decay=cfg.diffusion.training.weight_decay
    )
    scaler = GradScaler(enabled=cfg.diffusion.training.mixed_precision)
    ema = EMA(unet, decay=cfg.diffusion.model.ema.decay) if cfg.diffusion.model.ema.enabled else None

    ckpt_dir = Path(cfg.paths.checkpoints.diffusion)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    accum_steps = cfg.diffusion.training.grad_accum_steps
    uncond_p = cfg.diffusion.model.cfg.unconditional_prob

    with mlflow_run(cfg, run_name="diffusion_training"):
        log_config_artifact(cfg)
        step = 0
        for epoch in range(cfg.diffusion.training.epochs):
            unet.train()
            pbar = tqdm(loader, desc=f"Diffusion epoch {epoch}")
            optimizer.zero_grad()

            for i, (images, labels) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)

                with torch.no_grad():
                    latents = vqvae.encode(images)

                # classifier-free guidance: randomly drop conditioning
                drop_mask = torch.rand(labels.shape[0], device=device) < uncond_p
                cond_labels = labels.clone()
                cond_labels[drop_mask] = conditioner.null_token_id
                cond_emb = conditioner(cond_labels)

                with autocast(enabled=cfg.diffusion.training.mixed_precision):
                    loss = diffusion.training_loss(latents, cond_emb) / accum_steps

                scaler.scale(loss).backward()

                if (i + 1) % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, cfg.diffusion.training.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if ema is not None and step % cfg.diffusion.model.ema.update_every == 0:
                        ema.update(unet)

                if step % 50 == 0:
                    mlflow.log_metric("loss/diffusion", loss.item() * accum_steps, step=step)
                pbar.set_postfix(loss=loss.item() * accum_steps)
                step += 1

            if (epoch + 1) % cfg.diffusion.training.checkpoint_every_n_epochs == 0:
                ckpt_path = ckpt_dir / f"diffusion_epoch{epoch + 1}.pt"
                torch.save(
                    {
                        "unet": unet.state_dict(),
                        "conditioner": conditioner.state_dict(),
                        "ema": ema.state_dict() if ema else None,
                    },
                    ckpt_path,
                )
                mlflow.log_artifact(str(ckpt_path))

        final_path = ckpt_dir / "diffusion_final.pt"
        torch.save(
            {
                "unet": unet.state_dict(),
                "conditioner": conditioner.state_dict(),
                "ema": ema.state_dict() if ema else None,
            },
            final_path,
        )
        mlflow.log_artifact(str(final_path))


if __name__ == "__main__":
    main()
