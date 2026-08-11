"""Train the VQ-VAE latent encoder (Module 6).

Usage:
    python train_vqvae.py                          # uses configs/config.yaml
    python train_vqvae.py vqvae.training.epochs=50  # Hydra overrides
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

from src.models.vqvae.losses import VQVAELoss
from src.models.vqvae.model import VQVAE
from src.utils.mlflow_utils import log_config_artifact, mlflow_run
from src.utils.seed import set_seed


class PatchImageDataset(Dataset):
    """Loads normalized patches from dataset/normalized/<class>/*.png."""

    def __init__(self, root_dir: str, image_size: int = 256):
        self.paths = sorted(Path(root_dir).rglob("*.png"))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        from PIL import Image

        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.vqvae.training.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PatchImageDataset(cfg.paths.normalized)
    loader = DataLoader(
        dataset,
        batch_size=cfg.vqvae.training.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.dataloader.num_workers,
        pin_memory=True,
    )

    model = VQVAE(**cfg.vqvae.model).to(device)
    loss_fn = VQVAELoss(**cfg.vqvae.loss).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.vqvae.training.lr, weight_decay=cfg.vqvae.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.vqvae.training.epochs)
    scaler = GradScaler(enabled=cfg.vqvae.training.mixed_precision)

    ckpt_dir = Path(cfg.paths.checkpoints.vqvae)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with mlflow_run(cfg, run_name="vqvae_training"):
        log_config_artifact(cfg)
        step = 0
        for epoch in range(cfg.vqvae.training.epochs):
            model.train()
            pbar = tqdm(loader, desc=f"VQ-VAE epoch {epoch}")
            for batch in pbar:
                batch = batch.to(device)
                optimizer.zero_grad()

                with autocast(enabled=cfg.vqvae.training.mixed_precision):
                    recon, vq_loss, vq_info = model(batch)
                    total_loss, log_dict = loss_fn(recon, batch, vq_info)
                    total_loss = total_loss + vq_loss

                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.vqvae.training.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                if step % 50 == 0:
                    mlflow.log_metrics(log_dict, step=step)
                pbar.set_postfix(loss=log_dict["loss/total"])
                step += 1

            scheduler.step()

            if (epoch + 1) % cfg.vqvae.training.checkpoint_every_n_epochs == 0:
                ckpt_path = ckpt_dir / f"vqvae_epoch{epoch + 1}.pt"
                torch.save(model.state_dict(), ckpt_path)
                mlflow.log_artifact(str(ckpt_path))

        final_path = ckpt_dir / "vqvae_final.pt"
        torch.save(model.state_dict(), final_path)
        mlflow.log_artifact(str(final_path))


if __name__ == "__main__":
    main()
