"""Train the VQ-VAE latent encoder (Module 6).

Usage:
    python train_vqvae.py                          # uses configs/config.yaml
    python train_vqvae.py vqvae.training.epochs=50  # Hydra overrides
"""
from __future__ import annotations

import random
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.dataset.esd_dataset import esd_slide_paths, is_esd_dataset_root
from src.dataset.wsi_loader import SlideReader, discover_input_files
from src.models.vqvae.losses import VQVAELoss
from src.models.vqvae.model import VQVAE
from src.normalization.normalize import build_normalizer
from src.preprocessing.tissue_detection import detect_tissue_mask, tissue_percentage
from src.utils.mlflow_utils import log_config_artifact, mlflow_run
from src.utils.seed import set_seed


def _log(message: str) -> None:
    print(message, flush=True)


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
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)


class RawSlidePatchDataset(Dataset):
    """Samples tissue-rich patches directly from raw slides each epoch."""

    def __init__(self, cfg: DictConfig):
        training_cfg = cfg.vqvae.training
        self.raw_dir = Path(cfg.paths.data.raw)
        self.patch_size = int(cfg.dataset.patching.patch_size)
        self.minimum_tissue_percentage = float(cfg.dataset.patching.minimum_tissue_percentage)
        self.tissue_threshold_method = str(cfg.dataset.patching.tissue_threshold_method)
        self.image_size = int(self.patch_size)
        self.patches_per_slide_per_epoch = int(training_cfg.raw_patches_per_slide_per_epoch)
        self.max_sampling_attempts = int(training_cfg.raw_max_sampling_attempts)
        self.base_seed = int(training_cfg.seed)
        self.enable_normalization = bool(training_cfg.raw_apply_normalization)
        self.normalizer = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        self.slide_paths = self._discover_slide_paths(cfg)
        if not self.slide_paths:
            raise ValueError(f"No raw slide/image inputs found under {self.raw_dir}")

        if self.enable_normalization:
            self.normalizer = build_normalizer(
                method=cfg.normalization.method,
                alpha=cfg.normalization.alpha,
                beta=cfg.normalization.beta,
                luminosity_threshold=cfg.normalization.luminosity_threshold,
            )
            if cfg.normalization.target_image:
                target = np.array(Image.open(cfg.normalization.target_image).convert("RGB"))
                self.normalizer.fit(target)

    def _discover_slide_paths(self, cfg: DictConfig) -> list[Path]:
        if is_esd_dataset_root(self.raw_dir):
            return esd_slide_paths(self.raw_dir)

        grouped_paths: list[Path] = []
        excluded = list(cfg.dataset.ingestion.get("excluded_dir_names", []))
        for class_name in cfg.dataset.classes:
            class_dir = self.raw_dir / class_name
            if not class_dir.exists():
                continue
            grouped_paths.extend(
                discover_input_files(
                    class_dir,
                    cfg.dataset.wsi.input_format,
                    recursive=True,
                    excluded_dir_names=excluded,
                )
            )

        if grouped_paths:
            return grouped_paths

        return discover_input_files(
            self.raw_dir,
            cfg.dataset.wsi.input_format,
            recursive=True,
            excluded_dir_names=excluded,
        )

    def __len__(self) -> int:
        return len(self.slide_paths) * self.patches_per_slide_per_epoch

    def _rng_for_index(self, idx: int) -> random.Random:
        return random.Random(self.base_seed + idx)

    def _sample_patch(self, slide_path: Path, rng: random.Random) -> Image.Image:
        reader = SlideReader(slide_path)
        try:
            width, height = reader.dimensions
            if width < self.patch_size or height < self.patch_size:
                raise ValueError(
                    f"Slide {slide_path} is smaller than patch size {self.patch_size}"
                )

            for _ in range(self.max_sampling_attempts):
                max_x = max(width - self.patch_size, 0)
                max_y = max(height - self.patch_size, 0)
                x = 0 if max_x == 0 else rng.randint(0, max_x)
                y = 0 if max_y == 0 else rng.randint(0, max_y)
                patch = reader.read_region((x, y), level=0, size=(self.patch_size, self.patch_size))
                if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                    continue
                tissue_mask = detect_tissue_mask(patch, method=self.tissue_threshold_method)
                if tissue_percentage(tissue_mask) < self.minimum_tissue_percentage:
                    continue
                if self.normalizer is not None:
                    patch = self.normalizer.transform(patch)
                return Image.fromarray(patch)

            raise RuntimeError(
                f"Failed to sample a tissue-rich patch from {slide_path.name} after "
                f"{self.max_sampling_attempts} attempts"
            )
        finally:
            reader.close()

    def __getitem__(self, idx: int) -> torch.Tensor:
        slide_path = self.slide_paths[idx % len(self.slide_paths)]
        rng = self._rng_for_index(idx)
        image = self._sample_patch(slide_path, rng)
        return self.transform(image)


def _build_training_dataset(cfg: DictConfig) -> Dataset:
    input_mode = str(cfg.vqvae.training.get("input_mode", "normalized"))
    if input_mode == "raw":
        dataset = RawSlidePatchDataset(cfg)
        _log(
            f"Training VQ-VAE from raw slides in {cfg.paths.data.raw} "
            f"with {len(dataset.slide_paths)} inputs and {len(dataset)} sampled patches per epoch"
        )
        return dataset

    dataset = PatchImageDataset(cfg.paths.normalized)
    _log(f"Training VQ-VAE from normalized patches in {cfg.paths.normalized}: {len(dataset)} files")
    return dataset


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.vqvae.training.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Using device: {device}")

    dataset = _build_training_dataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.vqvae.training.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.dataloader.num_workers,
        pin_memory=True,
    )
    _log(
        f"DataLoader ready: batch_size={cfg.vqvae.training.batch_size}, "
        f"num_workers={cfg.dataset.dataloader.num_workers}, batches_per_epoch={len(loader)}"
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
    _log(f"Checkpoint directory: {ckpt_dir}")

    with mlflow_run(cfg, run_name="vqvae_training"):
        log_config_artifact(cfg)
        step = 0
        for epoch in range(cfg.vqvae.training.epochs):
            model.train()
            _log(f"Starting epoch {epoch + 1}/{cfg.vqvae.training.epochs}")
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
                _log(f"Saved checkpoint: {ckpt_path}")

        final_path = ckpt_dir / "vqvae_final.pt"
        torch.save(model.state_dict(), final_path)
        mlflow.log_artifact(str(final_path))
        _log(f"Saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
