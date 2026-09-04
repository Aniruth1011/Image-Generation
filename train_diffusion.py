"""Train the conditional latent diffusion model (Module 7).

Requires a trained VQ-VAE checkpoint (see train_vqvae.py) to encode images
into the latent space the diffusion model operates on.

Usage:
    python train_diffusion.py vqvae_checkpoint=checkpoints/vqvae/vqvae_final.pt
"""
from __future__ import annotations

import random
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.dataset.esd_dataset import ESDPatchLabeler, esd_class_names, is_esd_dataset_root
from src.dataset.label_space import resolve_class_names
from src.dataset.wsi_loader import SlideReader, discover_input_files
from src.models.diffusion.conditioning import ClassConditioning
from src.models.diffusion.ddpm import EMA, GaussianDiffusion
from src.models.diffusion.unet import UNetModel
from src.models.vqvae.model import VQVAE
from src.normalization.normalize import build_normalizer
from src.preprocessing.tissue_detection import detect_tissue_mask, tissue_percentage
from src.utils.mlflow_utils import log_config_artifact, mlflow_run
from src.utils.seed import set_seed


def _log(message: str) -> None:
    print(message, flush=True)


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


class RawLabeledSlideDataset(Dataset):
    """Samples tissue-rich, labeled patches directly from raw inputs each epoch."""

    def __init__(self, cfg: DictConfig, classes: list[str], image_size: int = 256):
        training_cfg = cfg.diffusion.training
        self.raw_dir = Path(cfg.paths.data.raw)
        self.classes = classes
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        self.patch_size = int(cfg.dataset.patching.patch_size)
        self.minimum_tissue_percentage = float(cfg.dataset.patching.minimum_tissue_percentage)
        self.tissue_threshold_method = str(cfg.dataset.patching.tissue_threshold_method)
        self.image_size = int(image_size)
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
        self.slide_entries = self._discover_slide_entries(cfg)
        if not self.slide_entries:
            raise ValueError(f"No labeled raw slide/image inputs found under {self.raw_dir}")

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

    def _discover_slide_entries(self, cfg: DictConfig) -> list[dict]:
        if is_esd_dataset_root(self.raw_dir):
            entries = []
            for slide_path in sorted(self.raw_dir.glob("*.svs")):
                entries.append(
                    {
                        "slide_path": slide_path,
                        "class_name": None,
                        "labeler": ESDPatchLabeler(
                            self.raw_dir,
                            slide_path,
                            downsample_factor=cfg.dataset.ingestion.get(
                                "esd_annotation_downsample_factor", 64
                            ),
                            label_mode=cfg.dataset.ingestion.get("esd_label_mode", "fine"),
                            min_label_fraction=cfg.dataset.ingestion.get(
                                "esd_min_label_fraction", 0.05
                            ),
                        ),
                    }
                )
            return entries

        excluded = list(cfg.dataset.ingestion.get("excluded_dir_names", []))
        grouped_entries: list[dict] = []
        for class_name in self.classes:
            class_dir = self.raw_dir / class_name
            if not class_dir.exists():
                continue
            for slide_path in discover_input_files(
                class_dir,
                cfg.dataset.wsi.input_format,
                recursive=True,
                excluded_dir_names=excluded,
            ):
                grouped_entries.append(
                    {"slide_path": slide_path, "class_name": class_name, "labeler": None}
                )
        return grouped_entries

    def __len__(self) -> int:
        return len(self.slide_entries) * self.patches_per_slide_per_epoch

    def _rng_for_index(self, idx: int) -> random.Random:
        return random.Random(self.base_seed + idx)

    def _sample_patch(self, entry: dict, rng: random.Random) -> tuple[Image.Image, int]:
        slide_path = entry["slide_path"]
        labeler = entry["labeler"]
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

                class_name = entry["class_name"]
                if labeler is not None:
                    patch_label = labeler.label_patch(x, y, self.patch_size)
                    if patch_label is None:
                        continue
                    class_name = patch_label.class_name

                if class_name not in self.class_to_idx:
                    continue

                if self.normalizer is not None:
                    patch = self.normalizer.transform(patch)

                return Image.fromarray(patch), self.class_to_idx[class_name]

            raise RuntimeError(
                f"Failed to sample a labeled tissue-rich patch from {slide_path.name} after "
                f"{self.max_sampling_attempts} attempts"
            )
        finally:
            reader.close()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rng = self._rng_for_index(idx)
        num_slides = len(self.slide_entries)
        start_offset = idx % num_slides
        last_error: RuntimeError | None = None

        for offset in range(num_slides):
            entry = self.slide_entries[(start_offset + offset) % num_slides]
            try:
                image, label = self._sample_patch(entry, rng)
                return self.transform(image), label
            except RuntimeError as exc:
                last_error = exc
                continue

        raise RuntimeError(
            "Failed to sample a labeled tissue-rich patch from any slide in the dataset"
        ) from last_error


def _build_training_dataset(cfg: DictConfig, class_names: list[str]) -> Dataset:
    input_mode = str(cfg.diffusion.training.get("input_mode", "normalized"))
    if input_mode == "raw":
        dataset = RawLabeledSlideDataset(cfg, class_names)
        _log(
            f"Training diffusion from raw slides in {cfg.paths.data.raw} "
            f"with {len(dataset.slide_entries)} inputs and {len(dataset)} sampled patches per epoch"
        )
        return dataset

    dataset = LabeledPatchDataset(cfg.paths.normalized, class_names)
    _log(f"Training diffusion from normalized patches in {cfg.paths.normalized}: {len(dataset)} files")
    return dataset


def _resolve_training_class_names(cfg: DictConfig) -> list[str]:
    input_mode = str(cfg.diffusion.training.get("input_mode", "normalized"))
    if input_mode == "raw":
        raw_dir = Path(cfg.paths.data.raw)
        if is_esd_dataset_root(raw_dir):
            return esd_class_names(cfg.dataset.ingestion.get("esd_label_mode", "fine"))

        grouped_classes = [
            class_name for class_name in cfg.dataset.classes if (raw_dir / class_name).exists()
        ]
        if grouped_classes:
            return grouped_classes

    return resolve_class_names(cfg.paths.normalized, cfg.paths.data.metadata)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.diffusion.training.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Using device: {device}")
    class_names = _resolve_training_class_names(cfg)

    # --- frozen VQ-VAE for latent encoding ---
    vqvae = VQVAE(**cfg.vqvae.model).to(device).eval()
    vqvae_ckpt = cfg.get("vqvae_checkpoint", str(Path(cfg.paths.checkpoints.vqvae) / "vqvae_final.pt"))
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
    for p in vqvae.parameters():
        p.requires_grad_(False)

    dataset = _build_training_dataset(cfg, class_names)
    loader = DataLoader(
        dataset,
        batch_size=cfg.diffusion.training.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.dataloader.num_workers,
        pin_memory=True,
    )
    _log(
        f"DataLoader ready: batch_size={cfg.diffusion.training.batch_size}, "
        f"num_workers={cfg.dataset.dataloader.num_workers}, batches_per_epoch={len(loader)}"
    )

    unet = UNetModel(
        **cfg.diffusion.model.unet, cond_dim=cfg.diffusion.model.conditioning.embedding_dim
    ).to(device)
    diffusion_cfg = OmegaConf.to_container(cfg.diffusion.model.noise_schedule, resolve=True)
    diffusion_cfg["schedule_type"] = diffusion_cfg.pop("type")
    diffusion = GaussianDiffusion(unet, **diffusion_cfg).to(device)
    conditioner = ClassConditioning(
        len(class_names), cfg.diffusion.model.conditioning.embedding_dim
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
        mlflow_enabled = mlflow.active_run() is not None
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

                if mlflow_enabled and step % 50 == 0:
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
                if mlflow_enabled:
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
        if mlflow_enabled:
            mlflow.log_artifact(str(final_path))


if __name__ == "__main__":
    main()
