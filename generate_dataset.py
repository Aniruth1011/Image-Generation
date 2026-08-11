"""Generate a synthetic dataset (Module 8): by class, by seed, or a fixed
count, with domain randomization (Module 9) and full metadata export
(Module 10).

Usage:
    python generate_dataset.py generation.num_images=1000
    python generate_dataset.py generation.classes=[HP,IM]
"""
from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from src.dataset.label_space import resolve_class_names
from src.augmentation.domain_randomization import apply_domain_randomization
from src.models.diffusion.conditioning import ClassConditioning
from src.models.diffusion.ddpm import GaussianDiffusion
from src.models.diffusion.unet import UNetModel
from src.models.vqvae.model import VQVAE
from src.utils.metadata import ImageMetadata, MetadataStore
from src.utils.seed import set_seed

GENERATOR_VERSION = "0.1.0"


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = resolve_class_names(cfg.paths.normalized, cfg.paths.data.metadata)
    num_images = cfg.get("generation", {}).get("num_images", 100)
    target_classes = cfg.get("generation", {}).get("classes", class_names)
    seed = cfg.get("generation", {}).get("seed", 42)
    set_seed(seed)

    vqvae = VQVAE(**cfg.vqvae.model).to(device).eval()
    vqvae.load_state_dict(
        torch.load(Path(cfg.paths.checkpoints.vqvae) / "vqvae_final.pt", map_location=device)
    )

    unet = UNetModel(
        **cfg.diffusion.model.unet, cond_dim=cfg.diffusion.model.conditioning.embedding_dim
    ).to(device)
    diffusion = GaussianDiffusion(unet, **cfg.diffusion.model.noise_schedule).to(device)
    conditioner = ClassConditioning(
        len(class_names), cfg.diffusion.model.conditioning.embedding_dim
    ).to(device)

    ckpt = torch.load(
        Path(cfg.paths.checkpoints.diffusion) / "diffusion_final.pt", map_location=device
    )
    unet.load_state_dict(ckpt["unet"])
    conditioner.load_state_dict(ckpt["conditioner"])
    unet.eval()

    out_dir = Path(cfg.paths.outputs.synthetic)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_store = MetadataStore()

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    images_per_class = num_images // len(target_classes)

    for class_name in target_classes:
        class_idx = class_to_idx[class_name]
        class_dir = out_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for batch_start in tqdm(
            range(0, images_per_class, 16), desc=f"Generating {class_name}"
        ):
            batch_size = min(16, images_per_class - batch_start)
            labels = torch.full((batch_size,), class_idx, dtype=torch.long, device=device)
            cond_emb = conditioner(labels)
            null_emb = conditioner.null_conditioning(batch_size, device)

            latent_h = latent_w = 256 // (2 ** (len(cfg.vqvae.model.channel_multipliers) - 1))
            latents = diffusion.ddim_sample(
                shape=(batch_size, cfg.vqvae.model.latent_channels, latent_h, latent_w),
                cond_emb=cond_emb,
                null_emb=null_emb,
                guidance_scale=cfg.diffusion.model.cfg.guidance_scale,
                num_inference_steps=cfg.diffusion.sampling.num_inference_steps,
                eta=cfg.diffusion.sampling.eta,
                device=device,
            )

            with torch.no_grad():
                images = vqvae.decoder(latents)
            images = ((images.clamp(-1, 1) + 1) / 2 * 255).byte().permute(0, 2, 3, 1).cpu().numpy()

            for i, img_arr in enumerate(images):
                img_idx = batch_start + i
                augmented = apply_domain_randomization(img_arr, cfg.augmentation)

                filename = f"{class_name}_{img_idx:05d}_seed{seed}.png"
                out_path = class_dir / filename
                Image.fromarray(augmented).save(out_path)

                metadata_store.add(
                    ImageMetadata(
                        filename=str(out_path),
                        class_label=class_name,
                        generator_version=GENERATOR_VERSION,
                        seed=seed,
                        augmentation="domain_randomization" if cfg.augmentation.enabled else "none",
                        latent_model="vqvae+diffusion",
                        image_size=img_arr.shape[0],
                        guidance_scale=cfg.diffusion.model.cfg.guidance_scale,
                        sampler=cfg.diffusion.sampling.sampler,
                        num_inference_steps=cfg.diffusion.sampling.num_inference_steps,
                    )
                )

    metadata_store.export_csv(out_dir / "metadata.csv")
    metadata_store.export_json(out_dir / "metadata.json")
    print(f"Generated {len(metadata_store._records)} images -> {out_dir}")


if __name__ == "__main__":
    main()
