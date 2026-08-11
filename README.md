# Synthetic Histopathology Image Generation Pipeline

Gastric H&E synthesis pipeline for four classes — **Normal, HP (H. pylori),
IM (Intestinal Metaplasia), Mixed** — using a VQ-VAE latent space and a
conditional latent diffusion model, with full MLflow experiment tracking
and Hydra configuration.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --break-system-packages   # or drop the flag in a venv
```

Some pathology foundation-model encoders (Virchow2, UNI, CONCH) are gated on
Hugging Face. Before running embedding extraction:

```bash
huggingface-cli login
```
and accept the model licenses on the respective HF model pages.

## Data layout expected by `build_dataset.py`

```
dataset/raw/
    Normal/    *.svs | *.ndpi | *.tiff | *.png | *.jpg
    HP/        ...
    IM/        ...
    Mixed/     ...
```

## Pipeline order

```bash
# 1. Preprocess: tissue detection -> patch extraction -> artifact scoring
#    -> stain normalization -> embedding extraction (Modules 1-4)
python build_dataset.py

# 2. Inspect the embedding space before training (Module 5)
#    (see src/visualization/dataset_analysis.py — call from a notebook,
#    or wire into a small analyze_dataset.py if you want a CLI entry point)

# 3. Train the VQ-VAE latent encoder (Module 6)
python train_vqvae.py

# 4. Train the conditional latent diffusion model (Module 7)
python train_diffusion.py

# 5. Generate a synthetic dataset (Modules 8-10)
python generate_dataset.py generation.num_images=2000

# 6. Evaluate image quality / embedding / diversity metrics (Module 11)
python evaluate.py

# 7. Run the privacy audit BEFORE releasing any synthetic data (Module 11)
python privacy_test.py

# Optional: local MLflow UI
python mlflow_server.py --port 5000
```

## Configuration

All settings live under `configs/` (Hydra). Override anything from the
command line, e.g.:

```bash
python train_diffusion.py diffusion.training.epochs=1000 diffusion.model.cfg.guidance_scale=6.0
python build_dataset.py normalization.method=vahadane
python generate_dataset.py generation.classes=[HP,IM] generation.num_images=500
```

## Project structure

```
synthetic_histopathology/
  configs/                  Hydra configs (dataset, normalization, vqvae, diffusion, ...)
  src/
    dataset/                WSI loading + patch extraction
    preprocessing/          Tissue detection, artifact removal
    normalization/          Macenko / Vahadane stain normalization
    embeddings/             Virchow2 / UNI / CONCH / DINOv3 encoders
    models/
      vqvae/                VQ-VAE encoder/decoder + composite loss
      diffusion/             Conditional U-Net, DDPM/DDIM schedule, EMA
    evaluation/              Image quality, embedding, diversity, downstream utility
    privacy/                 Nearest-neighbor / duplicate / MIA privacy audits
    augmentation/             Domain randomization
    visualization/            PCA/t-SNE/UMAP dataset analysis
    utils/                    Seeding, MLflow helpers, metadata schema
  build_dataset.py           Modules 1-4 pipeline
  train_vqvae.py             Module 6
  train_diffusion.py         Module 7
  generate_dataset.py        Modules 8-10
  evaluate.py                Module 11 (quality/embedding/diversity)
  privacy_test.py            Module 11 (privacy audit)
  mlflow_server.py           Module 12 (local tracking UI)
```

## Notes / things you'll need to fill in for your environment

- **GPU compute.** VQ-VAE + diffusion training need a real GPU; nothing here
  was trained or benchmarked in this environment.
- **Downstream classifier code (Module 11) is for internal evaluation only** —
  it estimates whether synthetic data is useful, and isn't a submission
  deliverable.
- **Stain-matrix targets:** `normalization.target_image` in
  `configs/normalization/default.yaml` is `null` by default (uses a canonical
  H&E reference); point it at one of your own reference patches for
  slide-specific normalization.
- **`analyze_dataset.py`** (Module 5 CLI) isn't wired up as a standalone
  script — the functions in `src/visualization/dataset_analysis.py` are
  ready to call from a notebook or a thin script you add.
- Adjust `dataset.classes`, patch size, magnification, and encoder choice in
  `configs/` to match your actual cohort and hardware budget.
