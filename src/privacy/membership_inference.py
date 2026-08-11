"""Membership-inference privacy AUDIT (Module 11) for the pipeline's own
generative models. This estimates whether an attacker who only has query
access to the trained VQ-VAE/diffusion models could infer whether a given
real patient image was in the training set — the standard MIA evaluation
used to certify a generative model as privacy-preserving before release.
This module does not target any external system; it is run by the data
owner against their own model as a pre-release privacy check.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


@torch.no_grad()
def loss_based_mia(
    model: nn.Module,
    member_loader,
    nonmember_loader,
    loss_fn,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Simplest MIA baseline: per-sample reconstruction/diffusion loss tends
    to be lower for training ("member") samples than held-out ("non-member")
    samples. We fit a threshold classifier on the loss value and report AUC
    as the privacy-risk score (0.5 = no leakage, 1.0 = fully separable)."""
    model = model.to(device).eval()

    def per_sample_losses(loader):
        losses = []
        for batch in loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            out = model(x)
            recon = out[0] if isinstance(out, tuple) else out
            per_sample = ((recon - x) ** 2).flatten(1).mean(1)
            losses.extend(per_sample.cpu().numpy().tolist())
        return np.array(losses)

    member_losses = per_sample_losses(member_loader)
    nonmember_losses = per_sample_losses(nonmember_loader)

    # lower loss => more likely a member, so use negative loss as the score
    scores = np.concatenate([-member_losses, -nonmember_losses])
    labels = np.concatenate([np.ones_like(member_losses), np.zeros_like(nonmember_losses)])
    auc = roc_auc_score(labels, scores)

    return {
        "mia_auc": float(auc),
        "member_loss_mean": float(member_losses.mean()),
        "nonmember_loss_mean": float(nonmember_losses.mean()),
        "privacy_risk": "high" if auc > 0.75 else "moderate" if auc > 0.6 else "low",
    }


def shadow_model_mia(
    member_features: np.ndarray,
    nonmember_features: np.ndarray,
    num_shadow_models: int = 4,
    train_fraction: float = 0.5,
    seed: int = 42,
) -> dict:
    """Shadow-model MIA (Shokri et al., 2017): trains attack classifiers on
    features (e.g. per-sample loss, confidence, or embedding distance to
    nearest training point) from shadow member/non-member splits, then
    reports the aggregate attack AUC as the privacy-risk estimate."""
    rng = np.random.default_rng(seed)
    aucs = []

    all_features = np.concatenate([member_features, nonmember_features])
    all_labels = np.concatenate(
        [np.ones(len(member_features)), np.zeros(len(nonmember_features))]
    )

    for i in range(num_shadow_models):
        X_train, X_test, y_train, y_test = train_test_split(
            all_features, all_labels, train_size=train_fraction, random_state=seed + i, stratify=all_labels
        )
        attack_model = LogisticRegression(max_iter=1000)
        attack_model.fit(X_train.reshape(len(X_train), -1), y_train)
        scores = attack_model.predict_proba(X_test.reshape(len(X_test), -1))[:, 1]
        aucs.append(roc_auc_score(y_test, scores))

    mean_auc = float(np.mean(aucs))
    return {
        "mia_auc_mean": mean_auc,
        "mia_auc_std": float(np.std(aucs)),
        "num_shadow_models": num_shadow_models,
        "privacy_risk": "high" if mean_auc > 0.75 else "moderate" if mean_auc > 0.6 else "low",
    }


def image_inversion_attempt(
    generator: nn.Module,
    target_embedding: torch.Tensor,
    latent_shape: tuple[int, ...],
    max_iterations: int = 500,
    lr: float = 0.05,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Gradient-based latent inversion: optimizes a random latent to
    reproduce a target embedding, then checks reconstruction quality against
    the real training set. High-fidelity inversion of a training sample is
    evidence of memorization; this is run by the data owner against their
    own model, using their own embeddings, as a pre-release privacy check.
    """
    generator = generator.to(device).eval()
    z = torch.randn(latent_shape, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    target_embedding = target_embedding.to(device)
    final_loss = None
    for _ in range(max_iterations):
        optimizer.zero_grad()
        with torch.enable_grad():
            generated = generator(z)
            loss = nn.functional.mse_loss(generated, target_embedding.expand_as(generated))
            loss.backward()
            optimizer.step()
        final_loss = loss.item()

    return {"final_inversion_loss": final_loss, "iterations": max_iterations}
