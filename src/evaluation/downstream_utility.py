"""Downstream-utility evaluation (Module 11, internal only): train a
lightweight classifier under real-only / real+synthetic / synthetic-only
regimes and compare accuracy, precision, recall, F1, ROC-AUC, balanced
accuracy and confusion matrix. Not part of the competition deliverable.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


def build_classifier(model_name: str, num_classes: int) -> nn.Module:
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module, test_loader: DataLoader, num_classes: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    model = model.to(device).eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }
    if num_classes == 2:
        metrics["roc_auc"] = roc_auc_score(all_labels, [p[1] for p in all_probs])
    else:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    return metrics


def run_downstream_utility_suite(
    real_only_loaders: tuple[DataLoader, DataLoader],
    real_plus_synthetic_loaders: tuple[DataLoader, DataLoader],
    synthetic_only_loaders: tuple[DataLoader, DataLoader],
    model_name: str,
    num_classes: int,
    epochs: int,
    lr: float,
) -> dict:
    """Each *_loaders tuple is (train_loader, test_loader); test_loader is
    always the same held-out real test set across regimes."""
    results = {}
    for regime, (train_loader, test_loader) in {
        "real_only": real_only_loaders,
        "real_plus_synthetic": real_plus_synthetic_loaders,
        "synthetic_only": synthetic_only_loaders,
    }.items():
        model = build_classifier(model_name, num_classes)
        model = train_classifier(model, train_loader, epochs, lr)
        results[regime] = evaluate_classifier(model, test_loader, num_classes)
    return results
