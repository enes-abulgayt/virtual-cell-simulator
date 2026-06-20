"""Training utilities for the virtual cell model."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import SimulatorConfig
from .dataset import NormalizationStats
from .model import VirtualCellModel


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_default_device() -> torch.device:
    """Use CUDA when available while remaining fully CPU-compatible."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: VirtualCellModel,
    data_loader: DataLoader[dict[str, torch.Tensor]],
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    description: str = "Epoch",
) -> float:
    """Run one training or validation epoch and return mean MSE."""

    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(data_loader, desc=description, leave=False)
    for batch in progress:
        batch = _move_batch(batch, device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(
                batch["normalized_control_expression"],
                batch["perturbation_id"],
                batch["control_expression"],
            )
            loss = loss_function(outputs["predicted_delta"], batch["target_delta"])

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = batch["control_expression"].shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(mse=f"{loss.item():.5f}")

    return total_loss / max(total_samples, 1)


def save_checkpoint(
    path: Path,
    model: VirtualCellModel,
    config: SimulatorConfig,
    normalization: NormalizationStats,
    epoch: int,
    validation_loss: float,
    gene_names: np.ndarray,
    perturbation_names: np.ndarray,
) -> None:
    """Save model weights and all metadata needed for inference."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_metadata": {
                "num_perturbation_groups": model.num_perturbation_groups,
                "perturbation_group_ids": (
                    model.perturbation_group_ids.detach().cpu().tolist()
                    if model.group_embedding is not None
                    else None
                ),
            },
            "config": config.to_dict(),
            "normalization_mean": normalization.mean,
            "normalization_std": normalization.std,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "gene_names": gene_names.tolist(),
            "perturbation_names": perturbation_names.tolist(),
        },
        path,
    )


def load_model_from_checkpoint(
    path: Path,
    device: torch.device | None = None,
) -> tuple[VirtualCellModel, dict[str, Any]]:
    """Restore a trained model and its inference metadata."""

    device = device or get_default_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = SimulatorConfig.from_dict(checkpoint["config"])
    model_metadata = checkpoint.get("model_metadata", {})
    has_group_embedding = "group_embedding.weight" in checkpoint["model_state_dict"]
    num_groups = (
        int(model_metadata.get("num_perturbation_groups", 0))
        if has_group_embedding
        else 0
    )
    group_ids = model_metadata.get("perturbation_group_ids")
    model = VirtualCellModel(
        num_genes=config.num_genes,
        num_perturbations=config.num_perturbations,
        embedding_dim=config.embedding_dim,
        encoder_dim=config.encoder_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        num_perturbation_groups=num_groups,
        perturbation_group_ids=group_ids,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def save_json(path: Path, values: dict[str, Any]) -> None:
    """Write a dictionary as formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(values, file, indent=2)
