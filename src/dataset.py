"""PyTorch datasets and data loaders for perturbation prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import SimulatorConfig


@dataclass
class NormalizationStats:
    """Per-gene normalization statistics for control expression."""

    mean: np.ndarray
    std: np.ndarray

    def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert statistics to float tensors."""

        return (
            torch.from_numpy(self.mean.astype(np.float32)),
            torch.from_numpy(self.std.astype(np.float32)),
        )


class PerturbationDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset returning paired control and perturbed cell states."""

    def __init__(
        self,
        data: dict[str, np.ndarray],
        split: str,
        normalization: NormalizationStats,
    ) -> None:
        self.indices = np.flatnonzero(data["splits"] == split)
        if len(self.indices) == 0:
            raise ValueError(f"No samples found for split '{split}'.")

        self.control_expression = torch.from_numpy(
            data["control_expression"][self.indices].astype(np.float32)
        )
        self.perturbation_ids = torch.from_numpy(
            data["perturbation_ids"][self.indices].astype(np.int64)
        )
        self.target_delta = torch.from_numpy(
            data["target_delta"][self.indices].astype(np.float32)
        )
        self.target_expression = torch.from_numpy(
            data["target_expression"][self.indices].astype(np.float32)
        )
        self.normalization_mean, self.normalization_std = normalization.to_tensors()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        control = self.control_expression[index]
        normalized_control = (
            control - self.normalization_mean
        ) / self.normalization_std
        return {
            "control_expression": control,
            "normalized_control_expression": normalized_control,
            "perturbation_id": self.perturbation_ids[index],
            "target_delta": self.target_delta[index],
            "target_perturbed_expression": self.target_expression[index],
        }


def compute_normalization_stats(
    data: dict[str, np.ndarray],
) -> NormalizationStats:
    """Compute normalization statistics using only training controls."""

    train_mask = data["splits"] == "train"
    training_controls = data["control_expression"][train_mask]
    mean = training_controls.mean(axis=0).astype(np.float32)
    std = training_controls.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-5)
    return NormalizationStats(mean=mean, std=std)


def create_data_loaders(
    data: dict[str, np.ndarray],
    config: SimulatorConfig,
) -> tuple[
    dict[str, DataLoader[dict[str, torch.Tensor]]],
    NormalizationStats,
]:
    """Create train, validation, and test data loaders."""

    normalization = compute_normalization_stats(data)
    datasets = {
        split: PerturbationDataset(data, split, normalization)
        for split in ("train", "validation", "test")
    }

    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            generator=generator,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
    }
    return loaders, normalization
