"""Evaluation helpers and baseline comparisons."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import VirtualCellModel


def _metric_summary(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    error = predicted - target
    return {
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
    }


def pearson_correlation(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    """Calculate a stable flattened Pearson correlation."""

    predicted_flat = predicted.astype(np.float64, copy=False).ravel()
    target_flat = target.astype(np.float64, copy=False).ravel()
    if predicted_flat.std() < 1e-12 or target_flat.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(predicted_flat, target_flat)[0, 1])


def top_k_changed_gene_overlap(
    predicted: np.ndarray,
    target: np.ndarray,
    top_k: int = 20,
) -> float:
    """Return mean overlap between predicted and true top changed genes."""

    if predicted.ndim == 1:
        predicted = predicted[None, :]
        target = target[None, :]

    top_k = min(top_k, predicted.shape[1])
    predicted_top = np.argpartition(
        np.abs(predicted),
        -top_k,
        axis=1,
    )[:, -top_k:]
    target_top = np.argpartition(
        np.abs(target),
        -top_k,
        axis=1,
    )[:, -top_k:]

    overlaps = [
        len(set(predicted_indices) & set(target_indices)) / top_k
        for predicted_indices, target_indices in zip(
            predicted_top,
            target_top,
            strict=True,
        )
    ]
    return float(np.mean(overlaps))


@torch.no_grad()
def collect_predictions(
    model: VirtualCellModel,
    data_loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Collect model predictions and targets from a data loader."""

    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        device_batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(
            device_batch["normalized_control_expression"],
            device_batch["perturbation_id"],
            device_batch["control_expression"],
        )

        collected["control_expression"].append(
            batch["control_expression"].numpy()
        )
        collected["perturbation_ids"].append(batch["perturbation_id"].numpy())
        collected["target_delta"].append(batch["target_delta"].numpy())
        collected["target_expression"].append(
            batch["target_perturbed_expression"].numpy()
        )
        collected["predicted_delta"].append(
            outputs["predicted_delta"].cpu().numpy()
        )
        collected["predicted_expression"].append(
            outputs["predicted_expression"].cpu().numpy()
        )

    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def calculate_evaluation_metrics(
    predictions: dict[str, np.ndarray],
    heldout_perturbation_id: int,
    perturbation_names: np.ndarray | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    """Calculate aggregate and per-perturbation evaluation metrics."""

    baseline_delta = np.zeros_like(predictions["target_delta"])
    baseline_expression = predictions["control_expression"]
    model_delta_metrics = _metric_summary(
        predictions["predicted_delta"],
        predictions["target_delta"],
    )
    model_delta_metrics["correlation"] = pearson_correlation(
        predictions["predicted_delta"],
        predictions["target_delta"],
    )
    model_delta_metrics["top_k_changed_gene_overlap"] = top_k_changed_gene_overlap(
        predictions["predicted_delta"],
        predictions["target_delta"],
        top_k=top_k,
    )
    model_delta_metrics["top_k"] = top_k

    metrics: dict[str, Any] = {
        "model": {
            "delta": model_delta_metrics,
            "expression": _metric_summary(
                predictions["predicted_expression"],
                predictions["target_expression"],
            ),
        },
        "no_change_baseline": {
            "delta": _metric_summary(
                baseline_delta,
                predictions["target_delta"],
            ),
            "expression": _metric_summary(
                baseline_expression,
                predictions["target_expression"],
            ),
        },
        "per_perturbation": {},
    }

    perturbation_ids = predictions["perturbation_ids"]
    for group_name, mask in {
        "seen_perturbations": perturbation_ids != heldout_perturbation_id,
        "unseen_perturbation": perturbation_ids == heldout_perturbation_id,
    }.items():
        if not np.any(mask):
            continue
        metrics[group_name] = {
            "sample_count": int(mask.sum()),
            "model_delta": {
                **_metric_summary(
                    predictions["predicted_delta"][mask],
                    predictions["target_delta"][mask],
                ),
                "correlation": pearson_correlation(
                    predictions["predicted_delta"][mask],
                    predictions["target_delta"][mask],
                ),
                "top_k_changed_gene_overlap": top_k_changed_gene_overlap(
                    predictions["predicted_delta"][mask],
                    predictions["target_delta"][mask],
                    top_k=top_k,
                ),
            },
            "baseline_delta": _metric_summary(
                baseline_delta[mask],
                predictions["target_delta"][mask],
            ),
        }

    for perturbation_id in np.unique(perturbation_ids):
        mask = perturbation_ids == perturbation_id
        perturbation_name = (
            str(perturbation_names[perturbation_id])
            if perturbation_names is not None
            else f"perturbation_{perturbation_id}"
        )
        model_values = predictions["predicted_delta"][mask]
        target_values = predictions["target_delta"][mask]
        metrics["per_perturbation"][perturbation_name] = {
            "perturbation_id": int(perturbation_id),
            "sample_count": int(mask.sum()),
            "seen_in_training": bool(perturbation_id != heldout_perturbation_id),
            "model_mse": _metric_summary(model_values, target_values)["mse"],
            "baseline_mse": _metric_summary(
                np.zeros_like(target_values),
                target_values,
            )["mse"],
            "correlation": pearson_correlation(model_values, target_values),
            "top_k_changed_gene_overlap": top_k_changed_gene_overlap(
                model_values,
                target_values,
                top_k=top_k,
            ),
        }

    baseline_mse = metrics["no_change_baseline"]["delta"]["mse"]
    model_mse = metrics["model"]["delta"]["mse"]
    metrics["model_mse_improvement_percent"] = float(
        100.0 * (baseline_mse - model_mse) / max(baseline_mse, 1e-12)
    )
    metrics["delta_correlation"] = model_delta_metrics["correlation"]
    metrics["top_k_changed_gene_overlap"] = model_delta_metrics[
        "top_k_changed_gene_overlap"
    ]
    return metrics
