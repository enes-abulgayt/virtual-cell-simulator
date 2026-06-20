"""Matplotlib visualizations for training, evaluation, and the app."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curve(
    train_losses: list[float],
    validation_losses: list[float],
    output_path: Path,
) -> None:
    """Save train and validation loss curves."""

    figure, axis = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(train_losses) + 1)
    axis.plot(epochs, train_losses, marker="o", label="Training MSE")
    axis.plot(epochs, validation_losses, marker="o", label="Validation MSE")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss")
    axis.set_title("Virtual Cell Model Training")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_predicted_vs_true_delta(
    true_delta: np.ndarray,
    predicted_delta: np.ndarray,
    output_path: Path,
    seed: int = 42,
    max_points: int = 25_000,
) -> None:
    """Save a sampled predicted-vs-true delta scatter plot."""

    true_flat = true_delta.ravel()
    predicted_flat = predicted_delta.ravel()
    rng = np.random.default_rng(seed)
    if len(true_flat) > max_points:
        indices = rng.choice(len(true_flat), size=max_points, replace=False)
        true_flat = true_flat[indices]
        predicted_flat = predicted_flat[indices]

    limits = [
        min(float(true_flat.min()), float(predicted_flat.min())),
        max(float(true_flat.max()), float(predicted_flat.max())),
    ]
    figure, axis = plt.subplots(figsize=(6.5, 6))
    axis.scatter(true_flat, predicted_flat, s=7, alpha=0.18, color="#2563eb")
    axis.plot(limits, limits, linestyle="--", color="#dc2626", linewidth=1.5)
    axis.set_xlabel("True delta")
    axis.set_ylabel("Predicted delta")
    axis.set_title("Predicted vs. True Expression Change")
    axis.grid(alpha=0.20)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_top_changed_genes(
    true_delta: np.ndarray,
    predicted_delta: np.ndarray,
    gene_names: np.ndarray,
    output_path: Path,
    top_k: int = 15,
) -> None:
    """Compare true and predicted mean absolute changes for top genes."""

    true_change = np.mean(np.abs(true_delta), axis=0)
    predicted_change = np.mean(np.abs(predicted_delta), axis=0)
    top_indices = np.argsort(true_change)[-top_k:]
    positions = np.arange(top_k)

    figure, axis = plt.subplots(figsize=(9, 6))
    width = 0.38
    axis.barh(
        positions - width / 2,
        true_change[top_indices],
        height=width,
        label="True |delta|",
        color="#0f766e",
    )
    axis.barh(
        positions + width / 2,
        predicted_change[top_indices],
        height=width,
        label="Predicted |delta|",
        color="#7c3aed",
    )
    axis.set_yticks(positions, gene_names[top_indices])
    axis.set_xlabel("Mean absolute expression change")
    axis.set_title("Top Changed Genes")
    axis.legend()
    axis.grid(axis="x", alpha=0.20)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_baseline_comparison(
    model_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    output_path: Path,
) -> None:
    """Save a grouped model-versus-baseline error chart."""

    metric_names = ["MSE", "MAE"]
    model_values = [model_metrics["mse"], model_metrics["mae"]]
    baseline_values = [baseline_metrics["mse"], baseline_metrics["mae"]]
    positions = np.arange(len(metric_names))
    width = 0.36

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(
        positions - width / 2,
        baseline_values,
        width,
        label="No-change baseline",
        color="#94a3b8",
    )
    axis.bar(
        positions + width / 2,
        model_values,
        width,
        label="Virtual cell model",
        color="#2563eb",
    )
    axis.set_xticks(positions, metric_names)
    axis.set_ylabel("Error (lower is better)")
    axis.set_title("Baseline vs. Model Error")
    axis.legend()
    axis.grid(axis="y", alpha=0.20)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_per_perturbation_mse(
    per_perturbation_metrics: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Save per-perturbation model and baseline MSE values."""

    names = list(per_perturbation_metrics)
    model_values = [
        per_perturbation_metrics[name]["model_mse"] for name in names
    ]
    baseline_values = [
        per_perturbation_metrics[name]["baseline_mse"] for name in names
    ]
    positions = np.arange(len(names))
    width = 0.38

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(
        positions - width / 2,
        baseline_values,
        width,
        label="No-change baseline",
        color="#94a3b8",
    )
    axis.bar(
        positions + width / 2,
        model_values,
        width,
        label="Virtual cell model",
        color="#2563eb",
    )
    axis.set_xticks(positions, names, rotation=45, ha="right")
    axis.set_ylabel("Delta MSE")
    axis.set_title("Error by Perturbation")
    axis.legend()
    axis.grid(axis="y", alpha=0.20)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def make_cell_state_figure(
    control_expression: np.ndarray,
    predicted_expression: np.ndarray,
    predicted_delta: np.ndarray,
    gene_names: np.ndarray,
    top_k: int = 12,
) -> plt.Figure:
    """Create a before/after dumbbell chart for the most changed genes."""

    top_indices = np.argsort(np.abs(predicted_delta))[-top_k:]
    positions = np.arange(top_k)
    before = control_expression[top_indices]
    after = predicted_expression[top_indices]
    colors = np.where(predicted_delta[top_indices] >= 0, "#dc2626", "#2563eb")

    figure, axis = plt.subplots(figsize=(9, 6))
    for position, start, end, color in zip(
        positions,
        before,
        after,
        colors,
        strict=True,
    ):
        axis.plot([start, end], [position, position], color=color, linewidth=3, alpha=0.7)
    axis.scatter(
        before,
        positions,
        s=70,
        label="Before perturbation",
        color="#94a3b8",
        edgecolor="white",
        zorder=3,
    )
    axis.scatter(
        after,
        positions,
        s=70,
        label="Predicted after perturbation",
        color=colors,
        edgecolor="white",
        zorder=3,
    )
    axis.set_yticks(positions, gene_names[top_indices])
    axis.set_xlabel("Synthetic expression")
    axis.set_title("Control vs. Predicted Perturbed Cell State")
    axis.legend()
    axis.grid(axis="x", alpha=0.20)
    figure.tight_layout()
    return figure


def make_pathway_activity_figure(
    pathway_summary: list[dict[str, float | str]],
    top_k: int = 10,
) -> plt.Figure:
    """Create a signed pathway response chart."""

    selected = sorted(
        pathway_summary,
        key=lambda row: float(row["mean_absolute_delta"]),
    )[-top_k:]
    names = [str(row["pathway"]) for row in selected]
    signed_values = np.array(
        [float(row["mean_delta"]) for row in selected],
        dtype=np.float32,
    )
    colors = np.where(signed_values >= 0, "#dc2626", "#2563eb")
    positions = np.arange(len(selected))

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(positions, signed_values, color=colors)
    axis.set_yticks(positions, names)
    axis.axvline(0.0, color="#111827", linewidth=0.8)
    axis.set_xlabel("Mean predicted delta across pathway genes")
    axis.set_title("Predicted Pathway-Level Response")
    axis.grid(axis="x", alpha=0.20)
    figure.tight_layout()
    return figure


def make_model_baseline_figure(
    model_mse: float,
    baseline_mse: float,
) -> plt.Figure:
    """Create a compact selected-simulation error comparison."""

    figure, axis = plt.subplots(figsize=(6.5, 4))
    bars = axis.bar(
        ["Virtual cell model", "No-change baseline"],
        [model_mse, baseline_mse],
        color=["#2563eb", "#94a3b8"],
    )
    axis.set_ylabel("MSE against synthetic expected delta")
    axis.set_title("Selected Simulation: Model vs. Baseline")
    axis.grid(axis="y", alpha=0.20)
    for bar, value in zip(bars, [model_mse, baseline_mse], strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    return figure


def make_delta_figure(
    predicted_delta: np.ndarray,
    gene_names: np.ndarray,
    top_k: int = 16,
) -> plt.Figure:
    """Create a signed delta chart for the strongest predictions."""

    top_indices = np.argsort(np.abs(predicted_delta))[-top_k:]
    values = predicted_delta[top_indices]
    colors = np.where(values >= 0, "#dc2626", "#2563eb")
    positions = np.arange(top_k)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(positions, values, color=colors)
    axis.set_yticks(positions, gene_names[top_indices])
    axis.axvline(0.0, color="#111827", linewidth=0.8)
    axis.set_xlabel("Predicted expression delta")
    axis.set_title("Strongest Predicted Gene Responses")
    axis.grid(axis="x", alpha=0.20)
    figure.tight_layout()
    return figure
