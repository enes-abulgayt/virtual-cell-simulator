"""Evaluate the trained Mini Virtual Cell Simulator.

Usage:
    python evaluate.py
"""

from __future__ import annotations

import torch

from src.config import (
    CHECKPOINT_PATH,
    EVALUATION_PATH,
    RESULTS_DIR,
    SimulatorConfig,
    ensure_project_directories,
)
from src.dataset import create_data_loaders
from src.evaluate_utils import calculate_evaluation_metrics, collect_predictions
from src.synthetic_data import load_synthetic_data
from src.train_utils import (
    get_default_device,
    load_model_from_checkpoint,
    save_json,
)
from src.visualization import (
    plot_baseline_comparison,
    plot_per_perturbation_mse,
    plot_predicted_vs_true_delta,
    plot_top_changed_genes,
)


def main() -> None:
    """Load the best checkpoint, evaluate it, and save metrics and plots."""

    ensure_project_directories()
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {CHECKPOINT_PATH}. Run `python train.py` first."
        )

    device = get_default_device()
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    model, checkpoint = load_model_from_checkpoint(CHECKPOINT_PATH, device)
    simulator_config = SimulatorConfig.from_dict(checkpoint["config"])
    data = load_synthetic_data(config=simulator_config)
    checkpoint_data_version = int(
        checkpoint["config"].get("synthetic_data_version", 1)
    )
    data_version = int(data["data_version"].item())
    if checkpoint_data_version != data_version:
        raise RuntimeError(
            "Checkpoint and dataset schema versions differ. Run "
            "`python train.py --regenerate-data` before evaluation."
        )
    loaders, _ = create_data_loaders(data, simulator_config)
    predictions = collect_predictions(model, loaders["test"], device)
    metrics = calculate_evaluation_metrics(
        predictions,
        heldout_perturbation_id=simulator_config.heldout_perturbation_id,
        perturbation_names=data["perturbation_names"],
        top_k=20,
    )
    metrics["checkpoint_epoch"] = int(checkpoint["epoch"])
    metrics["test_sample_count"] = int(len(predictions["perturbation_ids"]))
    metrics["heldout_perturbation"] = str(
        data["perturbation_names"][simulator_config.heldout_perturbation_id]
    )
    save_json(EVALUATION_PATH, metrics)

    plot_predicted_vs_true_delta(
        predictions["target_delta"],
        predictions["predicted_delta"],
        RESULTS_DIR / "predicted_vs_true_delta.png",
        seed=simulator_config.seed,
    )
    plot_top_changed_genes(
        predictions["target_delta"],
        predictions["predicted_delta"],
        data["gene_names"],
        RESULTS_DIR / "top_changed_genes.png",
    )
    plot_baseline_comparison(
        metrics["model"]["delta"],
        metrics["no_change_baseline"]["delta"],
        RESULTS_DIR / "baseline_vs_model.png",
    )
    plot_per_perturbation_mse(
        metrics["per_perturbation"],
        RESULTS_DIR / "per_perturbation_mse.png",
    )

    model_mse = metrics["model"]["delta"]["mse"]
    baseline_mse = metrics["no_change_baseline"]["delta"]["mse"]
    print(f"Model delta MSE:       {model_mse:.6f}")
    print(f"No-change delta MSE:   {baseline_mse:.6f}")
    print(
        "MSE improvement:       "
        f"{metrics['model_mse_improvement_percent']:.2f}%"
    )
    print(f"Delta correlation:     {metrics['delta_correlation']:.4f}")
    print(
        "Top-20 gene overlap:   "
        f"{metrics['top_k_changed_gene_overlap']:.4f}"
    )
    print(f"Evaluation saved to:   {EVALUATION_PATH}")


if __name__ == "__main__":
    main()
