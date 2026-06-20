"""Train the Mini Virtual Cell Simulator.

Usage:
    python train.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import torch
from torch import nn

from src.config import (
    CHECKPOINT_PATH,
    DATA_PATH,
    METRICS_PATH,
    RESULTS_DIR,
    SimulatorConfig,
    ensure_project_directories,
)
from src.dataset import create_data_loaders
from src.model import VirtualCellModel
from src.synthetic_data import (
    generate_synthetic_data,
    is_synthetic_data_current,
    load_synthetic_data,
)
from src.train_utils import (
    get_default_device,
    run_epoch,
    save_checkpoint,
    save_json,
    set_reproducible_seed,
)
from src.visualization import plot_loss_curve


def parse_args() -> argparse.Namespace:
    """Parse optional training overrides."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the default number of training epochs.",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Regenerate the synthetic dataset even if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate data, train the model, and save all training artifacts."""

    args = parse_args()
    config = SimulatorConfig()
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1.")
        config = replace(config, epochs=args.epochs)

    ensure_project_directories()
    set_reproducible_seed(config.seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))

    data_needs_generation = not is_synthetic_data_current(DATA_PATH, config)
    if args.regenerate_data or data_needs_generation:
        print(
            f"Generating {config.num_cells:,} cells x "
            f"{config.num_genes:,} genes..."
        )
        generate_synthetic_data(config=config, force=args.regenerate_data)

    data = load_synthetic_data(config=config)
    loaders, normalization = create_data_loaders(data, config)
    device = get_default_device()

    model = VirtualCellModel(
        num_genes=config.num_genes,
        num_perturbations=config.num_perturbations,
        embedding_dim=config.embedding_dim,
        encoder_dim=config.encoder_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        num_perturbation_groups=config.num_perturbation_groups,
        perturbation_group_ids=data["perturbation_group_ids"].tolist(),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.MSELoss()

    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_time = time.perf_counter()

    print(f"Training on {device} with {len(loaders['train'].dataset):,} cells.")
    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(
            model,
            loaders["train"],
            loss_function,
            device,
            optimizer=optimizer,
            description=f"Train {epoch}/{config.epochs}",
        )
        validation_loss = run_epoch(
            model,
            loaders["validation"],
            loss_function,
            device,
            optimizer=None,
            description=f"Valid {epoch}/{config.epochs}",
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        print(
            f"Epoch {epoch:02d} | train MSE: {train_loss:.6f} | "
            f"validation MSE: {validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                CHECKPOINT_PATH,
                model,
                config,
                normalization,
                epoch,
                validation_loss,
                data["gene_names"],
                data["perturbation_names"],
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    elapsed_seconds = time.perf_counter() - start_time
    plot_loss_curve(
        train_losses,
        validation_losses,
        RESULTS_DIR / "loss_curve.png",
    )

    split_counts = {
        split: int((data["splits"] == split).sum())
        for split in ("train", "validation", "test")
    }
    metrics = {
        "seed": config.seed,
        "device": str(device),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "split_counts": split_counts,
        "heldout_perturbation": str(
            data["perturbation_names"][config.heldout_perturbation_id]
        ),
        "epochs_completed": len(train_losses),
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_loss,
        "training_seconds": elapsed_seconds,
        "train_mse": train_losses,
        "validation_mse": validation_losses,
    }
    save_json(METRICS_PATH, metrics)

    print(f"Best checkpoint: {CHECKPOINT_PATH}")
    print(f"Training metrics: {METRICS_PATH}")
    print(f"Loss curve: {RESULTS_DIR / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
