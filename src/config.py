"""Project-wide configuration and path definitions."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
MATPLOTLIB_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"

# Keep Matplotlib's font cache in a project-local writable directory. This is
# especially useful in containers and restricted execution environments.
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))

DATA_PATH = DATA_DIR / "synthetic_expression.npz"
METADATA_PATH = DATA_DIR / "metadata.csv"
CHECKPOINT_PATH = CHECKPOINT_DIR / "model.pt"
METRICS_PATH = RESULTS_DIR / "metrics.json"
EVALUATION_PATH = RESULTS_DIR / "evaluation.json"


@dataclass(frozen=True)
class SimulatorConfig:
    """Hyperparameters for data generation, modeling, and training."""

    synthetic_data_version: int = 2
    seed: int = 42
    num_cells: int = 5_000
    num_genes: int = 1_000
    num_perturbations: int = 10
    num_perturbation_groups: int = 3
    latent_dim: int = 24
    num_pathways: int = 18
    genes_per_pathway: int = 45
    direct_targets_per_perturbation: int = 25
    heldout_perturbation_id: int = 10

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    embedding_dim: int = 32
    encoder_dim: int = 128
    hidden_dim: int = 192
    dropout: float = 0.10

    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 12
    early_stopping_patience: int = 4
    num_workers: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a JSON-serializable dictionary."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SimulatorConfig":
        """Create a configuration while ignoring unknown checkpoint fields."""

        valid_fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in valid_fields})


def ensure_project_directories() -> None:
    """Create all runtime output directories if they do not exist."""

    for path in (DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, MATPLOTLIB_CONFIG_DIR):
        path.mkdir(parents=True, exist_ok=True)
