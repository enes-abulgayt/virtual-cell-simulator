"""Synthetic single-cell perturbation data generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PATH, METADATA_PATH, SimulatorConfig, ensure_project_directories


PATHWAY_NAMES = (
    "Cell Cycle",
    "Growth Signaling",
    "Ribosome Biogenesis",
    "Stress Response",
    "DNA Repair",
    "Apoptosis",
    "Innate Immune Signaling",
    "Interferon-like Response",
    "Energy Metabolism",
    "Oxidative Phosphorylation",
    "Protein Folding",
    "Chromatin Regulation",
    "RNA Processing",
    "Vesicle Transport",
    "Cytoskeleton",
    "Lipid Metabolism",
    "Amino Acid Metabolism",
    "Cell Adhesion",
)

PERTURBATION_GROUP_NAMES = (
    "Control",
    "Growth & Proliferation",
    "Stress & Genome Maintenance",
    "Immune & Metabolic Signaling",
)

REQUIRED_DATA_KEYS = {
    "data_version",
    "control_expression",
    "perturbation_ids",
    "target_delta",
    "target_expression",
    "splits",
    "cell_ids",
    "gene_names",
    "perturbation_names",
    "perturbation_signatures",
    "perturbation_group_ids",
    "perturbation_group_names",
    "perturbation_primary_pathway_ids",
    "perturbation_secondary_pathway_ids",
    "pathway_names",
    "pathway_basis",
    "pathway_gene_mask",
    "gene_module_ids",
    "cell_context_scale",
}


def _softplus(values: np.ndarray) -> np.ndarray:
    """Numerically stable softplus transformation."""

    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0)


def _build_pathway_basis(
    rng: np.random.Generator,
    config: SimulatorConfig,
) -> np.ndarray:
    """Create overlapping sparse gene modules that mimic pathways."""

    pathways = np.zeros((config.num_pathways, config.num_genes), dtype=np.float32)
    shared_hub_genes = rng.choice(config.num_genes, size=35, replace=False)

    for pathway_idx in range(config.num_pathways):
        module_genes = rng.choice(
            config.num_genes,
            size=config.genes_per_pathway,
            replace=False,
        )
        hub_genes = rng.choice(shared_hub_genes, size=5, replace=False)
        genes = np.unique(np.concatenate([module_genes, hub_genes]))
        weights = rng.normal(0.0, 0.42, size=len(genes))
        weights += np.sign(weights) * rng.uniform(0.08, 0.20, size=len(genes))
        pathways[pathway_idx, genes] = weights.astype(np.float32)

    return pathways


def _assign_perturbation_groups(config: SimulatorConfig) -> np.ndarray:
    """Assign perturbations to related biological families."""

    group_ids = np.zeros(config.num_perturbations + 1, dtype=np.int64)
    for perturbation_id in range(1, config.num_perturbations + 1):
        group_ids[perturbation_id] = (
            (perturbation_id - 1) % config.num_perturbation_groups
        ) + 1

    # The held-out perturbation belongs to a family observed during training.
    # This creates a meaningful grouped extrapolation challenge.
    group_ids[config.heldout_perturbation_id] = 1
    return group_ids


def _build_perturbation_signatures(
    rng: np.random.Generator,
    pathways: np.ndarray,
    config: SimulatorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create related perturbation signatures using shared group programs."""

    signatures = np.zeros(
        (config.num_perturbations + 1, config.num_genes),
        dtype=np.float32,
    )
    primary_pathway_ids = np.full(
        config.num_perturbations + 1,
        -1,
        dtype=np.int64,
    )
    secondary_pathway_ids = np.full(
        config.num_perturbations + 1,
        -1,
        dtype=np.int64,
    )
    group_ids = _assign_perturbation_groups(config)

    group_pathway_pools = {
        1: np.array([0, 1, 2], dtype=np.int64),
        2: np.array([3, 4, 5], dtype=np.int64),
        3: np.array([6, 7, 8], dtype=np.int64),
    }
    group_prototypes: dict[int, np.ndarray] = {}
    for group_id, pathway_pool in group_pathway_pools.items():
        group_prototypes[group_id] = (
            0.55 * pathways[pathway_pool[0]]
            + 0.38 * pathways[pathway_pool[1]]
            + 0.20 * pathways[pathway_pool[2]]
        )

    group_member_counts = {group_id: 0 for group_id in group_pathway_pools}
    for perturbation_id in range(1, config.num_perturbations + 1):
        group_id = int(group_ids[perturbation_id])
        pathway_pool = group_pathway_pools[group_id]
        member_index = group_member_counts[group_id]
        group_member_counts[group_id] += 1

        primary_id = int(pathway_pool[member_index % len(pathway_pool)])
        secondary_id = int(pathway_pool[(member_index + 1) % len(pathway_pool)])
        primary_pathway_ids[perturbation_id] = primary_id
        secondary_pathway_ids[perturbation_id] = secondary_id

        direct_genes = rng.choice(
            config.num_genes,
            size=config.direct_targets_per_perturbation,
            replace=False,
        )
        direct_effect = np.zeros(config.num_genes, dtype=np.float32)
        group_direction = 1.0 if group_id in (1, 3) else -1.0
        direct_effect[direct_genes] = group_direction * rng.uniform(
            0.35,
            0.75,
            size=config.direct_targets_per_perturbation,
        )

        individual_program = (
            0.28 * pathways[primary_id] + 0.16 * pathways[secondary_id]
        )
        signature = group_prototypes[group_id] + individual_program + direct_effect
        signature[np.abs(signature) < 0.07] = 0.0
        signatures[perturbation_id] = signature.astype(np.float32)

    return signatures, primary_pathway_ids, secondary_pathway_ids, group_ids


def _cosine_similarity_matrix(signatures: np.ndarray) -> np.ndarray:
    """Calculate perturbation signature similarity."""

    norms = np.linalg.norm(signatures, axis=1, keepdims=True)
    normalized = signatures / np.maximum(norms, 1e-8)
    return (normalized @ normalized.T).astype(np.float32)


def _assign_conditions_and_splits(
    rng: np.random.Generator,
    config: SimulatorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign balanced conditions and cell-level splits."""

    condition_ids = np.arange(config.num_perturbations + 1, dtype=np.int64)
    perturbation_ids = np.resize(condition_ids, config.num_cells)
    rng.shuffle(perturbation_ids)

    splits = np.empty(config.num_cells, dtype="<U10")
    for perturbation_id in condition_ids:
        indices = np.flatnonzero(perturbation_ids == perturbation_id)
        rng.shuffle(indices)

        if perturbation_id == config.heldout_perturbation_id:
            splits[indices] = "test"
            continue

        train_end = int(len(indices) * config.train_fraction)
        validation_end = train_end + int(len(indices) * config.validation_fraction)
        splits[indices[:train_end]] = "train"
        splits[indices[train_end:validation_end]] = "validation"
        splits[indices[validation_end:]] = "test"

    return perturbation_ids, splits


def is_synthetic_data_current(
    path: Path = DATA_PATH,
    config: SimulatorConfig | None = None,
) -> bool:
    """Return whether an existing dataset matches the current schema."""

    config = config or SimulatorConfig()
    if not path.exists():
        return False

    try:
        with np.load(path, allow_pickle=False) as archive:
            if not REQUIRED_DATA_KEYS.issubset(archive.files):
                return False
            version = int(np.asarray(archive["data_version"]).item())
            return (
                version == config.synthetic_data_version
                and archive["control_expression"].shape
                == (config.num_cells, config.num_genes)
                and len(archive["perturbation_names"])
                == config.num_perturbations + 1
            )
    except (OSError, ValueError, KeyError):
        return False


def generate_synthetic_data(
    config: SimulatorConfig | None = None,
    output_path: Path = DATA_PATH,
    metadata_path: Path = METADATA_PATH,
    force: bool = False,
) -> Path:
    """Generate and save a structured synthetic perturbation dataset."""

    config = config or SimulatorConfig()
    ensure_project_directories()

    if (
        output_path.exists()
        and metadata_path.exists()
        and not force
        and is_synthetic_data_current(output_path, config)
    ):
        return output_path

    rng = np.random.default_rng(config.seed)
    gene_names = np.array(
        [f"GENE_{index:04d}" for index in range(config.num_genes)],
        dtype="<U16",
    )
    perturbation_names = np.array(
        ["control"]
        + [f"PERT_GENE_{index:02d}" for index in range(1, config.num_perturbations + 1)],
        dtype="<U20",
    )
    pathway_names = np.asarray(PATHWAY_NAMES[: config.num_pathways], dtype="<U32")
    group_names = np.asarray(
        PERTURBATION_GROUP_NAMES[: config.num_perturbation_groups + 1],
        dtype="<U40",
    )

    latent_loadings = rng.normal(
        0.0,
        0.18,
        size=(config.latent_dim, config.num_genes),
    ).astype(np.float32)
    latent_states = rng.normal(
        0.0,
        1.0,
        size=(config.num_cells, config.latent_dim),
    ).astype(np.float32)
    gene_bias = rng.normal(0.75, 0.35, size=config.num_genes).astype(np.float32)

    biological_signal = latent_states @ latent_loadings + gene_bias
    control_expression = _softplus(biological_signal)
    control_expression += rng.normal(
        0.0,
        0.045,
        size=control_expression.shape,
    ).astype(np.float32)
    control_expression = np.clip(control_expression, 0.0, None).astype(np.float32)

    pathways = _build_pathway_basis(rng, config)
    (
        signatures,
        primary_pathway_ids,
        secondary_pathway_ids,
        perturbation_group_ids,
    ) = _build_perturbation_signatures(rng, pathways, config)
    perturbation_ids, splits = _assign_conditions_and_splits(rng, config)

    context_signal = np.tanh(
        0.65 * latent_states[:, 0] - 0.35 * latent_states[:, 1]
    ).astype(np.float32)
    cell_context_scale = (1.0 + 0.22 * context_signal).astype(np.float32)
    selected_signatures = signatures[perturbation_ids]
    expression_sensitivity = 1.0 + 0.08 * np.tanh(control_expression - 1.0)
    target_delta = (
        selected_signatures
        * cell_context_scale[:, None]
        * expression_sensitivity
    )

    perturbation_noise = rng.normal(
        0.0,
        0.035,
        size=target_delta.shape,
    ).astype(np.float32)
    control_rows = perturbation_ids == 0
    perturbation_noise[control_rows] *= 0.35
    target_delta = target_delta + perturbation_noise

    target_expression = np.clip(
        control_expression + target_delta,
        0.0,
        None,
    ).astype(np.float32)
    target_delta = (target_expression - control_expression).astype(np.float32)

    pathway_gene_mask = (pathways != 0.0).astype(np.uint8)
    strongest_pathway = np.argmax(np.abs(pathways), axis=0)
    has_module = np.max(np.abs(pathways), axis=0) > 0
    gene_module_ids = np.where(has_module, strongest_pathway, -1).astype(np.int64)
    similarity_matrix = _cosine_similarity_matrix(signatures)

    seen_in_training = perturbation_ids != config.heldout_perturbation_id
    cell_ids = np.array(
        [f"CELL_{index:05d}" for index in range(config.num_cells)],
        dtype="<U16",
    )

    np.savez(
        output_path,
        data_version=np.array(config.synthetic_data_version, dtype=np.int64),
        control_expression=control_expression,
        perturbation_ids=perturbation_ids,
        target_delta=target_delta,
        target_expression=target_expression,
        splits=splits,
        cell_ids=cell_ids,
        gene_names=gene_names,
        perturbation_names=perturbation_names,
        perturbation_signatures=signatures,
        perturbation_group_ids=perturbation_group_ids,
        perturbation_group_names=group_names,
        perturbation_primary_pathway_ids=primary_pathway_ids,
        perturbation_secondary_pathway_ids=secondary_pathway_ids,
        perturbation_similarity=similarity_matrix,
        pathway_names=pathway_names,
        pathway_basis=pathways,
        pathway_gene_mask=pathway_gene_mask,
        gene_module_ids=gene_module_ids,
        cell_context_scale=cell_context_scale,
        seen_in_training=seen_in_training,
    )

    primary_names = np.where(
        primary_pathway_ids[perturbation_ids] >= 0,
        pathway_names[np.maximum(primary_pathway_ids[perturbation_ids], 0)],
        "None",
    )
    secondary_names = np.where(
        secondary_pathway_ids[perturbation_ids] >= 0,
        pathway_names[np.maximum(secondary_pathway_ids[perturbation_ids], 0)],
        "None",
    )
    metadata = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "perturbation_id": perturbation_ids,
            "perturbation": perturbation_names[perturbation_ids],
            "perturbation_group": group_names[
                perturbation_group_ids[perturbation_ids]
            ],
            "primary_pathway": primary_names,
            "secondary_pathway": secondary_names,
            "split": splits,
            "seen_in_training": seen_in_training,
        }
    )
    metadata.to_csv(metadata_path, index=False)
    return output_path


def load_synthetic_data(
    path: Path = DATA_PATH,
    config: SimulatorConfig | None = None,
    generate_if_missing: bool = True,
) -> dict[str, np.ndarray]:
    """Load current synthetic data, regenerating stale schemas when allowed."""

    config = config or SimulatorConfig()
    if not is_synthetic_data_current(path, config):
        if not generate_if_missing:
            raise RuntimeError(
                "Synthetic data is missing or outdated. Run "
                "`python train.py --regenerate-data`."
            )
        generate_synthetic_data(
            config=config,
            output_path=path,
            force=True,
        )

    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def expected_delta_for_cell(
    data: dict[str, np.ndarray],
    cell_index: int,
    perturbation_id: int,
) -> np.ndarray:
    """Return the deterministic synthetic response before measurement noise."""

    control = data["control_expression"][cell_index].astype(np.float32)
    signature = data["perturbation_signatures"][perturbation_id].astype(np.float32)
    context_scale = float(data["cell_context_scale"][cell_index])
    sensitivity = 1.0 + 0.08 * np.tanh(control - 1.0)
    raw_delta = signature * context_scale * sensitivity
    expected_expression = np.clip(control + raw_delta, 0.0, None)
    return (expected_expression - control).astype(np.float32)
