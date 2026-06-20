"""Interactive Streamlit application for virtual cell simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.config import CHECKPOINT_PATH, DATA_PATH
from src.evaluate_utils import pearson_correlation, top_k_changed_gene_overlap
from src.synthetic_data import (
    expected_delta_for_cell,
    is_synthetic_data_current,
    load_synthetic_data,
)
from src.train_utils import load_model_from_checkpoint
from src.visualization import (
    make_cell_state_figure,
    make_delta_figure,
    make_model_baseline_figure,
    make_pathway_activity_figure,
)


st.set_page_config(
    page_title="Mini Virtual Cell Simulator",
    page_icon="🧬",
    layout="wide",
)


@st.cache_resource
def load_trained_model() -> tuple[torch.nn.Module, dict, torch.device]:
    """Load and cache the trained PyTorch model."""

    device = torch.device("cpu")
    model, checkpoint = load_model_from_checkpoint(CHECKPOINT_PATH, device)
    return model, checkpoint, device


@st.cache_data
def load_app_data() -> dict[str, np.ndarray]:
    """Load and cache arrays used by the interactive app."""

    return load_synthetic_data(DATA_PATH, generate_if_missing=False)


def initialize_random_cell(num_cells: int) -> None:
    """Initialize the selected cell in Streamlit session state."""

    if "selected_cell_index" not in st.session_state:
        st.session_state.selected_cell_index = int(
            np.random.default_rng().integers(0, num_cells)
        )


def pathway_summary(
    predicted_delta: np.ndarray,
    data: dict[str, np.ndarray],
    perturbation_id: int,
) -> list[dict[str, float | int | str]]:
    """Aggregate a gene-level prediction into pathway response scores."""

    primary_id = int(data["perturbation_primary_pathway_ids"][perturbation_id])
    secondary_id = int(data["perturbation_secondary_pathway_ids"][perturbation_id])
    rows: list[dict[str, float | int | str]] = []

    for pathway_id, pathway_name in enumerate(data["pathway_names"]):
        gene_mask = data["pathway_gene_mask"][pathway_id].astype(bool)
        pathway_delta = predicted_delta[gene_mask]
        relationship = "Other module"
        if pathway_id == primary_id:
            relationship = "Primary target"
        elif pathway_id == secondary_id:
            relationship = "Secondary target"

        rows.append(
            {
                "pathway": str(pathway_name),
                "relationship": relationship,
                "genes": int(gene_mask.sum()),
                "mean_delta": float(pathway_delta.mean()),
                "mean_absolute_delta": float(np.abs(pathway_delta).mean()),
            }
        )

    return rows


def simulation_results_table(
    gene_names: np.ndarray,
    control_expression: np.ndarray,
    predicted_expression: np.ndarray,
    predicted_delta: np.ndarray,
    expected_delta: np.ndarray,
    data: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build a downloadable gene-level result table."""

    module_ids = data["gene_module_ids"]
    module_names = np.full(len(gene_names), "Unassigned", dtype="<U40")
    assigned = module_ids >= 0
    module_names[assigned] = data["pathway_names"][module_ids[assigned]]

    table = pd.DataFrame(
        {
            "gene": gene_names,
            "pathway_module": module_names,
            "control_expression": control_expression,
            "predicted_delta": predicted_delta,
            "predicted_expression": predicted_expression,
            "synthetic_expected_delta": expected_delta,
            "absolute_delta_error": np.abs(predicted_delta - expected_delta),
        }
    )
    return table.sort_values("predicted_delta", key=np.abs, ascending=False)


def main() -> None:
    """Render the simulator interface and run model inference."""

    st.title("🧬 Mini Virtual Cell Simulator")
    st.caption(
        "A CPU-friendly PyTorch model for exploring synthetic single-cell "
        "responses to grouped genetic perturbations."
    )

    if not CHECKPOINT_PATH.exists():
        st.error(
            "No trained model was found. Run `python train.py` from the project "
            "directory first, then refresh this page."
        )
        st.stop()
    if not is_synthetic_data_current(DATA_PATH):
        st.error(
            "The synthetic dataset is missing or uses an older schema. Run "
            "`python train.py --regenerate-data`, then refresh this page."
        )
        st.stop()

    model, checkpoint, device = load_trained_model()
    data = load_app_data()
    checkpoint_data_version = int(
        checkpoint["config"].get("synthetic_data_version", 1)
    )
    data_version = int(np.asarray(data["data_version"]).item())
    if checkpoint_data_version != data_version:
        st.error(
            "The model checkpoint and synthetic dataset use different schema "
            "versions. Run `python train.py --regenerate-data`."
        )
        st.stop()

    gene_names = data["gene_names"]
    perturbation_names = data["perturbation_names"]
    initialize_random_cell(len(data["control_expression"]))

    with st.sidebar:
        st.header("Simulation controls")
        perturbation_name = st.selectbox(
            "Perturbation gene",
            options=perturbation_names[1:].tolist(),
            help="Choose the synthetic gene perturbation to apply.",
        )
        perturbation_id = int(
            np.flatnonzero(perturbation_names == perturbation_name)[0]
        )
        group_id = int(data["perturbation_group_ids"][perturbation_id])
        primary_id = int(data["perturbation_primary_pathway_ids"][perturbation_id])
        secondary_id = int(
            data["perturbation_secondary_pathway_ids"][perturbation_id]
        )

        st.markdown("#### Biological context")
        st.write(
            f"**Perturbation family:** "
            f"{data['perturbation_group_names'][group_id]}"
        )
        st.write(f"**Primary pathway:** {data['pathway_names'][primary_id]}")
        st.write(f"**Secondary pathway:** {data['pathway_names'][secondary_id]}")

        similarity = data["perturbation_similarity"][perturbation_id].copy()
        similarity[[0, perturbation_id]] = -np.inf
        nearest_id = int(np.argmax(similarity))
        st.write(
            f"**Most similar perturbation:** {perturbation_names[nearest_id]} "
            f"(cosine similarity {similarity[nearest_id]:.2f})"
        )

        if st.button("🎲 Select random control cell", width="stretch"):
            st.session_state.selected_cell_index = int(
                np.random.default_rng().integers(
                    0,
                    len(data["control_expression"]),
                )
            )

        selected_index = int(st.session_state.selected_cell_index)
        st.write(f"Control cell: `{data['cell_ids'][selected_index]}`")
        run_simulation = st.button(
            "Run Simulation",
            type="primary",
            width="stretch",
        )

        with st.expander("How the model works", expanded=False):
            st.markdown(
                """
                1. Encode the 1,000-gene control cell.
                2. Look up perturbation and perturbation-family embeddings.
                3. Fuse those representations in a compact MLP.
                4. Predict a gene-wise expression delta.
                5. Add the delta back to the control cell.

                The no-change baseline predicts a zero delta.
                """
            )

    st.info(
        "The simulator predicts a gene-by-gene response: "
        "**predicted perturbed expression = control expression + predicted delta**."
    )

    if not run_simulation:
        st.subheader("Ready to simulate")
        st.write(
            "Choose a perturbation, sample a control cell, and click "
            "**Run Simulation**. The app will summarize gene and pathway "
            "responses and compare the model with a no-change baseline."
        )
        return

    control_expression = data["control_expression"][selected_index].astype(np.float32)
    normalization_mean = np.asarray(
        checkpoint["normalization_mean"],
        dtype=np.float32,
    )
    normalization_std = np.asarray(
        checkpoint["normalization_std"],
        dtype=np.float32,
    )
    normalized_control = (control_expression - normalization_mean) / normalization_std

    with torch.no_grad():
        outputs = model(
            torch.from_numpy(normalized_control).unsqueeze(0).to(device),
            torch.tensor([perturbation_id], dtype=torch.long, device=device),
            torch.from_numpy(control_expression).unsqueeze(0).to(device),
        )
    predicted_delta = outputs["predicted_delta"].squeeze(0).cpu().numpy()
    predicted_expression = (
        outputs["predicted_expression"].squeeze(0).cpu().numpy()
    )
    expected_delta = expected_delta_for_cell(
        data,
        selected_index,
        perturbation_id,
    )

    if perturbation_id == checkpoint["config"]["heldout_perturbation_id"]:
        st.warning(
            "This perturbation was intentionally held out from training. Its "
            "family and pathways are known, but its individual ID is unseen, "
            "so this remains a grouped extrapolation test."
        )

    upregulated = np.argsort(predicted_delta)[-10:][::-1]
    downregulated = np.argsort(predicted_delta)[:10]
    model_mse = float(np.mean(np.square(predicted_delta - expected_delta)))
    baseline_mse = float(np.mean(np.square(expected_delta)))
    correlation = pearson_correlation(predicted_delta, expected_delta)
    overlap = top_k_changed_gene_overlap(
        predicted_delta,
        expected_delta,
        top_k=20,
    )

    st.subheader(f"Simulation result: {perturbation_name}")
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Largest increase",
        gene_names[upregulated[0]],
        f"{predicted_delta[upregulated[0]]:+.3f}",
    )
    metric_columns[1].metric(
        "Largest decrease",
        gene_names[downregulated[0]],
        f"{predicted_delta[downregulated[0]]:+.3f}",
    )
    metric_columns[2].metric("Delta correlation", f"{correlation:.3f}")
    metric_columns[3].metric("Top-20 gene overlap", f"{overlap:.0%}")

    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown("#### Top upregulated genes")
        up_table = pd.DataFrame(
            {
                "Gene": gene_names[upregulated],
                "Predicted delta": predicted_delta[upregulated],
            }
        )
        st.dataframe(
            up_table.style.format({"Predicted delta": "{:+.4f}"}),
            width="stretch",
            hide_index=True,
        )

    with right_column:
        st.markdown("#### Top downregulated genes")
        down_table = pd.DataFrame(
            {
                "Gene": gene_names[downregulated],
                "Predicted delta": predicted_delta[downregulated],
            }
        )
        st.dataframe(
            down_table.style.format({"Predicted delta": "{:+.4f}"}),
            width="stretch",
            hide_index=True,
        )

    st.markdown("### Before vs. after expression")
    st.pyplot(
        make_cell_state_figure(
            control_expression,
            predicted_expression,
            predicted_delta,
            gene_names,
        ),
        width="stretch",
    )

    st.markdown("### Pathway-level summary")
    pathway_rows = pathway_summary(predicted_delta, data, perturbation_id)
    pathway_table = pd.DataFrame(pathway_rows).sort_values(
        "mean_absolute_delta",
        ascending=False,
    )
    pathway_left, pathway_right = st.columns([1.15, 1])
    with pathway_left:
        st.pyplot(
            make_pathway_activity_figure(pathway_rows),
            width="stretch",
        )
    with pathway_right:
        st.dataframe(
            pathway_table.head(10).style.format(
                {
                    "mean_delta": "{:+.4f}",
                    "mean_absolute_delta": "{:.4f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown("### Model vs. no-change baseline")
    comparison_left, comparison_right = st.columns([1, 1])
    with comparison_left:
        st.pyplot(
            make_model_baseline_figure(model_mse, baseline_mse),
            width="stretch",
        )
    with comparison_right:
        st.metric("Model MSE", f"{model_mse:.5f}")
        st.metric(
            "No-change baseline MSE",
            f"{baseline_mse:.5f}",
            delta=f"{baseline_mse - model_mse:+.5f} model advantage",
            delta_color="normal",
        )
        st.write(
            "Because this app uses synthetic data, the generator provides a "
            "noise-free expected response for benchmarking this selected cell. "
            "The baseline assumes the perturbation causes no change."
        )

    st.markdown("### Strongest signed responses")
    st.pyplot(
        make_delta_figure(predicted_delta, gene_names),
        width="stretch",
    )

    results_table = simulation_results_table(
        gene_names,
        control_expression,
        predicted_expression,
        predicted_delta,
        expected_delta,
        data,
    )
    st.download_button(
        "Download simulation results as CSV",
        data=results_table.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"{data['cell_ids'][selected_index]}_{perturbation_name}_simulation.csv"
        ),
        mime="text/csv",
        width="stretch",
    )

    st.markdown("#### What this means")
    st.write(
        f"The model predicts that **{perturbation_name}** changes both individual "
        f"genes and coordinated pathways in **{data['cell_ids'][selected_index]}**. "
        "Positive deltas indicate predicted activation; negative deltas indicate "
        "predicted repression. This is a synthetic educational model, not a "
        "biological or clinical prediction."
    )


if __name__ == "__main__":
    main()
