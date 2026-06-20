# Mini Virtual Cell Simulator

An interactive, CPU-friendly PyTorch project that predicts how a cell's gene
expression profile changes after a genetic perturbation. The repository includes
a structured synthetic single-cell dataset, a delta-prediction neural network,
evaluation against a no-change baseline, and a Streamlit dashboard.

The project is fully self-contained: it downloads no external data and runs on a
normal laptop.

## What is a virtual cell simulator?

A virtual cell simulator is a computational model that estimates how a cell may
respond when its state is changed. In this project, a cell is represented by a
vector of 1,000 synthetic gene-expression values.

A **genetic perturbation** is an intervention that changes gene activity, such
as a knockout, knockdown, or activation experiment. The model predicts the
change caused by that intervention:

```text
predicted perturbed expression = control expression + predicted delta
```

Predicting the delta focuses the model on the perturbation response while the
original control profile is preserved through an explicit residual connection.

## Project highlights

- 5,000 synthetic cells, 1,000 genes, 10 perturbations, and a control condition
- 18 overlapping synthetic gene modules/pathways
- Three perturbation families with biologically related signatures
- Similar perturbations share pathway programs but keep individual effects
- Cell-state-dependent effects and realistic measurement noise
- Group-aware and perturbation-specific learned embeddings
- PyTorch `Dataset` and `DataLoader` pipeline
- Train, validation, and test splits
- One intentionally held-out perturbation for grouped extrapolation testing
- MSE, MAE, correlation, top-k gene overlap, and per-perturbation metrics
- Streamlit pathway summaries, baseline comparison, and CSV download
- Reproducible generation and training

## Synthetic biology simulation

Control expression profiles are generated from latent cell-state programs.
Genes are organized into overlapping modules such as:

- Cell Cycle
- Growth Signaling
- DNA Repair
- Stress Response
- Innate Immune Signaling
- Energy Metabolism
- Chromatin Regulation
- Protein Folding

Perturbations belong to one of three families:

1. Growth & Proliferation
2. Stress & Genome Maintenance
3. Immune & Metabolic Signaling

Members of a family share a pathway-level prototype. Each perturbation then adds
its own primary pathway, secondary pathway, and direct target genes. This makes
within-family perturbations more similar than unrelated perturbations while
preserving an intervention-specific signal.

For each cell, the perturbation signature is modulated by the cell's latent
state and baseline expression, then measurement noise is added. The result is a
learnable dataset rather than random input-label pairs.

The saved dataset includes:

- control and perturbed expression matrices
- target expression deltas
- cell IDs and split labels
- perturbation families
- primary and secondary pathway assignments
- pathway-to-gene membership
- perturbation signature similarities

## Model architecture

```text
Control expression (1,000 genes)
              |
              v
     Control MLP encoder
              |
              +-------------------+
                                  |
Perturbation ID --> ID embedding  |
                                  +--> Feature fusion MLP --> Delta head
Perturbation family               |                         (1,000 genes)
       --> Group embedding -------+                              |
                                                                 v
                                              predicted expression
                                         = control + predicted delta
```

The group embedding gives related perturbations a shared representation. The ID
embedding captures effects unique to an individual perturbation. The model
remains compact at roughly a few hundred thousand parameters and trains quickly
on CPU.

## Evaluation

The no-change baseline predicts:

```text
predicted delta = 0
predicted perturbed expression = control expression
```

`python evaluate.py` reports:

- overall delta MSE and MAE
- Pearson correlation between true and predicted deltas
- top-20 changed-gene overlap
- per-perturbation model and baseline MSE
- separate seen and held-out perturbation performance

It also creates:

- `results/predicted_vs_true_delta.png`
- `results/top_changed_genes.png`
- `results/baseline_vs_model.png`
- `results/per_perturbation_mse.png`

## Project structure

```text
virtual-cell-simulator/
|-- app.py
|-- train.py
|-- evaluate.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- synthetic_data.py
|   |-- dataset.py
|   |-- model.py
|   |-- train_utils.py
|   |-- evaluate_utils.py
|   `-- visualization.py
|-- data/
|   `-- .gitkeep
|-- checkpoints/
|   `-- .gitkeep
`-- results/
    `-- .gitkeep
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/enes-abulgayt/virtual-cell-simulator.git
cd virtual-cell-simulator
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py
```

The training script generates or upgrades the synthetic dataset when needed and
saves:

- `data/synthetic_expression.npz`
- `data/metadata.csv`
- `checkpoints/model.pt`
- `results/metrics.json`
- `results/loss_curve.png`

Optional commands:

```bash
python train.py --epochs 5
python train.py --regenerate-data
```

## Evaluate

```bash
python evaluate.py
```

Metrics are written to `results/evaluation.json`.

## Run the Streamlit app

```bash
streamlit run app.py
```

If the `streamlit` executable is not on your shell's `PATH`, use:

```bash
python -m streamlit run app.py
```

The app lets you:

- select a perturbation and random control cell
- inspect its perturbation family and affected pathways
- view top predicted up- and downregulated genes
- explore a pathway-level response summary
- compare the model with a no-change baseline
- view a before-versus-after dumbbell chart
- download all gene-level simulation results as CSV

## LinkedIn-ready explanation

> I built a Mini Virtual Cell Simulator with PyTorch and Streamlit. The model
> starts from a 1,000-gene control-cell profile, combines it with learned
> perturbation and pathway-family embeddings, and predicts the gene-expression
> delta caused by a genetic intervention. I designed a structured synthetic
> single-cell dataset with related perturbations, overlapping gene modules, and
> cell-state-dependent responses, then evaluated the model against a no-change
> baseline using MSE, correlation, and top-gene recovery. The interactive app
> turns the prediction into gene-level and pathway-level explanations and lets
> users export each simulation as CSV.

Suggested topics: `#PyTorch`, `#Bioinformatics`, `#SingleCell`,
`#MachineLearning`, `#Streamlit`, `#ComputationalBiology`.

## Example screenshots

Add screenshots or an animated GIF after running the app:

```text
docs/
|-- simulator-overview.png
|-- pathway-summary.png
`-- before-after-chart.png
```

## Limitations

- The data is synthetic and does not represent a specific organism or cell type.
- Pathway names are biologically inspired, but their gene members are synthetic.
- Gene expression is modeled as a dense continuous vector rather than raw
  count data with realistic sequencing depth and dropout.
- A learned perturbation ID does not provide true zero-shot generalization.
  The held-out perturbation can use its known family, but its individual
  embedding has not been trained.
- The model predicts an average point estimate and does not quantify uncertainty.
- There are no covariates for dose, time, donor, batch, or cell type.
- Results are educational and must not be used for clinical decisions.

## Roadmap to real scRNA-seq perturbation data

The next version can move from synthetic data to public CRISPR perturbation
experiments in stages:

1. Add AnnData and Scanpy loaders while keeping the current NumPy path.
2. Ingest a public dataset from scPerturb, such as a Perturb-seq or CRISPRi
   study with control and perturbed cells.
3. Apply quality control, library-size normalization, log transformation, and
   highly variable gene selection.
4. Build control-to-perturbation training pairs by cell type, batch, and donor.
5. Add pseudobulk evaluation for more stable perturbation-level comparisons.
6. Replace synthetic pathway membership with curated gene sets.
7. Represent perturbations using gene sequence, ontology, pathway, or network
   features to support stronger unseen-perturbation generalization.
8. Add dose, time, cell type, and experimental batch as model covariates.
9. Report uncertainty and validate across datasets rather than only random
   cell splits.

