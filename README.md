# Drug Discovery with Graph Neural Networks

Small end-to-end project for molecular property prediction using Geometric Deep Learning.

It trains GNNs on MoleculeNet datasets and exposes an interactive Streamlit app for single-molecule inference and atom-level gradient explanations.

## Features

- Tasks: `ESOL` (regression), `HIV` (binary classification), `TOX21` (12-label classification)
- Architectures: `gcn`, `gat`, `mpnn`
- Artifacts: best checkpoint, per-epoch metrics CSV, test metrics JSON
- App: SMILES input, prediction output, saved test metrics display, atom-level importance highlighting

## Project Structure

```text
drugDiscovery/
├── src/
│   ├── train.py           # main training pipeline
│   ├── dataset.py         # MoleculeNet loading + train/val/test split
│   ├── featurize.py       # SMILES -> PyG graph
│   ├── models.py          # GCN / GAT / MPNN + predictor head
│   ├── tasks.py           # task metadata + task-specific loss
│   ├── metrics.py         # RMSE/MAE/ROC-AUC/PR-AUC
│   ├── explain.py         # atom-level gradient attribution
│   └── train_zero.py      # older/alternate trainer
├── app/
│   ├── streamlit_app2.py  # main app (multi-task + explainability)
│   └── streamlit_app.py   # older minimal ESOL demo
├── requirements.txt
└── cmd                    # example training commands
```

## Requirements

- Python 3.10+ (commands below use `python3`)
- pip
- internet access for first dataset download (MoleculeNet)

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Main trainer:

```bash
python3 -m src.train --dataset ESOL --arch gcn --seed 42
```

Available options:

```bash
python3 -m src.train --help
```

Important flags:

- `--dataset {ESOL,HIV,TOX21}`
- `--arch {gcn,gat,mpnn}`
- `--epochs` (default `200`)
- `--batch_size` (default `64`)
- `--lr` (default `1e-3`)
- `--seed` (default `42`)
- `--patience` (default `25`)
- `--out_dir` (default `checkpoints`)
- `--log_dir` (default `logs`)

### Example Runs

```bash
# ESOL
python3 -m src.train --dataset ESOL --arch gcn --seed 42
python3 -m src.train --dataset ESOL --arch gat --seed 42
python3 -m src.train --dataset ESOL --arch mpnn --seed 42

# HIV
python3 -m src.train --dataset HIV --arch gcn --seed 42
python3 -m src.train --dataset HIV --arch gat --seed 42
python3 -m src.train --dataset HIV --arch mpnn --seed 42

# TOX21
python3 -m src.train --dataset TOX21 --arch gcn --seed 42
python3 -m src.train --dataset TOX21 --arch gat --seed 42
python3 -m src.train --dataset TOX21 --arch mpnn --seed 42
```

## Outputs

For run name `<DATASET>_<ARCH>_seed<SEED>`, training writes:

- `checkpoints/<RUN>_best.pt`
- `logs/<RUN>_metrics.csv`
- `logs/<RUN>_test.json`
- `logs/<RUN>_config.json`

Example:

- `checkpoints/TOX21_gcn_seed42_best.pt`
- `logs/TOX21_gcn_seed42_test.json`

## Streamlit App

Main app:

```bash
streamlit run app/streamlit_app2.py
```

In the sidebar, choose:

- Dataset/task
- Architecture
- Seed
- Checkpoint/log directories (or override checkpoint path)

Then input a SMILES string and click **Predict**.

## Metrics by Task

- `ESOL`: validation/test `rmse`, `mae`; early stopping on `rmse` (lower is better)
- `HIV`: validation/test `roc_auc`, `pr_auc`; early stopping on `roc_auc` (higher is better)
- `TOX21`: validation/test macro `roc_auc` and macro `pr_auc` with NaN masking; early stopping on macro `roc_auc` (higher is better)

## Notes

- `src/train.py` is the recommended trainer for current checkpoints and app compatibility.
- `app/streamlit_app2.py` expects checkpoints that include `task`, `in_dim`, and `edge_dim` (saved by `src/train.py`).
- `src/featurize.py` currently uses `Chem.AddHs(mol)`. If atom highlighting looks misaligned in Streamlit, removing `AddHs` can improve 2D atom index alignment.

## Troubleshooting

- `python: command not found`: use `python3` instead of `python`.
- `Checkpoint missing in_dim/edge_dim`: re-train with `python3 -m src.train ...` and use the new checkpoint.
- `Invalid SMILES in app`: verify your SMILES string is valid RDKit syntax.
