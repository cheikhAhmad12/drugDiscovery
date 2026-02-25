# app/streamlit_app.py
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from rdkit import Chem
from rdkit.Chem import Draw

from src.featurize import smiles_to_pyg
from src.models import (
    GraphBackboneGCN,
    GraphBackboneGAT,
    GraphBackboneMPNN,
    GraphPredictor,
)
from src.tasks import get_task
from src.explain import atom_importance_gradient


TOX21_TASKS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def default_ckpt_path(dataset: str, arch: str, seed: int, ckpt_dir: str = "checkpoints") -> str:
    run_name = f"{dataset}_{arch}_seed{seed}"
    return str(Path(ckpt_dir) / f"{run_name}_best.pt")


def default_test_json_path(dataset: str, arch: str, seed: int, log_dir: str = "logs") -> str:
    run_name = f"{dataset}_{arch}_seed{seed}"
    return str(Path(log_dir) / f"{run_name}_test.json")


def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    args = ckpt.get("args", {})
    task = ckpt.get("task", None)  # contains {"type":..., "out_dim":...} in our train.py
    if task is None:
        # fallback: infer from args.dataset
        task = get_task(args.get("dataset", "ESOL"))

    in_dim = ckpt.get("in_dim", None)
    edge_dim = ckpt.get("edge_dim", None)

    if in_dim is None or edge_dim is None:
        raise ValueError("Checkpoint missing in_dim/edge_dim. Re-train with the upgraded train.py.")

    arch = args.get("arch", "gcn")
    hidden = int(args.get("hidden", 128))
    layers = int(args.get("layers", 3))
    heads = int(args.get("heads", 4))

    if arch == "gcn":
        backbone = GraphBackboneGCN(in_dim, hidden=hidden, layers=layers)
    elif arch == "gat":
        backbone = GraphBackboneGAT(in_dim, hidden=hidden, heads=heads, layers=layers)
    elif arch == "mpnn":
        backbone = GraphBackboneMPNN(in_dim, edge_dim=edge_dim, hidden=hidden, layers=layers)
    else:
        raise ValueError(f"Unknown arch in ckpt: {arch}")

    model = GraphPredictor(backbone, hidden=hidden, out_dim=int(task["out_dim"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, args, task


def render_molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(450, 300))
    return img


def highlight_atoms(smiles: str, atom_scores: np.ndarray):
    """
    Highlight top atoms based on importance scores.
    Uses a simple top-k highlight; colors are auto.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n = mol.GetNumAtoms()
    if n == 0:
        return None

    scores = np.asarray(atom_scores).reshape(-1)
    if scores.shape[0] != n:
        # mismatch can happen if you AddHs in featurize; in that case rendering without Hs mismatches
        # We’ll fall back to non-highlighted render.
        return Draw.MolToImage(mol, size=(450, 300))

    k = min(6, n)
    top_idx = list(np.argsort(-scores)[:k])

    # Create per-atom color intensity
    # (Streamlit/RDKit needs explicit RGB tuples)
    atom_colors = {}
    for i in top_idx:
        s = float(scores[i])
        # Map score->intensity (no fixed “style”, just a simple interpolation)
        atom_colors[int(i)] = (1.0, 1.0 - 0.6 * s, 0.2)

    img = Draw.MolToImage(
        mol,
        size=(450, 300),
        highlightAtoms=top_idx,
        highlightAtomColors=atom_colors,
    )
    return img


st.set_page_config(page_title="Drug Discovery GNN Screening", layout="wide")
st.title("🧪 Drug Discovery — Geometric Deep Learning (GNN) Screening Demo")

with st.sidebar:
    st.header("Run selection")
    dataset = st.selectbox("Dataset / Task", ["ESOL", "HIV", "TOX21"], index=0)
    arch = st.selectbox("Architecture", ["gcn", "gat", "mpnn"], index=0)
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)
    ckpt_dir = st.text_input("Checkpoints dir", value="checkpoints")
    log_dir = st.text_input("Logs dir", value="logs")

    st.divider()
    st.caption("Optionnel: override chemin checkpoint")
    override_ckpt = st.text_input("Checkpoint path (override)", value="")

ckpt_path = override_ckpt.strip() if override_ckpt.strip() else default_ckpt_path(dataset, arch, int(seed), ckpt_dir=ckpt_dir)
test_json_path = default_test_json_path(dataset, arch, int(seed), log_dir=log_dir)

colA, colB = st.columns([1.2, 1])

with colA:
    st.subheader("Input")
    smiles = st.text_input("SMILES", value="CCO")
    st.caption("Exemples: CCO (ethanol), CC(=O)Oc1ccccc1C(=O)O (aspirin)")

with colB:
    st.subheader("Artifacts")
    st.write("**Checkpoint:**", ckpt_path)
    st.write("**Test metrics JSON:**", test_json_path)

# Show saved test metrics if present
if os.path.exists(test_json_path):
    try:
        with open(test_json_path, "r", encoding="utf-8") as f:
            test_info = json.load(f)
        st.info(f"Saved TEST metrics found for **{test_info.get('run_name','(run)')}**")
        st.json(test_info.get("test", {}))
    except Exception as e:
        st.warning(f"Could not read test metrics JSON: {e}")
else:
    st.warning("No saved TEST metrics found yet. Train first to generate logs/*.json.")

st.divider()

# Render molecule
mol_img = render_molecule(smiles)
if mol_img is not None:
    st.image(mol_img, caption="Molecule preview (RDKit)", use_container_width=False)
else:
    st.error("Invalid SMILES. Please fix the input before predicting.")
    st.stop()

predict_clicked = st.button("🚀 Predict", type="primary")

if predict_clicked:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(ckpt_path):
        st.error(
            "Checkpoint not found.\n\n"
            "Train first, e.g.\n"
            f"`python -m src.train --dataset {dataset} --arch {arch} --seed {int(seed)}`\n\n"
            "Or set the correct path in the sidebar."
        )
        st.stop()

    # Build data
    data = smiles_to_pyg(smiles, y=0.0)
    if data is None:
        st.error("Could not featurize SMILES into a graph.")
        st.stop()

    # Single-graph batch vector
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

    # Load model
    try:
        model, args, task = load_model_from_ckpt(ckpt_path, device)
    except Exception as e:
        st.error(f"Failed to load model from checkpoint: {e}")
        st.stop()

    # Predict
    data = data.to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr, data.batch).detach().cpu().numpy().reshape(-1)

    task_type = task["type"]
    out_dim = int(task["out_dim"])

    st.subheader("Prediction")

    if task_type == "regression":
        pred = float(logits[0])
        st.metric("ESOL predicted solubility (logS)", f"{pred:.4f}")
        st.caption("Plus logS est élevé, plus la molécule est soluble (en général).")

    elif task_type == "binary":
        prob = float(sigmoid(logits)[0])
        st.metric("HIV activity probability", f"{prob:.3f}")
        st.caption("C’est une proba modèle (pas une vérité clinique).")

    else:
        probs = sigmoid(logits)
        if probs.shape[0] != out_dim:
            st.warning(f"Output dim mismatch: got {probs.shape[0]} expected {out_dim}")
        task_names = TOX21_TASKS[:out_dim] if out_dim <= len(TOX21_TASKS) else [f"task_{i}" for i in range(out_dim)]
        df = pd.DataFrame({"task": task_names, "probability": probs[:out_dim]})
        df = df.sort_values("probability", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True)
        st.caption("Tox21 est multi-label: chaque tâche est une toxicité différente (probas indépendantes).")

    st.divider()
    st.subheader("Atom-level explainability (gradient attribution)")

    try:
        # NOTE: featurize.py uses AddHs; RDKit render here uses no Hs -> may mismatch.
        # We'll try anyway; if mismatch, fallback image without highlights.
        imp = atom_importance_gradient(model, data, device=device).numpy()
        st.write("Importance (normalized) — top atoms highlighted on the 2D structure.")
        hl_img = highlight_atoms(smiles, imp)
        st.image(hl_img, caption="Highlighted atoms (top contributors)", use_container_width=False)

        # Show as table
        st.caption("Scores par atome (index RDKit) :")
        st.dataframe(pd.DataFrame({"atom_index": list(range(len(imp))), "importance": imp}).sort_values("importance", ascending=False),
                     use_container_width=True)
    except Exception as e:
        st.warning(f"Explainability failed: {e}")
        st.caption("Tip: if H-addition causes mismatch, remove Chem.AddHs in src/featurize.py for perfect alignment.")