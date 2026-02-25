import streamlit as st
import torch

from src.featurize import smiles_to_pyg
from src.models import GCNRegressor

st.title("Drug Discovery GNN — ESOL Solubility (demo)")

smiles = st.text_input("SMILES", value="CCO")  # ethanol
model_path = st.text_input("Checkpoint path", value="checkpoints/best.pt")

if st.button("Predict"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device)
    args = ckpt.get("args", {})
    # Rebuild model (GCN par défaut ici; adapte si tu veux gat/mpnn)
    # On doit connaître in_dim : on featurize d'abord
    data = smiles_to_pyg(smiles, y=0.0)
    if data is None:
        st.error("SMILES invalide.")
        st.stop()

    in_dim = data.x.size(1)
    model = GCNRegressor(in_dim=in_dim, hidden=args.get("hidden", 128), layers=args.get("layers", 3)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data = data.to(device)
    with torch.no_grad():
        pred = model(data.x, data.edge_index, data.edge_attr, data.batch).item()

    st.success(f"Predicted ESOL solubility (logS): {pred:.4f}")
