from __future__ import annotations
from typing import List, Tuple

import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from .featurize import smiles_to_pyg

def load_esol(root: str = "data") -> List:
    base = MoleculeNet(root=root, name="ESOL")  # fournit data + smiles
    data_list = []
    for d in base:
        smiles = getattr(d, "smiles", None)
        if smiles is None:
            continue
        y = float(d.y.item())
        pyg = smiles_to_pyg(smiles, y)
        if pyg is not None:
            data_list.append(pyg)
    return data_list

def split_loaders(data_list, batch_size=64, seed=42):
    idx = list(range(len(data_list)))
    train_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=seed)

    train = [data_list[i] for i in train_idx]
    val = [data_list[i] for i in val_idx]
    test = [data_list[i] for i in test_idx]

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size, shuffle=False),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )
