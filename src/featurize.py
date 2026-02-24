from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from rdkit import Chem

ATOM_LIST = ["H","B","C","N","O","F","Si","P","S","Cl","Br","I"]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

def one_hot(x, choices):
    v = [0] * len(choices)
    if x in choices:
        v[choices.index(x)] = 1
    return v

def atom_features(atom: Chem.Atom) -> List[float]:
    sym = atom.GetSymbol()
    feats = []
    feats += one_hot(sym, ATOM_LIST)                      # type atome
    feats += [atom.GetAtomicNum()]                        # Z
    feats += [atom.GetTotalDegree()]                      # degré
    feats += [atom.GetFormalCharge()]                     # charge formelle
    feats += [int(atom.GetIsAromatic())]                  # aromaticité
    feats += [atom.GetTotalNumHs()]                       # H attachés
    return feats

def bond_features(bond: Chem.Bond) -> List[float]:
    bt = bond.GetBondType()
    feats = []
    feats += one_hot(bt, BOND_TYPES)
    feats += [int(bond.GetIsConjugated())]
    feats += [int(bond.IsInRing())]
    return feats

def smiles_to_pyg(smiles: str, y: float) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)  # optionnel; tu peux enlever si tu veux plus léger

    # Nodes
    x = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(np.array(x, dtype=np.float32))

    # Edges (undirected -> two directed edges)
    edge_index = []
    edge_attr = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bf = bond_features(b)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr, dtype=np.float32))
    y = torch.tensor([y], dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)
