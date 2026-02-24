from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.nn import NNConv

class GCNRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return self.mlp(g)

class GATRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, heads: int = 4, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout))
        for _ in range(layers - 1):
            self.convs.append(GATConv(hidden, hidden // heads, heads=heads, dropout=dropout))
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return self.mlp(g)

class MPNNRegressor(nn.Module):
    """
    NNConv ~ message passing qui utilise edge_attr via un petit réseau.
    """
    def __init__(self, in_dim: int, edge_dim: int, hidden: int = 128, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.edge_nn = nn.Sequential(
            nn.Linear(edge_dim, hidden * in_dim),
            nn.ReLU(),
            nn.Linear(hidden * in_dim, hidden * in_dim)
        )
        self.convs = nn.ModuleList()
        self.convs.append(NNConv(in_dim, hidden, self.edge_nn, aggr="mean"))
        for _ in range(layers - 1):
            edge_nn_h = nn.Sequential(
                nn.Linear(edge_dim, hidden * hidden),
                nn.ReLU(),
                nn.Linear(hidden * hidden, hidden * hidden)
            )
            self.convs.append(NNConv(hidden, hidden, edge_nn_h, aggr="mean"))

        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        return self.mlp(g)
