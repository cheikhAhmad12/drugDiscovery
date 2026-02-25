from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, NNConv, global_mean_pool

class GraphBackboneGCN(nn.Module):
    def __init__(self, in_dim, hidden=128, layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden)])
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

class GraphBackboneGAT(nn.Module):
    def __init__(self, in_dim, hidden=128, heads=4, layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout)])
        for _ in range(layers - 1):
            self.convs.append(GATConv(hidden, hidden // heads, heads=heads, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

class GraphBackboneMPNN(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=128, layers=3, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        edge_nn0 = nn.Sequential(
            nn.Linear(edge_dim, hidden * in_dim),
            nn.ReLU(),
            nn.Linear(hidden * in_dim, hidden * in_dim),
        )
        self.convs.append(NNConv(in_dim, hidden, edge_nn0, aggr="mean"))

        for _ in range(layers - 1):
            edge_nn = nn.Sequential(
                nn.Linear(edge_dim, hidden * hidden),
                nn.ReLU(),
                nn.Linear(hidden * hidden, hidden * hidden),
            )
            self.convs.append(NNConv(hidden, hidden, edge_nn, aggr="mean"))

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

class GraphPredictor(nn.Module):
    def __init__(self, backbone: nn.Module, hidden=128, out_dim=1, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        g = self.backbone(x, edge_index, edge_attr, batch)
        return self.head(g)