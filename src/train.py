from __future__ import annotations
import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm

from .dataset import load_esol, split_loaders
from .models import GCNRegressor, GATRegressor, MPNNRegressor

def rmse(pred, y):
    return torch.sqrt(torch.mean((pred - y) ** 2))

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    preds, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        preds.append(out)
        ys.append(batch.y.view(-1, 1))
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return float(rmse(pred, y).item()), float(torch.mean(torch.abs(pred - y)).item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gcn","gat","mpnn"], default="gcn")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--out", type=str, default="checkpoints/best.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_list = load_esol(root="data")
    train_loader, val_loader, test_loader = split_loaders(data_list, batch_size=args.batch_size, seed=args.seed)

    in_dim = data_list[0].x.size(1)
    edge_dim = data_list[0].edge_attr.size(1)

    if args.model == "gcn":
        model = GCNRegressor(in_dim, hidden=args.hidden, layers=args.layers)
    elif args.model == "gat":
        model = GATRegressor(in_dim, hidden=args.hidden, layers=args.layers)
    else:
        model = MPNNRegressor(in_dim, edge_dim=edge_dim, hidden=args.hidden, layers=args.layers)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    best_val = math.inf
    bad = 0

    import os
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            y = batch.y.view(-1, 1)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total_loss += float(loss.item()) * batch.num_graphs

        train_loss = total_loss / len(train_loader.dataset)
        val_rmse, val_mae = eval_epoch(model, val_loader, device)

        print(f"[{epoch:03d}] train_mse={train_loss:.4f} | val_rmse={val_rmse:.4f} val_mae={val_mae:.4f}")

        if val_rmse < best_val - 1e-4:
            best_val = val_rmse
            bad = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.out)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_rmse, test_mae = eval_epoch(model, test_loader, device)
    print(f"TEST: rmse={test_rmse:.4f} mae={test_mae:.4f}")

if __name__ == "__main__":
    main()
