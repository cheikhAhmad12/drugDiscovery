# src/train.py
from __future__ import annotations

import argparse
import csv
import json
import math
import os

import torch
from tqdm import tqdm

from .dataset import load_moleculenet, split_loaders
from .tasks import get_task, compute_loss
from .metrics import rmse, mae, multilabel_auc_pr, binary_auc_pr
from .models import (
    GraphBackboneGCN,
    GraphBackboneGAT,
    GraphBackboneMPNN,
    GraphPredictor,
)


@torch.no_grad()
def evaluate(model, loader, device, task_type: str, out_dim: int):
    model.eval()
    all_logits, all_y = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_logits.append(logits.detach().cpu())
        all_y.append(batch.y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)

    if task_type == "regression":
        y = y.view(-1, 1).float()
        return {"rmse": rmse(logits, y), "mae": mae(logits, y)}

    # classification
    log_np = logits.numpy()
    y_np = y.numpy()

    if task_type == "binary":
        roc, pr = binary_auc_pr(log_np, y_np)
        return {"roc_auc": float(roc), "pr_auc": float(pr)}

    # multilabel
    y_np = y_np.reshape(-1, out_dim)
    log_np = log_np.reshape(-1, out_dim)
    roc, pr = multilabel_auc_pr(log_np, y_np)
    return {"roc_auc_macro": roc, "pr_auc_macro": pr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ESOL", "HIV", "TOX21"], default="ESOL")
    ap.add_argument("--arch", choices=["gcn", "gat", "mpnn"], default="gcn")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--out_dir", type=str, default="checkpoints")
    ap.add_argument("--log_dir", type=str, default="logs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    task = get_task(args.dataset)
    task_type = task["type"]
    out_dim = task["out_dim"]

    # Data
    data_list = load_moleculenet(args.dataset, root="data")
    train_loader, val_loader, test_loader = split_loaders(
        data_list, batch_size=args.batch_size, seed=args.seed
    )

    in_dim = data_list[0].x.size(1)
    edge_dim = data_list[0].edge_attr.size(1)

    # Model
    if args.arch == "gcn":
        backbone = GraphBackboneGCN(in_dim, hidden=args.hidden, layers=args.layers)
    elif args.arch == "gat":
        backbone = GraphBackboneGAT(
            in_dim, hidden=args.hidden, heads=args.heads, layers=args.layers
        )
    else:
        backbone = GraphBackboneMPNN(
            in_dim, edge_dim=edge_dim, hidden=args.hidden, layers=args.layers
        )

    model = GraphPredictor(backbone, hidden=args.hidden, out_dim=out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Pick early-stopping key
    if task_type == "regression":
        best_key = "rmse"
        best = math.inf
        is_better = lambda s, b: s < b - 1e-4
    elif task_type == "binary":
        best_key = "roc_auc"
        best = -math.inf
        is_better = lambda s, b: s > b + 1e-4
    else:
        best_key = "roc_auc_macro"
        best = -math.inf
        is_better = lambda s, b: s > b + 1e-4

    # Filenames
    run_name = f"{args.dataset}_{args.arch}_seed{args.seed}"
    ckpt_path = os.path.join(args.out_dir, f"{run_name}_best.pt")
    csv_path = os.path.join(args.log_dir, f"{run_name}_metrics.csv")
    test_json_path = os.path.join(args.log_dir, f"{run_name}_test.json")
    config_json_path = os.path.join(args.log_dir, f"{run_name}_config.json")

    # Save config
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # CSV header
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # store main val metric + whatever else
        header = ["epoch", "train_loss", f"val_{best_key}"]
        if task_type == "regression":
            header += ["val_mae"]
        elif task_type == "binary":
            header += ["val_pr_auc"]
        else:
            header += ["val_pr_auc_macro"]
        writer.writerow(header)

    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            y = batch.y.float()
            if task_type in ("regression", "binary"):
                y = y.view(-1, 1)
            else:
                y = y.view(-1, out_dim)

            loss = compute_loss(task_type, logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            total_loss += float(loss.item()) * batch.num_graphs

        train_loss = total_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device, task_type, out_dim)
        val_score = val_metrics[best_key]

        # Console log
        print(f"[{epoch:03d}] train_loss={train_loss:.4f} | val={val_metrics}")

        # CSV append
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            row = [epoch, train_loss, val_score]
            if task_type == "regression":
                row += [val_metrics["mae"]]
            elif task_type == "binary":
                row += [val_metrics["pr_auc"]]
            else:
                row += [val_metrics["pr_auc_macro"]]
            writer.writerow(row)

        # Early stopping + checkpointing
        if is_better(val_score, best):
            best = val_score
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "task": task,
                    "in_dim": in_dim,
                    "edge_dim": edge_dim,
                },
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    # Load best and test
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device, task_type, out_dim)
    print(f"BEST CKPT: {ckpt_path}")
    print(f"TEST: {test_metrics}")

    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "dataset": args.dataset,
                "arch": args.arch,
                "seed": args.seed,
                "best_key": best_key,
                "best_val": best,
                "test": test_metrics,
                "ckpt_path": ckpt_path,
                "csv_path": csv_path,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()