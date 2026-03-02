from __future__ import annotations
import argparse, math, os
import torch
from tqdm import tqdm

from .dataset import load_moleculenet, split_loaders
from .tasks import get_task, compute_loss
from .metrics import rmse, mae, multilabel_auc_pr, binary_auc_pr
from .models import (
    GraphBackboneGCN, GraphBackboneGAT, GraphBackboneMPNN, GraphPredictor
)

@torch.no_grad()
def evaluate(model, loader, device, task_type: str):
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
        return {"rmse": rmse(logits, y.view(-1,1)), "mae": mae(logits, y.view(-1,1))}
    else:
        # classification
        log_np = logits.numpy()
        y_np = y.numpy()
        if task_type == "binary":
            roc, pr = binary_auc_pr(log_np, y_np)
            return {"roc_auc": float(roc), "pr_auc": float(pr)}
        else:
            roc, pr = multilabel_auc_pr(log_np, y_np)
            return {"roc_auc_macro": roc, "pr_auc_macro": pr}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ESOL","HIV","TOX21"], default="ESOL")
    ap.add_argument("--arch", choices=["gcn","gat","mpnn"], default="gcn")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--out", type=str, default="checkpoints/best.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    task = get_task(args.dataset)
    data_list = load_moleculenet(args.dataset, root="data")
    train_loader, val_loader, test_loader = split_loaders(data_list, batch_size=args.batch_size, seed=args.seed)

    in_dim = data_list[0].x.size(1)
    edge_dim = data_list[0].edge_attr.size(1)

    if args.arch == "gcn":
        backbone = GraphBackboneGCN(in_dim, hidden=args.hidden, layers=args.layers)
    elif args.arch == "gat":
        backbone = GraphBackboneGAT(in_dim, hidden=args.hidden, heads=args.heads, layers=args.layers)
    else:
        backbone = GraphBackboneMPNN(in_dim, edge_dim=edge_dim, hidden=args.hidden, layers=args.layers)

    model = GraphPredictor(backbone, hidden=args.hidden, out_dim=task["out_dim"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_key = "rmse" if task["type"] == "regression" else ("roc_auc" if task["type"] == "binary" else "roc_auc_macro")
    best = math.inf if best_key == "rmse" else -math.inf
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            # reshape targets
            y = batch.y
            if task["type"] == "regression":
                y = y.view(-1, 1).float()
            elif task["type"] == "binary":
                y = y.view(-1, 1).float()
            else:
                y = y.view(-1, task["out_dim"]).float()

            loss = compute_loss(task["type"], logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.item()) * batch.num_graphs

        val_metrics = evaluate(model, val_loader, device, task["type"])
        score = val_metrics[best_key]
        print(f"[{epoch:03d}] train_loss={total/len(train_loader.dataset):.4f} | val={val_metrics}")

        improved = (score < best - 1e-4) if best_key == "rmse" else (score > best + 1e-4)
        if improved:
            best = score
            bad = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.out)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device, task["type"])
    print(f"TEST: {test_metrics}")

if __name__ == "__main__":
    main()