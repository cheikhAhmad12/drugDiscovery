from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

def rmse(yhat, y):
    return float(torch.sqrt(torch.mean((yhat - y) ** 2)).item())

def mae(yhat, y):
    return float(torch.mean(torch.abs(yhat - y)).item())

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def multilabel_auc_pr(logits: np.ndarray, y: np.ndarray):
    """
    logits: [N,T], y: [N,T] with NaNs possible
    returns macro ROC-AUC and macro PR-AUC over tasks with enough labels
    """
    T = y.shape[1]
    rocs, prs = [], []
    probs = sigmoid(logits)
    for t in range(T):
        yt = y[:, t]
        pt = probs[:, t]
        mask = np.isfinite(yt)
        yt = yt[mask]
        pt = pt[mask]
        if yt.size < 10:
            continue
        # need both classes
        if len(np.unique(yt)) < 2:
            continue
        rocs.append(roc_auc_score(yt, pt))
        prs.append(average_precision_score(yt, pt))
    return (float(np.mean(rocs)) if rocs else float("nan"),
            float(np.mean(prs)) if prs else float("nan"))

def binary_auc_pr(logits: np.ndarray, y: np.ndarray):
    probs = sigmoid(logits.reshape(-1))
    y = y.reshape(-1)
    return roc_auc_score(y, probs), average_precision_score(y, probs)