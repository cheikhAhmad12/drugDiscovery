from __future__ import annotations
import torch

TASKS = {
    "ESOL": {"type": "regression", "out_dim": 1},
    "HIV": {"type": "binary", "out_dim": 1},
    "TOX21": {"type": "multilabel", "out_dim": 12},
}

def get_task(name: str):
    name = name.upper()
    if name not in TASKS:
        raise ValueError(f"Unknown task {name}. Choose from {list(TASKS)}")
    return TASKS[name]

def compute_loss(task_type: str, logits, y):
    """
    y shape:
      - regression: [B,1]
      - binary: [B,1] (0/1)
      - multilabel: [B,T] with possible NaNs
    """
    if task_type == "regression":
        return torch.nn.functional.mse_loss(logits, y)

    # BCE losses expect float targets
    if task_type == "binary":
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

    if task_type == "multilabel":
        # mask NaNs (missing labels)
        mask = torch.isfinite(y)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=y.device, requires_grad=True)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits[mask], y[mask]
        )

    raise ValueError(task_type)