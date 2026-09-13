"""Train and evaluate the surrogate, config-driven instead of the original
notebook's hardcoded 4-stage loop (each stage doubling epochs / unfreezing
layer4 -- now `surrogate.unfreeze_after_epoch` and `surrogate.total_epochs`
in config/default.yaml).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from ..config import SurrogateConfig
from .model import ProxyModel


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    mae: float
    rmse: float
    accuracy_within_0_15: float
    pearson_r: float
    spearman_r: float


def accuracy_within(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 0.15) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true)) <= tol)) * 100.0


def evaluate_proxy(model: ProxyModel, X_val: torch.Tensor, y_val_np: np.ndarray, device: str) -> EpochMetrics:
    model.eval()
    with torch.no_grad():
        y_pred = np.clip(model.predict_from_embedding(X_val.to(device)).cpu().numpy(), 0, 1)
    mae = mean_absolute_error(y_val_np, y_pred)
    rmse = float(np.sqrt(np.mean((y_val_np - y_pred) ** 2)))
    acc = accuracy_within(y_val_np, y_pred)
    rp, _ = pearsonr(y_val_np, y_pred) if len(y_val_np) > 1 else (float("nan"), None)
    rs, _ = spearmanr(y_val_np, y_pred) if len(y_val_np) > 1 else (float("nan"), None)
    return EpochMetrics(epoch=-1, train_loss=float("nan"), mae=mae, rmse=rmse, accuracy_within_0_15=acc, pearson_r=rp, spearman_r=rs)


def train_proxy(
    embeddings: np.ndarray,
    scores: np.ndarray,
    cfg: SurrogateConfig,
    device: str | None = None,
    seed: int = 42,
) -> Tuple[ProxyModel, List[EpochMetrics]]:
    """Staged fine-tuning: head-only until `unfreeze_after_epoch`, then
    `unfreeze_layer` joins the trainable set for the remaining epochs."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X_tr_np, X_val_np, y_tr_np, y_val_np = train_test_split(
        embeddings.astype(np.float32), scores.astype(np.float32), test_size=cfg.val_fraction, random_state=seed
    )
    X_tr = torch.tensor(X_tr_np)
    y_tr = torch.tensor(y_tr_np)
    X_val = torch.tensor(X_val_np)

    model = ProxyModel().to(device)
    model.set_backbone_trainable(None)  # start head-only

    loss_fn = nn.HuberLoss(delta=cfg.huber_delta)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=cfg.batch_size, shuffle=True)

    history: List[EpochMetrics] = []
    optimiser = torch.optim.Adam(model.trainable_parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, cfg.total_epochs + 1):
        if epoch == cfg.unfreeze_after_epoch + 1:
            model.set_backbone_trainable(cfg.unfreeze_layer)
            optimiser = torch.optim.Adam(model.trainable_parameters(), lr=2e-4, weight_decay=1e-4)

        model.train()
        total_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model.predict_from_embedding(Xb)
            loss = loss_fn(pred, yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += float(loss.item())

        metrics = evaluate_proxy(model, X_val, y_val_np, device)
        metrics.epoch = epoch
        metrics.train_loss = total_loss / max(1, len(loader))
        history.append(metrics)

        if epoch % 10 == 0 or epoch == 1 or epoch == cfg.total_epochs:
            logging.info(
                "epoch %03d: loss=%.4f MAE=%.4f acc±.15=%.1f%% r=%.3f",
                epoch, metrics.train_loss, metrics.mae, metrics.accuracy_within_0_15, metrics.pearson_r,
            )

    return model, history
