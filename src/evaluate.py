"""
evaluate.py — IITF Per-Trace Evaluation
=========================================
Provides ``evaluate_model()``, which collects predictions across an entire
DataLoader, then reports MAE and RMSE broken down by trace source
(fastStorage / rnd) as well as global aggregates.

The per-trace split is essential because the training DataLoader uses
``WeightedRandomSampler``, meaning batches contain a mix of both traces;
a flat average would mask the performance gap between the stable
``fastStorage`` trace and the bursty ``rnd`` trace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Trace-name lookup — mirrors TRACE_ID in data_loader.py
_TRACE_NAMES: dict[int, str] = {0: "fastStorage", 1: "rnd"}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Structured container for per-trace and global evaluation metrics.

    Attributes
    ----------
    per_trace:
        Dict mapping trace name → ``{"mae": float, "rmse": float}``.
    global_mae:
        MAE averaged over all samples regardless of trace.
    global_rmse:
        RMSE averaged over all samples regardless of trace.
    n_samples:
        Total number of prediction samples evaluated.
    """
    per_trace:   dict[str, dict[str, float]] = field(default_factory=dict)
    global_mae:  float = 0.0
    global_rmse: float = 0.0
    n_samples:   int   = 0

    def __str__(self) -> str:  # pragma: no cover
        lines = [
            "=" * 54,
            f"  {'Metric':<28} {'MAE':>10}  {'RMSE':>10}",
            "-" * 54,
        ]
        for trace_name, metrics in self.per_trace.items():
            lines.append(
                f"  {trace_name:<28} {metrics['mae']:>10.4f}  {metrics['rmse']:>10.4f}"
            )
        lines.append("-" * 54)
        lines.append(
            f"  {'Global':<28} {self.global_mae:>10.4f}  {self.global_rmse:>10.4f}"
        )
        lines.append(f"  Samples evaluated: {self.n_samples:,}")
        lines.append("=" * 54)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    verbose: bool = True,
) -> EvalResult:
    """Collect predictions across *dataloader* and compute per-trace metrics.

    The function streams all batches under ``torch.no_grad()``, accumulates
    ``y_true``, ``y_pred``, and ``trace_id`` tensors in CPU memory, then
    computes MAE and RMSE independently for each trace and globally.

    Parameters
    ----------
    model:
        A ``DualTraceLSTM`` (or compatible) model in eval mode.
    dataloader:
        DataLoader yielding ``(X, trace_id, y)`` batches.  May be the
        validation or test loader.
    device:
        Device to run inference on.
    verbose:
        If ``True``, print the formatted result table after evaluation.

    Returns
    -------
    EvalResult
        Structured object with per-trace and global MAE / RMSE.
    """
    model.eval()

    all_y_true:   list[Tensor] = []
    all_y_pred:   list[Tensor] = []
    all_trace_ids: list[Tensor] = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="evaluate", unit="batch", leave=False)
        for x, trace_id, y in pbar:
            x        = x.to(device)
            trace_id = trace_id.to(device)

            y_pred = model(x, trace_id)

            # Collect on CPU to avoid OOM on large validation sets
            all_y_pred.append(y_pred.cpu())
            all_y_true.append(y.cpu())           # y already on CPU from loader
            all_trace_ids.append(trace_id.cpu())

    y_pred_all  = torch.cat(all_y_pred,   dim=0)   # (N, F)
    y_true_all  = torch.cat(all_y_true,   dim=0)   # (N, F)
    trace_id_all = torch.cat(all_trace_ids, dim=0)  # (N,)

    n_total = y_pred_all.shape[0]
    result  = EvalResult(n_samples=n_total)

    # ── Per-trace metrics ─────────────────────────────────────────────────
    present_traces = trace_id_all.unique().tolist()
    for tid in present_traces:
        tid = int(tid)
        mask = trace_id_all == tid
        yp   = y_pred_all[mask]   # (n_trace, F)
        yt   = y_true_all[mask]

        mae  = _mae(yp, yt)
        rmse = _rmse(yp, yt)

        trace_name = _TRACE_NAMES.get(tid, f"trace_{tid}")
        result.per_trace[trace_name] = {"mae": mae, "rmse": rmse}
        logger.info("%s — MAE: %.4f  RMSE: %.4f", trace_name, mae, rmse)

    # ── Global metrics ────────────────────────────────────────────────────
    result.global_mae  = _mae(y_pred_all, y_true_all)
    result.global_rmse = _rmse(y_pred_all, y_true_all)
    logger.info("Global  — MAE: %.4f  RMSE: %.4f", result.global_mae, result.global_rmse)

    if verbose:
        print(result)

    return result


# ---------------------------------------------------------------------------
# Checkpoint loader helper
# ---------------------------------------------------------------------------

def load_best_model(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[nn.Module, int, float]:
    """Load model weights from a checkpoint saved by ``Trainer``.

    Parameters
    ----------
    model:
        Uninitialised model instance with the same architecture as the
        checkpoint.
    checkpoint_path:
        Path to the ``.pth`` file written by :class:`Trainer`.
    device:
        Device to map tensors to when loading.

    Returns
    -------
    tuple[nn.Module, int, float]
        ``(model, epoch, val_loss)`` where *model* has weights loaded.
    """
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    epoch    = payload.get("epoch", -1)
    val_loss = payload.get("val_loss", float("nan"))
    logger.info(
        "Loaded checkpoint from '%s'  (epoch=%d, val_loss=%.4f)",
        checkpoint_path,
        epoch,
        val_loss,
    )
    return model, epoch, val_loss


# ---------------------------------------------------------------------------
# Private metric helpers
# ---------------------------------------------------------------------------

def _mae(y_pred: Tensor, y_true: Tensor) -> float:
    """Mean Absolute Error averaged over all elements."""
    return (y_pred - y_true).abs().mean().item()


def _rmse(y_pred: Tensor, y_true: Tensor) -> float:
    """Root Mean Squared Error averaged over all elements."""
    return (y_pred - y_true).pow(2).mean().sqrt().item()
