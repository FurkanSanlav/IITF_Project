"""
trainer.py — IITF Training Loop
================================
Provides the ``Trainer`` class, which encapsulates one complete training
lifecycle: forward pass, backward pass, validation, and checkpoint saving.

Checkpoint policy: ``best_model.pth`` is written whenever the validation loss
strictly improves; ``last_model.pth`` is overwritten every epoch so training
can always be resumed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Trainer:
    """Encapsulates the training and validation loops for ``DualTraceLSTM``.

    Parameters
    ----------
    model:
        The ``nn.Module`` to train.
    optimizer:
        Any PyTorch optimizer wrapping ``model.parameters()``.
    loss_fn:
        Callable criterion, e.g. ``get_loss_fn()`` → ``nn.HuberLoss``.
    device:
        ``torch.device`` to move tensors to.
    save_dir:
        Directory where ``best_model.pth`` and ``last_model.pth`` are written.
    grad_clip_norm:
        Maximum L2 norm for gradient clipping.  ``None`` disables clipping.
    lr_scheduler:
        Optional learning-rate scheduler.  ``ReduceLROnPlateau`` is the
        recommended choice; it is stepped with the validation loss at the
        end of every epoch.  Any other ``LRScheduler`` is stepped without
        arguments (i.e. epoch-based schedulers like ``CosineAnnealingLR``).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        save_dir: str | Path = "checkpoints",
        grad_clip_norm: Optional[float] = 1.0,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.save_dir = Path(save_dir)
        self.grad_clip_norm = grad_clip_norm
        self.lr_scheduler = lr_scheduler

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._best_val_loss: float = float("inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_one_epoch(self, train_loader: DataLoader) -> float:
        """Run one full pass over *train_loader* with gradient updates.

        Parameters
        ----------
        train_loader:
            DataLoader yielding ``(X, trace_id, y)`` batches.

        Returns
        -------
        float
            Mean Huber loss over all batches in the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = len(train_loader)

        pbar = tqdm(train_loader, desc="  train", unit="batch", leave=False)
        for x, trace_id, y in pbar:
            x        = x.to(self.device)
            trace_id = trace_id.to(self.device)
            y        = y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            y_pred: Tensor = self.model(x, trace_id)
            loss: Tensor   = self.loss_fn(y_pred, y)
            loss.backward()

            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / n_batches if n_batches > 0 else 0.0

    def validate(self, val_loader: DataLoader) -> float:
        """Evaluate *model* on *val_loader* without gradient tracking.

        Parameters
        ----------
        val_loader:
            DataLoader yielding ``(X, trace_id, y)`` batches.

        Returns
        -------
        float
            Mean Huber loss over all validation batches.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = len(val_loader)

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="  val  ", unit="batch", leave=False)
            for x, trace_id, y in pbar:
                x        = x.to(self.device)
                trace_id = trace_id.to(self.device)
                y        = y.to(self.device)

                y_pred = self.model(x, trace_id)
                loss   = self.loss_fn(y_pred, y)
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / n_batches if n_batches > 0 else 0.0

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        log_every: int = 1,
    ) -> dict[str, list[float]]:
        """Run the full training loop for *n_epochs*.

        Saves ``best_model.pth`` whenever validation loss strictly improves
        and ``last_model.pth`` at the end of every epoch.

        Parameters
        ----------
        train_loader:
            Training DataLoader.
        val_loader:
            Validation DataLoader.
        n_epochs:
            Number of epochs to train.
        log_every:
            Print a summary line every this many epochs.

        Returns
        -------
        dict[str, list[float]]
            History dict with keys ``"train_loss"`` and ``"val_loss"``.
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        logger.info(
            "Starting training — %d epochs | device=%s | save_dir=%s",
            n_epochs,
            self.device,
            self.save_dir,
        )

        for epoch in range(1, n_epochs + 1):
            t0 = time.perf_counter()
            train_loss = self.train_one_epoch(train_loader)
            val_loss   = self.validate(val_loader)
            elapsed    = time.perf_counter() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # ── LR scheduler step ─────────────────────────────────────────
            if self.lr_scheduler is not None:
                lr_before = self._current_lr()
                # ReduceLROnPlateau requires the monitored metric
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
                lr_after = self._current_lr()
                if lr_after < lr_before - 1e-12:
                    logger.info(
                        "LR reduced: %.2e → %.2e (epoch %d)",
                        lr_before, lr_after, epoch,
                    )

            # ── Checkpoint saving ─────────────────────────────────────────
            self._save_checkpoint("last_model.pth", epoch, val_loss)
            improved = val_loss < self._best_val_loss
            if improved:
                self._best_val_loss = val_loss
                self._save_checkpoint("best_model.pth", epoch, val_loss)

            if epoch % log_every == 0:
                lr_tag = f"  lr={self._current_lr():.2e}" if self.lr_scheduler else ""
                tag = " ★ best" if improved else ""
                logger.info(
                    "Epoch %3d/%d  train=%.4f  val=%.4f  (%.1fs)%s%s",
                    epoch,
                    n_epochs,
                    train_loss,
                    val_loss,
                    elapsed,
                    lr_tag,
                    tag,
                )

        logger.info(
            "Training complete.  Best val loss: %.4f", self._best_val_loss
        )
        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self, filename: str, epoch: int, val_loss: float
    ) -> None:
        """Persist model + optimizer + scheduler state to *save_dir/filename*."""
        payload = {
            "epoch":            epoch,
            "val_loss":         val_loss,
            "model_state":      self.model.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "scheduler_state":  (
                self.lr_scheduler.state_dict()
                if self.lr_scheduler is not None
                else None
            ),
        }
        path = self.save_dir / filename
        torch.save(payload, path)
        logger.debug("Checkpoint saved → %s", path)

    def _current_lr(self) -> float:
        """Return the current learning rate of the first param group."""
        return self.optimizer.param_groups[0]["lr"]
