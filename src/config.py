"""
config.py — IITF Hyperparameter Configuration
===============================================
Defines ``IITFConfig``, a frozen-friendly dataclass that centralises every
tunable hyperparameter for the dual-trace training pipeline.

Serialisation to / from JSON means every training run is fully reproducible:
the config file saved alongside ``best_model.pth`` is the single source of
truth for reproducing results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class IITFConfig:
    """All hyperparameters for one IITF training run.

    Data
    ----
    seq_len:
        Length of the sliding-window input sequence (timesteps).
    window_stride:
        Step size between consecutive windows. Default is 10 to speed up training.
    batch_size:
        Mini-batch size for train / val / test loaders.
    num_features:
        Number of numeric telemetry features ``F``.
    train_split:
        Fraction of CSV files assigned to training (file-level).
    val_split:
        Fraction of CSV files assigned to validation.
    scaler_cache_dir:
        Directory for persisting fitted ``RobustScaler`` objects.
        ``None`` disables caching.

    Model
    -----
    hidden_size:
        LSTM hidden state dimensionality.
    num_lstm_layers:
        Number of stacked LSTM layers.
    embedding_dim:
        Dimensionality of the trace-ID embedding vector.
    dropout_rate:
        Dropout probability (LSTM inter-layer + MLP head).

    Optimiser & scheduler
    ---------------------
    learning_rate:
        Initial AdamW learning rate.
    weight_decay:
        AdamW L2 regularisation coefficient.
    grad_clip_norm:
        Maximum gradient L2 norm; ``None`` disables clipping.
    scheduler_factor:
        ``ReduceLROnPlateau`` multiplicative decay factor.
    scheduler_patience:
        Epochs without val-loss improvement before LR reduction.
    scheduler_min_lr:
        Floor on the learning rate.

    Training
    --------
    epochs:
        Total number of training epochs.
    num_workers:
        DataLoader worker processes (keep 0 on WSL).
    pin_memory:
        Pin CPU memory for faster GPU transfers.
    save_dir:
        Directory for ``best_model.pth`` / ``last_model.pth``.
    """

    # ── Data ──────────────────────────────────────────────────────────────
    seq_len:          int            = 60
    window_stride:    int            = 10
    batch_size:       int            = 256
    num_features:     int            = 8
    train_split:      float          = 0.70
    val_split:        float          = 0.15
    scaler_cache_dir: Optional[str]  = "data/processed/scalers"

    # ── Model ─────────────────────────────────────────────────────────────
    hidden_size:      int   = 128
    num_lstm_layers:  int   = 2
    embedding_dim:    int   = 16
    dropout_rate:     float = 0.2

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    learning_rate:       float          = 1e-3
    weight_decay:        float          = 1e-4
    grad_clip_norm:      Optional[float]= 1.0
    scheduler_factor:    float          = 0.5
    scheduler_patience:  int            = 3
    scheduler_min_lr:    float          = 1e-6

    # ── Training ──────────────────────────────────────────────────────────
    epochs:       int  = 50
    num_workers:  int  = 0
    pin_memory:   bool = True
    save_dir:     str  = "checkpoints"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        """Persist this configuration to *path* as a JSON file.

        Parameters
        ----------
        path:
            Destination file path (created / overwritten).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "IITFConfig":
        """Load an ``IITFConfig`` from a JSON file written by :meth:`to_json`.

        Parameters
        ----------
        path:
            Path to the JSON config file.

        Returns
        -------
        IITFConfig
            Reconstructed configuration object.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)

    def __str__(self) -> str:  # pragma: no cover
        lines = ["IITFConfig:"]
        for k, v in asdict(self).items():
            lines.append(f"  {k:<24} = {v}")
        return "\n".join(lines)
