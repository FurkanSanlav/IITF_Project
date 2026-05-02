"""
model.py — IITF Dual-Trace LSTM Baseline
=========================================
Defines the ``DualTraceLSTM`` architecture that jointly processes:
  * Telemetry time-series via an ``nn.LSTM`` encoder.
  * Source-trace identity (fastStorage vs. rnd) via an ``nn.Embedding`` layer.

The final hidden state of the LSTM is concatenated with the trace embedding
and passed through a two-layer MLP to predict the next timestep ``y``.

Expected I/O shapes (B = batch, T = seq_len, F = num_features):
    Input:  X        → (B, T, F)
            trace_id → (B,)   — long tensor of 0 or 1
    Output: y_pred   → (B, F)
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DualTraceLSTM(nn.Module):
    """Multi-input LSTM baseline for dual-trace telemetry forecasting.

    Architecture
    ------------
    1. **LSTM encoder** — processes the full ``(B, T, F)`` sequence;
       the final layer's hidden state ``h_n[-1]`` is used as the sequence
       representation.
    2. **Trace embedding** — maps the integer ``trace_id`` (0 or 1) to a
       learned dense vector, giving the model an explicit inductive bias
       about the data source (stable vs. bursty).
    3. **MLP head** — ``Linear → LayerNorm → ReLU → Dropout → Linear``
       maps the concatenated ``[h_n ‖ embed]`` vector to ``y_pred``.

    Parameters
    ----------
    num_features:
        Number of telemetry features ``F`` (input and output dimensionality).
    hidden_size:
        Number of units in each LSTM hidden state.
    num_lstm_layers:
        Number of stacked LSTM layers.
    embedding_dim:
        Dimensionality of the trace-ID embedding vector.
    dropout_rate:
        Dropout probability applied inside the LSTM (between layers, only
        when ``num_lstm_layers > 1``) and in the MLP head.
    num_traces:
        Total number of distinct trace sources.  Default ``2``
        (fastStorage=0, rnd=1).
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 128,
        num_lstm_layers: int = 2,
        embedding_dim: int = 16,
        dropout_rate: float = 0.2,
        num_traces: int = 2,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate

        # ── 1. LSTM encoder ───────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0.0,
        )

        # ── 2. Trace-source embedding ─────────────────────────────────────
        self.trace_embedding = nn.Embedding(
            num_embeddings=num_traces,
            embedding_dim=embedding_dim,
        )

        # ── 3. MLP projection head ────────────────────────────────────────
        mlp_input_dim = hidden_size + embedding_dim
        self.head = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_input_dim * 2),
            nn.LayerNorm(mlp_input_dim * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(mlp_input_dim * 2, num_features),
        )

        self._init_weights()
        logger.info(
            "DualTraceLSTM initialised — features=%d, hidden=%d, "
            "layers=%d, embed_dim=%d, dropout=%.2f  |  params=%s",
            num_features,
            hidden_size,
            num_lstm_layers,
            embedding_dim,
            dropout_rate,
            f"{self._count_parameters():,}",
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, trace_id: Tensor) -> Tensor:
        """Compute next-step predictions for a batch of sequences.

        Parameters
        ----------
        x:
            Telemetry input of shape ``(B, T, F)``.
        trace_id:
            Source-trace identifiers of shape ``(B,)`` with values in
            ``{0, 1}``.  Must be a ``torch.long`` tensor.

        Returns
        -------
        Tensor
            Predicted next-step tensor of shape ``(B, F)``.
        """
        # ── LSTM: extract final hidden state ──────────────────────────────
        # lstm_out: (B, T, hidden_size) — we discard the per-step outputs
        # h_n:      (num_layers, B, hidden_size)
        _, (h_n, _) = self.lstm(x)

        # Take the hidden state of the topmost LSTM layer
        h_last: Tensor = h_n[-1]  # (B, hidden_size)

        # ── Trace embedding ───────────────────────────────────────────────
        trace_emb: Tensor = self.trace_embedding(trace_id.long())  # (B, embed_dim)

        # ── Concatenate and project ───────────────────────────────────────
        combined = torch.cat([h_last, trace_emb], dim=-1)  # (B, hidden+embed_dim)
        y_pred: Tensor = self.head(combined)                # (B, num_features)

        return y_pred

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply principled weight initialisation.

        * LSTM weights — orthogonal for recurrent kernels, Xavier uniform
          for input kernels; biases zeroed.
        * Linear layers — Kaiming uniform (He init, pairs well with ReLU).
        * Embedding — normal(0, 0.01) for a smooth embedding space at init.
        """
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1 (Jozefowicz et al., 2015)
                hidden = param.shape[0] // 4
                param.data[hidden : 2 * hidden].fill_(1.0)

        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.normal_(self.trace_embedding.weight, mean=0.0, std=0.01)

    def _count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss function factory
# ---------------------------------------------------------------------------


def get_loss_fn(delta: float = 1.0) -> nn.HuberLoss:
    """Return the Huber loss criterion configured for dual-trace telemetry.

    Huber loss (smooth L1) was chosen over MSE because the ``rnd`` trace
    contains inherently bursty, high-variance VM telemetry with frequent
    extreme outliers.  MSE would square those residuals and cause the
    gradient signal to be dominated by a handful of anomalous timesteps,
    destabilising training.  Huber loss behaves like L2 for small residuals
    (smooth gradient near zero) but degrades to L1 for large residuals
    (bounded gradient, outlier-robust), controlled by the ``delta`` threshold.

    Parameters
    ----------
    delta:
        The transition threshold between L2 and L1 behaviour.  The default
        of ``1.0`` corresponds to ``torch.nn.SmoothL1Loss`` semantics and
        works well across normalised (RobustScaler) feature ranges.

    Returns
    -------
    nn.HuberLoss
    """
    return nn.HuberLoss(delta=delta, reduction="mean")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # ── Configuration ──────────────────────────────────────────────────────
    BATCH = 32
    SEQ_LEN = 60
    NUM_FEATURES = 8
    HIDDEN = 128
    LAYERS = 2
    EMBED_DIM = 16
    DROPOUT = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running smoke test on device: %s", device)

    # ── Dummy inputs ───────────────────────────────────────────────────────
    x = torch.randn(BATCH, SEQ_LEN, NUM_FEATURES, device=device)
    trace_id = torch.randint(0, 2, (BATCH,), device=device)
    y_true = torch.randn(BATCH, NUM_FEATURES, device=device)

    # ── Model ──────────────────────────────────────────────────────────────
    model = DualTraceLSTM(
        num_features=NUM_FEATURES,
        hidden_size=HIDDEN,
        num_lstm_layers=LAYERS,
        embedding_dim=EMBED_DIM,
        dropout_rate=DROPOUT,
    ).to(device)

    model.train()
    y_pred = model(x, trace_id)

    # ── Shape assertions ───────────────────────────────────────────────────
    assert y_pred.shape == (BATCH, NUM_FEATURES), (
        f"Expected ({BATCH}, {NUM_FEATURES}), got {tuple(y_pred.shape)}"
    )

    loss_fn = get_loss_fn(delta=1.0)
    loss: Tensor = loss_fn(y_pred, y_true)

    assert loss.ndim == 0, "Loss must be a scalar tensor"
    assert loss.item() >= 0.0, "Huber loss must be non-negative"

    logger.info("✓  y_pred shape : %s", tuple(y_pred.shape))
    logger.info("✓  loss value   : %.6f", loss.item())
    logger.info("✓  Smoke test passed.")
    sys.exit(0)
