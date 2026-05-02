"""
tests/test_model.py — DualTraceLSTM Test Suite
===============================================
Tests forward pass shapes, backward pass gradient flow, loss function
properties, and the evaluate_model() function — all running on CPU
for CI speed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import DualTraceLSTM, get_loss_fn
from evaluate import EvalResult, _mae, _rmse, evaluate_model


# ===========================================================================
# Shared fixtures
# ===========================================================================

BATCH      = 8
SEQ_LEN    = 12
N_FEATURES = 6
HIDDEN     = 32
LAYERS     = 2
EMBED_DIM  = 8
DROPOUT    = 0.1
DEVICE     = torch.device("cpu")


@pytest.fixture()
def model() -> DualTraceLSTM:
    """Small CPU model suitable for fast unit tests."""
    return DualTraceLSTM(
        num_features=N_FEATURES,
        hidden_size=HIDDEN,
        num_lstm_layers=LAYERS,
        embedding_dim=EMBED_DIM,
        dropout_rate=DROPOUT,
    ).to(DEVICE)


@pytest.fixture()
def dummy_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(X, trace_id, y)`` with realistic shapes."""
    x        = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
    trace_id = torch.randint(0, 2, (BATCH,))
    y        = torch.randn(BATCH, N_FEATURES)
    return x, trace_id, y


# ===========================================================================
# Section 1 — DualTraceLSTM construction
# ===========================================================================

class TestDualTraceLSTMInit:
    def test_instantiates_without_error(self) -> None:
        DualTraceLSTM(num_features=4)

    def test_lstm_is_batch_first(self, model: DualTraceLSTM) -> None:
        assert model.lstm.batch_first is True

    def test_embedding_num_embeddings(self, model: DualTraceLSTM) -> None:
        assert model.trace_embedding.num_embeddings == 2

    def test_embedding_dim(self, model: DualTraceLSTM) -> None:
        assert model.trace_embedding.embedding_dim == EMBED_DIM

    def test_trainable_parameters_nonzero(self, model: DualTraceLSTM) -> None:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total > 0

    def test_custom_num_traces(self) -> None:
        m = DualTraceLSTM(num_features=4, num_traces=5)
        assert m.trace_embedding.num_embeddings == 5


# ===========================================================================
# Section 2 — Forward pass shape tests
# ===========================================================================

class TestForwardPass:
    def test_output_shape(
        self, model: DualTraceLSTM, dummy_batch
    ) -> None:
        x, trace_id, _ = dummy_batch
        y_pred = model(x, trace_id)
        assert y_pred.shape == (BATCH, N_FEATURES)

    def test_output_dtype_float32(
        self, model: DualTraceLSTM, dummy_batch
    ) -> None:
        x, trace_id, _ = dummy_batch
        y_pred = model(x, trace_id)
        assert y_pred.dtype == torch.float32

    def test_all_faststorage_trace_ids(self, model: DualTraceLSTM) -> None:
        x        = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        trace_id = torch.zeros(BATCH, dtype=torch.long)
        y_pred   = model(x, trace_id)
        assert y_pred.shape == (BATCH, N_FEATURES)

    def test_all_rnd_trace_ids(self, model: DualTraceLSTM) -> None:
        x        = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        trace_id = torch.ones(BATCH, dtype=torch.long)
        y_pred   = model(x, trace_id)
        assert y_pred.shape == (BATCH, N_FEATURES)

    def test_batch_size_one(self, model: DualTraceLSTM) -> None:
        x        = torch.randn(1, SEQ_LEN, N_FEATURES)
        trace_id = torch.tensor([0])
        y_pred   = model(x, trace_id)
        assert y_pred.shape == (1, N_FEATURES)

    def test_single_layer_lstm(self) -> None:
        m = DualTraceLSTM(num_features=N_FEATURES, num_lstm_layers=1)
        x        = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        trace_id = torch.zeros(BATCH, dtype=torch.long)
        assert m(x, trace_id).shape == (BATCH, N_FEATURES)

    def test_different_seq_len(self, model: DualTraceLSTM) -> None:
        for t in [1, 30, 128]:
            x        = torch.randn(BATCH, t, N_FEATURES)
            trace_id = torch.zeros(BATCH, dtype=torch.long)
            assert model(x, trace_id).shape == (BATCH, N_FEATURES)

    def test_output_differs_per_trace_embedding(
        self, model: DualTraceLSTM
    ) -> None:
        """Same X, different trace_id → different outputs."""
        x   = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        y0  = model(x, torch.zeros(BATCH, dtype=torch.long))
        y1  = model(x, torch.ones(BATCH, dtype=torch.long))
        assert not torch.allclose(y0, y1), (
            "trace embedding should cause different predictions"
        )

    def test_eval_mode_no_dropout_variance(
        self, model: DualTraceLSTM
    ) -> None:
        """In eval mode, two identical forward passes must be identical."""
        model.eval()
        x        = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
        trace_id = torch.zeros(BATCH, dtype=torch.long)
        with torch.no_grad():
            y1 = model(x, trace_id)
            y2 = model(x, trace_id)
        assert torch.allclose(y1, y2)


# ===========================================================================
# Section 3 — Backward pass / gradient flow
# ===========================================================================

class TestBackwardPass:
    def test_loss_backward_populates_grads(
        self, model: DualTraceLSTM, dummy_batch
    ) -> None:
        x, trace_id, y = dummy_batch
        loss_fn = get_loss_fn()
        y_pred  = model(x, trace_id)
        loss    = loss_fn(y_pred, y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_grads_are_finite(
        self, model: DualTraceLSTM, dummy_batch
    ) -> None:
        x, trace_id, y = dummy_batch
        loss_fn = get_loss_fn()
        loss_fn(model(x, trace_id), y).backward()
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), "Non-finite gradient detected"

    def test_optimizer_step_changes_weights(
        self, model: DualTraceLSTM, dummy_batch
    ) -> None:
        x, trace_id, y = dummy_batch
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        loss_fn = get_loss_fn()
        optimizer.zero_grad()
        loss_fn(model(x, trace_id), y).backward()
        optimizer.step()
        changed = any(
            not torch.equal(before[n], p.data)
            for n, p in model.named_parameters()
            if p.requires_grad
        )
        assert changed, "No parameter was updated after optimizer.step()"

    def test_trace_embedding_grad_populated(
        self, model: DualTraceLSTM, dummy_batch
    ) -> None:
        x, trace_id, y = dummy_batch
        get_loss_fn()(model(x, trace_id), y).backward()
        assert model.trace_embedding.weight.grad is not None


# ===========================================================================
# Section 4 — get_loss_fn
# ===========================================================================

class TestGetLossFn:
    def test_returns_huber_loss(self) -> None:
        assert isinstance(get_loss_fn(), nn.HuberLoss)

    def test_default_delta(self) -> None:
        loss_fn = get_loss_fn()
        assert loss_fn.delta == 1.0

    def test_custom_delta(self) -> None:
        assert get_loss_fn(delta=0.5).delta == 0.5

    def test_loss_is_scalar(self) -> None:
        loss_fn = get_loss_fn()
        yp = torch.randn(BATCH, N_FEATURES)
        yt = torch.randn(BATCH, N_FEATURES)
        loss = loss_fn(yp, yt)
        assert loss.ndim == 0

    def test_loss_is_nonnegative(self) -> None:
        loss_fn = get_loss_fn()
        yp = torch.randn(BATCH, N_FEATURES)
        yt = torch.randn(BATCH, N_FEATURES)
        assert loss_fn(yp, yt).item() >= 0.0

    def test_loss_is_zero_for_perfect_prediction(self) -> None:
        loss_fn = get_loss_fn()
        y = torch.randn(BATCH, N_FEATURES)
        assert loss_fn(y, y).item() == pytest.approx(0.0, abs=1e-6)

    def test_reduction_is_mean(self) -> None:
        assert get_loss_fn().reduction == "mean"


# ===========================================================================
# Section 5 — evaluate_model integration
# ===========================================================================

class TestEvaluateModel:
    """Light-weight integration test using a tiny synthetic DataLoader."""

    @staticmethod
    def _make_loader(n_batches: int = 4):
        """Build a list-backed iterable that mimics a real DataLoader."""
        batches = []
        for i in range(n_batches):
            x        = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
            trace_id = torch.cat([
                torch.zeros(BATCH // 2, dtype=torch.long),
                torch.ones(BATCH // 2, dtype=torch.long),
            ])
            y = torch.randn(BATCH, N_FEATURES)
            batches.append((x, trace_id, y))
        return batches

    def test_returns_eval_result(self, model: DualTraceLSTM) -> None:
        loader = self._make_loader()
        result = evaluate_model(model, loader, DEVICE, verbose=False)
        assert isinstance(result, EvalResult)

    def test_both_traces_present_in_result(self, model: DualTraceLSTM) -> None:
        loader = self._make_loader()
        result = evaluate_model(model, loader, DEVICE, verbose=False)
        assert "fastStorage" in result.per_trace
        assert "rnd" in result.per_trace

    def test_global_mae_nonnegative(self, model: DualTraceLSTM) -> None:
        loader = self._make_loader()
        result = evaluate_model(model, loader, DEVICE, verbose=False)
        assert result.global_mae >= 0.0

    def test_global_rmse_nonnegative(self, model: DualTraceLSTM) -> None:
        loader = self._make_loader()
        result = evaluate_model(model, loader, DEVICE, verbose=False)
        assert result.global_rmse >= 0.0

    def test_n_samples_correct(self, model: DualTraceLSTM) -> None:
        n_batches = 4
        loader = self._make_loader(n_batches)
        result = evaluate_model(model, loader, DEVICE, verbose=False)
        assert result.n_samples == n_batches * BATCH

    def test_per_trace_mae_nonnegative(self, model: DualTraceLSTM) -> None:
        loader = self._make_loader()
        result = evaluate_model(model, loader, DEVICE, verbose=False)
        for metrics in result.per_trace.values():
            assert metrics["mae"] >= 0.0
            assert metrics["rmse"] >= 0.0


# ===========================================================================
# Section 6 — Private metric helpers
# ===========================================================================

class TestMetricHelpers:
    def test_mae_is_zero_for_identical_tensors(self) -> None:
        t = torch.randn(10, 4)
        assert _mae(t, t) == pytest.approx(0.0, abs=1e-6)

    def test_rmse_is_zero_for_identical_tensors(self) -> None:
        t = torch.randn(10, 4)
        assert _rmse(t, t) == pytest.approx(0.0, abs=1e-6)

    def test_mae_known_value(self) -> None:
        yp = torch.tensor([[1.0, 2.0]])
        yt = torch.tensor([[3.0, 4.0]])
        assert _mae(yp, yt) == pytest.approx(2.0, abs=1e-5)

    def test_rmse_known_value(self) -> None:
        yp = torch.tensor([[0.0]])
        yt = torch.tensor([[3.0]])
        assert _rmse(yp, yt) == pytest.approx(3.0, abs=1e-5)
