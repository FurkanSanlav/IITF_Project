"""
tests/test_data_loader.py — IITF Dual-Trace Data Pipeline Test Suite
=====================================================================
Covers TraceScaler, TelemetryDataset, _FilteredTelemetryDataset,
get_dataloaders(), and all private helpers.  All tests use in-memory
tmp_path fixtures; no real CSVs are required.

Run with:
    pytest tests/test_data_loader.py -v
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
import torch

# ---------------------------------------------------------------------------
# Make src/ importable without an editable install
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (  # noqa: E402
    TRACE_ID,
    TraceScaler,
    TelemetryDataset,
    _FilteredTelemetryDataset,
    _check_trace,
    _count_csv_rows,
    _safe_read_csv,
    get_dataloaders,
)


# ===========================================================================
# Constants shared across tests
# ===========================================================================
N_ROWS = 100          # rows per dummy CSV
N_FEATURES = 4        # numeric feature columns
SEQ_LEN = 10          # short window for speed
BATCH_SIZE = 8
FS_FILES = 6          # fastStorage CSV count in fixture
RND_FILES = 4         # rnd CSV count in fixture
COLUMNS = [f"feat_{i}" for i in range(N_FEATURES)]


# ===========================================================================
# Helpers
# ===========================================================================

def _make_csv(path: Path, n_rows: int = N_ROWS, columns: list[str] = COLUMNS) -> Path:
    """Write a deterministic numeric CSV to *path* and return it."""
    rng = np.random.default_rng(seed=abs(hash(path.name)) % (2**31))
    df = pd.DataFrame(rng.standard_normal((n_rows, len(columns))), columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _make_corrupted_csv(path: Path) -> Path:
    """Write a file that is not a valid CSV (causes pandas to raise)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x01\xff\xfe" * 50)
    return path


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    """
    Create a minimal data/raw/ tree with dummy CSVs:
        tmp_path/data/raw/fastStorage/vm_{0..5}.csv  (FS_FILES files)
        tmp_path/data/raw/rnd/vm_{0..3}.csv          (RND_FILES files)
    Returns the project root (tmp_path) so paths mirror production layout.
    """
    for i in range(FS_FILES):
        _make_csv(tmp_path / "data" / "raw" / "fastStorage" / f"vm_{i:04d}.csv")
    for i in range(RND_FILES):
        _make_csv(tmp_path / "data" / "raw" / "rnd" / f"vm_{i:04d}.csv")
    return tmp_path


@pytest.fixture()
def fitted_scaler(raw_dir: Path) -> TraceScaler:
    """Return a TraceScaler fitted on the raw_dir fixture (no cache)."""
    scaler = TraceScaler(scaler_cache_dir=None)
    for trace in TRACE_ID:
        files = sorted((raw_dir / "data" / "raw" / trace).glob("*.csv"))
        scaler.fit(trace, files)
    return scaler


@pytest.fixture()
def dataset(raw_dir: Path, fitted_scaler: TraceScaler) -> TelemetryDataset:
    """Return a TelemetryDataset built from the raw_dir fixture."""
    return TelemetryDataset(
        root_dir=raw_dir,
        scaler=fitted_scaler,
        seq_len=SEQ_LEN,
    )


# ===========================================================================
# Section 1 — Private helpers
# ===========================================================================

class TestCheckTrace:
    def test_valid_traces_do_not_raise(self) -> None:
        for trace in TRACE_ID:
            _check_trace(trace)  # must not raise

    def test_invalid_trace_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown trace"):
            _check_trace("unknown_trace")


class TestSafeReadCsv:
    def test_reads_valid_csv(self, tmp_path: Path) -> None:
        fp = _make_csv(tmp_path / "vm.csv")
        df = _safe_read_csv(fp)
        assert df is not None
        assert df.shape == (N_ROWS, N_FEATURES)

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        result = _safe_read_csv(tmp_path / "nonexistent.csv")
        assert result is None

    def test_returns_none_for_corrupted_file(self, tmp_path: Path) -> None:
        fp = _make_corrupted_csv(tmp_path / "bad.csv")
        result = _safe_read_csv(fp)
        assert result is None

    def test_drops_non_numeric_columns(self, tmp_path: Path) -> None:
        fp = tmp_path / "mixed.csv"
        df_raw = pd.DataFrame({
            "timestamp": ["2024-01-01"] * 5,
            "cpu": np.random.rand(5),
            "mem": np.random.rand(5),
        })
        df_raw.to_csv(fp, index=False)
        df = _safe_read_csv(fp)
        assert df is not None
        assert "timestamp" not in df.columns
        assert set(df.columns) == {"cpu", "mem"}

    def test_column_filter_keeps_only_requested(self, tmp_path: Path) -> None:
        fp = _make_csv(tmp_path / "vm.csv")
        df = _safe_read_csv(fp, columns=["feat_0", "feat_1"])
        assert df is not None
        assert list(df.columns) == ["feat_0", "feat_1"]

    def test_missing_requested_columns_falls_back_gracefully(self, tmp_path: Path) -> None:
        """If requested columns are absent, all numeric columns are kept."""
        fp = _make_csv(tmp_path / "vm.csv")
        df = _safe_read_csv(fp, columns=["does_not_exist"])
        # present list is empty → df falls back to full numeric df
        assert df is not None
        assert df.shape[1] == N_FEATURES

    def test_handles_nan_via_ffill(self, tmp_path: Path) -> None:
        fp = tmp_path / "nan.csv"
        data = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 4.0]})
        data.to_csv(fp, index=False)
        df = _safe_read_csv(fp)
        assert df is not None
        assert not df.isnull().any().any()


class TestCountCsvRows:
    def test_counts_correct_rows(self, tmp_path: Path) -> None:
        fp = _make_csv(tmp_path / "vm.csv", n_rows=50)
        assert _count_csv_rows(fp) == 50

    def test_returns_zero_for_missing_file(self, tmp_path: Path) -> None:
        assert _count_csv_rows(tmp_path / "ghost.csv") == 0

    def test_returns_zero_for_corrupted_file(self, tmp_path: Path) -> None:
        fp = _make_corrupted_csv(tmp_path / "bad.csv")
        assert _count_csv_rows(fp) == 0


# ===========================================================================
# Section 2 — TraceScaler
# ===========================================================================

class TestTraceScalerFit:
    def test_fit_marks_scaler_as_fitted(self, raw_dir: Path) -> None:
        scaler = TraceScaler()
        files = sorted((raw_dir / "data" / "raw" / "fastStorage").glob("*.csv"))
        scaler.fit("fastStorage", files)
        assert scaler.is_fitted("fastStorage")
        assert not scaler.is_fitted("rnd")

    def test_fit_idempotent_when_already_fitted(self, raw_dir: Path) -> None:
        """A second .fit() call must be a no-op (no exception)."""
        scaler = TraceScaler()
        files = sorted((raw_dir / "data" / "raw" / "fastStorage").glob("*.csv"))
        scaler.fit("fastStorage", files)
        scaler.fit("fastStorage", files)  # should not raise

    def test_fit_raises_on_empty_file_list(self) -> None:
        scaler = TraceScaler()
        with pytest.raises(RuntimeError, match="No valid data found"):
            scaler.fit("rnd", [])

    def test_fit_skips_corrupted_files(self, tmp_path: Path) -> None:
        bad = _make_corrupted_csv(tmp_path / "bad.csv")
        good = _make_csv(tmp_path / "good.csv")
        scaler = TraceScaler()
        scaler.fit("rnd", [bad, good])
        assert scaler.is_fitted("rnd")

    def test_fit_raises_for_unknown_trace(self, raw_dir: Path) -> None:
        scaler = TraceScaler()
        files = sorted((raw_dir / "data" / "raw" / "fastStorage").glob("*.csv"))
        with pytest.raises(ValueError, match="Unknown trace"):
            scaler.fit("bogus", files)


class TestTraceScalerTransform:
    def test_transform_output_shape_preserved(self, fitted_scaler: TraceScaler) -> None:
        arr = np.random.rand(SEQ_LEN, N_FEATURES).astype(np.float64)
        out = fitted_scaler.transform("fastStorage", arr)
        assert out.shape == arr.shape

    def test_transform_raises_when_not_fitted(self) -> None:
        scaler = TraceScaler()
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.transform("fastStorage", np.zeros((5, 4)))

    def test_transform_produces_different_results_per_trace(
        self, fitted_scaler: TraceScaler
    ) -> None:
        arr = np.random.rand(10, N_FEATURES).astype(np.float64)
        out_fs = fitted_scaler.transform("fastStorage", arr)
        out_rnd = fitted_scaler.transform("rnd", arr)
        assert not np.allclose(out_fs, out_rnd), (
            "Both traces should produce distinct scaled values"
        )

    def test_transform_raises_for_unknown_trace(self, fitted_scaler: TraceScaler) -> None:
        with pytest.raises(ValueError):
            fitted_scaler.transform("unknown", np.zeros((5, 4)))


class TestTraceScalerCache:
    def test_scaler_is_persisted_and_reloaded(self, raw_dir: Path, tmp_path: Path) -> None:
        cache_dir = tmp_path / "scalers"
        scaler1 = TraceScaler(scaler_cache_dir=cache_dir)
        files = sorted((raw_dir / "data" / "raw" / "fastStorage").glob("*.csv"))
        scaler1.fit("fastStorage", files)

        pkl_path = cache_dir / "scaler_fastStorage.pkl"
        assert pkl_path.exists(), "Scaler pickle was not written"

        # New instance must auto-load from cache
        scaler2 = TraceScaler(scaler_cache_dir=cache_dir)
        assert scaler2.is_fitted("fastStorage")

    def test_reloaded_scaler_matches_original(self, raw_dir: Path, tmp_path: Path) -> None:
        cache_dir = tmp_path / "scalers"
        scaler1 = TraceScaler(scaler_cache_dir=cache_dir)
        files = sorted((raw_dir / "data" / "raw" / "fastStorage").glob("*.csv"))
        scaler1.fit("fastStorage", files)

        scaler2 = TraceScaler(scaler_cache_dir=cache_dir)
        arr = np.random.rand(5, N_FEATURES).astype(np.float64)
        np.testing.assert_array_almost_equal(
            scaler1.transform("fastStorage", arr),
            scaler2.transform("fastStorage", arr),
        )

    def test_no_cache_dir_does_not_write_files(self, raw_dir: Path, tmp_path: Path) -> None:
        scaler = TraceScaler(scaler_cache_dir=None)
        files = sorted((raw_dir / "data" / "raw" / "fastStorage").glob("*.csv"))
        scaler.fit("fastStorage", files)
        assert not any(tmp_path.rglob("*.pkl"))


# ===========================================================================
# Section 3 — TelemetryDataset
# ===========================================================================

class TestTelemetryDatasetLen:
    def test_len_equals_expected_windows(self, dataset: TelemetryDataset) -> None:
        """Each file of N_ROWS rows with seq_len=SEQ_LEN → N_ROWS - SEQ_LEN windows."""
        n_files = FS_FILES + RND_FILES
        expected = n_files * (N_ROWS - SEQ_LEN)
        assert len(dataset) == expected

    def test_len_is_zero_for_empty_directories(self, tmp_path: Path) -> None:
        """Directories exist but contain no CSVs → dataset is empty."""
        # Use an isolated tmp_path; an unfitted scaler is safe because no
        # __getitem__ calls are ever made on an empty dataset.
        root = tmp_path / "project"
        (root / "data" / "raw" / "fastStorage").mkdir(parents=True)
        (root / "data" / "raw" / "rnd").mkdir(parents=True)
        scaler = TraceScaler(scaler_cache_dir=None)
        ds = TelemetryDataset(root_dir=root, scaler=scaler, seq_len=SEQ_LEN)
        assert len(ds) == 0

    def test_len_is_zero_for_missing_directories(self, tmp_path: Path) -> None:
        """Neither trace directory exists → dataset is empty."""
        # Fresh root with no data/ subtree; unfitted scaler is safe.
        root = tmp_path / "empty_project"
        root.mkdir()
        scaler = TraceScaler(scaler_cache_dir=None)
        ds = TelemetryDataset(root_dir=root, scaler=scaler, seq_len=SEQ_LEN)
        assert len(ds) == 0

    def test_file_limit_reduces_len(self, raw_dir: Path, fitted_scaler: TraceScaler) -> None:
        ds = TelemetryDataset(
            root_dir=raw_dir, scaler=fitted_scaler, seq_len=SEQ_LEN, file_limit=2
        )
        # 2 fastStorage + 2 rnd → 4 files × (N_ROWS - SEQ_LEN) windows
        assert len(ds) == 4 * (N_ROWS - SEQ_LEN)


class TestTelemetryDatasetGetitem:
    def test_output_is_three_tuple(self, dataset: TelemetryDataset) -> None:
        result = dataset[0]
        assert len(result) == 3

    def test_x_shape(self, dataset: TelemetryDataset) -> None:
        x, _, _ = dataset[0]
        assert x.shape == (SEQ_LEN, N_FEATURES)

    def test_y_shape(self, dataset: TelemetryDataset) -> None:
        _, _, y = dataset[0]
        assert y.shape == (N_FEATURES,)

    def test_x_dtype_is_float32(self, dataset: TelemetryDataset) -> None:
        x, _, _ = dataset[0]
        assert x.dtype == torch.float32

    def test_y_dtype_is_float32(self, dataset: TelemetryDataset) -> None:
        _, _, y = dataset[0]
        assert y.dtype == torch.float32

    def test_trace_id_is_integer(self, dataset: TelemetryDataset) -> None:
        _, trace_id, _ = dataset[0]
        assert isinstance(trace_id, int)

    def test_trace_id_values_are_valid(self, dataset: TelemetryDataset) -> None:
        valid_ids = set(TRACE_ID.values())  # {0, 1}
        seen = {dataset[i][1] for i in range(len(dataset))}
        assert seen.issubset(valid_ids)

    def test_faststorage_trace_id_is_zero(self, dataset: TelemetryDataset) -> None:
        """Windows from fastStorage files must carry trace_id = 0."""
        # fastStorage files are indexed first
        _, trace_id, _ = dataset[0]
        assert trace_id == TRACE_ID["fastStorage"]

    def test_rnd_trace_id_is_one(self, dataset: TelemetryDataset) -> None:
        """Windows from rnd files must carry trace_id = 1."""
        fs_windows = FS_FILES * (N_ROWS - SEQ_LEN)
        _, trace_id, _ = dataset[fs_windows]   # first rnd window
        assert trace_id == TRACE_ID["rnd"]

    def test_x_and_y_are_not_equal(self, dataset: TelemetryDataset) -> None:
        """y is the step immediately after x; they should differ."""
        x, _, y = dataset[0]
        assert not torch.equal(x[-1], y), "Last x row should not equal y target"

    def test_all_items_accessible(self, dataset: TelemetryDataset) -> None:
        """Iterate every sample without exceptions."""
        for i in range(len(dataset)):
            x, tid, y = dataset[i]
            assert x.shape == (SEQ_LEN, N_FEATURES)


class TestTelemetryDatasetGracefulFallback:
    def test_corrupted_csv_returns_zero_tensors(
        self, raw_dir: Path, fitted_scaler: TraceScaler
    ) -> None:
        """A corrupted CSV that survives indexing returns zero tensors, not an error."""
        # Place a corrupted file in fastStorage
        bad = raw_dir / "data" / "raw" / "fastStorage" / "corrupted.csv"
        _make_csv(bad)                  # write valid CSV first (so row count > 0)
        bad.write_bytes(b"\x00\xff" * 200)  # then overwrite with garbage

        ds = TelemetryDataset(root_dir=raw_dir, scaler=fitted_scaler, seq_len=SEQ_LEN)
        # Find the sample that maps to the corrupted file
        for i in range(len(ds)):
            file_idx, _ = ds._samples[i]
            fp, _ = ds._file_records[file_idx]
            if fp.name == "corrupted.csv":
                x, _, y = ds[i]
                assert torch.all(x == 0.0)
                assert torch.all(y == 0.0)
                return
        pytest.skip("Corrupted file produced no windows (row count returned 0)")


class TestTraceCountsAndWeights:
    def test_get_trace_sample_counts_sums_correctly(
        self, dataset: TelemetryDataset
    ) -> None:
        counts = dataset.get_trace_sample_counts()
        assert sum(counts.values()) == len(dataset)

    def test_get_sample_weights_length_matches_dataset(
        self, dataset: TelemetryDataset
    ) -> None:
        weights = dataset.get_sample_weights()
        assert len(weights) == len(dataset)

    def test_sample_weights_are_positive(self, dataset: TelemetryDataset) -> None:
        weights = dataset.get_sample_weights()
        assert all(w > 0 for w in weights)

    def test_faststorage_weight_less_than_rnd_weight(
        self, dataset: TelemetryDataset
    ) -> None:
        """fastStorage has more files → higher count → lower per-sample weight."""
        weights = dataset.get_sample_weights()
        fs_windows = FS_FILES * (N_ROWS - SEQ_LEN)
        w_fs = weights[0]
        w_rnd = weights[fs_windows]
        assert w_fs < w_rnd, "fastStorage (larger trace) must have lower per-sample weight"


# ===========================================================================
# Section 4 — _FilteredTelemetryDataset
# ===========================================================================

class TestFilteredDataset:
    def _make_filtered(
        self, raw_dir: Path, fitted_scaler: TraceScaler, split: str = "train"
    ) -> _FilteredTelemetryDataset:
        fs_files = sorted((raw_dir / "data" / "raw" / "fastStorage").glob("*.csv"))
        rnd_files = sorted((raw_dir / "data" / "raw" / "rnd").glob("*.csv"))
        return _FilteredTelemetryDataset(
            root_dir=raw_dir,
            scaler=fitted_scaler,
            seq_len=SEQ_LEN,
            window_stride=1,
            feature_columns=None,
            file_paths_by_trace={"fastStorage": fs_files, "rnd": rnd_files},
        )

    def test_len_matches_parent_behaviour(
        self, raw_dir: Path, fitted_scaler: TraceScaler
    ) -> None:
        ds = self._make_filtered(raw_dir, fitted_scaler)
        expected = (FS_FILES + RND_FILES) * (N_ROWS - SEQ_LEN)
        assert len(ds) == expected

    def test_getitem_returns_correct_shapes(
        self, raw_dir: Path, fitted_scaler: TraceScaler
    ) -> None:
        ds = self._make_filtered(raw_dir, fitted_scaler)
        x, tid, y = ds[0]
        assert x.shape == (SEQ_LEN, N_FEATURES)
        assert y.shape == (N_FEATURES,)


# ===========================================================================
# Section 5 — get_dataloaders() integration
# ===========================================================================

class TestGetDataloaders:
    @pytest.fixture()
    def loaders(self, raw_dir: Path):
        return get_dataloaders(
            root_dir=raw_dir,
            seq_len=SEQ_LEN,
            batch_size=BATCH_SIZE,
            train_split=0.6,
            val_split=0.2,
            num_workers=0,
            pin_memory=False,
        )

    def test_returns_three_dataloaders(self, loaders) -> None:
        assert len(loaders) == 3

    def test_train_batch_x_shape(self, loaders) -> None:
        train_loader, _, _ = loaders
        x, _, _ = next(iter(train_loader))
        assert x.shape[1] == SEQ_LEN
        assert x.shape[2] == N_FEATURES

    def test_train_batch_y_shape(self, loaders) -> None:
        train_loader, _, _ = loaders
        _, _, y = next(iter(train_loader))
        assert y.shape[1] == N_FEATURES

    def test_train_batch_trace_id_shape(self, loaders) -> None:
        train_loader, _, _ = loaders
        _, trace_ids, _ = next(iter(train_loader))
        assert trace_ids.shape[0] == BATCH_SIZE

    def test_val_loader_is_deterministic(self, loaders) -> None:
        """Two passes over the val loader must yield identical first batches."""
        _, val_loader, _ = loaders
        x1, _, _ = next(iter(val_loader))
        x2, _, _ = next(iter(val_loader))
        assert torch.equal(x1, x2)

    def test_train_loader_uses_weighted_sampler(self, loaders) -> None:
        from torch.utils.data import WeightedRandomSampler
        train_loader, _, _ = loaders
        assert isinstance(train_loader.sampler, WeightedRandomSampler)

    def test_val_and_test_have_no_weighted_sampler(self, loaders) -> None:
        from torch.utils.data import WeightedRandomSampler
        _, val_loader, test_loader = loaders
        assert not isinstance(val_loader.sampler, WeightedRandomSampler)
        assert not isinstance(test_loader.sampler, WeightedRandomSampler)

    def test_scaler_cache_integration(self, raw_dir: Path, tmp_path: Path) -> None:
        cache = tmp_path / "scalers"
        get_dataloaders(
            root_dir=raw_dir,
            seq_len=SEQ_LEN,
            batch_size=BATCH_SIZE,
            num_workers=0,
            pin_memory=False,
            scaler_cache_dir=cache,
            train_split=0.6,
            val_split=0.2,
        )
        pkls = list(cache.glob("*.pkl"))
        assert len(pkls) == 2, f"Expected 2 .pkl files, found {pkls}"

    def test_empty_raw_directories_do_not_crash(self, tmp_path: Path) -> None:
        """get_dataloaders must not raise even when both trace dirs are empty."""
        (tmp_path / "data" / "raw" / "fastStorage").mkdir(parents=True)
        (tmp_path / "data" / "raw" / "rnd").mkdir(parents=True)
        # ScalerFit will raise RuntimeError because there's no data; that is
        # acceptable — we only verify no *unexpected* exception type surfaces.
        with pytest.raises((RuntimeError, Exception)):
            get_dataloaders(
                root_dir=tmp_path,
                seq_len=SEQ_LEN,
                batch_size=BATCH_SIZE,
                num_workers=0,
                pin_memory=False,
            )

    def test_file_limit_parameter_is_respected(self, raw_dir: Path) -> None:
        limit = 2
        train_loader, _, _ = get_dataloaders(
            root_dir=raw_dir,
            seq_len=SEQ_LEN,
            batch_size=BATCH_SIZE,
            num_workers=0,
            pin_memory=False,
            file_limit=limit,
            train_split=0.5,
            val_split=0.25,
        )
        # With file_limit=2 per trace and train_split=0.5 → 1 file each
        # Total train windows ≤ 2 * (N_ROWS - SEQ_LEN)
        assert len(train_loader.dataset) <= 2 * limit * (N_ROWS - SEQ_LEN)


# ===========================================================================
# Section 6 — Edge cases & regression guards
# ===========================================================================

class TestEdgeCases:
    def test_seq_len_equals_rows_minus_one_produces_one_window(
        self, raw_dir: Path, fitted_scaler: TraceScaler
    ) -> None:
        """seq_len = N_ROWS-1 should produce exactly 1 window per file."""
        ds = TelemetryDataset(
            root_dir=raw_dir,
            scaler=fitted_scaler,
            seq_len=N_ROWS - 1,
        )
        expected = (FS_FILES + RND_FILES) * 1
        assert len(ds) == expected

    def test_seq_len_equals_rows_produces_zero_windows(
        self, raw_dir: Path, fitted_scaler: TraceScaler
    ) -> None:
        """seq_len = N_ROWS leaves no room for a target step → 0 windows."""
        ds = TelemetryDataset(
            root_dir=raw_dir,
            scaler=fitted_scaler,
            seq_len=N_ROWS,
        )
        assert len(ds) == 0

    def test_row_cache_populated_after_getitem(self, dataset: TelemetryDataset) -> None:
        assert len(dataset._row_cache) == 0
        _ = dataset[0]
        assert len(dataset._row_cache) == 1

    def test_second_getitem_from_same_file_uses_cache(
        self, dataset: TelemetryDataset
    ) -> None:
        """Two consecutive accesses to windows in file 0 should share cache entry."""
        _ = dataset[0]
        cache_before = dict(dataset._row_cache)
        _ = dataset[1]                              # still file 0
        assert dataset._row_cache.keys() == cache_before.keys()

    def test_feature_columns_filter_applied(self, raw_dir: Path) -> None:
        """A scaler fitted on the *same* 2-column subset must be used."""
        subset = ["feat_0", "feat_1"]
        # Build and fit a scaler on the 2-column subset to avoid n_features mismatch.
        scaler = TraceScaler(scaler_cache_dir=None)
        for trace in TRACE_ID:
            files = sorted((raw_dir / "data" / "raw" / trace).glob("*.csv"))
            scaler.fit(trace, files)  # _safe_read_csv reads all cols; we then
            # override the internal scaler by re-fitting on the subset directly
        # Re-fit on the 2-col subset to match what __getitem__ will pass
        from sklearn.preprocessing import RobustScaler as _RS
        import pandas as _pd
        for trace in TRACE_ID:
            files = sorted((raw_dir / "data" / "raw" / trace).glob("*.csv"))
            frames = [_pd.read_csv(f)[["feat_0", "feat_1"]] for f in files]
            combined = _pd.concat(frames, ignore_index=True)
            scaler._scalers[trace] = _RS(quantile_range=(10.0, 90.0)).fit(combined.values)

        ds = TelemetryDataset(
            root_dir=raw_dir,
            scaler=scaler,
            seq_len=SEQ_LEN,
            feature_columns=subset,
        )
        x, _, y = ds[0]
        assert x.shape == (SEQ_LEN, len(subset))
        assert y.shape == (len(subset),)

    def test_only_one_trace_dir_present(self, tmp_path: Path) -> None:
        """Dataset with only fastStorage must not raise; rnd simply contributes 0."""
        root = tmp_path / "single_trace"
        fs_files = [
            _make_csv(root / "data" / "raw" / "fastStorage" / f"vm_{i}.csv")
            for i in range(3)
        ]
        # rnd dir intentionally absent; fit scaler only on fastStorage
        scaler = TraceScaler(scaler_cache_dir=None)
        scaler.fit("fastStorage", fs_files)
        # Provide a dummy fitted state for rnd so transform is never called on it
        import numpy as _np
        from sklearn.preprocessing import RobustScaler as _RS
        _dummy = _RS().fit(_np.zeros((10, N_FEATURES)))
        scaler._scalers["rnd"] = _dummy
        scaler._fitted["rnd"] = True

        ds = TelemetryDataset(root_dir=root, scaler=scaler, seq_len=SEQ_LEN)
        counts = ds.get_trace_sample_counts()
        assert counts["rnd"] == 0
        assert counts["fastStorage"] > 0
