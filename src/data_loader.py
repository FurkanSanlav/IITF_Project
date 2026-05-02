"""
data_loader.py — IITF Dual-Trace Data Pipeline
================================================
Implements a WSL-optimised, lazy-loading PyTorch Dataset for two telemetry
traces (fastStorage / rnd), modular per-trace RobustScaling, source-origin
trace-ID tagging, and balanced 50/50 WeightedRandomSampler batching.

Directory layout expected:
    data/raw/fastStorage/   ← 1 250 VM CSVs
    data/raw/rnd/           ← 500  VM CSVs

Each CSV must contain a numeric column for every feature; the first column
may optionally be a timestamp / string index (it is dropped automatically).
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRACE_ID: dict[str, int] = {"fastStorage": 0, "rnd": 1}
_DTYPE = torch.float32


# ===========================================================================
# TraceScaler — one RobustScaler per trace, fit-once / persist
# ===========================================================================
class TraceScaler:
    """Maintains independent ``RobustScaler`` instances for each trace.

    Parameters
    ----------
    scaler_cache_dir:
        Directory used to persist fitted scalers.  Pass ``None`` to disable
        caching (scalers must be re-fitted on every run).
    quantile_range:
        IQR quantile range forwarded to :class:`sklearn.preprocessing.RobustScaler`.
    """

    def __init__(
        self,
        scaler_cache_dir: Optional[Path] = None,
        quantile_range: Tuple[float, float] = (10.0, 90.0),
    ) -> None:
        self._scalers: dict[str, RobustScaler] = {
            trace: RobustScaler(quantile_range=quantile_range)
            for trace in TRACE_ID
        }
        self._fitted: dict[str, bool] = {trace: False for trace in TRACE_ID}
        self._cache_dir = Path(scaler_cache_dir) if scaler_cache_dir else None

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._try_load_from_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, trace_name: str, file_paths: Sequence[Path]) -> None:
        """Fit the scaler for *trace_name* by reservoir-sampling rows from CSVs.

        Loads every file but only keeps a stratified random subset of rows to
        bound memory usage on WSL systems.

        Parameters
        ----------
        trace_name:
            One of ``"fastStorage"`` or ``"rnd"``.
        file_paths:
            All CSV paths belonging to this trace.
        """
        _check_trace(trace_name)
        if self._fitted[trace_name]:
            logger.info("Scaler for '%s' already fitted — skipping.", trace_name)
            return

        frames: list[pd.DataFrame] = []
        for fp in file_paths:
            df = _safe_read_csv(fp)
            if df is not None and not df.empty:
                # reservoir sample: at most 512 rows per file
                frames.append(df.sample(min(len(df), 512), random_state=42))

        if not frames:
            raise RuntimeError(
                f"No valid data found for trace '{trace_name}' — cannot fit scaler."
            )

        combined = pd.concat(frames, ignore_index=True)
        self._scalers[trace_name].fit(combined.values)
        self._fitted[trace_name] = True
        logger.info(
            "Fitted RobustScaler for '%s' on %d rows × %d features.",
            trace_name,
            len(combined),
            combined.shape[1],
        )
        self._try_save_to_cache(trace_name)

    def transform(self, trace_name: str, array: np.ndarray) -> np.ndarray:
        """Apply the fitted scaler for *trace_name* to *array*.

        Parameters
        ----------
        trace_name:
            One of ``"fastStorage"`` or ``"rnd"``.
        array:
            2-D NumPy array of shape ``(T, F)``.

        Returns
        -------
        np.ndarray
            Scaled array of identical shape.
        """
        _check_trace(trace_name)
        if not self._fitted[trace_name]:
            raise RuntimeError(
                f"Scaler for '{trace_name}' is not fitted.  "
                "Call TraceScaler.fit() before transform()."
            )
        return self._scalers[trace_name].transform(array)

    def is_fitted(self, trace_name: str) -> bool:
        """Return whether the scaler for *trace_name* has been fitted."""
        return self._fitted[trace_name]

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _cache_path(self, trace_name: str) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"scaler_{trace_name}.pkl"

    def _try_load_from_cache(self) -> None:
        for trace in TRACE_ID:
            path = self._cache_path(trace)
            if path and path.exists():
                with open(path, "rb") as fh:
                    self._scalers[trace] = pickle.load(fh)
                self._fitted[trace] = True
                logger.info("Loaded cached scaler for '%s' from %s.", trace, path)

    def _try_save_to_cache(self, trace_name: str) -> None:
        path = self._cache_path(trace_name)
        if path is None:
            return
        with open(path, "wb") as fh:
            pickle.dump(self._scalers[trace_name], fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved scaler for '%s' to %s.", trace_name, path)


# ===========================================================================
# TelemetryDataset — lazy, file-indexed PyTorch Dataset
# ===========================================================================
class TelemetryDataset(Dataset):
    """Lazy-loading Dataset that yields ``(X_sequence, trace_id, y_target)`` tuples.

    Each sample is a sliding-window slice of length ``seq_len`` drawn from one
    VM's CSV file.  The target ``y_target`` is the *next* timestep immediately
    following the window (single-step forecast).

    Parameters
    ----------
    root_dir:
        Project root; ``data/raw/fastStorage`` and ``data/raw/rnd`` are
        resolved relative to this.
    scaler:
        A **fitted** :class:`TraceScaler` instance.
    seq_len:
        Number of consecutive timesteps in each input window.
    feature_columns:
        Explicit list of column names to use.  If ``None``, all numeric
        columns are used (inferred from the first valid file of each trace).
    file_limit:
        Cap the number of files loaded per trace (useful for smoke tests).
    """

    def __init__(
        self,
        root_dir: Path,
        scaler: TraceScaler,
        seq_len: int = 60,
        feature_columns: Optional[List[str]] = None,
        file_limit: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.scaler = scaler
        self.seq_len = seq_len
        self.feature_columns = feature_columns

        # ---- Index all file paths ------------------------------------------
        self._index: list[tuple[Path, str, int, int]] = []
        # Each entry: (file_path, trace_name, start_row, n_available_rows)
        # We defer computing slices to __getitem__ for memory efficiency.

        # We store (file_path, trace_name, trace_id) and build a slice index.
        self._file_records: list[tuple[Path, str]] = []
        for trace_name in TRACE_ID:
            trace_dir = self.root_dir / "data" / "raw" / trace_name
            if not trace_dir.exists():
                logger.warning("Trace directory not found: %s", trace_dir)
                continue
            csv_files = sorted(trace_dir.glob("*.csv"))
            if file_limit:
                csv_files = csv_files[:file_limit]
            for fp in csv_files:
                self._file_records.append((fp, trace_name))

        # ---- Build flat slice index ----------------------------------------
        #   For each file we count how many valid windows it contributes.
        #   We record (file_idx, window_start_row) for every window.
        self._samples: list[tuple[int, int]] = []   # (file_idx, start_row)
        self._row_cache: dict[int, pd.DataFrame] = {}   # file_idx → dataframe
        # Pre-scan row counts without loading data fully.
        self._file_row_counts: list[int] = []
        for file_idx, (fp, _) in enumerate(self._file_records):
            n_rows = _count_csv_rows(fp)
            self._file_row_counts.append(n_rows)
            n_windows = max(0, n_rows - seq_len)   # seq_len rows → 1 window
            for start in range(n_windows):
                self._samples.append((file_idx, start))

        logger.info(
            "Dataset indexed: %d files → %d windows (seq_len=%d).",
            len(self._file_records),
            len(self._samples),
            seq_len,
        )

    # ------------------------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, Tensor]:
        """Return ``(X_sequence, trace_id, y_target)`` for sample *idx*.

        * ``X_sequence`` — ``Tensor`` of shape ``(seq_len, n_features)``
        * ``trace_id``   — ``int`` (0 = fastStorage, 1 = rnd)
        * ``y_target``   — ``Tensor`` of shape ``(n_features,)``
        """
        file_idx, start_row = self._samples[idx]
        fp, trace_name = self._file_records[file_idx]

        df = self._get_df(file_idx, fp, trace_name)
        if df is None or len(df) < self.seq_len + 1:
            # Graceful fallback: return zeros
            n_feat = len(self.feature_columns) if self.feature_columns else 1
            x = torch.zeros(self.seq_len, n_feat, dtype=_DTYPE)
            y = torch.zeros(n_feat, dtype=_DTYPE)
            return x, TRACE_ID[trace_name], y

        window = df.iloc[start_row : start_row + self.seq_len + 1].values  # (T+1, F)
        x_np = window[:self.seq_len]     # (seq_len, F)
        y_np = window[self.seq_len]      # (F,)

        # Apply per-trace scaling
        x_scaled = self.scaler.transform(trace_name, x_np)
        y_scaled = self.scaler.transform(trace_name, y_np.reshape(1, -1)).squeeze(0)

        x_tensor = torch.tensor(x_scaled, dtype=_DTYPE)
        y_tensor = torch.tensor(y_scaled, dtype=_DTYPE)
        return x_tensor, TRACE_ID[trace_name], y_tensor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_df(
        self, file_idx: int, fp: Path, trace_name: str
    ) -> Optional[pd.DataFrame]:
        """Lazy-load a DataFrame, caching it for repeated window lookups."""
        if file_idx not in self._row_cache:
            df = _safe_read_csv(fp, columns=self.feature_columns)
            self._row_cache[file_idx] = df   # may be None
        return self._row_cache[file_idx]

    # ---- Introspection helpers -------------------------------------------
    def get_trace_sample_counts(self) -> dict[str, int]:
        """Return the number of windows contributed by each trace."""
        counts: dict[str, int] = {t: 0 for t in TRACE_ID}
        for file_idx, _ in self._samples:
            _, trace_name = self._file_records[file_idx]
            counts[trace_name] += 1
        return counts

    def get_sample_weights(self) -> list[float]:
        """Compute per-sample inverse-frequency weights for balanced sampling.

        Ensures each trace contributes equally regardless of its file count.
        """
        trace_counts = self.get_trace_sample_counts()
        total = sum(trace_counts.values())
        # Weight for a sample from trace T = total / (n_traces × count_T)
        n_traces = len([c for c in trace_counts.values() if c > 0])
        weights: list[float] = []
        for file_idx, _ in self._samples:
            _, trace_name = self._file_records[file_idx]
            count = trace_counts[trace_name]
            w = total / (n_traces * count) if count > 0 else 0.0
            weights.append(w)
        return weights


# ===========================================================================
# Factory function — get_dataloaders()
# ===========================================================================
def get_dataloaders(
    root_dir: str | Path = ".",
    seq_len: int = 60,
    batch_size: int = 64,
    train_split: float = 0.7,
    val_split: float = 0.15,
    # test_split is implicitly 1 - train - val
    num_workers: int = 0,
    feature_columns: Optional[List[str]] = None,
    scaler_cache_dir: Optional[str | Path] = None,
    file_limit: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build and return ``(train_loader, val_loader, test_loader)``.

    The training loader uses a :class:`~torch.utils.data.WeightedRandomSampler`
    to guarantee 50 / 50 representation of ``fastStorage`` and ``rnd`` traces.
    Validation and test loaders are sequential (no resampling).

    Parameters
    ----------
    root_dir:
        Project root directory.  ``data/raw/`` is resolved relative to this.
    seq_len:
        Sliding-window length in timesteps.
    batch_size:
        Mini-batch size for all three loaders.
    train_split:
        Fraction of files assigned to training (file-level split).
    val_split:
        Fraction of files assigned to validation.
    num_workers:
        DataLoader workers.  Set to ``0`` on WSL to avoid fork issues.
    feature_columns:
        Explicit feature columns.  ``None`` → all numeric columns.
    scaler_cache_dir:
        Directory for caching fitted scalers.  ``None`` → no caching.
    file_limit:
        Cap files per trace (smoke-test flag).
    pin_memory:
        Pin memory for GPU transfer.
    persistent_workers:
        Keep DataLoader workers alive between iterations.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        ``(train_loader, val_loader, test_loader)``
    """
    root_dir = Path(root_dir).resolve()

    # ------------------------------------------------------------------ #
    # 1. Collect and split file paths per trace                           #
    # ------------------------------------------------------------------ #
    split_files: dict[str, dict[str, list[Path]]] = {}   # trace → split → paths

    for trace_name in TRACE_ID:
        trace_dir = root_dir / "data" / "raw" / trace_name
        if not trace_dir.exists():
            logger.warning("Missing trace directory: %s", trace_dir)
            split_files[trace_name] = {"train": [], "val": [], "test": []}
            continue

        all_files = sorted(trace_dir.glob("*.csv"))
        if file_limit:
            all_files = all_files[:file_limit]

        n = len(all_files)
        n_train = int(n * train_split)
        n_val = int(n * val_split)

        split_files[trace_name] = {
            "train": all_files[:n_train],
            "val":   all_files[n_train : n_train + n_val],
            "test":  all_files[n_train + n_val :],
        }
        logger.info(
            "Trace '%s': %d train | %d val | %d test files.",
            trace_name,
            n_train,
            n_val,
            n - n_train - n_val,
        )

    # ------------------------------------------------------------------ #
    # 2. Fit scalers on training files only (no leakage)                  #
    # ------------------------------------------------------------------ #
    cache = Path(scaler_cache_dir) if scaler_cache_dir else None
    scaler = TraceScaler(scaler_cache_dir=cache)
    for trace_name in TRACE_ID:
        if not scaler.is_fitted(trace_name):
            scaler.fit(trace_name, split_files[trace_name]["train"])

    # ------------------------------------------------------------------ #
    # 3. Build datasets for each split                                     #
    # ------------------------------------------------------------------ #
    def _make_dataset(split: str) -> TelemetryDataset:
        # Create a temporary root that exposes only the relevant slice of files.
        # We use a custom subclass that accepts pre-filtered file lists.
        return _FilteredTelemetryDataset(
            root_dir=root_dir,
            scaler=scaler,
            seq_len=seq_len,
            feature_columns=feature_columns,
            file_paths_by_trace={
                trace: split_files[trace][split] for trace in TRACE_ID
            },
        )

    train_ds = _make_dataset("train")
    val_ds   = _make_dataset("val")
    test_ds  = _make_dataset("test")

    # ------------------------------------------------------------------ #
    # 4. Balanced WeightedRandomSampler for training                      #
    # ------------------------------------------------------------------ #
    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    # ------------------------------------------------------------------ #
    # 5. Assemble DataLoaders                                             #
    # ------------------------------------------------------------------ #
    _common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=persistent_workers and num_workers > 0,
    )
    train_loader = DataLoader(train_ds, sampler=sampler, **_common)
    val_loader   = DataLoader(val_ds,  shuffle=False, **_common)
    test_loader  = DataLoader(test_ds, shuffle=False, **_common)

    logger.info(
        "DataLoaders ready — train: %d samples | val: %d | test: %d.",
        len(train_ds),
        len(val_ds),
        len(test_ds),
    )
    return train_loader, val_loader, test_loader


# ===========================================================================
# Internal: _FilteredTelemetryDataset
# ===========================================================================
class _FilteredTelemetryDataset(TelemetryDataset):
    """Internal variant that accepts pre-filtered per-trace file lists.

    This avoids re-scanning the full directory for every split and allows
    clean train/val/test separation at the file level.
    """

    def __init__(
        self,
        root_dir: Path,
        scaler: TraceScaler,
        seq_len: int,
        feature_columns: Optional[List[str]],
        file_paths_by_trace: dict[str, list[Path]],
    ) -> None:
        # Bypass TelemetryDataset.__init__ directory scan
        Dataset.__init__(self)
        self.root_dir = root_dir
        self.scaler = scaler
        self.seq_len = seq_len
        self.feature_columns = feature_columns

        self._file_records: list[tuple[Path, str]] = []
        for trace_name, fps in file_paths_by_trace.items():
            for fp in fps:
                self._file_records.append((fp, trace_name))

        self._samples: list[tuple[int, int]] = []
        self._file_row_counts: list[int] = []
        self._row_cache: dict[int, Optional[pd.DataFrame]] = {}

        for file_idx, (fp, _) in enumerate(self._file_records):
            n_rows = _count_csv_rows(fp)
            self._file_row_counts.append(n_rows)
            for start in range(max(0, n_rows - seq_len)):
                self._samples.append((file_idx, start))

        logger.debug(
            "_FilteredTelemetryDataset: %d files → %d windows.",
            len(self._file_records),
            len(self._samples),
        )


# ===========================================================================
# Private helpers
# ===========================================================================

def _check_trace(trace_name: str) -> None:
    """Raise ``ValueError`` for unknown trace names."""
    if trace_name not in TRACE_ID:
        raise ValueError(
            f"Unknown trace '{trace_name}'.  Expected one of {list(TRACE_ID)}."
        )


def _safe_read_csv(
    fp: Path,
    columns: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """Read a CSV, drop non-numeric columns, and return a clean DataFrame.

    Returns ``None`` on missing / corrupted files instead of raising.

    Parameters
    ----------
    fp:
        Path to the CSV file.
    columns:
        If provided, only these columns are kept (after numeric coercion).
    """
    if not fp.exists():
        logger.warning("File not found, skipping: %s", fp)
        return None
    try:
        df = pd.read_csv(fp, engine="c", low_memory=False)
    except Exception as exc:
        logger.warning("Could not read '%s': %s", fp, exc)
        return None

    # Drop non-numeric columns (e.g. timestamp strings)
    df = df.select_dtypes(include=[np.number])

    if columns is not None:
        present = [c for c in columns if c in df.columns]
        missing = set(columns) - set(present)
        if missing:
            logger.debug("Columns %s absent in %s — using available.", missing, fp.name)
        df = df[present] if present else df

    # Drop all-NaN columns then forward-fill remaining NaNs
    df = df.dropna(axis=1, how="all")
    df = df.ffill().bfill()

    if df.empty or df.shape[1] == 0:
        logger.warning("Empty DataFrame after cleaning: %s", fp)
        return None

    return df.reset_index(drop=True)


def _count_csv_rows(fp: Path) -> int:
    """Count data rows in a CSV cheaply (without loading values).

    Falls back to 0 on any error.
    """
    if not fp.exists():
        return 0
    try:
        # Read only the index column for speed
        df = pd.read_csv(fp, usecols=[0], engine="c", low_memory=False)
        return len(df)
    except Exception:
        return 0
