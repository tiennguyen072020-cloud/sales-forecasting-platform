"""Chronological dataset splitting and common regression evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class TemporalDataSplit:
    """Chronological train, validation, and test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def summary(self, date_column: str = "date") -> pd.DataFrame:
        """Describe row counts and date ranges for each partition."""
        rows = []
        for name, frame in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            rows.append(
                {
                    "split": name,
                    "rows": len(frame),
                    "start": frame[date_column].min(),
                    "end": frame[date_column].max(),
                }
            )
        return pd.DataFrame(rows)


def chronological_split(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TemporalDataSplit:
    """Split rows into ordered train/validation/test partitions without shuffling."""
    if date_column not in frame.columns:
        raise KeyError(f"Missing date column: {date_column}")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1.")

    ordered = frame.copy()
    ordered[date_column] = pd.to_datetime(ordered[date_column], errors="raise")
    if ordered.empty:
        raise ValueError("Cannot split an empty time series.")
    if ordered[date_column].isna().any():
        raise ValueError("Dates must not contain missing values.")
    if ordered[date_column].duplicated().any():
        raise ValueError("Dates must be unique before splitting.")
    ordered = ordered.sort_values(date_column).reset_index(drop=True)

    row_count = len(ordered)
    train_end = int(row_count * train_fraction)
    validation_end = int(row_count * (train_fraction + validation_fraction))
    if train_end == 0 or validation_end <= train_end or validation_end >= row_count:
        raise ValueError(
            "The row count is too small for non-empty train, validation, and test splits."
        )

    return TemporalDataSplit(
        train=ordered.iloc[:train_end].copy(),
        validation=ordered.iloc[train_end:validation_end].copy(),
        test=ordered.iloc[validation_end:].copy(),
    )


def summarize_time_series_folds(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    n_splits: int = 5,
) -> pd.DataFrame:
    """Describe expanding time-series CV folds using their calendar bounds."""
    if date_column not in frame.columns:
        raise KeyError(f"Missing date column: {date_column}")
    if n_splits < 2 or len(frame) <= n_splits:
        raise ValueError("n_splits must be at least 2 and smaller than the row count.")

    dates = pd.to_datetime(frame[date_column], errors="raise").reset_index(drop=True)
    if dates.isna().any():
        raise ValueError("Dates must not contain missing values.")
    if dates.duplicated().any():
        raise ValueError("Dates must be unique before creating time-series folds.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Rows must be in chronological order before creating folds.")

    rows = []
    for fold, (train_indices, validation_indices) in enumerate(
        TimeSeriesSplit(n_splits=n_splits).split(frame),
        start=1,
    ):
        rows.append(
            {
                "fold": fold,
                "train_rows": len(train_indices),
                "train_start": dates.iloc[train_indices[0]],
                "train_end": dates.iloc[train_indices[-1]],
                "cv_rows": len(validation_indices),
                "cv_start": dates.iloc[validation_indices[0]],
                "cv_end": dates.iloc[validation_indices[-1]],
            }
        )
    return pd.DataFrame(rows)


def safe_mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Calculate MAPE on non-zero actuals; return NaN if all actuals are zero."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1 or actual.size != predicted.size:
        raise ValueError("y_true and y_pred must be one-dimensional and equal in length.")
    if actual.size == 0:
        raise ValueError("At least one observation is required.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("y_true and y_pred must contain only finite values.")

    non_zero_actuals = actual != 0
    if not non_zero_actuals.any():
        return float("nan")
    percentage_errors = np.abs(
        (actual[non_zero_actuals] - predicted[non_zero_actuals])
        / actual[non_zero_actuals]
    )
    return float(percentage_errors.mean() * 100)


def evaluate_regression(
    model_name: str,
    split_name: str,
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, str | float]:
    """Return the project's standard MAE, RMSE, MAPE, and R² metrics."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1 or actual.size != predicted.size:
        raise ValueError("y_true and y_pred must be one-dimensional and equal in length.")
    if actual.size == 0:
        raise ValueError("At least one observation is required.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("y_true and y_pred must contain only finite values.")

    return {
        "model": model_name,
        "split": split_name,
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE_pct": safe_mape(actual, predicted),
        "R2": float(r2_score(actual, predicted)),
    }


def compare_metrics(
    results: pd.DataFrame | Sequence[dict[str, str | float]],
    *,
    sort_by: Sequence[str] = ("split", "RMSE"),
) -> pd.DataFrame:
    """Sort model result rows by split and metric, with lower errors first."""
    comparison = pd.DataFrame(results).copy()
    required_columns = {"model", "split", "MAE", "RMSE", "MAPE_pct", "R2"}
    missing_columns = required_columns - set(comparison.columns)
    if missing_columns:
        raise ValueError(f"Metrics are missing columns: {sorted(missing_columns)}")
    missing_sort_columns = set(sort_by) - set(comparison.columns)
    if missing_sort_columns:
        raise ValueError(
            f"Sort columns are missing from metrics: {sorted(missing_sort_columns)}"
        )
    return comparison.sort_values(list(sort_by)).reset_index(drop=True)