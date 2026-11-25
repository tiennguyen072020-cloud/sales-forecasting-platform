"""Leakage-safe time-series feature engineering for sales forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_LAGS: tuple[int, ...] = (1, 7, 14, 28, 30, 60, 90)
DEFAULT_WINDOWS: tuple[int, ...] = (7, 14, 30)


@dataclass(frozen=True)
class TimeFeatureConfig:
    date_column: str = "date"
    target_column: str = "net_revenue"
    lags: tuple[int, ...] = DEFAULT_LAGS
    rolling_windows: tuple[int, ...] = DEFAULT_WINDOWS
    expanding_min_periods: int = 7
    add_cyclical_features: bool = True


def _positive_unique(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(sorted(set(int(value) for value in values)))
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers.")
    return result


def validate_daily_series(
    df: pd.DataFrame,
    date_column: str = "date",
    target_column: str = "net_revenue",
) -> pd.DataFrame:
    """Return a sorted daily series and fail fast on invalid input.

    Phase 5 should already have decided how missing calendar dates are handled.
    This function deliberately does not fill missing target values because doing
    so here could hide data-quality problems or introduce leakage.
    """
    missing = [column for column in (date_column, target_column) if column not in df]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    result = df.copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="raise")
    result[target_column] = pd.to_numeric(result[target_column], errors="raise")
    result = result.sort_values(date_column).reset_index(drop=True)

    if result[date_column].isna().any() or result[target_column].isna().any():
        raise ValueError("Date and target columns must not contain missing values.")
    if result[date_column].duplicated().any():
        duplicates = result.loc[result[date_column].duplicated(), date_column].head(5)
        raise ValueError(f"Each date must be unique. Duplicate examples: {duplicates.tolist()}")
    if not result[date_column].is_monotonic_increasing:
        raise ValueError("Dates must be sortable in ascending order.")

    return result


def add_calendar_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Add deterministic features known at prediction time."""
    result = df.copy()
    dates = result[date_column].dt

    result["day_of_week"] = dates.dayofweek
    result["day_of_month"] = dates.day
    result["day_of_year"] = dates.dayofyear
    result["week_of_year"] = dates.isocalendar().week.astype("int16")
    result["month"] = dates.month
    result["quarter"] = dates.quarter
    result["year"] = dates.year
    result["is_weekend"] = (dates.dayofweek >= 5).astype("int8")
    result["is_month_start"] = dates.is_month_start.astype("int8")
    result["is_month_end"] = dates.is_month_end.astype("int8")
    result["is_quarter_start"] = dates.is_quarter_start.astype("int8")
    result["is_quarter_end"] = dates.is_quarter_end.astype("int8")
    return result


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode repeating calendar cycles without an artificial boundary."""
    result = df.copy()
    cycles = {
        "day_of_week": 7,
        "month": 12,
        "quarter": 4,
        "day_of_year": 365.25,
    }
    for column, period in cycles.items():
        result[f"{column}_sin"] = np.sin(2 * np.pi * result[column] / period)
        result[f"{column}_cos"] = np.cos(2 * np.pi * result[column] / period)
    return result


def add_lag_features(
    df: pd.DataFrame,
    target_column: str = "net_revenue",
    lags: Sequence[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    result = df.copy()
    for lag in _positive_unique(lags, "lags"):
        result[f"{target_column}_lag_{lag}"] = result[target_column].shift(lag)
    return result


def add_rolling_features(
    df: pd.DataFrame,
    target_column: str = "net_revenue",
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Add past-only rolling statistics.

    shift(1) is essential: the feature for date t is computed only from dates
    <= t-1, never from the target being predicted on date t.
    """
    result = df.copy()
    past_target = result[target_column].shift(1)
    for window in _positive_unique(windows, "windows"):
        rolling = past_target.rolling(window=window, min_periods=window)
        result[f"{target_column}_rolling_mean_{window}"] = rolling.mean()
        result[f"{target_column}_rolling_std_{window}"] = rolling.std(ddof=0)
        result[f"{target_column}_rolling_min_{window}"] = rolling.min()
        result[f"{target_column}_rolling_max_{window}"] = rolling.max()
    return result


def add_expanding_features(
    df: pd.DataFrame,
    target_column: str = "net_revenue",
    min_periods: int = 7,
) -> pd.DataFrame:
    """Add expanding statistics using only observations before the row."""
    if min_periods <= 0:
        raise ValueError("min_periods must be positive.")
    result = df.copy()
    expanding = result[target_column].shift(1).expanding(min_periods=min_periods)
    result[f"{target_column}_expanding_mean"] = expanding.mean()
    result[f"{target_column}_expanding_std"] = expanding.std(ddof=0)
    result[f"{target_column}_expanding_min"] = expanding.min()
    result[f"{target_column}_expanding_max"] = expanding.max()
    return result


def build_time_features(
    df: pd.DataFrame,
    config: TimeFeatureConfig | None = None,
    drop_incomplete_rows: bool = True,
) -> pd.DataFrame:
    """Build the shared model-ready dataset in chronological order."""
    config = config or TimeFeatureConfig()
    result = validate_daily_series(df, config.date_column, config.target_column)
    result = add_calendar_features(result, config.date_column)
    if config.add_cyclical_features:
        result = add_cyclical_features(result)
    result = add_lag_features(result, config.target_column, config.lags)
    result = add_rolling_features(result, config.target_column, config.rolling_windows)
    result = add_expanding_features(
        result, config.target_column, config.expanding_min_periods
    )

    if drop_incomplete_rows:
        engineered_columns = [
            column
            for column in result.columns
            if column not in df.columns or "_lag_" in column
        ]
        result = result.dropna(subset=engineered_columns).reset_index(drop=True)
    return result


def get_model_columns(
    feature_df: pd.DataFrame,
    date_column: str = "date",
    target_column: str = "net_revenue",
    excluded_columns: Sequence[str] = (),
) -> tuple[list[str], str]:
    """Return a stable feature-column list and the target name."""
    excluded = {date_column, target_column, *excluded_columns}
    feature_columns = [column for column in feature_df.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("No model features are available.")
    return feature_columns, target_column