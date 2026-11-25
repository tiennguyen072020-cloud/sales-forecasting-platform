"""Reusable Prophet model construction and input preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


DEFAULT_PROPHET_PARAMETERS: dict[str, float | str] = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "holidays_prior_scale": 10.0,
    "seasonality_mode": "additive",
}


def prepare_prophet_frame(
    frame: pd.DataFrame,
    date_column: str = "date",
    target_column: str = "net_revenue",
) -> pd.DataFrame:
    """Validate and convert a chronological target series to Prophet's ds/y schema."""
    missing_columns = [
        column for column in (date_column, target_column) if column not in frame.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    prepared = frame[[date_column, target_column]].copy()
    prepared[date_column] = pd.to_datetime(prepared[date_column], errors="raise")
    prepared[target_column] = pd.to_numeric(prepared[target_column], errors="raise")
    prepared = prepared.rename(columns={date_column: "ds", target_column: "y"})
    prepared = prepared.sort_values("ds").reset_index(drop=True)

    if prepared.empty:
        raise ValueError("Prophet training data must not be empty.")
    if prepared["ds"].isna().any() or prepared["y"].isna().any():
        raise ValueError("Prophet dates and target values must not be missing.")
    if prepared["ds"].duplicated().any():
        raise ValueError("Prophet training data must contain one row per date.")
    if not np.isfinite(prepared["y"].to_numpy(dtype=float)).all():
        raise ValueError("Prophet target values must be finite.")

    return prepared


def build_prophet_model(
    parameters: Mapping[str, Any] | None = None,
    *,
    country_holidays: str | None = "GB",
    interval_width: float = 0.8,
    uncertainty_samples: int = 1000,
):
    """Build Prophet with the project's weekly, yearly, and holiday effects.

    Country holidays default to Great Britain because the analyzed retail data
    is UK-based. Pass ``None`` to disable country-specific holidays.
    """
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "Prophet support requires the 'prophet' package to be installed."
        ) from exc

    model_parameters = DEFAULT_PROPHET_PARAMETERS.copy()
    if parameters is not None:
        unknown_parameters = set(parameters) - set(model_parameters)
        if unknown_parameters:
            raise ValueError(
                f"Unsupported Prophet parameters: {sorted(unknown_parameters)}"
            )
        model_parameters.update(parameters)

    if model_parameters["seasonality_mode"] not in {"additive", "multiplicative"}:
        raise ValueError("seasonality_mode must be 'additive' or 'multiplicative'.")
    if not 0 < interval_width < 1:
        raise ValueError("interval_width must be between 0 and 1.")
    if uncertainty_samples < 0:
        raise ValueError("uncertainty_samples must not be negative.")

    model = Prophet(
        growth="linear",
        weekly_seasonality=False,
        yearly_seasonality=False,
        daily_seasonality=False,
        interval_width=interval_width,
        uncertainty_samples=uncertainty_samples,
        **model_parameters,
    )
    model.add_seasonality(name="weekly", period=7, fourier_order=3)
    model.add_seasonality(name="yearly", period=365.25, fourier_order=10)
    if country_holidays is not None:
        model.add_country_holidays(country_name=country_holidays)
    return model


def fit_prophet_model(
    frame: pd.DataFrame,
    parameters: Mapping[str, Any] | None = None,
    *,
    date_column: str = "date",
    target_column: str = "net_revenue",
    country_holidays: str | None = "GB",
    interval_width: float = 0.8,
    uncertainty_samples: int = 1000,
):
    """Prepare a target series and fit a reusable Prophet instance."""
    training_frame = prepare_prophet_frame(frame, date_column, target_column)
    model = build_prophet_model(
        parameters,
        country_holidays=country_holidays,
        interval_width=interval_width,
        uncertainty_samples=uncertainty_samples,
    )
    model.fit(training_frame)
    return model


def predict_prophet(
    model,
    dates: Sequence[pd.Timestamp | str] | pd.Series,
    *,
    components: Sequence[str] = (),
) -> pd.DataFrame:
    """Return Prophet forecasts and optional named components for supplied dates."""
    prediction_dates = pd.to_datetime(pd.Series(dates), errors="raise")
    if prediction_dates.empty:
        raise ValueError("At least one prediction date is required.")
    if prediction_dates.isna().any():
        raise ValueError("Prediction dates must not be missing.")

    # Prophet consumes only the date column at prediction time.
    predictions = model.predict(pd.DataFrame({"ds": prediction_dates}))
    forecast_columns = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    required_columns = set(forecast_columns)
    requested_components = list(components)
    if len(requested_components) != len(set(requested_components)):
        raise ValueError("components must not contain duplicate column names.")
    required_columns.update(requested_components)
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"Prophet output is missing columns: {sorted(missing_columns)}"
        )
    return predictions[forecast_columns + requested_components].copy()


def tune_prophet(
    frame: pd.DataFrame,
    parameter_grid: Sequence[Mapping[str, Any]],
    *,
    date_column: str = "ds",
    target_column: str = "y",
    n_splits: int = 3,
    country_holidays: str | None = "GB",
    interval_width: float = 0.8,
    uncertainty_samples: int = 1000,
) -> pd.DataFrame:
    """Rank Prophet parameter sets using expanding chronological CV and RMSE.

    Each fold fits only on dates preceding its validation block. The input is
    normalized to Prophet's ``ds``/``y`` columns before folds are constructed.
    """
    training_frame = prepare_prophet_frame(frame, date_column, target_column)
    if n_splits < 2 or len(training_frame) <= n_splits:
        raise ValueError("n_splits must be at least 2 and smaller than the row count.")
    if not parameter_grid:
        raise ValueError("parameter_grid must contain at least one parameter set.")

    time_series_split = TimeSeriesSplit(n_splits=n_splits)
    results: list[dict[str, Any]] = []
    for parameters in parameter_grid:
        fold_rmse: list[float] = []
        for fit_indices, validation_indices in time_series_split.split(training_frame):
            fold_train = training_frame.iloc[fit_indices]
            fold_validation = training_frame.iloc[validation_indices]
            model = fit_prophet_model(
                fold_train,
                parameters,
                date_column="ds",
                target_column="y",
                country_holidays=country_holidays,
                interval_width=interval_width,
                uncertainty_samples=uncertainty_samples,
            )
            forecast = predict_prophet(model, fold_validation["ds"])
            rmse = float(
                np.sqrt(
                    mean_squared_error(
                        fold_validation["y"],
                        forecast["yhat"],
                    )
                )
            )
            fold_rmse.append(rmse)

        results.append(
            {
                **parameters,
                "mean_cv_RMSE": float(np.mean(fold_rmse)),
                "std_cv_RMSE": float(np.std(fold_rmse)),
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("mean_cv_RMSE")
        .reset_index(drop=True)
    )