"""Simple forecasting baselines used to benchmark learned models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validated_history(history: Sequence[float]) -> np.ndarray:
    """Convert historical targets to a finite, non-empty numeric array."""
    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("history must be a non-empty one-dimensional sequence.")
    if not np.isfinite(values).all():
        raise ValueError("history must contain only finite values.")
    return values


def naive_forecast(history: Sequence[float], periods: int = 1) -> np.ndarray:
    """Repeat the latest observed value for each requested future period."""
    values = _validated_history(history)
    if periods <= 0:
        raise ValueError("periods must be a positive integer.")

    return np.full(periods, values[-1], dtype=float)


def moving_average_forecast(
    history: Sequence[float],
    window: int = 7,
    periods: int = 1,
) -> np.ndarray:
    """Forecast recursively from a trailing moving average.

    Each forecast is appended to the available history before predicting the
    next period. This matches recursive multi-step forecasting and prevents
    future actual values from leaking into a forecast.
    """
    values = _validated_history(history)
    if window <= 0:
        raise ValueError("window must be a positive integer.")
    if periods <= 0:
        raise ValueError("periods must be a positive integer.")

    rolling_history = values.tolist()
    predictions = np.empty(periods, dtype=float)
    for index in range(periods):
        prediction = float(np.mean(rolling_history[-window:]))
        predictions[index] = prediction
        rolling_history.append(prediction)

    return predictions