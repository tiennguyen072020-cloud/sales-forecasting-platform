import numpy as np
import pandas as pd
import pytest

import ml.models.ml_models as ml_models
from ml.models.baseline import (
    moving_average_forecast,
    naive_forecast,
)
from ml.models.evaluation import (
    chronological_split,
    compare_metrics,
    evaluate_regression,
    safe_mape,
    summarize_time_series_folds,
)
from ml.models.ml_models import (
    build_linear_regression_pipeline,
    build_random_forest_pipeline,
    fit_regressor,
    tune_regressor,
)
from ml.models.prophet_model import (
    build_prophet_model,
    prepare_prophet_frame,
    predict_prophet,
    tune_prophet,
)


def test_naive_forecast_repeats_last_observed_value():
    assert naive_forecast([4.0, 5.0, 6.0], periods=3).tolist() == [6.0] * 3


def test_moving_average_forecast_uses_prior_predictions_recursively():
    predictions = moving_average_forecast([1.0, 2.0, 3.0], window=2, periods=2)

    assert predictions.tolist() == [2.5, 2.75]


@pytest.mark.parametrize(
    ("history", "window", "periods"),
    [([], 2, 1), ([1.0], 0, 1), ([1.0], 1, 0), ([np.inf], 1, 1)],
)
def test_baseline_forecasts_reject_invalid_inputs(history, window, periods):
    with pytest.raises(ValueError):
        moving_average_forecast(history, window=window, periods=periods)


def test_chronological_split_sorts_dates_without_shuffling_rows():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=20).take(
                [5, 2, 4, 0, 3, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            ),
            "target": range(20),
        }
    )

    split = chronological_split(frame)

    assert len(split.train) == 14
    assert len(split.validation) == 3
    assert len(split.test) == 3
    assert split.train["date"].max() < split.validation["date"].min()
    assert split.validation["date"].max() < split.test["date"].min()
    assert split.summary().columns.tolist() == ["split", "rows", "start", "end"]


def test_chronological_split_rejects_duplicate_dates():
    frame = pd.DataFrame(
        {"date": pd.to_datetime(["2026-01-01"] * 6), "target": range(6)}
    )

    with pytest.raises(ValueError, match="unique"):
        chronological_split(frame)


def test_summarize_time_series_folds_reports_chronological_date_bounds():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=12),
            "target": np.arange(12),
        }
    )

    summary = summarize_time_series_folds(frame, n_splits=3)

    assert summary["fold"].tolist() == [1, 2, 3]
    assert summary["train_rows"].tolist() == [3, 6, 9]
    assert summary["cv_rows"].tolist() == [3, 3, 3]
    assert (summary["train_end"] < summary["cv_start"]).all()
    assert summary.loc[0, "train_rows"] < summary.loc[1, "train_rows"]


def test_summarize_time_series_folds_rejects_unsorted_dates():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-01", "2026-01-03"]),
            "target": [2, 1, 3],
        }
    )

    with pytest.raises(ValueError, match="chronological"):
        summarize_time_series_folds(frame, n_splits=2)


def test_regression_metrics_match_project_metric_names_and_values():
    result = evaluate_regression(
        "test model",
        "validation",
        y_true=[10.0, 0.0, 20.0],
        y_pred=[12.0, 5.0, 18.0],
    )

    assert result["model"] == "test model"
    assert result["split"] == "validation"
    assert result["MAE"] == pytest.approx(3.0)
    assert result["RMSE"] == pytest.approx(np.sqrt(11.0))
    assert result["MAPE_pct"] == pytest.approx(15.0)
    assert np.isfinite(result["R2"])
    assert np.isnan(safe_mape([0.0, 0.0], [1.0, 2.0]))


def test_compare_metrics_sorts_by_split_then_rmse():
    rows = [
        {"model": "slow", "split": "test", "MAE": 2, "RMSE": 3, "MAPE_pct": 2, "R2": 0},
        {"model": "fast", "split": "test", "MAE": 1, "RMSE": 2, "MAPE_pct": 1, "R2": 0},
    ]

    assert compare_metrics(rows)["model"].tolist() == ["fast", "slow"]


@pytest.mark.parametrize(
    "pipeline",
    [build_linear_regression_pipeline(), build_random_forest_pipeline()],
)
def test_regression_pipeline_fits_and_returns_a_clone(pipeline):
    features = pd.DataFrame({"feature": [1.0, np.nan, 3.0, 4.0]})
    target = pd.Series([2.0, 4.0, 6.0, 8.0])

    fitted = fit_regressor(pipeline, features, target)

    assert fitted is not pipeline
    assert np.isfinite(fitted.predict(features)).all()


def test_tuner_requires_enough_rows_for_time_series_folds():
    features = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    target = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="row count"):
        tune_regressor("random forest", features, target, n_splits=3)


def test_tuner_runs_expanding_time_series_search(monkeypatch):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    # Use tiny trees and a two-value grid so the test covers the real search
    # mechanics without paying the notebook's production tuning cost.
    monkeypatch.setattr(
        ml_models,
        "build_regression_pipeline",
        lambda model_name, random_state: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=2,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    )
    monkeypatch.setattr(
        ml_models,
        "_parameter_distributions",
        lambda model_name: {"model__max_depth": [1, 2]},
    )
    features = pd.DataFrame({"feature": np.arange(12, dtype=float)})
    target = pd.Series(np.arange(12, dtype=float) * 2)

    search = tune_regressor(
        "random forest",
        features,
        target,
        n_iter=2,
        n_splits=2,
        n_jobs=1,
    )

    first_train, first_validation = next(search.cv.split(features))
    assert first_train.max() < first_validation.min()
    assert search.best_estimator_ is not None


def test_prepare_prophet_frame_sorts_and_renames_inputs():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-01"]),
            "revenue": [20.0, 10.0],
        }
    )

    result = prepare_prophet_frame(frame, target_column="revenue")

    assert result.columns.tolist() == ["ds", "y"]
    assert result["ds"].is_monotonic_increasing
    assert result["y"].tolist() == [10.0, 20.0]


def test_prophet_model_uses_project_seasonality_configuration():
    model = build_prophet_model(
        country_holidays=None,
        uncertainty_samples=0,
    )

    assert model.seasonalities["weekly"]["period"] == 7
    assert model.seasonalities["yearly"]["period"] == 365.25


def test_predict_prophet_can_include_requested_components():
    class FakeModel:
        def predict(self, future):
            return pd.DataFrame(
                {
                    "ds": future["ds"],
                    "yhat": [1.0],
                    "yhat_lower": [0.0],
                    "yhat_upper": [2.0],
                    "trend": [1.0],
                    "weekly": [0.1],
                }
            )

    result = predict_prophet(
        FakeModel(),
        ["2026-01-01"],
        components=["trend", "weekly"],
    )

    assert result.columns.tolist() == [
        "ds",
        "yhat",
        "yhat_lower",
        "yhat_upper",
        "trend",
        "weekly",
    ]


def test_tune_prophet_uses_chronological_folds_and_ranks_parameters(monkeypatch):
    import ml.models.prophet_model as prophet_model

    fitted_dates = []

    class FakeModel:
        pass

    def fake_fit(frame, parameters, **kwargs):
        fitted_dates.append(frame["ds"].max())
        return FakeModel()

    def fake_predict(model, dates):
        dates = pd.to_datetime(pd.Series(dates))
        return pd.DataFrame(
            {
                "ds": dates,
                "yhat": np.zeros(len(dates)),
                "yhat_lower": np.zeros(len(dates)),
                "yhat_upper": np.zeros(len(dates)),
            }
        )

    monkeypatch.setattr(prophet_model, "fit_prophet_model", fake_fit)
    monkeypatch.setattr(prophet_model, "predict_prophet", fake_predict)
    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=12),
            "y": np.arange(12, dtype=float),
        }
    )

    results = tune_prophet(
        frame,
        [{"changepoint_prior_scale": 0.01}, {"changepoint_prior_scale": 0.05}],
        n_splits=3,
    )

    assert len(results) == 2
    assert results["mean_cv_RMSE"].is_monotonic_increasing
    assert len(fitted_dates) == 6
    assert all(fit_end < validation_start for fit_end, validation_start in zip(
        fitted_dates[:3],
        [frame["ds"].iloc[3], frame["ds"].iloc[6], frame["ds"].iloc[9]],
    ))
