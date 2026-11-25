"""Reusable scikit-learn pipelines and time-aware model tuning utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_RANDOM_STATE = 42


def build_linear_regression_pipeline() -> Pipeline:
    """Create the median-imputation, scaling, and linear-regression pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def build_random_forest_pipeline(
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Create the Random Forest pipeline used by the model-selection notebook."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Create an XGBoost pipeline, importing the optional estimator on demand."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost support requires the 'xgboost' package to be installed."
        ) from exc

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    tree_method="hist",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_regression_pipeline(
    model_name: str,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Build a supported model pipeline by its user-facing name."""
    normalized_name = model_name.strip().casefold().replace("_", " ").replace("-", " ")
    if normalized_name in {"linear", "linear regression"}:
        return build_linear_regression_pipeline()
    if normalized_name in {"random forest", "randomforest"}:
        return build_random_forest_pipeline(random_state)
    if normalized_name in {"xgboost", "xgb"}:
        return build_xgboost_pipeline(random_state)
    raise ValueError(
        "model_name must be one of: linear regression, random forest, xgboost."
    )


def _parameter_distributions(model_name: str) -> dict[str, Any]:
    """Return notebook-aligned randomized-search distributions."""
    normalized_name = model_name.strip().casefold().replace("_", " ").replace("-", " ")
    if normalized_name in {"random forest", "randomforest"}:
        return {
            "model__n_estimators": randint(200, 801),
            "model__max_depth": [None, 6, 10, 14, 20],
            "model__min_samples_split": randint(2, 15),
            "model__min_samples_leaf": randint(1, 8),
            "model__max_features": ["sqrt", "log2", 0.6, 1.0],
        }
    if normalized_name in {"xgboost", "xgb"}:
        return {
            "model__n_estimators": randint(200, 1001),
            "model__max_depth": randint(2, 9),
            "model__learning_rate": uniform(0.02, 0.18),
            "model__subsample": uniform(0.65, 0.35),
            "model__colsample_bytree": uniform(0.65, 0.35),
            "model__min_child_weight": randint(1, 12),
            "model__reg_alpha": uniform(0.0, 1.0),
            "model__reg_lambda": uniform(0.5, 2.5),
        }
    raise ValueError("Only random forest and xgboost support randomized tuning.")


def tune_regressor(
    model_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_iter: int | None = None,
    n_splits: int = 5,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = -1,
    verbose: int = 0,
) -> RandomizedSearchCV:
    """Tune Random Forest or XGBoost using expanding chronological CV folds.

    The provided rows must already be in chronological order. Test data must
    not be included; select hyperparameters with the training period only.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")
    if features.empty or len(features) != len(target):
        raise ValueError("features and target must have the same non-zero row count.")
    if n_splits < 2 or len(features) <= n_splits:
        raise ValueError("n_splits must be at least 2 and smaller than the row count.")
    if n_iter is not None and n_iter <= 0:
        raise ValueError("n_iter must be a positive integer.")

    search_iterations = n_iter
    if search_iterations is None:
        normalized_name = model_name.strip().casefold()
        search_iterations = 25 if normalized_name in {"random forest", "random_forest"} else 30

    # TimeSeriesSplit never shuffles: each validation fold follows its training fold.
    search = RandomizedSearchCV(
        estimator=build_regression_pipeline(model_name, random_state),
        param_distributions=_parameter_distributions(model_name),
        n_iter=search_iterations,
        scoring="neg_root_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=n_splits),
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        error_score="raise",
    )
    search.fit(features, target)
    return search


def fit_regressor(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
) -> BaseEstimator:
    """Fit a fresh clone of an estimator, leaving the caller's instance intact."""
    if features.empty or len(features) != len(target):
        raise ValueError("features and target must have the same non-zero row count.")

    fitted_estimator = clone(estimator)
    fitted_estimator.fit(features, target)
    return fitted_estimator