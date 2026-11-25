"""Leakage-safe 30/60/90-day recursive forecasting service."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.features.time_features import TimeFeatureConfig, build_time_features


MODEL_FILES = {
    "prophet": "prophet_final.joblib",
    "xgboost": "xgboost_final.joblib",
    "random forest": "random_forest_final.joblib",
    "linear regression": "linear_regression_baseline.joblib",
}


class ForecastService:
    """Load the Phase 10 winner and generate future-only forecasts."""

    def __init__(self, project_root: str | Path):
        self.root = Path(project_root).resolve()
        self.model_dir = self.root / "artifacts" / "models"
        self.report_dir = self.root / "artifacts" / "reports"
        self.evaluation_dir = self.root / "artifacts" / "evaluation"

        decision_path = self.evaluation_dir / "final_model_decision.json"
        feature_metadata_path = self.root / "data" / "features" / "sales_time_features_metadata.json"
        history_path = self.root / "data" / "features" / "sales_time_features.csv"

        self.decision = json.loads(decision_path.read_text(encoding="utf-8"))
        self.feature_metadata = json.loads(feature_metadata_path.read_text(encoding="utf-8"))
        self.model_name = str(self.decision["selected_model"])
        self.model_key = self.model_name.casefold()
        self.features = list(self.feature_metadata["feature_columns"])
        self.target = str(self.feature_metadata["target_column"])

        history = pd.read_csv(history_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        self.feature_frame = history.copy()
        self.history = history[["date", self.target]].copy()
        if not self.history["date"].is_unique:
            raise ValueError("Historical dates must be unique.")

        self.residual_low, self.residual_high = self._load_residual_interval()
        self.model = self._load_model_if_needed()

    def _load_model_if_needed(self):
        if "naive" in self.model_key or "moving average" in self.model_key:
            return None
        matched = next((key for key in MODEL_FILES if key in self.model_key), None)
        if matched is None:
            raise ValueError(f"Unsupported selected model: {self.model_name}")
        path = self.model_dir / MODEL_FILES[matched]
        # Load artifact
        if not path.exists():
            raise FileNotFoundError(f"Selected model artifact not found: {path}")
        model = joblib.load(path)
        if matched == "prophet":
            return model
        # use every observed row, including the former holdout period.
        model.fit(self.feature_frame[self.features], 
                  self.feature_frame[self.target])
        return model

    def _load_residual_interval(self) -> tuple[float, float]:
        path = self.evaluation_dir / "selected_model_test_predictions.csv"
        if not path.exists():
            return 0.0, 0.0
        frame = pd.read_csv(path)
        if "residual" not in frame:
            if {"actual", "prediction"}.issubset(frame.columns):
                frame["residual"] = frame["actual"] - frame["prediction"]
            else:
                return 0.0, 0.0
        residuals = pd.to_numeric(frame["residual"], errors="coerce").dropna()
        if residuals.empty:
            return 0.0, 0.0
        return float(residuals.quantile(0.10)), float(residuals.quantile(0.90))

    def forecast(self, days: int) -> pd.DataFrame:
        if days not in {30, 60, 90}:
            raise ValueError("days must be one of: 30, 60, 90")
        if "prophet" in self.model_key:
            return self._forecast_prophet(days)
        return self._forecast_recursive(days)

    def _forecast_prophet(self, days: int) -> pd.DataFrame:
        start = self.history["date"].max() + pd.Timedelta(days=1)
        future = pd.DataFrame({"ds": pd.date_range(start, periods=days, freq="D")})
        forecast = self.model.predict(future)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
            columns={"ds": "Date", "yhat": "Forecast", "yhat_lower": "Lower", "yhat_upper": "Upper"}
        )
        return self._finish(result)

    def _forecast_recursive(self, days: int) -> pd.DataFrame:
        history = self.history.copy()
        rows: list[dict[str, float | pd.Timestamp]] = []
        config = TimeFeatureConfig(
            date_column="date",
            target_column=self.target,
            lags=tuple(self.feature_metadata["lags"]),
            rolling_windows=tuple(self.feature_metadata["rolling_windows"]),
            expanding_min_periods=int(self.feature_metadata["expanding_min_periods"]),
            add_cyclical_features=True,
        )

        for _ in range(days):
            future_date = history["date"].max() + pd.Timedelta(days=1)
            if "naive" in self.model_key:
                prediction = float(history[self.target].iloc[-1])
            elif "moving average" in self.model_key:
                digits = "".join(character for character in self.model_name if character.isdigit())
                window = int(digits) if digits else 7
                prediction = float(history[self.target].tail(window).mean())
            else:
                # Placeholder target is never used by current-row features because
                # lag, rolling, and expanding statistics all depend on shifted history.
                candidate = pd.concat([
                    history,
                    pd.DataFrame({"date": [future_date], self.target: [0.0]}),
                ], ignore_index=True)
                engineered = build_time_features(candidate, config, drop_incomplete_rows=False)
                model_row = engineered.iloc[[-1]][self.features]
                prediction = float(self.model.predict(model_row)[0])

            rows.append({
                "Date": future_date,
                "Forecast": prediction,
                "Lower": prediction + self.residual_low,
                "Upper": prediction + self.residual_high,
            })
            history = pd.concat([
                history,
                pd.DataFrame({"date": [future_date], self.target: [prediction]}),
            ], ignore_index=True)

        return self._finish(pd.DataFrame(rows))

    def _finish(self, result: pd.DataFrame) -> pd.DataFrame:
        result = result.copy()
        result["HorizonDay"] = np.arange(1, len(result) + 1)
        result["Model"] = self.model_name
        result["GeneratedFrom"] = self.history["date"].max()
        if not (result["Date"] > self.history["date"].max()).all():
            raise AssertionError("Forecast output contains historical dates.")
        return result[["Date", "Forecast", "Lower", "Upper", "HorizonDay", "Model", "GeneratedFrom"]]

    def export(self, output_dir: str | Path | None = None) -> dict[int, Path]:
        destination = Path(output_dir) if output_dir else self.root / "artifacts" / "forecasts"
        destination.mkdir(parents=True, exist_ok=True)
        forecast_90 = self.forecast(90)
        paths: dict[int, Path] = {}
        for horizon in (30, 60, 90):
            path = destination / f"forecast_{horizon}_days.csv"
            forecast_90.iloc[:horizon].to_csv(path, index=False)
            paths[horizon] = path
        return paths
