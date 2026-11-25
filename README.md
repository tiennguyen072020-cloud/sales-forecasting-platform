# Sales Forecasting & Business Intelligence Platform

A machine learning and business analytics project designed to analyze
historical sales data and forecast future revenue.

## Business Objective

The goal of this project is to forecast business revenue for:

- 30 days
- 60 days
- 90 days

The system combines:

- Data Analytics
- Time Series Analysis
- Machine Learning
- Forecasting
- Business Intelligence

## Machine Learning Models

The project will evaluate:

- Naive Forecast
- Moving Average
- Linear Regression
- Random Forest
- XGBoost
- Prophet

## Business Dashboard

The final Power BI dashboard will provide:

- Total Revenue
- Growth Rate
- Sales Trends
- Top Products
- Regional Performance
- Revenue Forecast
- Forecast Confidence

## Architecture

Historical Sales
→ Data Cleaning
→ Exploratory Data Analysis
→ Feature Engineering
→ Machine Learning
→ Forecasting
→ Business Intelligence Dashboard

## Requirements

- Python 3.12 or newer.

## Installation

Install runtime and development dependencies through the project manifest:

```powershell
python -m pip install -r requirements.txt
```

## Agent context refresh automation

The repository supports three refresh paths:

- VS Code: press `Ctrl+Shift+B` to run the default `Build and refresh agent context` task.
- CI: push and pull-request builds run the wheel build, refresh the handoff, and verify generated context files are committed.
- Git hooks: after installing development dependencies, run `pre-commit install` once. Each pre-commit run refreshes `AI agent Workspace/project-context.md` and its state baseline.

The refresh implementation is `scripts/update_agent_context.py`; it reads `pyproject.toml`, `README.md`, `ml/`, `api/`, and `build/lib/`.

## Running the API

From the project root in PowerShell, run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.app.main:app --reload --log-level info
```

The API documentation is available at `http://127.0.0.1:8000/docs`.

## Reusable machine-learning modules

Reusable forecasting code is under `ml/models/`:

- `baseline.py`: naive and recursive moving-average forecasts.
- `ml_models.py`: Linear Regression, Random Forest, and XGBoost pipelines; Random Forest and XGBoost tuning uses chronological `TimeSeriesSplit`.
- `prophet_model.py`: Prophet input preparation, model construction, fitting, and prediction.
- `evaluation.py`: chronological train/validation/test splitting, time-series CV fold summaries, and common regression metrics.

The notebooks remain the project workflow for experimentation and reporting. They can import these modules to avoid duplicating model logic. Build time-series features before splitting, keep rows chronological, and use only the training partition for hyperparameter tuning.