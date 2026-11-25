---
name: sales-forecasting-workflow
description: "Use when changing or investigating sales data processing, time-series features, model training or evaluation, 30/60/90-day forecasts, or the forecasting API in this project."
---

# Sales Forecasting Workflow

Use this workflow for feature work, debugging, and tests across the sales forecasting pipeline.

## Procedure

1. Identify the affected layer before editing:
   - Data ingestion and cleaning: `ml/data/`
   - Time features: `ml/features/`
   - Models and evaluation: `ml/models/`
   - Forecast generation: `ml/forecasting/`
   - HTTP API: `api/app/`
   - Exploratory and business analysis: `notebooks/`
2. Read the relevant implementation, callers, tests, and metadata or schema files. Trace inputs and outputs through adjacent layers before changing a shared contract.
3. For time-series work:
   - Preserve date ordering and the existing daily forecasting contract.
   - Use chronological validation; do not randomly shuffle temporal observations.
   - Ensure features for a prediction use only information available before or at that prediction time.
   - For recursive forecasts, feed each prediction into the next step only where that matches the existing design.
   - Preserve supported horizons of 30, 60, and 90 days unless the request explicitly changes them.
4. Make the smallest complete change that follows nearby code patterns. Do not modify raw data, model artifacts, or notebook outputs unless the task requires it.
5. Add or adjust tests for the changed behavior. Cover edge cases such as missing inputs, invalid horizons, date boundaries, and output shape when relevant.
6. Run focused tests with `python -m pytest <relevant-test-file>`. If tests cannot run, report the exact blocker instead of claiming success.
7. Summarize changed files, key behavior, validation performed, and any remaining limitation.

## Output expectations

- Forecast results must remain future-only and maintain the established response or dataframe schema.
- Errors for invalid inputs or missing required artifacts must be explicit; do not silently return plausible-looking forecasts.
- Avoid claiming model quality or business impact without measured evaluation results.
