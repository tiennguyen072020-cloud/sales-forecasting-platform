import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.app.routers import analytics, forecast
from api.app.services.analytics_service import AnalyticsService
from ml.forecasting.forecast_service import ForecastService


def resolve_project_root() -> Path:
    configured = os.getenv("SALES_FORECASTING_ROOT")

    if configured:
        return Path(configured).resolve()

    return Path(__file__).resolve().parents[2]


@asynccontextmanager
async def lifespan(app: FastAPI):
    project_root = resolve_project_root()

    app.state.forecast_service = ForecastService(project_root)
    app.state.analytics_service = AnalyticsService(project_root)

    yield


app = FastAPI(
    title="Sales Forecasting API",
    description="Forecast and analytics APIs for Phase 11.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(forecast.router)
app.include_router(analytics.router)


@app.get("/health")
def health():
    return {"status": "ok"}