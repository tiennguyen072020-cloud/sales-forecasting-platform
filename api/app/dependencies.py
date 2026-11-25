from fastapi import Request

from api.app.services.analytics_service import AnalyticsService
from ml.forecasting.forecast_service import ForecastService


def get_forecast_service(request: Request) -> ForecastService:
    return request.app.state.forecast_service


def get_analytics_service(request: Request) -> AnalyticsService:
    return request.app.state.analytics_service