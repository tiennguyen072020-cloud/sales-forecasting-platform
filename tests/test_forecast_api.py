from datetime import timedelta

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.app.dependencies import get_forecast_service
from api.app.routers.forecast import router


class StubForecastService:
    model_name = "test model"

    def __init__(self):
        self.requested_days = None

    def forecast(self, days: int) -> pd.DataFrame:
        self.requested_days = days
        generated_from = pd.Timestamp("2026-01-01")
        return pd.DataFrame(
            {
                "Date": [
                    generated_from + timedelta(days=day)
                    for day in range(1, days + 1)
                ],
                "Forecast": [100.0] * days,
                "Lower": [90.0] * days,
                "Upper": [110.0] * days,
                "HorizonDay": range(1, days + 1),
                "GeneratedFrom": [generated_from] * days,
            }
        )


@pytest.fixture
def client_and_service():
    service = StubForecastService()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_forecast_service] = lambda: service

    with TestClient(app) as client:
        yield client, service


@pytest.mark.parametrize("days", [30, 60, 90])
def test_forecast_accepts_supported_query_horizons(client_and_service, days):
    client, service = client_and_service

    response = client.get("/api/forecast", params={"days": days})

    assert response.status_code == 200
    assert response.json()["horizon_days"] == days
    assert len(response.json()["values"]) == days
    assert service.requested_days == days


def test_forecast_rejects_unsupported_query_horizon(client_and_service):
    client, _ = client_and_service

    response = client.get("/api/forecast", params={"days": 45})

    assert response.status_code == 422
    assert response.json()["detail"] == "days must be one of: 30, 60, 90"
