from fastapi import APIRouter, Depends, HTTPException, Query

from api.app.dependencies import get_forecast_service
from api.app.schemas import ForecastResponse
from ml.forecasting.forecast_service import ForecastService


router = APIRouter(
    prefix="/api",
    tags=["Forecast"],
)


@router.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    days: int = Query(default=30),
    service: ForecastService = Depends(get_forecast_service),
):
    if days not in {30, 60, 90}:
        raise HTTPException(
            status_code=422,
            detail="days must be one of: 30, 60, 90",
        )

    frame = service.forecast(days)

    return {
        "model": service.model_name,
        "generated_from": frame["GeneratedFrom"].iloc[0].date(),
        "horizon_days": days,
        "values": [
            {
                "date": row.Date.date(),
                "forecast": float(row.Forecast),
                "lower": float(row.Lower),
                "upper": float(row.Upper),
                "horizon_day": int(row.HorizonDay),
            }
            for row in frame.itertuples(index=False)
        ],
    }