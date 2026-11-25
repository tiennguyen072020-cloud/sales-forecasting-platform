from datetime import date
from pydantic import BaseModel


class ForecastPoint(BaseModel):
    date: date
    forecast: float
    lower: float
    upper: float
    horizon_day: int


class ForecastResponse(BaseModel):
    model: str
    generated_from: date
    horizon_days: int
    values: list[ForecastPoint]


class RevenuePoint(BaseModel):
    date: date
    revenue: float


class ProductMetric(BaseModel):
    rank: int
    product_code: str
    product_name: str
    quantity: float
    revenue: float