from typing import Literal

from fastapi import APIRouter, Depends, Query

from api.app.dependencies import get_analytics_service
from api.app.services.analytics_service import AnalyticsService


router = APIRouter(
    prefix="/api/analytics",
    tags=["Analytics"],
)


@router.get("/revenue")
def get_revenue(
    service: AnalyticsService = Depends(get_analytics_service),
):
    return service.revenue()


@router.get("/growth")
def get_growth(
    frequency: Literal["month", "quarter", "year"] = "month",
    service: AnalyticsService = Depends(get_analytics_service),
):
    return service.growth(frequency)


@router.get("/top-products")
def get_top_products(
    limit: int = Query(default=10, ge=1, le=100),
    service: AnalyticsService = Depends(get_analytics_service),
):
    return service.top_products(limit)


@router.get("/categories")
def get_categories(
    service: AnalyticsService = Depends(get_analytics_service),
):
    return service.categories()