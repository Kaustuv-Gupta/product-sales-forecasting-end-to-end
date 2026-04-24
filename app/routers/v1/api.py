from fastapi import APIRouter
from .endpoints import product_sales_forecast_router

from app.core.config import settings

from app.utils.logger import get_logger
logger = get_logger(__name__, log_file="app_api")

logger.debug("Include prediction Router")

api_router = APIRouter(prefix=settings.api_base_path)

logger.debug("Including Product Sales Forecast Router")
api_router.include_router(product_sales_forecast_router.router,prefix="/forecast", tags=["Forecast APIs"])


# @api_router.get("/ping")
# def ping():
#     return {
#         "status": "ok",
#         "message": "Forecast API is working"
#     }