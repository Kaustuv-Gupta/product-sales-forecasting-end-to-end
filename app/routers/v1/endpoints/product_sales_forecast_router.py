from fastapi import APIRouter, HTTPException, status,Request
from app.schemas.product_sales_forecast_schemas import *
from app.features.order_sales_forecast.service import get_recursive_forecast
from typing import List

from app.utils.logger import get_logger
logger = get_logger(__name__, log_file="app_main")


router=APIRouter()

@router.post("/recursive_order_sales_forecast",response_model=List[RecursiveForecastOrderSalesResponseSchema],status_code=status.HTTP_200_OK)
async def recursive_forecast_order_sales(
    payload: Product_Sales_Forecast_PayloadSchema,
    request: Request
):
    try:

        if not payload.Store_id or str(payload.Store_id).strip() =="":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                details = "Strore ID is required for prediction."
            )
        
        return await get_recursive_forecast(payload, request)

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"forecast_order_sales: An unexpected error occured while forecasting orders and sales: {str(e)}"
        )
    