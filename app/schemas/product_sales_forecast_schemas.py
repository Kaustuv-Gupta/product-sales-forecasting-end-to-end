from datetime import date
from pydantic import BaseModel, Field, create_model
from typing import Optional,Literal


from app.utils.logger import get_logger
logger = get_logger(__name__, log_file="app_product_sales_forecast_schemas")

DEFAULT_DATE_COL = "Date"
LAST_N_DAYS = 8


EXCLUDED_FEATURES = ['ID', 
                     'Store_Type', 
                    'Location_Type', 
                    'Region_Code',
                    'Date', 
                    'Discount',
                    'Holiday'
                    ]

PREDICT_DATE_COLS = ['year', 'month', 'day']
TARGET_COLS = ['Order', 'Sales']
ORIGINAL_TARGET_COLS = ['#Order', 'Sales']
PREDICT_OUTPUT_COLS = ['Store_id','Date',TARGET_COLS[0], TARGET_COLS[1],f'Pre_{TARGET_COLS[0]}', f'Pre_{TARGET_COLS[1]}']
RECUSSIVE_PREDICT_OUTPUT_COLS = ['Store_id','Date',f'Pre_{TARGET_COLS[0]}', f'Pre_{TARGET_COLS[1]}']
column_rename_mapping = {'#Order':f'Pre_{TARGET_COLS[0]}',TARGET_COLS[1]:f'Pre_{TARGET_COLS[1]}'}


dynamic_fields = {
    "Store_id": (int, ...),
    "Date": (date, ...),
    f"Pre_{TARGET_COLS[0]}": (float, ...),
    f"Pre_{TARGET_COLS[1]}": (float, ...)
}


class Product_Sales_Forecast_PayloadSchema(BaseModel):
    Store_id: int = Field(gt=0, description="Store ID (must be greater than 0)")
    Store_Type : Literal["S1", "S2", "S3", "S4"] = Field(description="Allowed Store Type (S1-S4)")
    Location_Type : Literal["L1", "L2", "L3", "L4"] = Field(description="Allowed location types (L1-L4)")
    Region_Code : Literal["R1", "R2", "R3", "R4"] = Field(description="Allowed Regional Code (R1-R4)")
    Prediction_Start_Date: date = Field(
        description="Start date for prediction in YYYY-MM-DD format",
        examples=["2024-12-31"]
    )
    period: int = Field(gt=0, description="Prediction period (must be greater than 0)")


RecursiveForecastOrderSalesResponseSchema=create_model(
    "Recursive_Forecast_Order_Sales", 
    **dynamic_fields
)
# class RecursiveForecastOrderSalesResponseSchema(BaseModel):
#     Store_id: int
#     Date : date
#     Pre_Order: float
#     Pre_Sales: float

