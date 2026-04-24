import pandas as pd
from typing import Optional
#from pydantic import BaseModel, field_validator



# class DataRecord(BaseModel):
#     Date: pd.Timestamp
#     Store_id: int
#     Sales: float
#     Order: float

#     Store_Type: Optional[str] = None
#     Location_Type: Optional[str] = None
#     Region_Code: Optional[str] = None
#     Holiday: Optional[bool] = None
#     ID: Optional[str] = None

#     @field_validator("Sales", "Order")
#     @classmethod
#     def non_negative(cls, v):
#         if v < 0:
#             raise ValueError("Values must be non-negative")
#         return v


INPUT_FEATURES = [
                    'ID', 
                    'Store_id', 
                    'Store_Type', 
                    'Location_Type', 
                    'Region_Code', 
                    'Date', 
                    'Holiday', 
                    'Discount', 
                    '#Order', 
                    'Sales'
                    ]

FORMATED_FEATURES = [
                    'Store_id', 
                    'Store_Type', 
                    'Location_Type', 
                    'Region_Code', 
                    'Date', 
                    'Holiday', 
                    'Discount', 
                    'Order', 
                    'Sales'
                    ]

EXCLUDED_FEATURES = ['ID', 
                     'Store_Type', 
                    'Location_Type', 
                    'Region_Code',
                    'Date', 
                    'Discount',
                    'Holiday'
                    ]

TARGET_COLS = ['Order', 'Sales']
ORIGINAL_TARGET_COLS = ['#Order', 'Sales']

PREDICT_DATE_COLS = ['year', 'month', 'day']

PREDICT_OUTPUT_COLS = ['Store_id','Date',TARGET_COLS[0], TARGET_COLS[1],f'Pre_{TARGET_COLS[0]}', f'Pre_{TARGET_COLS[1]}']


RECUSSIVE_PREDICT_OUTPUT_COLS = ['Store_id','Date',f'Pre_{TARGET_COLS[0]}', f'Pre_{TARGET_COLS[1]}']

column_rename_mapping = {'#Order':f'Pre_{TARGET_COLS[0]}',TARGET_COLS[1]:f'Pre_{TARGET_COLS[1]}'}

# MODEL_TRAIN_FEATURES = [
#                         'Store_id', 
#                         # 'Store_Type', 
#                         # 'Location_Type', 
#                         # 'Region_Code', 
#                         'Date', 
#                         'Holiday', 
#                         #'Discount',  
#                         'day', 
#                         'month', 
#                         'year', 
#                         'weekday', 
#                         'weekofyear', 
#                         'is_weekend', 
#                         'Store_Type_S1', 
#                         'Store_Type_S2', 
#                         'Store_Type_S3', 
#                         'Store_Type_S4', 
#                         'Location_Type_L1', 
#                         'Location_Type_L2', 
#                         'Location_Type_L3', 
#                         'Location_Type_L4', 
#                         'Location_Type_L5', 
#                         'Region_Code_R1', 
#                         'Region_Code_R2', 
#                         'Region_Code_R3', 
#                         'Region_Code_R4', 
#                         'lag_1_Sales', 
#                         'lag_7_Sales', 
#                         'rolling_Sales_7', 
#                         'lag_1_Order', 
#                         'lag_7_Order', 
#                         'rolling_Order_7', 
#                         'Discount_Flag', 
#                         'lag_1_aov'
#                         'Order', 
#                         'Sales',
#                     ]