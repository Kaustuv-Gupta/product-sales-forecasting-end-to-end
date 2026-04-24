import pandas as pd
import numpy as np
import pickle
from ml.models.train import prepare_data
from ml.config.schema import PREDICT_OUTPUT_COLS, TARGET_COLS, PREDICT_DATE_COLS, ORIGINAL_TARGET_COLS, RECUSSIVE_PREDICT_OUTPUT_COLS,column_rename_mapping
from ml.config.constants import ARTIFACTS_PATH, PIPELINE_FILE_NAME, ORDERS_MODEL_FILE_NAME, SALES_MODEL_FILE_NAME, LAST_N_RECORDS_PER_STORE_FILE_NAME,LAST_N_DAYS, DEFAULT_DATE_COL


from ml.utils.logger import get_logger
logger = get_logger(__name__, log_file="model_prediction")


def predict(df, feature_pipeline, orders_model, sales_model):
    logger.info("Starting prediction process...")
    logger.info("Input data shape: %s", df.shape)
    logger.info("Input data columns: %s", df.columns.tolist())

    # Apply feature pipeline transformations
    logger.info("Applying feature pipeline transformations...")
    transformed_df = feature_pipeline.transform(df)

    logger.info("Transformed data shape: %s", transformed_df.shape)
    logger.info("Transformed data columns: %s", transformed_df.columns.tolist())

    (
        X_pred, y_pred_orders, y_pred_sales
    ) = prepare_data(transformed_df)

    # Predict orders and sales
    logger.info("Predicting orders and sales...")
    X_pred[f'Pre_{TARGET_COLS[0]}'] = orders_model.predict(X_pred)
    X_pred[f'Pre_{TARGET_COLS[0]}'] = X_pred[f'Pre_{TARGET_COLS[0]}'].astype(float)
    X_pred[f'Pre_{TARGET_COLS[1]}'] = sales_model.predict(X_pred)
    X_pred[f'Pre_{TARGET_COLS[1]}'] = X_pred[f'Pre_{TARGET_COLS[1]}'].astype(float)

    OUTPUT_DF = pd.concat([
    X_pred, 
    pd.Series(y_pred_orders, name=TARGET_COLS[0], index=X_pred.index),
    pd.Series(y_pred_sales, name=TARGET_COLS[1], index=X_pred.index)
    ], axis=1)

    OUTPUT_DF[TARGET_COLS[0]] = OUTPUT_DF[TARGET_COLS[0]].astype(float)
    OUTPUT_DF[TARGET_COLS[1]] = OUTPUT_DF[TARGET_COLS[1]].astype(float)

    OUTPUT_DF['Date'] = pd.to_datetime({
    'year': OUTPUT_DF[PREDICT_DATE_COLS[0]],
    'month': OUTPUT_DF[PREDICT_DATE_COLS[1]],
    'day': OUTPUT_DF[PREDICT_DATE_COLS[2]]
})


    logger.info("Prediction process completed.")

    OUTPUT_DF = OUTPUT_DF[PREDICT_OUTPUT_COLS]
    logger.info("Output data shape: %s", OUTPUT_DF.shape)
    logger.info("Output data columns: %s", OUTPUT_DF.columns.tolist())

    return OUTPUT_DF

def recusive_predict(start_date, period, store_payload):
    is_existing_store=True
    logger.info("Starting recursive prediction process...")
    logger.info("Input start_date: %s", start_date)
    logger.info("Input period: %s", period)
    logger.info("Input store payload: %s", store_payload)

    with open(f'{ARTIFACTS_PATH}/{PIPELINE_FILE_NAME}', 'rb') as file:
        feature_pipeline = pickle.load(file)

    with open(f'{ARTIFACTS_PATH}/{ORDERS_MODEL_FILE_NAME}', 'rb') as file:
        orders_model = pickle.load(file)

    with open(f'{ARTIFACTS_PATH}/{SALES_MODEL_FILE_NAME}', 'rb') as file:
        sales_model = pickle.load(file)

    train_df= pd.read_csv(f'{ARTIFACTS_PATH}/{LAST_N_RECORDS_PER_STORE_FILE_NAME}')
    logger.info("Train data loaded for recursive prediction with shape: %s", train_df.shape)
    logger.info("Train data columns: %s", train_df.columns.tolist())

    date_start_date=pd.to_datetime(start_date)

    df=train_df[(train_df['Store_id'] == store_payload['Store_id']) & 
    (pd.to_datetime(train_df[DEFAULT_DATE_COL]) < date_start_date) ].sort_values(DEFAULT_DATE_COL).tail(LAST_N_DAYS)

    if df.empty:
        is_existing_store=False
        logger.error("Not enough historical data for Store_id %s to perform recursive prediction. Required: %d, Available: %d", store_payload['Store_id'], LAST_N_DAYS, len(df))
        
        dates = pd.date_range(start=start_date, periods=period, freq='D')
        data = {
                DEFAULT_DATE_COL: dates,
                'Store_id': store_payload['Store_id'],
                'Store_Type': store_payload['Store_Type'],
                'Location_Type': store_payload['Location_Type'],
                'Region_Code': store_payload['Region_Code']
            }
        forecast_df = pd.DataFrame(data)
        forecast_df = forecast_df.reindex(columns=train_df.columns)

        for idx in range(len(forecast_df)):
            logger.info("Forecasting period: %s", idx+1)
            predict_df=forecast_df.iloc[:idx+1]
            predict_output=predict(predict_df, feature_pipeline, orders_model, sales_model)

            forecast_df.loc[idx, ORIGINAL_TARGET_COLS[0]] = predict_output[f'Pre_{TARGET_COLS[0]}'].iloc[-1]
            forecast_df.loc[idx, ORIGINAL_TARGET_COLS[1]] = predict_output[f'Pre_{TARGET_COLS[1]}'].iloc[-1]

        OUTPUT_DF=forecast_df.copy()
    else:
        logger.info('Historical data exists..')
        df[DEFAULT_DATE_COL] = pd.to_datetime(df[DEFAULT_DATE_COL])
        max_date = pd.to_datetime(df[DEFAULT_DATE_COL].max())

        logger.error("Store_id %s max train date: %s", store_payload['Store_id'], str(max_date))
        target_start = pd.to_datetime(start_date)

        forecast_end = target_start + pd.Timedelta(days=period - 1)
        full_forecast_range = pd.date_range(start=max_date + pd.Timedelta(days=1), end=forecast_end, freq='D')
        logger.info("full_forecast_range range: %s to %s", full_forecast_range.min(), full_forecast_range.max())
        data = {
            DEFAULT_DATE_COL: full_forecast_range,
            'Store_id': store_payload['Store_id'],
            'Store_Type': store_payload['Store_Type'],
            'Location_Type': store_payload['Location_Type'],
            'Region_Code': store_payload['Region_Code']
        }

        forecast_df = pd.DataFrame(data)
        forecast_df = forecast_df.reindex(columns=train_df.columns)

        combined_df = pd.concat([df, forecast_df], axis=0).reset_index(drop=True)
        history_len = len(df)
        total_forecast_len = len(forecast_df)

        logger.info("Combined shape: %s", combined_df.shape)
        logger.info("Min and max dates in Combined data for selected store: %s to %s", combined_df['Date'].min(), combined_df['Date'].max())
        
        for i in range(total_forecast_len):
            idx = history_len + i
            
            predict_df = combined_df.iloc[:idx + 1]
            
            predict_output = predict(predict_df, feature_pipeline, orders_model, sales_model)

            
            pred_order = predict_output[f'Pre_{TARGET_COLS[0]}'].iloc[-1]
            pred_sales = predict_output[f'Pre_{TARGET_COLS[1]}'].iloc[-1]

            
            combined_df.loc[idx, ORIGINAL_TARGET_COLS[0]] = pred_order
            combined_df.loc[idx, ORIGINAL_TARGET_COLS[1]] = pred_sales
            forecast_df.loc[i, ORIGINAL_TARGET_COLS[0]] = pred_order
            forecast_df.loc[i, ORIGINAL_TARGET_COLS[1]] = pred_sales

        OUTPUT_DF = forecast_df[forecast_df[DEFAULT_DATE_COL] >= target_start]



    for key, val in column_rename_mapping.items():
        if key in forecast_df.columns:
            OUTPUT_DF = OUTPUT_DF.rename(columns={key: val})


    return is_existing_store,OUTPUT_DF[RECUSSIVE_PREDICT_OUTPUT_COLS]
    
        



# def recusive_predict(df, feature_pipeline, orders_model, sales_model, date_col='Date', store_id_col='Store_id'):
#     logger.info("Starting recursive prediction process...")
#     logger.info("Input data shape: %s", df.shape)
#     logger.info("Input data columns: %s", df.columns.tolist())

#     # Ensure date column is datetime
#     df[date_col] = pd.to_datetime(df[date_col])

#     # Sort by date and store_id
#     df = df.sort_values(by=[store_id_col, date_col])

#     # Apply feature pipeline transformations
#     logger.info("Applying feature pipeline transformations...")
#     transformed_df = feature_pipeline.transform(df)

#     logger.info("Transformed data shape: %s", transformed_df.shape)
#     logger.info("Transformed data columns: %s", transformed_df.columns.tolist())

#     (
#         X_pred, y_pred_orders, y_pred_sales
#     ) = prepare_data(transformed_df)

#     # Predict orders and sales
#     logger.info("Predicting orders and sales recursively...")
    
#     for idx in range(len(X_pred)):
#         X_pred.loc[idx, f'Pre_{TARGET_COLS[0]}'] = orders_model.predict(X_pred.loc[idx:idx])
#         X_pred.loc[idx, f'Pre_{TARGET_COLS[0]}'] = X_pred.loc[idx, f'Pre_{TARGET_COLS[0]}'].astype(float)
#         X_pred.loc[idx, f'Pre_{TARGET_COLS[1]}'] = sales_model.predict(X_pred.loc[idx:idx])
#         X_pred.loc[idx, f'Pre_{TARGET_COLS[1]}'] = X_pred.loc[idx, f'Pre_{TARGET_COLS[1]}'].astype(float)

#         if idx < len(X_pred) - 1:
#             next_date = X_pred.loc[idx + 1, date_col]
#             next_store_id = X_pred.loc[idx + 1, store_id_col]

#             if next_date > X_pred.loc[idx, date_col] and next_store_id == X_pred.loc[idx, store_id_col]:
#                 X_pred.loc[idx + 1, f'Pre_{TARGET_COLS[0]}'] = X_pred.loc[idx, f'Pre_{TARGET_COLS[0]}']
#                 X_pred.loc[idx + 1, f'Pre_{TARGET_COLS[1]}'] = X_pred.loc[idx, f'Pre_{TARGET_COLS[1]}']

#     OUTPUT_DF = pd.concat([