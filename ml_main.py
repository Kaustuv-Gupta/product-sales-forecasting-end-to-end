import pickle
import pandas as pd
from ml.data.load_data import load_data, time_based_split
from ml.config.constants import DATA_FILE_PATH, TEST_START_DATE, VAL_START_DATE,DEFAULT_DATE_COL, ARTIFACTS_PATH, PIPELINE_FILE_NAME, ORDERS_MODEL_FILE_NAME, SALES_MODEL_FILE_NAME, PREDICTIONS_FILE_NAME, EVALUATION_METRICS_FILE_NAME, LAST_N_RECORDS_PER_STORE_FILE_NAME, LAST_N_DAYS
from ml.config.schema import TARGET_COLS
from ml.features.pipeline import FeaturePipeline
from ml.models.train import train_xgboost_model
from ml.models.predict import predict, recusive_predict
from ml.models.evaluate import evaluate_predictions
from ml.utils.logger import get_logger

logger = get_logger(__name__, log_file="main")



def main():
    #LOAD DATA
    logger.info("Loading data from %s", DATA_FILE_PATH)

    df = load_data(DATA_FILE_PATH)

    logger.info("Data loaded successfully with shape: %s", df.shape)
    logger.info("Columns in loaded data: %s", df.columns.tolist())
    #SPLIT DATA
    #train_val_df, test_df = time_based_split(df, TEST_START_DATE)
    #train_df, val_df = time_based_split(train_val_df, VAL_START_DATE)

    train_df, test_df = time_based_split(df, TEST_START_DATE)

    logger.info("Data split into train, validation, and test sets.")
    logger.info("Train shape: %s", train_df.shape)
    #logger.info("Validation shape: %s", val_df.shape)
    logger.info("Test shape: %s", test_df.shape)


    pipeline = FeaturePipeline()
    transformed_df = pipeline.fit_transform(train_df)

    with open(f'{ARTIFACTS_PATH}/{PIPELINE_FILE_NAME}', 'wb') as f:
        pickle.dump(pipeline, f)

    orders_model, sales_model = train_xgboost_model(transformed_df)

    with open(f'{ARTIFACTS_PATH}/{ORDERS_MODEL_FILE_NAME}', 'wb') as f:
        pickle.dump(orders_model, f)

    with open(f'{ARTIFACTS_PATH}/{SALES_MODEL_FILE_NAME}', 'wb') as f:
        pickle.dump(sales_model, f)

    logger.info("Models and feature pipeline saved to artifacts directory.")
    logger.info("Starting prediction on test set...")
    predict_output = predict(test_df, pipeline, orders_model, sales_model)


    predict_output.to_csv(f'{ARTIFACTS_PATH}/{PREDICTIONS_FILE_NAME}', index=False)
    logger.info("Predictions saved to %s", f'{ARTIFACTS_PATH}/{PREDICTIONS_FILE_NAME}')

    # Evaluate predictions
    results, df_metrics = evaluate_predictions(predict_output)
    logger.info("Evaluation results: %s", results)
    
    df_metrics.to_csv(f'{ARTIFACTS_PATH}/{EVALUATION_METRICS_FILE_NAME}', index=False)
    logger.info("Evaluation metrics saved to %s", f'{ARTIFACTS_PATH}/{EVALUATION_METRICS_FILE_NAME}')


    ##############################################################################################################
    # df_last_N_store_wise = (
    #             train_df.sort_values(['Store_id', 'Date'])
    #             .groupby('Store_id', group_keys=False)
    #             .tail(LAST_N_DAYS)
    #         ).reset_index(drop=True)
    df_last_N_store_wise = train_df.copy()
    
    df_last_N_store_wise.to_csv(f'{ARTIFACTS_PATH}/{LAST_N_RECORDS_PER_STORE_FILE_NAME}', index=False)
    logger.info("Last %d records per store saved to %s", LAST_N_DAYS, f'{ARTIFACTS_PATH}/{LAST_N_RECORDS_PER_STORE_FILE_NAME}')

    random_store_id = test_df['Store_id'].sample(1).iloc[0]

    #####################################3

    logger.info("Randomly selected Store_id for testing: %s", random_store_id)
    random_date = '2019-05-12'
    period = 7
    test_df_random_store = test_df[(test_df['Store_id'] == random_store_id) & (test_df['Date'] >= random_date)].sort_values('Date').head(period)

    random_store_payload = {
        'Store_id': test_df_random_store['Store_id'].iloc[0],   # for new store replace id by -99
        'Store_Type': test_df_random_store['Store_Type'].iloc[0],
        'Location_Type': test_df_random_store['Location_Type'].iloc[0],
        'Region_Code': test_df_random_store['Region_Code'].iloc[0]
        }
    

    is_existing_store_flag,rec_predict_output=recusive_predict(random_date, period, random_store_payload)

    rec_predict_output.to_csv(f'{ARTIFACTS_PATH}/rec_{PREDICTIONS_FILE_NAME}', index=False)
    logger.info("Predictions saved to %s", f'{ARTIFACTS_PATH}/rec_{PREDICTIONS_FILE_NAME}')


    if is_existing_store_flag:
        df2=df.copy()
        rec_predict_output2=rec_predict_output.copy()
        df2[DEFAULT_DATE_COL] = pd.to_datetime(df2[DEFAULT_DATE_COL])
        rec_predict_output2[DEFAULT_DATE_COL] = pd.to_datetime(rec_predict_output2[DEFAULT_DATE_COL])
        combined_results = rec_predict_output2.merge(df2[['Store_id', DEFAULT_DATE_COL , '#Order', 'Sales']]
                                                     ,on=['Store_id', DEFAULT_DATE_COL]
                                                     ,how='inner')
        
        final_view = combined_results[[
        'Store_id', 
        'Date', 
        '#Order', 
        'Sales', 
        f'Pre_{TARGET_COLS[0]}', 
        f'Pre_{TARGET_COLS[1]}'
        ]]
        final_view = final_view.rename(columns={'#Order': 'Order'})

        if not final_view.empty:
            rec_results, rec_df_metrics = evaluate_predictions(final_view)

        rec_df_metrics.to_csv(f'{ARTIFACTS_PATH}/rec_{EVALUATION_METRICS_FILE_NAME}', index=False)
        logger.info("Evaluation metrics saved to %s", f'{ARTIFACTS_PATH}/rec_{EVALUATION_METRICS_FILE_NAME}')

if __name__ == "__main__":
    main()