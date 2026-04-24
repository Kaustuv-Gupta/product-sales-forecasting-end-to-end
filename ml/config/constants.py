DEFAULT_DATE_COL = "Date"

DEFAULT_SORT_COLS = ['Date','Store_id']

DATA_FILE_PATH = "data/raw/TRAIN.csv"
ARTIFACTS_PATH = "artifacts/v1"
LOG_DIR = "logs"

TEST_START_DATE = "2019-05-01"
VAL_START_DATE = "2019-02-01"

CATEGORICAL_COLS = ['Store_Type','Location_Type','Region_Code','Holiday','Discount']

CATEGORICAL_ENCODER_COLS = ['Store_Type', 'Location_Type', 'Region_Code']

LAG_COLS = ['Sales', 'Order']

FLAG_COLS = ['Holiday', 'Discount']

XGBOOST_ORDERS_MODEL_PARAMS = {
        'n_estimators': 201,
        'max_depth': 10,
        'learning_rate': 0.013960663099383598,
        'subsample': 0.9208699091164867,
        'colsample_bytree': 0.9170140235850499,
        'reg_alpha': 4.824825650018285,
        'reg_lambda': 2.4528319109319674
        }

XGBOOST_SALES_MODEL_PARAMS = {
            'n_estimators': 323,
            'max_depth': 5,
            'learning_rate': 0.026530195738347556,
            'subsample': 0.7388544536271039,
            'colsample_bytree': 0.8628252390092686,
            'reg_alpha': 1.1499809816719329,
            'reg_lambda': 4.984633296547041
        }

PIPELINE_FILE_NAME = "feature_pipeline.pkl"
ORDERS_MODEL_FILE_NAME = "xgb_orders_model.pkl"
SALES_MODEL_FILE_NAME = "xgb_sales_model.pkl"
PREDICTIONS_FILE_NAME = "predictions.csv"
EVALUATION_METRICS_FILE_NAME = "evaluation_metrics.csv"
LAST_N_DAYS = 8
LAST_N_RECORDS_PER_STORE_FILE_NAME = "train_data.csv"