import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from ml.config.schema import EXCLUDED_FEATURES, TARGET_COLS
from ml.config.constants import XGBOOST_ORDERS_MODEL_PARAMS, XGBOOST_SALES_MODEL_PARAMS
from ml.utils.logger import get_logger

logger = get_logger(__name__, log_file="model_training")

def prepare_data(df):

    logger.info("Preparing data...")

    missing_targets = [col for col in TARGET_COLS if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    X = df.drop(columns=EXCLUDED_FEATURES + TARGET_COLS, errors='ignore')
    y_order = df[TARGET_COLS[0]]
    y_sales = df[TARGET_COLS[1]]

    logger.info(f"Feature shape: {X.shape}")

    return X, y_order, y_sales


def train_xgboost_model(df):
    logger.info("Starting train_xgboost_model")
    (
        X_train, y_train_orders, y_train_sales
    ) = prepare_data(df)

  
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train_orders shape: {y_train_orders.shape}")
    logger.info(f"y_train_sales shape: {y_train_sales.shape}")
    logger.info("Columns in X_train: %s", X_train.columns.tolist())

    logger.info("Training XGBoost models...")
    logger.info("Orders model parameters: %s", XGBOOST_ORDERS_MODEL_PARAMS)
    logger.info("Sales model parameters: %s", XGBOOST_SALES_MODEL_PARAMS)

    orders_model = XGBRegressor(**XGBOOST_ORDERS_MODEL_PARAMS)
    sales_model = XGBRegressor(**XGBOOST_SALES_MODEL_PARAMS)


    logger.info("Fitting orders model...")
    orders_model.fit(X_train, y_train_orders)


    logger.info("Predicting orders for training data...")
    X_train[f'Pre_{TARGET_COLS[0]}'] = orders_model.predict(X_train)
    X_train[f'Pre_{TARGET_COLS[0]}'] = X_train[f'Pre_{TARGET_COLS[0]}'].astype(float)
    
    logger.info("Fitting sales model...")
    sales_model.fit(X_train, y_train_sales)


    logger.info("Finished training XGBoost models.")
    return orders_model, sales_model
        
        


        