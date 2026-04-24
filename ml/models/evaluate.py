import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ml.utils.metrics import wape
from ml.config.schema import TARGET_COLS

from ml.utils.logger import get_logger
logger = get_logger(__name__, log_file="model_evaluation")

def evaluate_predictions(df):
    logger.info("Starting evaluation of predictions...")
    logger.info("Input data shape: %s", df.shape)

    results = {}

    for target in TARGET_COLS:
        logger.info(f"Evaluating target: {target}")

        pred_col = f"Pre_{target}"
        if pred_col not in df.columns or target not in df.columns:
            logger.warning(f"Missing column: {pred_col} or {target}")
            continue
        
        y_true = df[target]
        y_pred = df[pred_col]

        metrics  = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MSE": mean_squared_error(y_true, y_pred),
            "WAPE": wape(y_true, y_pred)

        }
        results[target] = metrics

        logger.info(f"{target} → {metrics}")

    logger.info("Evaluation completed: %s", results)

    df_metrics = pd.DataFrame(results).T.rename_axis("Target").reset_index()
    return  results, df_metrics