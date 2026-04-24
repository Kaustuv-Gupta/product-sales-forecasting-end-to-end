import pickle
import pandas as pd
from app.core.config import settings

from app.utils.logger import get_logger
logger = get_logger(__name__, log_file="app_model_loader")



def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# logger.info("Artifacts Loading Started..")

# sales_model = load_pickle(f"{settings.arifacts_path}/{settings.sales_file_name}")
# orders_model = load_pickle(f"{settings.arifacts_path}/{settings.orders_file_name}")
# feature_pipeline = load_pickle(f"{settings.arifacts_path}/{settings.pipeline_file_name}")
# train_df = pd.read_csv(f'{settings.arifacts_path}/{settings.train_datafile_name}')

# logger.info("Artifacts Loading Completed..")

def load_models():
    logger.info("Artifacts Loading Started..")

    sales_model = load_pickle(f"{settings.arifacts_path}/{settings.sales_file_name}")
    orders_model = load_pickle(f"{settings.arifacts_path}/{settings.orders_file_name}")
    feature_pipeline = load_pickle(f"{settings.arifacts_path}/{settings.pipeline_file_name}")
    train_df = pd.read_csv(f"{settings.arifacts_path}/{settings.train_datafile_name}")

    logger.info("Artifacts Loading Completed..")

    return {
        "sales_model": sales_model,
        "orders_model": orders_model,
        "feature_pipeline": feature_pipeline,
        "train_df": train_df
    }