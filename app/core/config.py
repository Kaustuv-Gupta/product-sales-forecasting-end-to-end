import os
from enum import Enum
from functools import lru_cache

from dotenv import load_dotenv

class ApiResponseType(Enum):
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"

class Settings:
    def __init__(self, **kwargs):
        load_dotenv()

        self._load_constants()


    def _load_constants(self):
        """
        Load constants 
        """

        self.api_version = os.getenv("API_VERSION","v1")
        self.app_name = os.getenv("APP_NAME","PRODUCT_SALES_FORECASTING")
        self.api_base_path_template = os.getenv("API_BASE_PATH","/product_sales_forecasting/{}")
        self.app_name = os.getenv("API_VERSION","v1")
        self.LOG_DIR = os.getenv("APP_LOG_DIR","logs")


        #ml
        self.artifact_version = os.getenv("ARTIFACTS_VERSION","v1")
        self.arifacts_path_template = os.getenv("ARTIFACTS_PATH","artifacts/{}")
        self.pipeline_file_name = os.getenv("PIPELINE_FILE_NAME","")
        self.orders_file_name = os.getenv("ORDERS_MODEL_FILE_NAME","")
        self.sales_file_name = os.getenv("SALES_MODEL_FILE_NAME","")
        self.train_datafile_name = os.getenv("TRAIN_DATA_FILE_NAME","")

        #version pass path
        self.api_base_path = self.api_base_path_template.format(self.api_version)
        self.arifacts_path = self.arifacts_path_template.format(self.artifact_version)


@lru_cache()
def get_settings() ->Settings:
    """Get cached settings instance"""
    return Settings()

settings = get_settings()