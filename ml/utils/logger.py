import logging
import os
from datetime import datetime
from ml.config.constants import LOG_DIR


def get_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Create and return a configured logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create log directory
    os.makedirs(LOG_DIR, exist_ok=True)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(LOG_DIR, f"{log_file}_{timestamp}.log")

        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger