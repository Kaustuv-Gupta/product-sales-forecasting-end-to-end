import os
import pandas as pd
from ml.config.constants import DEFAULT_DATE_COL, DEFAULT_SORT_COLS
from ml.config.schema import INPUT_FEATURES


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    return df


def basic_cleaning(df: pd.DataFrame, date_col: str = DEFAULT_DATE_COL) -> pd.DataFrame:
    df = df.copy()

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df = df.drop_duplicates()

    if date_col in df.columns:
        df = df.dropna(subset=[date_col])


    df = df.sort_values(DEFAULT_SORT_COLS).reset_index(drop=True)

    return df


def validate_schema(df: pd.DataFrame):
    """
    Ensure required columns exist
    """
    required_cols = [
        "Date",
        "Store_id",
        "Sales",
        "#Order"
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    

def load_data(path: str) -> pd.DataFrame:
    df = load_csv(path)
    validate_schema(df)
    df = basic_cleaning(df)
    return df[INPUT_FEATURES]


def time_based_split(df: pd.DataFrame, split_date: str):
    split_date = pd.to_datetime(split_date)

    split1 = df[df["Date"] < split_date]
    split2 = df[df["Date"] >= split_date]

    return split1, split2