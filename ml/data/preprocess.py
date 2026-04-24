import pandas as pd
from ml.config.constants import CATEGORICAL_COLS, DEFAULT_SORT_COLS , CATEGORICAL_ENCODER_COLS
from ml.config.schema import FORMATED_FEATURES


def standardize_columns(df):
    df = df.copy()
    
    # fix problematic names
    df = df.rename(columns={"#Order": "Order"})
    
    return df


def enforce_dtypes(df):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df["Store_id"] = df["Store_id"].astype(int)
    df["Sales"] = df["Sales"].astype(float)
    df["Order"] = df["Order"].astype(float)

    df['Holiday'] = df['Holiday'].fillna(0).astype(int)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype('category')

    return df


def clean_data(df):
    df = df.copy()

    # remove duplicates
    df = df.drop_duplicates()

    # remove invalid rows
    df = df.dropna(subset=["Date", "Store_id"])

    # remove negative values
    df = df[(df["Sales"] >= 0) & (df["Order"] >= 0)]

    return df


def sort_data(df):
    return df.sort_values(DEFAULT_SORT_COLS).reset_index(drop=True)


def preprocess_data(df):

    df = standardize_columns(df)
    df = enforce_dtypes(df)
    #df = clean_data(df)
    df = sort_data(df)

    return df[FORMATED_FEATURES]