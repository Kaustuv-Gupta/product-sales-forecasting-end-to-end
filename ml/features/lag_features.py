import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from ml.config.constants import DEFAULT_DATE_COL

class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, target_cols, lags=[1, 7], rolling_windows=[7],add_aov=True):
        self.group_col = group_col
        self.target_cols = target_cols
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.add_aov = add_aov

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[DEFAULT_DATE_COL] = pd.to_datetime(df[DEFAULT_DATE_COL])
        # Ensure proper sorting
        df = df.sort_values([self.group_col, DEFAULT_DATE_COL])

        #grouped = df.groupby(self.group_col)

        for col in self.target_cols:

            # Lags
            for lag in self.lags:
                df[f'lag_{lag}_{col}'] = (
                    df.groupby(self.group_col)[col].shift(lag)
                ).astype(float)

            # Rolling
            for window in self.rolling_windows:
                df[f'rolling_{col}_{window}'] = (
                    df.groupby(self.group_col)[col]
                    .transform(lambda x: x.shift(1).rolling(window).mean())
                ).astype(float)
        return df