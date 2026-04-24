import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from ml.config.constants import DEFAULT_DATE_COL

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        X = X.copy()
        X[DEFAULT_DATE_COL] = pd.to_datetime(X[DEFAULT_DATE_COL])

        X['day'] = X[DEFAULT_DATE_COL].dt.day.astype(int)
        X['month'] = X[DEFAULT_DATE_COL].dt.month.astype(int)
        X['year'] = X[DEFAULT_DATE_COL].dt.year.astype(int)
        X['weekday'] = X[DEFAULT_DATE_COL].dt.weekday.astype(int)
        X['weekofyear'] = X[DEFAULT_DATE_COL].dt.isocalendar().week.astype(int)
        X['is_weekend'] = X['weekday'].isin([5, 6]).astype(int)

        return X