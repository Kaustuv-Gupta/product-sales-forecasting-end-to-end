import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DerivedFeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Discount_Flag'] = (X['Discount']=='Yes').astype(int)
        X['Holiday_Flag'] = X['Holiday'].astype(int)
        X['lag_1_aov'] = X['lag_1_Sales'] / (X['lag_1_Order'] + 1)  # Adding 1 to avoid division by zero
        X['lag_1_aov'] = X['lag_1_aov'].astype(float)
        return X