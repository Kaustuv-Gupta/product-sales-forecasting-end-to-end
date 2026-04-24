import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols
        self.encoder = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            dtype=int
        )
        self.feature_names = None

    def fit(self, X, y=None):
        self.encoder.fit(X[self.cols])
        self.feature_names = self.encoder.get_feature_names_out(self.cols)
        return self

    def transform(self, X):
        X = X.copy()

        encoded = self.encoder.transform(X[self.cols])
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.feature_names,
            index=X.index
        )

        #X = X.drop(columns=self.cols)
        X = pd.concat([X, encoded_df], axis=1)

        return X