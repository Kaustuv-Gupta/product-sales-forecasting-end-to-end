from ml.data.preprocess import preprocess_data
from ml.features.date_features import DateFeatureExtractor
from ml.features.lag_features import LagFeatureTransformer
from ml.features.categorical_features import CategoricalEncoder
from ml.features.derived_features import DerivedFeatureTransformer
from ml.config.constants import CATEGORICAL_ENCODER_COLS

class FeaturePipeline:
    def __init__(self):
        self.date = DateFeatureExtractor()
        self.lag = LagFeatureTransformer(
            group_col='Store_id',
            target_cols=['Sales', 'Order'],
            lags=[1, 7],
            rolling_windows=[7]
        )
        self.cat = CategoricalEncoder(
            cols=CATEGORICAL_ENCODER_COLS
        )
        self.derived = DerivedFeatureTransformer()

    def fit_transform(self, df):
        df = preprocess_data(df)
        df = self.date.fit_transform(df)
        df = self.cat.fit_transform(df)
        df = self.lag.fit_transform(df)
        df = self.derived.fit_transform(df)

        return df

    def transform(self, df):
        df = preprocess_data(df)
        df = self.date.transform(df)
        df = self.cat.transform(df)
        df = self.lag.transform(df)
        df = self.derived.transform(df)

        return df