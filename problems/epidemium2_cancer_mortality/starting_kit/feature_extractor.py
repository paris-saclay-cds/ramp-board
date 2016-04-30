import pandas as pd
from sklearn.base import TransformerMixin


class FeatureExtractor(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, df, y):
        return self

    def transform(self, df):
        df_ = pd.get_dummies(df, drop_first=False, columns=['country'])
        df_ = pd.get_dummies(df_, drop_first=True, columns=['gender'])
        return df_.values
