import pandas as pd
from sklearn.base import TransformerMixin


class FeatureExtractor(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        X_df_ = pd.get_dummies(X_df, drop_first=False, columns=['country'])
        X_df_ = pd.get_dummies(X_df_, drop_first=True, columns=['gender'])
        return X_df_.values
