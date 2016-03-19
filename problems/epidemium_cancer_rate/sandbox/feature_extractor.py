import pandas as pd


class FeatureExtractor(object):
    core_cols = ['Year']
    region_cols = ['RegionType', 'Part of', 'Region']
    categ_cols = ['Gender', 'Age', 'MainOrigin']
    additional_cols = ['HIV_15_49']

    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        ret = X_df[['Year']].copy()
        # dummify the categorical variables
        for col in self.categ_cols:
            ret = ret.join(pd.get_dummies(X_df[col], prefix=col[:3]))
        # add extra information
        for col in self.additional_cols:
            ret[col] = X_df[col]
        return ret.values
