class FeatureExtractor(object):
    # The columns you want to include without pre-processing
    core_cols = ['Year']
    # Categorical columns. They must be processed (use pd.get_dummies for the simplest way)
    categ_cols = ['RegionType','Part of', 'Region', 'Gender', 'MainOrigin', 'cancer_type']
    # the different factors to include in the model
    additional_cols = []
 
    def __init__(self):
        pass
 
    def fit(self, X_df, y_array):
        pass
 
    def transform(self, X_df):
        import pandas as pd
        ret = X_df[self.core_cols].copy()
        # dummify the categorical variables
        for col in self.categ_cols:
            ret = ret.join(pd.get_dummies(X_df[col], prefix=col[:3]))
        # add extra information
        for col in self.additional_cols:
            ret[col] = X_df[col]
        return ret.values
