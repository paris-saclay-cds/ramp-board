import numpy as np
# import pandas as pd


labels = np.array(['A', 'B', 'Q', 'R'])

class FeatureExtractorReg():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        XX = np.array([np.array(dd) for dd in X_df['spectra']])
        XX -= np.median(XX, axis=1)[:, None]
        XX /= np.sqrt(np.sum(XX ** 2, axis=1))[:, None]
        XX = np.concatenate([XX, X_df[labels].values], axis=1)
        return XX
