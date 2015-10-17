import numpy as np

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_dict, y):
        pass

    def transform(self, X_dict):
        cols = X_dict[0].keys()
        return np.array([[instance[col] for col in cols]
                         for instance in X_dict])
