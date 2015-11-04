import numpy as np


class FeatureExtractor():

    def __init__(self):
        pass

    def fit(self, X_dict, y):
        pass

    def transform(self, X_dict):
        cols = [
            'magnitude_b',
            'magnitude_r',
            'period',
            'div_period',
        ]
        return np.array(
            [[instance[col] for col in cols] for instance in X_dict])
