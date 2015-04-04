import numpy as np
#
class FeatureExtractor():

    def __init__(self):
        pass

    def fit(self, X_dict):
        pass

    def transform(self, X_dict):
        #print X_dict.keys()
        cols = [
            'magnitude_b', 
            'magnitude_r',
            'period',
            'asym_b', 
            'asym_r', 
            'log_p_not_variable', 
            'sigma_flux_b', 
            'sigma_flux_r', 
            'quality', 
            'div_period',
        ]
        return np.array([[instance[col] for col in cols] for instance in X_dict])
