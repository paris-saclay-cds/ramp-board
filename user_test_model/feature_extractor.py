import numpy as np

class FeatureExtractor():

    def __init__(self):
        pass

    def transform(self, X_dict):
        #print X_dict.keys()
        return np.array([X_dict[col] for col in [
            'magnitude_b', 
            'magnitude_r',
            'period',
            'asym_b', 
            'asym_r', 
            'log_p_not_variable', 
            'flux_b', 
            'flux_r', 
            'quality', 
            'div_period',
        ]]).T