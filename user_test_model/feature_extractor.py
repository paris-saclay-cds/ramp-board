def fold_time_series(time_point, period, div_period):
    return time_point - 1.0 * int(time_point / (period / div_period)) * period / div_period

def get_bin_means(instance, num_bins):
    period = instance['period']
    div_period = instance['div_period']
    real_period = period / div_period
    bins = [i * real_period / num_bins for i in range(num_bins + 1)]
    feature_array = np.empty(2 * num_bins)
    for band in ['b', 'r']:
        time_points = np.array(instance['time_points_' + band])
        light_points = np.array(instance['light_points_' + band])
        time_points_folded = np.array([fold_time_series(time_point, period, div_period) 
                                       for time_point in time_points])
        time_points_folded_digitized = np.digitize(time_points_folded, bins) - 1
        binned_means = np.array([light_points[time_points_folded_digitized == i].mean()
                                for i in range(num_bins)])
        if band == 'b':
            feature_array[:num_bins] = binned_means
        else:
            feature_array[num_bins:] = binned_means
    return feature_array

import numpy as np

class FeatureExtractor():

    def __init__(self):
        pass

    def fit(self, X_dict, y):
        pass

    def transform(self, X_dict):
        cols = [
            'magnitude_b', 
            'magnitude_r'
        ]
        X_array = np.array([[instance[col] for col in cols] for instance in X_dict])
        real_period = np.array([instance['period'] / instance['div_period']
            for instance in X_dict])
        X_array = np.concatenate((X_array.T, [real_period])).T
        num_bins = 5
        X_array_variable_features = np.array([get_bin_means(instance, num_bins) for instance in X_dict])
        X_array = np.concatenate((X_array.T, X_array_variable_features.T)).T
        return X_array