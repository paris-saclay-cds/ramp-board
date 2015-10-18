# Author: Balazs Kegl
# License: BSD 3 clause

import numpy as np

from .base_prediction import BasePrediction


class Prediction(BasePrediction):

    def __init__(self, indices=None, y_pred=None, y_proba=None, y_true=None):
        if y_pred is not None:
            self.y_pred = y_pred

        # XXX : after I am lost
        elif 'prediction_list' in kwargs.keys():
            self.y_pred_array = np.array(
                [prediction for prediction in kwargs['prediction_list']])
        elif 'f_name' in kwargs.keys():
            # loading from file
            f_name = kwargs['f_name']
            self.y_pred_array = np.genfromtxt(f_name)
        elif 'y_combined_array' in kwargs.keys():
            self.y_pred_array = kwargs['y_combined_array']
        else:
            raise ValueError("Unkonwn init argument, {}".format(kwargs))

    def save(self, f_name):
        np.savetxt(self.y_proba, delimiter=',', fmt="%f")

    @property
    def y_comp(self):
        """Returns an array which can be combined by taking means"""
        return self.y_pred


def get_nan_combineable_predictions(num_points):
    predictions = np.empty(num_points, dtype=float)
    predictions.fill(np.nan)
    return predictions
