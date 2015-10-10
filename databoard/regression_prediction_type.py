# Author: Balazs Kegl
# License: BSD 3 clause

import csv
import numpy as np


class PredictionType:

    def __init__(self, prediction):
        self.y_pred = prediction

    def __str__(self):
        return "y_pred = ".format(self.y_pred)

    def get_prediction(self):
        return self.y_pred


class PredictionArrayType:

    def __init__(self, *args, **kwargs):
        if 'y_pred_array' in kwargs.keys():
            self.y_pred_array = kwargs['y_pred_array']
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

    def save_predictions(self, f_name):
        with open(f_name, "w") as f:
            for y_pred in self.y_pred_array:
                f.write(str(y_pred))
                f.write("\n")

    def get_predictions(self):
        return self.y_pred_array

    def get_combineable_predictions(self):
        """Returns an array which can be combined by taking means"""
        return self.get_predictions()

    def combine(self, indexes=[]):
        # Not yet used

        # usually the class contains arrays corresponding to predictions
        # for a set of different data points. Here we assume that it is
        # a list of predictions produced by different functions on the same
        # data point. We return a single PrdictionType

        # Just saving here in case we want to go back there how to
        # combine based on simply ranks, k = len(indexes)
        #n = len(y_preds[0])
        # n_ones = n * k - y_preds[indexes].sum() # number of zeros
        if len(indexes) == 0:  # we combine the full list
            indexes = range(len(self.y_probas_array))
        combined_y_preds = self.y_preds_array.mean()
        combined_prediction = PredictionType(combined_y_preds)
        return combined_prediction


def get_nan_combineable_predictions(num_points):
    predictions = np.empty(num_points, dtype=float)
    predictions.fill(np.nan)
    return predictions


def get_y_pred_array(y_probas_array):
    return np.array([labels[y_probas.argmax()] for y_probas in y_probas_array])

# also think about working with matrices of PredictionType instead of
# lists of PredictionArrayType


def transpose(predictions_list):
    pass
