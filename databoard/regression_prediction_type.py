# Author: Balazs Kegl
# License: BSD 3 clause

import numpy as np


class PredictionArrayType:

    def __init__(self, *args, **kwargs):
        if 'y_prediction_array' in kwargs.keys():
            self.y_prediction_array = kwargs['y_prediction_array']
        # should match the way the target is represented when y_test is saved
        elif 'ground_truth_f_name' in kwargs.keys():
            # loading from file
            f_name = kwargs['ground_truth_f_name']
            self.y_prediction_array = np.genfromtxt(f_name)
        # should match save_prediction: representation of the target returned
        # by specific.test_model
        elif 'predictions_f_name' in kwargs.keys():
            # loading from file
            f_name = kwargs['predictions_f_name']
            self.y_prediction_array = np.genfromtxt(f_name)
        else:
            raise ValueError("Unkonwn init argument, {}".format(kwargs))

    def save_predictions(self, f_name):
        with open(f_name, "w") as f:
            for y_prediction in self.y_prediction_array:
                f.write(str(y_prediction))
                f.write("\n")

    def get_prediction_array(self):
        return self.y_prediction_array

    def get_combineable_predictions(self):
        """Returns an array which can be combined by taking means"""
        return self.y_prediction_array

#    def __iter__(self):
#        for y_pred, y_probas in self.get_predictions():
#            yield y_pred, y_probas

    # def combine(self, indexes=[]):
        # Not yet used

        # usually the class contains arrays corresponding to predictions
        # for a set of different data points. Here we assume that it is
        # a list of predictions produced by different functions on the same
        # data point. We return a single PrdictionType

        # Just saving here in case we want to go back there how to
        # combine based on simply ranks, k = len(indexes)
        # n = len(y_preds[0])
        # n_ones = n * k - y_preds[indexes].sum() # number of zeros
        # if len(indexes) == 0:  # we combine the full list
        #    indexes = range(len(self.y_probas_array))
        # combined_y_preds = self.y_preds_array.mean()
        # combined_prediction = PredictionType(combined_y_preds)
        # return combined_prediction


def get_nan_combineable_predictions(num_points):
    y_prediction_array = np.empty(num_points, dtype=float)
    y_prediction_array.fill(np.nan)
    return y_prediction_array


# also think about working with matrices of PredictionType instead of
# lists of PredictionArrayType


def transpose(predictions_list):
    pass
