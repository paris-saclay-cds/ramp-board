# Author: Balazs Kegl
# License: BSD 3 clause

import csv
import string
import numpy as np

# Global static that should be set by specific (or somebody)
labels = []


class PredictionArrayType(object):

    def __init__(self, *args, **kwargs):
        if 'y_prediction_array' in kwargs.keys():  # probability matrix
            self.y_probas_array = np.array(
                kwargs['y_prediction_array'], dtype=np.float64)
        elif 'y_pred_label_array' in kwargs.keys():
            y_pred_label_array = kwargs['y_pred_label_array']
            self._init_from_pred_label_array(y_pred_label_array)
        elif 'y_pred_index_array' in kwargs.keys():
            y_pred_index_array = kwargs['y_pred_index_array']
            self.y_probas_array = np.zeros(
                (len(y_pred_index_array), len(labels)), dtype=np.float64)
            for y_probas, label_index in \
                    zip(self.y_probas_array, y_pred_index_array):
                y_probas[label_index] = 1.0
        # should match the way the target is represented when y_test is saved
        elif 'ground_truth_f_name' in kwargs.keys():
            # loading from ground truth file
            f_name = kwargs['ground_truth_f_name']
            with open(f_name) as f:
                y_pred_label_array = list(csv.reader(f))
            self._init_from_pred_label_array(y_pred_label_array)
        # should match save_prediction: representation of the target returned
        # by specific.test_model
        elif 'predictions_f_name' in kwargs.keys():
            f_name = kwargs['predictions_f_name']
            with open(f_name) as f:
                self.y_probas_array = np.array(
                    list(csv.reader(f)), dtype=np.float64)
        else:
            raise ValueError("Unkonwn init argument, {}".format(kwargs))

    def _init_from_pred_label_array(self, y_pred_label_array):
        type_of_label = type(labels[0])
        self.y_probas_array = np.zeros(
            (len(y_pred_label_array), len(labels)), dtype=np.float64)
        for y_probas, label_list in\
                zip(self.y_probas_array, y_pred_label_array):
            label_list = map(type_of_label, label_list)
            for label in label_list:
                y_probas[labels.index(label)] = 1.0 / len(label_list)


#    def __iter__(self):
#        for y_pred, y_probas in self.get_predictions():
#            yield y_pred, y_probas

    def save_predictions(self, f_name):
        with open(f_name, "w") as f:
            for y_probas in self.y_probas_array:
                f.write(string.join(map(str, y_probas), ',') + '\n')

    def get_prediction_array(self):
        return self.y_probas_array

    def get_combineable_predictions(self):
        """Returns an array which can be combined by taking means"""
        return self.y_probas_array

    def get_pred_index_array(self):
        return np.argmax(self.y_probas_array, axis=1)

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
        #     indexes = range(len(self.y_probas_array))
        # combined_y_probas = self.y_probas_array.mean(axis=0)
        # combined_prediction = PredictionType((labels[0], combined_y_probas))
        # combined_prediction.make_consistent()
        # return combined_prediction


def get_nan_combineable_predictions(num_points):
    predictions = np.empty((num_points, len(labels)), dtype=float)
    predictions.fill(np.nan)
    return predictions
