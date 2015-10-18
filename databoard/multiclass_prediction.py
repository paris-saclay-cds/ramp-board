# Author: Balazs Kegl
# License: BSD 3 clause

import csv
import numpy as np

from .base_prediction import BasePrediction

# Global static that should be set by specific (or somebody)
labels = []


class Prediction(BasePrediction):

    def __init__(self, indices=None, y_pred=None, y_proba=None, y_true=None):
        self.indices = indices
        if y_pred is not None:  # probability matrix
            self.y_proba = np.array(y_pred, dtype=np.float64)


        # XXX : the rest I am lost
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

    @property
    def y_comb(self):
        """Returns an array which can be combined by taking means"""
        return self.y_proba

    def save(self, fname):
        np.savetxt(self.y_proba, delimiter=',', fmt="%f")

    @property
    def y(self):
        return self.y_proba

    def get_index_max(self):  # not super happy about this one
        return np.argmax(self.y_proba, axis=1)


def get_nan_combineable_predictions(n_samples):
    predictions = np.empty((n_samples, len(labels)), dtype=float)
    predictions.fill(np.nan)
    return predictions
