import numpy as np
from databoard.base_prediction import BasePrediction

# Global static that should be set by specific (or somebody)
labels = []


class Predictions(BasePrediction):

    def __init__(self, y_pred=None, y_pred_labels=None, y_pred_indexes=None,
                 y_true=None, f_name=None, n_samples=None):
        if y_pred is not None:
            self.y_proba = np.array(y_pred)
        elif y_pred_labels is not None:
            self._init_from_pred_labels(y_pred_labels)
        elif y_true is not None:
            self._init_from_pred_labels(y_true)
        elif f_name is not None:
            self.y_proba = np.load(f_name)
        elif n_samples is not None:
            self.y_proba = np.empty(
                (n_samples, len(labels)), dtype=np.float64)
            self.y_proba.fill(np.nan)
        else:
            raise ValueError('Missing init argument: y_pred, y_pred_labels, '
                             'y_pred_indexes, y_true, f_name, or n_samples)')
        shape = self.y_proba.shape
        if len(shape) != 2:
            raise ValueError('Multiclass y_proba should be 2-dimensional, '
                             'instead it is {}-dimensional'.format(len(shape)))
        # if shape[1] != len(labels):
        #    raise ValueError('Vectors in multiclass y_proba should be '
        #                     '{}-dimensional, instead they are {}-dimensional'.
        #                     format(len(labels), shape[1]))

    def _init_from_pred_labels(self, y_pred_labels):
        type_of_label = type(labels[0])
        self.y_proba = np.zeros(
            (len(y_pred_labels), len(labels)), dtype=np.float64)
        for ps_i, label_list in zip(self.y_proba, y_pred_labels):
            # converting single labels to list of labels, assumed below
            if type(label_list) != np.ndarray and type(label_list) != list:
                label_list = [label_list]
            label_list = map(type_of_label, label_list)
            for label in label_list:
                ps_i[labels.index(label)] = 1.0 / len(label_list)

    def set_valid_in_train(self, predictions, test_is):
        self.y_proba[test_is] = predictions.y_proba

    @property
    def valid_indexes(self):
        return ~np.isnan(self.y_proba[:, 0])

    @property
    def y_pred(self):
        return self.y_proba

    @property
    def y_pred_label_index(self):
        """Multi-class y_pred is the index of the predicted label."""
        return np.argmax(self.y_proba, axis=1)

    @property
    def y_pred_label(self):
        return labels[self.y_pred_label_index]

    @property
    def y_pred_comb(self):
        """Return an array which can be combined by taking means."""
        return self.y_proba

    @property
    def n_samples(self):
        return len(self.y_proba)

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
        #     indexes = range(len(self.y_proba_array))
        # combined_y_proba = self.y_proba_array.mean(axis=0)
        # combined_prediction = PredictionType((labels[0], combined_y_proba))
        # combined_prediction.make_consistent()
        # return combined_prediction
