import numpy as np
from databoard.base_prediction import BasePrediction
import databoard.multiclass_prediction as multiclass_prediction
import databoard.regression_prediction as regression_prediction

# Global static that should be set by specific (or somebody)
labels = []


class Predictions(BasePrediction):

    def __init__(self, y_pred=None, y_true=None, f_name=None, n_samples=None):
        multiclass_prediction.labels = labels
        if y_pred is not None:
            self.multiclass_prediction = multiclass_prediction.Predictions(
                y_pred=y_pred[:, :-1])
            self.regression_prediction = regression_prediction.Predictions(
                y_pred=y_pred[:, -1])
        elif y_true is not None:
            self.multiclass_prediction = multiclass_prediction.Predictions(
                y_true=y_true[:, 0])
            self.regression_prediction = regression_prediction.Predictions(
                y_true=y_true[:, 1])
        elif f_name is not None:
            y_true = np.load(f_name)
            self.multiclass_prediction = multiclass_prediction.Predictions(
                y_true=y_true[:, 0])
            self.regression_prediction = regression_prediction.Predictions(
                y_true=y_true[:, 1])
        elif n_samples is not None:
            self.multiclass_prediction = multiclass_prediction.Predictions(
                n_samples=n_samples)
            self.regression_prediction = regression_prediction.Predictions(
                n_samples=n_samples)

    def set_valid_in_train(self, predictions, test_is):
        self.multiclass_prediction.set_valid_in_train(
            predictions.multiclass_prediction, test_is)
        self.regression_prediction.set_valid_in_train(
            predictions.regression_prediction, test_is)

    @property
    def valid_indexes(self):
        return self.multiclass_prediction.valid_indexes

    @property
    def y_pred(self):
        return np.concatenate(
            [self.multiclass_prediction.y_pred,
             self.regression_prediction.y_pred.reshape(-1, 1)],
            axis=1)

    @property
    def y_pred_comb(self):
        """Return an array which can be combined by taking means."""
        return self.y_pred

    @property
    def n_samples(self):
        return self.multiclass_prediction.n_samples
