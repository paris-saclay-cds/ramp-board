import numpy as np
from databoard.base_prediction import BasePrediction
import databoard.multiclass_prediction as multiclass_prediction
import databoard.regression_prediction as regression_prediction


class Predictions(BasePrediction):

    def __init__(self, labels=None, y_pred=None, y_true=None, f_name=None,
                 shape=None):
        self.labels = labels
        # multiclass_prediction.labels = labels
        if y_pred is not None:
            self.multiclass_prediction = multiclass_prediction.Predictions(
                labels=self.labels, y_pred=y_pred[:, :-1])
            self.regression_prediction = regression_prediction.Predictions(
                labels=self.labels, y_pred=y_pred[:, -1])
        elif y_true is not None:
            self.multiclass_prediction = multiclass_prediction.Predictions(
                labels=self.labels, y_true=y_true[:, 0])
            self.regression_prediction = regression_prediction.Predictions(
                labels=self.labels, y_true=y_true[:, 1])
#        elif f_name is not None:
#            y_true = np.load(f_name)
#            self.multiclass_prediction = multiclass_prediction.Predictions(
#                labels=self.labels, y_true=y_true[:, 0])
#            self.regression_prediction = regression_prediction.Predictions(
#                labels=self.labels, y_true=y_true[:, 1])
        elif shape is not None:
            # last col is reg, first shape[1] - 1 cols are clf
            self.multiclass_prediction = multiclass_prediction.Predictions(
                labels=self.labels, shape=(shape[0], shape[1] - 1))
            self.regression_prediction = regression_prediction.Predictions(
                labels=self.labels, shape=shape[0])

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
