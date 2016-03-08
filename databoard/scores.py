"""All score classes have a score function taking a ground truth vector
    and a prediction vector and returning a score; and a boolean
    higher_the_better."""

# Author: Balazs Kegl
# License: BSD 3 clause

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class Score(object):

    def __init__(self, score, eps=10 ** -8):
        self.score = score
        self.eps = eps
        self.dtype = np.dtype(np.float)

    def __add__(self, right):
        return Score(self.score + right.score, self.eps)

    def __div__(self, divider):
        return Score(self.score / divider, self.eps)

    def __mul__(self, multiplier):
        return Score(self.score * multiplier.score, self.eps)

    def __sub__(self, diffand):
        return Score(self.score - diffand.score, self.eps)

    def __truediv__(self, divider):
        return Score(self.score / divider, self.eps)

    def __float__(self):
        return float(self.score)

    def __str__(self):
        return "{:.4f}".format(self.score)

    __repr__ = __str__

    def __eq__(self, right):
        return abs(self.score - right.score) <= self.eps


class ScoreLowerTheBetter(Score):

    def __init__(self, score, eps=10 ** -8):
        Score.__init__(self, score, eps)

    def __add__(self, right):
        return Score(self.score + right.score, self.eps)

    def __lt__(self, right):
        return self.score > right.score + self.eps

    def __gt__(self, right):
        return self.score < right.score - self.eps


class ScoreHigherTheBetter(Score):

    def __init__(self, score, eps=10 ** -8):
        Score.__init__(self, score, eps)

    def __add__(self, right):
        return Score(self.score + right.score, self.eps)

    def __lt__(self, right):
        return self.score < right.score - self.eps

    def __gt__(self, right):
        return self.score > right.score + self.eps


class ScoreFunction(object):

    def __init__(self, eps=10 ** -8):
        self.eps = eps
        self.higher_the_better = True  # default

    def set_labels(self, labels):
        self.labels = np.sort(labels)  # sklearn policy
        self.label_index_dict = dict(zip(self.labels, range(len(self.labels))))

    def set_eps(self, eps):
        self.eps = eps


class Accuracy(ScoreFunction):

    def __call__(self, true_predictions, predictions, valid_indexes=None):
        if valid_indexes is None:
            y_pred_label_index = predictions.y_pred_label_index
            y_true_label_index = true_predictions.y_pred_label_index
        else:
            y_pred_label_index = predictions.y_pred_label_index[valid_indexes]
            y_true_label_index = \
                true_predictions.y_pred_label_index[valid_indexes]
        return ScoreHigherTheBetter(
            accuracy_score(y_true_label_index, y_pred_label_index), self.eps)

    @property
    def zero(self):
        return ScoreHigherTheBetter(0.0, self.eps)

    @staticmethod
    def convert(x):
        return ScoreHigherTheBetter(x)


class Error(ScoreFunction):

    def __call__(self, true_predictions, predictions, valid_indexes=None):
        if valid_indexes is None:
            y_pred_label_index = predictions.y_pred_label_index
            y_true_label_index = true_predictions.y_pred_label_index
        else:
            y_pred_label_index = predictions.y_pred_label_index[valid_indexes]
            y_true_label_index = \
                true_predictions.y_pred_label_index[valid_indexes]
        return ScoreLowerTheBetter(
            1 - accuracy_score(y_true_label_index, y_pred_label_index),
            self.eps)

    @property
    def zero(self):
        return ScoreLowerTheBetter(0.0, self.eps)

    @staticmethod
    def convert(x):
        return ScoreLowerTheBetter(x)


class NegativeLogLikelihood(ScoreFunction):

    def __call__(self, true_predictions, predictions, valid_indexes=None):
        # We need valid_indexes because in cv bagging not all instances
        # have valid predictions.
        if valid_indexes is None:
            y_proba = predictions.y_pred
            y_true_proba = true_predictions.y_pred
        else:
            y_proba = predictions.y_pred[valid_indexes]
            y_true_proba = true_predictions.y_pred[valid_indexes]
        # Normalize rows
        y_proba_normalized = \
            y_proba / np.sum(y_proba, axis=1, keepdims=True)
        # Kaggle's rule
        y_proba_normalized = np.maximum(
            y_proba_normalized, 10 ** -15)
        y_proba_normalized = np.minimum(
            y_proba_normalized, 1 - 10 ** -15)
        scores = - np.sum(np.log(y_proba_normalized) *
                          y_true_proba,
                          axis=1)
        score = np.mean(scores)
        return ScoreLowerTheBetter(score)

    @property
    def zero(self):
        return ScoreLowerTheBetter(0.0, self.eps)

    @staticmethod
    def convert(x):
        return ScoreLowerTheBetter(x)


class RMSE(ScoreFunction):

    def __call__(self, true_predictions, predictions, valid_indexes=None):
        if valid_indexes is None:
            y_pred = predictions.y_pred
            y_true = true_predictions.y_pred
        else:
            y_pred = predictions.y_pred[valid_indexes]
            y_true = true_predictions.y_pred[valid_indexes]
        score = np.sqrt(np.mean(np.square(y_true - y_pred)))
        return ScoreLowerTheBetter(score)

    @property
    def zero(self):
        return ScoreLowerTheBetter(0.0)

    @staticmethod
    def convert(x):
        return ScoreLowerTheBetter(x)


class RegressionF1(ScoreFunction):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, true_predictions, predictions, valid_indexes=None):
        if valid_indexes is None:
            y_pred = predictions.y_pred
            y_true = true_predictions.y_pred
        else:
            y_pred = predictions.y_pred[valid_indexes]
            y_true = true_predictions.y_pred[valid_indexes]
        score = f1_score(y_true > self.threshold, y_pred > self.threshold)
        return ScoreHigherTheBetter(score)

    @property
    def zero(self):
        return ScoreHigherTheBetter(0.0)

    @staticmethod
    def convert(x):
        return ScoreHigherTheBetter(x)

