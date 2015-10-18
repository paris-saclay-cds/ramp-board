"""All score classes have a score function taking a ground truth vector
    and a prediction vector and returning a score; and a boolean
    higher_the_better."""

# Author: Balazs Kegl
# License: BSD 3 clause

import numpy as np
from sklearn.metrics import accuracy_score


class Score(object):

    def __init__(self, score, eps=10 ** -8):
        self.score = score
        self.eps = eps
        self.dtype = np.dtype(np.float)

    def __add__(self, right):
        return Score(self.score + right.score, self.eps)

    def __div__(self, divider):
        return Score(self.score / divider, self.eps)

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

    def __call__(self, ground_truth, predictions):
        y_pred_array = predictions.get_pred_index_array()
        ground_truth_pred_array = ground_truth.get_pred_index_array()
        # print y_pred_array
        # print ground_truth_pred_array
        # print accuracy_score(ground_truth_pred_array, y_pred_array)
        # print "\n"
        return ScoreHigherTheBetter(
            accuracy_score(ground_truth_pred_array, y_pred_array), self.eps)

    def zero(self):
        return ScoreHigherTheBetter(0.0, self.eps)


class Error(ScoreFunction):

    def __call__(self, ground_truth, predictions):
        y_pred_array = predictions.get_pred_index_array()
        ground_truth_pred_array = ground_truth.get_pred_index_array()
        return ScoreLowerTheBetter(
            1 - accuracy_score(ground_truth_pred_array, y_pred_array),
            self.eps)

    def zero(self):
        return ScoreLowerTheBetter(0.0, self.eps)


class NegativeLogLikelihood(ScoreFunction):

    def __call__(self, ground_truth, predictions):
        y_probas_array = predictions.get_prediction_array()
        ground_truth_probas_array = ground_truth.get_prediction_array()
        # Normalize rows
        y_probas_array_normalized = \
            y_probas_array / np.sum(y_probas_array, axis=1, keepdims=True)
        # Kaggle's rule
        y_probas_array_normalized = np.maximum(
            y_probas_array_normalized, 10 ** -15)
        y_probas_array_normalized = np.minimum(
            y_probas_array_normalized, 1 - 10 ** -15)
        scores = - np.sum(np.log(y_probas_array_normalized) *
                          ground_truth_probas_array,
                          axis=1)
        score = np.mean(scores)
        return ScoreLowerTheBetter(score)

    def zero(self):
        return ScoreLowerTheBetter(0.0, self.eps)


class RMSE(ScoreFunction):

    def __call__(self, ground_truth_list, predictions):
        y_pred_array = predictions.get_predictions()
        score = np.sqrt(np.mean(np.square(ground_truth_list - y_pred_array)))
        return ScoreLowerTheBetter(score)

    def zero(self):
        return ScoreLowerTheBetter(0.0, self.eps)
