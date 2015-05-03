"""All score classes have a score function taking a ground truth vector
    and a prediction vector and returning a score; and a boolean 
    higher_the_better."""

# Author: Balazs Kegl
# License: BSD 3 clause

import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc

class ScoreObject():
    def __init__(self, score, eps = 10 ** -8):
        self.score = score
        self.eps = eps
        self.dtype = np.dtype(np.float)

    def __add__(self, right):
        return ScoreObject(self.score + right.score, self.eps)

    def __div__(self, divider):
        return ScoreObject(self.score / divider, self.eps)

    def __truediv__(self, divider):
        return ScoreObject(self.score / divider, self.eps)

    def __float__(self):
        return float(self.score)

    def __str__(self):
        return "{:.4f}".format(self.score)
    
    __repr__ = __str__

    def __eq__(self, right):
        return abs(self.score - right.score) <= self.eps

class ScoreLowerTheBetter(ScoreObject):
    def __init__(self, score, eps = 10 ** -8):
        ScoreObject.__init__(self, score, eps)

    def __add__(self, right):
        return ScoreObject(self.score + right.score, self.eps)

    def __lt__(self, right):
        return self.score > right.score + self.eps

    def __gt__(self, right):
        return self.score < right.score - self.eps

class ScoreHigherTheBetter(ScoreObject):
    def __init__(self, score, eps = 10 ** -8):
        ScoreObject.__init__(self, score, eps)

    def __add__(self, right):
        return ScoreObject(self.score + right.score, self.eps)

    def __lt__(self, right):
        return self.score < right.score - self.eps

    def __gt__(self, right):
        return self.score > right.score + self.eps

class ScoreFunction():
    def __init__(self, eps=10 ** -8):
        self.eps = eps
        self.higher_the_better = True #default

    def set_labels(self, labels):
        self.labels = np.sort(labels) # sklearn policy
        self.label_index_dict = dict(zip(self.labels, range(len(self.labels))))

    def set_eps(self, eps):
        self.eps = eps

class Accuracy(ScoreFunction):
    def __call__(self, ground_truth_list, predictions):
        y_pred_array, y_probas_array = predictions.get_predictions()
        return ScoreHigherTheBetter(
            accuracy_score(ground_truth_list, y_pred_array), self.eps)

    def zero(self):
        return ScoreHigherTheBetter(0.0, self.eps)

class Error(ScoreFunction):
    def __call__(self, ground_truth_list, predictions):
        y_pred_array, y_probas_array = predictions.get_predictions()
        return ScoreLowerTheBetter(
            1 - accuracy_score(ground_truth_list, y_pred_array), self.eps)

    def zero(self):
        return ScoreLowerTheBetter(0.0, self.eps)

class NegativeLogLikelihood(ScoreFunction):
    def __call__(self, ground_truth_list, predictions):
        y_pred_array, y_probas_array = predictions.get_predictions()
        ground_truth_index_list = [self.label_index_dict[ground_truth] 
                                   for ground_truth in ground_truth_list]
        # Normalize rows
        y_probas_array_normalized = \
            y_probas_array / np.sum(y_probas_array, axis=1, keepdims=True)
        probas = np.array([y_probas[ground_truth_index] 
                           for y_probas, ground_truth_index 
                           in zip(y_probas_array_normalized, ground_truth_index_list)])
        # Kaggle's rule
        probas = np.array([max(10 ** -15 , min(1 - 10 ** -15, p))
                           for p in probas])
        score = - np.mean(np.log(probas))
        return ScoreLowerTheBetter(score)

    def zero(self):
        return ScoreLowerTheBetter(0.0, self.eps)
