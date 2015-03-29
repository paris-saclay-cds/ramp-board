"""All score classes have a score function taking a ground truth vector
    and a prediction vector and returning a score; and a boolean 
    higher_the_better."""

# Author: Balazs Kegl
# License: BSD 3 clause

from sklearn.metrics import accuracy_score, roc_curve, auc

class Score():
    def __init__(self):
        self.higher_the_better = True #default

class ScoreAuc(Score):
    def score(self, y_test_list, y_pred_list):
        fpr, tpr, _ = roc_curve(y_test_list, y_pred_list)
        return auc(fpr, tpr)

class ScoreAccuracy(Score):
    def score(self, y_test_list, y_pred_list):
        return accuracy_score(y_test_list, y_pred_list)

class ScoreError(Score):
    def __init__(self):
        self.higher_the_better = False

    def score(self, y_test_list, y_pred_list):
        return 1 - accuracy_score(y_test_list, y_pred_list)
