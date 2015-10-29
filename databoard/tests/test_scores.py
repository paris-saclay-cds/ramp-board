# Author: Balazs Kegl
# License: BSD 3 clause

from numpy.testing import assert_equal
from .. import scores
from .. import multiclass_prediction
from ..multiclass_prediction import Predictions

multiclass_prediction.labels = [0, 1]


def test_score_accuracy():
    predictions_1 = Predictions(y_pred_labels=[[0], [0], [0], [1], [1], [1]])
    predictions_2 = Predictions(y_true=[[0], [0], [0], [1], [1], [1]])
    score_function = scores.Accuracy()
    score = score_function(predictions_1, predictions_2)
    assert_equal(score, scores.ScoreHigherTheBetter(1.0))
    assert_equal(float(score), 1.0)


def test_score_error():
    predictions_1 = Predictions(y_pred_labels=[[0], [0], [0], [1], [1], [1]])
    predictions_2 = Predictions(y_true=[[0], [0], [0], [1], [1], [1]])
    score_function = scores.Error()
    score = score_function(predictions_1, predictions_2)
    assert_equal(score, scores.ScoreLowerTheBetter(0.0))
    assert_equal(float(score), 0.0)

test_score_accuracy()
test_score_error()
