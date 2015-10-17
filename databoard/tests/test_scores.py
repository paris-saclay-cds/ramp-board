# Author: Balazs Kegl
# License: BSD 3 clause

from numpy.testing import assert_equal
from .. import scores
from .. import multiclass_prediction_type

multiclass_prediction_type.labels = [0, 1]


def test_score_accuracy():
    prediction_array_1 = multiclass_prediction_type.PredictionArrayType(
        y_pred_label_array=[[0], [0], [0], [1], [1], [1]])
    prediction_array_2 = multiclass_prediction_type.PredictionArrayType(
        y_pred_label_array=[[0], [0], [0], [1], [1], [1]])
    score_function = scores.Accuracy()
    score = score_function(prediction_array_1, prediction_array_2)
    assert_equal(score, scores.ScoreHigherTheBetter(1.0))
    assert_equal(float(score), 1.0)


def test_score_error():
    prediction_array_1 = multiclass_prediction_type.PredictionArrayType(
        y_pred_label_array=[[0], [0], [0], [1], [1], [1]])
    prediction_array_2 = multiclass_prediction_type.PredictionArrayType(
        y_pred_label_array=[[0], [0], [0], [1], [1], [1]])
    score_function = scores.Error()
    score = score_function(prediction_array_1, prediction_array_2)
    assert_equal(score, scores.ScoreLowerTheBetter(0.0))
    assert_equal(float(score), 0.0)

test_score_accuracy()
test_score_error()
