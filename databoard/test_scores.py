# Author: Balazs Kegl
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_equal
from scores import ScoreAuc, ScoreAccuracy, ScoreError


def test_score_auc():
    y_test_list = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred_list = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    score_function = ScoreAuc()
    score = score_function.score(y_test_list, y_pred_list)
    assert_equal(score, 1.0)

    y_test_list = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred_list = np.array([5, 6, 7, 8, 1, 2, 3, 4])
    score_function = ScoreAuc()
    score = score_function.score(y_test_list, y_pred_list)
    assert_equal(score, 0.0)


def test_score_accuracy():
    y_test_list = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred_list = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score_function = ScoreAccuracy()
    score = score_function.score(y_test_list, y_pred_list)
    assert_equal(score, 1.0)


def test_score_error():
    y_test_list = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred_list = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score_function = ScoreError()
    score = score_function.score(y_test_list, y_pred_list)
    assert_equal(score, 0.0)

test_score_auc()
test_score_accuracy()
test_score_error()
