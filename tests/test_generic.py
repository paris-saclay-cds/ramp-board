import numpy as np
from numpy.testing import assert_array_equal
from generic import combine_models


def test_combine_models():
    y_preds = np.array([[1, 1],
                        [0, 0]])
    y_ranks = np.array([[0, 1],
                        [0, 1]])
    indexes = np.array([0, 1])

    out = combine_models(y_preds, y_ranks, indexes)
    assert_array_equal(out, [0, 1])
