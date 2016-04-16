import numpy as np


def score_function(true_predictions, predictions, valid_indexes=None):
    if valid_indexes is None:
        y_pred = predictions.y_pred
        y_true = true_predictions.y_pred
    else:
        y_pred = predictions.y_pred[valid_indexes]
        y_true = true_predictions.y_pred[valid_indexes]
    score = np.sqrt(np.mean(np.square(y_true - y_pred)))
    return score

# default display precision in n_digits
precision = 2
