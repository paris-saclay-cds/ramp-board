import databoard.specific.score_types.mare as mare


def score_function(true_predictions, predictions, valid_indexes=None):
    return mare.score_function(
        true_predictions.regression_prediction,
        predictions.regression_prediction,
        valid_indexes)

# default display precision in n_digits
precision = 2
