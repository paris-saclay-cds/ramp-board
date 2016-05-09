import databoard.specific.score_types.error as error


def score_function(true_predictions, predictions, valid_indexes=None):
    return error.score_function(
        true_predictions.multiclass_prediction,
        predictions.multiclass_prediction,
        valid_indexes)

# default display precision in n_digits
precision = 2
