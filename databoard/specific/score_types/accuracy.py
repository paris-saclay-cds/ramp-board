from sklearn.metrics import accuracy_score


def score_function(true_predictions, predictions, valid_indexes=None):
    if valid_indexes is None:
        valid_indexes = range(true_predictions.n_samples)
    y_pred_label_index = predictions.y_pred_label_index[valid_indexes]
    y_true_label_index = \
        true_predictions.y_pred_label_index[valid_indexes]
    score = accuracy_score(y_true_label_index, y_pred_label_index)
    return score

# default display precision in n_digits
precision = 2
