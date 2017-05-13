from sklearn.metrics import roc_auc_score


def score_function(true_predictions, predictions, valid_indexes=None):
    if valid_indexes is None:
        valid_indexes = range(true_predictions.n_samples)
    y_proba = predictions.y_pred[valid_indexes]
    y_true_proba = true_predictions.y_pred_label_index[valid_indexes]
    score = roc_auc_score(y_true_proba, y_proba[:, 1])
    return score

# default display precision in n_digits
precision = 2
