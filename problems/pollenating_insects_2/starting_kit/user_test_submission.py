# coding=utf-8
import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from batch_classifier_workflow import train_submission
from batch_classifier_workflow import test_submission
from batch_classifier_workflow import ArrayContainer

attrs = {
    'chunk_size': 256,
    'n_jobs': 8,
    'test_batch_size': 16,
    'folder': 'imgs',
    'n_classes': 209
}


def read_data(filename):
    df = pd.read_csv(filename)
    X_values = df['id'].values
    X = ArrayContainer(X_values, attrs=attrs)
    y = df['class'].values
    return X, y


def get_cv(y_train_array):
    return StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=43)

score_function = accuracy_score


def score_function_2(y_true, y_pred):
    """Rate of classes with f1 score above 0.5."""
    f1 = f1_score(y_true, y_pred, average=None)
    score = 1. * len(f1[f1 > 0.5]) / len(f1)
    return score


if __name__ == '__main__':
    # PIL image loading logger is a bit annoying, so disable it.
    logging.getLogger('PIL.PngImagePlugin').disabled = True
    X, y = read_data('train.csv')
    cv = get_cv(y)
    for i, (train_is, test_is) in enumerate(cv.split(X, y)):
        print("Training ...")
        trained_model = train_submission('', X[train_is], y[train_is])
        print("Testing ...")
        y_pred = test_submission(trained_model, X[test_is])
        score = score_function(y[test_is], y_pred.argmax(axis=1))
        np.savez(
            'pred_{}.npz'.format(i), y_true=y[test_is],
            y_pred=y_pred.argmax(axis=1))
        print('accuracy = ', score)
        score_2 = score_function_2(y[test_is], y_pred.argmax(axis=1))
        print('rate of classes with f1 score > 0.5 = ', score_2)
