from __future__ import division, print_function

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from classifier import Classifier


def nll(y_true, y_proba):
    y_true_proba = np.array([1 - y_true, y_true]).T
    # Normalize rows
    y_proba_normalized = y_proba / np.sum(y_proba, axis=1, keepdims=True)
    # Kaggle's rule
    y_proba_normalized = np.maximum(y_proba_normalized, 10 ** -15)
    y_proba_normalized = np.minimum(y_proba_normalized, 1 - 10 ** -15)
    scores = - np.sum(np.log(y_proba_normalized) * y_true_proba, axis=1)
    score = np.mean(scores)
    return score


if __name__ == '__main__':
    print("Reading file ...")
    data = pd.read_csv('public_train.csv.gz', compression='gzip')
    y = data['isSkewed'].values
    X = data.drop(['isSkewed'], axis=1).values

    skf = StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=43)
    print("Training ...")
    for valid_train_is, valid_test_is in skf.split(X, y):
        print('-------------------------------------------------------------')

        X_valid_train = X[valid_train_is]
        y_valid_train = y[valid_train_is]
        X_valid_test = X[valid_test_is]
        y_valid_test = y[valid_test_is]

        clf = Classifier()
        clf.fit(X_valid_train, y_valid_train)
        y_valid_pred = clf.predict_proba(X_valid_test)
        print('accuracy = ', accuracy_score(
            y_valid_test, y_valid_pred[:, 1] > y_valid_pred[:, 0]))
        print('ROC AUC = ', roc_auc_score(y_valid_test, y_valid_pred[:, 1]))
        print('nll = ', nll(y_valid_test, y_valid_pred))
