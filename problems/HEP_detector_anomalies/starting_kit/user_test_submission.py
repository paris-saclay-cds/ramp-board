from __future__ import division, print_function

import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from classifier import Classifier

if __name__ == '__main__':
    print("Reading file ...")
    data = pd.read_csv('public_train.csv.gz', compression='gzip')
    y = data[['isSkewed']].values.ravel()
    X = data.drop(['isSkewed'], axis=1).reset_index(drop=True)

    skf = StratifiedShuffleSplit(y, n_iter=2, test_size=0.5, random_state=43)
    print("Training ...")
    for valid_train_is, valid_test_is in skf:
        print('-------------------------------------------------------------')

        X_valid_train = X.ix[valid_train_is]
        y_valid_train = y[valid_train_is]
        X_valid_test = X.ix[valid_test_is]
        y_valid_test = y[valid_test_is]

        clf = Classifier()
        clf.fit(X_valid_train, y_valid_train)
        y_valid_pred = clf.predict_proba(X_valid_test)
        print('accuracy = ', accuracy_score(
            y_valid_test, y_valid_pred[:, 1] > y_valid_pred[:, 0]))
        print('ROC AUC = ', roc_auc_score(y_valid_test, y_valid_pred[:, 1]))
