# Author: Balazs Kegl
# License: BSD 3 clause

import os
import sys
import pandas as pd
from importlib import import_module
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
# menu
import scores
# menu polymorphism example
import multiclass_prediction
from .multiclass_prediction import Predictions
from .config import config, submissions_path, raw_data_path, public_data_path
from .config import private_data_path
# import config_databoard

# sys.path.append(os.path.dirname(os.path.abspath(config_databoard.submissions_path)))
sys.path.append(os.path.dirname(os.path.abspath(submissions_path)))

hackaton_title = 'Iris classification (test)'
multiclass_prediction.labels = ['setosa', 'versicolor', 'virginica']
target_column_name = 'species'

cv_test_size = config.cv_test_size
held_out_test_size = 0.2
random_state = config.random_state
n_CV = config.num_cpus  # XXX

raw_filename = os.path.join(raw_data_path, 'iris.csv')
train_filename = os.path.join(public_data_path, 'train.csv')
test_filename = os.path.join(private_data_path, 'test.csv')

score = scores.Accuracy()


def read_data(filename):
    data = pd.read_csv(filename)
    y_array = data[target_column_name].values
    X_array = data.drop([target_column_name], axis=1).values
    return X_array, y_array


def prepare_data():
    df = pd.read_csv(raw_filename)
    df_train, df_test = train_test_split(
        df, test_size=held_out_test_size, random_state=random_state)
    df_train.to_csv(train_filename)
    df_test.to_csv(test_filename)


def get_train_data():
    X_train_array, y_train_array = read_data(train_filename)
    return X_train_array, y_train_array


def get_test_data():
    X_test_array, y_test_array = read_data(test_filename)
    return X_test_array, y_test_array


def get_cv(y_train_array):
    cv = StratifiedShuffleSplit(
        y_train_array, n_iter=n_CV, test_size=cv_test_size,
        random_state=random_state)
    return cv


def train_model(module_path, X_array, y_array, cv_is):
    train_is, _ = cv_is
    classifier = import_module('.classifier', module_path)
    clf = classifier.Classifier()
    clf.fit(X_array[train_is], y_array[train_is])
    return clf


def test_model(trained_model, X_array, cv_is):
    _, test_is = cv_is
    clf = trained_model
    y_proba = clf.predict_proba(X_array[test_is])
    return Predictions(y_pred=y_proba)
