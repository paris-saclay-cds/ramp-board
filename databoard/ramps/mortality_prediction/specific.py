# Author: Balazs Kegl
# License: BSD 3 clause

import os
import sys
import pandas as pd
from importlib import import_module
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
# menu
import databoard.scores as scores
# menu polymorphism example
import databoard.multiclass_prediction as multiclass_prediction
from databoard.multiclass_prediction import Predictions
from databoard.config import config_object, submissions_path
from databoard.config import raw_data_path, public_data_path, private_data_path

sys.path.append(os.path.dirname(os.path.abspath(submissions_path)))

ramp_title = 'Mortality prediction'
multiclass_prediction.labels = ['0.0', '1.0']
target_column_name = 'TARGET'

cv_test_size = config_object.cv_test_size
held_out_test_size = 0.2
random_state = config_object.random_state
n_CV = config_object.n_cpus

raw_filename = os.path.join(raw_data_path, 'data.csv')
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


def train_submission(module_path, X_array, y_array, train_is):
    classifier = import_module('.classifier', module_path)
    clf = classifier.Classifier()
    clf.fit(X_array[train_is], y_array[train_is])
    return clf


def test_submission(trained_model, X_array, test_is):
    clf = trained_model
    y_probas_array = clf.predict_proba(X_array[test_is])
    return Predictions(y_pred=y_probas_array)
