# Author: Balazs Kegl
# License: BSD 3 clause

import os
import sys
import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
# menu
import scores
import multiclass_prediction_type as prediction_type # menu polymorphism example
import config_databoard

sys.path.append(os.path.dirname(os.path.abspath(config_databoard.models_path)))

hackaton_title = 'IRIS'
prediction_type.labels = ["setosa",  'versicolor',  "virginica"]

cv_test_size = config_databoard.get_ramp_field('cv_test_size')
random_state = config_databoard.get_ramp_field('random_state')
n_CV = config_databoard.get_ramp_field('num_cpus')

raw_filename = os.path.join(config_databoard.raw_data_path, 'iris.csv')
train_filename = os.path.join(config_databoard.public_data_path, 'train.csv')
test_filename = os.path.join(config_databoard.private_data_path, 'test.csv')

score = scores.Accuracy()

# X is a list of dicts, each dict is indexed by column
def read_data(filename):
    data = pd.read_csv(filename)
    y_array = data['species'].values
    X_array = data.drop(['species'], axis=1).values
    return X_array, y_array

def prepare_data():
    X_array, y_array = read_data(raw_filename)
    cv = StratifiedShuffleSplit(
        y_array, 1, test_size=0.2, random_state=random_state)
    train_is, test_is = list(cv)[0]
    X_train_array = X_array[train_is]
    X_test_array = X_array[test_is]
    y_train_array = y_array[train_is]
    y_test_array = y_array[test_is]
    np.savez(train_filename, X=X_train_array, y=y_train_array)
    np.savez(test_filename, X=X_test_array, y=y_test_array)

def get_train_data():
    X_train_array, y_train_array = read_data(train_filename)
    return X_train_array, y_train_array

def get_test_data():
    X_test_array, y_test_array = read_data(test_filename)
    return X_test_array, y_test_array

def get_cv(y_train_array):
    cv = StratifiedShuffleSplit(y_train_array,
        n_iter=n_CV, test_size=cv_test_size, random_state=random_state)
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
    y_probas_array = clf.predict_proba(X_array[test_is])
    return prediction_type.PredictionArrayType(y_prediction_array=y_probas_array)
