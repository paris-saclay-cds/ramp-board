# Author: Balazs Kegl
# License: BSD 3 clause

import os
import sys
import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.cross_validation import StratifiedShuffleSplit
# menu
import databoard.scores as scores
# menu polymorphism example
import databoard.multiclass_prediction as multiclass_prediction
from databoard.multiclass_prediction import Predictions
from databoard.config import config_object, submissions_path
from databoard.config import public_data_path, private_data_path

sys.path.append(os.path.dirname(os.path.abspath(submissions_path)))

ramp_title = 'Kaggle Otto product classification'
target_column_name = 'target'
multiclass_prediction.labels = ['Class_1', 'Class_2', 'Class_3', 'Class_4',
                                'Class_5', 'Class_6', 'Class_7', 'Class_8',
                                'Class_9']

cv_test_size = config_object.cv_test_size
random_state = config_object.random_state
n_CV = config_object.num_cpus

train_filename = os.path.join(public_data_path, 'train.csv')
test_filename = os.path.join(private_data_path, 'test.csv')

score = scores.NegativeLogLikelihood()
# score = scores.Accuracy()


def read_data(df_filename):
    df = pd.read_csv(df_filename, index_col=0)  # this drops the id actually
    y_array = df[target_column_name].values
    X_dict = df.drop(target_column_name, axis=1).to_dict(orient='records')
    return X_dict, y_array


def prepare_data():
    pass
    # train and tes splits are given


def get_train_data():
    X_train_dict, y_train_array = read_data(train_filename)
    return X_train_dict, y_train_array


def get_test_data():
    X_test_dict, y_test_array = read_data(test_filename)
    return X_test_dict, y_test_array


def get_cv(y_train_array):
    cv = StratifiedShuffleSplit(
        y_train_array, n_iter=n_CV,
        test_size=cv_test_size, random_state=random_state)
    return cv


def train_submission(module_path, X_dict, y_array, train_is):
    # Preparing the training set
    X_train_dict = [X_dict[i] for i in train_is]
    y_train_array = np.array([y_array[i] for i in train_is])

    # Feature extraction
    feature_extractor = import_module('.feature_extractor', module_path)
    fe = feature_extractor.FeatureExtractor()
    fe.fit(X_train_dict, y_train_array)
    X_train_array = fe.transform(X_train_dict)

    classifier = import_module('.classifier', module_path)
    clf = classifier.Classifier()

    # Calibration
    try:
        calibrator = import_module('.calibrator', module_path)
        calib = calibrator.Calibrator()

        # Train/valid cut for holding out calibration set
        cv = StratifiedShuffleSplit(
            y_train_array, n_iter=1, test_size=0.1, random_state=57)
        calib_train_is, calib_test_is = list(cv)[0]

        X_train_train_array = X_train_array[calib_train_is]
        y_train_train_array = y_train_array[calib_train_is]
        X_calib_train_array = X_train_array[calib_test_is]
        y_calib_train_array = y_train_array[calib_test_is]

        # Classification
        clf = classifier.Classifier()
        clf.fit(X_train_train_array, y_train_train_array)

        # Calibration
        y_probas_array = clf.predict_proba(X_calib_train_array)
        calib.fit(y_probas_array, y_calib_train_array)
        return fe, clf, calib
    except ImportError:
        # Classification
        clf.fit(X_train_array, y_train_array)
        return fe, clf


def test_submission(trained_model, X_dict, test_is):
    # Preparing the test (or valid) set
    X_test_dict = [X_dict[i] for i in test_is]

    if len(trained_model) == 3:  # calibrated classifier
        fe, clf, calib = trained_model

        # Feature extraction
        X_test_array = fe.transform(X_test_dict)

        # Classification
        y_probas_array = clf.predict_proba(X_test_array)

        # Calibration
        y_calib_probas_array = calib.predict_proba(y_probas_array)
        return Predictions(y_pred=y_calib_probas_array)

    else:  # uncalibrated classifier
        fe, clf = trained_model

        # Feature extraction
        X_test_array = fe.transform(X_test_dict)

        # Classification
        y_probas_array = clf.predict_proba(X_test_array)
        return Predictions(y_pred=y_probas_array)
