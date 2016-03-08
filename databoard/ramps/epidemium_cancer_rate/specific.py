# Author: Balazs Kegl
# License: BSD 3 clause

import os
import sys
import pandas as pd
from importlib import import_module
from sklearn.cross_validation import ShuffleSplit, train_test_split
# menu
import databoard.scores as scores
# menu polymorphism example
from databoard.regression_prediction import Predictions
from databoard.config import config_object, submissions_path, raw_data_path
from databoard.config import public_data_path, private_data_path

sys.path.append(os.path.dirname(os.path.abspath(submissions_path)))

ramp_title = 'Epidemium cancer rate prediction'
target_column_name = 'target'

cv_test_size = config_object.cv_test_size
held_out_test_size = 0.4
public_train_size = 0.5  # so public train and train have the same size
random_state = config_object.random_state
n_CV = config_object.n_cpus

raw_filename = os.path.join(raw_data_path, 'data.csv')
public_train_filename = os.path.join(public_data_path, 'public_train.csv')
train_filename = os.path.join(private_data_path, 'train.csv')
test_filename = os.path.join(private_data_path, 'test.csv')

score = scores.RMSE()

workflow_element_types = [
    {'name': 'feature_extractor'},
    {'name': 'regressor'},
]


def prepare_data():
    df = pd.read_csv(raw_filename)
    df_train, df_test = train_test_split(
        df, test_size=held_out_test_size, random_state=random_state)
#    df_public_train, df_private_train = train_test_split(
#        df_train, train_size=public_train_size, random_state=random_state)
    df_train.to_csv(train_filename, index=False)
#    df_private_train.to_csv(train_filename, index=False)
    # test data was too big
#    _, df_test = train_test_split(
#        df_test, test_size=0.5, random_state=random_state)
    df_test.to_csv(test_filename, index=False)


def read_data(filename):
    data = pd.read_csv(filename)
    y_array = data[target_column_name].values
    X_df = data.drop([target_column_name], axis=1)
    return X_df, y_array


def get_train_data():
    X_train_df, y_train_array = read_data(train_filename)
    return X_train_df, y_train_array


def get_test_data():
    X_test_df, y_test_array = read_data(test_filename)
    return X_test_df, y_test_array


def get_cv(y_train_array):
    cv = ShuffleSplit(
        len(y_train_array), n_iter=n_CV, test_size=cv_test_size,
        random_state=random_state)
    return cv


def train_submission(module_path, X_df, y_array, train_is):
    # Feature extraction
    feature_extractor = import_module('.feature_extractor', module_path)
    fe = feature_extractor.FeatureExtractor()
    fe.fit(X_df.iloc[train_is], y_array[train_is])
    X_array = fe.transform(X_df.iloc[train_is])
    # Regression
    regressor = import_module('.regressor', module_path)
    reg = regressor.Regressor()
    reg.fit(X_array, y_array[train_is])
    return fe, reg


def test_submission(trained_model, X_df, test_is):
    fe, reg = trained_model
    # Feature extraction
    X_array = fe.transform(X_df.iloc[test_is])
    # Regression
    y_pred = reg.predict(X_array)
    return Predictions(y_pred=y_pred)
