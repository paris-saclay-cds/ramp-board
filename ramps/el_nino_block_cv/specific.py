# Author: Balazs Kegl
# License: BSD 3 clause

import os
import sys
import xray
import numpy as np
import pandas as pd
from importlib import import_module
# menu
# menu polymorphism example
from .regression_prediction import Predictions
import scores
from .config import config, submissions_path
from .config import raw_data_path, public_data_path, private_data_path

sys.path.append(os.path.dirname(os.path.abspath(submissions_path)))

hackaton_title = 'El Nino prediction'
target_column_name = 'target'
cv_test_size = config.cv_test_size
cv_bag_size = config.cv_bag_size
random_state = config.random_state
n_CV = config.num_cpus

en_lat_bottom = -5
en_lat_top = 5
en_lon_left = 360 - 170
en_lon_right = 360 - 120
n_burn_in = 120
n_lookahead = 6

train_filename = os.path.join(public_data_path,
    'resampled_tas_Amon_CCSM4_piControl_r1i1p1_080001-130012.nc')
test_filename = os.path.join(private_data_path,
    'resampled_tas_Amon_CCSM4_piControl_r2i1p1_095301-110812.nc')

score = scores.RMSE()


def get_enso_mean(tas):
    return tas.loc[:, en_lat_bottom:en_lat_top, en_lon_left:en_lon_right].mean(
        dim=('lat', 'lon'))


def read_data(xray_filename):
    temperatures_xray = xray.open_dataset(xray_filename, decode_times=False)
    # ridiculous as it sounds, there is simply no way to convert a date
    # starting with the year 800 into pd array
    temperatures_xray['time'] = pd.date_range(
        '1/1/1700', periods=temperatures_xray['time'].shape[0], freq='M') - \
        np.timedelta64(15, 'D')
    return temperatures_xray


def prepare_data():
    pass
    # train and tes splits are given


def get_train_data():
    X_train_xray = read_data(train_filename)
    y_train_array = X_train_xray[target_column_name].values[
        n_burn_in:-n_lookahead]
    return X_train_xray, y_train_array


def get_test_data():
    X_test_xray = read_data(test_filename)
    y_test_array = X_test_xray[target_column_name].values[
        n_burn_in:-n_lookahead]
    return X_test_xray, y_test_array


def get_check_data():
    X_test_xray = read_data(test_filename)
    y_test_array = X_test_xray[target_column_name].values[
        n_burn_in:-n_lookahead]
    return X_test_xray, y_test_array


def get_cv(y_train_array):
    print y_train_array.shape
    n = y_train_array.shape[0]
    test_start = int(n * (1 - cv_test_size))
    n_test = n - test_start
    block_size = int(n_test / n_CV)
    cv = []
    for i in range(n_CV):
        train_is = np.arange(min(n, test_start + i * block_size))
        test_is = np.arange(min(n, test_start + i * block_size), n)
        cv.append((train_is, test_is))
    return cv


def check_model(module_path, X_xray, y_array, cv_is):
    check_index = 250
    feature_extractor = import_module('.ts_feature_extractor', module_path)
    ts_fe = feature_extractor.FeatureExtractor()
    X1 = ts_fe.transform(X_xray, n_burn_in, n_lookahead, cv_is)
    check_xray = X_xray.copy(deep=True)
    check_xray['tas'][n_burn_in + check_index:] += \
        np.random.normal(
            0.0, 10.0, check_xray['tas'][n_burn_in + check_index:].shape)
    X2 = ts_fe.transform(check_xray, n_burn_in, n_lookahead, cv_is)
    first_modified_index = np.argmax(np.not_equal(X1, X2)[:, 0])
    if first_modified_index < check_index:
        message = "The feature extractor looks into the future by {} months".\
            format(check_index - first_modified_index)
        raise AssertionError(message)


def train_model(module_path, X_xray, y_array, cv_is):
    X_xray = X_xray.copy(deep=True)
    # Feature extraction
    feature_extractor = import_module('.ts_feature_extractor', module_path)
    ts_fe = feature_extractor.FeatureExtractor()
    X_array = ts_fe.transform(X_xray, n_burn_in, n_lookahead, cv_is)
    # Regression
    train_is, _ = cv_is
    X_train_array = np.array([X_array[i] for i in train_is])
    y_train_array = np.array([y_array[i] for i in train_is])
    regressor = import_module('.regressor', module_path)
    reg = regressor.Regressor()
    reg.fit(X_train_array, y_train_array)
    return ts_fe, reg


def test_model(trained_model, X_xray, cv_is):
    X_xray = X_xray.copy(deep=True)
    ts_fe, reg = trained_model
    # Feature extraction
    X_array = ts_fe.transform(X_xray, n_burn_in, n_lookahead, cv_is)
    # Regression
    _, test_is = cv_is
    X_test_array = np.array([X_array[i] for i in test_is])
    y_prediction_array = reg.predict(X_test_array)
    return Predictions(y_pred=y_prediction_array)
