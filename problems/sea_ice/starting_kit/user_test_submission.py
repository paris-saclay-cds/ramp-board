# coding=utf-8

import numpy as np
import pandas as pd
import xarray as xr
from importlib import import_module

# 10 years; also set in the Dataset file so fe has access
n_burn_in = 120
# y_train array is blocked in the following way:
# n_burn_in | n_common_block | n_cv x block_size
# first cv fold takes 0 blocks (so only n_burn_in and n_common_block)
# last cv fold takes n_burn_in + n_common_block + (n_cv - 1) x block_size
# block_size should be divisible by 12 if score varies within year ->
# n_validation = n_train - int(n_train / 2) = n_train / 2
# should be divisible by 12 * n_cv -> n_train should be
# multiply of 24 * n_cv
# n_train should also be > 2 * n_burn_in
n_cv = 8
n_train = 24 * n_cv * 12  # 24 x n_cv years


def get_cv(y_train_array):
    n = len(y_train_array)
    n_common_block = int(n / 2) - n_burn_in
    n_validation = n - n_common_block - n_burn_in
    block_size = int(n_validation / n_cv)
    print 'length of burn in:', n_burn_in, 'months =',\
        n_burn_in / 12, 'years'
    print 'length of common block:', n_common_block, 'months =',\
        n_common_block / 12, 'years'
    print 'length of validation block:', n_validation, 'months =',\
        n_validation / 12, 'years'
    print 'length of each cv block:', block_size, 'months =',\
        block_size / 12, 'years'
    for i in range(n_cv):
        train_is = np.arange(0, n_burn_in + n_common_block + i * block_size)
        test_is = np.arange(n_burn_in + n_common_block + i * block_size, n)
        yield (train_is, test_is)


def score(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def read_data():
    X_ds = xr.open_dataset('sea_ice_X_public_train.nc', decode_times=False)
    y_array = np.load('sea_ice_y_public_train.npy')
    return X_ds, y_array


def get_train_data():
    X_ds, y_array = read_data()
    X_train_ds = X_ds.isel(time=slice(None, n_train))
    y_train_array = y_array[:n_train]
    print 'length of training array:', len(y_train_array), 'months =',\
        len(y_train_array) / 12, 'years'
    return X_train_ds, y_train_array


def get_test_data():
    X_ds, y_array = read_data()
    X_test_ds = X_ds.isel(time=slice(n_train, None))
    y_test_array = y_array[n_train:]
    print 'length of test array:', len(y_test_array), 'months =',\
        len(y_test_array) / 12, 'years'
    return X_test_ds, y_test_array


def train_submission(module_path, X_ds, y_array, train_is):
    n_burn_in = X_ds.attrs['n_burn_in']
    X_train_ds = X_ds.isel(time=train_is)
    y_train_array = y_array[train_is]

    # X_xray = X_xray.copy(deep=True)
    # Feature extraction
    ts_feature_extractor = import_module('ts_feature_extractor', module_path)
    ts_fe = ts_feature_extractor.FeatureExtractor()
    X_train_array = ts_fe.transform(X_train_ds)

    # Checking if feature extractor looks ahead: we change the input array
    # after index n_burn_in + check_index, and check if the first
    # check_features have changed
    check_size = 130
    check_index = 120
    X_check_ds = X_ds.isel(time=slice(0, n_burn_in + check_size))
    # Assigning Dataset slices is not yet supported so we need to iterate
    # over the arrays
    data_var_names = X_check_ds.data_vars.keys()
    for data_var_name in data_var_names:
        X_check_ds[data_var_name][dict(time=slice(
            n_burn_in + check_index, None))] += np.random.normal()
    X_check_array = ts_fe.transform(X_check_ds)
    X_neq = np.not_equal(
        X_train_array[:check_size], X_check_array[:check_size])
    x_neq = np.all(X_neq, axis=1)
    first_modified_index = np.min(x_neq.nonzero())
    if first_modified_index < check_index:
        message = 'The feature extractor looks into the future by' +\
            ' at least {} time steps'.format(
                check_index - first_modified_index)
        raise AssertionError(message)

    # Regression
    regressor = import_module('regressor', module_path)
    reg = regressor.Regressor()
    reg.fit(X_train_array, y_train_array[n_burn_in:])
    return ts_fe, reg


def test_submission(trained_model, X_ds, test_is):
    X_test_ds = X_ds.isel(time=test_is)

    # X_xray = X_xray.copy(deep=True)
    ts_fe, reg = trained_model
    # Feature extraction
    X_test_array = ts_fe.transform(X_test_ds)
    # Regression
    y_pred_array = reg.predict(X_test_array)
    return y_pred_array


X_train_ds, y_train_array = get_train_data()
X_test_ds, y_test_array = get_test_data()
train_scores = []
valid_scores = []
test_scores = []
for train_is, valid_is in get_cv(y_train_array):
    trained_model = train_submission('.', X_train_ds, y_train_array, train_is)
    y_train_pred_array = test_submission(trained_model, X_train_ds, train_is)
    train_score = score(
        y_train_array[train_is][n_burn_in:], y_train_pred_array)

    # we burn in _before_ the test fold so we have prediction for all
    # y_train_array[valid_is]
    burn_in_range = np.arange(valid_is[0] - n_burn_in, valid_is[0])
    extended_valid_is = np.concatenate((burn_in_range, valid_is))
    y_valid_pred_array = test_submission(
        trained_model, X_train_ds, extended_valid_is)
    valid_score = score(y_train_array[valid_is], y_valid_pred_array)
    y_test_pred_array = test_submission(
        trained_model, X_test_ds, range(len(y_test_array)))
    test_score = score(y_test_array[n_burn_in:], y_test_pred_array)
    print 'train RMSE =', train_score, '; valid RMSE =', valid_score,\
        '; test RMSE =', test_score
    train_scores.append(train_score)
    valid_scores.append(valid_score)
    test_scores.append(test_score)
print 'mean train RMSE =', np.mean(train_scores), '±', np.std(train_scores)
print 'mean valid RMSE =', np.mean(valid_scores), u'±', np.std(valid_scores)
print 'mean test RMSE =', np.mean(test_scores), u'±', np.std(test_scores)


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

