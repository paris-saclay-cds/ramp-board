import numpy as np
from importlib import import_module


def train_submission(module_path, X_ds, y_array, train_is):
    # train_is wrt range(len(y_array)) so we need to make X_test_ds longer
    # by n_burn_in
    n_burn_in = X_ds.attrs['n_burn_in']
    burn_in_range = np.arange(train_is[-1], train_is[-1] + n_burn_in)
    extended_train_is = np.concatenate((train_is, burn_in_range))
    X_train_ds = X_ds.isel(time=extended_train_is)
    y_train_array = y_array[train_is]

    # Feature extraction
    ts_feature_extractor = import_module('.ts_feature_extractor', module_path)
    ts_fe = ts_feature_extractor.FeatureExtractor()
    # ts_fe.transform should return array corresponding to time points
    # without burn in, so X_train_array and y_train_array should new have
    # the same length.
    X_train_array = ts_fe.transform(X_train_ds)

    # Checking if feature extractor looks ahead: we change the input array
    # after index n_burn_in + check_index, and check if the first
    # check_features have changed
    check_size = 132
    check_index = 13
    # We use a short prefix to save time
    X_check_ds = X_ds.isel(
        time=slice(0, n_burn_in + check_size)).copy(deep=True)
    # Assigning Dataset slices is not yet supported so we need to iterate
    # over the arrays. To generalize we should maybe check the types.
    data_var_names = X_check_ds.data_vars.keys()
    for data_var_name in data_var_names:
        X_check_ds[data_var_name][dict(time=slice(
            n_burn_in + check_index, None))] += np.random.normal()
    X_check_array = ts_fe.transform(X_check_ds)
    X_neq = np.not_equal(
        X_train_array[:check_size], X_check_array[:check_size])
    x_neq = np.any(X_neq, axis=1)
    x_neq_nonzero = x_neq.nonzero()
    if len(x_neq_nonzero[0]) == 0:  # no change anywhere
        first_modified_index = check_index
    else:
        first_modified_index = np.min(x_neq_nonzero)
    # Normally, the features should not have changed before check_index
    if first_modified_index < check_index:
        message = 'The feature extractor looks into the future by' +\
            ' at least {} time steps'.format(
                check_index - first_modified_index)
        raise AssertionError(message)

    # Regression
    regressor = import_module('.regressor', module_path)
    reg = regressor.Regressor()
    reg.fit(X_train_array, y_train_array)
    return ts_fe, reg


def test_submission(trained_model, X_ds, test_is):
    # test_is is range(len(y)) so we need to make X_test_ds longer by n_burn_in
    n_burn_in = X_ds.attrs['n_burn_in']
    burn_in_range = np.arange(test_is[-1], test_is[-1] + n_burn_in)
    extended_test_is = np.concatenate((test_is, burn_in_range))
    X_test_ds = X_ds.isel(time=extended_test_is)
    # X_xray = X_xray.copy(deep=True)
    ts_fe, reg = trained_model
    # Feature extraction
    X_test_array = ts_fe.transform(X_test_ds)
    # Regression
    y_pred_array = reg.predict(X_test_array)
    return y_pred_array
