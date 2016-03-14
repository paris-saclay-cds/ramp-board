from importlib import import_module


def train_submission(module_path, X_array, y_array, train_is):
    regressor = import_module('.regressor', module_path)
    reg = regressor.Regressor()
    reg.fit(X_array[train_is], y_array[train_is])
    return reg


def test_submission(trained_model, X_array, test_is):
    reg = trained_model
    y_pred = reg.predict(X_array[test_is])
    return y_pred
