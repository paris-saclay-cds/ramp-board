from importlib import import_module

def train_submission(module_path, X_df, y_array, train_is):
    # Preparing the training set
    X_train_df = X_df.iloc[train_is]
    y_train_array = y_array[train_is]

    # Feature extraction
    feature_extractor = import_module('.feature_extractor', module_path)
    fe = feature_extractor.FeatureExtractor()
    fe.fit(X_train_df, y_train_array)
    X_train_array = fe.transform(X_train_df)

    regressor = import_module('.regressor', module_path)
    reg = regressor.Regressor()
    reg.fit(X_train_array, y_train_array)
    return fe, reg


def test_submission(trained_model, X_df, test_is):

    # Preparing the test (or valid) set
    X_test_df = X_df.iloc[test_is]

    fe, reg = trained_model

    # Feature extraction
    X_test_array = fe.transform(X_test_df)

    # Regression
    y_pred = reg.predict(X_test_array)
    return y_pred
