from importlib import import_module


def train_submission(module_path, X_df, y_array, train_is=None):
    if train_is is None:
        train_is = range(len(y_array))
    feature_extractor = import_module('.feature_extractor', module_path)
    fe = feature_extractor.FeatureExtractor()
    fe.fit(X_df.iloc[train_is], y_array[train_is])
    return fe


def test_submission(trained_model, X_df):
    fe = trained_model
    X_test_array = fe.transform(X_df)
    return X_test_array
