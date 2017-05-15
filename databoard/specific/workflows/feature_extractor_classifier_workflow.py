from importlib import import_module
from databoard.specific.workflows import classifier_workflow


def train_submission(module_path, X_df, y_array, train_is=None):
    if train_is is None:
        train_is = range(len(y_array))
    # Preparing the training set
    X_train_df = X_df.iloc[train_is]
    y_train_array = y_array[train_is]

    # Feature extraction
    feature_extractor = import_module('.feature_extractor', module_path)
    fe = feature_extractor.FeatureExtractor()
    fe.fit(X_train_df, y_train_array)
    X_train_array = fe.transform(X_train_df)

    clf = classifier_workflow.train_submission(
        module_path, X_train_array, y_train_array)
    return fe, clf


def test_submission(trained_model, X_df):
    fe, clf = trained_model

    # Feature extraction
    X_test_array = fe.transform(X_df)

    # Classification
    y_proba = classifier_workflow.test_submission(clf, X_test_array)
    return y_proba
