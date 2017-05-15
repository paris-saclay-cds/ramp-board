from databoard.specific.workflows import feature_extractor_workflow
from databoard.specific.workflows import classifier_workflow


def train_submission(module_path, X_df, y_array, train_is=None):
    if train_is is None:
        train_is = range(len(y_array))
    fe = feature_extractor_workflow.train_submission(
        module_path, X_df, y_array, train_is)
    X_train_array = feature_extractor_workflow.test_submission(
        fe, X_df.iloc[train_is])
    clf = classifier_workflow.train_submission(
        module_path, X_train_array, y_array[train_is])
    return fe, clf


def test_submission(trained_model, X_df):
    fe, clf = trained_model
    X_test_array = fe.transform(X_df)
    y_proba = classifier_workflow.test_submission(clf, X_test_array)
    return y_proba
