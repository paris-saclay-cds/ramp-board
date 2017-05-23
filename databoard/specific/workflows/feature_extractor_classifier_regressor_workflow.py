import numpy as np
from importlib import import_module
# TODO: this should change when labels are in the db
labels = ['A', 'B', 'Q', 'R']


def train_submission(module_path, X_df, y_array, train_is=None):
    if train_is is None:
        train_is = range(len(y_array))
    # Preparing the training set
    X_train_df = X_df.iloc[train_is].copy()
    y_train_array = y_array[train_is].copy()
    y_train_clf_array = y_train_array[:, 0]
    y_train_reg_array = y_train_array[:, 1].astype(float)

    # Classifier feature extraction
    feature_extractor_clf = import_module(
        '.feature_extractor_clf', module_path)
    fe_clf = feature_extractor_clf.FeatureExtractorClf()
    fe_clf.fit(X_train_df, y_train_clf_array)
    X_train_array_clf = fe_clf.transform(X_train_df)

    # Classifier
    classifier = import_module('.classifier', module_path)
    clf = classifier.Classifier()
    clf.fit(X_train_array_clf, y_train_clf_array)

    # Concatenating y_proba (derived from labels) to X_train_df
    # TODO: this should change when labels are in the db
    for i, label in enumerate(labels):
        X_train_df.loc[:, label] = (y_train_clf_array == label)

    # Regressor feature extraction
    feature_extractor_reg = import_module(
        '.feature_extractor_reg', module_path)
    fe_reg = feature_extractor_reg.FeatureExtractorReg()
    fe_reg.fit(X_train_df, y_train_reg_array)
    X_train_array_reg = fe_reg.transform(X_train_df)

    # Regressor
    regressor = import_module('.regressor', module_path)
    reg = regressor.Regressor()
    reg.fit(X_train_array_reg, y_train_reg_array)
    return fe_clf, clf, fe_reg, reg


def test_submission(trained_model, X_df):
    fe_clf, clf, fe_reg, reg = trained_model

    # Preparing the test set
    X_test_df = X_df#.copy()

    # Classifier feature extraction
    X_test_array_clf = fe_clf.transform(X_test_df)

    # Classifier
    y_proba_clf = clf.predict_proba(X_test_array_clf)

    # Concatenating y_proba (derived from labels) to X_test_df
    # TODO: this should change when labels are in the db
    for i, label in enumerate(labels):
        X_test_df.loc[:, label] = y_proba_clf[:, i]

    # Regressor feature extraction
    X_test_array_reg = fe_reg.transform(X_test_df)

    # Regressor
    y_pred_reg = reg.predict(X_test_array_reg)

    return np.concatenate([y_proba_clf, y_pred_reg.reshape(-1, 1)], axis=1)
