import numpy as np
from importlib import import_module
from sklearn.cross_validation import StratifiedShuffleSplit


def train_submission(module_path, X_dict, y_array, train_is):
    # Preparing the training set
    X_train_dict = [X_dict[i] for i in train_is]
    y_train_array = np.array([y_array[i] for i in train_is])

    # Feature extraction
    feature_extractor = import_module('.feature_extractor', module_path)
    fe = feature_extractor.FeatureExtractor()
    fe.fit(X_train_dict, y_train_array)
    X_train_array = fe.transform(X_train_dict)

    # Train/valid cut for holding out calibration set
    cv = StratifiedShuffleSplit(
        y_train_array, n_iter=1, test_size=0.1, random_state=57)
    calib_train_is, calib_test_is = list(cv)[0]

    X_train_train_array = X_train_array[calib_train_is]
    y_train_train_array = y_train_array[calib_train_is]
    X_calib_train_array = X_train_array[calib_test_is]
    y_calib_train_array = y_train_array[calib_test_is]

    # Classification
    classifier = import_module('.classifier', module_path)
    clf = classifier.Classifier()
    clf.fit(X_train_train_array, y_train_train_array)

    # Calibration
    calibrator = import_module('.calibrator', module_path)
    calib = calibrator.Calibrator()
    y_probas_array = clf.predict_proba(X_calib_train_array)
    calib.fit(y_probas_array, y_calib_train_array)
    return fe, clf, calib


def test_submission(trained_model, X_dict, test_is):
    # Preparing the test (or valid) set
    X_test_dict = [X_dict[i] for i in test_is]

    fe, clf, calib = trained_model

    # Feature extraction
    X_test_array = fe.transform(X_test_dict)

    # Classification
    y_probas = clf.predict_proba(X_test_array)

    # Calibration
    y_calib_probas = calib.predict_proba(y_probas)
    return y_calib_probas
