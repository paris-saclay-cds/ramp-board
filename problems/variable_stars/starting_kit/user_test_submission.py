import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

train_filename = 'public_train.csv'
vf_train_filename = 'public_train_varlength_features.csv.gz'
target_column_name = 'type'

labels = [1.0, 2.0, 3.0, 4.0]

def csv_array_to_float(csv_array_string):
    return map(float, csv_array_string[1:-1].split(','))


def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    z = x.copy()
    z.update(y)
    return z


# X is a column-indexed dict, y is a numpy array
def read_data(df_filename, vf_filename):
    df = pd.read_csv(df_filename, index_col=0)
    y_array = df[target_column_name].values
    X_dict = df.drop(target_column_name, axis=1).to_dict(orient='records')
    vf_raw = pd.read_csv(vf_filename, index_col=0, compression='gzip')
    vf_dict = vf_raw.applymap(csv_array_to_float).to_dict(orient='records')
    X_dict = [merge_two_dicts(d_inst, v_inst) for d_inst, v_inst
              in zip(X_dict, vf_dict)]
    return X_dict, y_array


def train_submission(module_path, X_dict, y_array, train_is):
    # Preparing the training set
    X_train_dict = [X_dict[i] for i in train_is]
    y_train_array = np.array([y_array[i] for i in train_is])

    # Feature extraction
    feature_extractor = import_module('feature_extractor', module_path)
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
    classifier = import_module('classifier', module_path)
    clf = classifier.Classifier()
    clf.fit(X_train_train_array, y_train_train_array)

    # Calibration
    calibrator = import_module('calibrator', module_path)
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
    y_probas_array = clf.predict_proba(X_test_array)

    # Calibration
    y_calib_probas_array = calib.predict_proba(y_probas_array)
    return y_calib_probas_array


if __name__ == '__main__':
    print("Reading file ...")
    X_dict, y_array = read_data(train_filename, vf_train_filename)
    skf = StratifiedShuffleSplit(
        y_array, n_iter=2, test_size=0.5, random_state=67)
    print("Training file ...")
    for valid_train_is, valid_test_is in skf:
        X_valid_train_dict = [X_dict[i] for i in valid_train_is]
        y_valid_train = y_array[valid_train_is]
        X_valid_test_dict = [X_dict[i] for i in valid_test_is]
        y_valid_test = y_array[valid_test_is]

        trained_model = train_submission('.', X_dict, y_array, valid_train_is)
        y_calib_probas_array = test_submission(
            trained_model, X_dict, valid_test_is)
        y_pred = [labels[np.argmax(probas)] for probas in y_calib_probas_array]
        print 'accuracy = ', accuracy_score(y_pred, y_valid_test)
