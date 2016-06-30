import os
from importlib import import_module
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
# import databoard.multiclass_prediction as prediction
# from databoard.config import submissions_path, problems_path,\
#     starting_kit_d_name

# sys.path.append(os.path.dirname(os.path.abspath(submissions_path)))

problem_name = 'variable_stars'  # should be the same as the file name

random_state = 57
held_out_test_size = 0.7

raw_filename = 'data.csv'
vf_raw_filename = 'data_varlength_features.csv.gz'
public_train_filename = 'public_train.csv'
vf_public_train_filename = 'public_train_varlength_features.csv'
train_filename = 'train.csv'
vf_train_filename = 'train_varlength_features.csv'
test_filename = 'test.csv'
vf_test_filename = 'test_varlength_features.csv'

target_column_name = 'type'


def csv_array_to_float(csv_array_string):
    return map(float, csv_array_string[1:-1].split(','))


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


# X is a list of dicts, each dict is indexed by column
def read_data(static_filename, variable_filename):
    static_df = pd.read_csv(static_filename, index_col=0)
    y_array = static_df[target_column_name].values
    X_static_dict = static_df.drop(
        target_column_name, axis=1).to_dict(orient='records')
    variable_df = pd.read_csv(variable_filename, index_col=0)
    X_variable_dict = variable_df.applymap(
        csv_array_to_float).to_dict(orient='records')
    X_dict = [pd.Series(merge_two_dicts(d_inst, v_inst))
              for d_inst, v_inst in zip(X_static_dict, X_variable_dict)]
    return X_dict, y_array


def prepare_data(raw_data_path):
    df = pd.read_csv(os.path.join(raw_data_path, raw_filename), index_col=0)
    # we drop the "unkown" class for this ramp
    index_list = df[df[target_column_name] < 5].index
    df = df.loc[index_list]

    vf_raw = pd.read_csv(os.path.join(raw_data_path, vf_raw_filename),
                         index_col=0, compression='gzip')
    vf_raw = vf_raw.loc[index_list]
    vf = vf_raw.applymap(csv_array_to_float)
    df_public_train, df_test, vf_public_train, vf_test = train_test_split(
        df, vf, test_size=held_out_test_size, random_state=random_state)

    df_public_train = pd.DataFrame(df_public_train, columns=df.columns)
    df_public_train.to_csv(os.path.join(raw_data_path, public_train_filename),
                           index=True)
    vf_public_train = pd.DataFrame(vf_public_train, columns=vf.columns)
    vf_public_train.to_csv(os.path.join(raw_data_path,
                                        vf_public_train_filename), index=True)

    df_train, df_test, vf_train, vf_test = train_test_split(
        df_test, vf_test, test_size=held_out_test_size,
        random_state=random_state)

    df_train = pd.DataFrame(df_train, columns=df.columns)
    df_train.to_csv(os.path.join(raw_data_path, train_filename), index=True)
    vf_train = pd.DataFrame(vf_train, columns=vf.columns)
    vf_train.to_csv(os.path.join(raw_data_path, vf_train_filename), index=True)

    df_test = pd.DataFrame(df_test, columns=df.columns)
    df_test.to_csv(os.path.join(raw_data_path, test_filename), index=True)
    vf_test = pd.DataFrame(vf_test, columns=vf.columns)
    vf_test.to_csv(os.path.join(raw_data_path, vf_test_filename), index=True)


def get_train_data(raw_data_path):
    X_train_dict, y_train_array = read_data(os.path.join(raw_data_path,
                                                         train_filename),
                                            os.path.join(raw_data_path,
                                                         vf_train_filename))
    return X_train_dict, y_train_array


def get_test_data(raw_data_path):
    X_test_dict, y_test_array = read_data(os.path.join(raw_data_path,
                                                       test_filename),
                                          os.path.join(raw_data_path,
                                                       vf_test_filename))
    return X_test_dict, y_test_array


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
