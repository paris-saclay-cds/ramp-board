import os
from importlib import import_module
import pandas as pd
from sklearn.model_selection import train_test_split


problem_name = 'HEP_detector_anomalies'  # should be the same as the file name

random_state = 57
public_train_size = 100000
train_size = 200000

raw_filename = 'data.csv.gz'
public_train_filename = 'public_train.csv.gz'
train_filename = 'train.csv'
test_filename = 'test.csv'

target_column_name = 'isSkewed'
workflow_name = 'classifier_workflow'


def prepare_data(raw_data_path):
    df = pd.read_csv(os.path.join(raw_data_path, raw_filename), index_col=0)
    df_public_train, df_private = train_test_split(
        df, train_size=public_train_size, random_state=random_state)
    df_train, df_test = train_test_split(
        df_private, train_size=train_size,
        random_state=random_state)

    df_public_train.to_csv(
        os.path.join(raw_data_path, public_train_filename),
        index=False, compression='gzip')
    df_train.to_csv(os.path.join(raw_data_path, train_filename), index=False)
    df_test.to_csv(os.path.join(raw_data_path, test_filename), index=False)


def read_data(static_filename):
    df = pd.read_csv(static_filename)
    y_array = df[target_column_name].values
    X_array = df.drop([target_column_name], axis=1).values
    return X_array, y_array


def get_train_data(raw_data_path):
    X_train_array, y_train_array = read_data(os.path.join(raw_data_path,
                                                          train_filename))
    return X_train_array, y_train_array


def get_test_data(raw_data_path):
    X_test_array, y_test_array = read_data(os.path.join(raw_data_path,
                                                        test_filename))
    return X_test_array, y_test_array


def train_submission(module_path, X_array, y_array, train_is):
    classifier = import_module('.classifier', module_path)
    clf = classifier.Classifier()
    clf.fit(X_array[train_is], y_array[train_is])
    return clf


def test_submission(trained_model, X_array, test_is):
    clf = trained_model
    y_proba = clf.predict_proba(X_array[test_is])
    return y_proba
