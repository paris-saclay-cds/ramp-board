import os
import sys
import pandas as pd
from sklearn.cross_validation import train_test_split
import databoard.multiclass_prediction as prediction
from databoard.config import submissions_path, problems_path,\
    starting_kit_d_name

sys.path.append(os.path.dirname(os.path.abspath(submissions_path)))

problem_name = 'HEP_detector_anomalies'  # should be the same as the file name

random_state = 57
public_train_size = 100000
train_size = 200000

raw_filename = os.path.join(
    problems_path, problem_name, 'data', 'raw', 'data.csv.gz')
public_train_filename = os.path.join(
    problems_path, problem_name, starting_kit_d_name, 'public_train.csv.gz')
train_filename = os.path.join(
    problems_path, problem_name, 'data', 'private', 'train.csv')
test_filename = os.path.join(
    problems_path, problem_name, 'data', 'private', 'test.csv')

prediction.labels = [0, 1]
target_column_name = 'isSkewed'
workflow_name = 'classifier_workflow'
extra_files = [os.path.join(problems_path, problem_name,
                            'HEP_detector_anomalies_datarun.py')]


def prepare_data():
    df = pd.read_csv(raw_filename, index_col=0)
    df_public_train, df_private = train_test_split(
        df, train_size=public_train_size, random_state=random_state)
    df_train, df_test = train_test_split(
        df_private, train_size=train_size,
        random_state=random_state)

    df_public_train.to_csv(
        public_train_filename, index=False, compression='gzip')
    df_train.to_csv(train_filename, index=False)
    df_test.to_csv(test_filename, index=False)


def read_data(static_filename):
    df = pd.read_csv(static_filename)
    y_array = df[target_column_name].values
    X_array = df.drop([target_column_name], axis=1).values
    return X_array, y_array


def get_train_data():
    X_train_array, y_train_array = read_data(train_filename)
    return X_train_array, y_train_array


def get_test_data():
    X_test_array, y_test_array = read_data(test_filename)
    return X_test_array, y_test_array
