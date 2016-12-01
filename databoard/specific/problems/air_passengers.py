import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import databoard.regression_prediction as prediction  # noqa
from databoard.config import submissions_path, problems_path,\
    starting_kit_d_name

sys.path.append(os.path.dirname(os.path.abspath(submissions_path)))

# should be the same as the file name
problem_name = 'air_passengers'

random_state = 57
held_out_test_size = 0.2
public_train_size = 0.2

raw_filename = os.path.join(
    problems_path, problem_name, 'data', 'raw', 'data.csv')
public_train_filename = os.path.join(
    problems_path, problem_name, starting_kit_d_name, 'public_train.csv')
train_filename = os.path.join(
    problems_path, problem_name, 'data', 'private', 'train.csv')
test_filename = os.path.join(
    problems_path, problem_name, 'data', 'private', 'test.csv')

target_column_name = 'log_PAX'
workflow_name = 'feature_extractor_regressor_with_external_data_workflow'
prediction_labels = None
extra_files = [os.path.join(problems_path, problem_name,
                            'air_passengers_datarun.py')]


def read_data(df_filename):
    data = pd.read_csv(df_filename)
    return data


def prepare_data():
    df = read_data(raw_filename)
    df_train, df_test = train_test_split(
        df, test_size=held_out_test_size, random_state=random_state)
    df_public_train, df_private_train = train_test_split(
        df_train, train_size=public_train_size, random_state=random_state)
    df_public_train.to_csv(public_train_filename, index=False)
    df_private_train.to_csv(train_filename, index=False)
    df_test.to_csv(test_filename, index=False)


def get_train_data():
    df_train = read_data(train_filename)
    X_train_df = df_train.drop(target_column_name, axis=1)
    y_train_array = df_train[target_column_name].values
    return X_train_df, y_train_array


def get_test_data():
    df_test = read_data(test_filename)
    X_test_df = df_test.drop(target_column_name, axis=1)
    y_test_array = df_test[target_column_name].values
    return X_test_df, y_test_array
