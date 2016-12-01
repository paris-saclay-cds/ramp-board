import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from databoard.config import submissions_path, problems_path
import databoard.regression_prediction as prediction  # noqa
from distutils.dir_util import mkpath

sys.path.append(os.path.dirname(os.path.abspath(submissions_path)))

problem_name = 'boston_housing'  # should be the same as the file name

random_state = 57
n_CV = 2
held_out_test_size = 0.2

raw_filename = os.path.join(
    problems_path, problem_name, 'data', 'raw', 'boston_housing.csv')
train_filename = os.path.join(
    problems_path, problem_name, 'data', 'public', 'train.csv')
test_filename = os.path.join(
    problems_path, problem_name, 'data', 'private', 'test.csv')

target_column_name = 'medv'
workflow_name = 'regressor_workflow'
prediction_labels = None
extra_files = [os.path.join(problems_path, problem_name,
                            'boston_housing_datarun.py')]


def prepare_data():
    df = pd.read_csv(raw_filename, index_col=0)
    df_train, df_test = train_test_split(
        df, test_size=held_out_test_size, random_state=random_state)
    mkpath(os.path.dirname(train_filename))
    df_train.to_csv(train_filename, index=False)
    mkpath(os.path.dirname(test_filename))
    df_test.to_csv(test_filename, index=False)


def read_data(filename, index_col=None):
    data = pd.read_csv(filename, index_col=index_col)
    y_array = data[target_column_name].values
    X_array = data.drop([target_column_name], axis=1).values
    return X_array, y_array


def get_train_data():
    X_train_array, y_train_array = read_data(train_filename)
    return X_train_array, y_train_array


def get_test_data():
    X_test_array, y_test_array = read_data(test_filename)
    return X_test_array, y_test_array
