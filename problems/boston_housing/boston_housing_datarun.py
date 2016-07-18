import os
from importlib import import_module
import pandas as pd
from sklearn.cross_validation import train_test_split

problem_name = 'boston_housing'  # should be the same as the file name

random_state = 57
n_CV = 2
held_out_test_size = 0.2

raw_filename = 'boston_housing.csv'
train_filename = 'train.csv'
test_filename = 'test.csv'

target_column_name = 'medv'
workflow_name = 'regressor_workflow'


def prepare_data(raw_data_path):
    df = pd.read_csv(os.path.join(raw_data_path, raw_filename), index_col=0)
    df_train, df_test = train_test_split(
        df, test_size=held_out_test_size, random_state=random_state)
    df_train.to_csv(os.path.join(raw_data_path, train_filename), index=False)
    df_test.to_csv(os.path.join(raw_data_path, test_filename), index=False)


def read_data(filename, index_col=None):
    data = pd.read_csv(filename, index_col=index_col)
    y_array = data[target_column_name].values
    X_array = data.drop([target_column_name], axis=1).values
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
    regressor = import_module('.regressor', module_path)
    reg = regressor.Regressor()
    reg.fit(X_array[train_is], y_array[train_is])
    return reg


def test_submission(trained_model, X_array, test_is):
    reg = trained_model
    y_pred = reg.predict(X_array[test_is])
    return y_pred
