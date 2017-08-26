import os
import sys
import pandas as pd
import rampwf as rw
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import databoard.multiclass_prediction as prediction  # noqa
from databoard.config import submissions_path, problems_path
from distutils.dir_util import mkpath

sys.path.append(os.path.dirname(os.path.abspath(submissions_path)))

problem_name = 'iris'  # should be the same as the file name
problem_title = 'Iris classification'


raw_filename = os.path.join(
    problems_path, problem_name, 'data', 'raw', 'iris.csv')
train_filename = os.path.join(
    problems_path, problem_name, 'data', 'public', 'train.csv')
test_filename = os.path.join(
    problems_path, problem_name, 'data', 'private', 'test.csv')

prediction_type = rw.prediction_types.multiclass
prediction_labels = ['setosa', 'versicolor', 'virginica']
target_column_name = 'species'
workflow = rw.workflows.Classifier()
score_types = [
    rw.score_types.Accuracy(name='acc', n_columns=len(prediction_labels)),
    rw.score_types.ClassificationError(
        name='err', n_columns=len(prediction_labels)),
    rw.score_types.NegativeLogLikelihood(
        name='nll', n_columns=len(prediction_labels)),
    rw.score_types.F1Above(
        name='f1_70', n_columns=len(prediction_labels), threshold=0.7),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=57)
    return cv.split(X, y)


def prepare_data():
    df = pd.read_csv(raw_filename)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=57)
    mkpath(os.path.dirname(train_filename))
    df_train.to_csv(train_filename, index=False)
    mkpath(os.path.dirname(test_filename))
    df_test.to_csv(test_filename, index=False)


def read_data(filename):
    data = pd.read_csv(filename)
    y_array = data[target_column_name].values
    X_array = data.drop([target_column_name], axis=1).values
    return X_array, y_array


def get_train_data():
    X_train_array, y_train_array = read_data(train_filename)
    return X_train_array, y_train_array


def get_test_data():
    X_test_array, y_test_array = read_data(test_filename)
    return X_test_array, y_test_array
