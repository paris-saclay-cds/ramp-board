import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import databoard.multiclass_prediction as prediction
from databoard.config import submissions_path, problems_path
from distutils.dir_util import mkpath

sys.path.append(os.path.dirname(os.path.abspath(submissions_path)))

problem_name = 'titanic'  # should be the same as the file name

random_state = 57
held_out_test_size = 0.2


raw_filename = os.path.join(
    problems_path, problem_name, 'data', 'raw', 'all.csv')
train_filename = os.path.join(
    problems_path, problem_name, 'data', 'private', 'train.csv')
test_filename = os.path.join(
    problems_path, problem_name, 'data', 'private', 'test.csv')

# prediction.labels = ['setosa', 'versicolor', 'virginica']
prediction_labels = [0, 1]
workflow_name = 'feature_extractor_classifier_workflow'


def prepare_data():
    pass


def read_data(filename):
    data = pd.read_csv(filename)
    y_array = data['Survived'].values
    X_df = data.drop(['Survived', 'PassengerId'], axis=1)
    return X_df, y_array


def get_train_data():
    X_train_df, y_train_array = read_data(train_filename)
    return X_train_df, y_train_array


def get_test_data():
    X_test_df, y_test_array = read_data(test_filename)
    return X_test_df, y_test_array
