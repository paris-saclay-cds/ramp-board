import os
import sys
import numpy as np
import pandas as pd
from scipy import io
from sklearn.model_selection import ShuffleSplit
import databoard.clustering_prediction as prediction
from databoard.config import submissions_path, problems_path

sys.path.append(os.path.dirname(os.path.abspath(submissions_path)))

problem_name = 'HEP_tracking'  # should be the same as the file name
problem_title = 'Particle tracking in the LHC ATLAS detector'

random_state = 57
held_out_test_size = 0.3


train_filename = os.path.join(
    problems_path, problem_name, 'data', 'private', 'train.csv')
test_filename = os.path.join(
    problems_path, problem_name, 'data', 'private', 'test.csv')

workflow_name = 'clusterer_workflow'
prediction_labels = None


def prepare_data():
    pass


def read_data(filename):
    df = pd.read_csv(filename)
    y_df = df[['event_id', 'cluster_id']]
    X_df = df.drop(['cluster_id'], axis=1)
    return X_df.values, y_df.values


def get_train_data():
    X_train_array, y_train_array = read_data(train_filename)
    return X_train_array, y_train_array


def get_test_data():
    X_test_array, y_test_array = read_data(test_filename)
    return X_test_array, y_test_array
