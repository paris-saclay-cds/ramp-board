import os
import sys
import socket
import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.calibration import CalibratedClassifierCV
# menu
from output_type import MultiClassClassification as OutputType
import scores

from .config_databoard import (
    local_deployment,
    raw_data_path,
    public_data_path,
    private_data_path,
    n_processes,
    models_path,
    root_path
)

sys.path.append(os.path.dirname(os.path.abspath(models_path)))
server_port = '8081'
dest_path = '/mnt/datacamp/databoard_04_8081_test'

hackaton_title = 'Kaggle Otto product classification'
target_column_name = 'target'
labels = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6", 
          "Class_7", "Class_8", "Class_9"]
#held_out_test_size = 0.7
skf_test_size = 0.5
random_state = 57
#raw_filename = os.path.join(raw_data_path, 'data.csv')
train_filename = os.path.join(public_data_path, 'train.csv')
test_filename = os.path.join(private_data_path, 'test.csv')

n_CV = 2 if local_deployment else 1 * n_processes

#score = scores.Accuracy()
#score = scores.Error()
score = scores.NegativeLogLikelihood()
score.set_labels(labels)


# X is a list of dicts, each dict is indexed by column
def read_data(df_filename):
    df = pd.read_csv(df_filename, index_col=0)
    y_array = df[target_column_name].values
    X_dict = df.drop(target_column_name, axis=1).to_dict(orient='records')
    return X_dict, y_array

def prepare_data():
    pass
    # train and tes splits are given

def split_data():
    X_train_dict, y_train_array = read_data(train_filename)
    X_test_dict, y_test_array = read_data(test_filename)
    skf = StratifiedShuffleSplit(y_train_array, n_iter=n_CV, 
        test_size=skf_test_size, random_state=random_state)
    return X_train_dict, y_train_array, X_test_dict, y_test_array, skf

def train_model(module_path, X_valid_train_dict, y_valid_train):
     # Feature extraction
    feature_extractor = import_module('.feature_extractor', module_path)
    fe = feature_extractor.FeatureExtractor()
    fe.fit(X_valid_train_dict, y_valid_train)
    X_valid_train_array = fe.transform(X_valid_train_dict)

    # Classification
    classifier = import_module('.classifier', module_path)
    clf = classifier.Classifier()
    clf.fit(X_valid_train_array, y_valid_train)
    return fe, clf

def test_model(trained_model, X_test_dict):
    fe, clf = trained_model

    # Feature extraction
    X_test_array = fe.transform(X_test_dict)

    # Classification
    y_pred_array = clf.predict(X_test_array)
    y_probas_array = clf.predict_proba(X_test_array)
    return OutputType(y_pred_array=y_pred_array, y_probas_array=y_probas_array)