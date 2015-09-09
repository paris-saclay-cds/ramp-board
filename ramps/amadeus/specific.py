import os
import sys
import xray
import socket
import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.cross_validation import ShuffleSplit, train_test_split
# menu
import regression_prediction_type as prediction_type # menu polymorphism example
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

# should be the same as the directory in ramp, and fab publish should also be 
# called with the same name as parameter
ramp_name = 'amadeus'
# will be displayed on the web site
hackaton_title = 'Amadeus number of passengers prediction'
target_column_name = 'log_PAX'
drop_column_names = ['PAX']
skf_test_size = 0.5
held_out_test_size = 0.5
random_state = 57
raw_filename = os.path.join(raw_data_path, 'data_amadeus.csv')
train_filename = os.path.join(public_data_path, 'train.csv')
test_filename = os.path.join(private_data_path, 'test.csv')

from multiprocessing import cpu_count
n_CV = 2 if local_deployment else cpu_count() #n_processes

score = scores.RMSE()

def read_data(df_filename):
    data = pd.read_csv(df_filename)
    return data

def prepare_data():
    try:
        df = read_data(raw_filename)
        df = df.drop(drop_column_names, axis=1)
    except IOError, e:
        print e
        print raw_filename + " should be placed in " + raw_data_path + " before running make setup"
        raise

    df_train, df_test = train_test_split(
        df, test_size=held_out_test_size, random_state=random_state)

    if not os.path.exists(public_data_path):
        os.mkdir(public_data_path)
    if not os.path.exists(private_data_path):
        os.mkdir(private_data_path)

    df_train.to_csv(train_filename) 
    df_test.to_csv(test_filename)

def split_data():
    df_train = read_data(train_filename)
    X_train_df = df_train.drop(target_column_name, axis=1)
    y_train_array = df_train[target_column_name].values
    df_test = read_data(test_filename)
    X_test_df = df_test.drop(target_column_name, axis=1)
    y_test_array = df_test[target_column_name].values
    skf = ShuffleSplit(
        y_train_array.shape[0], n_iter=n_CV, test_size=skf_test_size, 
        random_state=random_state)
    return X_train_df, y_train_array, X_test_df, y_test_array, skf

def train_model(module_path, X_df, y_array, skf_is):
    # Feature extraction
    feature_extractor = import_module('.feature_extractor', module_path)
    fe = feature_extractor.FeatureExtractor()
    fe.fit(X_df, y_array)
    X_array = fe.transform(X_df)
    # Regression
    train_is, _ = skf_is
    X_train_array = np.array([X_array[i] for i in train_is])
    y_train_array = np.array([y_array[i] for i in train_is])
    regressor = import_module('.regressor', module_path)
    reg = regressor.Regressor()
    reg.fit(X_train_array, y_train_array)
    return fe, reg

def test_model(trained_model, X_df, skf_is):
    fe, reg = trained_model
    # Feature extraction
    X_array = fe.transform(X_df)
    # Regression
    _, test_is = skf_is
    X_test_array = np.array([X_array[i] for i in test_is])
    y_pred_array = reg.predict(X_test_array)
    return prediction_type.PredictionArrayType(y_pred_array=y_pred_array)
