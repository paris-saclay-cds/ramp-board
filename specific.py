import socket
import os
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from config_databoard import (
    local_deployment,
    raw_data_path,
    public_data_path,
    private_data_path,
    cachedir,
)

from sklearn.externals.joblib import Memory

hackaton_title = 'Mortality prediction'
target_column_name = 'TARGET'
held_out_test_size = 0.2
skf_test_size = 0.5
random_state = 57
raw_filename = os.path.join(raw_data_path, 'data.csv')
train_filename = os.path.join(public_data_path, 'train.csv')
test_filename = os.path.join(private_data_path, 'test.csv')
n_CV = 2 if local_deployment else 5 * n_processes


mem = Memory(cachedir=cachedir)


def read_data(filename):
    df = pd.read_csv(filename)
    y = df[target_column_name].values
    X = df.drop(target_column_name, axis=1).values
    return X, y

def prepare_data():
    try:
        df = pd.read_csv(raw_filename)
    except IOError, e:
        print e
        print raw_filename + " should be placed in " + raw_data_path + " before running make setup"
        raise

    df_train, df_test = train_test_split(df, 
        test_size=held_out_test_size, random_state=random_state)

    df_train = pd.DataFrame(df_train, columns=df.columns)
    df_test = pd.DataFrame(df_test, columns=df.columns)

    if not os.path.exists(public_data_path):
        os.mkdir(public_data_path)
    df_train.to_csv(train_filename, index=False) 
    if not os.path.exists(private_data_path):
        os.mkdir(private_data_path)
    df_test.to_csv(test_filename, index=False) 

def split_data():
    X_train, y_train = read_data(train_filename)
    X_test, y_test = read_data(test_filename)
    skf = StratifiedShuffleSplit(y_train, n_iter=n_CV, 
        test_size=skf_test_size, random_state=random_state)
    return X_train, y_train, X_test, y_test, skf

@mem.cache
def run_model(model, X_valid_train, y_valid_train, X_valid_test, X_test):
    clf = model.Classifier()
    clf.fit(X_valid_train, y_valid_train)
    y_valid_pred = clf.predict(X_valid_test)
    y_valid_score = clf.predict_proba(X_valid_test)
    y_test_pred = clf.predict(X_test)
    y_test_score = clf.predict_proba(X_test)
    return y_valid_pred, y_valid_score, y_test_pred, y_test_score


