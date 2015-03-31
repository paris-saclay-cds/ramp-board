import os
import socket
import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from .scores import ScoreAuc as Score

from .config_databoard import (
    local_deployment,
    raw_data_path,
    public_data_path,
    private_data_path,
    n_processes,
)

hackaton_title = 'Variable star type prediction'
target_column_name = 'type'
held_out_test_size = 0.2
skf_test_size = 0.5
random_state = 57
raw_filename = os.path.join(raw_data_path, 'data.csv')
train_filename = os.path.join(public_data_path, 'train.csv')
test_filename = os.path.join(private_data_path, 'test.csv')

vf_raw_filename = os.path.join(raw_data_path, 'varlength_features.csv.gz')
vf_train_filename = os.path.join(public_data_path, 'train_varlength_features.csv')
vf_test_filename = os.path.join(private_data_path, 'test_varlength_features.csv')

n_CV = 2 if local_deployment else 1 * n_processes

def csv_array_to_float(csv_array_string):
    return map(float, csv_array_string[1:-1].split())

# I'm not prepared for this kind of shit late at night. Pandas can handle 
# variable length vectors as db elements. In the notebook, when I do to_csv, 
# it turns them into strings "[elem1 elem2 ... elemn]". When I call python it 
# turns them into strings "[elem1, elem2, ... ,elemn]" (notice the commas).
def csv_array_to_float_comma(csv_array_string):
    return map(float, csv_array_string[1:-1].split(','))

# X is a column-indexed dict, y is a numpy array
def read_data(df_filename, vf_filename):
    df = pd.read_csv(df_filename, index_col=0)
    y_array = df[target_column_name].values
    X_dict = df.drop(target_column_name, axis=1).to_dict(orient='list')
    vf_raw = pd.read_csv(vf_filename, index_col=0)
    vf_dict = vf_raw.applymap(csv_array_to_float_comma).to_dict(orient='list')
    X_dict = dict(X_dict.items() + vf_dict.items())
    return X_dict, y_array

def prepare_data():
    try:
        df = pd.read_csv(raw_filename, index_col=0)
    except IOError, e:
        print e
        print raw_filename + " should be placed in " + raw_data_path + " before running make setup"
        raise

    try:
        vf_raw = pd.read_csv(vf_raw_filename, index_col=0, compression = 'gzip')
        vf = vf_raw.applymap(csv_array_to_float)
    except IOError, e:
        print e
        print vf_raw_filename + " should be placed in " + raw_data_path + " before running make setup"
        raise

    df_train, df_test, vf_train, vf_test = train_test_split(df, vf,
        test_size=held_out_test_size, random_state=random_state)


    if not os.path.exists(public_data_path):
        os.mkdir(public_data_path)
    if not os.path.exists(private_data_path):
        os.mkdir(private_data_path)

    df_train = pd.DataFrame(df_train, columns=df.columns)
    df_test = pd.DataFrame(df_test, columns=df.columns)
    df_train.to_csv(train_filename, index=True) 
    df_test.to_csv(test_filename, index=True)

    vf_train = pd.DataFrame(vf_train, columns=vf.columns)
    vf_test = pd.DataFrame(vf_test, columns=vf.columns)
    vf_train.to_csv(vf_train_filename, index=True) 
    vf_test.to_csv(vf_test_filename, index=True)

def split_data():
    X_train, y_train = read_data(train_filename, vf_train_filename)
    X_test, y_test = read_data(test_filename, vf_test_filename)
    skf = StratifiedShuffleSplit(y_train, n_iter=n_CV, 
        test_size=skf_test_size, random_state=random_state)
    return X_train, y_train, X_test, y_test, skf

def run_model(model, X_valid_train, y_valid_train, X_valid_test, X_test):
    clf = model.Classifier()
    clf_c = CalibratedClassifierCV(clf, cv=2, method='isotonic')
    clf_c.fit(X_valid_train, y_valid_train)
    y_valid_pred = clf_c.predict(X_valid_test)
    y_valid_proba = clf_c.predict_proba(X_valid_test)
    y_test_pred = clf_c.predict(X_test)
    y_test_proba = clf_c.predict_proba(X_test)
    return y_valid_pred, y_valid_proba, y_test_pred, y_test_proba

