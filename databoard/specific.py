import os
import sys
import socket
import pandas as pd
from importlib import import_module
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.calibration import CalibratedClassifierCV
# menu
from .scores import ScoreAccuracy as Score
from .output_type import save_multi_class_predictions as save_model_predictions
from .output_type import load_multi_class_predictions as load_model_predictions

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

hackaton_title = 'Variable star type prediction'
target_column_name = 'type'
labels = [1, 2, 3, 4]
held_out_test_size = 0.7
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
    return map(float, csv_array_string[1:-1].split(','))

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

# X is a list of dicts, each dict is indexed by column
def read_data(static_filename, variable_filename):
    static_df = pd.read_csv(static_filename, index_col=0)
    y_array = static_df[target_column_name].values
    X_static_dict = static_df.drop(target_column_name, axis=1).to_dict(orient='records')
    variable_df = pd.read_csv(variable_filename, index_col=0)
    X_variable_dict = variable_df.applymap(csv_array_to_float).to_dict(orient='records')
    X_dict = [merge_two_dicts(d_inst, v_inst)
              for d_inst, v_inst in zip(X_static_dict, X_variable_dict)]
    return X_dict, y_array

def prepare_data():
    try:
        df = pd.read_csv(raw_filename, index_col=0)
        # we drop the "unkown" class for this ramp
        index_list = df[df['type'] < 5].index
        df = df.loc[index_list]
    except IOError, e:
        print e
        print raw_filename + " should be placed in " + raw_data_path + " before running make setup"
        raise

    try:
        vf_raw = pd.read_csv(vf_raw_filename, index_col=0, compression = 'gzip')
        vf_raw = vf_raw.loc[index_list]
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
    X_train_dict, y_train_array = read_data(train_filename, vf_train_filename)
    X_test_dict, y_test_array = read_data(test_filename, vf_test_filename)
    skf = StratifiedShuffleSplit(y_train_array, n_iter=n_CV, 
        test_size=skf_test_size, random_state=random_state)
    return X_train_dict, y_train_array, X_test_dict, y_test_array, skf

def run_model(module_path, X_valid_train_dict, y_valid_train, 
              X_valid_test_dict, X_test_dict):
     # Feature extraction
    feature_extractor = import_module('.feature_extractor', module_path)
    fe = feature_extractor.FeatureExtractor()
    fe.fit(X_valid_train_dict, y_valid_train)
    X_valid_train_array = fe.transform(X_valid_train_dict)
    X_valid_test_array = fe.transform(X_valid_test_dict)
    X_test_array = fe.transform(X_test_dict)

    # Classification
    classifier = import_module('.classifier', module_path)
    clf = classifier.Classifier()
    clf_c = CalibratedClassifierCV(clf, cv=2, method='isotonic')
    clf_c.fit(X_valid_train_array, y_valid_train)
    y_valid_pred = clf_c.predict(X_valid_test_array)
    y_valid_score = clf_c.predict_proba(X_valid_test_array)
    y_test_pred = clf_c.predict(X_test_array)
    y_test_score = clf_c.predict_proba(X_test_array)
    return y_valid_pred, y_valid_score, y_test_pred, y_test_score
