import os
import sys
import xray
import socket
import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.cross_validation import ShuffleSplit
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

hackaton_title = 'El Nino prediction'
target_column_name = 'target'
#held_out_test_size = 0.7
skf_test_size = 0.5
random_state = 57

en_lat_bottom = -5
en_lat_top = 5
en_lon_left = 360-170
en_lon_right = 360-120
n_burn_in = 120
n_lookahead = 6

#raw_filename = os.path.join(raw_data_path, 'data.csv')
train_filename = os.path.join(
    public_data_path, 'resampled_tas_Amon_CCSM4_piControl_r1i1p1_080001-130012.nc')
test_filename = os.path.join(
    private_data_path, 'resampled_tas_Amon_CCSM4_piControl_r2i1p1_095301-110812.nc')

n_CV = 2 if local_deployment else n_processes
score = scores.RMSE()

def get_enso_mean(tas):
    return tas.loc[:, en_lat_bottom:en_lat_top, en_lon_left:en_lon_right].mean(dim=('lat','lon'))

def read_data(xray_filename):
    temperatures_xray = xray.open_dataset(xray_filename, decode_times=False)
    # ridiculous as it sounds, there is simply no way to convert a date starting 
    # with the year 800 into pd array
    temperatures_xray['time'] = pd.date_range(
        '1/1/1700', periods=temperatures_xray['time'].shape[0], freq='M') - \
        np.timedelta64(15, 'D')
    #target = get_enso_mean(temperatures_xray['tas'])
    #target['time'] = np.roll(target['time'], n_lookahead)
    #target[:n_lookahead] = np.NaN
    #temperatures_xray['target'] = target
    return temperatures_xray

def prepare_data():
    pass
    # train and tes splits are given

def split_data():
    X_train_xray = read_data(train_filename)
    y_train_array = X_train_xray['target'].values[n_burn_in:-n_lookahead]
    X_test_xray = read_data(test_filename)
    y_test_array = X_test_xray['target'].values[n_burn_in:-n_lookahead]
    skf = ShuffleSplit(
        X_train_xray['time'].shape[0] - n_burn_in - n_lookahead, 
        n_iter=n_CV, test_size=skf_test_size, random_state=random_state)
    return X_train_xray, y_train_array, X_test_xray, y_test_array, skf

def train_model(module_path, X_xray, y_array, skf_is):
    # Feature extraction
    feature_extractor = import_module('.ts_feature_extractor', module_path)
    ts_fe = feature_extractor.FeatureExtractor()
    X_array = ts_fe.transform(X_xray, n_burn_in, n_lookahead, skf_is)
    # Regression
    train_is, _ = skf_is
    X_train_array = np.array([X_array[i] for i in train_is])
    y_train_array = np.array([y_array[i] for i in train_is])
    regressor = import_module('.regressor', module_path)
    reg = regressor.Regressor()
    reg.fit(X_train_array, y_train_array)
    return ts_fe, reg

def test_model(trained_model, X_xray, skf_is):
    ts_fe, reg = trained_model
    # Feature extraction
    X_array = ts_fe.transform(X_xray, n_burn_in, n_lookahead, skf_is)
    # Regression
    _, test_is = skf_is
    X_test_array = np.array([X_array[i] for i in test_is])
    y_pred_array = reg.predict(X_test_array)
    return prediction_type.PredictionArrayType(y_pred_array=y_pred_array)
