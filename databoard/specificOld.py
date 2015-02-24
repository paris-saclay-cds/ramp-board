import socket
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
from .config_databoard import local_deployment

def read_data(filename='input/train.csv'):
    test_size = 0.5
    random_state = 57
    n_CV = 2 if local_deployment else 5 * n_processes
    df = pd.read_csv(filename)
    y = df['TARGET'].values
    X = df.drop('TARGET', axis=1).values
    skf = StratifiedShuffleSplit(y, n_iter=n_CV, test_size=test_size, random_state=random_state)
    return X, y, skf

def run_model(model, X_train, y_train, X_test):
    return model.model(X_train, y_train, X_test)

