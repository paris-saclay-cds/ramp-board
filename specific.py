import socket
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from config_databoard import local_deployment
import pandas as pd

target_column_name = 'TARGET'
held_out_test_size = 0.2
skf_test_size = 0.5
random_state = 57
raw_filename = 'data/raw/data.csv'
train_filename = 'data/public/train.csv'
test_filename = 'data/private/test.csv'
n_CV = 2 if local_deployment else 5 * n_processes

def prepare_data():
    df = pd.read_csv(raw_filename)
    df_y = df[target_column_name]
    df_X = df.drop(target_column_name, axis=1)
    df_train, df_test = train_test_split(df, 
        test_size=held_out_test_size, random_state=random_state)
    df_train.to_csv(train_filename, index=False) 
    df_test.to_csv(test_filename, index=False) 

def read_data():
    df = pd.read_csv(train_filename)
    y = df[target_column_name].values
    X = df.drop(target_column_name, axis=1).values
    skf = StratifiedShuffleSplit(y, n_iter=n_CV, 
        test_size=skf_test_size, random_state=random_state)
    return X, y, skf

def run_model(model, X_train, y_train, X_test):
    clf = model.Classifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)
    return y_pred, y_score


