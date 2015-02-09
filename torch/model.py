import numpy as np

from tempfile import NamedTemporaryFile
import pandas as pd

import os
import subprocess

def model(X_train, y_train, X_test):
    where_are_NaNs = np.isnan(X_train)
    X_train[where_are_NaNs] = -1
    where_are_NaNs = np.isnan(X_test)
    X_test[where_are_NaNs] = -1

    y_train_shape_dim = len(y_train.shape)
    if y_train_shape_dim == 1:
        y_train = y_train[:, np.newaxis]

    train = np.concatenate((y_train, X_train), axis=1)
    train = pd.DataFrame(train)

    test = pd.DataFrame(X_test)

    train_tmp_file = NamedTemporaryFile(delete=False)
    train.to_csv(train_tmp_file, header=False, index=False)
    train_tmp_file.close()

    test_tmp_file = NamedTemporaryFile(delete=False)
    test.to_csv(test_tmp_file, header=False, index=False)
    test_tmp_file.close()

    pred_tmp_file = NamedTemporaryFile(delete=False)
    pred_tmp_file.close()

    subprocess.call("luajit model_generic.lua --train %s --test %s --output %s" % (train_tmp_file.name, test_tmp_file.name, pred_tmp_file.name), shell=True)

    pred = pd.read_csv(pred_tmp_file.name, header=False).values
    if y_train_shape_dim == 2:
        pred = pred[:, np.newaxis]

    os.remove(train_tmp_file.name)
    os.remove(test_tmp_file.name)
    os.remove(pred_tmp_file.name)
    return pred

if __name__ == "__main__":
    X_size = (100, 1)
    X_test_size = (20, 1)
    X_train = np.random.normal(size=X_size)
    y_train = 1*(X_train >= 0.2)[:, 0]

    X_test = np.random.normal(size=X_test_size)
    model(X_train, y_train, X_test)
