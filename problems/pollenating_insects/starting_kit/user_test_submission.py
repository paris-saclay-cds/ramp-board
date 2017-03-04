# coding=utf-8
import time
import os
import re
import glob
import logging
from importlib import import_module

import numpy as np
import pandas as pd
from skimage.io import imread

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer

from batch_classifier import train_submission
from batch_classifier import test_submission

def read_data(filename):
    df = pd.read_csv(filename)
    return df['our_unique_id'].values, df['class'].values

def get_cv(y_train_array):
    return StratifiedShuffleSplit(n_splits=2, test_size=0.5, random_state=43)

score_function = accuracy_score

if __name__ == '__main__':
    # PIL image loading logger is a bit annoying, so disable it.
    logging.getLogger('PIL.PngImagePlugin').disabled = True

    X, y = read_data('train.csv')
    cv = get_cv(y)
    for train_is, test_is in cv.split(X, y):
        print("Training ...")
        trained_model = train_submission('', X, y, train_is)
        print("Testing ...")
        y_pred = test_submission(trained_model, X, test_is)
        print(y[test_is].shape, y_pred[test_is].shape)
        score = score_function(y[test_is], y_pred[test_is].argmax(axis=1))
        print('accuracy = ', score)
