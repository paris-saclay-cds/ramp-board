# coding=utf-8
from __future__ import division
import time
import os
import re
import glob
from importlib import import_module

import numpy as np
from skimage.io import imread
import pandas as pd

from data import minibatch_img_iterator

from classifier import Classifier
from feature_extractor import FeatureExtractor

if __name__ == '__main__':
    df = pd.read_csv('pub_train/data.csv')
    fe = FeatureExtractor()
    clf = Classifier()

    t0 = time.time()
    nb_minibatches = 0
    for _ in range(30):
        for X, y in minibatch_img_iterator(df, batch_size=1024, include_y=True, folder='pub_train'):
            X = fe.transform(X)
            y = fe.transform_output(y)
            clf.partial_fit(X, y)
            nb_minibatches += 1
    elapsed = time.time() - t0
    print('duration : {:.3f} secs'.format(elapsed))
    elapsed_per_minibatch = elapsed / nb_minibatches
    print('{:.3f} sec/minibatch'.format(elapsed_per_minibatch))
