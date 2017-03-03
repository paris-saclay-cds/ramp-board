# coding=utf-8
import time
import os
import re
import glob

import numpy as np
from skimage.io import imread
import pandas as pd

from classifier import Classifier
from feature_extractor import FeatureExtractor

from databoard.specific.workflows.batch_classifier_workflow import BatchGeneratorBuilder

if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    df = df.iloc[0:1024]
    id_array = df['our_unique_id'].values
    y_array = df['class'].values
    fe = FeatureExtractor()
    clf = Classifier()
    gen_builder = BatchGeneratorBuilder(
        id_array, y_array, 
        fe, folder='../data/raw/imgs',
        chunk_size=1024,
        n_jobs=1)
    clf.fit(gen_builder)
