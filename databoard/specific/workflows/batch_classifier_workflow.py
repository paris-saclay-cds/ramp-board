import os
import time
import threading

from joblib import delayed
from joblib import Parallel

import numpy as np
from skimage.io import imread

from importlib import import_module
from databoard.specific.problems.pollenating_insects import img_folder
from databoard.specific.problems.pollenating_insects import chunk_size
from databoard.specific.problems.pollenating_insects import n_img_load_jobs
from databoard.specific.problems.pollenating_insects import _get_image_filename

def train_submission(module_path, X_array, y_array, train_is):
    classifier = import_module('.classifier', module_path)
    feature_extractor = import_module('.feature_extractor', module_path)
    fe = feature_extractor.FeatureExtractor(n_jobs=n_img_load_jobs)
    clf = classifier.Classifier()
    gen_builder = BatchGeneratorBuilder(
        X_array[train_is], y_array[train_is], 
        feature_extractor,
        chunk_size=chunk_size)
    clf.fit(gen_builder)
    return clf


def test_submission(trained_model, X_array, test_is):
    clf = trained_model
    it = chunk_iterator(
        X_array, 
        chunk_size=chunk_size, 
        folder=img_folder)
    y_proba = []
    for X in it:
        y_proba.append(clf.predict_proba(X))
    y_proba = np.concatenate(y_proba, axis=0)
    return y_proba


class BatchGeneratorBuilder(object):
    
    def __init__(self, id_array, y_array, 
                feature_extractor, 
                folder=img_folder, 
                chunk_size=1024):
        self.id_array = id_array
        self.y_array = y_array
        self.feature_extractor = feature_extractor
        self.folder = folder
        self.chunk_size = chunk_size
        self.nb_examples = len(id_array)
    
    def get_train_valid_generators(self, batch_size=256, valid_ratio=0.1, n_jobs=8):
        nb_valid = int(valid_ratio * self.nb_examples)
        nb_train = self.nb_examples - nb_valid
        indices = np.arange(self.nb_examples)
        train_indices = indices[0:nb_train]
        valid_indices = indices[nb_train:]
        gen_train = self._get_generator(
            indices=train_indices, batch_size=batch_size, n_jobs=n_jobs)
        gen_valid = self._get_generator(
            indices=valid_indices, batch_size=batch_size, n_jobs=n_jobs)
        return gen_train, gen_valid, nb_train, nb_valid

    def _get_generator(self, indices=None, batch_size=256, n_jobs=8):
        if indices is None:
            indices = np.arange(self.nb_examples)
        while True:
            it = chunk_iterator(
                id_array=self.id_array[indices],
                y_array=self.y_array[indices], 
                chunk_size=self.chunk_size, 
                folder=self.folder,
                n_jobs=n_jobs)
            for X, y in it:
                X = self.feature_extractor.transform(X)
                y = self.feature_extractor.transform_output(y) 
                for i in range(0, len(X), batch_size):
                    yield X[i:i + batch_size], y[i:i + batch_size]

def chunk_iterator(id_array, y_array=None, chunk_size=1024, folder='imgs', n_jobs=8):
    """
    Generator function that yields chunks of images, optionally with their labels.

    Parameters
    ==========
    
    id_array : vector of int
        image ids to load
    y_array : vector of int
        labels corresponding to each image from id_array
    chunk_size : int
        chunk size
    folder : str
        folder where the images are

    Yields
    ======

    if y_array is provided:
        it yields each time a tuple (X, y) where X is a list
        of numpy arrays of images and y is a list of ints (labels).
        The length of X and y is 'chunk_size' at most (it can be smaller).
    if y_array is not provided 
        it yields each time X where X is a list of numpy arrays
        of images. The length of X is 'chunk_size' at most (it can be smaller).
    The shape of each element of X in both cases
    is (height, width, color), where color=3 and height/width
    vary according to examples (hence the fact that X is a list instead of numpy array).
    """
    for i in range(0, len(id_array), chunk_size):
        id_cur_chunk = id_array[i:i + chunk_size]
        y = y_array[i:i + chunk_size]
        filenames = map(_get_image_filename, id_cur_chunk)
        filenames = map(lambda filename:os.path.join(folder, filename), filenames)
        X = Parallel(n_jobs=n_jobs, backend='threading')(delayed(imread)(filename) for filename in filenames)
        if y_array is not None:
            yield X, y
        else:
            yield X
