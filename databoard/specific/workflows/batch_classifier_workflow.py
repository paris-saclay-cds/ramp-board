import os
import time
import threading

from joblib import delayed
from joblib import Parallel

import numpy as np
from skimage.io import imread

from keras.utils.np_utils import to_categorical

from importlib import import_module

def train_submission(module_path, X_array, y_array, train_is):
    """
    module_path : str
        folder where the submission is. the folder have to contain
        batch_classifier.py and image_preprocessor.py.
    X_array : ArrayContainer vector of int
        vector of image IDs to train on
        (it is named X_array to be coherent with the current API,
         but as said here, it does not represent the data itself,
         only image IDs).
    y_array : vector of int
        vector of image labels corresponding to X_train
    train_is : vector of int
       indices from X_array to train on 
    """
    batch_classifier = import_module('batch_classifier', module_path)
    image_preprocessor = import_module('image_preprocessor', module_path)
    transform_img = image_preprocessor.transform
    clf = batch_classifier.BatchClassifier()
    attrs = X_array.attrs
    test_batch_size = attrs['test_batch_size']
    chunk_size = attrs['chunk_size']
    n_jobs = attrs['n_jobs']
    n_classes = attrs['n_classes']
    gen_builder = BatchGeneratorBuilder(
        X_array[train_is], y_array[train_is],
        transform_img,
        chunk_size=chunk_size,
        n_classes=n_classes,
        n_jobs=n_jobs)
    clf.fit(gen_builder)
    return transform_img, clf


def test_submission(trained_model, X_array, test_is):
    """
    trained_model : tuple (function, Classifier)
        tuple of a trained model returned by `train_submission`.
    X_array : ArrayContainer of int
        vector of image IDs to test on.
        (it is named X_array to be coherent with the current API,
         but as said here, it does not represent the data itself,
         only image IDs).
    test_is : vector of int
        ##TODO : not used, it should be removed from the API.
    """
    transform_img, clf = trained_model
    attrs = X_array.attrs
    test_batch_size = attrs['test_batch_size']
    chunk_size = attrs['chunk_size']
    n_jobs = attrs['n_jobs']
    folder = attrs['folder']
    it = chunk_iterator(
        X_array, 
        chunk_size=chunk_size, 
        folder=folder)
    y_proba = []
    for X in it:
        for i in range(0, len(X), test_batch_size):
            X_batch = X[i:i + test_batch_size]
            X_batch = Parallel(n_jobs=n_jobs, backend='threading')(delayed(transform_img)(x) for x in X_batch)
            X_batch = [x[np.newaxis, :, :, :] for x in X_batch]
            X_batch = np.concatenate(X_batch, axis=0)
            y_proba_batch = clf.predict_proba(X_batch)
            y_proba.append(y_proba_batch)
    y_proba = np.concatenate(y_proba, axis=0)
    return y_proba


class BatchGeneratorBuilder(object):
    """
    This class is a way to build training and 
    validation generators that yield each time a tuple (X, y) of mini-batches. 
    The generators are built in a way to fit into keras API of `fit_generator`
    (see https://keras.io/models/model/).
    An instance of this class is exposed to users `Classifier` through
    the `fit` function : model fitting is called by using
    "clf.fit(gen_builder)" where `gen_builder` is an instance
    of this class : `BatchGeneratorBuilder`.
    The fit function from `Classifier` should then use the instance
    to build train and validation generators, using the method
    `get_train_valid_generators`

    Parameters
    ==========
        
    X_array : ArrayContainer of int
        vector of image IDs to train on
         (it is named X_array to be coherent with the current API,
         but as said here, it does not represent the data itself,
         only image IDs).

    y_array : vector of int
        vector of image labels corresponding to `X_array`

    folder : str
        folder where the images are

    chunk_size : int
        size of the chunk used to load data from disk into memory.
        (see at the top of the file what a chunk is and its difference
         with the mini-batch size of neural nets).

    n_classes : int
        Total number of classes. This is needed because the array
        of labels, which is a vector of ints, is transformed into
        a onehot representation.

    n_jobs : int
        the number of jobs used to load images from disk to memory as `chunks`.
    """
    def __init__(self, X_array, y_array, 
                 transform_img,
                 folder='imgs', 
                 chunk_size=1024,
                 n_classes=18,
                 n_jobs=8):
        self.X_array = X_array
        self.y_array = y_array
        self.transform_img = transform_img
        self.folder = folder
        self.chunk_size = chunk_size
        self.n_classes = n_classes
        self.n_jobs = n_jobs
        self.nb_examples = len(X_array)
    
    def get_train_valid_generators(self, batch_size=256, valid_ratio=0.1):
        """
        This method is used by the user defined `Classifier` to o build train and 
        valid generators that will be used in keras `fit_generator`.

        Parameters
        ==========

        batch_size : int
            size of mini-batches
        valid_ratio : float between 0 and 1
            ratio of validation data

        Returns
        =======

        a 4-tuple (gen_train, gen_valid, nb_train, nb_valid) where:
            - gen_train is a generator function for training data
            - gen_valid is a generator function for valid data
            - nb_train is the number of training examples
            - nb_valid is the number of validation examples
        The number of training and validation data are necessary
        so that we can use the keras method `fit_generator`.
        """
        nb_valid = int(valid_ratio * self.nb_examples)
        nb_train = self.nb_examples - nb_valid
        indices = np.arange(self.nb_examples)
        train_indices = indices[0:nb_train]
        valid_indices = indices[nb_train:]
        gen_train = self._get_generator(
            indices=train_indices, batch_size=batch_size)
        gen_valid = self._get_generator(
            indices=valid_indices, batch_size=batch_size)
        return gen_train, gen_valid, nb_train, nb_valid

    def _get_generator(self, indices=None, batch_size=256):
        if indices is None:
            indices = np.arange(self.nb_examples)
        while True:
            it = chunk_iterator(
                X_array=self.X_array[indices],
                y_array=self.y_array[indices], 
                chunk_size=self.chunk_size, 
                folder=self.folder,
                n_jobs=self.n_jobs)
            for X, y in it:
                X = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self.transform_img)(x) for x in X)
                X = [x[np.newaxis, :, :, :] for x in X]
                X = np.concatenate(X, axis=0)
                X = np.array(X, dtype='float32')
                y = to_categorical(y, nb_classes=self.n_classes)
                for i in range(0, len(X), batch_size):
                    yield X[i:i + batch_size], y[i:i + batch_size]

def chunk_iterator(X_array, y_array=None, chunk_size=1024, folder='imgs', n_jobs=8):
    """
    Generator function that yields chunks of images, optionally with their labels.

    Parameters
    ==========
    
    X_array : ArrayContainer of int
        image ids to load
        (it is named X_array to be coherent with the current API,
         but as said here, it does not represent the data itself,
         only image IDs).

    y_array : vector of int
        labels corresponding to each image from X_array

    chunk_size : int
        chunk size

    folder : str
        folder where the images are

    n_jobs : int
        number of jobs used to load images in parallel

    Yields
    ======

    if y_array is provided (not None):
        it yields each time a tuple (X, y) where X is a list
        of numpy arrays of images and y is a list of ints (labels).
        The length of X and y is `chunk_size` at most (it can be smaller).

    if y_array is not provided (it is None) 
        it yields each time X where X is a list of numpy arrays
        of images. The length of X is `chunk_size` at most (it can be smaller).
        This is used for testing, where we don't have/need the labels.

    The shape of each element of X in both cases
    is (height, width, color), where color=3 and height/width
    vary according to examples (hence the fact that X is a list instead of numpy array).
    """
    for i in range(0, len(X_array), chunk_size):
        X_chunk = X_array[i:i + chunk_size]
        filenames = map(_get_image_filename, X_chunk)
        filenames = map(lambda filename:os.path.join(folder, filename), filenames)
        X = Parallel(n_jobs=n_jobs, backend='threading')(delayed(imread)(filename) for filename in filenames)
        if y_array is not None:
            y = y_array[i:i + chunk_size]
            yield X, y
        else:
            yield X

def _get_image_filename(unique_id):
    return 'id_{}.jpg'.format(unique_id)

class ArrayContainer(np.ndarray):
    """
    This is an extension of numpy arrays with attributes.
    It follows the guidelines from numpy documentation to
    subclass numpy arrays to have new attributes.
    (Check : https://docs.scipy.org/doc/numpy/user/basics.subclassing.html).
    It is used to pass a set of global variables from the problem
    to the workflow `batch_classifier_workflow`. 
    The global variables that are passed (and they are obligatory):
    - chunk_size,
    - n_jobs,
    - test_batch_size,
    - folder
    - n_classes
    """
    
    def __new__(cls, input_array, attrs=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.attrs = attrs
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.attrs = getattr(obj, 'attrs', None)

