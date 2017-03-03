import numpy as np
from skimage.transform import resize
from joblib import Parallel, delayed
from keras.utils.np_utils import to_categorical

class FeatureExtractor(object):
    
    def transform(self, X):
        n_jobs = 8
        X = Parallel(n_jobs=n_jobs)(delayed(_transform_single)(x) for x in X)
        X = np.array(X, dtype='float32')
        X = X / 255.
        return X
    
    def transform_output(self, y):
        return to_categorical(y, nb_classes=18)

def _transform_single(x):
    return resize(x, (3, 64, 64), preserve_range=True)
