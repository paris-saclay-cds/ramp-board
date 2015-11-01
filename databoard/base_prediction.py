import numpy as np


class BasePrediction(object):

    def __init__(self, y_pred):
        self.y_pred = y_pred

    def __str__(self):
        return "y_pred = ".format(self.y_pred)

    def save(self, f_name):
        np.save(f_name, self.y_pred)

    @property
    def n_samples(self):
        return len(self.y_pred)

