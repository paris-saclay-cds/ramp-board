import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline


class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = make_pipeline(
            Imputer(),
            RandomForestRegressor(n_estimators=20, max_depth=None))

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
