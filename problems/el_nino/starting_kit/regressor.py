from sklearn.base import BaseEstimator
from sklearn import linear_model


class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = linear_model.BayesianRidge()

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)
