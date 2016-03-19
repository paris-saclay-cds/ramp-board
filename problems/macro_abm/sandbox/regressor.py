from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = RandomForestRegressor(n_estimators=100, max_leaf_nodes=100)

    def fit(self, X, y):
        self.clf.fit(X, y, sample_weight=y + 1)

    def predict(self, X):
        return self.clf.predict(X)
