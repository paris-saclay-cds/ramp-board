from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=30, max_leaf_nodes=10)

    def fit(self, X, y):
        self.clf.fit(X, y > 0.0005)

    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1] * 0.001 / 0.3
