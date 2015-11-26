from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


class Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.clf = RandomForestClassifier(
            n_estimators=1, max_leaf_nodes=2, random_state=61)
        self.clf.fit(X, y)
        self.X = X

    def predict(self, X):
        return self.clf.predict(self.X)

    def predict_proba(self, X):
        return self.clf.predict_proba(self.X)
