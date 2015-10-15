from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


class Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        X_vectorized = X.reshape(
            (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        self.clf = RandomForestClassifier(
            n_estimators=10, max_features=2, max_leaf_nodes=5)
        self.clf.fit(X_vectorized, y)


    def predict(self, X):
        X_vectorized = X.reshape(
            (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        return self.clf.predict(X_vectorized)


    def predict_proba(self, X):
        X_vectorized = X.reshape(
            (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        return self.clf.predict_proba(X_vectorized)
