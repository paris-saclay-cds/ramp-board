from importlib import import_module
from databoard.multiclass_prediction import Predictions


def train_submission(self, module_path, X_array, y_array, train_is):
    classifier = import_module('.classifier', module_path)
    clf = classifier.Classifier()
    clf.fit(X_array[train_is], y_array[train_is])
    return clf


def test_submission(self, trained_model, X_array, test_is):
    clf = trained_model
    y_proba = clf.predict_proba(X_array[test_is])
    return Predictions(y_pred=y_proba)
