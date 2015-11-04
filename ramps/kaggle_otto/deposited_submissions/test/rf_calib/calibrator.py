import numpy as np
from sklearn.isotonic import IsotonicRegression


class Calibrator():
    def __init__(self):
        pass

    def fit(self, X_array, y_array):
        labels = np.sort(np.unique(y_array))
        self.calibrators = []
        for class_index in range(X_array.shape[1]):
            calibrator = IsotonicRegression(
                y_min=0., y_max=1., out_of_bounds='clip')
            class_indicator = np.array([1 if y == labels[class_index] else 0
                                        for y in y_array])
            calibrator.fit(X_array[:, class_index], class_indicator)
            self.calibrators.append(calibrator)

    def predict_proba(self, y_probas_array_uncalibrated):
        num_classes = y_probas_array_uncalibrated.shape[1]
        y_probas_array_transpose = np.array(
            [self.calibrators[class_index].predict(
                y_probas_array_uncalibrated[:, class_index])
             for class_index in range(num_classes)])
        sum_rows = np.sum(y_probas_array_transpose, axis=0)
        y_probas_array_normalized_transpose = np.divide(
            y_probas_array_transpose, sum_rows)
        return y_probas_array_normalized_transpose.T
