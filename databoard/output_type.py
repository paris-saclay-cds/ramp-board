import csv
import numpy as np

# Fixme: should be classes
# Binary classification: to be tested
def save_binary_prediction(y_pred, y_proba, f_name):
    output = np.transpose(np.array([y_pred, y_proba[:,1]]))
    np.savetxt(f_name, output, fmt='%d,%lf')

def save_binary_predictions(model_output, f_name_valid, f_name_test):
    y_valid_pred, y_valid_proba, y_test_pred, y_test_proba = model_output
    save_single_class_prediction(y_valid_pred, y_valid_proba, f_name_valid)
    save_single_class_prediction(y_test_pred, y_test_proba, f_name_test)

def load_binary_predictions(model_output, f_name_valid, f_name_test):
    y_valid_pred, y_valid_proba, y_test_pred, y_test_proba = model_output
    save_single_class_prediction(y_valid_pred, y_valid_proba, f_name_valid)
    save_single_class_prediction(y_test_pred, y_test_proba, f_name_test)

# Multi-class classification
class MultiClassClassification:
    def __init__(self, y_pred_array, y_probas_array):
        self.y_pred_array = y_pred_array
        self.y_probas_array = y_probas_array

    def __init__(self, f_name):
        output = np.genfromtxt(f_name, delimiter=',')
        self.y_pred_array = np.array(output[:,0], dtype=int)
        self.y_probas_array = np.array(output[:,1:], dtype=float)

    def save_predictions(self, f_name):
        num_classes = self.y_probas_array.shape[1]
        output = [np.append(y_pred, y_probas) for y_pred, y_probas
                  in zip(self.y_pred_array, self.y_probas_array)]
        fmt = "%d"
        for i in range(num_classes):
            fmt = fmt + ",%lf"
        np.savetxt(f_name, output, fmt=fmt)

    def get_predictions(self):
        return self.y_pred_array, self.y_probas_array