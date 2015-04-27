import csv
import numpy as np
import pandas as pd

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
    def __init__(self, *args, **kwargs):
        try:
            self.y_pred_array = kwargs['y_pred_array']
            self.y_probas_array = kwargs['y_probas_array']
        except KeyError:
            # loading from file
            f_name = kwargs['f_name']
            input = np.genfromtxt(f_name, delimiter=',')
            self.y_pred_array = input[:,0]
            self.y_probas_array = np.array(input[:,1:], dtype=float)

    def save_predictions(self, f_name):
        num_classes = self.y_probas_array.shape[1]
        with open(f_name, "w") as f:
            for y_pred, y_probas in zip(self.y_pred_array, self.y_probas_array):
                f.write(y_pred)
                for y_proba in y_probas:
                    f.write("," + str(y_proba))
                f.write("\n")

    def get_predictions(self):
        return self.y_pred_array, self.y_probas_array