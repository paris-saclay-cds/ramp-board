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
def save_multi_class_prediction(y_pred, y_proba, f_name):
    num_classes = y_proba.shape[1]
    output = [np.append(y, p) for y, p in zip(y_pred, y_proba)]
    fmt = "%d"
    for i in range(num_classes):
        fmt = fmt + ",%lf"
    np.savetxt(f_name, output, fmt=fmt)

def save_multi_class_predictions(model_output, f_name_valid, f_name_test):
    y_valid_pred, y_valid_proba, y_test_pred, y_test_proba = model_output
    save_multi_class_prediction(y_valid_pred, y_valid_proba, f_name_valid)
    save_multi_class_prediction(y_test_pred, y_test_proba, f_name_test)

def load_multi_class_predictions(predictions_path):
    csv_file = open(predictions_path)
    predictions = []
    for row in csv_file:
        number_strings = row.split(',')
        probas = map(float, number_strings[1:])
        prediction = [
            int(number_strings[0]), # class label
            map(float, number_strings[1:])  # class probas list
        ]
        predictions.append(prediction)
    return predictions
