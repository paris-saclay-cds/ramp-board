import numpy as np

# Binary classification
def save_binary_prediction(y_pred, y_proba, f_name):
    output = np.transpose(np.array([y_pred, y_proba[:,1]]))
    np.savetxt(f_name, output, fmt='%d,%lf')

def save_binary_ouput(model_output, f_name_valid, f_name_test):
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

def save_multi_class_ouput(model_output, f_name_valid, f_name_test):
    y_valid_pred, y_valid_proba, y_test_pred, y_test_proba = model_output
    save_multi_class_prediction(y_valid_pred, y_valid_proba, f_name_valid)
    save_multi_class_prediction(y_test_pred, y_test_proba, f_name_test)