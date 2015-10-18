from .. import regression_prediction_type
from numpy.testing import assert_array_equal, assert_array_almost_equal


y_prediction_array = [0.7, 0.1, 0.2]

prediction_array = regression_prediction_type.PredictionArrayType(
    y_prediction_array=y_prediction_array)

assert_array_equal(prediction_array.get_prediction_array(), y_prediction_array)

prediction_array.save_predictions('/tmp/predictions.csv')
prediction_array_loaded = regression_prediction_type.PredictionArrayType(
    predictions_f_name='/tmp/predictions.csv')
assert_array_almost_equal(prediction_array.get_prediction_array(),
                          prediction_array_loaded.get_prediction_array())

prediction_array_loaded = regression_prediction_type.PredictionArrayType(
    ground_truth_f_name='/tmp/predictions.csv')
assert_array_almost_equal(prediction_array.get_prediction_array(),
                          prediction_array_loaded.get_prediction_array())

