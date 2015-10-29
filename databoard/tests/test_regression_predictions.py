from ..regression_prediction import Predictions
import numpy as np
from numpy.testing import assert_equal, assert_array_equal


y_pred = [0.7, 0.1, 0.2]
predictions = Predictions(y_pred=y_pred)
assert_array_equal(predictions.y_pred, y_pred)

predictions.save_predictions('/tmp/predictions.csv')
loaded_predictions = Predictions(f_name='/tmp/predictions.csv')
assert_array_equal(predictions.y_pred, loaded_predictions.y_pred)

assert_equal(predictions.n_samples, 3)

assert_array_equal(predictions.y_pred_comb, predictions.y_pred)

cv_y_pred = [0.6, 0.3]
cv_predictions = Predictions(y_pred=cv_y_pred)
updated_y_pred = [0.6, 0.1, 0.3]
updated_predictions = Predictions(y_pred=updated_y_pred)
predictions.set_valid_predictions(cv_predictions, [0, 2])
assert_array_equal(predictions.y_pred, updated_predictions.y_pred)

predictions = Predictions(n_samples=3)
assert_array_equal(predictions.y_pred, np.array([np.nan, np.nan, np.nan]))
