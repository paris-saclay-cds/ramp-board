from .. import multiclass_prediction
from ..multiclass_prediction import Predictions
from numpy.testing import assert_array_equal, assert_array_almost_equal

multiclass_prediction.labels = ['Class_1', 'Class_2', 'Class_3']

ps_1 = [0.7, 0.1, 0.2]
ps_2 = [0.1, 0.1, 0.8]
ps_3 = [0.2, 0.5, 0.3]

predictions = Predictions(y_pred=[ps_1, ps_2, ps_3])

assert_array_equal(predictions.y_pred, [ps_1, ps_2, ps_3])
assert_array_equal(predictions.y_pred_label_index, [0, 2, 1])

predictions.save_predictions('/tmp/predictions.csv')
loaded_predictions = Predictions(f_name='/tmp/predictions.csv')
assert_array_almost_equal(predictions.y_pred, loaded_predictions.y_pred)


y_pred_label_1 = 'Class_2'
y_pred_label_2 = 'Class_1'
y_pred_label_3 = 'Class_3'

predictions = Predictions(
    y_pred_labels=[y_pred_label_1, y_pred_label_2, y_pred_label_3])
predictions = Predictions(
    y_true=[y_pred_label_1, y_pred_label_2, y_pred_label_3])

assert_array_equal(predictions.y_pred, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
assert_array_equal(predictions.y_pred_label_index, [1, 0, 2])


y_pred_label_1 = ['Class_2']
y_pred_label_2 = ['Class_1']
y_pred_label_3 = ['Class_3']

predictions = Predictions(
    y_pred_labels=[y_pred_label_1, y_pred_label_2, y_pred_label_3])
predictions = Predictions(
    y_true=[y_pred_label_1, y_pred_label_2, y_pred_label_3])

assert_array_equal(predictions.y_pred, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
assert_array_equal(predictions.y_pred_label_index, [1, 0, 2])


predictions.save_predictions('/tmp/predictions.csv')
loaded_predictions = Predictions(f_name='/tmp/predictions.csv')
assert_array_almost_equal(predictions.y_pred, loaded_predictions.y_pred)

# y_pred_index_1 = 1
# y_pred_index_2 = 0
# y_pred_index_3 = 2

# predictions = Predictions(
#     y_pred_index_array=[y_pred_index_1, y_pred_index_2, y_pred_index_3])

# assert_array_equal(predictions.y_pred,
#                    [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
# assert_array_equal(predictions.get_pred_index_array(), [1, 0, 2])

# predictions.save_predictions('/tmp/predictions.csv')
# loaded_predictions = Predictions(
#     predictions_f_name='/tmp/predictions.csv')
# assert_array_almost_equal(predictions.y_pred,
#                           loaded_predictions.y_pred)

# assert_array_equal(predictions.y_pred,
#                    [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
# assert_array_equal(predictions.get_pred_index_array(), [1, 0, 2])

# predictions.save_predictions('/tmp/predictions.csv')
# loaded_predictions = Predictions(f_name='/tmp/predictions.csv')
# assert_array_almost_equal(predictions.y_pred, loaded_predictions.y_pred)

multiclass_prediction.labels = [1, 2, 3]

y_pred_label_1 = [2]
y_pred_label_2 = [1]
y_pred_label_3 = [3]

predictions = Predictions(
    y_pred_labels=[y_pred_label_1, y_pred_label_2, y_pred_label_3])
predictions = Predictions(
    y_true=[y_pred_label_1, y_pred_label_2, y_pred_label_3])

assert_array_equal(predictions.y_pred, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
assert_array_equal(predictions.y_pred_label_index, [1, 0, 2])

y_pred_label_1 = 2
y_pred_label_2 = 1
y_pred_label_3 = 3

predictions = Predictions(
    y_pred_labels=[y_pred_label_1, y_pred_label_2, y_pred_label_3])
predictions = Predictions(
    y_true=[y_pred_label_1, y_pred_label_2, y_pred_label_3])

assert_array_equal(predictions.y_pred, [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
assert_array_equal(predictions.y_pred_label_index, [1, 0, 2])

predictions.save_predictions('/tmp/predictions.csv')
loaded_predictions = Predictions(f_name='/tmp/predictions.csv')
assert_array_almost_equal(predictions.y_pred, loaded_predictions.y_pred)

y_pred_label_1 = [1, 3]
y_pred_label_2 = [2]
y_pred_label_3 = [1, 2, 3]

predictions = Predictions(
    y_pred_labels=[y_pred_label_1, y_pred_label_2, y_pred_label_3])
predictions = Predictions(
    y_true=[y_pred_label_1, y_pred_label_2, y_pred_label_3])

assert_array_equal(predictions.y_pred,
                   [[0.5, 0, 0.5], [0, 1, 0], [1.0 / 3, 1.0 / 3, 1.0 / 3]])
# in case of ties, argmax returns the index of the first max
assert_array_equal(predictions.y_pred_label_index, [0, 1, 0])

predictions.save_predictions('/tmp/predictions.csv')
loaded_predictions = Predictions(f_name='/tmp/predictions.csv')
assert_array_almost_equal(predictions.y_pred, loaded_predictions.y_pred)
