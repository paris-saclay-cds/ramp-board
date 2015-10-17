from .. import multiclass_prediction_type
from numpy.testing import assert_array_equal, assert_array_almost_equal

multiclass_prediction_type.labels = ['Class_1', 'Class_2', 'Class_3']

y_probas1 = [0.7, 0.1, 0.2]
y_probas2 = [0.1, 0.1, 0.8]
y_probas3 = [0.2, 0.5, 0.3]

prediction_array = multiclass_prediction_type.PredictionArrayType(
    y_prediction_array=[y_probas1, y_probas2, y_probas3])

assert_array_equal(prediction_array.get_prediction_array(),
                   [y_probas1, y_probas2, y_probas3])
assert_array_equal(prediction_array.get_pred_index_array(), [0, 2, 1])

prediction_array.save_predictions('/tmp/predictions.csv')
prediction_array_loaded = multiclass_prediction_type.PredictionArrayType(
    predictions_f_name='/tmp/predictions.csv')
assert_array_almost_equal(prediction_array.get_prediction_array(),
                          prediction_array_loaded.get_prediction_array())

y_pred_label1 = ['Class_2']
y_pred_label2 = ['Class_1']
y_pred_label3 = ['Class_3']

prediction_array = multiclass_prediction_type.PredictionArrayType(
    y_pred_label_array=[y_pred_label1, y_pred_label2, y_pred_label3])

assert_array_equal(prediction_array.get_prediction_array(),
                   [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
assert_array_equal(prediction_array.get_pred_index_array(), [1, 0, 2])


prediction_array.save_predictions('/tmp/predictions.csv')
prediction_array_loaded = multiclass_prediction_type.PredictionArrayType(
    predictions_f_name='/tmp/predictions.csv')
assert_array_almost_equal(prediction_array.get_prediction_array(),
                          prediction_array_loaded.get_prediction_array())

y_pred_index1 = 1
y_pred_index2 = 0
y_pred_index3 = 2

prediction_array = multiclass_prediction_type.PredictionArrayType(
    y_pred_index_array=[y_pred_index1, y_pred_index2, y_pred_index3])

assert_array_equal(prediction_array.get_prediction_array(),
                   [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
assert_array_equal(prediction_array.get_pred_index_array(), [1, 0, 2])

prediction_array.save_predictions('/tmp/predictions.csv')
prediction_array_loaded = multiclass_prediction_type.PredictionArrayType(
    predictions_f_name='/tmp/predictions.csv')
assert_array_almost_equal(prediction_array.get_prediction_array(),
                          prediction_array_loaded.get_prediction_array())

with open('/tmp/y_pred_label_array_string.csv', 'w') as f:
    f.write(multiclass_prediction_type.labels[y_pred_index1] + '\n')
    f.write(multiclass_prediction_type.labels[y_pred_index2] + '\n')
    f.write(multiclass_prediction_type.labels[y_pred_index3] + '\n')

prediction_array = multiclass_prediction_type.PredictionArrayType(
    ground_truth_f_name='/tmp/y_pred_label_array_string.csv')

assert_array_equal(prediction_array.get_prediction_array(),
                   [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
assert_array_equal(prediction_array.get_pred_index_array(), [1, 0, 2])

prediction_array.save_predictions('/tmp/predictions.csv')
prediction_array_loaded = multiclass_prediction_type.PredictionArrayType(
    predictions_f_name='/tmp/predictions.csv')
assert_array_almost_equal(prediction_array.get_prediction_array(),
                          prediction_array_loaded.get_prediction_array())

with open('/tmp/y_pred_label_array_string_multilabel.csv', 'w') as f:
    f.write(multiclass_prediction_type.labels[0] + ',')
    f.write(multiclass_prediction_type.labels[2] + '\n')
    f.write(multiclass_prediction_type.labels[1] + '\n')
    f.write(multiclass_prediction_type.labels[0] + ',')
    f.write(multiclass_prediction_type.labels[1] + ',')
    f.write(multiclass_prediction_type.labels[2] + '\n')

prediction_array = multiclass_prediction_type.PredictionArrayType(
    ground_truth_f_name='/tmp/y_pred_label_array_string_multilabel.csv')

assert_array_equal(prediction_array.get_prediction_array(),
                   [[0.5, 0, 0.5], [0, 1, 0], [1.0 / 3, 1.0 / 3, 1.0 / 3]])
# in case of ties, argmax returns the index of the first max
assert_array_equal(prediction_array.get_pred_index_array(), [0, 1, 0])

prediction_array.save_predictions('/tmp/predictions.csv')
prediction_array_loaded = multiclass_prediction_type.PredictionArrayType(
    predictions_f_name='/tmp/predictions.csv')
assert_array_almost_equal(prediction_array.get_prediction_array(),
                          prediction_array_loaded.get_prediction_array())

multiclass_prediction_type.labels = [1, 2, 3]

y_pred_label1 = [2]
y_pred_label2 = [1]
y_pred_label3 = [3]

prediction_array = multiclass_prediction_type.PredictionArrayType(
    y_pred_label_array=[y_pred_label1, y_pred_label2, y_pred_label3])

assert_array_equal(prediction_array.get_prediction_array(),
                   [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
assert_array_equal(prediction_array.get_pred_index_array(), [1, 0, 2])

prediction_array.save_predictions('/tmp/predictions.csv')
prediction_array_loaded = multiclass_prediction_type.PredictionArrayType(
    predictions_f_name='/tmp/predictions.csv')
assert_array_almost_equal(prediction_array.get_prediction_array(),
                          prediction_array_loaded.get_prediction_array())

y_pred_label1 = [1, 3]
y_pred_label2 = [2]
y_pred_label3 = [1, 2, 3]

prediction_array = multiclass_prediction_type.PredictionArrayType(
    y_pred_label_array=[y_pred_label1, y_pred_label2, y_pred_label3])

assert_array_equal(prediction_array.get_prediction_array(),
                   [[0.5, 0, 0.5], [0, 1, 0], [1.0 / 3, 1.0 / 3, 1.0 / 3]])
# in case of ties, argmax returns the index of the first max
assert_array_equal(prediction_array.get_pred_index_array(), [0, 1, 0])

prediction_array.save_predictions('/tmp/predictions.csv')
prediction_array_loaded = multiclass_prediction_type.PredictionArrayType(
    predictions_f_name='/tmp/predictions.csv')
assert_array_almost_equal(prediction_array.get_prediction_array(),
                          prediction_array_loaded.get_prediction_array())
