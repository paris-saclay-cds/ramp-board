import numpy as np
import multiclass_prediction_type

multiclass_prediction_type.labels = ['Class_1', 'Class_2', 'Class_3']
y_pred1 = 'Class_1'
y_probas1 = np.array([0.7,0.1,0.2])

y_pred2 = 'Class_3'
y_probas2 = np.array([0.1,0.1,0.8])

y_pred3 = 'Class_2'
y_probas3 = np.array([0.2,0.5,0.3])

prediction_array_type = multiclass_prediction_type.PredictionArrayType(
    y_pred_array=np.array([y_pred1, y_pred2, y_pred3]), 
    y_probas_array=np.array([y_probas1, y_probas2, y_probas3]))

combined_prediction = prediction_array_type.combine()
print combined_prediction

prediction_array_type = multiclass_prediction_type.PredictionArrayType(
    prediction_list=[(y_pred1, y_probas1), 
                     (y_pred2, y_probas2), 
                     (y_pred3, y_probas3)])

combined_prediction = prediction_array_type.combine()
print combined_prediction
