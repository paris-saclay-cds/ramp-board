from generic import combine_models
import numpy as np

y_preds = np.array([[1,1], 
	                [0,0]])
y_ranks = np.array([[0,1], 
	                [0,1]])
indexes = np.array([0,1])

# expecting [0 1]
print combine_models(y_preds, y_ranks, indexes)