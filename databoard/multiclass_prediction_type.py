import csv
import numpy as np

class PredictionType:
    def __init__(self, *args, **kwargs):
        try:
            self.y_pred_array = kwargs['y_pred_array']
            self.y_probas_array = kwargs['y_probas_array']
        except KeyError:
            # loading from file
            f_name = kwargs['f_name']
            with open(f_name) as f:
                input = list(csv.reader(f))
                input = map(list,map(None,*input))

                #print np.array(input[1:]).astype(float)
                self.y_pred_array = np.array(input[0])
                self.y_probas_array = np.array(input[1:]).astype(float).T
                #print self.y_probas_array

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