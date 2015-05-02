import csv
import numpy as np

# Global static that should be set by specific (or somebody)
labels = []

class PredictionType:
    def __init__(self, prediction):
        self.y_pred, self.y_probas = prediction

    def __str__(self):
        return "y_pred = {}, y_probas = {}".format(self.y_pred, self.y_probas)

    def get_prediction(self):
        return self.y_pred, self.y_probas

    def make_consistent(self):
        """ Making the prediction consistent by setting the pred to the argmax
        of the probas. Not sure what it's for here.
        """
        self.y_pred = labels[self.y_probas.argmax()]

class PredictionArrayType:
    def __init__(self, *args, **kwargs):
        if 'y_pred_array' in kwargs.keys() and 'y_probas_array' in kwargs.keys():
            self.y_pred_array = kwargs['y_pred_array']
            self.y_probas_array = kwargs['y_probas_array']
        elif 'prediction_list' in kwargs.keys():
            self.y_pred_array = np.array(
                [prediction[0] for prediction in kwargs['prediction_list']])
            self.y_probas_array = np.array(
                [prediction[1] for prediction in kwargs['prediction_list']])
        elif 'f_name' in kwargs.keys():
            # loading from file
            f_name = kwargs['f_name']
            with open(f_name) as f:
                input = list(csv.reader(f))
                input = map(list,map(None,*input))

                #print np.array(input[1:]).astype(float)
                self.y_pred_array = np.array(input[0])
                self.y_probas_array = np.array(input[1:]).astype(float).T
                #print self.y_probas_array
        else:
            raise ValueError("Unkonwn init argument, {}".format(kwargs))

#    def __iter__(self):
#        for y_pred, y_probas in self.get_predictions():
#            yield y_pred, y_probas

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


    def combine(self, indexes = []):
        # Not yet used

        # usually the class contains arrays corresponding to predictions
        # for a set of different data points. Here we assume that it is 
        # a list of predictions produced by different functions on the same
        # data point. We return a single PrdictionType

        #Just saving here in case we want to go bakc there how to 
        #combine based on simply ranks, k = len(indexes)
        #n = len(y_preds[0])
        #n_ones = n * k - y_preds[indexes].sum() # number of zeros
        if len(indexes) == 0: # we combine the full list
            indexes = range(len(self.y_probas_array))
        combined_y_probas = self.y_probas_array.mean(axis=0)
        combined_prediction = PredictionType((labels[0], combined_y_probas))
        combined_prediction.make_consistent()
        return combined_prediction

def get_y_pred_array(y_probas_array):
    return np.array([labels[y_probas.argmax()] for y_probas in y_probas_array])

# also think about working with matrices of PredictionType instead of 
# lists of PredictionArrayType
def transpose(predictions_list):
    pass


def combine(predictions_list, indexes = []):
    if len(indexes) == 0: # we combine the full list
        indexes = range(len(predictions_list))
    #selected_predictions_list = [predictions_list[index].get_predictions() 
    #                             for index in indexes]
    #y_probas_arrays = np.array([predictions.get_predictions()[1] 
    #                   for predictions in selected_predictions_list])

    y_probas_arrays = []
    for index in indexes:
        y_pred_array, y_probas_array = predictions_list[index].get_predictions()
        y_probas_arrays.append(y_probas_array)
    y_probas_arrays = np.array(y_probas_arrays)

    # We do take the mean of probas because sum(log(probas)) have problems at zero
    combined_y_probas_array = y_probas_arrays.mean(axis=0)
    combined_y_pred_array = get_y_pred_array(combined_y_probas_array)
    return PredictionArrayType(
        y_pred_array=combined_y_pred_array, 
        y_probas_array=combined_y_probas_array)


