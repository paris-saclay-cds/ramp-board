class BasePrediction(object):

    def __init__(self, y_pred):
        self.y_pred = y_pred

    def __str__(self):
        return "y_pred = ".format(self.y_pred)
