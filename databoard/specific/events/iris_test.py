# Author: Balazs Kegl
# License: BSD 3 clause

from sklearn.cross_validation import StratifiedShuffleSplit
import databoard.scores as scores
from databoard.specific.problems.iris import problem_name  # noqa

event_name = 'iris_test'  # should be the same as the file name
event_title = 'Iris classification (test)'

random_state = 57
cv_test_size = 0.5
n_CV = 2
score = scores.Accuracy()


def get_cv(y_train_array):
    cv = StratifiedShuffleSplit(
        y_train_array, n_iter=n_CV, test_size=cv_test_size,
        random_state=random_state)
    return cv
