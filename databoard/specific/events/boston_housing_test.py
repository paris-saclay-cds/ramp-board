# Author: Balazs Kegl
# License: BSD 3 clause

from sklearn.cross_validation import ShuffleSplit
import databoard.scores as scores
from databoard.specific.problems.boston_housing import problem_name  # noqa

event_name = 'boston_housing_test'  # should be the same as the file name
event_title = 'Boston housing regression (test)'

random_state = 57
cv_test_size = 0.5
n_CV = 2
score = scores.RMSE()

ramp_title = 'Boston housing regression (test)'


def get_cv(y_train_array):
    cv = ShuffleSplit(
        len(y_train_array), n_iter=n_CV, test_size=cv_test_size,
        random_state=random_state)
    return cv
