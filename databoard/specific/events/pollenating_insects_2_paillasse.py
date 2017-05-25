import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from databoard.specific.problems.pollenating_insects_2 import problem_name  # noqa

event_name = 'pollenating_insects_2_paillasse'  # should be the same as the file name

# Unmutable config parameters that we always read from here

event_title = 'La Paillasse / Futur en Seine'

random_state = 57
cv_test_size = 0.2
n_cv = 1
score_type_descriptors = [
    {'name': 'accuracy', 'precision': 3},
    {'name': 'negative_log_likelihood', 'new_name': 'nll', 'precision': 3},
    {'name': 'f1_above', 'new_name': 'f1a', 'precision': 2},
]


def get_cv(y_train_array):
    cv = StratifiedShuffleSplit(n_splits=n_cv, test_size=cv_test_size,
                                random_state=random_state)
    return cv.split(y_train_array, y_train_array)

# Mutable config parameters to initialize database fields

max_members_per_team = 1
max_n_ensemble = 80  # max number of submissions in greedy ensemble
is_send_trained_mails = True
is_send_submitted_mails = True
min_duration_between_submissions = 24 * 60 * 60  # sec
opening_timestamp = datetime.datetime(2000, 1, 1, 0, 0, 0)
# before links to submissions in leaderboard are not alive
public_opening_timestamp = datetime.datetime(2017, 6, 7, 19, 0, 0)
closing_timestamp = datetime.datetime(4000, 1, 1, 0, 0, 0)
is_public = True
