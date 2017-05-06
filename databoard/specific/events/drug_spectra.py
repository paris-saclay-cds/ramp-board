import datetime
from sklearn.model_selection import ShuffleSplit
from databoard.specific.problems.drug_spectra import problem_name  # noqa

event_name = 'drug_spectra'  # should be the same as the file name

# Unmutable config parameters that we always read from here

event_title = 'initial single-day RAMP 2016; Saclay Data Camp 2016/17'

random_state = 57
cv_test_size = 0.2
n_cv = 8
score_type_descriptors = [
    {'name': 'error_mare_mixed', 'precision': 3, 'new_name': 'combined'},
    {'name': 'error_mixed', 'precision': 3, 'new_name': 'err'},
    {'name': 'mare_mixed', 'precision': 3, 'new_name': 'mare'},
]
official_score_name = 'combined'


def get_cv(y_train_array):
    cv = ShuffleSplit(n_splits=n_cv, test_size=cv_test_size,
                      random_state=random_state)
    return cv.split(y_train_array)

# Mutable config parameters to initialize database fields

max_members_per_team = 1
max_n_ensemble = 80  # max number of submissions in greedy ensemble
is_send_trained_mails = True
is_send_submitted_mails = True
min_duration_between_submissions = 15 * 60  # sec
opening_timestamp = datetime.datetime(2000, 1, 1, 0, 0, 0)
# before links to submissions in leaderboard are not alive
public_opening_timestamp = datetime.datetime(2000, 1, 1, 0, 0, 0)
closing_timestamp = datetime.datetime(4000, 1, 1, 0, 0, 0)
is_public = True
