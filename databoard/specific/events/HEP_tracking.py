import datetime
import numpy as np
from sklearn.model_selection import ShuffleSplit
from databoard.specific.problems.HEP_tracking import problem_name  # noqa

event_name = 'HEP_tracking'  # should be the same as the file name

# Unmutable config parameters that we always read from here

event_title = 'Particle tracking in the LHC ATLAS detector'

random_state = 57
cv_test_size = 0.5
n_cv = 1
score_type_descriptors = [
    {'name': 'clustering_efficiency',
    'precision': 3,
    'new_name': 'efficiency'},
]

# We do a single fold because blending would not work anyway:
# mean of cluster_ids make no sense
def get_cv(y_train_array):
    unique_event_ids = np.unique(y_train_array[:, 0])
    event_cv = ShuffleSplit(
    	n_splits=n_cv, test_size=cv_test_size, random_state=random_state)
    for train_event_is, test_event_is in event_cv.split(unique_event_ids):
        train_is = np.where(
            np.in1d(y_train_array[:, 0], unique_event_ids[train_event_is]))
        test_is = np.where(
            np.in1d(y_train_array[:, 0], unique_event_ids[test_event_is]))
        yield train_is, test_is

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
