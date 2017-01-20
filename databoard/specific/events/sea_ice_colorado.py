import numpy as np
import datetime
from databoard.specific.problems.sea_ice import problem_name  # noqa

event_name = 'sea_ice_colorado'  # should be the same as the file name

# Unmutable config parameters that we always read from here

event_title = 'Northern hemisphere sea ice prediction'

score_type_descriptors = [
    {'name': 'rmse', 'precision': 3},
]
official_score_name = 'rmse'

# y_train array is blocked in the following way:
# note that y_train does not contain the burn_in prefix, data preparation
# shuold take care of that
# n_common_block | n_cv x block_size
# first cv fold takes 0 blocks (so only n_common_block)
# last cv fold takes n_common_block + (n_cv - 1) x block_size
# block_size should be divisible by 12 if score varies within year ->
# n_validation = n_train - int(n_train / 2) = n_train / 2
# should be divisible by 12 * n_cv -> n_train should be
# multiply of 24 * n_cv
# n_train should also be > 2 * n_burn_in
n_cv = 8


def get_cv(y_train_array):
    n = len(y_train_array)
    block_size = int(n / 2 / n_cv / 12 * 12)
    n_common_block = n - block_size * n_cv
    n_validation = n - n_common_block
    print 'length of common block:', n_common_block, 'months =',\
        n_common_block / 12, 'years'
    print 'length of validation block:', n_validation, 'months =',\
        n_validation / 12, 'years'
    print 'length of each cv block:', block_size, 'months =',\
        block_size / 12, 'years'
    for i in range(n_cv):
        train_is = np.arange(0, n_common_block + i * block_size)
        test_is = np.arange(n_common_block + i * block_size, n)
        yield (train_is, test_is)

# Mutable config parameters to initialize database fields

max_members_per_team = 1
max_n_ensemble = 80  # max number of submissions in greedy ensemble
is_send_trained_mails = True
is_send_submitted_mails = True
min_duration_between_submissions = 15 * 60  # sec
opening_timestamp = datetime.datetime(2000, 1, 1, 0, 0, 0)
# before links to submissions in leaderboard are not alive
public_opening_timestamp = datetime.datetime(2016, 9, 21, 18, 0, 0)
closing_timestamp = datetime.datetime(4000, 1, 1, 0, 0, 0)
is_public = True
