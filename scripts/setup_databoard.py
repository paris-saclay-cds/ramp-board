# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

import os
import glob

from sklearn.cross_validation import StratifiedShuffleSplit

from generic import setup_ground_truth, read_data
from config_databoard import root_path, n_CV, test_size, random_state, cachedir

# cleanup prediction files
fnames = []
if os.path.exists('ground_truth'):
    fnames = glob.glob('ground_truth/pred_*')
else:
    os.mkdir('ground_truth')
fnames += glob.glob('models/*/pred_*')
fnames += glob.glob('models/*/*/pred_*')
for fname in fnames:
    os.remove(fname)

old_fnames = ["leaderboard1.csv", "leaderboard2.csv",
              "failed_submissions.csv", "submissions.csv",
              "trained_submissions.csv"]
for fname in old_fnames:
    if os.path.exists(fname):
        os.remove(fname)

# Create last_trained_timestamp.py file
gt_path = os.path.join(root_path, 'ground_truth')
os.rmdir(gt_path)  # cleanup the ground_truth
os.mkdir(gt_path)
X, y = read_data()
skf = StratifiedShuffleSplit(y, n_iter=n_CV, test_size=test_size, random_state=random_state)
setup_ground_truth(gt_path, y, skf)

# Flush joblib cache
from sklearn.externals.joblib import Memory
mem = Memory(cachedir=cachedir)
mem.clear()