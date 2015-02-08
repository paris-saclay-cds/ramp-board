import os
import glob

import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit

from databoard.generic import setup_ground_truth, read_data
from config_databoard import root_path, n_CV, test_size, random_state

# cleanup prediction files
fnames = glob.glob('ground_truth/pred_*')
fnames += glob.glob('models/*/pred_*')
fnames += glob.glob('models/*/*/pred_*')
for fname in fnames:
    os.remove(fname)

# Create last_trained_timestamp.py file
gt_path = os.path.join(root_path, 'ground_truth')
X, y = read_data()
skf = StratifiedShuffleSplit(y, n_iter=n_CV, test_size=test_size, random_state=random_state)
setup_ground_truth(gt_path, y, skf)
with open('last_trained_timestamp.py', 'w') as f:
  f.write('last_trained_timestamp = 0')
with open('trained_submissions.csv', 'w') as f:
  f.write('team,model,path\n')
