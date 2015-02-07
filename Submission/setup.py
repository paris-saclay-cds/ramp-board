import os
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from generic import setup_ground_truth, read_data
from config import root_path, n_CV, test_size, random_state
import glob

gt_path = os.path.join(root_path, 'Submission', 'GroundTruth')
X, y = read_data()
skf = StratifiedShuffleSplit(y, n_iter=n_CV, test_size=test_size, random_state=random_state)
setup_ground_truth(gt_path, y, skf)
with open('last_trained_timestamp.py', 'w') as f:
	f.write('last_trained_timestamp = 0')
