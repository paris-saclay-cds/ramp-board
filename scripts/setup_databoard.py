# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

import os
import sys
import glob

# FIXME: use relative imports instead
prog_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, prog_path)

from git import Repo, Submodule
from generic import setup_ground_truth, read_data
from sklearn.cross_validation import StratifiedShuffleSplit
from config_databoard import (
    root_path, 
    n_CV, 
    test_size, 
    random_state, 
    cachedir,
    repos_path,
)

# cleanup prediction files
fnames = []
if os.path.exists('ground_truth'):
    fnames = glob.glob('ground_truth/pred_*')
else:
    os.mkdir('ground_truth')

if not os.path.exists('output'):
    os.mkdir('output')

if not os.path.exists('models'):
    os.mkdir('models')
open('models/__init__.py', 'a').close()

fnames += glob.glob('models/*/pred_*')
fnames += glob.glob('models/*/*/pred_*')
for fname in fnames:
    os.remove(fname)

old_fnames = glob.glob('output/*.csv') 
for fname in old_fnames:
    if os.path.exists(fname):
        os.remove(fname)

# Prepare the teams repo submodules
repo = Repo.init(repos_path)  # does nothing if already exists

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