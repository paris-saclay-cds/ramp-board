# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

import os
import sys
import glob

# FIXME: use relative imports instead
prog_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, prog_path)

from git import Repo, Submodule
from generic import setup_ground_truth
from config_databoard import (
    root_path, 
    cachedir,
    repos_path,
)
from specific import prepare_data

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

# Preparing the data set, typically public train/private held-out test cut
prepare_data()

# Set up the ground truth predictions for the CV folds
setup_ground_truth()

# Flush joblib cache
from sklearn.externals.joblib import Memory
mem = Memory(cachedir=cachedir)
mem.clear()