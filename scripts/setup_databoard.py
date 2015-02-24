# Author: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD 3 clause

import os
import sys
import glob

from git import Repo, Submodule

from databoard.generic import setup_ground_truth
from databoard.specific import prepare_data
from databoard.config_databoard import (
    root_path, 
    cachedir,
    repos_path,
    ground_truth_path,
    output_path,
    models_path
)

# cleanup prediction files
fnames = []
if os.path.exists(ground_truth_path):
    fnames = glob.glob(os.path.join(ground_truth_path, 'pred_*'))
else:
    os.mkdir(ground_truth_path)

if not os.path.exists(output_path):
    os.mkdir(output_path)

if not os.path.exists(models_path):
    os.mkdir(models_path)
open(os.path.join(models_path, '__init__.py'), 'a').close()


# TODO: the following will be removed after switching to a database
fnames += glob.glob(os.path.join(models_path, '*', '*', 'pred_*'))
fnames += glob.glob(os.path.join(models_path, '*', '*', 'score.csv'))
fnames += glob.glob(os.path.join(models_path, '*', '*', 'error.txt'))
for fname in fnames:
    if os.path.exists(fname):
        os.remove(fname)

old_fnames = glob.glob(os.path.join(output_path, '*.csv'))
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