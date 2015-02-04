import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
from generic import setup, root_path, n_CV, test_size, random_state
import glob

data = io.loadmat('dataMarathon.mat')
Z = np.c_[data['data_target'].astype(np.int), data['X']]
label_col = u'TARGET'
columns = [label_col] + [d[0] for d in data['header'].ravel()]
df = pd.DataFrame(Z, columns=columns)
Z = df.values
y = Z[:, 0]
X = Z[:, 1:]

gt_path = root_path + "/Submission/GroundTruth"
skf = StratifiedShuffleSplit(y, n_iter=n_CV, test_size=test_size, random_state=random_state)
setup(gt_path, y, skf)