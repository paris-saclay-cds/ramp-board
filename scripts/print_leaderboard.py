import numpy as np
import pandas as pd
import os
from databoard.generic import leaderboard_classical, leaderboard_combination
import glob
from config_databoard import root_path

trained_models = pd.read_csv("trained_submissions.csv")
print trained_models

gt_path = os.path.join(root_path, 'ground_truth')
print gt_path

l1 = leaderboard_classical(gt_path, trained_models)
print l1

l2 = leaderboard_combination(gt_path, trained_models)
print l2
