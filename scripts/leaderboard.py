import numpy as np
import pandas as pd
import os
from databoard.generic import leaderboard_classical, leaderboard_combination, leaderboard_to_html
import glob
from config_databoard import root_path

trained_models = pd.read_csv("trained_submissions.csv")
print trained_models

gt_path = os.path.join(root_path, 'ground_truth')

l1 = leaderboard_classical(gt_path, trained_models)
print l1
#l1_html = leaderboard_to_html(l1)
#print l1_html


#gt_path = os.path.join(root_path, 'Submission', 'GroundTruth')
#l2 = leaderboard_combination(trained_models, gt_path)
#print l2
#l2_html = leaderboard_to_html(l2)
#print l2_html