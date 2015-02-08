import numpy as np
import pandas as pd
import os
from generic import leaderboard_classical, leaderboard_combination, leaderboard_to_html
import glob

trained_models = pd.read_csv("trained_submissions.csv")
print trained_models

#gt_path = root_path + "/Submission/GroundTruth"
l1 = leaderboard_classical(trained_models)
l1_html = leaderboard_to_html(l1)
print l1_html
#print trained_models
#print leaderboard_combination(gt_path, m_paths)