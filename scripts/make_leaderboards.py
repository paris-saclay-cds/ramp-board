import os
import pandas as pd

from generic import leaderboard_classical, leaderboard_combination
from config_databoard import root_path

groundtruth_path = os.path.join(root_path, 'ground_truth')

submissions_path = os.path.join(root_path, 'trained_submissions.csv')
trained_models = pd.read_csv(submissions_path)

l1 = leaderboard_classical(groundtruth_path, trained_models)
l2 = leaderboard_combination(groundtruth_path, trained_models)

l1.to_csv("leaderboard1.csv", index=False)
l2.to_csv("leaderboard2.csv", index=False)
