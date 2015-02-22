import os
import sys
import pandas as pd

# FIXME: use relative imports instead
prog_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, prog_path)

from generic import leaderboard_classical, leaderboard_combination, private_leaderboard_classical
from config_databoard import root_path

groundtruth_path = os.path.join(root_path, 'ground_truth')

submissions_path = os.path.join(root_path, 'output/trained_submissions.csv')
trained_models = pd.read_csv(submissions_path)

l1 = leaderboard_classical(groundtruth_path, trained_models)
l2 = leaderboard_combination(groundtruth_path, trained_models)
l3 = private_leaderboard_classical(trained_models)

l1.to_csv("output/leaderboard1.csv", index=False)
l2.to_csv("output/leaderboard2.csv", index=False)
l3.to_csv("output/leaderboard3.csv", index=False)
