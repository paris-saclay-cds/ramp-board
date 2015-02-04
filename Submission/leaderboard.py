import numpy as np
from generic import leaderboard_classical, leaderboard_combination, root_path
import glob

model_path = root_path + "/Submission/Models/*"
m_paths = glob.glob(model_path)
gt_path = root_path + "/Submission/GroundTruth"
print leaderboard_classical(m_paths)
print leaderboard_combination(gt_path, m_paths)