import os
import sys
import glob

import numpy as np
import pandas as pd

from databoard.generic import leaderboard_classical, leaderboard_combination
from databoard.config_databoard import (
    root_path, 
    repos_path, 
    serve_port,
    server_name,
    local_deployment,
    tag_len_limit,
)

trained_models = pd.read_csv("output/trained_submissions.csv")
print trained_models

groundtruth_path = os.path.join(root_path, 'ground_truth')
print groundtruth_path

l1 = leaderboard_classical(groundtruth_path, trained_models)
print l1

l2 = leaderboard_combination(groundtruth_path, trained_models)
print l2
