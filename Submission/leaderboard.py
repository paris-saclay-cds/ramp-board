import numpy as np
from generic import leaderboard_classical
import glob

model_path = "/Users/kegl/Research/Samples/HealthcareBootcamp/Submission/Models/*"
m_paths = glob.glob(model_path)
print leaderboard_classical(m_paths)