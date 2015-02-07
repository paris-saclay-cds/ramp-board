import os
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
from generic import train_model, read_data
from config import root_path, n_CV, test_size, random_state
from last_trained_timestamp import last_trained_timestamp

X, y = read_data()
skf = StratifiedShuffleSplit(y, n_iter=n_CV, test_size=test_size, random_state=random_state)

models = pd.read_csv("submissions.csv")
models_to_train = models[models['timestamp'] > last_trained_timestamp]
if len(models_to_train) > 0:
	last_trained_timestamp = max(models_to_train['timestamp'])
	print last_trained_timestamp
	with open('last_trained_timestamp.py', 'w') as f:
		f.write('last_trained_timestamp = ' + str(last_trained_timestamp))
	m_paths = [os.path.join(root_path, 'Submission', 'Models', path) for path in models_to_train['path']]
	print m_paths
	for m_path in m_paths:
		print m_paths
		train_model(m_path, X, y, skf)
