import os
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit

from databoard.generic import train_model, read_data

from config_databoard import root_path, n_CV, test_size, random_state
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
    for team, model, path in zip(models_to_train['team'],
                                 models_to_train['model'],
                                 models_to_train['path']):
        m_path = os.path.join(root_path, 'models', path)
        print m_path

        try:
            train_model(m_path, X, y, skf)
            with open('trained_submissions.csv', 'a') as f:
                f.write(team + "," + model + "," + path + "\n")
        except:
            print "ERROR: Model not trained."
