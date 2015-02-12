import os
import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.externals.joblib import Memory

from generic import train_model, read_data

from config_databoard import root_path, n_CV, test_size
from config_databoard import random_state, cachedir


X, y = read_data()
skf = StratifiedShuffleSplit(y, n_iter=n_CV, test_size=test_size,
                             random_state=random_state)

models = pd.read_csv("submissions.csv")

mem = Memory(cachedir=cachedir)
train_model = mem.cache(train_model)


def train_models(models, last_time_stamp=None):
    models = models.sort("timestamp")
    models.index = range(1, len(models) + 1)
    # models = models[models.index < 50]  # XXX to make things fast

    failed_models = models.copy()
    trained_models = models.copy()

    for idx, team, model, timestamp, path in zip(models.index.values,
                                                 models['team'],
                                                 models['model'],
                                                 models['timestamp'],
                                                 models['path']):
        m_path = os.path.join(root_path, 'models', path)

        print "Training : %s" % m_path

        try:
            train_model(m_path, X, y, skf)
            failed_models.drop(idx, axis=0, inplace=True)
        except Exception, e:
            trained_models.drop(idx, axis=0, inplace=True)
            print e
            with open(os.path.join(m_path, 'error.txt'), 'w') as f:
                f.write("%s" % e)
            print "ERROR (non fatal): Model not trained."

    return trained_models, failed_models

trained_models, failed_models = train_models(models)

print trained_models
print failed_models

trained_models.to_csv("trained_submissions.csv", index=False)
failed_models.to_csv("failed_submissions.csv", index=False)
