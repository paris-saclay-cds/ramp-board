import os
import sys
import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.externals.joblib import Memory

# FIXME: use relative imports instead
prog_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, prog_path)

from generic import train_model, train_models, read_data

from config_databoard import (
    root_path, 
    n_CV, 
    test_size, 
    random_state, 
    cachedir
)


X, y = read_data()
skf = StratifiedShuffleSplit(y, n_iter=n_CV, test_size=test_size,
                             random_state=random_state)

models = pd.read_csv("output/submissions.csv")

mem = Memory(cachedir=cachedir)
train_model = mem.cache(train_model)

trained_models, failed_models = train_models(models, X, y, skf)

print trained_models
print failed_models

trained_models.to_csv("output/trained_submissions.csv", index=False)
failed_models.to_csv("output/failed_submissions.csv", index=False)
