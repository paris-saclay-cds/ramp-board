import os
import sys
import pandas as pd
from sklearn.externals.joblib import Memory

# FIXME: use relative imports instead
prog_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, prog_path)

from generic import train_models, train_model

from config_databoard import ( 
    cachedir
)

models = pd.read_csv("output/submissions.csv")

mem = Memory(cachedir=cachedir)
train_model = mem.cache(train_model)

trained_models, failed_models = train_models(models)

print trained_models
print failed_models

trained_models.to_csv("output/trained_submissions.csv", index=False)
failed_models.to_csv("output/failed_submissions.csv", index=False)
