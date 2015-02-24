import os
import sys
import pandas as pd

from sklearn.externals.joblib import Memory

from incentiboard.generic import train_models
from incentiboard.config_databoard import cachedir


models = pd.read_csv("output/submissions.csv")

trained_models, failed_models = train_models(models)

print trained_models
print failed_models

trained_models.to_csv("output/trained_submissions.csv", index=False)
failed_models.to_csv("output/failed_submissions.csv", index=False)
