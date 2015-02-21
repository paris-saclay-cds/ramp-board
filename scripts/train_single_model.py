import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
import click
from generic import train_models, read_data
from config_databoard import (
    root_path, 
    n_CV, 
    test_size, 
    random_state
)

@click.command()
@click.option('--alias', default='kegl/rf300')
def train(alias):
	models = pd.read_csv("output/submissions.csv")
	model = models[models['alias'] == alias]
	X, y = read_data()
	skf = StratifiedShuffleSplit(y, n_iter=n_CV, test_size=test_size,
                             random_state=random_state)

	train_models(model, X, y, skf)

if __name__ == '__main__':
    train()